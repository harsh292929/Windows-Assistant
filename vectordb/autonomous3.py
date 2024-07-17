import tkinter as tk
from tkinter import ttk, messagebox
import os
import time
import threading
from queue import Queue
from PIL import ImageGrab
import easyocr
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import requests
import json
import numpy as np
import pyautogui

current_dir = os.getcwd()
db_dir = os.path.join(current_dir, "chroma_db")
client = chromadb.Client(Settings(persist_directory=db_dir))
collection = client.get_or_create_collection("screenshots_collection")
model = SentenceTransformer('all-MiniLM-L6-v2')
reader = easyocr.Reader(['en'])
screenshot_queue = Queue()
query_entry = None
results_text = None
db_status_label = None
n_results_entry = None


def capture_screenshots():
    while True:
        screenshot = ImageGrab.grab()  # Capture the entire screen
        screenshot_queue.put((screenshot, time.time()))
        time.sleep(10)


def process_screenshots():
    while True:
        screenshot, timestamp = screenshot_queue.get()
        image_np = np.array(screenshot)
        results = reader.readtext(image_np)
        text = " ".join([detection[1] for detection in results])
        add_documents([text], metadatas=[{"timestamp": timestamp}], ids=[f"doc_{timestamp}"])


def generate_multi_vectors(text):
    full_text_vector = model.encode(text).tolist()
    first_sentence_vector = model.encode(text.split('.')[0]).tolist()
    keywords = " ".join([word for word in text.split() if len(word) > 5])
    keyword_vector = model.encode(keywords).tolist()
    return [full_text_vector, first_sentence_vector, keyword_vector]

def add_documents(texts, metadatas=None, ids=None):
    if ids is None:
        ids = [f"doc_{time.time()}_{i}" for i in range(len(texts))]
    for i, text in enumerate(texts):
        multi_vectors = generate_multi_vectors(text)
        for j, vector in enumerate(multi_vectors):
            collection.add(
                embeddings=[vector],
                documents=[text],
                metadatas=[metadatas[i] if metadatas else None],
                ids=[f"{ids[i]}_aspect_{j}"]
            )


def query_documents(query_text, n_results=5):
    query_embedding = model.encode([query_text]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=n_results, include=["documents", "metadatas"])
    documents_with_timestamps = [(doc, metadata.get("timestamp", None)) for doc, metadata in zip(results["documents"][0], results["metadatas"][0])]
    return {"documents": documents_with_timestamps}


def query_local_lm(prompt, context, retries=3):
    url = "http://localhost:1234/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "local-model",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the user's query."},
            {"role": "user", "content": f"Context: {context}\n\nQuery: {prompt}"}
        ],
        "temperature": 0.7
    }
    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data), timeout=100)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except requests.RequestException as e:
            if attempt < retries - 1:
                time.sleep(1)
    return f"Error: Unable to get a response from the local LLM after {retries} attempts."


def check_database_population():
    try:
        results = collection.query(query_embeddings=model.encode(["test query"]).tolist(), n_results=1)
        print(f"Database query test result: {results}")
        print(f"Number of items in collection: {collection.count()}")
    except Exception as e:
        print(f"Error querying database: {e}")


def on_query_click():
    query = query_entry.get()
    if not query:
        messagebox.showwarning("Warning", "Please enter a query.")
        return

    try:
        n_results = int(n_results_entry.get())
        if n_results <= 0:
            raise ValueError("Number of results must be a positive integer.")
    except ValueError:
        messagebox.showerror("Error", "Invalid number of results.")
        return

    results = query_documents(query, n_results)
    context = "\n".join([doc for doc, _ in results['documents']])
    response = query_local_lm(query, context)

    results_text.config(state=tk.NORMAL)
    results_text.delete(1.0, tk.END)  # Clear previous results
    results_text.insert(tk.END, f"LLM Response: {response}\n\n")
    for i, (doc, timestamp) in enumerate(results['documents'], 1):
        results_text.insert(tk.END, f"\n--- Result {i} ---\n")
        results_text.insert(
            tk.END, f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))}\n"
        )
        results_text.insert(tk.END, f"Text: {doc[:200]}...\n")
    results_text.config(state=tk.DISABLED)


def on_check_status_click():
    check_database_population()
    db_status_label.config(text=f"Items in collection: {collection.count()}")

def create_gui():
    global window, query_entry, results_text, n_results_entry, db_status_label

    window = tk.Tk()
    window.title("Screenshot Query System")

    query_label = ttk.Label(window, text="Enter your query:")
    query_label.pack(pady=10)
    query_entry = ttk.Entry(window, width=50)
    query_entry.pack()

    n_results_label = ttk.Label(window, text="Number of results:")
    n_results_label.pack(pady=5)
    n_results_entry = ttk.Entry(window, width=10)
    n_results_entry.insert(0, "5")  # Default to 5 results
    n_results_entry.pack()

    query_button = ttk.Button(window, text="Query", command=on_query_click)
    query_button.pack(pady=10)

    status_button = ttk.Button(window, text="Check Status", command=on_check_status_click)
    status_button.pack(pady=5)

    db_status_label = ttk.Label(window, text="Database status: Not checked yet")
    db_status_label.pack(pady=5)

    results_text = tk.Text(window, wrap=tk.WORD, width=80, height=20)
    results_text.pack(pady=10)
    results_text.config(state=tk.DISABLED)  # Make it read-only initially


    # ... (start capture and processing threads as you were doing before)
    window.mainloop()



if __name__ == "__main__":
    # Start the screenshot capture thread
    screenshot_thread = threading.Thread(target=capture_screenshots, daemon=True)
    screenshot_thread.start()

    # Start the screenshot processing thread
    processing_thread = threading.Thread(target=process_screenshots, daemon=True)
    processing_thread.start()

    create_gui()