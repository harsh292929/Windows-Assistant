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

client = chromadb.Client(Settings(persist_directory="./chroma_db"))
collection = client.get_or_create_collection("screenshots_collection")
model = SentenceTransformer('all-MiniLM-L6-v2')
reader = easyocr.Reader(['en'])  # Adjust languages if needed

# Queue for communication between threads
screenshot_queue = Queue()

def capture_screenshots():
    while True:
        screenshot = ImageGrab.grab()
        screenshot_queue.put((screenshot, time.time()))
        time.sleep(10)  # Capture a screenshot every 10 seconds

def process_screenshots():
    while True:
        screenshot, timestamp = screenshot_queue.get()
        image_np = np.array(screenshot)
        results = reader.readtext(image_np)
        text = " ".join([detection[1] for detection in results])
        add_documents([text], metadatas=[{"timestamp": timestamp}], ids=[f"doc_{timestamp}"])
        print(f"Processed screenshot from {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))}")

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
    url = "http://localhost:1234/v1/chat/completions"  # Replace with the actual address of your local LLM
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

def query_interface():
    while True:
        print("\nScreenshot Query Interface")
        print("1. Query stored screenshots")
        print("2. Exit")
        choice = input("Enter your choice (1-2): ")

        if choice == '1':
            query = input("Enter your query: ")
            n_results = int(input("Enter the number of results to use as context: "))
            results = query_documents(query, n_results)
            
            context = "\n".join([doc for doc, _ in results['documents']])
            response = query_local_lm(query, context)
            
            print("\nQuery Results:")
            print(f"LLM Response: {response}")
            
            print("\nRelevant Screenshots:")
            for i, (doc, timestamp) in enumerate(results['documents'], 1):
                print(f"\n--- Result {i} ---")
                print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))}")
                print(f"Text: {doc[:200]}...")  # Print first 200 characters
        elif choice == '2':
            print("Exiting the program.")
            return
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    # Start the screenshot capture thread
    capture_thread = threading.Thread(target=capture_screenshots, daemon=True)
    capture_thread.start()

    # Start the screenshot processing thread
    process_thread = threading.Thread(target=process_screenshots, daemon=True)
    process_thread.start()

    # Run the query interface in the main thread
    query_interface()