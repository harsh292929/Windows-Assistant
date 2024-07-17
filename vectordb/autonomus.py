import time
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

def capture_and_store_screenshot():
    print(f"Capturing screenshot at {time.strftime('%Y-%m-%d %H:%M:%S')}")    
    screenshot = ImageGrab.grab()
    image_np = np.array(screenshot)
    results = reader.readtext(image_np)
    text = " ".join([detection[1] for detection in results])
    timestamp = time.time()
    add_documents([text], metadatas=[{"timestamp": timestamp}], ids=[f"doc_{timestamp}"])
    print(f"Processed and stored screenshot data. Extracted text: {text[:100]}...")  # Print first 100 chars of extracted text



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
    expanded_queries = query_expansion(query_text)
    all_results = []
    for query in expanded_queries:
        query_embedding = model.encode([query]).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=n_results, include=["documents", "metadatas"])
        documents_with_timestamps = [(doc, metadata.get("timestamp", None)) for doc, metadata in zip(results["documents"][0], results["metadatas"][0])]
        all_results.extend(documents_with_timestamps)
    unique_results = list(dict.fromkeys(all_results))
    return {"documents": unique_results[:n_results]}

def query_expansion(query):
    prompt = f"Generate 3 alternative phrasings or related terms for the following query: '{query}'"
    expansions = query_local_lm(prompt).split('\n')
    return [query] + expansions

def query_local_lm(prompt, retries=3):
    url = "http://localhost:1234/v1/chat/completions"  # Replace with the actual address of your local LLM
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "local-model",  
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
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


if __name__ == "__main__":
    while True:
        capture_and_store_screenshot()
        time.sleep(10)
