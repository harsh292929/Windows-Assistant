import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import requests
import json
import logging
import PyPDF2
from tqdm import tqdm
import time
from elasticsearch import Elasticsearch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ChromaDB setup
chroma_client = chromadb.Client(Settings(persist_directory="./chroma_db"))
chroma_collection = chroma_client.get_or_create_collection("my_collection")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Elasticsearch setup
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

def extract_text_from_pdf(file_path, chunk_size=500, chunk_overlap=50):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        text_chunks = []
        for page_num in tqdm(range(num_pages), desc="Processing PDF"):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            for i in range(0, len(text), chunk_size - chunk_overlap):
                chunk = text[i:i + chunk_size]
                text_chunks.append(chunk)
    return text_chunks

def add_documents(texts, metadatas=None, ids=None):
    embeddings = model.encode(texts).tolist()
    if ids is None:
        ids = [f"doc_{i}" for i in range(len(texts))]
    
    # Add to ChromaDB
    chroma_collection.add(embeddings=embeddings, documents=texts, metadatas=metadatas, ids=ids)
    
    # Add to Elasticsearch
    for i, text in enumerate(texts):
        es.index(index="document_index", id=ids[i], body={"content": text})

def hybrid_search(query_text, n_results=5, keyword_weight=0.3, semantic_weight=0.7):
    # Semantic search using ChromaDB
    query_embedding = model.encode([query_text]).tolist()
    semantic_results = chroma_collection.query(query_embeddings=query_embedding, n_results=n_results)
    
    # Keyword search using Elasticsearch
    es_results = es.search(index="document_index", body={
        "query": {
            "match": {
                "content": query_text
            }
        },
        "size": n_results
    })

    # Combine results
    combined_results = {}
    for i, doc in enumerate(semantic_results['documents'][0]):
        combined_results[semantic_results['ids'][0][i]] = {
            'document': doc,
            'semantic_score': 1 - i/n_results  # Normalize score
        }

    for hit in es_results['hits']['hits']:
        doc_id = hit['_id']
        if doc_id in combined_results:
            combined_results[doc_id]['keyword_score'] = hit['_score']
        else:
            combined_results[doc_id] = {
                'document': hit['_source']['content'],
                'keyword_score': hit['_score']
            }

    # Normalize Elasticsearch scores
    max_keyword_score = max(result['keyword_score'] for result in combined_results.values() if 'keyword_score' in result)
    for result in combined_results.values():
        if 'keyword_score' in result:
            result['keyword_score'] /= max_keyword_score
        else:
            result['keyword_score'] = 0
        if 'semantic_score' not in result:
            result['semantic_score'] = 0

    # Calculate final scores
    for result in combined_results.values():
        result['final_score'] = (keyword_weight * result['keyword_score'] + 
                                 semantic_weight * result['semantic_score'])

    # Sort by final score and return top n_results
    sorted_results = sorted(combined_results.items(), key=lambda x: x[1]['final_score'], reverse=True)[:n_results]
    return [item[1]['document'] for item in sorted_results]

def query_local_lm(prompt, retries=3):
    url = "http://localhost:1234/v1/chat/completions"
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
            logging.error(f"Attempt {attempt+1}/{retries} failed: {e}")
            if attempt < retries - 1:
                logging.info(f"Retrying in 1 second...")
                time.sleep(1)
    return f"Error: Unable to get a response from the local LLM after {retries} attempts."

def rag_query(query, n_results=3):
    results = hybrid_search(query, n_results)
    context = "\n".join(results)
    prompt = f"""Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {query} Answer:"""
    response = query_local_lm(prompt)
    return response

def add_pdf_documents(file_path):
    text_chunks = extract_text_from_pdf(file_path)
    add_documents(texts=text_chunks)

if __name__ == "__main__":
    pdf_file_path = "your_document.pdf"
    add_pdf_documents(pdf_file_path)

    while True:
        query = input("Enter your query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        answer = rag_query(query)
        print("\nAnswer:", answer)