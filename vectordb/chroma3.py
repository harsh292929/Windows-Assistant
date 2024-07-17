import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import requests
import json
import logging
import PyPDF2
from tqdm import tqdm
import time
from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

client = chromadb.Client(Settings(persist_directory="./chroma_db"))
collection = client.get_or_create_collection("my_collection")
model = SentenceTransformer('all-MiniLM-L6-v2')

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
    collection.add(embeddings=embeddings, documents=texts, metadatas=metadatas, ids=ids)

def hybrid_search(query_text, n_results=5, top_k_semantic=10):
    query_embedding = model.encode([query_text]).tolist()
    semantic_results = collection.query(query_embeddings=query_embedding, n_results=top_k_semantic)
    
    # Check if results exist and wrap in a list if needed
    if semantic_results['documents'][0]:
        documents = semantic_results['documents'][0]
        if isinstance(documents[0], str):
            documents = [documents]  # Wrap in a list if it's a single string
        return documents[:n_results]
    else:
        return []  # Return an empty list if no results found


def rerank_results(results, documents, query):
    tokenized_docs = [[word.lower() for word in doc.split()] for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    query_tokens = [word.lower() for word in query.split()]
    bm25_scores = bm25.get_scores(query_tokens)
    reranked_results = sorted(results, key=lambda x: x['score'], reverse=True) 
    return reranked_results

def query_expansion(query_text):
    prompt = f"Expand the following query with relevant keywords: {query_text}"
    expanded_query = query_local_lm(prompt)
    return expanded_query

def filter_results(results, filter_criteria):
    filtered_results = [result for result in results if all(result['metadata'].get(key) == value for key, value in filter_criteria.items())]
    return filtered_results

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

def contextual_summarization(context):
    summarization_prompt = f"Summarize the following text:\n{context}"
    summary = query_local_lm(summarization_prompt) 
    return summary

def rag_query(query, n_results=3, filter_criteria=None):
    results = hybrid_search(query, n_results)
    if filter_criteria:
        results = filter_results(results, filter_criteria)
    context = "\n".join(results) if results else ""
    summarized_context = contextual_summarization(context)
    prompt = f"""
    You are a helpful and informative assistant.
    Use the following context to answer the question.
    If you can't find the answer, say "I'm not sure."

    Context: {summarized_context}

    Question: {query}

    Answer: 
    """
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

        response = rag_query(query)

        print("\nAnswer:", response)