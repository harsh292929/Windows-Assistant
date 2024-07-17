import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import requests
import json
import logging
import PyPDF2
from tqdm import tqdm
import time
import networkx as nx
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ChromaDB setup
chroma_client = chromadb.Client(Settings(persist_directory="./chroma_db"))
chroma_collection = chroma_client.get_or_create_collection("graphrag_collection")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Knowledge Graph
knowledge_graph = nx.Graph()

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

def create_knowledge_graph(texts):
    for i, text in enumerate(texts):
        prompt = f"Extract key entities and relationships from the following text:\n\n{text}\n\nProvide the output as a list of tuples: (entity1, relationship, entity2)"
        entities_and_relationships = query_local_lm(prompt)
        
        for item in entities_and_relationships.split('\n'):
            try:
                entity1, relationship, entity2 = eval(item.strip())
                knowledge_graph.add_edge(entity1, entity2, relationship=relationship)
            except:
                logging.warning(f"Failed to parse: {item}")

def create_semantic_clusters(texts, n_clusters=10):
    embeddings = model.encode(texts)
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = clustering.fit_predict(embeddings)
    return cluster_labels

def add_documents(texts, metadatas=None, ids=None):
    embeddings = model.encode(texts).tolist()
    if ids is None:
        ids = [f"doc_{i}" for i in range(len(texts))]
    
    # Add to ChromaDB
    chroma_collection.add(embeddings=embeddings, documents=texts, metadatas=metadatas, ids=ids)
    
    # Create knowledge graph
    create_knowledge_graph(texts)
    
    # Create semantic clusters
    cluster_labels = create_semantic_clusters(texts)
    
    # Add cluster information to metadatas
    if metadatas is None:
        metadatas = [{} for _ in texts]
    for i, cluster in enumerate(cluster_labels):
        metadatas[i]['cluster'] = int(cluster)
    
    # Update ChromaDB with cluster information
    chroma_collection.update(ids=ids, metadatas=metadatas)

def graphrag_search(query_text, n_results=5):
    # Semantic search using ChromaDB
    query_embedding = model.encode([query_text]).tolist()
    semantic_results = chroma_collection.query(query_embeddings=query_embedding, n_results=n_results)
    
    # Extract entities from the query
    query_entities = query_local_lm(f"Extract key entities from the following text:\n\n{query_text}\n\nProvide the output as a list of entities.")
    query_entities = [entity.strip() for entity in query_entities.split(',')]
    
    # Find relevant subgraph
    relevant_nodes = set()
    for entity in query_entities:
        if entity in knowledge_graph:
            relevant_nodes.update(nx.neighbors(knowledge_graph, entity))
    relevant_subgraph = knowledge_graph.subgraph(relevant_nodes)
    
    # Rank results based on graph centrality
    centrality = nx.degree_centrality(relevant_subgraph)
    
    # Combine semantic search results with graph-based ranking
    combined_results = []
    for i, doc in enumerate(semantic_results['documents'][0]):
        doc_id = semantic_results['ids'][0][i]
        metadata = semantic_results['metadatas'][0][i]
        cluster = metadata.get('cluster', -1)
        
        # Calculate graph-based score
        graph_score = sum(centrality.get(entity, 0) for entity in query_entities if entity in centrality)
        
        combined_results.append({
            'id': doc_id,
            'document': doc,
            'semantic_score': 1 - i/n_results,  # Normalize score
            'graph_score': graph_score,
            'cluster': cluster
        })
    
    # Sort by combined score (you can adjust the weights)
    sorted_results = sorted(combined_results, key=lambda x: (0.5 * x['semantic_score'] + 0.5 * x['graph_score']), reverse=True)
    
    return sorted_results[:n_results]

def rag_query(query, n_results=3):
    results = graphrag_search(query, n_results)
    context = "\n".join([result['document'] for result in results])
    
    # Extract cluster information
    clusters = set(result['cluster'] for result in results if result['cluster'] != -1)
    cluster_summaries = []
    for cluster in clusters:
        cluster_docs = [doc for doc in results if doc['cluster'] == cluster]
        cluster_text = "\n".join([doc['document'] for doc in cluster_docs])
        summary_prompt = f"Summarize the main theme of the following texts:\n\n{cluster_text}"
        cluster_summary = query_local_lm(summary_prompt)
        cluster_summaries.append(f"Cluster {cluster} summary: {cluster_summary}")
    
    cluster_context = "\n".join(cluster_summaries)
    
    prompt = f"""Use the following pieces of context and cluster summaries to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Cluster Summaries:
{cluster_context}

Question: {query}
Answer:"""
    
    response = query_local_lm(prompt)
    return response

def add_pdf_documents(file_path):
    text_chunks = extract_text_from_pdf(file_path)
    add_documents(texts=text_chunks)

def initialize_system():
    # Initialize ChromaDB collection
    global chroma_collection
    chroma_collection = chroma_client.get_or_create_collection("graphrag_collection")
    logging.info("ChromaDB collection initialized.")

def main():
    initialize_system()

    # Check if documents have been loaded
    if chroma_collection.count() == 0:
        pdf_dir = "pdfs"
        for filename in os.listdir(pdf_dir):
            if filename.endswith(".pdf"):
                file_path = os.path.join(pdf_dir, filename)
                logging.info(f"Processing {filename}")
                add_pdf_documents(file_path)
        logging.info("All PDFs processed and added to the system.")
    else:
        logging.info(f"Found {chroma_collection.count()} existing documents in the system.")

    while True:
        query = input("Enter your query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        answer = rag_query(query)
        print("\nAnswer:", answer)

if __name__ == "__main__":
    main()