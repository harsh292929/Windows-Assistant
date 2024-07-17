import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import requests
import json
import logging
import PyPDF2
from tqdm import tqdm
import networkx as nx
import time
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

client = chromadb.Client(Settings(persist_directory="./chroma_db"))
collection = client.get_or_create_collection("my_collection")
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(file_path, chunk_size=500, chunk_overlap=50):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        text_chunks = []
        for page_num in tqdm(range(num_pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            for i in range(0, len(text), chunk_size - chunk_overlap):
                chunk = text[i:i + chunk_size]
                text_chunks.append(chunk)
    return text_chunks

def generate_multi_vectors(text):
    full_text_vector = model.encode(text).tolist()
    first_sentence_vector = model.encode(text.split('.')[0]).tolist()
    keywords = " ".join([word for word in text.split() if len(word) > 5])
    keyword_vector = model.encode(keywords).tolist()
    return [full_text_vector, first_sentence_vector, keyword_vector]

def add_documents(texts, metadatas=None, ids=None):
    if ids is None:
        ids = [f"doc_{i}" for i in range(len(texts))]
    for i, text in enumerate(texts):
        multi_vectors = generate_multi_vectors(text)
        for j, vector in enumerate(multi_vectors):
            collection.add(
                embeddings=[vector],
                documents=[text],
                metadatas=[metadatas[i] if metadatas else None],
                ids=[f"{ids[i]}_aspect_{j}"]
            )

def query_expansion(query):
    prompt = f"Generate 3 alternative phrasings or related terms for the following query: '{query}'"
    expansions = query_local_lm(prompt).split('\n')
    return [query] + expansions

def query_documents(query_text, n_results=5):
    expanded_queries = query_expansion(query_text)
    all_results = []
    for query in expanded_queries:
        query_embedding = model.encode([query]).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=n_results)
        all_results.extend(results['documents'][0])
    unique_results = list(dict.fromkeys(all_results))
    return {'documents': [unique_results[:n_results]]}

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


def build_knowledge_graph(documents):
    G = nx.Graph()
    for doc in documents:
        entities = query_local_lm(f"Extract key entities (people, places, organizations, etc.) from this text: {doc}").split('\n')
        for entity in entities:
            G.add_node(entity)
        relations = query_local_lm(f"Identify relationships between entities in this text: {doc}").split('\n')
        for relation in relations:
            if relation.count(',') == 2:
                subject, predicate, object = relation.split(',')
                G.add_edge(subject.strip(), object.strip(), predicate=predicate.strip())
            else:
                logging.warning(f"Skipping invalid relation: {relation}")
    return G

def augment_context_with_graph(query, context, graph):
    relevant_nodes = []
    for node in graph.nodes:
        if node.lower() in query.lower() or any(node.lower() in c.lower() for c in context):
            relevant_nodes.append(node)
    subgraph = graph.subgraph(relevant_nodes)
    extra_context = "\n".join([f"{u} {attrs.get('predicate', '')} {v}" for u, v, attrs in subgraph.edges(data=True)])
    return context + "\n" + extra_context

def visualize_graph(graph, query=None):
    plt.figure(figsize=(15, 10))  # Larger canvas for better readability

    # Node positioning and layout
    pos = nx.spring_layout(graph, k=0.3, iterations=50, seed=42)  # More spread-out layout

    # Color Palette
    node_colors = {
        "Person": "#FF5733",    # Reddish for people
        "Location": "#3498DB",  # Blue for locations
        "Organization": "#2ECC71",  # Green for organizations
        "Other": "#F1C40F"      # Yellow for others
    }

    edge_colors = "#888888"  # Grey for edges

    # Node Styling
    node_size = 2000
    node_border_color = "#333333"  # Dark grey border
    node_border_width = 1.5

    # Edge Styling
    edge_width = 1.2

    # Node Labels (with entity types)
    node_labels = {}
    for node in graph.nodes:
        entity_type = query_local_lm(f"Classify the entity type of '{node}': (Person, Location, Organization, or Other)").strip()
        node_labels[node] = f"{node}\n({entity_type})"  # Add type below node name

    # Drawing Nodes and Edges
    nx.draw_networkx_nodes(
        graph, pos,
        node_size=node_size,
        node_color=[node_colors.get(node_labels[node].split("\n")[1][1:-1], node_colors["Other"]) for node in graph.nodes],
        edgecolors=node_border_color,
        linewidths=node_border_width
    )
    nx.draw_networkx_edges(graph, pos, width=edge_width, edge_color=edge_colors)
    nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=12, font_family="sans-serif")

    # Edge Labels (Predicate)
    edge_labels = nx.get_edge_attributes(graph, 'predicate')
    nx.draw_networkx_edge_labels(
        graph, pos, 
        edge_labels=edge_labels, 
        font_size=10, 
        font_color="#555555",   # Slightly darker grey for edge labels
        bbox=dict(facecolor='white', edgecolor='white', alpha=0.8)  # Box around edge labels
    )

    # Title & Background
    plt.title(
        f"Knowledge Graph Visualization (Query: '{query}')" if query else "Knowledge Graph Visualization",
        fontsize=16,
        fontweight="bold"
    )
    plt.gca().set_facecolor("#F0F0F0")  # Light grey background

    plt.axis("off")  # Remove axis lines
    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()


def rag_query_with_graph(query, n_results=3):
    results = query_documents(query, n_results)
    context = "\n".join(results['documents'][0])
    graph = build_knowledge_graph(results['documents'][0])
    augmented_context = augment_context_with_graph(query, context, graph)
    prompt = f"""Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {augmented_context}

Question: {query} Answer:"""
    response = query_local_lm(prompt)
    return response


def add_pdf_documents(file_path):
    text_chunks = extract_text_from_pdf(file_path)
    add_documents(texts=text_chunks)

if __name__ == "__main__":
    pdf_file_path = "your_document1.pdf"
    add_pdf_documents(pdf_file_path)

    while True:
        query = input("Enter your query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        answer = rag_query_with_graph(query)
        print("\nAnswer:", answer)

        results = query_documents(query, n_results=3)
        graph = build_knowledge_graph(results['documents'][0])
        visualize_graph(graph, query)
