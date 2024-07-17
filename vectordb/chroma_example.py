import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import requests
import json

# Initialize Chroma client
client = chromadb.Client(Settings(persist_directory="./chroma_db"))

# Create or get a collection
collection = client.get_or_create_collection("my_collection")

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')
def generate_multi_vectors(text):
    full_text_vector = model.encode(text).tolist()
    first_sentence_vector = model.encode(text.split('.')[0]).tolist()
    keywords = " ".join([word for word in text.split() if len(word) > 5])
    keyword_vector = model.encode(keywords).tolist()
    return [full_text_vector, first_sentence_vector, keyword_vector]
# Function to add documents
def add_documents(texts, metadatas=None, ids=None):
    embeddings = model.encode(texts).tolist()
    collection.add(
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )

# Function to query documents
def query_documents(query_text, n_results=5):
    query_embedding = model.encode([query_text]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    return results

# Function to interact with your local Llama 3 model
def query_local_lm(prompt):
    url = "http://localhost:1234/v1/chat/completions"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": "local-model",  # This might need to be adjusted
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # Raise an exception for bad status codes
        result = response.json()
        return result['choices'][0]['message']['content']
    except requests.RequestException as e:
        print(f"Error querying local LLM: {e}")
        return f"Error: Unable to get a response from the local LLM. Details: {str(e)}"

# Function to perform RAG
def rag_query(query, n_results=3):
    # Retrieve relevant documents
    results = query_documents(query, n_results)
    
    # Construct prompt with retrieved information
    context = "\n".join(results['documents'][0])
    prompt = f"""Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {query}

Answer:"""
    
    # Generate response using local LM
    response = query_local_lm(prompt)
    
    return response

# Example usage
if __name__ == "__main__":
    # Add some documents
    add_documents(
        texts=[
            "The quick brown fox jumps over the lazy dog",
            "Machine learning is a subset of artificial intelligence",
            "Natural language processing deals with the interaction between computers and living beings using natural language"
        ],
        ids=["doc1", "doc2", "doc3"]
    )

    # Perform a RAG query
    query = "What is NLP?"
    response = rag_query(query)
    
    print(f"Query: {query}")
    print(f"RAG Response: {response}")