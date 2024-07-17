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
import speech_recognition as sr
import schedule
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get the current working directory
current_dir = os.getcwd()
db_dir = os.path.join(current_dir, "chroma_db")

# Initialize ChromaDB
client = chromadb.Client(Settings(persist_directory=db_dir))
collection = client.get_or_create_collection("screenshots_collection")

# Initialize other components
model = SentenceTransformer('all-MiniLM-L6-v2')
reader = easyocr.Reader(['en'])
nlp = spacy.load("en_core_web_sm")

# Queue for communication between threads
screenshot_queue = Queue()

# User feedback storage
user_feedback = {}

def capture_screenshots():
    while True:
        try:
            screenshot = ImageGrab.grab()
            screenshot_queue.put((screenshot, time.time()))
            time.sleep(10)  # Capture a screenshot every 10 seconds
        except Exception as e:
            logging.error(f"Error capturing screenshot: {e}")

def process_screenshots():
    while True:
        try:
            screenshot, timestamp = screenshot_queue.get()
            image_np = np.array(screenshot)
            results = reader.readtext(image_np)
            text = " ".join([detection[1] for detection in results])
            add_documents([text], metadatas=[{"timestamp": timestamp}], ids=[f"doc_{timestamp}"])
            logging.info(f"Processed screenshot from {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))}")
        except Exception as e:
            logging.error(f"Error processing screenshot: {e}")

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
            else:
                logging.error(f"Error querying local LLM: {e}")
    return f"Error: Unable to get a response from the local LLM after {retries} attempts."

def check_database_population():
    try:
        results = collection.query(query_embeddings=model.encode(["test query"]).tolist(), n_results=1)
        logging.info(f"Database query test result: {results}")
        logging.info(f"Number of items in collection: {collection.count()}")
    except Exception as e:
        logging.error(f"Error querying database: {e}")

def get_voice_input(duration=5):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak your query (max 5 seconds)...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = recognizer.listen(source, timeout=duration)
            text = recognizer.recognize_google(audio)
            print(f"Recognized query: {text}")
            return text
        except sr.WaitTimeoutError:
            print("No speech detected within the time limit.")
        except sr.UnknownValueError:
            print("Speech recognition could not understand the audio.")
        except sr.RequestError as e:
            print(f"Could not request results from the speech recognition service; {e}")
    return None

def schedule_tasks():
    schedule.every(1).hour.do(analyze_recent_activity)
    schedule.every().day.at("09:00").do(daily_summary)

    while True:
        schedule.run_pending()
        time.sleep(1)

def analyze_recent_activity():
    recent_docs = query_documents("", n_results=10)
    summary = query_local_lm("Summarize the recent screen activity:", "\n".join([doc for doc, _ in recent_docs['documents']]))
    logging.info(f"Recent Activity Summary: {summary}")

def daily_summary():
    daily_docs = query_documents("", n_results=100)
    summary = query_local_lm("Provide a comprehensive summary of today's screen activity:", "\n".join([doc for doc, _ in daily_docs['documents']]))
    logging.info(f"Daily Summary: {summary}")

def enhanced_query_understanding(query):
    doc = nlp(query)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ']]
    return {"entities": entities, "keywords": keywords}

def proactive_retrieval():
    recent_docs = query_documents("", n_results=5)
    recent_text = " ".join([doc for doc, _ in recent_docs['documents']])
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([recent_text])
    feature_names = vectorizer.get_feature_names_out()
    important_terms = [feature_names[i] for i in tfidf_matrix.sum(axis=0).argsort()[0, -5:]]
    
    proactive_query = " ".join(important_terms)
    results = query_documents(proactive_query, n_results=3)
    
    print(f"Proactive Information: Based on recent activity, you might be interested in:")
    for doc, timestamp in results['documents']:
        print(f"- {doc[:100]}...")

def collect_user_feedback(query, response, rating):
    user_feedback[query] = {"response": response, "rating": rating}

def adaptive_query(query):
    similar_queries = []
    for past_query, data in user_feedback.items():
        similarity = cosine_similarity(
            model.encode([query]).reshape(1, -1),
            model.encode([past_query]).reshape(1, -1)
        )[0][0]
        if similarity > 0.8:
            similar_queries.append((past_query, data['rating']))
    
    if similar_queries:
        best_query = max(similar_queries, key=lambda x: x[1])[0]
        logging.info(f"Adapting query based on past feedback. Original: '{query}', Adapted: '{best_query}'")
        return best_query
    return query

def perform_web_search(query):
    search_url = f"https://api.duckduckgo.com/?q={query}&format=json"
    try:
        response = requests.get(search_url)
        response.raise_for_status()
        results = response.json()
        return results.get('AbstractText', 'No clear result found')
    except requests.RequestException as e:
        logging.error(f"Error performing web search: {e}")
        return "Failed to perform web search"

def query_interface():
    while True:
        print("\nAgentic Screenshot Assistant")
        print("1. Query stored screenshots")
        print("2. Proactive information retrieval")
        print("3. Perform web search")
        print("4. Check database status")
        print("5. Exit")
        choice = input("Enter your choice (1-5): ")

        if choice == '1':
            query = input("Enter your query: ")
            enhanced_query = enhanced_query_understanding(query)
            print(f"Understood query: Entities - {enhanced_query['entities']}, Keywords - {enhanced_query['keywords']}")
            
            adapted_query = adaptive_query(query)
            results = query_documents(adapted_query, n_results=5)
            
            context = "\n".join([doc for doc, _ in results['documents']])
            response = query_local_lm(adapted_query, context)
            
            print(f"\nAssistant: {response}")
            
            rating = int(input("Please rate the response (1-5): "))
            collect_user_feedback(query, response, rating)

        elif choice == '2':
            proactive_retrieval()

        elif choice == '3':
            query = input("Enter your web search query: ")
            result = perform_web_search(query)
            print(f"Web Search Result: {result}")

        elif choice == '4':
            check_database_population()

        elif choice == '5':
            print("Exiting the program.")
            return

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    logging.info(f"Script running from directory: {current_dir}")
    
    if os.path.exists(db_dir):
        logging.info(f"ChromaDB directory found at: {db_dir}")
    else:
        logging.info(f"ChromaDB directory not found. It will be created at: {db_dir}")
    
    # Start the screenshot capture thread
    capture_thread = threading.Thread(target=capture_screenshots, daemon=True)
    capture_thread.start()

    # Start the screenshot processing thread
    process_thread = threading.Thread(target=process_screenshots, daemon=True)
    process_thread.start()

    # Start the task scheduler
    scheduler_thread = threading.Thread(target=schedule_tasks, daemon=True)
    scheduler_thread.start()

    # Run the query interface in the main thread
    query_interface()