import os
import time
import threading
from queue import Queue
from PIL import ImageGrab, Image
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
import pyautogui
from io import BytesIO
import base64
import tkinter as tk
import psutil
import win32api
import win32con
import win32gui
import win32process
import winreg
import subprocess 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

current_dir = os.getcwd()
db_dir = os.path.join(current_dir, "chroma_db")

client = chromadb.Client(Settings(persist_directory=db_dir))
collection = client.get_or_create_collection("screenshots_collection")

model = SentenceTransformer('all-MiniLM-L6-v2')
reader = easyocr.Reader(['en'])
nlp = spacy.load("en_core_web_sm")

screenshot_queue = Queue()
user_feedback = {}
latest_snippet = None

def capture_screenshots():
    while True:
        try:
            screenshot = ImageGrab.grab()
            screenshot_queue.put((screenshot, time.time()))
            time.sleep(10)
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

def file_system_operations():
    while True:
        print("\nFile System Operations:")
        print("1. List files in a directory")
        print("2. Create a new file")
        print("3. Delete a file")
        print("4. Read file content")
        print("5. Write to a file")
        print("6. Return to main menu")
        
        choice = input("Enter your choice (1-6): ")
        
        if choice == '1':
            directory = input("Enter directory path: ")
            try:
                files = os.listdir(directory)
                print(f"Files in {directory}:")
                for file in files:
                    print(file)
            except Exception as e:
                print(f"Error listing files: {e}")
        
        elif choice == '2':
            file_path = input("Enter file path to create: ")
            try:
                with open(file_path, 'w') as f:
                    pass
                print(f"File created: {file_path}")
            except Exception as e:
                print(f"Error creating file: {e}")
        
        elif choice == '3':
            file_path = input("Enter file path to delete: ")
            try:
                os.remove(file_path)
                print(f"File deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting file: {e}")
        
        elif choice == '4':
            file_path = input("Enter file path to read: ")
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                print(f"File content:\n{content}")
            except Exception as e:
                print(f"Error reading file: {e}")
        
        elif choice == '5':
            file_path = input("Enter file path to write to: ")
            content = input("Enter content to write: ")
            try:
                with open(file_path, 'w') as f:
                    f.write(content)
                print(f"Content written to: {file_path}")
            except Exception as e:
                print(f"Error writing to file: {e}")
        
        elif choice == '6':
            break
        
        else:
            print("Invalid choice. Please try again.")

def process_management():
    while True:
        print("\nProcess Management:")
        print("1. List running processes")
        print("2. Start a process")
        print("3. Stop a process")
        print("4. Get process details")
        print("5. Return to main menu")
        
        choice = input("Enter your choice (1-5): ")
        
        if choice == '1':
            for proc in psutil.process_iter(['pid', 'name', 'status']):
                print(f"PID: {proc.info['pid']}, Name: {proc.info['name']}, Status: {proc.info['status']}")
        
        elif choice == '2':
            process_name = input("Enter process name to start: ")
            try:
                os.startfile(process_name)
                print(f"Started process: {process_name}")
            except Exception as e:
                print(f"Error starting process: {e}")
        
        elif choice == '3':
            pid = input("Enter PID of process to stop: ")
            try:
                process = psutil.Process(int(pid))
                process.terminate()
                print(f"Terminated process with PID: {pid}")
            except Exception as e:
                print(f"Error stopping process: {e}")
        
        elif choice == '4':
            pid = input("Enter PID for process details: ")
            try:
                process = psutil.Process(int(pid))
                print(f"Process details for PID {pid}:")
                print(f"Name: {process.name()}")
                print(f"Status: {process.status()}")
                print(f"CPU Usage: {process.cpu_percent()}%")
                print(f"Memory Usage: {process.memory_info().rss / (1024 * 1024):.2f} MB")
            except Exception as e:
                print(f"Error getting process details: {e}")
        
        elif choice == '5':
            break
        
        else:
            print("Invalid choice. Please try again.")

def basic_automation():
    while True:
        print("\nBasic Automation:")
        print("1. Move mouse to position")
        print("2. Click at current position")
        print("3. Type text")
        print("4. Capture screen region")
        print("5. Return to main menu")
        
        choice = input("Enter your choice (1-5): ")
        
        if choice == '1':
            x = int(input("Enter x coordinate: "))
            y = int(input("Enter y coordinate: "))
            pyautogui.moveTo(x, y)
            print(f"Moved mouse to ({x}, {y})")
        
        elif choice == '2':
            pyautogui.click()
            print("Clicked at current position")
        
        elif choice == '3':
            text = input("Enter text to type: ")
            pyautogui.typewrite(text)
            print(f"Typed: {text}")
        
        elif choice == '4':
            left = int(input("Enter left coordinate: "))
            top = int(input("Enter top coordinate: "))
            width = int(input("Enter width: "))
            height = int(input("Enter height: "))
            screenshot = pyautogui.screenshot(region=(left, top, width, height))
            screenshot.save("captured_region.png")
            print("Screen region captured and saved as 'captured_region.png'")
        
        elif choice == '5':
            break
        
        else:
            print("Invalid choice. Please try again.")

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

def select_area():
    root = tk.Tk()
    root.attributes('-alpha', 0.3)
    root.attributes('-fullscreen', True)
    root.configure(background='grey')

    x1, y1, x2, y2 = 0, 0, 0, 0
    rect = None

    def on_mouse_down(event):
        nonlocal x1, y1, rect
        x1, y1 = event.x, event.y
        rect = canvas.create_rectangle(x1, y1, x1, y1, outline='red')

    def on_mouse_move(event):
        nonlocal rect, x2, y2
        x2, y2 = event.x, event.y
        canvas.coords(rect, x1, y1, x2, y2)

    def on_mouse_up(event):
        nonlocal x2, y2
        x2, y2 = event.x, event.y
        root.quit()

    canvas = tk.Canvas(root, cursor="cross")
    canvas.pack(fill=tk.BOTH, expand=True)

    canvas.bind("<ButtonPress-1>", on_mouse_down)
    canvas.bind("<B1-Motion>", on_mouse_move)
    canvas.bind("<ButtonRelease-1>", on_mouse_up)

    root.mainloop()
    root.destroy()

    return min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)

def capture_snippet():
    global latest_snippet
    try:
        print("Select the area for the snippet...")
        x, y, width, height = select_area()
        screenshot = pyautogui.screenshot(region=(x, y, width, height))
        timestamp = time.time()
        latest_snippet = (screenshot, timestamp)
        process_snippet(screenshot, timestamp)
        print("Snippet captured and processed.")
    except Exception as e:
        logging.error(f"Error capturing snippet: {e}")

def process_snippet(screenshot, timestamp):
    try:
        image_np = np.array(screenshot)
        results = reader.readtext(image_np)
        text = " ".join([detection[1] for detection in results])
        add_documents([text], metadatas=[{"timestamp": timestamp, "type": "snippet"}], ids=[f"snippet_{timestamp}"])
        logging.info(f"Processed snippet from {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))}")
    except Exception as e:
        logging.error(f"Error processing snippet: {e}")

def query_snippet(query_text, n_results=5):
    if latest_snippet is None:
        return {"error": "No snippet available. Please capture a snippet first."}
    
    query_embedding = model.encode([query_text]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        where={"type": "snippet"},
        include=["documents", "metadatas"]
    )
    documents_with_timestamps = [(doc, metadata.get("timestamp", None)) for doc, metadata in zip(results["documents"][0], results["metadatas"][0])]
    return {"documents": documents_with_timestamps}

def get_latest_snippet_image():
    if latest_snippet is None:
        return None
    screenshot, _ = latest_snippet
    buffered = BytesIO()
    screenshot.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def get_memory_intensive_processes(threshold_mb=500):
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
        mem_info = proc.info['memory_info']
        mem_mb = mem_info.rss / (1024 * 1024)
        if mem_mb > threshold_mb:
            processes.append(f"Process: {proc.info['name']}, PID: {proc.info['pid']}, Memory: {mem_mb:.2f} MB")
    return processes

def start_process(process_name, url=None):
    try:
        if url:
            subprocess.Popen([process_name, url])
        else:
            subprocess.Popen(process_name)
        logging.info(f"Started process: {process_name}")
    except FileNotFoundError:
        logging.error(f"Process not found: {process_name}")
    except Exception as e:
        logging.error(f"Error starting process: {e}")

def close_processes(process_name):
    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'].lower() == process_name.lower():
            try:
                process = psutil.Process(proc.info['pid'])
                process.terminate()
                logging.info(f"Terminated process: {process_name}, PID: {proc.info['pid']}")
            except Exception as e:
                logging.error(f"Error closing process: {e}")

def monitor_cpu_usage(process_name, threshold_percent=80):
    while True:
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
            if proc.info['name'].lower() == process_name.lower():
                cpu_usage = proc.info['cpu_percent']
                if cpu_usage > threshold_percent:
                    logging.warning(f"High CPU usage alert: {process_name}, CPU: {cpu_usage}%")
        time.sleep(5) 

def list_background_processes():
    background_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
        if proc.info['name'] not in ["conhost.exe", "svchost.exe", "dwm.exe"]: 
            background_processes.append(
                f"Process: {proc.info['name']}, PID: {proc.info['pid']}, "
                f"CPU: {proc.info['cpu_percent']:.2f}%, Memory: {proc.info['memory_percent']:.2f}%"
            )
    return background_processes

def restart_service(service_name):
    try:
        subprocess.run(["net", "stop", service_name], check=True)
        subprocess.run(["net", "start", service_name], check=True)
        logging.info(f"Restarted service: {service_name}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error restarting service: {e}")

def process_voice_command(command):
    logging.info(f"Processing voice command: {command}")
    if "memory" in command and "process" in command:
        try:
            threshold = int("".join([char for char in command if char.isdigit()]))
            processes = get_memory_intensive_processes(threshold)
            if processes:
                print("Memory-intensive processes:")
                for process in processes:
                    print(process)
            else:
                print(f"No processes found using more than {threshold}MB of memory.")
        except ValueError:
            print("Please specify a valid memory threshold in MB (e.g., 'processes using more than 500 MB').")

    elif "start" in command:
        process_name = command.split("start", 1)[1].strip()
        if "and navigate to" in process_name:
            process_name, url = process_name.split("and navigate to", 1)
            start_process(process_name.strip(), url.strip())
        else:
            start_process(process_name)

    elif "close" in command:
        process_name = command.split("close", 1)[1].strip()
        close_processes(process_name)

    elif "monitor" in command and "cpu" in command:
        process_name = command.split("cpu", 1)[1].replace("of", "").strip()
        try:
            threshold = int("".join([char for char in command if char.isdigit()]))
            threading.Thread(target=monitor_cpu_usage, args=(process_name, threshold), daemon=True).start()
            print(f"Monitoring CPU usage of '{process_name}'. Alert at {threshold}% usage.")
        except ValueError:
            print("Please specify a valid CPU threshold percentage (e.g., 'exceeds 80%').")

    elif "background" in command and "process" in command:
        processes = list_background_processes()
        if processes:
            print("Background Processes:")
            for process in processes:
                print(process)
        else:
            print("No background processes found.")
    
    elif "restart" in command and "service" in command:
        service_name = command.split("service", 1)[1].strip()
        restart_service(service_name)

    else:
        print("Sorry, I didn't understand that command.")

def process_text_command(command):
    logging.info(f"Processing text command: {command}")
    if "memory" in command and "process" in command:
        try:
            threshold = int("".join([char for char in command if char.isdigit()]))
            processes = get_memory_intensive_processes(threshold)
            if processes:
                print("Memory-intensive processes:")
                for process in processes:
                    print(process)
            else:
                print(f"No processes found using more than {threshold}MB of memory.")
        except ValueError:
            print("Please specify a valid memory threshold in MB (e.g., 'processes using more than 500 MB').")

    elif "start" in command:
        process_name = command.split("start", 1)[1].strip()
        if "and navigate to" in process_name:
            process_name, url = process_name.split("and navigate to", 1)
            start_process(process_name.strip(), url.strip())
        else:
            start_process(process_name)

    elif "close" in command:
        process_name = command.split("close", 1)[1].strip()
        close_processes(process_name)

    elif "monitor" in command and "cpu" in command:
        process_name = command.split("cpu", 1)[1].replace("of", "").strip()
        try:
            threshold = int("".join([char for char in command if char.isdigit()]))
            threading.Thread(target=monitor_cpu_usage, args=(process_name, threshold), daemon=True).start()
            print(f"Monitoring CPU usage of '{process_name}'. Alert at {threshold}% usage.")
        except ValueError:
            print("Please specify a valid CPU threshold percentage (e.g., 'exceeds 80%').")

    elif "background" in command and "process" in command:
        processes = list_background_processes()
        if processes:
            print("Background Processes:")
            for process in processes:
                print(process)
        else:
            print("No background processes found.")

    elif "restart" in command and "service" in command:
        service_name = command.split("service", 1)[1].strip()
        restart_service(service_name)

    else:
        print("Sorry, I didn't understand that command.")

def query_interface():
    global latest_snippet
    while True:
        print("\nAgentic Screenshot Assistant")
        print("1. Query stored screenshots")
        print("2. Proactive information retrieval")
        print("3. Perform web search")
        print("4. Check database status")
        print("5. Capture snippet")
        print("6. Query latest snippet")
        print("7. File system operations")
        print("8. Process management")
        print("9. Basic automation")
        print("10. Voice Command")
        print("11. Text Command")
        print("12. Exit")
        choice = input("Enter your choice (1-12): ")

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
            capture_snippet()

        elif choice == '6':
            if latest_snippet is None:
                print("No snippet available. Please capture a snippet first.")
            else:
                query = input("Enter your query about the latest snippet: ")
                results = query_snippet(query)
                context = "\n".join([doc for doc, _ in results['documents']])
                response = query_local_lm(query, context)
                print(f"\nAssistant: {response}")
        elif choice == '7':
            file_system_operations()
        elif choice == '8':
            process_management()
        elif choice == '9':
            basic_automation()
        elif choice == '10':
            command = get_voice_input()
            if command:
                process_voice_command(command) 
        elif choice == '11':
            command = input("Enter your text command: ")
            process_text_command(command) 
        elif choice == '12':
            print("Exiting the program.")
            return
        else:
            print("Invalid choice. Please try again.")
        time.sleep(1) # Add a small delay to avoid consuming too much CPU
if __name__ == "__main__":
    logging.info(f"Script running from directory: {current_dir}")
    
    if os.path.exists(db_dir):
        logging.info(f"ChromaDB directory found at: {db_dir}")
    else:
        logging.info(f"ChromaDB directory not found. It will be created at: {db_dir}")
    
    capture_thread = threading.Thread(target=capture_screenshots, daemon=True)
    capture_thread.start()

    process_thread = threading.Thread(target=process_screenshots, daemon=True)
    process_thread.start()

    scheduler_thread = threading.Thread(target=schedule_tasks, daemon=True)
    scheduler_thread.start()

    query_interface()