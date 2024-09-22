# from flask import Flask, request, jsonify, render_template, send_from_directory
# import sqlite3
# import hashlib
# import json
# import pandas as pd
# import nltk
# from nltk.stem import WordNetLemmatizer
# from tensorflow.keras.models import load_model
# import numpy as np
# import random
# import csv
# import os
# import pickle

# app = Flask(__name__)

# # Load necessary NLTK data
# nltk.download('punkt')
# nltk.download('wordnet')

# # Initialize the lemmatizer
# lemmatizer = WordNetLemmatizer()

# # Load the trained model
# model = load_model('magbot_model.keras')

# # Load intents and preprocessed data
# with open('intents.json', 'r') as f:
#     intents = json.load(f)

# words = pickle.load(open('words.pkl', 'rb'))
# classes = pickle.load(open('classes.pkl', 'rb'))

# # Connect to the SQLite database
# conn = sqlite3.connect('magbot.db', check_same_thread=False)
# cursor = conn.cursor()

# # For managing user context and leave requests
# context_state = {}
# leave_requests = {}

# # Function to hash passwords
# def hash_password(password):
#     return hashlib.sha256(password.encode()).hexdigest()

# # Function to preprocess user input
# def clean_up_sentence(sentence):
#     sentence_words = nltk.word_tokenize(sentence)
#     sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
#     return sentence_words

# # Function to convert user input into a bag of words
# def bag_of_words(sentence, words):
#     sentence_words = clean_up_sentence(sentence)
#     bag = [0] * len(words)
#     for s in sentence_words:
#         for i, w in enumerate(words):
#             if w == s:
#                 bag[i] = 1
#     return bag

# # Function to predict the class of user input
# def predict_class(sentence, model):
#     bow = bag_of_words(sentence, words)
#     bow = np.array([bow])  # Convert list to NumPy array
#     res = model.predict(bow)[0]
#     ERROR_THRESHOLD = 0.25
#     results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
#     results.sort(key=lambda x: x[1], reverse=True)
#     return_list = [{"intent": classes[r[0]], "probability": str(r[1]), "input": sentence} for r in results]
#     return return_list

# # Function to format reason for leave in a grammatically correct manner
# def format_reason(reason):
#     words = clean_up_sentence(reason)
#     return ' '.join(words).capitalize() + '.'

# # Function to extract leave details from user input
# def extract_leave_details(sentence):
#     leave_type = "general"
#     if "vacation" in sentence.lower():
#         leave_type = "vacation"
#     elif "sick" in sentence.lower():
#         leave_type = "sick"
    
#     leave_duration = [int(s) for s in sentence.split() if s.isdigit()]
#     leave_duration = leave_duration[0] if leave_duration else 1
    
#     reason = sentence
    
#     return leave_type, leave_duration, reason

# # Function to save leave requests to the database
# def save_leave_request_to_db(user_id, leave_type, leave_duration, reason):
#     cursor.execute("INSERT INTO leave_requests (user_id, leave_type, leave_duration, reason) VALUES (?, ?, ?, ?)",
#                    (user_id, leave_type, leave_duration, reason))
#     conn.commit()

# # Function to save leave requests to CSV
# def save_leave_request_to_csv(username, leave_duration, reason, approval='pending'):
#     formatted_reason = format_reason(reason)
#     with open('leave_requests.csv', mode='a', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow([username, formatted_reason, leave_duration, approval])

# # Function to get a response based on predicted intent and context
# def get_response(intents_list, intents_json, user_id='user'):
#     tag = intents_list[0]['intent'] if intents_list else 'fallback'
#     list_of_intents = intents_json['intents']

#     if user_id in context_state and context_state[user_id] in ["awaiting_leave_details", "confirm_leave_request"]:
#         if tag == "provide_leave_details":
#             leave_type, leave_duration, reason = extract_leave_details(intents_list[0]['input'])
#             leave_requests[user_id] = {"type": leave_type, "duration": leave_duration, "reason": reason}
#             context_state[user_id] = "confirm_leave_request"
#             return f"Got it. You want a {leave_type} leave for {leave_duration} days due to '{reason}'. Should I proceed with this request?"

#         if tag == "confirm_leave":
#             if "submit" in intents_list[0]['input'].lower() or "yes" in intents_list[0]['input'].lower():
#                 context_state[user_id] = None
#                 # Fetch username for CSV file storage
#                 cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
#                 username = cursor.fetchone()[0]
                
#                 # Save leave request to DB and CSV
#                 save_leave_request_to_db(user_id, leave_requests[user_id]['type'], leave_requests[user_id]['duration'], leave_requests[user_id]['reason'])
#                 save_leave_request_to_csv(username, leave_requests[user_id]['duration'], leave_requests[user_id]['reason'])
                
#                 return f"Your leave request for {leave_requests[user_id]['duration']} days due to '{leave_requests[user_id]['reason']}' has been submitted. The admin will validate it shortly."
#             else:
#                 context_state[user_id] = "awaiting_leave_details"
#                 return "Okay, I will change the request details. Please provide the new details."

#     for i in list_of_intents:
#         if i['tag'] == tag:
#             result = random.choice(i['responses'])
#             if 'context_set' in i:
#                 context_state[user_id] = i['context_set']
#             break

#     if tag == 'fallback':
#         result = "I'm not sure I understand. Could you please clarify or ask something else?"

#     return result

# # Function to handle chatbot response
# def chatbot_response(text, user_id='user'):
#     intents_list = predict_class(text, model)
#     response = get_response(intents_list, intents, user_id)
#     return response

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/login', methods=['POST'])
# def login():
#     username = request.form['username']
#     password = request.form['password']
#     cursor.execute("SELECT id FROM users WHERE username = ? AND password = ?", (username, hash_password(password)))
#     user = cursor.fetchone()
#     if user:
#         return jsonify({'status': 'success', 'user_id': user[0]})
#     return jsonify({'status': 'error', 'message': 'Invalid credentials'})

# @app.route('/signup', methods=['POST'])
# def signup():
#     username = request.form['username']
#     password = request.form['password']
#     try:
#         cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hash_password(password)))
#         conn.commit()
#         return jsonify({'status': 'success', 'message': 'Signup successful! Please log in.'})
#     except sqlite3.IntegrityError:
#         return jsonify({'status': 'error', 'message': 'Username already exists. Please choose a different username.'})

# @app.route('/chat', methods=['POST'])
# def chat():
#     user_id = request.form['user_id']
#     message = request.form['message']
#     response = chatbot_response(message, user_id=user_id)
#     return jsonify({'response': response})

# # Serve static files
# @app.route('/')
# def index():
#     return send_from_directory('static', 'leave_request.html')

# # Endpoint to get CSV data
# @app.route('/get_csv_data', methods=['GET'])
# def get_csv_data():
#     df = pd.read_csv('leave_requests.csv')
#     data = df.to_dict(orient='records')
#     return jsonify(data)

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, jsonify, render_template, send_from_directory
import sqlite3
import hashlib
import json
import pandas as pd
import nltk
import subprocess
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import numpy as np
import random
import csv
import os
import pickle

app = Flask(__name__)

# Load necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the trained model
model = load_model('magbot_model.keras')

# Load intents and preprocessed data
with open('intents.json', 'r') as f:
    intents = json.load(f)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Connect to the SQLite database
conn = sqlite3.connect('magbot.db', check_same_thread=False)
cursor = conn.cursor()

# For managing user context and leave requests
context_state = {}
leave_requests = {}

# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Function to preprocess user input
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Function to convert user input into a bag of words
def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return bag

# Function to predict the class of user input
def predict_class(sentence, model):
    bow = bag_of_words(sentence, words)
    bow = np.array([bow])  # Convert list to NumPy array
    res = model.predict(bow)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]], "probability": str(r[1]), "input": sentence} for r in results]
    return return_list

# Function to format reason for leave in a grammatically correct manner
def format_reason(reason):
    words = clean_up_sentence(reason)
    return ' '.join(words).capitalize() + '.'

# Function to extract leave details from user input
def extract_leave_details(sentence):
    leave_type = "general"
    if "vacation" in sentence.lower():
        leave_type = "vacation"
    elif "sick" in sentence.lower():
        leave_type = "sick"
    
    leave_duration = [int(s) for s in sentence.split() if s.isdigit()]
    leave_duration = leave_duration[0] if leave_duration else 1
    
    reason = sentence
    
    return leave_type, leave_duration, reason

# Function to save leave requests to the database
def save_leave_request_to_db(user_id, leave_type, leave_duration, reason):
    cursor.execute("INSERT INTO leave_requests (user_id, leave_type, leave_duration, reason) VALUES (?, ?, ?, ?)",
                   (user_id, leave_type, leave_duration, reason))
    conn.commit()

# Function to save leave requests to CSV
def save_leave_request_to_csv(username, leave_duration, reason, approval='pending'):
    formatted_reason = format_reason(reason)
    with open('leave_requests.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([username, formatted_reason, leave_duration, approval])

# Function to get a response based on predicted intent and context
def get_response(intents_list, intents_json, user_id='user'):
    tag = intents_list[0]['intent'] if intents_list else 'fallback'
    list_of_intents = intents_json['intents']

    if user_id in context_state and context_state[user_id] in ["awaiting_leave_details", "confirm_leave_request"]:
        if tag == "provide_leave_details":
            leave_type, leave_duration, reason = extract_leave_details(intents_list[0]['input'])
            leave_requests[user_id] = {"type": leave_type, "duration": leave_duration, "reason": reason}
            context_state[user_id] = "confirm_leave_request"
            return f"Got it. You want a {leave_type} leave for {leave_duration} days due to '{reason}'. Should I proceed with this request?"

        if tag == "confirm_leave":
            if "submit" in intents_list[0]['input'].lower() or "yes" in intents_list[0]['input'].lower():
                context_state[user_id] = None
                # Fetch username for CSV file storage
                cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
                username = cursor.fetchone()[0]
                
                # Save leave request to DB and CSV
                save_leave_request_to_db(user_id, leave_requests[user_id]['type'], leave_requests[user_id]['duration'], leave_requests[user_id]['reason'])
                save_leave_request_to_csv(username, leave_requests[user_id]['duration'], leave_requests[user_id]['reason'])
                
                return f"Your leave request for {leave_requests[user_id]['duration']} days due to '{leave_requests[user_id]['reason']}' has been submitted. The admin will validate it shortly."
            else:
                context_state[user_id] = "awaiting_leave_details"
                return "Okay, I will change the request details. Please provide the new details."

    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            if 'context_set' in i:
                context_state[user_id] = i['context_set']
            break

    if tag == 'fallback':
        result = "I'm not sure I understand. Could you please clarify or ask something else?"

    return result

# Function to handle chatbot response
def chatbot_response(text, user_id='user'):
    intents_list = predict_class(text, model)
    response = get_response(intents_list, intents, user_id)
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    cursor.execute("SELECT id FROM users WHERE username = ? AND password = ?", (username, hash_password(password)))
    user = cursor.fetchone()
    if user:
        return jsonify({'status': 'success', 'user_id': user[0]})
    return jsonify({'status': 'error', 'message': 'Invalid credentials'})

@app.route('/signup', methods=['POST'])
def signup():
    username = request.form['username']
    password = request.form['password']
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hash_password(password)))
        conn.commit()
        return jsonify({'status': 'success', 'message': 'Signup successful! Please log in.'})
    except sqlite3.IntegrityError:
        return jsonify({'status': 'error', 'message': 'Username already exists. Please choose a different username.'})

@app.route('/chat', methods=['POST'])
def chat():
    user_id = request.form['user_id']
    message = request.form['message']
    response = chatbot_response(message, user_id=user_id)
    return jsonify({'response': response})

def update_html_from_csv():
    try:
        # Run the csvtohtml.py script
        result = subprocess.run(['python', 'csvtohtml.py'], check=True, text=True, capture_output=True)
        print("CSV to HTML conversion successful:")
        print(result.stdout)  # Print the standard output from the script
    except subprocess.CalledProcessError as e:
        print(f"Error running csvtohtml.py: {e}")
        print(e.output)  # Print the standard output from the script (if any)
        print(e.stderr)  # Print the standard error from the script (if any)

@app.route('/leave_request')
def home():
    return render_template('leave_request.html')

if __name__ == '__main__':
    app.run(debug=True)
