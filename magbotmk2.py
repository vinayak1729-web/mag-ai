import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import requests  # For Gemini API requests

# Load necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the trained model
model = load_model('magbot_model.keras')  # Or 'magbot_model.h5' if you saved in HDF5 format

# Load intents and preprocessed data
with open('intents.json', 'r') as f:
    intents = json.load(f)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Function to preprocess user input
def clean_up_sentence(sentence):
    # Tokenize the sentence
    sentence_words = nltk.word_tokenize(sentence)
    # Lemmatize each word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Function to convert user input into a bag of words
def bag_of_words(sentence, words):
    # Tokenize and lemmatize user input
    sentence_words = clean_up_sentence(sentence)
    # Initialize bag with zeros
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:  # if the word is in our vocabulary
                bag[i] = 1
    return np.array(bag)

# Function to predict the class of user input
def predict_class(sentence, model):
    # Convert user input into a bag of words
    bow = bag_of_words(sentence, words)
    # Predict the intent
    res = model.predict(np.array([bow]))[0]
    # Only consider predictions above a certain threshold
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    # Sort by probability in descending order
    results.sort(key=lambda x: x[1], reverse=True)

    # Debugging: Print results for debugging
    print(f"Debug: Predicted results = {results}")

    # Return the predicted intent and its probability
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Function to get a response based on predicted intent
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            # Randomly select a response from the list of responses
            result = random.choice(i['responses'])
            break
    return result

# Function to update intents with new information
def update_intents(user_input, explanation):
    new_intent = {
        "tag": "user_defined_" + str(len(intents['intents']) + 1),
        "patterns": [user_input],
        "responses": [explanation]
    }
    intents['intents'].append(new_intent)

    # Save updated intents to intents.json
    with open('intents.json', 'w') as f:
        json.dump(intents, f, indent=4)
    
    # Retrain the model with the new intents
    retrain_model()

def retrain_model():
    # Placeholder for retraining the model with the updated intents.json
    # Add your model retraining code here.
    pass

# Function to search online using Gemini API
def search_online(query):
    gemini_api_key = "AIzaSyD7uo8yp4Bwd99SAiNyIzsxuA1SXhltj2Q"
    response = requests.get(f"https://api.gemini.com/v1/search?q={query}&key={gemini_api_key}")

    if response.status_code == 200:
        data = response.json()
        # Assuming the API response contains a 'summary' field
        return data.get('summary', '')
    else:
        print("Error: Unable to fetch data from Gemini.")
        return None

# Enhanced function to handle chatbot response
def chatbot_response(text):
    # Predict the intent of user input
    intents_list = predict_class(text, model)
    
    # Check if no intents are matched or if the highest probability is below the threshold
    if not intents_list or float(intents_list[0]['probability']) < 0.25:
        print("Debug: No matching intent found or confidence too low.")
        user_input = input("I don't know how to answer that. Can you explain it to me? (yes/no) ")
        
        if user_input.lower() == 'yes':
            explanation = input("Please provide the information: ")
            # Add the new intent to intents.json
            update_intents(text, explanation)
            return "Got it! I've learned something new."
        
        elif user_input.lower() == 'no':
            search_permission = input("Would you like me to search for this information online? (yes/no) ")
            if search_permission.lower() == 'yes':
                # Call the function to search online using the Gemini API
                explanation = search_online(text)
                if explanation:
                    update_intents(text, explanation)
                    return "I found some information online and learned something new!"
                else:
                    return "I couldn't find anything online."
            else:
                return "Alright, let me know if there's anything else I can help with."
        else:
            return "I didn't understand your response. Let's try again."
    else:
        # Get a response based on predicted intent
        response = get_response(intents_list, intents)
        return response

# Main loop for chatbot interaction
print("Magbot is ready to chat! (type 'quit' to exit)")
while True:
    message = input("You: ")
    if message.lower() == "quit":
        break

    # Get chatbot response
    response = chatbot_response(message)
    print("Magbot:", response)
