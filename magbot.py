import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

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

# Function to handle chatbot response
def chatbot_response(text):
    # Predict the intent of user input
    intents_list = predict_class(text, model)
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
