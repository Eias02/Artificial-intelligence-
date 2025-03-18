import os
import json
import numpy as np
import nltk
import pickle
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Suppress TensorFlow warnings for a cleaner output
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents (Ensure the JSON file path is correct)
json_file_path = 'D:/WorkCodingTest/AI_Projects/ChatBotInte/intents.json'
with open(json_file_path, 'r', encoding='utf-8') as file:
    intents = json.load(file)

# Load preprocessed words and classes
words = pickle.load(open('D:/WorkCodingTest/AI_Projects/ChatBotInte/words.pkl', 'rb'))
classes = pickle.load(open('D:/WorkCodingTest/AI_Projects/ChatBotInte/classes.pkl', 'rb'))

# Load trained model
model = load_model('D:/WorkCodingTest/AI_Projects/ChatBotInte/chatbot_model.h5')

# Compile model (optional, but prevents warnings)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Function to clean and tokenize a user input sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Convert sentence into a bag-of-words representation
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

# Predict intent class from user input
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

# Get a response based on predicted intent
def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm sorry, I didn't understand that."

    tag = intents_list[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return np.random.choice(intent['responses'])
    return "I'm sorry, I didn't understand that."

# Main chatbot function to process user input
def chatbot_response(user_input):
    intents_list = predict_class(user_input)
    return get_response(intents_list, intents)

# Test the chatbot in a loop
if __name__ == "__main__":
    print("Chatbot is running! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Chatbot: Goodbye!")
            break
        response = chatbot_response(user_input)
        print(f"Chatbot: {response}")
# End of chatbot.py
# Run this script to test the chatbot in a console or terminal. You can type messages and see the chatbot's responses.
# Type 'quit' to exit the chatbot loop.
