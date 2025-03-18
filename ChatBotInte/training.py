import json
import numpy as np
import nltk
import pickle
import random
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

lemmatizer = WordNetLemmatizer()

# Load JSON file
with open('D:/WorkCodingTest/AI_Projects/ChatBotInte/intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

# Preprocessing data
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Tokenize words
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize words
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

# Save words and classes
pickle.dump(words, open('D:/WorkCodingTest/AI_Projects/ChatBotInte/words.pkl', 'wb'))
pickle.dump(classes, open('D:/WorkCodingTest/AI_Projects/ChatBotInte/classes.pkl', 'wb'))

# Training data preparation
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = [lemmatizer.lemmatize(w.lower()) for w in doc[0]]
    for w in words:
        bag.append(1 if w in word_patterns else 0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Build the neural network model
model = Sequential([
    Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(train_y[0]), activation='softmax')
])

# Compile model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

# Train model
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save trained model
model.save('D:/WorkCodingTest/AI_Projects/ChatBotInte/chatbot_model.h5')

print("Training complete. Model saved.")
