import nltk
import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

with open('responses.json') as file:
    intents = json.load(file)

patterns = []
tags = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern.lower())
        tags.append(intent['tag'])

vectorizer = TfidfVectorizer()
pattern_vectors = vectorizer.fit_transform(patterns)

def predict_intent(message):
    message_vector = vectorizer.transform([message.lower()])
    similarity_scores = cosine_similarity(message_vector, pattern_vectors)
    best_match_index = similarity_scores.argmax()
    return tags[best_match_index]

def get_response(intent_tag):
    for intent in intents['intents']:
        if intent['tag'] == intent_tag:
            return random.choice(intent['responses'])
    return "Sorry, I don't understand that."

def chat():
    print("Start chatting with the chatbot (type 'quit' to exit):")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        intent = predict_intent(user_input)
        response = get_response(intent)
        print("Bot:", response)

if __name__ == "__main__":
    nltk.download('punkt')
    chat()
