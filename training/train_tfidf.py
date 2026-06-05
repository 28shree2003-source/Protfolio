import random
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.preprocessing import load_intents, preprocess_text

# Load dataset
data = load_intents("data/intents.json")

patterns = []
tags = []
responses = {}

# Extract patterns and tags
for intent in data["intents"]:
    tag = intent["tag"]

    responses[tag] = intent["responses"]

    for pattern in intent["patterns"]:
        processed = " ".join(preprocess_text(pattern))

        patterns.append(processed)
        tags.append(tag)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(patterns)

print("TF-IDF model trained successfully!")

# Chat loop
while True:

    user_input = input("\nYou: ")

    if user_input.lower() == "quit":
        print("Bot: Goodbye!")
        break

    # Preprocess user input
    processed_input = " ".join(preprocess_text(user_input))

    # Convert input to vector
    input_vector = vectorizer.transform([processed_input])

    # Compute similarity
    similarity = cosine_similarity(input_vector, X)

    # Find best match
    best_match_index = np.argmax(similarity)

    best_score = similarity[0][best_match_index]

    predicted_tag = tags[best_match_index]

    # Confidence threshold
    if best_score > 0.3:
        response = random.choice(responses[predicted_tag])

        print(f"Bot ({predicted_tag}): {response}")

    else:
        print("Bot: Sorry, I didn't understand that.")