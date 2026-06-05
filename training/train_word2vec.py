import random
import numpy as np

from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

from utils.preprocessing import load_intents, preprocess_text

# Load dataset
data = load_intents("data/intents.json")

sentences = []
tags = []
responses = {}

# Extract patterns
for intent in data["intents"]:

    tag = intent["tag"]

    responses[tag] = intent["responses"]

    for pattern in intent["patterns"]:

        processed = preprocess_text(pattern)

        sentences.append(processed)
        tags.append(tag)

# Train Word2Vec model
model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4
)

print("Word2Vec model trained successfully!")

# Function to get sentence vector
def sentence_vector(sentence_tokens):

    vectors = []

    for word in sentence_tokens:

        if word in model.wv:
            vectors.append(model.wv[word])

    if len(vectors) == 0:
        return np.zeros(100)

    return np.mean(vectors, axis=0)

# Create vectors for all patterns
sentence_vectors = []

for sentence in sentences:
    sentence_vectors.append(sentence_vector(sentence))

sentence_vectors = np.array(sentence_vectors)

# Chat loop
while True:

    user_input = input("\nYou: ")

    if user_input.lower() == "quit":
        print("Bot: Goodbye!")
        break

    # Preprocess user input
    processed_input = preprocess_text(user_input)

    # Convert input into vector
    input_vector = sentence_vector(processed_input)

    # Compute similarity
    similarity_scores = cosine_similarity(
        [input_vector],
        sentence_vectors
    )

    # Find best match
    best_match_index = np.argmax(similarity_scores)

    best_score = similarity_scores[0][best_match_index]

    predicted_tag = tags[best_match_index]

    # Confidence threshold
    if best_score > 0.5:

        response = random.choice(responses[predicted_tag])

        print(f"Bot ({predicted_tag}): {response}")

        print(f"Confidence: {best_score:.2f}")

    else:
        print("Bot: Sorry, I didn't understand that.")