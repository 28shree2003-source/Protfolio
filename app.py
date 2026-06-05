import random
import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.preprocessing import (
    load_intents,
    preprocess_text
)
from utils.preprocessing import load_intents

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ==========================
# RULE-BASED RESPONSES
# ==========================

rule_responses = {
    "hello": "Hello!",
    "hi": "Hi there!",
    "hey": "Hello!",
    "bye": "Goodbye!",
    "thanks": "You're welcome!"
}


# ==========================
# LOAD DATASET
# ==========================

data = load_intents("data/intents.json")

responses = {}

for intent in data["intents"]:
    responses[intent["tag"]] = intent["responses"]


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

patterns = []
tags = []
sentences = []

for intent in data["intents"]:

    tag = intent["tag"]

    for pattern in intent["patterns"]:

        processed_tokens = preprocess_text(pattern)

        processed_text = " ".join(
            processed_tokens
        )

        patterns.append(
            processed_text
        )

        sentences.append(
            processed_tokens
        )

        tags.append(tag)
# ==========================
# LOAD SAVED MODEL FILES
# ==========================

print("Loading model...")

model = load_model("models/lstm_model.h5")

with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("models/max_len.pkl", "rb") as f:
    max_len = pickle.load(f)

print("Model loaded successfully!")
print("Chatbot is ready.\n")

# TF-IDF

tfidf_vectorizer = TfidfVectorizer()

tfidf_matrix = tfidf_vectorizer.fit_transform(
    patterns
)

# Word2Vec

w2v_model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4
)

def sentence_vector(tokens):

    vectors = []

    for word in tokens:

        if word in w2v_model.wv:
            vectors.append(
                w2v_model.wv[word]
            )

    if len(vectors) == 0:
        return np.zeros(100)

    return np.mean(
        vectors,
        axis=0
    )

sentence_vectors = np.array([
    sentence_vector(s)
    for s in sentences
])
def predict_tfidf(text):

    processed = " ".join(
        preprocess_text(text)
    )

    input_vector = tfidf_vectorizer.transform(
        [processed]
    )

    similarity = cosine_similarity(
        input_vector,
        tfidf_matrix
    )

    idx = np.argmax(similarity)

    score = similarity[0][idx]

    return tags[idx], score
# ==========================
# LSTM PREDICTION FUNCTION
# ==========================

def predict_intent(text):

    sequence = tokenizer.texts_to_sequences(
        [text.lower()]
    )

    padded = pad_sequences(
        sequence,
        maxlen=max_len,
        padding="post"
    )

    prediction = model.predict(
        padded,
        verbose=0
    )

    predicted_index = np.argmax(prediction)

    confidence = prediction[0][predicted_index]

    predicted_tag = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    return predicted_tag, confidence


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():

    user_input = request.json["message"]

    # Rule Based
    if user_input.lower() in rule_responses:
        return jsonify({
            "response": rule_responses[user_input.lower()],
            "source": "RULE"
        })

    return get_response(user_input)


# ==========================
# RESPONSE ENGINE
# ==========================
def get_response(user_input):

    # TF-IDF
    tag, score = predict_tfidf(user_input)
    if score > 0.3:
        return jsonify({
            "response": random.choice(responses[tag]),
            "source": "TF-IDF"
        })
    #word2vec
    tag, score = predict_word2vec(user_input)
    if score > 0.5:
        return jsonify({
            "response": random.choice(responses[tag]),
             "source": "Word2Vec"
         })

    # LSTM
    tag, confidence = predict_intent(user_input)
    if confidence > 0.5:
        return jsonify({
            "response": random.choice(responses[tag]),
            "source": "LSTM"
        })

    # fallback
    return jsonify({
        "response": "Sorry, I didn't understand that.",
        "source": "fallback"
    })

if __name__ == "__main__":
    app.run(debug=True)