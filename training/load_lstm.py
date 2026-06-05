import pickle
import numpy as np
import random

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils.preprocessing import load_intents

# Load responses
data = load_intents("data/intents.json")

responses = {}

for intent in data["intents"]:
    responses[intent["tag"]] = intent["responses"]

# Load model
model = load_model("models/lstm_model.h5")

# Load tokenizer
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load label encoder
with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Load max length
with open("models/max_len.pkl", "rb") as f:
    max_len = pickle.load(f)

print("Model Loaded Successfully!")

while True:

    user_input = input("\nYou: ")

    if user_input.lower() == "quit":
        break

    sequence = tokenizer.texts_to_sequences(
        [user_input.lower()]
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

    tag = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    if confidence > 0.5:

        response = random.choice(
            responses[tag]
        )

        print(
            f"Bot ({tag}): {response}"
        )

        print(
            f"Confidence: {confidence:.2f}"
        )

    else:
        print(
            "Bot: Sorry, I didn't understand."
        )