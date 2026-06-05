import json
import random
import numpy as np

from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pickle


from utils.preprocessing import load_intents

# Load dataset
data = load_intents("data/intents.json")

sentences = []
labels = []
responses = {}

# Extract patterns and tags
for intent in data["intents"]:

    tag = intent["tag"]

    responses[tag] = intent["responses"]

    for pattern in intent["patterns"]:

        sentences.append(pattern.lower())
        labels.append(tag)

# Tokenization
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

sequences = tokenizer.texts_to_sequences(sentences)

# Padding
max_len = max(len(seq) for seq in sequences)

padded_sequences = pad_sequences(
    sequences,
    maxlen=max_len,
    padding='post'
)

# Encode labels
label_encoder = LabelEncoder()

encoded_labels = label_encoder.fit_transform(labels)

categorical_labels = to_categorical(encoded_labels)

# Vocabulary size
vocab_size = len(tokenizer.word_index) + 1

# Build LSTM model
model = Sequential()

model.add(
    Embedding(
        input_dim=vocab_size,
        output_dim=128,
        input_length=max_len
    )
)

model.add(
    LSTM(128)
)

model.add(
    Dense(64, activation='relu')
)

model.add(
    Dense(
        len(set(labels)),
        activation='softmax'
    )
)

# Compile model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train model
model.fit(
    padded_sequences,
    categorical_labels,
    epochs=200,
    batch_size=8,
    verbose=1
)

print("\nLSTM model trained successfully!")

# Save model
model.save("models/lstm_model.h5")

# Save tokenizer
with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Save label encoder
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Save max length
with open("models/max_len.pkl", "wb") as f:
    pickle.dump(max_len, f)

print("All files saved successfully!")

# Chat loop
while True:

    user_input = input("\nYou: ")

    if user_input.lower() == "quit":
        print("Bot: Goodbye!")
        break

    # Convert input to sequence
    input_sequence = tokenizer.texts_to_sequences(
        [user_input.lower()]
    )

    padded_input = pad_sequences(
        input_sequence,
        maxlen=max_len,
        padding='post'
    )

    # Predict intent
    prediction = model.predict(padded_input)

    predicted_index = np.argmax(prediction)

    confidence = prediction[0][predicted_index]

    predicted_tag = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    # Confidence threshold
    if confidence > 0.5:

        response = random.choice(
            responses[predicted_tag]
        )

        print(f"Bot ({predicted_tag}): {response}")

        print(f"Confidence: {confidence:.2f}")

    else:
        print("Bot: Sorry, I didn't understand that.")