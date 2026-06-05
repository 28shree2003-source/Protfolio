import json
import string
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords and lemmatize
    processed_tokens = []

    for word in tokens:
        if word not in stop_words:
            word = lemmatizer.lemmatize(word)
            processed_tokens.append(word)

    return processed_tokens


def load_intents(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    return data