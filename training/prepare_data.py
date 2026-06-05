from utils.preprocessing import load_intents, preprocess_text

data = load_intents("data/intents.json")

patterns = []
tags = []

for intent in data["intents"]:
    tag = intent["tag"]

    for pattern in intent["patterns"]:
        processed_pattern = preprocess_text(pattern)

        patterns.append(processed_pattern)
        tags.append(tag)

print("Total training samples:", len(patterns))

print("\nSample processed sentence:")
print(patterns[0])

print("\nCorresponding tag:")
print(tags[0])