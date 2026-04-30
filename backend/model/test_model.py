import pickle
import re
import nltk
from nltk.corpus import stopwords

# =========================
# LOAD MODEL
# =========================
model = pickle.load(open("../model/model.pkl", "rb"))
vectorizer = pickle.load(open("../model/vectorizer.pkl", "rb"))

# =========================
# SETUP CLEANING
# =========================
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', '', text)
    words = text.split()
    words = [w for w in words if len(w) > 2 and w not in stop_words]
    return " ".join(words)

# =========================
# PREDICTION FUNCTION
# =========================
def predict(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])

    pred = model.predict(vector)[0]
    prob = model.predict_proba(vector)[0]

    confidence = max(prob) * 100

    # Decision logic (same as backend)
    if confidence < 55:
        if pred == 1:
            result = "Real"
        else:
            result = "Uncertain"
    else:
        result = "Real" if pred == 1 else "Fake"

    return result, confidence, prob

# =========================
# INTERACTIVE MODE
# =========================
def interactive_mode():
    print("\n🔍 Fake News Model Tester (type 'exit' to quit)\n")

    while True:
        text = input("Enter text: ")

        if text.lower() == "exit":
            print("Exiting tester...")
            break

        result, confidence, prob = predict(text)

        print("\nTEXT:", text)
        print("Prediction:", result)
        print("Confidence:", round(confidence, 2))
        print("Raw Probabilities [Fake, Real]:", prob)
        print("-" * 50)

# =========================
# QUICK TEST MODE
# =========================
def quick_tests():
    tests = [
        "The government released an official report on economic growth.",
        "Researchers are studying climate change effects.",
        "Breaking shocking miracle cure found!!!",
        "Scientists secretly discovered a hidden energy source.",
        "Aliens landed in Jaipur today"
    ]

    print("\n⚡ Running Quick Tests...\n")

    for t in tests:
        result, confidence, _ = predict(t)

        print("TEXT:", t)
        print("Prediction:", result)
        print("Confidence:", round(confidence, 2))
        print("-" * 50)

# =========================
# MAIN ENTRY
# =========================
if __name__ == "__main__":
    print("\nChoose mode:")
    print("1 → Interactive")
    print("2 → Quick test")

    choice = input("Enter choice: ")

    if choice == "1":
        interactive_mode()
    elif choice == "2":
        quick_tests()
    else:
        print("Invalid choice")