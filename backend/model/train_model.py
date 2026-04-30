import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score

# Download stopwords (only first time)
nltk.download('stopwords', quiet=True)

# Load stopwords ONCE (important fix)
stop_words = set(stopwords.words('english'))

# Load datasets
fake = pd.read_csv("../../data/Fake.csv")
true = pd.read_csv("../../data/True.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine datasets
data = pd.concat([fake, true], axis=0, ignore_index=True)

# Balance dataset manually
fake_count = len(data[data["label"] == 0])
real_count = len(data[data["label"] == 1])

min_count = min(fake_count, real_count)

fake_data = data[data["label"] == 0].sample(min_count, random_state=42)
real_data = data[data["label"] == 1].sample(min_count, random_state=42)

data = pd.concat([fake_data, real_data]).sample(frac=1).reset_index(drop=True)

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', '', text)
    words = text.split()
    
    # Remove short words and stopwords
    words = [word for word in words if len(word) > 2 and word not in stop_words]
    
    return " ".join(words)

# Combine title and text so the model sees the headline as well as the article body
# Use BOTH short + long forms
data_long = data.copy()
data_short = data.copy()

# Long version
data_long["content"] = data_long["title"] + " " + data_long["text"]

# Short version (IMPORTANT)
data_short["content"] = data_short["title"]

# Combine both
data = pd.concat([data_long, data_short], ignore_index=True)

data["content"] = data["content"].apply(clean_text)

# Features and labels
X = data["content"]
y = data["label"]

print("🔄 Converting text to vectors... (this may take time)")
# Convert text to numerical features
vectorizer = TfidfVectorizer(
    max_features=20000,   # ↓ reduced
    ngram_range=(1, 2),   # ↓ removed 3-grams
    min_df=5,
    max_df=0.8,
    sublinear_tf=True,
    stop_words="english"
)

# Split before fitting so the evaluation is honest
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# Train model
model = SGDClassifier(
    loss="log_loss",
    class_weight="balanced",
    max_iter=2000,
    alpha=0.0005
)
model.fit(X_train, y_train)

# Check accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy:.4f}")
print(classification_report(y_test, predictions, target_names=["Fake", "Real"]))

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved successfully!")