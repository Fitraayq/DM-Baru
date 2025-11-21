
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib


true = pd.read_csv("data/True.csv")
fake = pd.read_csv("data/Fake.csv")

true["label"] = 1   # REAL
fake["label"] = 0   # FAKE

df = pd.concat([true, fake], axis=0).sample(frac=1, random_state=42)
df = df.reset_index(drop=True)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9.,;:!? ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["text"] = (df["title"] + " " + df["text"]).apply(clean_text)

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(
    max_iter=1500,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train_tfidf, y_train)

pred = model.predict(X_test_tfidf)
acc = accuracy_score(y_test, pred)

print("Akurasi Logistic Regression:", acc)
print("\nClassification Report:")
print(classification_report(y_test, pred))

joblib.dump(model, "model_lr.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel tersimpan! (Training selesai.)")
