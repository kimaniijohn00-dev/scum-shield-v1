"""
ScamShield — quick trainer (self-installing)
----------------------------------------------
1. Download a labeled dataset (pick ONE, takes 2 min):
   - "SMS Spam Collection Dataset" on Kaggle (search that exact name)
   - Or "Phishing Email Dataset" on Kaggle (search that exact name)
   Save it as data.csv next to this script.

2. Your CSV needs 2 columns: a text column and a label column
   (values like "spam"/"ham" or "scam"/"safe" both work — edit
   TEXT_COL / LABEL_COL below if your file's headers differ).

3. Just run:  python train_scamshield.py
   Missing packages install themselves automatically.
"""

import subprocess
import sys
import os


def ensure_installed(package, import_name=None):
    import_name = import_name or package
    try:
        __import__(import_name)
    except ImportError:
        print(f"Installing {package} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])


for pkg in ["pandas", "scikit-learn", "joblib"]:
    ensure_installed(pkg, "sklearn" if pkg == "scikit-learn" else pkg)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# ---- EDIT THESE TWO IF YOUR CSV HEADERS DIFFER ----
TEXT_COL = "text"
LABEL_COL = "label"
# ----------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data.csv")

if not os.path.exists(DATA_PATH):
    print(f"ERROR: Could not find data.csv at {DATA_PATH}")
    print("Download a dataset (e.g. 'SMS Spam Collection Dataset' on Kaggle),")
    print(f"rename it to data.csv, and place it in: {SCRIPT_DIR}")
    sys.exit(1)

df = pd.read_csv(DATA_PATH, encoding="latin-1")

# auto-detect the common Kaggle SMS Spam Collection column names
if "v1" in df.columns and "v2" in df.columns:
    TEXT_COL, LABEL_COL = "v2", "v1"

df = df[[TEXT_COL, LABEL_COL]].dropna()

# normalize labels to scam/safe (edit mapping if your dataset uses other words)
label_map = {
    "spam": "scam", "scam": "scam", "phishing": "scam", "1": "scam",
    "ham": "safe", "safe": "safe", "legit": "safe", "0": "safe",
}
df[LABEL_COL] = df[LABEL_COL].astype(str).str.lower().map(label_map).fillna(df[LABEL_COL])

X_train, X_test, y_train, y_test = train_test_split(
    df[TEXT_COL], df[LABEL_COL], test_size=0.2, random_state=42, stratify=df[LABEL_COL]
)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train_vec, y_train)

print(classification_report(y_test, clf.predict(X_test_vec)))

joblib.dump(clf, "scamshield_model.joblib")
joblib.dump(vectorizer, "scamshield_vectorizer.joblib")
print("Saved model + vectorizer.")


def predict(message: str, unsure_band=0.15):
    """Returns (label, confidence). label is 'scam', 'safe', or 'unsure'."""
    vec = vectorizer.transform([message])
    proba = clf.predict_proba(vec)[0]
    classes = clf.classes_
    top_idx = proba.argmax()
    top_label, top_conf = classes[top_idx], proba[top_idx]

    # if the model isn't confident, call it unsure instead of guessing
    if top_conf < 0.5 + unsure_band:
        return "unsure", float(top_conf)
    return top_label, float(top_conf)


if __name__ == "__main__":
    tests = [
        "Congratulations! You've won 50,000 KES, send your ID to claim.",
        "Hey are we still meeting at 3pm today?",
        "Your M-Pesa account will be suspended, verify now at bit.ly/xyz123",
    ]
    for t in tests:
        print(t, "->", predict(t))
