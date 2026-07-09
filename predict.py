"""
============================================================
predict.py — Prediction Module
============================================================
Author      : Govardhan N
Project     : Fake News Detection using Machine Learning
Description : Load saved model + vectorizer and predict
              whether a given news article is REAL or FAKE.
              Can be used standalone (CLI) or imported by
              the Streamlit app.
============================================================
"""

import os
import joblib
import numpy as np
from preprocess import preprocess_text


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH      = os.path.join("models", "model.pkl")
VECTORIZER_PATH = os.path.join("models", "vectorizer.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# Load Artefacts
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    """
    Load the trained classifier from disk.

    Returns
    -------
    sklearn estimator
        The trained ML model.

    Raises
    ------
    FileNotFoundError
        If models/model.pkl does not exist.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. "
            "Please run train.py first to generate the model."
        )
    return joblib.load(MODEL_PATH)


def load_vectorizer():
    """
    Load the fitted TF-IDF vectorizer from disk.

    Returns
    -------
    TfidfVectorizer
        The fitted vectorizer.

    Raises
    ------
    FileNotFoundError
        If models/vectorizer.pkl does not exist.
    """
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            f"Vectorizer not found at '{VECTORIZER_PATH}'. "
            "Please run train.py first to generate the vectorizer."
        )
    return joblib.load(VECTORIZER_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Core Prediction Function
# ─────────────────────────────────────────────────────────────────────────────

def predict(text: str, model=None, vectorizer=None) -> dict:
    """
    Predict whether the given text is REAL or FAKE news.

    Parameters
    ----------
    text : str
        Raw news article text provided by the user.
    model : sklearn estimator, optional
        Pre-loaded model (avoids repeated disk I/O when called in a loop).
    vectorizer : TfidfVectorizer, optional
        Pre-loaded vectorizer.

    Returns
    -------
    dict with keys:
        label       (str)   : "REAL" or "FAKE"
        confidence  (float) : Probability of the predicted class [0, 1]
        proba_real  (float) : Probability of being REAL
        proba_fake  (float) : Probability of being FAKE
    """
    # Load from disk only when not passed explicitly
    if model is None:
        model = load_model()
    if vectorizer is None:
        vectorizer = load_vectorizer()

    # Step 1 — clean the raw text
    cleaned = preprocess_text(text)

    # Step 2 — vectorize using the SAME fitted vectorizer used during training
    features = vectorizer.transform([cleaned])

    # Step 3 — predict class label
    prediction = model.predict(features)[0]

    # Step 4 — predict class probabilities (if supported by the model)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        classes = list(model.classes_)
        # Handle both numeric (0/1) and string ("FAKE"/"REAL") labels
        if 0 in classes or "0" in [str(c) for c in classes]:
            # Assume 0 = FAKE, 1 = REAL
            idx_fake = classes.index(0) if 0 in classes else 0
            idx_real = classes.index(1) if 1 in classes else 1
        else:
            idx_fake = classes.index("FAKE") if "FAKE" in classes else 0
            idx_real = classes.index("REAL") if "REAL" in classes else 1
        proba_fake = float(proba[idx_fake])
        proba_real = float(proba[idx_real])
    else:
        # Fallback for models without predict_proba (e.g., LinearSVC)
        proba_fake = 1.0 if str(prediction) in ("0", "FAKE") else 0.0
        proba_real = 1.0 - proba_fake

    # Normalise label to a human-readable string
    label = "FAKE" if str(prediction) in ("0", "FAKE") else "REAL"
    confidence = max(proba_fake, proba_real)

    return {
        "label"      : label,
        "confidence" : round(confidence, 4),
        "proba_real" : round(proba_real, 4),
        "proba_fake" : round(proba_fake, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  Fake News Detection — Prediction CLI")
    print("=" * 60)

    if len(sys.argv) > 1:
        # Accept text as a command-line argument
        news_text = " ".join(sys.argv[1:])
    else:
        # Interactive mode
        print("\nPaste or type the news article text below.")
        print("Press ENTER twice to submit.\n")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        news_text = " ".join(lines)

    if not news_text.strip():
        print("No text provided. Exiting.")
        sys.exit(1)

    result = predict(news_text)

    print("\n── Result ──────────────────────────────────────────────")
    print(f"  Label      : {result['label']}")
    print(f"  Confidence : {result['confidence'] * 100:.2f}%")
    print(f"  P(REAL)    : {result['proba_real'] * 100:.2f}%")
    print(f"  P(FAKE)    : {result['proba_fake'] * 100:.2f}%")
    print("─" * 60)
