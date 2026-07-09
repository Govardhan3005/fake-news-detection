"""
============================================================
preprocess.py — Text Preprocessing Module
============================================================
Project     : Fake News Detection using Machine Learning
Description : Reusable, modular text preprocessing pipeline.
              All cleaning steps are encapsulated here so that
              both train.py and predict.py use the SAME logic.
============================================================
"""

import re
import string
import nltk

# Download required NLTK resources (runs only if not already present)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ── Module-level singletons (created once, reused everywhere) ───────────────
STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()


# ─────────────────────────────────────────────────────────────────────────────
# Individual Cleaning Helpers
# ─────────────────────────────────────────────────────────────────────────────

def to_lowercase(text: str) -> str:
    """Convert all characters to lowercase."""
    return text.lower()


def remove_urls(text: str) -> str:
    """Remove HTTP/HTTPS URLs and bare www. links."""
    return re.sub(r'https?://\S+|www\.\S+', ' ', text)


def remove_html_tags(text: str) -> str:
    """Strip HTML tags such as <p>, <br/>, <a href=...>, etc."""
    return re.sub(r'<.*?>', ' ', text)


def remove_punctuation(text: str) -> str:
    """Remove all punctuation characters."""
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_numbers(text: str) -> str:
    """Remove standalone numbers and digits."""
    return re.sub(r'\b\d+\b', ' ', text)


def remove_special_characters(text: str) -> str:
    """Remove non-alphanumeric characters (keeps spaces)."""
    return re.sub(r'[^a-zA-Z\s]', ' ', text)


def remove_extra_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines into a single space."""
    return re.sub(r'\s+', ' ', text).strip()


def remove_stopwords(text: str) -> str:
    """Remove English stopwords."""
    tokens = text.split()
    filtered = [word for word in tokens if word not in STOP_WORDS]
    return ' '.join(filtered)


def lemmatize_text(text: str) -> str:
    """Lemmatize each token (e.g., 'running' → 'run')."""
    tokens = text.split()
    lemmatized = [LEMMATIZER.lemmatize(word) for word in tokens]
    return ' '.join(lemmatized)


# ─────────────────────────────────────────────────────────────────────────────
# Master Preprocessing Function
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """
    Apply the complete text cleaning pipeline in a fixed order.

    Pipeline steps
    --------------
    1. Convert to lowercase
    2. Remove URLs
    3. Remove HTML tags
    4. Remove punctuation
    5. Remove numbers
    6. Remove special characters
    7. Remove extra whitespace
    8. Remove stopwords
    9. Lemmatize tokens

    Parameters
    ----------
    text : str
        Raw news article text.

    Returns
    -------
    str
        Fully cleaned and normalized text.
    """
    if not isinstance(text, str):
        return ""

    text = to_lowercase(text)
    text = remove_urls(text)
    text = remove_html_tags(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_special_characters(text)
    text = remove_extra_whitespace(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)

    return text


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test (run this file directly to verify the pipeline)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = (
        "BREAKING NEWS!! <b>President</b> signs new bill. "
        "Visit https://example.com for details. "
        "100 protesters were arrested on 23rd March 2024!!!"
    )
    print("Original :", sample)
    print("Cleaned  :", preprocess_text(sample))
