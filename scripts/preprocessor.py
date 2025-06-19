# scripts/preprocessor.py

import spacy
import re

nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    # Lowercase, remove non-alphabetic chars
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    return text

def preprocess(text):
    cleaned = clean_text(text)
    doc = nlp(cleaned)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(tokens)
