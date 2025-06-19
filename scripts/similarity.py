# scripts/similarity.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def get_similarity(resume_text, job_text):
    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform([resume_text, job_text])
    similarity_score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

    # Save vectorizer if needed
    with open("models/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)

    return round(similarity_score * 100, 2)
