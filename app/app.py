import streamlit as st
import fitz  # PyMuPDF
import spacy
import re

nlp = spacy.load("en_core_web_sm")

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

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


# --- File Reading ---
def read_text_file(uploaded_file):
    return uploaded_file.read().decode("utf-8")

def read_pdf_file(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# --- Streamlit UI ---
st.set_page_config(page_title="Resume Matcher", layout="centered")
st.title("🤖 Resume vs Job Description Matcher")

uploaded_resume = st.file_uploader("📄 Upload your resume (PDF or TXT)", type=["pdf", "txt"])
job_description = st.text_area("📝 Paste the job description here")

resume_text = ""

# --- Handle Upload ---
if uploaded_resume:
    file_type = uploaded_resume.name.split(".")[-1]
    if file_type == "pdf":
        resume_text = read_pdf_file(uploaded_resume)
    else:
        resume_text = read_text_file(uploaded_resume)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

nlp = spacy.load("en_core_web_sm")

# --- Match Keywords ---
def extract_keywords(text):
    doc = nlp(text.lower())
    return set([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])

# --- On Button Click ---
if st.button("🔍 Match Resume"):
    if resume_text.strip() == "" or job_description.strip() == "":
        st.warning("Please upload a resume and paste a job description.")
    else:
        # --- Preprocess ---
        preprocessed_resume = preprocess(resume_text)
        preprocessed_job = preprocess(job_description)

        # --- Similarity ---
        score = get_similarity(preprocessed_resume, preprocessed_job)

        # --- Keyword Extraction ---
        resume_keywords = extract_keywords(resume_text)
        job_keywords = extract_keywords(job_description)

        matched = resume_keywords & job_keywords
        missing = job_keywords - resume_keywords

        # --- Display Results ---
        st.subheader(f"🔗 Similarity Score: {score}%")
        if score >= 70:
            st.success("✅ Strong match!")
        elif score >= 40:
            st.info("⚠️ Partial match – improve your resume.")
        else:
            st.error("❌ Low match – resume needs better alignment.")

        # --- Matched & Missing Keywords ---
        st.markdown("### ✅ Matched Keywords")
        st.write(", ".join(sorted(matched)) if matched else "None")

        st.markdown("### ❌ Missing Keywords")
        st.write(", ".join(sorted(missing)) if missing else "None")
