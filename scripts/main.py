# scripts/main.py

from scripts.preprocessor import preprocess
from scripts.similarity import get_similarity

def load_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    resume = load_file("data/sample_resume.txt")
    job_desc = load_file("data/sample_job_desc.txt")

    preprocessed_resume = preprocess(resume)
    preprocessed_job = preprocess(job_desc)

    score = get_similarity(preprocessed_resume, preprocessed_job)

    print(f"🔍 Similarity Score: {score}%")
    if score > 70:
        print("✅ This resume is a strong match for the job.")
    else:
        print("⚠️ Resume may need better alignment with job description.")
