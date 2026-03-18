import os
import sys

from graph import build_graph

SAMPLE_JOB_DESCRIPTION = """
Senior Python Developer — AI/ML Platform

About the role:
We are looking for a Senior Python Developer to join our AI/ML Platform team.
You will design, build, and maintain production-grade machine learning pipelines
and APIs that power our core product.

Requirements:
- 3+ years of professional experience with Python
- Strong experience with machine learning frameworks (PyTorch, TensorFlow, scikit-learn)
- Hands-on experience building REST APIs (FastAPI, Flask, or Django)
- Familiarity with LLMs, RAG pipelines, LangChain, or similar frameworks
- Experience with vector databases (FAISS, Pinecone, ChromaDB)
- Solid understanding of data structures, algorithms, and software design patterns
- Experience with cloud platforms (AWS, GCP, or Azure)
- Proficiency with Git, Docker, and CI/CD pipelines

Nice to have:
- Experience with LangGraph or multi-agent systems
- Knowledge of NLP and natural language understanding
- Contributions to open-source projects
- Experience deploying ML models at scale with Kubernetes
"""


if __name__ == "__main__":
    resume_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "resumes")
    job_description = sys.argv[2] if len(sys.argv) > 2 else SAMPLE_JOB_DESCRIPTION

    if not os.path.isdir(resume_dir):
        print(f"Error: '{resume_dir}' is not a valid directory.")
        sys.exit(1)

    app = build_graph()

    print(f"Resume directory: {resume_dir}")
    print(f"Job description: {job_description[:80]}...")
    print("\n[1/3] Parsing resumes...")

    result = app.invoke({
        "resume_dir": resume_dir,
        "job_description": job_description,
        "documents": [],
        "vectorstore": None,
        "results": "",
    })

    print("\nDone. Results are above.")
