from typing import TypedDict


class AgentState(TypedDict):
    resume_dir: str          # directory containing resume files
    job_description: str     # the job description to match against
    documents: list          # raw parsed documents
    vectorstore: object      # FAISS in-memory vectorstore
    results: str             # final LLM output
