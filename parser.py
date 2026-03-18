import os
import glob

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

from state import AgentState

LOADERS = {
    "*.pdf": lambda p: PyPDFLoader(p),
    "*.docx": lambda p: Docx2txtLoader(p),
    "*.txt": lambda p: TextLoader(p, encoding="utf-8"),
}


def parse_resumes(state: AgentState) -> dict:
    """Node 1: Load and parse all resume files (PDF, DOCX, TXT) from the given directory."""
    resume_dir = state["resume_dir"]
    documents = []

    for pattern, loader_fn in LOADERS.items():
        for file_path in glob.glob(os.path.join(resume_dir, pattern)):
            try:
                loader = loader_fn(file_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source_file"] = os.path.abspath(file_path)
                documents.extend(docs)
                print(f"  Parsed: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"  Failed to parse {file_path}: {e}")

    print(f"\n  Total documents loaded: {len(documents)}")
    return {"documents": documents}
