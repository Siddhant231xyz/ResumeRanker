from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

import config
from config import (
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP,
    SECTION_PATTERNS, SECTION_WEIGHTS,
)
from state import AgentState


def detect_section(text: str) -> str:
    """Check if a chunk's text itself contains a section header."""
    for section, pattern in SECTION_PATTERNS.items():
        if pattern.search(text):
            return section
    return "other"


def classify_chunks_with_context(full_text: str, chunks: list[Document]) -> None:
    """Classify each chunk by finding which section header precedes it in the full document."""
    section_spans = []
    for section, pattern in SECTION_PATTERNS.items():
        for match in pattern.finditer(full_text):
            section_spans.append((match.start(), section))
    section_spans.sort(key=lambda x: x[0])

    for chunk in chunks:
        chunk_pos = full_text.find(chunk.page_content[:80])
        if chunk_pos == -1:
            chunk.metadata["section"] = detect_section(chunk.page_content)
        else:
            assigned = "other"
            for pos, section in section_spans:
                if pos <= chunk_pos:
                    assigned = section
                else:
                    break
            chunk.metadata["section"] = assigned

        chunk.metadata["section_weight"] = SECTION_WEIGHTS[chunk.metadata["section"]]


def chunk_and_embed(state: AgentState) -> dict:
    """Node 2: Split documents into chunks, classify by section, embed, and store in FAISS."""
    documents = state["documents"]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"  Created {len(chunks)} chunks from {len(documents)} documents")

    # Build full text per file for section classification
    file_docs: dict[str, str] = {}
    for doc in documents:
        src = doc.metadata.get("source_file", "unknown")
        file_docs[src] = file_docs.get(src, "") + "\n" + doc.page_content

    file_chunks: dict[str, list[Document]] = {}
    for chunk in chunks:
        src = chunk.metadata.get("source_file", "unknown")
        file_chunks.setdefault(src, []).append(chunk)

    section_counts = {"experience": 0, "projects": 0, "skills": 0, "other": 0}
    for src, src_chunks in file_chunks.items():
        full_text = file_docs.get(src, "")
        classify_chunks_with_context(full_text, src_chunks)
        for c in src_chunks:
            section_counts[c.metadata["section"]] += 1

    print(f"  Section breakdown: {section_counts}")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=config.OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    print("  Vectorstore created (FAISS in-memory)")
    return {"vectorstore": vectorstore}
