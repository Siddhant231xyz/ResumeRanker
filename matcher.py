import os

from langchain_openai import ChatOpenAI

import config
from config import LLM_MODEL, LLM_TEMPERATURE, SECTION_WEIGHTS, SIMILARITY_TOP_K
from state import AgentState


def match_resumes(state: AgentState) -> dict:
    """Node 3: Retrieve relevant chunks, apply section-based weighting, and rank resumes via LLM."""
    vectorstore = state["vectorstore"]
    job_description = state["job_description"]

    retrieved = vectorstore.similarity_search_with_score(job_description, k=SIMILARITY_TOP_K)

    # Build weighted scores per resume
    file_data: dict[str, dict] = {}
    for doc, distance in retrieved:
        src = doc.metadata.get("source_file", "unknown")
        section = doc.metadata.get("section", "other")
        weight = SECTION_WEIGHTS.get(section, 1.0)
        similarity = 1.0 / (1.0 + distance)
        weighted_score = similarity * weight

        if src not in file_data:
            file_data[src] = {"chunks": [], "total_weighted_score": 0.0}

        file_data[src]["chunks"].append({
            "content": doc.page_content,
            "section": section,
            "weight": weight,
            "similarity": round(similarity, 4),
            "weighted_score": round(weighted_score, 4),
        })
        file_data[src]["total_weighted_score"] += weighted_score

    # Sort by weighted score
    ranked_files = sorted(
        file_data.items(),
        key=lambda x: x[1]["total_weighted_score"],
        reverse=True,
    )

    # Build LLM context
    context_parts = []
    for file_path, data in ranked_files:
        chunk_details = []
        for c in data["chunks"]:
            chunk_details.append(
                f"  [Section: {c['section'].upper()} | Weight: {c['weight']}x | "
                f"Similarity: {c['similarity']} | Weighted: {c['weighted_score']}]\n"
                f"  {c['content']}"
            )
        context_parts.append(
            f"--- Resume: {os.path.basename(file_path)} ---\n"
            f"File path: {file_path}\n"
            f"Total weighted score: {round(data['total_weighted_score'], 4)}\n"
            f"Relevant excerpts:\n" + "\n\n".join(chunk_details)
        )
    context = "\n\n".join(context_parts)

    print("\n  Pre-LLM weighted ranking:")
    for i, (fp, data) in enumerate(ranked_files, 1):
        sections_hit = {c["section"] for c in data["chunks"]}
        print(
            f"    {i}. {os.path.basename(fp)} — score: {round(data['total_weighted_score'], 4)} "
            f"(chunks: {len(data['chunks'])}, sections: {sections_hit})"
        )

    prompt = f"""You are an expert recruiter. Based on the job description and the resume excerpts below,
rank the resumes from best match to worst match.

IMPORTANT RANKING RULES:
- Matches from the EXPERIENCE section carry the HIGHEST weight (4x priority)
- Matches from the PROJECTS section carry the SECOND highest weight (3x priority)
- Matches from the SKILLS section carry THIRD priority (2x)
- Matches from OTHER sections carry the LEAST weight (1x)

Each excerpt is tagged with its section, similarity score, and weighted score.
The resumes are pre-sorted by total weighted score. Use this as your primary ranking signal,
but adjust if the content quality warrants it.

JOB DESCRIPTION:
{job_description}

RESUME EXCERPTS (sorted by weighted score):
{context}

Provide your response as a numbered ranking with:
- Rank
- Candidate file path
- Match score (out of 10)
- Section breakdown (how many matches came from Experience vs Projects vs Skills vs Other)
- Key reasons for the ranking
"""

    llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE, api_key=config.OPENAI_API_KEY)
    response = llm.invoke(prompt)
    results = response.content

    print("\n" + "=" * 60)
    print("RESUME MATCHING RESULTS")
    print("=" * 60)
    print(results)

    return {"results": results}
