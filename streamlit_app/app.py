import os
import sys
import tempfile
import streamlit as st

# Add parent directory to path so we can import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import OPENAI_API_KEY, SECTION_WEIGHTS
from graph import build_graph

st.set_page_config(page_title="ResumeRanker", page_icon="📄", layout="wide")

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Configuration")

    api_key = st.text_input(
        "OpenAI API Key",
        value=OPENAI_API_KEY or "",
        type="password",
        help="Your OpenAI API key. Required for embeddings and LLM ranking.",
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    st.divider()
    st.subheader("Section Weights")
    st.caption("Higher weight = higher priority in ranking")
    w_exp = st.slider("Experience", 1.0, 5.0, SECTION_WEIGHTS["experience"], 0.5)
    w_proj = st.slider("Projects", 1.0, 5.0, SECTION_WEIGHTS["projects"], 0.5)
    w_skill = st.slider("Skills", 1.0, 5.0, SECTION_WEIGHTS["skills"], 0.5)
    w_other = st.slider("Other", 1.0, 5.0, SECTION_WEIGHTS["other"], 0.5)

    # Update weights dynamically
    SECTION_WEIGHTS["experience"] = w_exp
    SECTION_WEIGHTS["projects"] = w_proj
    SECTION_WEIGHTS["skills"] = w_skill
    SECTION_WEIGHTS["other"] = w_other

# ── Main content ─────────────────────────────────────────────────────────────

st.title("📄 ResumeRanker")
st.markdown("**AI-powered resume ranking using LangGraph, FAISS, and GPT.**")
st.markdown("Upload resumes and provide a job description to get ranked results with section-weighted scoring.")

st.divider()

# ── Resume upload ────────────────────────────────────────────────────────────

st.subheader("1. Upload Resumes")

upload_method = st.radio(
    "Choose input method:",
    ["Upload files", "Provide folder path"],
    horizontal=True,
)

resume_dir = None

if upload_method == "Upload files":
    uploaded_files = st.file_uploader(
        "Upload resume files (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        # Save uploaded files to a temp directory
        resume_dir = tempfile.mkdtemp(prefix="resumes_")
        for f in uploaded_files:
            file_path = os.path.join(resume_dir, f.name)
            with open(file_path, "wb") as out:
                out.write(f.getbuffer())
        st.success(f"Uploaded {len(uploaded_files)} resume(s)")
else:
    folder_path = st.text_input(
        "Enter the full path to the folder containing resumes:",
        placeholder="e.g., C:/Users/admin/Desktop/resumes",
    )
    if folder_path and os.path.isdir(folder_path):
        resume_dir = folder_path
        file_count = sum(
            1 for f in os.listdir(folder_path)
            if f.lower().endswith((".pdf", ".docx", ".txt"))
        )
        st.success(f"Found {file_count} resume file(s) in the folder")
    elif folder_path:
        st.error("Invalid folder path. Please check and try again.")

# ── Job description ──────────────────────────────────────────────────────────

st.subheader("2. Job Description")

jd_method = st.radio(
    "Choose input method:",
    ["Type / paste text", "Upload text file"],
    horizontal=True,
    key="jd_method",
)

job_description = ""

if jd_method == "Type / paste text":
    job_description = st.text_area(
        "Paste the job description below:",
        height=250,
        placeholder="e.g., We are looking for a Senior Python Developer with experience in...",
    )
else:
    jd_file = st.file_uploader("Upload a .txt file with the job description", type=["txt"], key="jd_file")
    if jd_file:
        job_description = jd_file.read().decode("utf-8")
        st.text_area("Job description preview:", value=job_description, height=200, disabled=True)

# ── Run ranking ──────────────────────────────────────────────────────────────

st.divider()

can_run = resume_dir and job_description.strip() and api_key
if st.button("🚀 Rank Resumes", disabled=not can_run, type="primary", use_container_width=True):
    with st.spinner("Running the ranking pipeline..."):
        # Patch weights into config before running
        import config
        config.SECTION_WEIGHTS["experience"] = w_exp
        config.SECTION_WEIGHTS["projects"] = w_proj
        config.SECTION_WEIGHTS["skills"] = w_skill
        config.SECTION_WEIGHTS["other"] = w_other
        config.OPENAI_API_KEY = api_key
        os.environ["OPENAI_API_KEY"] = api_key

        app = build_graph()

        progress = st.progress(0, text="Parsing resumes...")
        result = app.invoke({
            "resume_dir": resume_dir,
            "job_description": job_description.strip(),
            "documents": [],
            "vectorstore": None,
            "results": "",
        })
        progress.progress(100, text="Done!")

    st.subheader("3. Results")
    st.markdown(result["results"])

elif not can_run:
    missing = []
    if not api_key:
        missing.append("OpenAI API key (sidebar)")
    if not resume_dir:
        missing.append("Resumes")
    if not job_description.strip():
        missing.append("Job description")
    if missing:
        st.info(f"Please provide: {', '.join(missing)}")

# ── Footer ───────────────────────────────────────────────────────────────────

st.divider()
st.caption("Built with LangGraph, FAISS, OpenAI, and Streamlit")
