# ResumeRanker — AI-Powered Resume Ranking System 📄

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-latest-red.svg)](https://streamlit.io)
[![OpenAI](https://img.shields.io/badge/OpenAI-integrated-green.svg)](https://openai.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-latest-purple.svg)](https://github.com/langchain-ai/langgraph)
[![LangChain](https://img.shields.io/badge/LangChain-latest-yellow.svg)](https://langchain.org)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

## Overview 🎯

ResumeRanker is an intelligent resume screening tool that ranks candidate resumes against a job description using a multi-node LangGraph pipeline. It uses retrieval-augmented generation (RAG) with section-aware weighted scoring — prioritizing matches from **Experience** over **Projects** over **Skills** — to produce accurate, explainable rankings.

## Features ✨

- 🤖 **Multi-Node LangGraph Pipeline**
  - Node 1: Parse resumes (PDF, DOCX, TXT)
  - Node 2: Chunk, classify sections, embed, and store in FAISS
  - Node 3: Weighted retrieval + LLM-powered ranking

- ⚖️ **Section-Aware Weighted Scoring**
  - Experience matches: **4x** weight
  - Project matches: **3x** weight
  - Skills matches: **2x** weight
  - Other sections: **1x** weight

- 💬 **Streamlit Web Interface**
  - Upload resumes or provide a folder path
  - Paste or upload a job description
  - Adjustable section weights via sidebar sliders
  - Shareable via public URL

- 📊 **Transparent Results**
  - Pre-LLM weighted scores for each resume
  - Section breakdown per candidate
  - Match score out of 10 with detailed reasoning

## Technical Architecture 🏗️

```
ResumeRanker/
├── main.py              # CLI entry point
├── config.py            # Configuration, section weights, patterns
├── state.py             # LangGraph state schema
├── parser.py            # Node 1: Resume file parsing
├── embedder.py          # Node 2: Chunking, section classification, FAISS
├── matcher.py           # Node 3: Weighted retrieval + LLM ranking
├── graph.py             # LangGraph pipeline definition
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (API keys)
├── .gitignore           # Git ignore rules
├── resumes/             # Sample resume files
└── streamlit_app/
    ├── app.py           # Streamlit web application
    └── .streamlit/
        └── config.toml  # Streamlit theme configuration
```

### How It Works

1. **Resume Parsing (Node 1)**
   - Scans a directory for PDF, DOCX, and TXT files
   - Loads each file and attaches the source file path in metadata

2. **Chunking & Embedding (Node 2)**
   - Splits documents into 500-character chunks with overlap
   - Classifies each chunk into a resume section (Experience, Projects, Skills, Other) by detecting section headers in the full document context
   - Generates vector embeddings using OpenAI's `text-embedding-3-small`
   - Stores everything in an in-memory FAISS vectorstore

3. **Weighted Matching & LLM Ranking (Node 3)**
   - Retrieves the top-40 most similar chunks for the job description
   - Applies section-based weight multipliers to similarity scores
   - Computes a total weighted score per resume
   - Sends the pre-ranked, section-tagged context to GPT-4o-mini for final ranking with explanations

```
START → parse_resumes → chunk_and_embed → match_resumes → END
```

## Installation 🚀

### Prerequisites

- Python 3.10+
- Git
- An OpenAI API key

### Setup Steps

1. **Clone the Repository:**

```bash
git clone https://github.com/Siddhant231xyz/ResumeRanker.git
cd ResumeRanker
```

2. **Create a Virtual Environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies:**

```bash
pip install -r requirements.txt
```

4. **Configure Environment:**

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your-api-key-here
```

## Usage Guide 📚

### CLI Mode

```bash
# Run with default sample job description and ./resumes folder
python main.py

# Run with custom resume folder and job description
python main.py /path/to/resumes "Your job description text here"
```

### Streamlit Web App

```bash
cd streamlit_app
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

**In the web app you can:**
- Upload resume files (PDF, DOCX, TXT) or provide a folder path
- Type/paste a job description or upload a `.txt` file
- Adjust section weights (Experience, Projects, Skills, Other) via sidebar sliders
- Click **Rank Resumes** to run the pipeline and see results

## Configuration ⚙️

Section weights can be adjusted in `config.py` or via the Streamlit sidebar:

| Section    | Default Weight | Priority |
|------------|---------------|----------|
| Experience | 4.0x          | Highest  |
| Projects   | 3.0x          | High     |
| Skills     | 2.0x          | Medium   |
| Other      | 1.0x          | Low      |

## Contributing 🤝

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## Dependencies 📦

- langgraph
- langchain & langchain-openai & langchain-community
- faiss-cpu
- pypdf
- python-docx & docx2txt
- python-dotenv
- streamlit

## License 📝

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
Created with ❤️ by Siddhant
</div>
