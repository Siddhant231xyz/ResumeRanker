import re
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SIMILARITY_TOP_K = 40

# Section detection patterns (order matters: first match wins)
SECTION_PATTERNS = {
    "experience": re.compile(
        r"(work\s*experience|professional\s*experience|experience|employment\s*history|work\s*history)",
        re.IGNORECASE,
    ),
    "projects": re.compile(
        r"(projects|personal\s*projects|academic\s*projects|key\s*projects)",
        re.IGNORECASE,
    ),
    "skills": re.compile(
        r"(skills|technical\s*skills|core\s*competencies|technologies|tools\s*&\s*technologies)",
        re.IGNORECASE,
    ),
}

# Section priority weights: higher = more important
SECTION_WEIGHTS = {
    "experience": 4.0,
    "projects": 3.0,
    "skills": 2.0,
    "other": 1.0,
}
