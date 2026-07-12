"""
config.py
---------
Central place for all configuration: environment variables, default models,
chunking parameters, and the clause taxonomy this project targets.

Keeping this in one file means a reviewer can understand every tunable knob
in the pipeline without hunting through the codebase.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
CACHE_DIR = PROJECT_ROOT / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# LLM provider selection
# ---------------------------------------------------------------------------
# Which provider to use by default. Can be overridden with --provider on the CLI.
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

# Default model per provider. Chosen for a good cost/speed/quality tradeoff.
DEFAULT_MODELS = {
    "groq": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    "openai": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    "anthropic": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
}

API_KEYS = {
    "groq": os.getenv("GROQ_API_KEY"),
    "openai": os.getenv("OPENAI_API_KEY"),
    "anthropic": os.getenv("ANTHROPIC_API_KEY"),
}

# ---------------------------------------------------------------------------
# Chunking / retrieval parameters
# ---------------------------------------------------------------------------
CHUNK_SIZE_CHARS = 1800          # ~ 450 tokens per chunk
CHUNK_OVERLAP_CHARS = 200
TOP_K_CHUNKS_PER_CLAUSE = 4      # how many chunks to feed the LLM per clause type
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Clause taxonomy for this assignment (Part A)
# ---------------------------------------------------------------------------
CLAUSE_TYPES = ["termination_clause", "confidentiality_clause", "liability_clause"]

# Natural-language queries used to *retrieve* the most relevant chunks for each
# clause type before we ever call the LLM. This keeps prompts small and
# accurate even for 40-page contracts.
CLAUSE_RETRIEVAL_QUERIES = {
    "termination_clause": (
        "termination of agreement, notice period, conditions for termination, "
        "termination for cause, termination for convenience, expiration of the contract"
    ),
    "confidentiality_clause": (
        "confidential information, non-disclosure obligations, protection of "
        "proprietary information, permitted disclosures, duty of confidentiality"
    ),
    "liability_clause": (
        "limitation of liability, indemnification, damages, disclaimer of "
        "warranties, cap on liability, consequential damages"
    ),
}

SUMMARY_WORD_RANGE = (100, 150)

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
REQUEST_TIMEOUT = 60
MAX_RETRIES = 4
