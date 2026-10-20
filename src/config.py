import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")
DATA_DIR = ROOT_DIR / "data"
PDF_PATH = DATA_DIR / "gpo_2018_annual_report.pdf"
CHROMA_DIR = ROOT_DIR / "storage" / "chroma"
COLLECTION_NAME = "gpo_2018_annual_report"

def _str_env(name: str, default: str) -> str:
    return os.getenv(name, "").strip() or default

def _int_env(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{name} must be a whole number, got {value!r}. Fix it in your .env file.") from None

LLM_PROVIDER = _str_env("LLM_PROVIDER", "openai").lower()
if LLM_PROVIDER == "openai":
    PROVIDER_NAME, API_KEY_NAME, API_BASE_URL = "OpenAI", "OPENAI_API_KEY", None
    CHAT_MODEL = _str_env("CHAT_MODEL", "gpt-4o-mini")
    EMBEDDING_MODEL = _str_env("EMBEDDING_MODEL", "text-embedding-3-small")
elif LLM_PROVIDER == "gemini":
    # Gemini is called through its OpenAI-compatible endpoint, so the same `openai` SDK works for both.
    PROVIDER_NAME, API_KEY_NAME = "Gemini", "GEMINI_API_KEY"
    API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    CHAT_MODEL = _str_env("GEMINI_CHAT_MODEL", "gemini-3.5-flash")
    EMBEDDING_MODEL = _str_env("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
else:
    raise ValueError(f"LLM_PROVIDER must be 'openai' or 'gemini', got {LLM_PROVIDER!r}. Fix it in your .env file.")
API_KEY = os.getenv(API_KEY_NAME, "").strip()
EMBEDDING_DIMENSIONS = _int_env("EMBEDDING_DIMENSIONS", 0) or None  # None = the model's default size
CHUNK_SIZE = _int_env("CHUNK_SIZE", 1000)
CHUNK_OVERLAP = _int_env("CHUNK_OVERLAP", 180)
TOP_K = _int_env("TOP_K", 5)
