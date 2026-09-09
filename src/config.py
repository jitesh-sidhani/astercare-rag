import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
CHROMA_DIR = str((ROOT_DIR / os.getenv("CHROMA_DIR", "chroma_db")).resolve()) if not os.getenv("CHROMA_DIR", "").startswith("/") else os.getenv("CHROMA_DIR")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "astercare_handbook")
TOP_K = int(os.getenv("TOP_K", "12"))
TOP_N = int(os.getenv("TOP_N", "5"))
HANDBOOK_PATH = (
    ROOT_DIR
    / "data"
    / "Knowledge Base - AsterCare Clinical Diagnosis and Therapeutics Handbook.docx"
)
