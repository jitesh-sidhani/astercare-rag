from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion import load_docx
from src.chunking import chunk_handbook
from src.config import HANDBOOK_PATH


def test_chunking_produces_chunks():
    records = load_docx(HANDBOOK_PATH)
    chunks = chunk_handbook(records)
    assert len(records) > 0
    assert len(chunks) > 0
    assert all(c["text"].strip() for c in chunks)
