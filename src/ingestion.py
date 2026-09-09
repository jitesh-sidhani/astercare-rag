from pathlib import Path
from typing import Dict, List
from docx import Document


def load_docx(path: str | Path) -> List[Dict]:
    """Load non-empty DOCX paragraphs while preserving order."""
    doc = Document(str(path))
    records = []
    for idx, paragraph in enumerate(doc.paragraphs, start=1):
        text = " ".join(paragraph.text.split()).strip()
        if not text:
            continue
        style = paragraph.style.name if paragraph.style else ""
        records.append({"paragraph_id": idx, "text": text, "style": style})
    return records
