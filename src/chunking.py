from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from docx import Document


SECTION_RE = re.compile(r"^(\d{1,2})\.\s+(.+)$")


@dataclass
class Chunk:
    chunk_id: str
    text: str

    section_number: str
    section_title: str
    heading: str

    paragraph_start: int
    paragraph_end: int

    authority: str
    restricted: bool

    chunk_type: str

    def to_dict(self) -> dict:
        return asdict(self)


def _looks_like_heading(text: str) -> bool:
    """
    Detect compact disease/topic headings.

    The handbook uses many unnumbered topic headings even though the
    DOCX does not reliably encode them with Word heading styles.
    """
    text = text.strip()

    if not text:
        return False

    # Numbered section headings are handled separately.
    if SECTION_RE.match(text):
        return True

    # Avoid classifying long prose as headings.
    if len(text) > 100:
        return False

    words = text.split()

    if not 1 <= len(words) <= 12:
        return False

    # Common sentence starters that indicate normal prose rather than headings.
    excluded_starts = (
        "If ",
        "When ",
        "A ",
        "An ",
        "The ",
        "For ",
        "Some ",
        "Similarly ",
        "Conversely ",
        "By ",
        "At ",
        "From ",
        "This ",
        "Patients ",
        "Patient ",
        "In ",
        "However ",
        "Because ",
        "As ",
        "During ",
        "Where ",
    )

    if text.startswith(excluded_starts):
        return False

    # Headings normally begin with an uppercase character.
    return text[0].isupper()


def _classify_authority(text: str) -> tuple[str, bool]:
    """
    Classify the source material.

    Important because the handbook deliberately mixes governing clinical
    rules with imported, historical, and restricted material.
    """
    lower = text.lower()

    # Restricted reference material.
    restricted = (
        "records three reference values" in lower
        or "brevalin device calibration ceiling" in lower
        or "miravel quarantine trigger" in lower
        or "tavroxen reserve-alert value" in lower
        or "remains governed by its access classification" in lower
    )

    if restricted:
        return "restricted", True

    # Historical material.
    historical_terms = [
        "archived clinical guidance notes extract",
        "historical review",
        "archived rule",
        "older boundary",
        "historical snapshot",
    ]

    if any(term in lower for term in historical_terms):
        return "historical", False

    # Imported/external material.
    imported_terms = [
        "external respiratory handoffs",
        "external respiratory handoff",
        "respiratory reconciliation service",
        "third-party medication reconciliation export",
        "external clinical guidance note",
        "external eye-care handout",
        "copied patient portal message",
        "copied rehabilitation note",
        "vendor-generated brevalin reconciliation certificate",
        "pharmacy attachment",
        "imported portal histories",
        "imported reminder systems",
        "external transfer service",
        "historical research abstract",
    ]

    if any(term in lower for term in imported_terms):
        return "imported", False

    return "governing", False


def _classify_chunk_type(text: str) -> str:
    """
    Conservative chunk classification.

    Classification is based on explicit signals and is metadata only.
    It should never override the actual chunk text.
    """
    lower = text.lower()

    # ---------------------------------------------------------
    # Source / authority material
    # ---------------------------------------------------------
    if any(
        phrase in lower
        for phrase in [
            "source status",
            "source authority",
            "source text",
            "source declaration",
            "imported source",
            "external source",
            "patient statement",
            "patient-authored history",
            "vendor-generated",
            "record integrity",
            "access rights",
            "internal - restricted",
            "restricted reference",
            "historical source",
        ]
    ):
        return "source_integrity"

    # ---------------------------------------------------------
    # Explicit exception / override
    # ---------------------------------------------------------
    if any(
        phrase in lower
        for phrase in [
            "this exception",
            "the exception",
            "override rule",
            "overrides the",
            "exception applies",
            "replaces both",
            "replaces the",
            "alternate medicine",
        ]
    ):
        return "exception_or_override"

    # ---------------------------------------------------------
    # Explicit hold / contraindication
    # ---------------------------------------------------------
    if any(
        phrase in lower
        for phrase in [
            "is held when",
            "is held if",
            "held pending review",
            "dose is held",
            "is omitted if",
            "should not be co-administered",
            "prevents initiation",
            "prevents continuation",
        ]
    ):
        return "contraindication_or_hold"

    # ---------------------------------------------------------
    # Age / population rules
    # ---------------------------------------------------------
    if any(
        phrase in lower
        for phrase in [
            "patients aged",
            "patients younger than",
            "patients seventy-five and older",
            "adults sixty-five and older",
            "twelve through seventeen",
            "younger than eighteen",
            "older than seventy-five",
            "pediatric",
            "adolescent",
            "age-specific",
        ]
    ):
        return "age_or_population_rule"

    # ---------------------------------------------------------
    # Monitoring / follow-up
    # ---------------------------------------------------------
    if any(
        phrase in lower
        for phrase in [
            "first review",
            "review occurs",
            "reviewed after",
            "reviewed at",
            "reassessed",
            "reassessment",
            "follow-up",
            "review clock",
        ]
    ):
        return "monitoring_or_followup"

    # ---------------------------------------------------------
    # Treatment
    # ---------------------------------------------------------
    if any(
        phrase in lower
        for phrase in [
            "first-line medicine",
            "standard medicine",
            "routine medicine",
            "the medicine is",
            "uses ",
            "dose is",
            "mg every",
            "mg once",
            "micro-units",
            "fictive-mg",
        ]
    ):
        return "treatment_rule"

    # ---------------------------------------------------------
    # Threshold
    # ---------------------------------------------------------
    if any(
        phrase in lower
        for phrase in [
            "exactly ",
            "at least ",
            "above ",
            "below ",
            "through ",
            "inclusive",
            "cutoff",
            "threshold",
        ]
    ):
        return "threshold_rule"

    # ---------------------------------------------------------
    # Disease / symptom profile
    # ---------------------------------------------------------
    if any(
        phrase in lower
        for phrase in [
            "presents with",
            "characterized by",
            "defined by",
            "associated with",
            "causes ",
            "produces ",
            "is identified by",
            "is considered",
        ]
    ):
        return "disease_or_symptom_profile"

    return "general_rule"



def _pack_paragraphs(
    items: list[tuple[int, str]],
    max_chars: int = 1800,
) -> list[tuple[int, int, str]]:
    """
    Pack nearby paragraphs while preserving paragraph boundaries.

    We do not split individual paragraphs in the middle.
    """
    packed: list[tuple[int, int, str]] = []

    current: list[tuple[int, str]] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len

        if not current:
            return

        start = current[0][0]
        end = current[-1][0]

        text = " ".join(
            text.strip()
            for _, text in current
        ).strip()

        packed.append((start, end, text))

        current = []
        current_len = 0

    for para_no, text in items:
        text = text.strip()

        if not text:
            continue

        separator_len = 1 if current else 0
        extra = separator_len + len(text)

        if current and current_len + extra > max_chars:
            flush()

        current.append((para_no, text))
        current_len += extra

    flush()

    return packed


def load_chunks(path: str | Path) -> list[Chunk]:
    """
    Load the AsterCare handbook and create structure-aware chunks.
    """
    doc = Document(str(path))

    # Extract non-empty paragraphs while retaining original paragraph numbers.
    raw = [
        (
            i + 1,
            paragraph.text.replace("\xa0", " ").strip(),
        )
        for i, paragraph in enumerate(doc.paragraphs)
    ]

    raw = [
        (paragraph_number, text)
        for paragraph_number, text in raw
        if text
    ]

    chunks: list[Chunk] = []

    current_section_number = ""
    current_section_title = ""
    current_heading = ""

    current_items: list[tuple[int, str]] = []

    counter = 0

    def flush() -> None:
        nonlocal current_items
        nonlocal counter

        if not current_items:
            return

        packed_chunks = _pack_paragraphs(
            current_items,
            max_chars=1800,
        )

        for start, end, text in packed_chunks:
            counter += 1

            authority, restricted = _classify_authority(text)
            chunk_type = _classify_chunk_type(text)

            chunks.append(
                Chunk(
                    chunk_id=f"astercare-{counter:04d}",
                    text=text,
                    section_number=current_section_number,
                    section_title=current_section_title,
                    heading=current_heading,
                    paragraph_start=start,
                    paragraph_end=end,
                    authority=authority,
                    restricted=restricted,
                    chunk_type=chunk_type,
                )
            )

        current_items = []

    for para_no, text in raw:

        # ---------------------------------------------------------
        # 1. Numbered handbook section
        # ---------------------------------------------------------
        section_match = SECTION_RE.match(text)

        if section_match:
            flush()

            current_section_number = section_match.group(1)
            current_section_title = section_match.group(2).strip()

            current_heading = text

            continue

        # ---------------------------------------------------------
        # 2. Disease/topic heading
        # ---------------------------------------------------------
        if _looks_like_heading(text):
            flush()

            current_heading = text

            continue

        # ---------------------------------------------------------
        # 3. Prevent different authority classes from sharing
        #    the same chunk.
        # ---------------------------------------------------------
        if current_items:

            previous_text = current_items[-1][1]

            previous_authority = _classify_authority(previous_text)
            current_authority = _classify_authority(text)

            if previous_authority != current_authority:
                flush()

        # ---------------------------------------------------------
        # 4. Add normal paragraph
        # ---------------------------------------------------------
        current_items.append(
            (para_no, text)
        )

    # Flush final chunk.
    flush()

    return chunks



if __name__ == "__main__":
    from pathlib import Path

    # Project root = parent of src/
    project_root = Path(__file__).resolve().parent.parent

    handbook_path = (
        project_root
        / "data"
        / "Knowledge Base - AsterCare Clinical Diagnosis and Therapeutics Handbook.docx"
    )

    if not handbook_path.exists():
        raise FileNotFoundError(
            f"Handbook not found at:\n{handbook_path}"
        )

    chunks = load_chunks(handbook_path)

    print(f"Created {len(chunks)} chunks")

    for chunk in chunks[:12]:
        print(
            chunk.chunk_id,
            "| Section:",
            chunk.section_number,
            "| Heading:",
            chunk.heading,
            "| Authority:",
            chunk.authority,
            "| Restricted:",
            chunk.restricted,
            "| Type:",
            chunk.chunk_type,
            "| Characters:",
            len(chunk.text),
        )

# if __name__ == "__main__":
#     from .config import settings

#     chunks = load_chunks(settings.handbook_path)

#     print(f"Created {len(chunks)} chunks")

#     for chunk in chunks[:12]:
#         print(
#             chunk.chunk_id,
#             "| Section:",
#             chunk.section_number,
#             "| Heading:",
#             chunk.heading,
#             "| Authority:",
#             chunk.authority,
#             "| Restricted:",
#             chunk.restricted,
#             "| Type:",
#             chunk.chunk_type,
#             "| Characters:",
#             len(chunk.text),
#         )