from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from src.pipeline import RAGPipeline


# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------

st.set_page_config(
    page_title="AsterCare Clinical RAG",
    page_icon="🩺",
    layout="centered",
)


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------

st.title("🩺 AsterCare Clinical RAG Assistant")

st.caption(
    "Source-grounded question answering over the "
    "AsterCare Clinical Diagnosis & Therapeutics Handbook"
)


# -------------------------------------------------------------------
# Pipeline
# -------------------------------------------------------------------

@st.cache_resource
def load_pipeline() -> RAGPipeline:
    """Load and cache the RAG pipeline."""
    return RAGPipeline()


# -------------------------------------------------------------------
# User input
# -------------------------------------------------------------------

st.subheader("Ask the Handbook")

question = st.text_area(
    "Question",
    placeholder=(
        "Example: What dose should a 15-year-old with "
        "Lumeris Type B receive?"
    ),
    height=120,
)


# -------------------------------------------------------------------
# Ask button
# -------------------------------------------------------------------

if st.button(
    "Ask Handbook",
    type="primary",
    use_container_width=True,
):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Searching the handbook..."):
                result = load_pipeline().run(question.strip())

            answer = result.get("answer", "").strip()
            sources = result.get("sources", [])

            # -------------------------------------------------------
            # Answer
            # -------------------------------------------------------

            st.subheader("Answer")

            if answer:
                st.markdown(answer)
            else:
                st.info(
                    "The handbook context does not provide enough "
                    "information to answer this question."
                )

            # -------------------------------------------------------
            # Sources
            # -------------------------------------------------------

            if sources:
                st.subheader("Sources")

                for index, source in enumerate(
                    sources,
                    start=1,
                ):
                    section_number = source.get(
                        "section_number",
                        "N/A",
                    )

                    section_title = source.get(
                        "section_title",
                        "N/A",
                    )

                    heading = source.get(
                        "heading",
                        "General rule",
                    )

                    authority = source.get(
                        "authority",
                        "N/A",
                    )

                    restricted = source.get(
                        "restricted",
                        False,
                    )

                    final_score = source.get(
                        "final_score",
                        0.0,
                    )

                    with st.expander(
                        f"Source {index}: {heading}",
                        expanded=index == 1,
                    ):
                        st.markdown(
                            f"**Section:** "
                            f"{section_number} — {section_title}"
                        )

                        st.markdown(
                            f"**Heading:** {heading}"
                        )

                        st.markdown(
                            f"**Authority:** {authority}"
                        )

                        if restricted:
                            st.warning(
                                "This source is marked as restricted. "
                                "Restricted information is not disclosed "
                                "in the generated answer."
                            )

                        st.markdown(
                            f"**Retrieval score:** "
                            f"{final_score:.4f}"
                        )

                        source_text = source.get(
                            "text",
                            "",
                        )

                        if source_text:
                            st.markdown("**Retrieved context**")
                            st.text(source_text)

            else:
                st.info(
                    "No supporting source passages were returned."
                )

        except Exception as exc:
            st.error(
                "An error occurred while processing the question."
            )

            with st.expander("Technical details"):
                st.exception(exc)


# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------

st.divider()

st.caption(
    "AsterCare is a fictional, non-operational handbook. "
    "This application is a retrieval and question-answering "
    "demonstration and is not intended to provide real-world "
    "medical advice."
)