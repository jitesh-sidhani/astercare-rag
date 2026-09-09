from __future__ import annotations

from .retriever import Retriever
from .generator import AnswerGenerator


class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.generator = AnswerGenerator()

    def run(self, question: str) -> dict:
        """
        Complete RAG pipeline:

            User Question
                  ↓
            Hybrid Retrieval
                  ↓
            Context Selection
                  ↓
            OpenAI Generation
                  ↓
            Answer + Sources
        """

        question = question.strip()

        # ---------------------------------------------------------
        # 1. Validate question
        # ---------------------------------------------------------
        if not question:
            return {
                "answer": "Please enter a question.",
                "sources": [],
            }

        # ---------------------------------------------------------
        # 2. Retrieve relevant handbook contexts
        # ---------------------------------------------------------
        contexts = self.retriever.retrieve(question)

        if not contexts:
            return {
                "answer": (
                    "I could not retrieve relevant content "
                    "from the AsterCare handbook."
                ),
                "sources": [],
            }

        # ---------------------------------------------------------
        # 3. Generate grounded answer
        # ---------------------------------------------------------
        answer = self.generator.answer(
            question,
            contexts,
        )

        # ---------------------------------------------------------
        # 4. Prepare source information
        # ---------------------------------------------------------
        sources = []

        for context in contexts:
            metadata = context.get(
                "metadata",
                {},
            )

            sources.append(
                {
                    "chunk_id": context.get(
                        "chunk_id"
                    ),

                    "section_number": metadata.get(
                        "section_number"
                    ),

                    "section_title": metadata.get(
                        "section_title"
                    ),

                    "heading": metadata.get(
                        "heading"
                    ),

                    "authority": metadata.get(
                        "authority"
                    ),

                    "restricted": metadata.get(
                        "restricted",
                        False,
                    ),

                    "chunk_type": metadata.get(
                        "chunk_type"
                    ),

                    "paragraph_start": metadata.get(
                        "paragraph_start"
                    ),

                    "paragraph_end": metadata.get(
                        "paragraph_end"
                    ),

                    # Retrieval diagnostics
                    "final_score": round(
                        context.get(
                            "final_score",
                            0.0,
                        ),
                        4,
                    ),

                    "semantic_score": (
                        round(
                            context["semantic_score"],
                            4,
                        )
                        if context.get(
                            "semantic_score"
                        ) is not None
                        else None
                    ),

                    "bm25_score": (
                        round(
                            context["bm25_score"],
                            4,
                        )
                        if context.get(
                            "bm25_score"
                        ) is not None
                        else None
                    ),

                    "semantic_rank": context.get(
                        "semantic_rank"
                    ),

                    "bm25_rank": context.get(
                        "bm25_rank"
                    ),
                }
            )

        # ---------------------------------------------------------
        # 5. Return answer + source metadata
        # ---------------------------------------------------------
        return {
            "answer": answer,
            "sources": sources,
        }