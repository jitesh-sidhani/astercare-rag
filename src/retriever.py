from __future__ import annotations

import re
from typing import Dict, List

from rank_bm25 import BM25Okapi

from .config import TOP_K, TOP_N
from .embeddings import EmbeddingService
from .vector_store import ChromaStore


class Retriever:
    def __init__(self):
        self.embedder = EmbeddingService()
        self.store = ChromaStore()

        # Build local BM25 index from the existing ChromaDB documents.
        # No OpenAI API call is made here.
        self._build_bm25_index()

    # =========================================================
    # Tokenization
    # =========================================================
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Tokenize text for BM25 and metadata matching.

        Numbers and decimals are retained because exact clinical
        thresholds are important in the handbook.
        """
        return re.findall(
            r"[A-Za-z]+(?:-[A-Za-z]+)*|\d+(?:\.\d+)?",
            text.lower(),
        )

    # =========================================================
    # BM25 INDEX
    # =========================================================
    def _build_bm25_index(self) -> None:
        """
        Build a local BM25 index over the chunks already stored
        in ChromaDB.
        """
        data = self.store.collection.get(
            include=["documents", "metadatas"],
        )

        self.bm25_ids = data.get("ids", [])
        self.bm25_documents = data.get("documents", [])
        self.bm25_metadatas = data.get("metadatas", [])

        tokenized_documents = [
            self._tokenize(document)
            for document in self.bm25_documents
        ]

        if tokenized_documents:
            self.bm25 = BM25Okapi(tokenized_documents)
        else:
            self.bm25 = None

    # =========================================================
    # SEMANTIC RETRIEVAL
    # =========================================================
    def _semantic_search(
        self,
        question: str,
        top_k: int,
    ) -> List[Dict]:
        """
        Retrieve candidates using OpenAI embeddings + ChromaDB.
        """
        query_vector = self.embedder.embed_query(question)

        result = self.store.query(
            query_vector,
            n_results=top_k,
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        ids = result.get("ids", [[]])[0]

        results: List[Dict] = []

        for rank, (
            chunk_id,
            document,
            metadata,
            distance,
        ) in enumerate(
            zip(
                ids,
                documents,
                metadatas,
                distances,
            ),
            start=1,
        ):
            results.append(
                {
                    "chunk_id": chunk_id,
                    "text": document,
                    "metadata": metadata or {},
                    "semantic_rank": rank,
                    "semantic_score": 1.0 - distance,
                }
            )

        return results

    # =========================================================
    # BM25 RETRIEVAL
    # =========================================================
    def _bm25_search(
        self,
        question: str,
        top_k: int,
    ) -> List[Dict]:
        """
        Retrieve candidates using BM25 lexical matching.
        """
        if self.bm25 is None:
            return []

        query_tokens = self._tokenize(question)

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        results: List[Dict] = []

        for rank, index in enumerate(
            ranked_indices,
            start=1,
        ):
            results.append(
                {
                    "chunk_id": self.bm25_ids[index],
                    "text": self.bm25_documents[index],
                    "metadata": self.bm25_metadatas[index] or {},
                    "bm25_rank": rank,
                    "bm25_score": float(scores[index]),
                }
            )

        return results

    # =========================================================
    # RECIPROCAL RANK FUSION
    # =========================================================
    def _rrf_fusion(
        self,
        semantic_results: List[Dict],
        bm25_results: List[Dict],
        rrf_k: int = 60,
    ) -> List[Dict]:
        """
        Fuse semantic and BM25 rankings using RRF.
        """
        fused: Dict[str, Dict] = {}

        # -----------------------------------------------------
        # Semantic results
        # -----------------------------------------------------
        for result in semantic_results:
            chunk_id = result["chunk_id"]
            rank = result["semantic_rank"]

            fused[chunk_id] = {
                "chunk_id": chunk_id,
                "text": result["text"],
                "metadata": result["metadata"],
                "rrf_score": 1.0 / (rrf_k + rank),
                "semantic_rank": rank,
                "bm25_rank": None,
                "semantic_score": result["semantic_score"],
                "bm25_score": None,
            }

        # -----------------------------------------------------
        # BM25 results
        # -----------------------------------------------------
        for result in bm25_results:
            chunk_id = result["chunk_id"]
            rank = result["bm25_rank"]

            contribution = 1.0 / (rrf_k + rank)

            if chunk_id in fused:
                fused[chunk_id]["rrf_score"] += contribution
                fused[chunk_id]["bm25_rank"] = rank
                fused[chunk_id]["bm25_score"] = result["bm25_score"]

            else:
                fused[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "rrf_score": contribution,
                    "semantic_rank": None,
                    "bm25_rank": rank,
                    "semantic_score": None,
                    "bm25_score": result["bm25_score"],
                }

        return sorted(
            fused.values(),
            key=lambda item: item["rrf_score"],
            reverse=True,
        )

    # =========================================================
    # SPECIFIC ENTITY EXTRACTION
    # =========================================================
    @staticmethod
    def _extract_specific_phrases(
        question: str,
    ) -> List[str]:
        """
        Extract specific disease/type/pattern phrases.

        Examples:
            Lumeris Type B
            Ardenic Pattern A
            Urovel Lower-Tract Syndrome
            Caroven Airspace Inflammation
        """
        question_lower = question.lower()

        phrases: List[str] = []

        patterns = [
            # Lumeris Type B
            r"\b[a-z]+(?:\s+[a-z]+){0,2}\s+type\s+[a-z]\b",

            # Ardenic Pattern A
            r"\b[a-z]+(?:\s+[a-z]+){0,2}\s+pattern\s+[a-z]\b",

            # Urovel Lower-Tract Syndrome
            r"\b[a-z]+(?:\s+[a-z]+){0,3}\s+syndrome\b",

            # Cavernic Pressure Disorder
            r"\b[a-z]+(?:\s+[a-z]+){0,3}\s+disorder\b",

            # Vexoral Febrile Illness
            r"\b[a-z]+(?:\s+[a-z]+){0,3}\s+illness\b",

            # Caroven Airspace Inflammation
            r"\b[a-z]+(?:\s+[a-z]+){0,3}\s+inflammation\b",
        ]

        for pattern in patterns:
            matches = re.findall(
                pattern,
                question_lower,
            )

            for match in matches:
                cleaned = match.strip()

                if cleaned:
                    phrases.append(cleaned)

        # Preserve order and remove duplicates.
        return list(dict.fromkeys(phrases))

    # =========================================================
    # GENERIC REFERENCE DETECTION
    # =========================================================
    @staticmethod
    def _is_generic_reference_heading(
        heading: str,
    ) -> bool:
        """
        Identify broad reference headings that should not outrank
        a directly named disease-specific rule.
        """
        lower = heading.lower()

        generic_reference_terms = [
            "family",
            "medication profiles",
            "medication timing",
            "commonly confused disease pairs",
            "disease summary notes",
            "core interpretation principles",
        ]

        return any(
            term in lower
            for term in generic_reference_terms
        )

    # =========================================================
    # METADATA + SPECIFICITY REFINEMENT
    # =========================================================
    def _metadata_refinement(
        self,
        question: str,
        candidates: List[Dict],
    ) -> List[Dict]:
        """
        Apply specificity-aware reranking.

        A directly named disease/type/pattern gets a strong boost.

        Generic medication/reference chunks receive a small penalty
        when a specific disease is present in the question.
        """
        question_lower = question.lower()

        question_terms = set(
            self._tokenize(question)
        )

        specific_phrases = self._extract_specific_phrases(
            question
        )

        has_specific_entity = bool(
            specific_phrases
        )

        for candidate in candidates:
            metadata = candidate["metadata"]

            heading = metadata.get(
                "heading",
                "",
            ).strip()

            section_title = metadata.get(
                "section_title",
                "",
            ).strip()

            chunk_type = metadata.get(
                "chunk_type",
                "",
            ).strip()

            authority = metadata.get(
                "authority",
                "",
            ).strip()

            restricted = metadata.get(
                "restricted",
                False,
            )

            heading_lower = heading.lower()

            bonus = 0.0
            penalty = 0.0

            # -------------------------------------------------
            # 1. Strong direct specific-phrase match
            # -------------------------------------------------
            for phrase in specific_phrases:
                if phrase in heading_lower:
                    bonus += 0.060

            # -------------------------------------------------
            # 2. Meaningful heading-token overlap
            # -------------------------------------------------
            heading_terms = set(
                self._tokenize(heading)
            )

            generic_terms = {
                "and",
                "the",
                "of",
                "for",
                "in",
                "to",
                "with",
                "type",
                "pattern",
                "section",
                "disease",
                "syndrome",
                "disorder",
                "illness",
                "conditions",
                "condition",
                "profile",
                "profiles",
                "presentations",
                "family",
                "note",
                "notes",
                "rule",
                "rules",
            }

            meaningful_heading_terms = (
                heading_terms - generic_terms
            )

            if meaningful_heading_terms:
                overlap = (
                    meaningful_heading_terms
                    & question_terms
                )

                overlap_ratio = (
                    len(overlap)
                    / len(meaningful_heading_terms)
                )

                if overlap_ratio >= 0.75:
                    bonus += 0.025

                elif overlap_ratio >= 0.50:
                    bonus += 0.015

                elif overlap_ratio >= 0.30:
                    bonus += 0.008

            # -------------------------------------------------
            # 3. Exact section title match
            # -------------------------------------------------
            if (
                section_title
                and section_title.lower()
                in question_lower
            ):
                bonus += 0.008

            # -------------------------------------------------
            # 4. Treatment intent
            # -------------------------------------------------
            treatment_terms = {
                "dose",
                "medicine",
                "medication",
                "treatment",
                "drug",
                "regimen",
            }

            if (
                chunk_type == "treatment_rule"
                and question_terms.intersection(
                    treatment_terms
                )
            ):
                bonus += 0.004

            # -------------------------------------------------
            # 5. Threshold intent
            # -------------------------------------------------
            threshold_terms = {
                "threshold",
                "exactly",
                "above",
                "below",
                "cutoff",
                "marker",
                "surrogate",
            }

            if (
                chunk_type == "threshold_rule"
                and question_terms.intersection(
                    threshold_terms
                )
            ):
                bonus += 0.004

            # -------------------------------------------------
            # 6. Age / population intent
            # -------------------------------------------------
            age_terms = {
                "year-old",
                "years",
                "old",
                "child",
                "children",
                "adolescent",
                "pediatric",
                "adult",
                "elderly",
            }

            if (
                chunk_type == "age_or_population_rule"
                and question_terms.intersection(
                    age_terms
                )
            ):
                bonus += 0.004

            # -------------------------------------------------
            # 7. Generic reference penalty
            # -------------------------------------------------
            #
            # Example:
            #
            # Question:
            # "Should Velorin be withheld in Ardenic Pattern A?"
            #
            # "Velorin family" is useful supporting context,
            # but the Ardenic disease rule should take precedence.
            #
            if (
                has_specific_entity
                and self._is_generic_reference_heading(
                    heading
                )
            ):
                penalty += 0.020

            # -------------------------------------------------
            # 8. Governing-content preference
            # -------------------------------------------------
            if (
                authority == "governing"
                and not restricted
            ):
                bonus += 0.002

            candidate["metadata_bonus"] = bonus
            candidate["specificity_penalty"] = penalty

            candidate["final_score"] = (
                candidate["rrf_score"]
                + bonus
                - penalty
            )

        return sorted(
            candidates,
            key=lambda item: item["final_score"],
            reverse=True,
        )

    # =========================================================
    # RESTRICTED CONTENT FILTER
    # =========================================================
    @staticmethod
    def _filter_restricted(
        candidates: List[Dict],
    ) -> List[Dict]:
        """
        Restricted chunks never reach the LLM.
        """
        return [
            candidate
            for candidate in candidates
            if not candidate["metadata"].get(
                "restricted",
                False,
            )
        ]

    # =========================================================
    # PUBLIC RETRIEVAL API
    # =========================================================
    def retrieve(
        self,
        question: str,
        top_k: int = TOP_K,
        top_n: int = TOP_N,
    ) -> List[Dict]:
        """
        Hybrid retrieval pipeline:

            Query
              ↓
        Semantic Search
              +
            BM25
              ↓
          RRF Fusion
              ↓
        Specificity-aware
          reranking
              ↓
        Restricted filter
              ↓
            Top-N
        """
        question = question.strip()

        if not question:
            return []

        # 1. Semantic retrieval
        semantic_results = self._semantic_search(
            question,
            top_k=top_k,
        )

        # 2. BM25 retrieval
        bm25_results = self._bm25_search(
            question,
            top_k=top_k,
        )

        # 3. Hybrid fusion
        candidates = self._rrf_fusion(
            semantic_results,
            bm25_results,
        )

        # 4. Specificity-aware reranking
        candidates = self._metadata_refinement(
            question,
            candidates,
        )

        # 5. Remove restricted content
        candidates = self._filter_restricted(
            candidates,
        )

        # 6. Final top-N
        return candidates[:top_n]