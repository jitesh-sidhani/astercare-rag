from pathlib import Path
from typing import Dict, List

import chromadb

from .config import CHROMA_DIR, COLLECTION_NAME


class ChromaStore:
    def __init__(self):
        Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=CHROMA_DIR
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine"
            },
        )

    def upsert(
        self,
        chunks: List[Dict],
        embeddings: List[List[float]],
    ) -> None:

        metadatas = []

        for chunk in chunks:
            metadatas.append(
                {
                    "section_number": chunk["section_number"],
                    "section_title": chunk["section_title"],
                    "heading": chunk["heading"],
                    "paragraph_start": chunk["paragraph_start"],
                    "paragraph_end": chunk["paragraph_end"],
                    "authority": chunk["authority"],
                    "restricted": chunk["restricted"],
                    "chunk_type": chunk["chunk_type"],
                }
            )

        self.collection.upsert(
            ids=[
                chunk["chunk_id"]
                for chunk in chunks
            ],
            embeddings=embeddings,
            documents=[
                chunk["text"]
                for chunk in chunks
            ],
            metadatas=metadatas,
        )

    def query(
        self,
        embedding: List[float],
        n_results: int = 8,
    ) -> Dict:

        return self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
        )