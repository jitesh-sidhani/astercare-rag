from __future__ import annotations

import sys
from pathlib import Path

# Make project root importable when running:
# python scripts\ingest.py
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import HANDBOOK_PATH
from src.chunking import load_chunks
from src.embeddings import EmbeddingService
from src.vector_store import ChromaStore


def main() -> None:
    print("Loading handbook...")

    # ---------------------------------------------------------
    # 1. Parse and chunk handbook
    # ---------------------------------------------------------
    chunks = load_chunks(HANDBOOK_PATH)

    print(f"Created chunks: {len(chunks)}")

    if not chunks:
        raise RuntimeError("No chunks were created.")

    # ---------------------------------------------------------
    # 2. Initialize OpenAI embedding service
    # ---------------------------------------------------------
    print("Initializing OpenAI embedding service...")

    embedder = EmbeddingService()

    # ---------------------------------------------------------
    # 3. Initialize ChromaDB
    # ---------------------------------------------------------
    print("Initializing ChromaDB...")

    store = ChromaStore()

    # ---------------------------------------------------------
    # 4. Generate embeddings and store in batches
    # ---------------------------------------------------------
    batch_size = 32

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]

        print(
            f"Generating embeddings for "
            f"{start + 1}-{min(start + batch_size, len(chunks))}"
            f"/{len(chunks)}..."
        )

        texts = [chunk.text for chunk in batch]

        embeddings = embedder.embed_documents(texts)

        chunk_dicts = [chunk.to_dict() for chunk in batch]

        store.upsert(
            chunk_dicts,
            embeddings,
        )

        print(
            f"Indexed "
            f"{min(start + batch_size, len(chunks))}"
            f"/{len(chunks)}"
        )

    print("\nIngestion complete.")
    print(f"Total chunks indexed: {len(chunks)}")


if __name__ == "__main__":
    main()