from typing import List
from openai import OpenAI
from .config import OPENAI_API_KEY, EMBEDDING_MODEL


class EmbeddingService:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env")
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
