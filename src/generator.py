from __future__ import annotations

from openai import OpenAI

from .config import OPENAI_API_KEY, OPENAI_MODEL
from .prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
)


class AnswerGenerator:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. "
                "Add it to .env"
            )

        if not OPENAI_MODEL:
            raise RuntimeError(
                "OPENAI_MODEL is not configured."
            )

        self.client = OpenAI(
            api_key=OPENAI_API_KEY
        )

    def answer(
        self,
        question: str,
        contexts: list[dict],
    ) -> str:
        """
        Generate a grounded answer using only the retrieved
        AsterCare handbook context.
        """
        if not question.strip():
            return "Please enter a question."

        if not contexts:
            return (
                "The handbook context does not provide "
                "enough information to answer this question."
            )

        prompt = build_user_prompt(
            question,
            contexts,
        )

        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        answer = (
            response.choices[0]
            .message
            .content
        )

        if not answer:
            return (
                "The model did not return an answer "
                "from the supplied handbook context."
            )

        return answer.strip()