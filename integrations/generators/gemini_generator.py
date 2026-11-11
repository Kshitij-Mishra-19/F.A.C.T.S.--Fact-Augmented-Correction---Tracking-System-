
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from integrations.generators.base_generator import BaseGenerator


load_dotenv()


class GeminiGenerator(BaseGenerator):
    def __init__(self, model=None):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing. Add it to your .env file."
            )

        self.model = model or os.getenv(
            "GEMINI_MODEL", "gemini-3.8-flash"
        )

        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("The generation prompt cannot be empty.")

        system_instruction = """
You are the answer-generation component of HalluGuard AI.

Your job is to answer the user's request accurately and clearly.

When reviewer feedback is included:
1. Read every rejected claim and the available evidence.
2. Correct claims that conflict with reliable evidence.
3. Do not repeat claims that the reviewer has rejected.
4. If evidence is insufficient, acknowledge the uncertainty.
5. Do not invent sources, evidence, or verification results.
6. Return the revised answer only, without claiming that it
   has been approved by the reviewer.
"""

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2,
                ),
            )

            answer = response.text

            if not answer or not answer.strip():
                raise RuntimeError(
                    "Gemini returned an empty text response."
                )

            return answer.strip()

        except Exception as exc:
            raise RuntimeError(
                f"Gemini generation failed: {exc}"
            ) from exc
