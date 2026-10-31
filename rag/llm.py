"""Gemini client with retry/backoff for rate limits."""
import os
import time


class GeminiClient:
    def __init__(self, api_key: str | None = None, model: str = "gemini-3.5-flash"):
        from google import genai

        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("No Gemini API key: set GEMINI_API_KEY or enter it in the app sidebar.")
        self.client = genai.Client(api_key=key)
        self.model = model

    def generate(self, system: str, prompt: str, schema=None) -> str:
        """Return the model's text. With `schema` (a Pydantic model) the output is guaranteed JSON."""
        from google.genai import types

        extra = {"response_mime_type": "application/json", "response_schema": schema} if schema else {}
        cfg = types.GenerateContentConfig(system_instruction=system, temperature=0, **extra)
        for attempt in range(5):
            try:
                resp = self.client.models.generate_content(model=self.model, contents=prompt, config=cfg)
                if not resp.text:
                    raise RuntimeError("Gemini returned an empty response (possibly blocked by safety filters).")
                return resp.text
            except Exception as e:
                msg = str(e)
                transient = any(x in msg for x in ("429", "500", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE"))
                if not transient or attempt == 4:
                    raise
                time.sleep(3 * 2 ** attempt)  # 3s, 6s, 12s, 24s
