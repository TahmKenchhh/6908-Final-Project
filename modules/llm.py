"""
LLM interface via Ollama with conversation history.
Vision queries route to Gemini Flash (if GEMINI_API_KEY is set) or
fall back to the local VISION_MODEL (qwen3-vl:2b).
"""

import base64
from typing import Generator
import ollama

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    OLLAMA_MODEL, VISION_MODEL, OLLAMA_HOST, SYSTEM_PROMPT,
    LLM_NUM_PREDICT, VISION_NUM_PREDICT,
    GEMINI_API_KEY, GEMINI_VISION_MODEL,
)

_VISION_BREVITY = "Answer in one short sentence. "

# Lazy-initialised Gemini client
_gemini_client = None

def _get_gemini():
    global _gemini_client
    if _gemini_client is None and GEMINI_API_KEY:
        from google import genai
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client


def _gemini_vision(prompt: str, image_b64: str) -> Generator[str, None, None]:
    """Stream a vision response from Gemini Flash."""
    from google.genai import types
    client = _get_gemini()
    image_bytes = base64.b64decode(image_b64)
    contents = [
        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        types.Part.from_text(text=_VISION_BREVITY + prompt),
    ]
    for chunk in client.models.generate_content_stream(
        model=GEMINI_VISION_MODEL,
        contents=contents,
    ):
        if chunk.text:
            yield chunk.text


class LLM:
    def __init__(self):
        self.client = ollama.Client(host=OLLAMA_HOST)
        self.history: list[dict] = []
        vision_backend = f"Gemini {GEMINI_VISION_MODEL}" if GEMINI_API_KEY else VISION_MODEL
        print(f"LLM ready (text: {OLLAMA_MODEL}, vision: {vision_backend}).")

    def chat(self, user_message: str, image_b64: str | None = None) -> Generator[str, None, None]:
        """
        Stream a response to user_message.
        Pass image_b64 for vision queries (one-shot, not added to history).
        Routes to Gemini if key is set, otherwise local qwen3-vl.
        """
        if image_b64:
            if GEMINI_API_KEY:
                try:
                    yield from _gemini_vision(user_message, image_b64)
                    return
                except Exception as e:
                    print(f"[gemini] failed ({e.__class__.__name__}: {str(e)[:80]}), falling back to local")
            yield from self._local_vision(user_message, image_b64)
            return

        # Text-only query
        self.history.append({"role": "user", "content": user_message})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history
        extra = {"think": False} if OLLAMA_MODEL.startswith("qwen3") else {}

        full_response = ""
        for chunk in self.client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            stream=True,
            options={"num_predict": LLM_NUM_PREDICT, "num_ctx": 1024},
            **extra,
        ):
            content = chunk["message"].get("content") or ""
            full_response += content
            yield content

        self.history.append({"role": "assistant", "content": full_response})

    def _local_vision(self, user_message: str, image_b64: str) -> Generator[str, None, None]:
        prompt = _VISION_BREVITY + user_message
        messages = [{"role": "user", "content": prompt, "images": [image_b64]}]
        extra = {"think": False} if VISION_MODEL.startswith("qwen3") else {}
        for chunk in self.client.chat(
            model=VISION_MODEL,
            messages=messages,
            stream=True,
            options={"num_predict": VISION_NUM_PREDICT},
            **extra,
        ):
            content = chunk["message"].get("content") or ""
            yield content

    def clear_history(self) -> None:
        self.history = []
        print("Conversation history cleared.")

    def unload(self) -> None:
        for m in (OLLAMA_MODEL, VISION_MODEL):
            try:
                self.client.chat(model=m, messages=[], keep_alive=0)
            except Exception:
                pass
