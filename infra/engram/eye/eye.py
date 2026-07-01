"""Engram's eye — a thin client to a local vision-language model.

Hand it a JPEG-encoded frame; it returns a :class:`Reading` (the model's text +
how long it took). That's the whole surface. No camera, no display, no threads —
those live in ``bench.py`` — so this same class drops straight into the perceiving
loop later as the cheap, always-on eye that gates the expensive model.

Backed by llama.cpp's ``llama-server`` running a SmolVLM GGUF (OpenAI-compatible
``/v1/chat/completions`` with image_url). Local, no API key, no cloud. Swapping to
a bigger VLM (or a different server) is a one-line URL/model change — the seam, as
with core.py's ModelDriver, is the point.
"""
from __future__ import annotations

import base64
import json
import time
import urllib.request
from dataclasses import dataclass

DEFAULT_SERVER = "http://127.0.0.1:8080"

# Small VLMs answer a SPECIFIC question far better than "describe everything"
# (verified on SmolVLM-500M: a directive prompt found the people + objects a bare
# "list objects" prompt collapsed). This is the bench's default; edit it live.
DEFAULT_PROMPT = (
    "Look at this webcam frame. How many people are in it? Describe each person "
    "(hair, facial hair, glasses, clothing) and list the main objects. Be concise."
)


@dataclass
class Reading:
    text: str
    latency: float          # seconds for the round-trip
    ok: bool = True


class Eye:
    def __init__(self, server: str = DEFAULT_SERVER, prompt: str = DEFAULT_PROMPT,
                 max_tokens: int = 220, temperature: float = 0.1,
                 timeout: float = 120.0) -> None:
        self.server = server.rstrip("/")
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

    def look(self, jpeg: bytes, prompt: str | None = None) -> Reading:
        """Send one JPEG frame to the VLM and return its reading."""
        b64 = base64.b64encode(jpeg).decode()
        body = {
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt or self.prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        t = time.time()
        try:
            req = urllib.request.Request(
                f"{self.server}/v1/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                out = json.load(r)
            txt = out["choices"][0]["message"]["content"].strip()
            return Reading(txt, time.time() - t, ok=True)
        except Exception as exc:  # noqa: BLE001 — never crash the bench loop
            return Reading(f"[eye error: {type(exc).__name__}: {exc}]",
                           time.time() - t, ok=False)

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.server}/health", timeout=3):
                return True
        except Exception:  # noqa: BLE001
            return False
