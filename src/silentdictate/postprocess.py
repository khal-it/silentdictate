"""Text post-processing: replacements, spoken punctuation, optional local LLM cleanup."""

from __future__ import annotations

import re
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx

from .config import PostprocessConfig

LLM_PROMPT = (
    "You correct dictated text. Fix punctuation, capitalization and obvious speech-recognition "
    "errors only. Never add, answer, explain, summarize or remove content. Reply with the "
    "corrected text and nothing else."
)
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
MAX_WORD_GROWTH = 1.2

# A trailing punctuation mark added by the recognizer is absorbed into the command.
_ANYWHERE = {
    "new paragraph": "\n\n",
    "new line": "\n",
    "comma": ",",
    "semicolon": ";",
    "open paren": "(",
    "close paren": ")",
}
# These are ordinary words too, so they only count at the end of a phrase.
_PHRASE_END = {
    "question mark": "?",
    "exclamation mark": "!",
    "exclamation point": "!",
    "full stop": ".",
    "period": ".",
    "colon": ":",
}
# German nouns are capitalized by Whisper; matching ignores case.
_ANYWHERE_DE = {
    "neuer absatz": "\n\n",
    "neue zeile": "\n",
    "komma": ",",
    "semikolon": ";",
    "klammer auf": "(",
    "klammer zu": ")",
}
_PHRASE_END_DE = {
    "fragezeichen": "?",
    "ausrufezeichen": "!",
    "doppelpunkt": ":",
    "punkt": ".",
}
_END = r"(?=[.,!?;:]*\s*$|[.,!?;:]+\s)"


def _sub_command(text: str, spoken: str, symbol: str, end: str) -> str:
    pattern = re.compile(rf"[\s,.]*\b{re.escape(spoken)}\b{end}[.,!?;:]*", re.IGNORECASE)
    return pattern.sub(lambda _m: symbol, text)


def _tables(language: str) -> tuple[dict[str, str], dict[str, str]]:
    """Return command tables; an empty language means auto-detect, so both apply."""
    if language == "de":
        return _ANYWHERE_DE, _PHRASE_END_DE
    if language:
        return _ANYWHERE, _PHRASE_END
    return {**_ANYWHERE, **_ANYWHERE_DE}, {**_PHRASE_END, **_PHRASE_END_DE}


def apply_spoken_punctuation(text: str, language: str = "en") -> str:
    """Replace spoken punctuation commands with symbols."""
    anywhere, phrase_end = _tables(language)
    for spoken, symbol in anywhere.items():
        text = _sub_command(text, spoken, symbol, "")
    for spoken, symbol in phrase_end.items():
        text = _sub_command(text, spoken, symbol, _END)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"([,;:.!?])(?=[^\s\d)\"'])", r"\1 ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    return text.strip(" \t")


def apply_replacements(text: str, replacements: dict[str, str]) -> str:
    """Apply case-insensitive whole-phrase replacements."""
    for source, target in replacements.items():
        pattern = rf"(?<!\w){re.escape(source)}(?!\w)"
        text = re.sub(pattern, target.replace("\\", r"\\"), text, flags=re.IGNORECASE)
    return text


class LlmCleanup:
    """Cleanup through a local OpenAI-compatible endpoint; failures return the input."""

    def __init__(self, endpoint: str, model: str, client: Any = None) -> None:
        if urlparse(endpoint).hostname not in LOCAL_HOSTS:
            raise ValueError(f"LLM endpoint must be local, got {endpoint}")
        self.url = endpoint.rstrip("/") + "/chat/completions"
        self.model = model
        self.client = client or httpx.Client(timeout=10, trust_env=False)

    def __call__(self, text: str) -> str:
        """Return cleaned text, or the input if the model failed or added content."""
        body = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": LLM_PROMPT},
                {"role": "user", "content": text},
            ],
        }
        try:
            response = self.client.post(self.url, json=body)
            response.raise_for_status()
            cleaned = str(response.json()["choices"][0]["message"]["content"]).strip()
        except (httpx.HTTPError, KeyError, IndexError, ValueError, TypeError) as exc:
            print(f"LLM cleanup skipped: {exc}")
            return text
        if not cleaned or len(cleaned.split()) > len(text.split()) * MAX_WORD_GROWTH:
            return text
        return cleaned


class PostProcessor(Protocol):
    """A pluggable text post-processor."""

    def process(self, text: str) -> str:
        """Transform recognized text."""
        ...


class DefaultPostProcessor:
    """Replacements, then spoken punctuation, then optional LLM cleanup."""

    def __init__(self, cfg: PostprocessConfig, cleanup: Any = None, language: str = "en") -> None:
        self.cfg = cfg
        self.language = language
        self.cleanup = cleanup
        if cleanup is None and cfg.llm_cleanup:
            self.cleanup = LlmCleanup(cfg.llm_endpoint, cfg.llm_model)

    def process(self, text: str) -> str:
        """Transform recognized text."""
        replaced = apply_replacements(text, self.cfg.replacements)
        text = apply_spoken_punctuation(replaced, self.language)
        return self.cleanup(text) if self.cleanup and text else text
