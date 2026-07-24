"""Local, offline content summarization of a transcript.

Uses a local LLM server - either **Ollama** or a **llama.cpp** server - so
summarization, like the rest of the tool, needs no cloud. Transport is plain
``urllib`` from the standard library, so this adds **no Python dependencies**.

Two backends:

* ``ollama``   - talks to the Ollama HTTP API (default http://localhost:11434).
                 Requires Ollama running and a pulled model (e.g. ``llama3.1``).
* ``llamacpp`` - talks to a llama.cpp ``llama-server`` (default
                 http://localhost:8080), using its native ``/completion`` API.

The summarization *logic* (chunking a long transcript, map-reduce combine) is
separated from the *transport* (the HTTP call), so it is unit-tested without a
network by injecting a fake call function.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Callable, List, Optional

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_LLAMACPP_URL = "http://localhost:8080"
DEFAULT_OLLAMA_MODEL = "llama3.1"
BACKENDS = ("ollama", "llamacpp")

# Keep each model call within a modest context budget. Transcripts longer than
# this are summarized in chunks and then combined (map-reduce).
DEFAULT_MAX_CHARS = 8000


class SummarizeError(RuntimeError):
    """Raised when summarization is unavailable or fails."""


# --------------------------------------------------------------------------- #
# HTTP helpers (stdlib only)
# --------------------------------------------------------------------------- #

def _post_json(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.URLError as exc:
        raise SummarizeError(
            f"could not reach the summarization server at {url} ({exc}). "
            "Is the local model server running?"
        ) from exc
    try:
        result = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise SummarizeError(
            f"the summarization server at {url} returned an unexpected "
            f"(non-JSON) response: {exc}"
        ) from exc
    if not isinstance(result, dict):
        raise SummarizeError(
            f"the summarization server at {url} returned JSON that is not an "
            f"object (got {type(result).__name__})."
        )
    return result


def _get_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Backend availability (used by the preflight/doctor check too)
# --------------------------------------------------------------------------- #

def ollama_available(url: str = DEFAULT_OLLAMA_URL) -> bool:
    return _get_ok(url.rstrip("/") + "/api/tags")


def llamacpp_available(url: str = DEFAULT_LLAMACPP_URL) -> bool:
    return _get_ok(url.rstrip("/") + "/health")


def detect_backend(
    ollama_url: str = DEFAULT_OLLAMA_URL,
    llamacpp_url: str = DEFAULT_LLAMACPP_URL,
) -> Optional[str]:
    """Return the first reachable backend name, or None if none respond."""
    if ollama_available(ollama_url):
        return "ollama"
    if llamacpp_available(llamacpp_url):
        return "llamacpp"
    return None


# --------------------------------------------------------------------------- #
# Backend calls
# --------------------------------------------------------------------------- #

def _ollama_call(prompt: str, model: str, url: str, timeout: float) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False}
    resp = _post_json(url.rstrip("/") + "/api/generate", payload, timeout)
    text = resp.get("response")
    if not text:
        raise SummarizeError(f"ollama returned no text (response: {resp!r})")
    return text.strip()


def _llamacpp_call(prompt: str, model: Optional[str], url: str, timeout: float) -> str:
    payload = {"prompt": prompt, "n_predict": 512, "temperature": 0.2, "stream": False}
    resp = _post_json(url.rstrip("/") + "/completion", payload, timeout)
    text = resp.get("content")
    if not text:
        raise SummarizeError(f"llama.cpp returned no text (response: {resp!r})")
    return text.strip()


# --------------------------------------------------------------------------- #
# Prompt + chunking logic (pure, unit-tested with an injected call)
# --------------------------------------------------------------------------- #

def build_prompt(text: str, *, combine: bool = False) -> str:
    if combine:
        head = (
            "You are a helpful assistant. Below are partial summaries of a longer "
            "transcript. Combine them into one concise final summary with key "
            "topics, decisions, and action items as bullet points."
        )
        label = "Partial summaries"
    else:
        head = (
            "You are a helpful assistant. Summarize the following transcript into "
            "concise bullet points capturing the key topics, decisions, and any "
            "action items. Preserve the original language of the transcript."
        )
        label = "Transcript"
    return f"{head}\n\n{label}:\n{text}\n\nSummary:"


def build_topics_prompt(text: str, *, combine: bool = False, max_words: int = 250) -> str:
    """Prompt for the dashboard's <=250-word 'critical topics' summary."""
    if combine:
        head = (
            "Below are partial notes from a longer transcript. Combine them into a "
            f"single summary of the most critical topics, in at most {max_words} "
            "words. No preamble. Preserve the original language of the transcript."
        )
        label = "Partial notes"
    else:
        head = (
            "Summarize the most critical topics discussed in the following "
            f"transcript in at most {max_words} words. Focus on what matters most; "
            "no preamble or meta commentary. Preserve the original language."
        )
        label = "Transcript"
    return f"{head}\n\n{label}:\n{text}\n\nSummary:"


def build_translate_prompt(text: str, target_language: str) -> str:
    """Prompt to translate text into ``target_language`` (LLM path)."""
    return (
        f"Translate the following text into {target_language}. Output ONLY the "
        "translation, preserving line breaks and meaning, with no preamble, notes, "
        f"or the original text.\n\nText:\n{text}\n\n{target_language} translation:"
    )


def _chunk(text: str, max_chars: int) -> List[str]:
    """Split text into <= max_chars pieces, preferring paragraph/line breaks.

    ``max_chars`` is clamped to at least 1 and the cut point always advances by
    at least one character, so this can never loop forever regardless of input.
    Empty/whitespace-only pieces are dropped.
    """
    max_chars = max(int(max_chars), 1)
    if len(text) <= max_chars:
        return [text] if text.strip() else []
    chunks: List[str] = []
    remaining = text
    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        cut = window.rfind("\n")
        if cut < max_chars // 2:  # no good break point; hard split
            cut = max_chars
        cut = max(cut, 1)  # guarantee forward progress
        piece = remaining[:cut].strip()
        if piece:
            chunks.append(piece)
        remaining = remaining[cut:]
    if remaining.strip():
        chunks.append(remaining.strip())
    return chunks


def summarize_text(
    text: str,
    call: Callable[[str], str],
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    prompt_builder: Callable[..., str] = build_prompt,
    _depth: int = 0,
) -> str:
    """Summarize ``text`` using ``call`` (prompt -> completion).

    Short text is summarized in one shot. Long text is chunked, each chunk
    summarized, and the partial summaries combined (recursively, bounded).
    ``prompt_builder`` lets callers swap the prompt (e.g. the dashboard's
    ``build_topics_prompt``); it must accept ``(text, *, combine=bool)``.
    """
    text = text.strip()
    if not text:
        raise SummarizeError("nothing to summarize: the transcript is empty.")

    max_chars = max(int(max_chars), 1)
    chunks = _chunk(text, max_chars)
    if len(chunks) == 1:
        return call(prompt_builder(chunks[0], combine=False)).strip()

    partials = [call(prompt_builder(c, combine=False)).strip() for c in chunks]
    combined = "\n\n".join(partials)
    if len(combined) <= max_chars or _depth >= 3:
        return call(prompt_builder(combined, combine=True)).strip()
    # Combined summaries are still huge - reduce again.
    return summarize_text(combined, call, max_chars=max_chars,
                          prompt_builder=prompt_builder, _depth=_depth + 1)


def translate_text(
    text: str,
    call: Callable[[str], str],
    *,
    target_language: str,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    """Translate ``text`` into ``target_language`` via ``call`` (chunked)."""
    text = text.strip()
    if not text:
        return ""
    chunks = _chunk(text, max_chars)
    return "\n".join(
        call(build_translate_prompt(c, target_language)).strip() for c in chunks
    )


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def _resolve_call(
    backend: str, model: Optional[str], url: Optional[str], timeout: float
) -> Callable[[str], str]:
    """Resolve a local LLM ``call`` (prompt -> completion) for the backend."""
    if backend == "auto":
        detected = detect_backend(
            url or DEFAULT_OLLAMA_URL, url or DEFAULT_LLAMACPP_URL
        )
        if detected is None:
            raise SummarizeError(
                "no local LLM server detected. Start one of:\n"
                "  - Ollama:    install from https://ollama.com/download, then "
                "`ollama pull llama3.1` and `ollama serve`\n"
                "  - llama.cpp: https://github.com/ggml-org/llama.cpp, then run "
                "`llama-server -m your-model.gguf`"
            )
        backend = detected

    if backend == "ollama":
        base = url or DEFAULT_OLLAMA_URL
        chosen_model = model or DEFAULT_OLLAMA_MODEL
        return lambda p: _ollama_call(p, chosen_model, base, timeout)
    if backend == "llamacpp":
        base = url or DEFAULT_LLAMACPP_URL
        return lambda p: _llamacpp_call(p, model, base, timeout)
    raise SummarizeError(
        f"unknown backend: {backend!r} (choose from {', '.join(BACKENDS)}, or auto)"
    )


def summarize(
    text: str,
    *,
    backend: str = "auto",
    model: Optional[str] = None,
    url: Optional[str] = None,
    timeout: float = 120.0,
    max_chars: int = DEFAULT_MAX_CHARS,
    topics: bool = False,
    max_words: int = 250,
) -> str:
    """Summarize a transcript with a local LLM backend.

    ``topics=True`` uses the dashboard's <=``max_words`` critical-topics prompt.
    """
    call = _resolve_call(backend, model, url, timeout)
    builder = build_prompt
    if topics:
        builder = lambda t, *, combine=False: build_topics_prompt(
            t, combine=combine, max_words=max_words
        )
    return summarize_text(text, call, max_chars=max_chars, prompt_builder=builder)


def translate(
    text: str,
    *,
    target_language: str = "English",
    backend: str = "auto",
    model: Optional[str] = None,
    url: Optional[str] = None,
    timeout: float = 120.0,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    """Translate a transcript into ``target_language`` with a local LLM backend.

    Used for non-English dashboard targets; English uses Whisper's translate task.
    """
    call = _resolve_call(backend, model, url, timeout)
    return translate_text(text, call, target_language=target_language, max_chars=max_chars)
