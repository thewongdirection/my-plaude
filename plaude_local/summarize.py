"""Local, offline content summarization of a transcript.

Uses a local LLM server - either **Ollama** or a **llama.cpp** server - so
summarization, like the rest of the tool, needs no cloud. Transport is plain
``urllib`` from the standard library, so this adds **no Python dependencies**.

Two backends:

* ``ollama``   - talks to the Ollama HTTP API (default http://127.0.0.1:11434).
                 Requires Ollama running and a pulled model (e.g. ``llama3.1``).
* ``llamacpp`` - talks to a llama.cpp ``llama-server`` (default
                 http://127.0.0.1:8080), using its native ``/completion`` API.

The summarization *logic* (chunking a long transcript, map-reduce combine) is
separated from the *transport* (the HTTP call), so it is unit-tested without a
network by injecting a fake call function.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Callable, List, Optional

# Use 127.0.0.1, not "localhost": on Windows the PowerShell port's HTTP client
# resolves "localhost" to IPv6 ::1 first and stalls for seconds before falling
# back to IPv4, which blew past the short backend-detection timeout and made the
# local LLM look unreachable. 127.0.0.1 connects immediately on both platforms.
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_LLAMACPP_URL = "http://127.0.0.1:8080"
BACKENDS = ("ollama", "llamacpp")

# Keep each model call within a modest context budget. Transcripts longer than
# this are summarized in chunks and then combined (map-reduce).
DEFAULT_MAX_CHARS = 8000
# Cap lines per translation request. Very large batches overflow the LLM context
# window, so the numbered reply comes back truncated/misformatted and parses to
# nothing. Small batches translate reliably and align cleanly.
DEFAULT_MAX_TRANSLATE_LINES = 40
# Fixed Ollama context window for all calls. Constant (not per-prompt) so the
# model stays warm; 2x the ~4k default so small translation batches and normal
# summary chunks aren't truncated.
_OLLAMA_NUM_CTX = 8192


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
    except OSError as exc:
        # OSError covers URLError/HTTPError, socket TimeoutError, and connection
        # errors. A raw TimeoutError must NOT escape uncaught: it would crash the
        # whole run instead of degrading to a summary note / "unavailable".
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


def list_ollama_models(url: str = DEFAULT_OLLAMA_URL, timeout: float = 3.0) -> List[str]:
    """Return the names of every model installed on the Ollama server.

    Reads ``/api/tags``; returns an empty list if the server is unreachable or
    has no models pulled.
    """
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/api/tags", timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models = data.get("models") or []
        return [m.get("name") for m in models if m.get("name")]
    except Exception:
        return []


def default_ollama_model(url: str = DEFAULT_OLLAMA_URL, timeout: float = 3.0) -> Optional[str]:
    """The model to use when none is specified: the first one Ollama has installed.

    Ollama has no notion of a designated default, so we take the first entry from
    ``/api/tags`` (the installed models). Returns ``None`` if the server is
    unreachable or has no models pulled.
    """
    models = list_ollama_models(url, timeout)
    return models[0] if models else None


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
    # Use a FIXED context window (not sized per prompt): num_ctx is a load-time
    # parameter, so a value that changes call-to-call forces Ollama to reload the
    # model every request, which thrashes it into timeouts/500s. One constant
    # keeps the model warm while still being well above the truncating ~4k
    # default (with translation batches capped small, this holds the full prompt).
    payload = {
        "model": model, "prompt": prompt, "stream": False,
        "options": {"num_ctx": _OLLAMA_NUM_CTX},
    }
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


def build_aligned_translate_prompt(numbered: str, target_language: str) -> str:
    """Prompt to translate numbered lines, preserving the line count."""
    return (
        f"Translate each numbered line below into {target_language}. Output EXACTLY "
        "the same number of lines, each starting with its number and a period, then "
        f"the {target_language} translation of that line only. Consider the whole "
        "passage for context, but do not merge, split, reorder, add, or drop lines, "
        f"and output nothing but the numbered {target_language} lines.\n\n{numbered}"
        f"\n\n{target_language} (numbered):"
    )


_NUM_LINE = re.compile(r"(?m)^\s*(\d+)[.\):]\s*(.*)$")
_THINK = re.compile(r"(?is)<think>.*?</think>")


def _translate_group(
    group: List[int],
    lines: List[str],
    call: Callable[[str], str],
    target_language: str,
    out: List[str],
) -> None:
    """Translate one group of line indices into ``out``.

    If the numbered reply maps back to fewer than half the lines (a sign the
    batch overflowed the context and was truncated/misformatted), split the group
    and retry each half — down to single lines — so a batch never silently comes
    back all-empty.
    """
    numbered = "\n".join(f"{n + 1}. {lines[idx]}" for n, idx in enumerate(group))
    resp = _THINK.sub("", call(build_aligned_translate_prompt(numbered, target_language)))
    got = {int(m.group(1)): m.group(2).strip() for m in _NUM_LINE.finditer(resp)}
    matched = sum(1 for n in range(len(group)) if got.get(n + 1))
    if len(group) == 1 or matched * 2 >= len(group):
        for n, idx in enumerate(group):
            out[idx] = got.get(n + 1, "")
        return
    mid = len(group) // 2
    _translate_group(group[:mid], lines, call, target_language, out)
    _translate_group(group[mid:], lines, call, target_language, out)


def translate_lines(
    lines: List[str],
    call: Callable[[str], str],
    *,
    target_language: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_lines: int = DEFAULT_MAX_TRANSLATE_LINES,
) -> List[str]:
    """Translate segment lines, keeping one output line per input line.

    Batches lines (with full-batch context) into requests bounded by both
    ``max_chars`` and ``max_lines`` and parses the numbered replies back per
    line, so the Side-by-Side stays aligned while the LLM does the (more
    accurate) translation. A reply that doesn't map back cleanly is retried on
    smaller sub-batches. Returns a list the same length as ``lines``.
    """
    out = [""] * len(lines)
    batch: List[int] = []
    blen = 0
    batches: List[List[int]] = []
    for i, ln in enumerate(lines):
        add = len(ln) + 6
        if batch and (blen + add > max_chars or len(batch) >= max_lines):
            batches.append(batch)
            batch, blen = [], 0
        batch.append(i)
        blen += add
    if batch:
        batches.append(batch)

    for group in batches:
        _translate_group(group, lines, call, target_language, out)
    return out


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
        # Default to whatever model Ollama has installed (no hard-coded name).
        chosen_model = model or default_ollama_model(base)
        if not chosen_model:
            raise SummarizeError(
                "no Ollama model available. Pull one (e.g. `ollama pull llama3.1`) "
                "or pass an explicit model via --summarize-model / --translate-model."
            )
        return lambda p: _ollama_call(p, chosen_model, base, timeout)
    if backend == "llamacpp":
        # llama.cpp serves a single preloaded model; no name needed.
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


def translate_segments(
    lines: List[str],
    *,
    target_language: str = "English",
    backend: str = "auto",
    model: Optional[str] = None,
    url: Optional[str] = None,
    timeout: float = 120.0,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_lines: int = DEFAULT_MAX_TRANSLATE_LINES,
) -> List[str]:
    """Translate per-segment lines with a local LLM (accuracy-first, aligned).

    Returns one translated line per input line so the Side-by-Side stays aligned.
    """
    call = _resolve_call(backend, model, url, timeout)
    return translate_lines(lines, call, target_language=target_language,
                           max_chars=max_chars, max_lines=max_lines)
