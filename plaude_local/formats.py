"""Output writers for transcripts.

Everything here is standard-library only. A transcript is represented as a list
of "segments", each a plain dict:

    {"start": float, "end": float, "text": str, "speaker": str | None}

Times are in seconds. ``speaker`` is populated only when diarization ran.
"""

from __future__ import annotations

import json
from typing import Iterable, List, Dict, Any, Optional


Segment = Dict[str, Any]


def _fmt_timestamp(seconds: float, *, comma: bool = False) -> str:
    """Format a number of seconds as HH:MM:SS,mmm (SRT) or HH:MM:SS.mmm (VTT)."""
    if seconds < 0:
        seconds = 0.0
    millis = int(round(seconds * 1000.0))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    sep = "," if comma else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"


def _clock(seconds: float) -> str:
    """Short [MM:SS] / [HH:MM:SS] clock for plain-text output."""
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def to_text(segments: Iterable[Segment], *, timestamps: bool = False) -> str:
    """Plain-text transcript.

    Without speakers or timestamps this is just the joined text. With speakers
    it becomes a readable turn-by-turn dialogue.
    """
    lines: List[str] = []
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        speaker = seg.get("speaker")
        prefix_parts: List[str] = []
        if timestamps:
            prefix_parts.append(f"[{_clock(seg['start'])}]")
        if speaker:
            prefix_parts.append(f"{speaker}:")
        prefix = " ".join(prefix_parts)
        lines.append(f"{prefix} {text}".strip() if prefix else text)
    return "\n".join(lines) + ("\n" if lines else "")


def to_srt(segments: Iterable[Segment]) -> str:
    """SubRip (.srt) subtitles."""
    blocks: List[str] = []
    index = 1
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        speaker = seg.get("speaker")
        if speaker:
            text = f"{speaker}: {text}"
        start = _fmt_timestamp(seg["start"], comma=True)
        end = _fmt_timestamp(seg["end"], comma=True)
        blocks.append(f"{index}\n{start} --> {end}\n{text}\n")
        index += 1
    return "\n".join(blocks)


def to_vtt(segments: Iterable[Segment]) -> str:
    """WebVTT (.vtt) subtitles."""
    blocks: List[str] = ["WEBVTT\n"]
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        speaker = seg.get("speaker")
        if speaker:
            text = f"<v {speaker}>{text}"
        start = _fmt_timestamp(seg["start"], comma=False)
        end = _fmt_timestamp(seg["end"], comma=False)
        blocks.append(f"{start} --> {end}\n{text}\n")
    return "\n".join(blocks)


def to_json(segments: Iterable[Segment], *, meta: Optional[Dict[str, Any]] = None) -> str:
    """Machine-readable JSON: metadata plus the full segment list."""
    payload: Dict[str, Any] = {"segments": list(segments)}
    if meta:
        payload["meta"] = meta
    return json.dumps(payload, ensure_ascii=False, indent=2)


_WRITERS = {
    "txt": lambda segs, meta: to_text(segs, timestamps=bool(meta and meta.get("timestamps"))),
    "srt": lambda segs, meta: to_srt(segs),
    "vtt": lambda segs, meta: to_vtt(segs),
    "json": lambda segs, meta: to_json(segs, meta=meta),
}

FORMATS = tuple(_WRITERS.keys())


def render(segments: Iterable[Segment], fmt: str, *, meta: Optional[Dict[str, Any]] = None) -> str:
    """Render ``segments`` to the requested format name."""
    try:
        writer = _WRITERS[fmt]
    except KeyError:
        raise ValueError(f"unknown output format: {fmt!r} (choose from {', '.join(FORMATS)})")
    segs = list(segments)
    return writer(segs, meta)
