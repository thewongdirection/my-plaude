"""Optional speaker diarization ("who spoke when") via pyannote.audio.

This is an optional extra. It is heavier than the core (pulls in torch and
pyannote.audio) and the pretrained pipeline requires a one-time, free Hugging
Face access-token step:

    1. Create a token at https://hf.co/settings/tokens
    2. Accept the model terms at https://hf.co/pyannote/speaker-diarization-3.1
    3. Pass --hf-token / set HF_TOKEN (first run downloads weights; then offline)

Diarization runs independently of transcription, then we merge the two by
timestamp overlap. When word-level timestamps are available we assign each word
to a speaker and regroup, which produces clean speaker turns even when a single
Whisper segment spans two speakers.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple


class DiarizeError(RuntimeError):
    """Raised when diarization is unavailable or fails."""


def _load_pipeline(hf_token: Optional[str]):
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise DiarizeError(
            "pyannote.audio is not installed. Install the optional extra with "
            "`pip install pyannote.audio` (also pulls in torch)."
        ) from exc

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
    except Exception as exc:
        raise DiarizeError(
            "could not load the pyannote diarization pipeline. On first use you "
            "must accept the model terms on Hugging Face and provide an access "
            "token via --hf-token or the HF_TOKEN environment variable. "
            f"Underlying error: {exc}"
        ) from exc

    if pipeline is None:
        raise DiarizeError(
            "pyannote returned no pipeline - this almost always means the access "
            "token is missing or the model terms have not been accepted."
        )

    # Move to GPU when available (torch is already a pyannote dependency here).
    try:
        import torch

        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
    except Exception:
        pass
    return pipeline


def _turns(annotation) -> List[Tuple[float, float, str]]:
    """Flatten a pyannote annotation to sorted (start, end, speaker) tuples."""
    out = [
        (float(seg.start), float(seg.end), str(label))
        for seg, _, label in annotation.itertracks(yield_label=True)
    ]
    out.sort(key=lambda t: t[0])
    return out


def _best_speaker(start: float, end: float, turns: List[Tuple[float, float, str]]) -> Optional[str]:
    """Return the speaker whose turn overlaps [start, end] the most."""
    best_label: Optional[str] = None
    best_overlap = 0.0
    for t_start, t_end, label in turns:
        overlap = min(end, t_end) - max(start, t_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = label
    return best_label


def _relabel(turns: List[Tuple[float, float, str]]) -> Dict[str, str]:
    """Map pyannote labels (SPEAKER_00...) to friendly, first-seen ordering."""
    mapping: Dict[str, str] = {}
    for _, _, label in turns:
        if label not in mapping:
            mapping[label] = f"Speaker {len(mapping) + 1}"
    return mapping


def diarize_and_merge(
    audio_path: str,
    segments: List[Dict[str, Any]],
    *,
    hf_token: Optional[str] = None,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Assign speakers to ``segments`` and return new, speaker-tagged segments.

    If the segments carry word-level timestamps, output is regrouped into
    contiguous same-speaker turns. Otherwise each input segment is tagged as a
    whole with its best-overlapping speaker.
    """
    pipeline = _load_pipeline(hf_token)

    kwargs: Dict[str, Any] = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    try:
        annotation = pipeline(audio_path, **kwargs)
    except Exception as exc:
        raise DiarizeError(f"diarization failed: {exc}") from exc

    turns = _turns(annotation)
    if not turns:
        return segments
    names = _relabel(turns)

    has_words = any(seg.get("words") for seg in segments)
    if not has_words:
        for seg in segments:
            label = _best_speaker(seg["start"], seg["end"], turns)
            seg["speaker"] = names.get(label) if label else None
        return segments

    # Word-level path: tag each word, then merge consecutive same-speaker words.
    merged: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for seg in segments:
        for w in seg.get("words", []):
            label = _best_speaker(w["start"], w["end"], turns)
            speaker = names.get(label) if label else None
            if current is not None and current["speaker"] == speaker:
                current["end"] = w["end"]
                current["text"] += w["word"]
            else:
                if current is not None:
                    merged.append(current)
                current = {
                    "start": w["start"],
                    "end": w["end"],
                    "text": w["word"],
                    "speaker": speaker,
                }
    if current is not None:
        merged.append(current)

    for seg in merged:
        seg["text"] = seg["text"].strip()
    return merged
