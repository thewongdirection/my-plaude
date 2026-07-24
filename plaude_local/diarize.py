"""Optional speaker diarization ("who spoke when").

Two backends are supported, selectable via ``backend=``:

* ``pyannote``  - uses ``pyannote.audio`` directly (lighter install).
* ``whisperx``  - uses WhisperX's diarization pipeline (bundles pyannote plus
                  WhisperX's conveniences; handy if you already use WhisperX).

Both backends produce the same intermediate representation - a list of
``(start, end, speaker)`` turns - which is then fed through one shared,
well-tested merge routine (:func:`merge_turns`). This keeps the transcript
output identical regardless of backend.

Both are optional extras (heavier than the core; they pull in torch) and both
require a one-time, free Hugging Face access-token step:

    1. Create a token at https://hf.co/settings/tokens
    2. Accept the model terms at https://hf.co/pyannote/speaker-diarization-3.1
    3. Pass --hf-token / set HF_TOKEN (first run downloads weights; then offline)
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple

BACKENDS = ("pyannote", "whisperx")

Turn = Tuple[float, float, str]


class DiarizeError(RuntimeError):
    """Raised when diarization is unavailable or fails."""


# --------------------------------------------------------------------------- #
# Shared, pure helpers (no heavy deps) - these are the regression-tested core
# --------------------------------------------------------------------------- #

def _best_speaker(start: float, end: float, turns: List[Turn]) -> Optional[str]:
    """Return the speaker whose turn overlaps ``[start, end]`` the most."""
    best_label: Optional[str] = None
    best_overlap = 0.0
    for t_start, t_end, label in turns:
        overlap = min(end, t_end) - max(start, t_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = label
    return best_label


def _relabel(turns: List[Turn]) -> Dict[str, str]:
    """Map raw backend labels (SPEAKER_00...) to friendly, first-seen names."""
    mapping: Dict[str, str] = {}
    for _, _, label in sorted(turns, key=lambda t: t[0]):
        if label not in mapping:
            mapping[label] = f"Speaker {len(mapping) + 1}"
    return mapping


def merge_turns(segments: List[Dict[str, Any]], turns: List[Turn]) -> List[Dict[str, Any]]:
    """Assign speakers from ``turns`` onto ``segments``.

    If segments carry word-level timestamps (a ``words`` list), the output is
    regrouped into contiguous same-speaker turns - so a single Whisper segment
    spanning two speakers is split correctly. Otherwise each segment is tagged
    as a whole with its best-overlapping speaker.

    ``turns`` empty -> segments returned unchanged. This function is pure and
    does not import any heavy dependencies.
    """
    if not turns:
        return segments

    names = _relabel(turns)
    has_words = any(seg.get("words") for seg in segments)

    if not has_words:
        for seg in segments:
            label = _best_speaker(seg["start"], seg["end"], turns)
            seg["speaker"] = names.get(label) if label else None
        return segments

    merged: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for seg in segments:
        words = seg.get("words")
        if not words:
            # This segment has no usable word timestamps (can happen at VAD /
            # silence edges even when word_timestamps was requested). Tag it as
            # a whole segment so its text is never dropped.
            if current is not None:
                merged.append(current)
                current = None
            text = seg["text"].strip()
            if not text:
                continue
            label = _best_speaker(seg["start"], seg["end"], turns)
            merged.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": text,
                "speaker": names.get(label) if label else None,
            })
            continue
        for w in words:
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


def _annotation_to_turns(annotation) -> List[Turn]:
    """Flatten a pyannote ``Annotation`` to sorted (start, end, speaker) turns."""
    out: List[Turn] = [
        (float(seg.start), float(seg.end), str(label))
        for seg, _, label in annotation.itertracks(yield_label=True)
    ]
    out.sort(key=lambda t: t[0])
    return out


def _dataframe_to_turns(df) -> List[Turn]:
    """Flatten a WhisperX diarization DataFrame to sorted turns.

    The DataFrame is expected to expose ``start``, ``end`` and ``speaker``
    columns (accessed via ``itertuples`` for pandas-free testability)."""
    out: List[Turn] = []
    for row in df.itertuples(index=False):
        out.append((float(row.start), float(row.end), str(row.speaker)))
    out.sort(key=lambda t: t[0])
    return out


def _speaker_kwargs(num, mn, mx) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if num is not None:
        kwargs["num_speakers"] = num
    if mn is not None:
        kwargs["min_speakers"] = mn
    if mx is not None:
        kwargs["max_speakers"] = mx
    return kwargs


# --------------------------------------------------------------------------- #
# Backend: pyannote.audio
# --------------------------------------------------------------------------- #

def _pyannote_from_pretrained(Pipeline, repo, token):
    """Load a pyannote pipeline across library versions.

    pyannote.audio >= 4 renamed the ``use_auth_token`` argument of
    ``Pipeline.from_pretrained`` to ``token``; older releases only accept the old
    name. Try the new name first and fall back so both work.
    """
    try:
        return Pipeline.from_pretrained(repo, token=token)
    except TypeError:
        return Pipeline.from_pretrained(repo, use_auth_token=token)


# The default diarization pipeline differs by pyannote.audio major version:
# 4.x replaced the 3.1 pipeline with the gated `speaker-diarization-community-1`
# model; 3.x uses `speaker-diarization-3.1`. `--diarize-model` overrides this.
PYANNOTE_MODEL_V4 = "pyannote/speaker-diarization-community-1"
PYANNOTE_MODEL_V3 = "pyannote/speaker-diarization-3.1"


def _model_for_pyannote_version(version: str) -> str:
    try:
        major = int(str(version).split(".")[0])
    except (ValueError, TypeError, AttributeError):
        major = 3
    return PYANNOTE_MODEL_V4 if major >= 4 else PYANNOTE_MODEL_V3


def _default_diarization_model() -> str:
    try:
        from pyannote.audio import __version__ as ver
    except Exception:  # pragma: no cover - environment dependent
        ver = "3"
    return _model_for_pyannote_version(ver)


def _pyannote_turns(audio_path, hf_token, device, num, mn, mx, model=None) -> List[Turn]:
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise DiarizeError(
            "pyannote.audio is not installed. Install the optional extra with "
            "`pip install pyannote.audio` (also pulls in torch)."
        ) from exc

    repo = model or _default_diarization_model()
    try:
        pipeline = _pyannote_from_pretrained(Pipeline, repo, hf_token)
    except Exception as exc:
        raise DiarizeError(
            f"could not load the pyannote pipeline {repo!r}. On first use you must "
            "accept the model terms on Hugging Face and provide an access token via "
            "--hf-token or the HF_TOKEN environment variable. Note: pyannote.audio "
            "4.x uses the gated 'pyannote/speaker-diarization-community-1' model, "
            "which needs its own one-time terms acceptance. "
            f"Underlying error: {exc}"
        ) from exc

    if pipeline is None:
        raise DiarizeError(
            "pyannote returned no pipeline - this almost always means the access "
            "token is missing or the model terms have not been accepted."
        )

    try:
        import torch

        if device == "cuda" and torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
    except Exception:
        pass

    try:
        annotation = pipeline(audio_path, **_speaker_kwargs(num, mn, mx))
    except Exception as exc:
        raise DiarizeError(f"pyannote diarization failed: {exc}") from exc
    return _annotation_to_turns(annotation)


# --------------------------------------------------------------------------- #
# Backend: WhisperX
# --------------------------------------------------------------------------- #

def _load_whisperx_pipeline(hf_token, device):
    try:
        # Newer WhisperX exposes it here; older releases at the top level.
        try:
            from whisperx.diarize import DiarizationPipeline
        except ImportError:
            from whisperx import DiarizationPipeline
    except ImportError as exc:
        raise DiarizeError(
            "whisperx is not installed. Install the optional extra with "
            "`pip install whisperx` (also pulls in torch)."
        ) from exc

    try:
        try:
            return DiarizationPipeline(token=hf_token, device=device)
        except TypeError:
            # whisperx pinned to older pyannote uses the old kwarg name.
            return DiarizationPipeline(use_auth_token=hf_token, device=device)
    except Exception as exc:
        raise DiarizeError(
            "could not create the WhisperX diarization pipeline. On first use "
            "you must accept the pyannote model terms on Hugging Face and "
            "provide an access token via --hf-token or HF_TOKEN. "
            f"Underlying error: {exc}"
        ) from exc


def _whisperx_turns(audio_path, hf_token, device, num, mn, mx) -> List[Turn]:
    pipeline = _load_whisperx_pipeline(hf_token, device)
    try:
        df = pipeline(audio_path, **_speaker_kwargs(num, mn, mx))
    except Exception as exc:
        raise DiarizeError(f"whisperx diarization failed: {exc}") from exc
    return _dataframe_to_turns(df)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def diarize_and_merge(
    audio_path: str,
    segments: List[Dict[str, Any]],
    *,
    backend: str = "pyannote",
    hf_token: Optional[str] = None,
    device: str = "cpu",
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    diarize_model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Diarize ``audio_path`` and return speaker-tagged ``segments``.

    ``diarize_model`` overrides the pipeline repo (default: auto by pyannote
    version — ``speaker-diarization-community-1`` on 4.x, ``…-3.1`` on 3.x).
    """
    if backend == "pyannote":
        turns = _pyannote_turns(audio_path, hf_token, device,
                                num_speakers, min_speakers, max_speakers,
                                model=diarize_model)
    elif backend == "whisperx":
        turns = _whisperx_turns(audio_path, hf_token, device,
                                num_speakers, min_speakers, max_speakers)
    else:
        raise DiarizeError(
            f"unknown diarization backend: {backend!r} "
            f"(choose from {', '.join(BACKENDS)})"
        )
    return merge_turns(segments, turns)
