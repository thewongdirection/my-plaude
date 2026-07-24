"""Transcription via faster-whisper (CTranslate2).

This is the one unavoidable heavy dependency. It runs the same Whisper weights
as OpenAI Whisper but through CTranslate2, which is fast and supports int8
quantization, so ``large-v3`` fits comfortably on an 8 GB GPU and still runs
(more slowly) on CPU.
"""

from __future__ import annotations

import os
import sys
from typing import List, Dict, Any, Optional, Tuple


class TranscribeError(RuntimeError):
    """Raised when the ASR backend is unavailable or fails."""


def nvidia_dll_dirs() -> List[str]:
    """Return existing ``site-packages/nvidia/*/bin`` directories (de-duplicated).

    These hold the CUDA runtime DLLs shipped by the ``nvidia-*-cu12`` wheels
    (``cublas64_12.dll``, ``cudnn64_9.dll`` ...). Empty when the wheels are not
    installed (e.g. a system-wide CUDA toolkit is used instead).
    """
    import glob
    import importlib.util

    dirs: List[str] = []
    seen = set()
    try:
        spec = importlib.util.find_spec("nvidia")
        roots = list(spec.submodule_search_locations) if spec and spec.submodule_search_locations else []
    except (ImportError, AttributeError, ValueError):
        roots = []
    for root in roots:
        for bindir in glob.glob(os.path.join(root, "*", "bin")):
            key = os.path.normcase(os.path.abspath(bindir))
            if os.path.isdir(bindir) and key not in seen:
                seen.add(key)
                dirs.append(bindir)
    return dirs


def add_cuda_dll_directories() -> None:
    """Make pip-installed CUDA runtime wheels loadable on Windows.

    CTranslate2 loads ``cublas64_12.dll`` / ``cudnn64_9.dll`` at run time. Since
    Python 3.8, Windows no longer searches ``PATH`` for a native extension's DLL
    dependencies, so CUDA libraries provided by the ``nvidia-*-cu12`` wheels are
    not found and GPU inference fails with "Library cublas64_12.dll is not
    found". Registering those directories with ``os.add_dll_directory`` fixes it.
    No-op off Windows or when the wheels are absent.
    """
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return
    for bindir in nvidia_dll_dirs():
        try:
            os.add_dll_directory(bindir)
        except OSError:  # pragma: no cover - defensive
            pass


def _maybe_float(value: Any) -> Optional[float]:
    return float(value) if value is not None else None


def resolve_device(device: str) -> str:
    """Resolve ``"auto"`` to ``"cuda"`` when a CUDA device is present, else ``"cpu"``.

    Uses CTranslate2's own probe so we don't need to import torch just to detect
    a GPU.
    """
    if device != "auto":
        return device
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def resolve_compute_type(compute_type: str, device: str) -> str:
    """Pick a sensible compute type per device when set to ``"auto"``.

    * CUDA -> ``float16`` (fast, accurate, fits 8 GB for large-v3)
    * CPU  -> ``int8``    (the only practical choice for large models)
    """
    if compute_type != "auto":
        return compute_type
    return "float16" if device == "cuda" else "int8"


class Transcriber:
    """Thin wrapper around ``faster_whisper.WhisperModel``."""

    def __init__(
        self,
        model: str = "large-v3",
        device: str = "auto",
        compute_type: str = "auto",
        *,
        download_root: Optional[str] = None,
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise TranscribeError(
                "faster-whisper is not installed. Run `pip install faster-whisper` "
                "(or `pip install -r requirements.txt`)."
            ) from exc

        self.device = resolve_device(device)
        self.compute_type = resolve_compute_type(compute_type, self.device)
        self.model_name = model
        if self.device == "cuda":
            add_cuda_dll_directories()
        try:
            self._model = WhisperModel(
                model,
                device=self.device,
                compute_type=self.compute_type,
                download_root=download_root,
            )
        except Exception as exc:
            raise TranscribeError(
                f"failed to load model {model!r} on {self.device} "
                f"({self.compute_type}): {exc}"
            ) from exc

    def transcribe(
        self,
        audio_path: str,
        *,
        language: Optional[str] = None,
        word_timestamps: bool = False,
        vad_filter: bool = True,
        beam_size: int = 5,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Transcribe ``audio_path``.

        ``language=None`` triggers Whisper's automatic language detection.
        Returns ``(segments, info)`` where segments are plain dicts and info
        carries detected language / duration metadata.
        """
        segments_iter, info = self._model.transcribe(
            audio_path,
            language=language,
            task="transcribe",
            beam_size=beam_size,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps,
        )

        segments: List[Dict[str, Any]] = []
        for seg in segments_iter:  # iterating is what actually runs inference
            entry: Dict[str, Any] = {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text,
                "speaker": None,
                # Quality signals faster-whisper reports per segment; used by
                # quality.assess() to flag bad/empty/noisy recordings.
                "avg_logprob": _maybe_float(getattr(seg, "avg_logprob", None)),
                "no_speech_prob": _maybe_float(getattr(seg, "no_speech_prob", None)),
                "compression_ratio": _maybe_float(getattr(seg, "compression_ratio", None)),
            }
            if word_timestamps and seg.words:
                entry["words"] = [
                    {"start": float(w.start), "end": float(w.end), "word": w.word}
                    for w in seg.words
                ]
            segments.append(entry)

        meta = {
            "model": self.model_name,
            "device": self.device,
            "compute_type": self.compute_type,
            "language": getattr(info, "language", language),
            "language_probability": getattr(info, "language_probability", None),
            "duration": getattr(info, "duration", None),
        }
        return segments, meta
