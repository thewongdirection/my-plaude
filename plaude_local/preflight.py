"""Prerequisite checks with actionable remedies.

Powers ``plaude-local --check`` (a "doctor" command) and the targeted checks the
CLI runs before a feature that needs an optional dependency. Every failing check
explains *what* is missing and *how* to get it.

Module availability is probed with :func:`importlib.util.find_spec` (cheap, and
does not import optional heavy libraries such as pyannote/whisperx). Binaries are
probed with :func:`shutil.which`. The one exception is the GPU check, which
imports ``ctranslate2`` (a core dependency) to count CUDA devices - guarded so it
never fails when ctranslate2 is not yet installed.
"""

from __future__ import annotations

import importlib.util
import platform
import shutil
import sys
from dataclasses import dataclass
from typing import List, Optional

MIN_PYTHON = (3, 9)

_IS_WINDOWS = platform.system() == "Windows"


@dataclass
class Check:
    name: str
    ok: bool
    required: bool
    detail: str = ""
    remedy: str = ""

    def symbol(self) -> str:
        if self.ok:
            return "OK  "
        return "FAIL" if self.required else "WARN"


def _ffmpeg_remedy() -> str:
    if _IS_WINDOWS:
        return ("install FFmpeg: `winget install ffmpeg` (or Chocolatey "
                "`choco install ffmpeg`, or download from "
                "https://www.gyan.dev/ffmpeg/builds/ and add it to PATH)")
    return ("install FFmpeg: Debian/Ubuntu `sudo apt install ffmpeg`, "
            "Fedora `sudo dnf install ffmpeg`, macOS `brew install ffmpeg`")


def check_python() -> Check:
    ok = sys.version_info[:2] >= MIN_PYTHON
    ver = ".".join(map(str, sys.version_info[:3]))
    return Check(
        name="Python >= 3.9",
        ok=ok,
        required=True,
        detail=f"found {ver}",
        remedy="install Python 3.9+ from https://www.python.org/downloads/",
    )


def check_ffmpeg() -> Check:
    path = shutil.which("ffmpeg")
    return Check(
        name="FFmpeg (audio decode/denoise)",
        ok=path is not None,
        required=True,
        detail=f"found at {path}" if path else "not found on PATH",
        remedy=_ffmpeg_remedy(),
    )


def _has_module(module: str) -> bool:
    """True if ``module`` is importable, without importing it.

    ``find_spec`` raises ``ModuleNotFoundError`` when a *parent* package (e.g.
    ``pyannote`` for ``pyannote.audio``) is absent, so guard against that.
    """
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _module_check(module: str, name: str, required: bool, pip: str,
                  extra_remedy: str = "") -> Check:
    present = _has_module(module)
    remedy = f"install with `{pip}`"
    if extra_remedy:
        remedy += f". {extra_remedy}"
    return Check(
        name=name,
        ok=present,
        required=required,
        detail="installed" if present else "not installed",
        remedy=remedy,
    )


def check_faster_whisper() -> Check:
    return _module_check(
        "faster_whisper", "faster-whisper (ASR engine)", True,
        "pip install faster-whisper",
    )


def check_gpu() -> Check:
    """Informational: report CUDA GPU availability (never fails)."""
    detail = "no CUDA GPU detected - will run on CPU"
    ok_gpu = False
    if _has_module("ctranslate2"):
        try:
            import ctranslate2

            n = ctranslate2.get_cuda_device_count()
            if n > 0:
                ok_gpu = True
                detail = f"{n} CUDA device(s) available - GPU acceleration on"
        except Exception:
            detail = "could not query CUDA (CTranslate2 present)"
    else:
        detail = "faster-whisper/ctranslate2 not installed yet"
    return Check(
        name="CUDA GPU (optional, faster)",
        ok=True,  # informational only
        required=False,
        detail=detail,
        remedy="" if ok_gpu else
               "for GPU: install NVIDIA driver + CUDA/cuDNN runtime; see "
               "https://github.com/SYSTRAN/faster-whisper#gpu (CPU works without it)",
    )


def check_deepfilternet() -> Check:
    return _module_check(
        "df", "DeepFilterNet (--denoise deepfilter)", False,
        "pip install deepfilternet",
    )


def check_pyannote() -> Check:
    return _module_check(
        "pyannote.audio", "pyannote.audio (--diarize, pyannote backend)", False,
        "pip install \"pyannote.audio>=3.1\"",
        "then accept terms at https://hf.co/pyannote/speaker-diarization-3.1 "
        "and set HF_TOKEN (create one at https://hf.co/settings/tokens)",
    )


def check_whisperx() -> Check:
    return _module_check(
        "whisperx", "whisperx (--diarize, whisperx backend)", False,
        "pip install whisperx",
        "also needs an HF token as above",
    )


def check_summarizer(ollama_url: Optional[str] = None,
                     llamacpp_url: Optional[str] = None) -> Check:
    """Check whether a local LLM server is reachable for --summarize."""
    from . import summarize as _sum

    o_url = ollama_url or _sum.DEFAULT_OLLAMA_URL
    l_url = llamacpp_url or _sum.DEFAULT_LLAMACPP_URL
    backend = _sum.detect_backend(o_url, l_url)
    return Check(
        name="Local LLM server (--summarize)",
        ok=backend is not None,
        required=False,
        detail=f"reachable via {backend}" if backend else "no server reachable",
        remedy="" if backend else
               "start Ollama (https://ollama.com/download, then `ollama pull "
               "llama3.1` && `ollama serve`) or a llama.cpp server "
               "(https://github.com/ggml-org/llama.cpp, `llama-server -m model.gguf`)",
    )


def run_all() -> List[Check]:
    """Run the full diagnostic set for ``--check``."""
    return [
        check_python(),
        check_ffmpeg(),
        check_faster_whisper(),
        check_gpu(),
        check_deepfilternet(),
        check_pyannote(),
        check_whisperx(),
        check_summarizer(),
    ]


def format_report(checks: List[Check]) -> str:
    lines = ["plaude-local environment check", "=" * 32, ""]
    width = max(len(c.name) for c in checks)
    for c in checks:
        lines.append(f"[{c.symbol()}] {c.name.ljust(width)}  {c.detail}")
        if not c.ok and c.remedy:
            lines.append(f"        -> {c.remedy}")
    missing_required = [c for c in checks if c.required and not c.ok]
    lines.append("")
    if missing_required:
        lines.append("Result: MISSING required prerequisites (see FAIL lines above).")
    else:
        lines.append("Result: all required prerequisites satisfied.")
    return "\n".join(lines) + "\n"


def missing_required(checks: List[Check]) -> List[Check]:
    return [c for c in checks if c.required and not c.ok]
