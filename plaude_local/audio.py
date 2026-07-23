"""Audio loading and denoising.

We lean on the FFmpeg *binary* (invoked via ``subprocess``) rather than pulling
in Python audio libraries. This means the tool accepts **any input that FFmpeg
can decode** - every audio codec/container FFmpeg supports, and the audio track
of video files too - with no per-format handling on our side. FFmpeg also
provides a capable denoise filter chain, keeping the Python dependency surface
small.

Two denoise strategies are offered:

* ``ffmpeg``     - a cheap DSP filter chain (default). No extra Python deps.
* ``deepfilter`` - DeepFilterNet, a small neural denoiser (optional extra),
                   noticeably better on hard/noisy recordings.
* ``none``       - skip denoising entirely.

Whisper wants 16 kHz mono audio, so every path here normalizes to that.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

TARGET_SR = 16000  # Whisper's expected sample rate

# A conservative speech-friendly filter chain:
#   highpass   - drop rumble / handling noise below 90 Hz
#   lowpass    - drop hiss above 7.5 kHz (speech energy sits below this)
#   afftdn     - FFmpeg's adaptive FFT denoiser
#   dynaudnorm - gentle single-pass loudness normalization
_FFMPEG_DENOISE_CHAIN = "highpass=f=90,lowpass=f=7500,afftdn=nf=-25,dynaudnorm"


class AudioError(RuntimeError):
    """Raised when audio decoding or denoising fails."""


def have_ffmpeg() -> bool:
    """True if an ``ffmpeg`` binary is on PATH."""
    return shutil.which("ffmpeg") is not None


def have_ffprobe() -> bool:
    """True if an ``ffprobe`` binary is on PATH (ships alongside ffmpeg)."""
    return shutil.which("ffprobe") is not None


def probe_audio_codec(path: str | Path) -> Optional[str]:
    """Return the codec of the first decodable audio stream, or None.

    Uses ``ffprobe`` to ask FFmpeg directly whether the file contains an audio
    stream it understands - so support tracks exactly what FFmpeg can decode,
    independent of the file's extension. Returns None when there is no audio
    stream (or ``ffprobe`` is unavailable / errors), letting the caller decide.
    """
    if not have_ffprobe():
        return None
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_name",
        "-of", "default=nokey=1:noprint_wrappers=1",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                errors="replace")
    except FileNotFoundError:  # pragma: no cover - race: ffprobe vanished
        return None
    codec = result.stdout.strip()
    return codec or None


def _run_ffmpeg(args: list[str]) -> None:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, errors="replace")
    except FileNotFoundError as exc:  # pragma: no cover - environment dependent
        raise AudioError(
            "ffmpeg was not found on PATH. Install it and try again "
            "(Windows: https://www.gyan.dev/ffmpeg/builds/ or `winget install ffmpeg`; "
            "Linux: `apt install ffmpeg` / `dnf install ffmpeg`)."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise AudioError(f"ffmpeg failed:\n{exc.stderr.strip()}") from exc


def _to_wav(src: Path, dst: Path, *, filters: Optional[str] = None) -> None:
    """Decode ``src`` to 16 kHz mono PCM WAV at ``dst``, optionally filtering."""
    args = ["-i", str(src)]
    if filters:
        args += ["-af", filters]
    args += ["-ac", "1", "-ar", str(TARGET_SR), "-c:a", "pcm_s16le", str(dst)]
    _run_ffmpeg(args)


def _denoise_deepfilter(src: Path, dst: Path) -> None:
    """Neural denoise via DeepFilterNet (optional dependency, lazily imported)."""
    try:
        import torch  # noqa: F401
        from df.enhance import init_df, enhance, load_audio, save_audio
    except ImportError as exc:
        raise AudioError(
            "DeepFilterNet is not installed. Install the optional extra with "
            "`pip install deepfilternet` (also pulls in torch), or use "
            "--denoise ffmpeg / --denoise none."
        ) from exc

    # DeepFilterNet operates at 48 kHz; feed it a clean 48 kHz mono decode first.
    tmp48 = dst.with_name(dst.stem + ".df48.wav")
    _to_wav(src, tmp48, filters=None)
    try:
        model, df_state, _ = init_df()
        audio, _ = load_audio(str(tmp48), sr=df_state.sr())
        enhanced = enhance(model, df_state, audio)
        save_audio(str(tmp48), enhanced, df_state.sr())
        # Resample the enhanced result down to Whisper's 16 kHz mono.
        _to_wav(tmp48, dst, filters=None)
    finally:
        tmp48.unlink(missing_ok=True)


def prepare(
    input_path: str | Path,
    workdir: str | Path,
    *,
    denoise: str = "ffmpeg",
) -> Path:
    """Produce a 16 kHz mono WAV ready for transcription.

    Returns the path to the prepared WAV inside ``workdir``.

    ``denoise`` is one of ``"ffmpeg"``, ``"deepfilter"`` or ``"none"``.
    """
    src = Path(input_path)
    if not src.is_file():
        raise AudioError(f"input file not found: {src}")

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    out = workdir / (src.stem + ".prepared.wav")

    if denoise == "none":
        _to_wav(src, out, filters=None)
    elif denoise == "ffmpeg":
        _to_wav(src, out, filters=_FFMPEG_DENOISE_CHAIN)
    elif denoise == "deepfilter":
        _denoise_deepfilter(src, out)
    else:
        raise AudioError(f"unknown denoise mode: {denoise!r}")

    return out
