"""Audio loading, denoising, and enhancement.

We lean on the FFmpeg *binary* (invoked via ``subprocess``) rather than pulling
in Python audio libraries. This means the tool accepts **any input that FFmpeg
can decode** - every audio codec/container FFmpeg supports, and the audio track
of video files too - with no per-format handling on our side. FFmpeg also
provides capable denoise and enhancement filter chains, keeping the Python
dependency surface small.

The preprocessing runs in two stages (denoise, then enhance):

Denoise (``denoise=``):
* ``ffmpeg``     - a cheap DSP filter chain (default). No extra Python deps.
* ``deepfilter`` - DeepFilterNet, a small neural denoiser (optional extra),
                   noticeably better on hard/noisy recordings.
* ``none``       - skip denoising entirely.

Enhance (``enhance=``) - for soft or garbled voice, applied after denoise:
* ``none``     - no enhancement (default).
* ``speech``   - FFmpeg speech normalization + loudness (fixes soft volume).
* ``strong``   - FFmpeg compression + presence EQ + normalization (garbled /
                 muffled / uneven voice).
* ``resemble`` - Resemble-Enhance neural speech restoration (optional extra).
Plus ``gain_db`` for a manual volume boost/cut in decibels.

Whisper wants 16 kHz mono audio, so every path here normalizes to that.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

TARGET_SR = 16000  # Whisper's expected sample rate

# A conservative speech-friendly filter chain:
#   highpass   - drop rumble / handling noise below 90 Hz
#   lowpass    - drop hiss above 7.5 kHz (speech energy sits below this)
#   afftdn     - FFmpeg's adaptive FFT denoiser
#   dynaudnorm - gentle single-pass loudness normalization
_FFMPEG_DENOISE_CHAIN = "highpass=f=90,lowpass=f=7500,afftdn=nf=-25,dynaudnorm"

# Enhancement filter chains (applied after denoise) for soft/garbled voice.
#   speechnorm - lifts soft passages of speech without crushing loud ones
#   loudnorm   - EBU R128 loudness normalization to a broadcast-ish target
#   acompressor- evens out level swings (mumbled/uneven delivery)
#   equalizer  - a presence boost around 3 kHz for consonant intelligibility
_FFMPEG_ENHANCE_CHAINS = {
    "speech": "highpass=f=80,speechnorm=e=6.25:r=0.0005:l=1,"
              "loudnorm=I=-16:TP=-1.5:LRA=11",
    "strong": "highpass=f=80,"
              "acompressor=threshold=-18dB:ratio=3:attack=20:release=250,"
              "equalizer=f=3000:width_type=q:w=1.5:g=4,"
              "speechnorm=e=12.5:r=0.0005:l=1,"
              "loudnorm=I=-16:TP=-1.5:LRA=11",
}

ENHANCE_MODES = ("none", "speech", "strong", "resemble")


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


def probe_duration(path: str | Path) -> Optional[float]:
    """Return the media duration in seconds via ffprobe, or None."""
    if not have_ffprobe():
        return None
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nokey=1:noprint_wrappers=1", str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    except FileNotFoundError:  # pragma: no cover
        return None
    try:
        return float(result.stdout.strip())
    except (TypeError, ValueError):
        return None


def probe_levels(path: str | Path) -> dict:
    """Measure loudness and silence with FFmpeg for a fast, model-free triage.

    Returns a dict with ``mean_volume_db``, ``max_volume_db`` (both may be None)
    and ``silence_ratio`` (0..1, fraction of the file detected as silence).
    Requires only the ffmpeg binary; used by ``--assess-only``.
    """
    import re

    stats: dict = {"mean_volume_db": None, "max_volume_db": None,
                   "silence_ratio": None}
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
        "-af", "volumedetect,silencedetect=noise=-30dB:d=0.5",
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                errors="replace")
    except FileNotFoundError as exc:  # pragma: no cover
        raise AudioError("ffmpeg not found on PATH.") from exc
    err = result.stderr or ""

    m = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?) dB", err)
    if m:
        stats["mean_volume_db"] = float(m.group(1))
    m = re.search(r"max_volume:\s*(-?\d+(?:\.\d+)?) dB", err)
    if m:
        stats["max_volume_db"] = float(m.group(1))

    silence = sum(float(x) for x in re.findall(r"silence_duration:\s*(\d+(?:\.\d+)?)", err))
    duration = probe_duration(path)
    if duration and duration > 0:
        stats["silence_ratio"] = max(0.0, min(1.0, silence / duration))
    return stats


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


def _enhance_filters(enhance: str, gain_db: float) -> Optional[str]:
    """Build the FFmpeg ``-af`` string for an enhancement mode + optional gain.

    Returns None when there is nothing to do (``enhance='none'`` and no gain).
    The ``resemble`` mode is neural (handled separately); here it contributes
    only the optional gain.
    """
    parts: List[str] = []
    chain = _FFMPEG_ENHANCE_CHAINS.get(enhance)
    if chain:
        parts.append(chain)
    if gain_db:
        parts.append(f"volume={gain_db}dB")
    return ",".join(parts) if parts else None


def _enhance_resemble(src: Path, dst: Path) -> None:
    """Neural speech restoration via Resemble-Enhance (optional, lazy import)."""
    try:
        import torch
        import torchaudio
        from resemble_enhance.enhancer.inference import enhance as re_enhance
    except ImportError as exc:
        raise AudioError(
            "Resemble-Enhance is not installed. Install the optional extra with "
            "`pip install resemble-enhance` (also pulls in torch), or use "
            "--enhance speech / --enhance strong / --enhance none."
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dwav, sr = torchaudio.load(str(src))
    dwav = dwav.mean(dim=0)  # mix down to mono
    try:
        wav, new_sr = re_enhance(dwav, sr, device)
    except Exception as exc:
        raise AudioError(f"Resemble-Enhance failed: {exc}") from exc

    tmp = dst.with_name(dst.stem + ".re.wav")
    torchaudio.save(str(tmp), wav.unsqueeze(0).cpu(), new_sr)
    try:
        _to_wav(tmp, dst, filters=None)  # normalize to 16 kHz mono
    finally:
        tmp.unlink(missing_ok=True)


def prepare(
    input_path: str | Path,
    workdir: str | Path,
    *,
    denoise: str = "ffmpeg",
    enhance: str = "none",
    gain_db: float = 0.0,
) -> Path:
    """Produce a 16 kHz mono WAV ready for transcription.

    Runs denoise, then (optionally) enhancement. Returns the path to the
    prepared WAV inside ``workdir``.

    * ``denoise``: ``"ffmpeg"``, ``"deepfilter"`` or ``"none"``.
    * ``enhance``: ``"none"``, ``"speech"``, ``"strong"`` or ``"resemble"``.
    * ``gain_db``: manual volume adjustment in dB (0 = none).
    """
    if enhance not in ENHANCE_MODES:
        raise AudioError(f"unknown enhance mode: {enhance!r}")

    src = Path(input_path)
    if not src.is_file():
        raise AudioError(f"input file not found: {src}")

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    final = workdir / (src.stem + ".prepared.wav")

    # If we'll enhance, denoise into an intermediate; otherwise straight to final.
    need_enhance = enhance != "none" or bool(gain_db)
    denoise_dst = (workdir / (src.stem + ".denoised.wav")) if need_enhance else final

    if denoise == "none":
        _to_wav(src, denoise_dst, filters=None)
    elif denoise == "ffmpeg":
        _to_wav(src, denoise_dst, filters=_FFMPEG_DENOISE_CHAIN)
    elif denoise == "deepfilter":
        _denoise_deepfilter(src, denoise_dst)
    else:
        raise AudioError(f"unknown denoise mode: {denoise!r}")

    if not need_enhance:
        return final  # denoise_dst is final

    if enhance == "resemble":
        _enhance_resemble(denoise_dst, final)
        if gain_db:  # apply the manual gain as a follow-up pass
            gained = workdir / (src.stem + ".gained.wav")
            _to_wav(final, gained, filters=f"volume={gain_db}dB")
            gained.replace(final)
    else:
        _to_wav(denoise_dst, final, filters=_enhance_filters(enhance, gain_db))

    return final
