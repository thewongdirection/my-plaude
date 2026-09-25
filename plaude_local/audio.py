"""Audio loading, denoising, and enhancement.

We lean on the FFmpeg *binary* (invoked via ``subprocess``) rather than pulling
in Python audio libraries. This means the tool accepts **any input that FFmpeg
can decode** - every audio codec/container FFmpeg supports, and the audio track
of video files too - with no per-format handling on our side. FFmpeg also
provides capable denoise and enhancement filter chains, keeping the Python
dependency surface small.

The preprocessing runs in up to three stages (repair, denoise, then enhance):

Repair (optional, off by default) - fixes damage from bad hardware, first:
* ``declip``  - FFmpeg ``adeclip``: rebuilds peaks flattened by a too-hot input.
* ``declick`` - FFmpeg ``adeclick``: removes clicks, pops and crackle/static bursts.
* ``dehum``   - notch filters on mains hum (50 or 60 Hz) and its harmonics;
                ``auto`` measures which (if either) is present.

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
from typing import Callable, List, Optional

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

# Repair stage (runs before denoise, on the decoded input).
DEHUM_MODES = ("none", "auto", "50", "60")
_HUM_HARMONICS = 8        # notch the fundamental + 7 harmonics (up to 400/480 Hz)
_HUM_NOTCH_Q = 30         # narrow notches (~2 Hz wide at 60 Hz) spare the voice
_HUM_MARGIN_DB = 6.0      # 50/60 Hz band must beat the 55 Hz reference by this much
_HUM_PROBE_SECONDS = 300  # auto-detect looks at the first 5 minutes only


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


def _to_wav(src: Path, dst: Path, *, filters: Optional[str] = None,
            native_float: bool = False) -> None:
    """Decode ``src`` to 16 kHz mono PCM WAV at ``dst``, optionally filtering.

    ``native_float`` keeps the source sample rate and writes 32-bit float
    instead: used for the repair intermediate, so peaks rebuilt by ``adeclip``
    above full scale are not re-clipped by 16-bit PCM, and DeepFilterNet still
    gets full-band audio.
    """
    args = ["-i", str(src)]
    if filters:
        args += ["-af", filters]
    if native_float:
        args += ["-ac", "1", "-c:a", "pcm_f32le", str(dst)]
    else:
        args += ["-ac", "1", "-ar", str(TARGET_SR), "-c:a", "pcm_s16le", str(dst)]
    _run_ffmpeg(args)


def _fmt_db(db: float) -> str:
    """Format a dB value compactly (6.0 -> '6', -3.5 -> '-3.5'), matching the
    PowerShell port's Format-Db so filter strings and manifests are identical."""
    return f"{db:g}"


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
        parts.append(f"volume={_fmt_db(gain_db)}dB")
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


def dehum_filters(hum_hz: int) -> str:
    """Narrow notch filters at ``hum_hz`` and its harmonics."""
    return ",".join(f"bandreject=f={hum_hz * k}:width_type=q:w={_HUM_NOTCH_Q}"
                    for k in range(1, _HUM_HARMONICS + 1))


def repair_filters(*, declip: bool = False, declick: bool = False,
                   hum_hz: Optional[int] = None) -> Optional[str]:
    """Build the repair ``-af`` chain (declip -> declick -> dehum), or None."""
    parts: List[str] = []
    if declip:
        parts.append("adeclip")
    if declick:
        parts.append("adeclick")
    if hum_hz:
        parts.append(dehum_filters(hum_hz))
    return ",".join(parts) if parts else None


def pick_hum_hz(levels: dict, margin_db: float = _HUM_MARGIN_DB) -> Optional[int]:
    """Choose 50 or 60 Hz from narrow-band mean levels ``{50: dB, 55: dB, 60: dB}``.

    Returns the louder of 50/60 only if it stands at least ``margin_db`` above
    the 55 Hz reference band (where neither hum nor voice sits); else None.
    """
    ref = levels.get(55)
    cands = [(levels[f], f) for f in (50, 60) if levels.get(f) is not None]
    if ref is None or not cands:
        return None
    level, hz = max(cands)
    return hz if level - ref >= margin_db else None


def _band_level(path: str | Path, hz: int) -> Optional[float]:
    """Mean level (dB) of a narrow band around ``hz`` over the first minutes."""
    import re
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-t", str(_HUM_PROBE_SECONDS),
           "-i", str(path),
           "-af", f"bandpass=f={hz}:width_type=q:w=20,volumedetect",
           "-f", "null", "-"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    except OSError as exc:
        raise AudioError(f"could not run ffmpeg for hum detection: {exc}") from exc
    if result.returncode != 0:
        # Don't let a failed measurement masquerade as "no hum detected".
        raise AudioError(f"hum detection failed (ffmpeg exit {result.returncode}):\n"
                         f"{(result.stderr or '').strip()}")
    m = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?) dB", result.stderr or "")
    return float(m.group(1)) if m else None


def detect_hum_hz(path: str | Path) -> Optional[int]:
    """Detect mains hum: 50, 60, or None when neither is clearly present."""
    return pick_hum_hz({hz: _band_level(path, hz) for hz in (50, 55, 60)})


def resolve_dehum(dehum: str, path: str | Path) -> Optional[int]:
    """Turn a ``dehum`` mode into the hum frequency to notch (or None)."""
    if dehum not in DEHUM_MODES:
        raise AudioError(f"unknown dehum mode: {dehum!r}")
    if dehum == "none":
        return None
    if dehum == "auto":
        return detect_hum_hz(path)
    return int(dehum)


def _save_stage(wav: Path, stages_dir: Path, stem: str, label: str) -> str:
    """Copy an intermediate ``wav`` into ``stages_dir`` as ``<stem>.<label>.wav``."""
    name = f"{stem}.{label}.wav"
    try:
        shutil.copyfile(wav, stages_dir / name)
    except OSError as exc:
        raise AudioError(f"could not save audio stage to {stages_dir}: {exc}") from exc
    return name


def _write_stages_manifest(stages_dir: Path, stem: str, manifest: dict) -> None:
    """Record which settings produced the saved stages (``<stem>.stages.json``)."""
    import json
    try:
        (stages_dir / f"{stem}.stages.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        raise AudioError(f"could not save audio stage to {stages_dir}: {exc}") from exc


def prepare(
    input_path: str | Path,
    workdir: str | Path,
    *,
    denoise: str = "ffmpeg",
    enhance: str = "none",
    gain_db: float = 0.0,
    declip: bool = False,
    declick: bool = False,
    dehum: str = "none",
    stages_dir: str | Path | None = None,
    on_hum: Optional[Callable[[Optional[int]], None]] = None,
) -> Path:
    """Produce a 16 kHz mono WAV ready for transcription.

    Runs (optional) repair, denoise, then (optional) enhancement. Returns the
    path to the prepared WAV inside ``workdir``.

    * ``denoise``: ``"ffmpeg"``, ``"deepfilter"`` or ``"none"``.
    * ``enhance``: ``"none"``, ``"speech"``, ``"strong"`` or ``"resemble"``.
    * ``gain_db``: manual volume adjustment in dB (0 = none).
    * ``declip`` / ``declick``: repair clipped peaks / clicks and crackle.
    * ``dehum``: ``"none"``, ``"auto"`` (detect 50/60 Hz), ``"50"`` or ``"60"``.
    * ``on_hum``: called with the detected hum frequency (or None) after
      ``dehum="auto"`` runs its detection - lets the CLI report the result.
    * ``stages_dir``: if set, also save each intermediate as a 16 kHz mono WAV
      for later inspection - ``<stem>.01-original.wav`` (unprocessed decode),
      ``<stem>.02-repaired.wav`` (only when repairing; source sample rate,
      32-bit float so rebuilt peaks survive),
      ``<stem>.03-denoised.wav`` (unless ``denoise="none"``),
      ``<stem>.04-enhanced.wav`` (only when enhancing / gain) - plus a
      ``<stem>.stages.json`` manifest of the settings that produced them.
    """
    if enhance not in ENHANCE_MODES:
        raise AudioError(f"unknown enhance mode: {enhance!r}")

    src = Path(input_path)
    if not src.is_file():
        raise AudioError(f"input file not found: {src}")

    if denoise not in ("none", "ffmpeg", "deepfilter"):
        raise AudioError(f"unknown denoise mode: {denoise!r}")

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    final = workdir / (src.stem + ".prepared.wav")

    # 1. Repair (declip -> declick -> dehum) into an intermediate that the
    #    denoise stage then reads instead of the raw input.
    hum_hz = resolve_dehum(dehum, src)
    if dehum == "auto" and on_hum is not None:
        on_hum(hum_hz)
    rep_filters = repair_filters(declip=declip, declick=declick, hum_hz=hum_hz)
    repaired: Optional[Path] = None
    if rep_filters:
        repaired = workdir / (src.stem + ".repaired.wav")
        _to_wav(src, repaired, filters=rep_filters, native_float=True)
    denoise_src = repaired or src

    # 2. Denoise. If we'll enhance, into an intermediate; otherwise to final.
    need_enhance = enhance != "none" or bool(gain_db)
    denoise_dst = (workdir / (src.stem + ".denoised.wav")) if need_enhance else final

    if denoise == "none":
        _to_wav(denoise_src, denoise_dst, filters=None)
    elif denoise == "ffmpeg":
        _to_wav(denoise_src, denoise_dst, filters=_FFMPEG_DENOISE_CHAIN)
    else:
        _denoise_deepfilter(denoise_src, denoise_dst)

    if need_enhance:
        if enhance == "resemble":
            _enhance_resemble(denoise_dst, final)
            if gain_db:  # apply the manual gain as a follow-up pass
                gained = workdir / (src.stem + ".gained.wav")
                _to_wav(final, gained, filters=f"volume={_fmt_db(gain_db)}dB")
                gained.replace(final)
        else:
            _to_wav(denoise_dst, final, filters=_enhance_filters(enhance, gain_db))

    if stages_dir is not None:
        _save_stages(src, workdir, Path(stages_dir), denoise=denoise,
                     enhance=enhance, gain_db=gain_db, declip=declip,
                     declick=declick, dehum=dehum, hum_hz=hum_hz,
                     repair_chain=rep_filters, repaired=repaired,
                     denoised=denoise_dst, enhanced=final if need_enhance else None)
    return final


def _save_stages(src: Path, workdir: Path, stages_dir: Path, *, denoise: str,
                 enhance: str, gain_db: float, declip: bool, declick: bool,
                 dehum: str, hum_hz: Optional[int], repair_chain: Optional[str],
                 repaired: Optional[Path], denoised: Path,
                 enhanced: Optional[Path]) -> None:
    """Save original / repaired / denoised / enhanced WAVs + manifest."""
    try:
        stages_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AudioError(f"could not create stages folder {stages_dir}: {exc}") from exc
    stem = src.stem
    files = {}
    if denoise == "none" and repaired is None:
        # The "denoise" pass was a plain decode: it *is* the original.
        files["original"] = _save_stage(denoised, stages_dir, stem, "01-original")
    else:
        original = workdir / (stem + ".original.wav")
        _to_wav(src, original, filters=None)
        files["original"] = _save_stage(original, stages_dir, stem, "01-original")
    if repaired is not None:
        files["repaired"] = _save_stage(repaired, stages_dir, stem, "02-repaired")
    if denoise != "none":
        files["denoised"] = _save_stage(denoised, stages_dir, stem, "03-denoised")
    if enhanced is not None:
        files["enhanced"] = _save_stage(enhanced, stages_dir, stem, "04-enhanced")
    denoise_filters = {"ffmpeg": _FFMPEG_DENOISE_CHAIN,
                       "deepfilter": "DeepFilterNet"}.get(denoise)
    enhance_filters = ("Resemble-Enhance" + (f" + volume={_fmt_db(gain_db)}dB" if gain_db else "")
                       if enhance == "resemble" else _enhance_filters(enhance, gain_db))
    _write_stages_manifest(stages_dir, stem, {
        "input": str(src),
        "sample_rate": TARGET_SR,
        "declip": declip,
        "declick": declick,
        "dehum": dehum,
        "hum_hz": hum_hz,
        "repair_filters": repair_chain,
        "denoise": denoise,
        "denoise_filters": denoise_filters,
        "enhance": enhance,
        # Integral gains as ints (6, not 6.0) - same JSON as the PowerShell port.
        "gain_db": int(gain_db) if float(gain_db).is_integer() else gain_db,
        "enhance_filters": enhance_filters,
        "files": files,
    })
