# plaude-local — Capabilities & Operating Summary

> **Living document.** Keep this in sync with the code as features land. When you
> add/change a flag, backend, output, or behavior, update the relevant row here
> in the same change (alongside `README.md` and `powershell/README.md`).
>
> _Last updated: 2026-07-24._

A **local, offline speech-to-text application** in the spirit of Plaud — built to
run entirely on your own machine and local-LLM infrastructure. Two implementations
kept at **feature parity**:

- **Python** (reference): `plaude-local` / `python -m plaude_local`
- **PowerShell** (Windows-first port): `powershell/Plaude-Local.ps1`

## Pipeline

```
audio/video (any FFmpeg-decodable format)
  → prepare   (denoise / enhance / gain)
  → transcribe(faster-whisper / CTranslate2)
  → translate (accuracy-first: local LLM→any language, or Whisper→English)
  → diarize   (optional: pyannote / whisperx)
  → summarize (local LLM: Ollama / llama.cpp)
  → HTML dashboard (default)  |  txt / srt / vtt / json
```

## Core capabilities

| Capability | Details |
|---|---|
| **Input** | Anything FFmpeg decodes (mp3, m4a, wav, flac, ogg/opus, wma, mp4/mkv…). ffprobe-gated; file extension is irrelevant. |
| **Transcription** | faster-whisper / CTranslate2 (PS: `whisper-ctranslate2`). Models `tiny`…`large-v3` (default **large-v3**). Auto device/compute resolve; `--language`, `--beam-size`, VAD silence-trim, `--model-dir`. |
| **HTML dashboard** (default output) | Self-contained, theme-aware page: header stats (**detected language, speech duration, word count**), a **provenance line** (transcription engine + translation engine/model actually used), a **≤250-word critical-topics summary**, and three tabs — **Transcribed**, **Translated-\<Language\>**, and a **segment-aligned Side-by-Side** (original left / translation right, lined up by timing). |
| **Translation** | `--translate-to` (default `en`), `--translate-engine {auto,whisper,llm}` (default `auto`), `--translate-model` (separate from the summary model). **Accuracy-first:** `auto` picks the **local LLM** whenever a server is reachable (line-aligned, one output per segment so Side-by-Side stays aligned), else falls back to Whisper's native translate task. `--translate-model` / summary model default to the **first model the local Ollama/llama.cpp server reports** when unset. Short-circuits when target == source. |
| **Summarization** | Local LLM (**Ollama** / **llama.cpp** over localhost), map-reduce for long transcripts, language-preserving. `--summarize` → `.summary.md` for non-HTML formats. `--summarize-timeout` raises the per-request LLM timeout for large reasoning models (e.g. deepseek-r1). |
| **Diarization** | `--diarize` via **pyannote** or **whisperx**. Version-aware model pick (pyannote 4.x → `speaker-diarization-community-1`, 3.x → `speaker-diarization-3.1`); `--diarize-model`, `--num/min/max-speakers`. |
| **Audio cleanup** | Denoise (`ffmpeg` default / `deepfilter` / `none`), enhance (`speech`/`strong`/`resemble`), `--gain DB`, `--keep-clean`. |
| **Quality triage** | Bad-recording detection (silence ratio, speech coverage, no-speech prob, compression ratio, avg logprob). `--assess-only` (fast, no model), `--on-bad warn/skip/fail`. |
| **Output formats** | `html` (default), `txt`, `srt`, `vtt`, `json`. `--split-outputs` also writes `transcription.txt` / `translation.txt` (paths overridable via `--transcription-file` / `--translation-file`). |

## Operating characteristics

- **Local-first / private.** Audio never leaves the machine; summary/translation use a local LLM on localhost. **Hugging Face telemetry disabled by default** (`HF_HUB_DISABLE_TELEMETRY` / `DISABLE_TELEMETRY`).
- **Air-gappable.** The only network touch is a one-time model download. `--offline` (`HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE`) forces cache-only runs; `tools/prepare_offline_bundle.py` (+ PS `Prepare-OfflineBundle.ps1`) pre-downloads FFmpeg, wheels, CUDA DLLs, and models, then zips them for offline install.
- **GPU out-of-the-box on Windows.** Auto-registers the `nvidia-*-cu12` wheel DLL directories (`os.add_dll_directory` / a PS bootstrap), so CUDA loads even though Python 3.8+ won't search `PATH` for a native extension's DLLs.
- **Always UTF-8** — CJK and any script round-trip; CJK-aware word counting and typography.
- **Cross-platform** — Windows + Linux; lean core deps, optional backends imported lazily.
- **FFmpeg auto-provisioning** — offers to install FFmpeg if missing (winget/apt/brew/download); `--ffmpeg-location`, `--install-missing`, `--no-provision`.
- **Stable exit-code contract** — 2 input · 3 ffmpeg · 4 diarize-token · 5 audio · 6 transcribe · 7 diarize · 8 summarize · 9 write · 10 no-audio-stream · 11 overwrite-declined · 12 bad-recording.
- **Overwrite guard** — prompt with 10 s auto-overwrite (`--yes` to skip); `--check` / `--doctor` preflight with remedies.
- **Tested & mirrored** — offline suites (Python unittest, PowerShell Pester); a hard parity rule keeps both implementations in lockstep.

## Prerequisites

**Required**
- Python 3.9+
- FFmpeg + ffprobe (or auto-provisioned)
- faster-whisper (Python) / whisper-ctranslate2 (PowerShell)

**Recommended**
- NVIDIA GPU + CUDA — via `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` pip wheels or a system CUDA install

**Optional (per feature)**
- Ollama or llama.cpp server → summarization & non-English translation
- pyannote.audio (+ free HF token & accepted model terms) or whisperx → diarization
- deepfilternet → `--denoise deepfilter`; resemble-enhance → `--enhance resemble`

## Known limitations / caveats

- **Diarization on Windows + Python 3.13 is blocked** by pyannote deps (4.x needs `k2` — no Windows wheels; 3.x needs an older `torchaudio` with no 3.13 wheels). Fix: a **Python 3.10/3.11 venv** with `torch/torchaudio 2.2` + `pyannote.audio<4`, or **WSL2/Linux**.
- **Translation accuracy vs. speed** — the `auto` engine prefers the local LLM for quality; on machines with no LLM server it falls back to Whisper's (English-only) translate task. Force either with `--translate-engine`.
- **Models originate from Hugging Face** (one-time download); the offline bundle mitigates this for locked-down setups.
