# plaude-local

A small, **fully local / offline** speech-to-text tool in the spirit of Plaud.
Point it at a `.wav` or `.mp3` recording and get a transcript back — no cloud,
nothing leaves your machine. Runs on an **NVIDIA GPU (CUDA)** or **CPU**, on
both **Windows and Linux**.

Pipeline:

```
  audio (WAV/MP3)
        │
        ▼
  1. denoise / normalize        (FFmpeg, or optional DeepFilterNet)
        │
        ▼
  2. transcribe                 (faster-whisper — Whisper large-v3, ~99 languages)
        │
        ▼
  3. speaker diarization        (optional — pyannote: "who said what")
        │
        ▼
  transcript (txt / srt / vtt / json)
```

## Design goals

- **Local & private** — no network calls at inference time.
- **Cross-platform** — Windows and Linux, GPU or CPU.
- **Lean dependencies** — the CLI, audio I/O, and output writers use only the
  Python standard library plus the **FFmpeg binary**. The single required pip
  package is `faster-whisper`. Denoising-by-neural-net and speaker diarization
  are *optional* extras you install only if you want them.
- **Best multilingual accuracy** — uses Whisper `large-v3`, trained on ~99
  languages with automatic language detection.

## Requirements

| Component | Required? | Notes |
|-----------|-----------|-------|
| Python 3.9+ | yes | |
| FFmpeg (binary) | yes | reads WAV/MP3, does default denoise. `winget install ffmpeg` / `apt install ffmpeg` |
| `faster-whisper` | yes | the ASR engine |
| NVIDIA GPU (8 GB+ VRAM) | recommended | `large-v3` in float16 fits ~8 GB; CPU works but is slower |
| `deepfilternet` | optional | better neural denoise (`--denoise deepfilter`) |
| `pyannote.audio` + HF token | optional | speaker diarization (`--diarize`) |

For GPU use you also need NVIDIA's CUDA/cuDNN runtime available to CTranslate2 —
see the [faster-whisper GPU notes](https://github.com/SYSTRAN/faster-whisper#gpu).

## Install

```bash
# core only
pip install -r requirements.txt

# or install as a package with extras
pip install .            # core
pip install ".[denoise]" # + DeepFilterNet
pip install ".[diarize]" # + speaker diarization
pip install ".[all]"     # everything
```

Install FFmpeg separately (it is a binary, not a pip package):

- **Windows:** `winget install ffmpeg` (or download a build and add it to PATH)
- **Linux:** `sudo apt install ffmpeg` / `sudo dnf install ffmpeg`

## Usage

```bash
# simplest: auto-detect GPU/CPU, auto-detect language, plain-text output
plaude-local meeting.mp3

# or without installing the entry point:
python -m plaude_local meeting.mp3
```

Common options:

```bash
# choose a smaller/faster model, force CPU
plaude-local note.wav --model small --device cpu

# write SubRip subtitles
plaude-local talk.mp3 --format srt -o talk.srt

# specify language instead of auto-detecting
plaude-local entrevista.mp3 --language es

# stronger neural denoise (needs: pip install deepfilternet)
plaude-local noisy.wav --denoise deepfilter

# tag speakers (needs: pip install pyannote.audio + a Hugging Face token)
export HF_TOKEN=hf_xxx
plaude-local interview.mp3 --diarize --format txt
```

### Output formats

`--format` accepts `txt` (default), `srt`, `vtt`, `json`. With `--diarize`,
speaker labels (`Speaker 1`, `Speaker 2`, …) and timestamps are added:

```
[00:03] Speaker 1: So how did the demo go yesterday?
[00:07] Speaker 2: Pretty well — they liked the transcription accuracy.
```

### Key flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--model` | `large-v3` | Whisper model size or local path |
| `--device` | `auto` | `auto` picks CUDA if present, else CPU |
| `--compute-type` | `auto` | `float16` on GPU, `int8` on CPU |
| `--language` | auto | language code, or auto-detect |
| `--denoise` | `ffmpeg` | `ffmpeg`, `deepfilter`, or `none` |
| `--diarize` | off | speaker tagging (optional extra) |
| `--no-vad` | off | disable silence trimming |
| `--keep-clean PATH` | — | also save the denoised 16 kHz wav |

## How the pieces work

- **Denoising.** The default (`--denoise ffmpeg`) runs a speech-tuned FFmpeg
  filter chain (`highpass`, `lowpass`, `afftdn`, `dynaudnorm`) — no extra
  dependencies. `--denoise deepfilter` swaps in DeepFilterNet, a small neural
  denoiser that is noticeably better on hard/noisy audio and still runs in real
  time on CPU. Everything is resampled to 16 kHz mono for Whisper.
- **Transcription.** `faster-whisper` runs Whisper via CTranslate2 with int8/
  float16 quantization, so `large-v3` fits on an 8 GB GPU and also runs on CPU.
  Voice-activity detection trims silence for speed and accuracy.
- **Diarization.** `pyannote` segments the audio by voice and we align it to the
  transcript by timestamp. With word-level timestamps we regroup into clean
  per-speaker turns. It labels *distinct* speakers (`Speaker 1/2/…`), not names.

## Hardware notes

- **Recommended:** NVIDIA GPU with ≥ 8 GB VRAM → `large-v3` in `float16` is fast.
- **CPU-only:** works with `--device cpu` (auto-selected if no GPU); `large-v3`
  is usable but slower — try `--model medium` or `--model small` for speed.
- Diarization adds a modest extra load; comfortable on a 6–8 GB GPU, slower on
  CPU.

## License

MIT
