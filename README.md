# plaude-local

A small, **fully local / offline** speech-to-text tool in the spirit of Plaud.
Point it at a `.wav` or `.mp3` recording and get a transcript back — no cloud,
nothing leaves your machine. Runs on an **NVIDIA GPU (CUDA)** or **CPU**, on
both **Windows and Linux**.

```
  audio (WAV / MP3 / M4A / FLAC / AAC / OGG / ... — any codec FFmpeg reads)
        │
        ▼
  1. denoise / normalize        (FFmpeg, or optional DeepFilterNet)
        │
        ▼
  2. transcribe                 (faster-whisper — Whisper large-v3, ~99 languages)
        │
        ▼
  3. speaker diarization        (optional — pyannote or WhisperX: "who said what")
        │
        ▼
  4. summarize                  (optional — local LLM via Ollama or llama.cpp)
        │
        ▼
  transcript (txt / srt / vtt / json)  +  summary.md      ← always UTF-8
```

> **Windows users:** a feature-parity **PowerShell edition** lives in
> [`powershell/`](powershell/README.md) (`Plaude-Local.ps1`) if you prefer a
> native PowerShell tool over the Python CLI.

## Features

- **Local & private** — no network calls at inference time.
- **Cross-platform** — Windows and Linux, GPU or CPU (auto-detected).
- **Any format FFmpeg can decode** — there is no fixed format list. Decoding is
  delegated entirely to FFmpeg, and the tool uses `ffprobe` to confirm a
  decodable audio stream regardless of file extension. That covers WAV, MP3,
  **M4A**, AAC, FLAC, OGG/Opus, WMA, AIFF, ALAC, APE, WavPack, AMR, AC3, CAF,
  and anything else FFmpeg supports — including pulling the audio track out of
  video files (MP4, MKV, MOV, WebM, …). If FFmpeg can read it, so can this tool.
- **Best multilingual accuracy** — Whisper `large-v3`, ~99 languages, auto-detect.
- **Always UTF-8 output** — Chinese, Japanese, Korean, and any non-Latin script
  round-trip correctly, to files and to stdout.
- **Optional** neural denoising, **voice enhancement** (soft/garbled audio),
  speaker diarization (two backends), and local-LLM summarization.
- **Bad-recording detection** — flags empty/no-speech, near-silent, or noisy
  recordings (warn by default), with a fast `--assess-only` triage mode.
- **Lean core** — the CLI, audio I/O, output writers, and summarization
  transport are standard-library only; `faster-whisper` is the sole required
  pip package. Everything else is an opt-in extra.
- **Built-in prerequisite checker** — `plaude-local --check` tells you exactly
  what's missing and how to get it.

---

## Prerequisites — what you need and where to get it

| Component | Required? | For | Where to obtain |
|-----------|-----------|-----|-----------------|
| **Python 3.9+** | ✅ required | everything | https://www.python.org/downloads/ |
| **FFmpeg** (binary, incl. `ffprobe`) | ✅ required | decoding **any** input format, default denoise | Windows: `winget install ffmpeg` · Linux: `apt install ffmpeg` · https://ffmpeg.org/download.html |
| **faster-whisper** (pip) | ✅ required | transcription | `pip install faster-whisper` |
| **NVIDIA GPU + CUDA/cuDNN** | ⭐ recommended | speed (`large-v3` in float16) | driver from NVIDIA; see [faster-whisper GPU notes](https://github.com/SYSTRAN/faster-whisper#gpu). CPU works without it. |
| **deepfilternet** (pip) | ⚪ optional | better denoise (`--denoise deepfilter`) | `pip install deepfilternet` |
| **resemble-enhance** (pip) | ⚪ optional | neural voice restoration (`--enhance resemble`) | `pip install resemble-enhance` |
| **pyannote.audio** (pip) + HF token | ⚪ optional | diarization, pyannote backend | `pip install "pyannote.audio>=3.1"`; token at https://hf.co/settings/tokens; accept terms at https://hf.co/pyannote/speaker-diarization-3.1 |
| **whisperx** (pip) + HF token | ⚪ optional | diarization, WhisperX backend | `pip install whisperx` (+ HF token as above) |
| **Ollama** *or* **llama.cpp** server | ⚪ optional | summarization (`--summarize`) | Ollama: https://ollama.com/download · llama.cpp: https://github.com/ggml-org/llama.cpp |

> Run `plaude-local --check` at any time to see which of these are present and
> get copy-pasteable install commands for whatever is missing.

---

## Setting up your local machine

### 1. Get the code and a Python environment

```bash
git clone <your-fork-url> my-plaude
cd my-plaude

# create an isolated environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 2. Install the core

```bash
pip install -r requirements.txt      # installs faster-whisper
# or, to get the CLI entry point:
pip install .
```

### 3. Install FFmpeg (required, it's a binary not a pip package)

- **Windows:** `winget install ffmpeg` (or `choco install ffmpeg`, or download
  from https://www.gyan.dev/ffmpeg/builds/ and add the `bin` folder to PATH)
- **Linux:** `sudo apt install ffmpeg` (Debian/Ubuntu) or `sudo dnf install ffmpeg`
- **macOS:** `brew install ffmpeg`

Verify: `ffmpeg -version` should print a version.

### 4. (Recommended) GPU acceleration

If you have an NVIDIA GPU (≥ 8 GB VRAM recommended), install the NVIDIA driver
and the CUDA/cuDNN runtime that CTranslate2 uses. See the
[faster-whisper GPU notes](https://github.com/SYSTRAN/faster-whisper#gpu). The
tool auto-detects the GPU; without one it runs on CPU automatically.

The CUDA runtime can come from the pip wheels
(`pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`) instead of a system-wide
CUDA install. On Windows, plaude-local registers those wheels' DLL directories
automatically before loading CUDA, so GPU inference works out of the box —
Python 3.8+ otherwise excludes `PATH` from a native extension's DLL search and
CTranslate2 would fail with `Library cublas64_12.dll is not found`.

### 5. (Optional) Install the extras you want

```bash
pip install ".[denoise]"          # DeepFilterNet neural denoise
pip install ".[diarize]"          # pyannote diarization
pip install ".[diarize-whisperx]" # WhisperX diarization
pip install ".[dev]"              # pytest, for running the test suite
pip install ".[all]"              # everything
```

For **diarization** you also need a free Hugging Face token:

```bash
# 1. create a token: https://hf.co/settings/tokens
# 2. accept model terms: https://hf.co/pyannote/speaker-diarization-3.1
# 3. expose it:
export HF_TOKEN=hf_xxx            # Windows: set HF_TOKEN=hf_xxx
```

For **summarization** start a local LLM server (pick one):

```bash
# Ollama
#   install from https://ollama.com/download
ollama pull llama3.1
ollama serve            # usually already running as a service

# --- or llama.cpp ---
#   build/download from https://github.com/ggml-org/llama.cpp
llama-server -m your-model.gguf     # serves on http://localhost:8080
```

### 6. Verify everything

```bash
plaude-local --check
```

You'll get a report like:

```
[OK  ] Python >= 3.9                                 found 3.11.9
[OK  ] FFmpeg (audio decode/denoise)                 found at /usr/bin/ffmpeg
[OK  ] faster-whisper (ASR engine)                   installed
[OK  ] CUDA GPU (optional, faster)                   1 CUDA device(s) available
[WARN] pyannote.audio (--diarize, pyannote backend)  not installed
        -> install with `pip install "pyannote.audio>=3.1"`. then accept terms ...
Result: all required prerequisites satisfied.
```

---

## Try it yourself (quick test)

You don't need your own recording to smoke-test the setup — generate a short
spoken clip with FFmpeg's synth, or just use any audio/video file you have:

```bash
# 1. confirm prerequisites
plaude-local --check

# 2. transcribe a file (auto GPU/CPU, auto language, writes ./output.txt)
plaude-local note.mp3

# 3. see it on screen instead of a file
plaude-local note.mp3 -o -

# 4. run the automated test suite (no GPU or downloads needed)
python -m unittest discover -s tests -v
```

---

## Usage examples

```bash
# Simplest: transcribe, auto-detect device + language, write ./output.txt
plaude-local note.mp3

# Choose the output file name
plaude-local note.mp3 -o meeting-notes.txt

# Overwrite an existing output without the 10-second prompt
plaude-local note.mp3 -o out.txt --yes

# Without the installed entry point
python -m plaude_local note.mp3

# Any FFmpeg-decodable format works the same way
plaude-local voice-memo.m4a          # iPhone/Android voice memo (AAC in M4A)
plaude-local podcast.opus
plaude-local lecture.mp4             # audio is extracted from the video

# Pick a smaller/faster model and force CPU
plaude-local call.wav --model small --device cpu

# Subtitles
plaude-local talk.mp3 --format srt -o talk.srt
plaude-local talk.mp3 --format vtt -o talk.vtt

# JSON with timestamps + metadata
plaude-local talk.mp3 --format json -o talk.json

# Force a language (skip auto-detect) — e.g. Mandarin Chinese
plaude-local 会议.mp3 --language zh

# Stronger neural denoise (needs: pip install deepfilternet)
plaude-local noisy.wav --denoise deepfilter

# Enhance a too-soft recording (loudness normalization, no extra deps)
plaude-local quiet.mp3 --enhance speech

# Garbled / muffled voice: compression + presence EQ + normalization
plaude-local muffled.m4a --enhance strong

# Just make it louder by 8 dB
plaude-local faint.wav --gain 8

# Neural restoration for badly degraded audio (needs: pip install resemble-enhance)
plaude-local bad-recording.mp3 --enhance resemble

# Speaker diarization — pyannote backend (needs HF_TOKEN)
export HF_TOKEN=hf_xxx
plaude-local interview.mp3 --diarize

# Speaker diarization — WhisperX backend
plaude-local interview.mp3 --diarize --diarize-backend whisperx

# Tell diarization exactly how many speakers there are
plaude-local panel.mp3 --diarize --num-speakers 3

# Summarize the transcript with a local LLM (auto-detect Ollama/llama.cpp)
plaude-local meeting.mp3 --summarize

# Summarize with a specific Ollama model and print summary to screen
plaude-local meeting.mp3 --summarize --summarize-model llama3.1 --summary-output -

# The full treatment: denoise + diarize + summarize
export HF_TOKEN=hf_xxx
plaude-local meeting.mp3 --denoise deepfilter --diarize --summarize -f txt

# Keep the cleaned audio for inspection
plaude-local noisy.mp3 --denoise deepfilter --keep-clean cleaned.wav

# Flag bad recordings (warn by default; transcript still written)
plaude-local maybe-empty.mp3

# Fast triage of a file without transcribing (loudness / silence check)
plaude-local suspect.wav --assess-only

# In batch jobs: exit non-zero (12) on a bad recording instead of writing
plaude-local clip.mp3 --on-bad fail

# Check your environment
plaude-local --check
```

### Output formats

`--format` accepts `txt` (default), `srt`, `vtt`, `json`. With `--diarize`,
speaker labels (`Speaker 1`, `Speaker 2`, …) and timestamps are added:

```
[00:03] Speaker 1: So how did the demo go yesterday?
[00:07] Speaker 2: Pretty well — they liked the transcription accuracy.
```

All outputs are written as **UTF-8**, so non-Latin scripts are preserved:

```
你好，欢迎使用本地语音转文字工具。
```

Summaries (from `--summarize`) are written to `<transcript>.summary.md` by
default (or to stdout with `--summary-output -`).

### Output file and overwriting

If you don't pass `-o/--output`, the transcript is written to **`output.<format>`**
in the current directory (e.g. `output.txt`). If that file already exists, you're
asked whether to overwrite; with no answer within **10 seconds it overwrites by
default**. Pass `--yes` (`-y`) to skip the prompt, or `-o -` to write to stdout.
In a non-interactive session (piped/redirected) it overwrites without waiting.

### Key flags

| Flag | Default | Meaning |
|------|---------|---------|
| `-o` / `--output` | `output.<format>` | output file path; `-` for stdout |
| `-y` / `--yes` / `--overwrite` | off | overwrite output without the 10 s prompt |
| `--check` / `--doctor` | — | verify prerequisites and exit |
| `--model` | `large-v3` | Whisper model size or local path |
| `--device` | `auto` | `auto` picks CUDA if present, else CPU |
| `--compute-type` | `auto` | `float16` on GPU, `int8` on CPU |
| `--language` | auto | language code, or auto-detect |
| `--format` | `txt` | `txt`, `srt`, `vtt`, `json` |
| `--denoise` | `ffmpeg` | `ffmpeg`, `deepfilter`, or `none` |
| `--enhance` | `none` | voice enhancement: `speech`, `strong`, `resemble` |
| `--gain DB` | `0` | manual volume adjustment in dB (e.g. `6`, `-3`) |
| `--diarize` | off | speaker tagging (optional extra) |
| `--diarize-backend` | `pyannote` | `pyannote` or `whisperx` |
| `--assess-only` | off | fast audio triage (loudness/silence), then exit |
| `--on-bad` | `warn` | on a bad recording: `warn`, `skip`, or `fail` (exit 12) |
| `--summarize` | off | summarize via local LLM |
| `--summarize-backend` | `auto` | `auto`, `ollama`, or `llamacpp` |
| `--summarize-model` | — | e.g. `llama3.1` (Ollama) |
| `--keep-clean PATH` | — | also save the denoised 16 kHz wav |
| `--no-vad` | off | disable silence trimming |

---

## How the pieces work

- **Input formats.** No format allow-list. The file is handed to FFmpeg, and
  `ffprobe` confirms it contains a decodable audio stream — so support tracks
  exactly what your FFmpeg build can decode, whatever the extension. Files with
  no audio stream get a clear error before any model runs.
- **Denoising.** Default (`--denoise ffmpeg`) runs a speech-tuned FFmpeg filter
  chain (`highpass`, `lowpass`, `afftdn`, `dynaudnorm`) — no extra deps.
  `--denoise deepfilter` swaps in DeepFilterNet, a small neural denoiser that's
  better on hard/noisy audio and still real-time on CPU. Everything is resampled
  to 16 kHz mono for Whisper.
- **Enhancement.** Applied *after* denoise, for soft or garbled voice.
  `--enhance speech` runs FFmpeg speech normalization + loudness (fixes
  too-quiet recordings); `--enhance strong` adds compression and a presence EQ
  boost for muffled/uneven delivery — both need no extra dependencies.
  `--enhance resemble` uses Resemble-Enhance, a neural model that *restores*
  degraded speech (optional extra, GPU-friendly). `--gain N` applies a manual
  N-dB volume change. Enhancement improves clarity but can't fully recover
  speech that's clipped or destroyed.
- **Transcription.** `faster-whisper` runs Whisper via CTranslate2 with int8/
  float16 quantization, so `large-v3` fits on an 8 GB GPU and also runs on CPU.
  Voice-activity detection trims silence.
- **Diarization.** Two backends — `pyannote` (default, lighter) and `whisperx`.
  Each produces `(start, end, speaker)` turns that flow through one shared merge
  routine, so output is identical either way. Labels *distinct* speakers
  (`Speaker 1/2/…`), not names.
- **Quality assessment.** After transcription, checks Whisper's own per-segment
  signals — `no_speech_prob` (no speech), `avg_logprob` (garbled/uncertain),
  `compression_ratio` (repetitive/hallucinated noise) — plus speech coverage, and
  reports a verdict (`ok`/`suspect`/`bad`, included in the JSON output). A bad
  recording triggers `--on-bad` (default `warn`: still writes the transcript;
  `skip`: writes nothing; `fail`: exit 12). `--assess-only` runs a fast,
  model-free triage from FFmpeg loudness/silence stats without transcribing.
  These are heuristics, so `warn` is the default — a transcript is never dropped
  silently.
- **Summarization.** Sends the transcript to a local LLM server (Ollama or
  llama.cpp) over HTTP using only the standard library. Long transcripts are
  chunked and combined (map-reduce). The prompt asks the model to preserve the
  transcript's original language.

---

## Testing

The regression suite is standard-library only (`unittest` + `unittest.mock`) and
mocks every heavy/network backend, so it runs offline with no GPU or model
downloads:

```bash
python -m unittest discover -s tests -v
# or, with pytest (pip install ".[dev]"):
pytest -q
```

It covers the output writers, device/compute resolution, the diarization merge
(including the mixed worded/wordless regression case), audio-preprocessing
dispatch, local summarization (chunking, backends, detection), the prerequisite
checker, and the CLI (argument parsing, `--check`, error codes, input probing,
UTF-8 output, and orchestration).

---

## Hardware notes

- **Recommended:** NVIDIA GPU with ≥ 8 GB VRAM → `large-v3` in `float16` is fast.
- **CPU-only:** works with `--device cpu` (auto-selected if no GPU); `large-v3`
  is usable but slower — try `--model medium` or `--model small` for speed.
- Diarization adds a modest extra load; summarization load depends on the local
  LLM model you run.

## Troubleshooting

- **`ffmpeg not found`** → install FFmpeg and ensure it's on PATH; re-run
  `plaude-local --check`.
- **GPU not used** → confirm `--check` shows a CUDA device; if not, verify the
  NVIDIA driver + CUDA/cuDNN runtime. `--device cpu` always works as a fallback.
- **Diarization can't load** → you need a Hugging Face token *and* must accept
  the pyannote model terms once (see Prerequisites).
- **`--summarize` says no server** → start Ollama (`ollama serve`) or a
  llama.cpp `llama-server`; check the URL with `--summarize-url` if non-default.

## Contributing

This project ships a Python implementation and a **feature-parity PowerShell
port** (`powershell/`). **Every change must be made to both, keeping them at
100% feature parity** — see [`CONTRIBUTING.md`](CONTRIBUTING.md) (and
[`CLAUDE.md`](CLAUDE.md) for automated/agent contributors) for the required
workflow and checklist.

## License

MIT
