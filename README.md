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

## Tech stack that works for both versions

Both the **Python** CLI and the **PowerShell** port (`powershell/Plaude-Local.ps1`)
share the same underlying engines, so one environment runs both. The table below
is the exact combination verified end-to-end on **Windows 11 + an 8 GB NVIDIA GPU**,
including GPU transcription, LLM translation/summary, **and** speaker diarization.
Linux/WSL2 is more forgiving (pyannote 4.x + `k2` install cleanly there).

| Layer | Package / tool | Verified version | Used by |
|-------|----------------|------------------|---------|
| Runtime | Python | **3.11** (diarization); 3.13 works for transcribe/translate only | both |
| Decode | FFmpeg + ffprobe | any recent (winget `Gyan.FFmpeg`) | both |
| Transcribe | `faster-whisper` (Py) / `whisper-ctranslate2` (PS) | on **`ctranslate2` 4.8.1** | both |
| GPU runtime | `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` / `nvidia-cuda-nvrtc-cu12` / `nvidia-cuda-runtime-cu12` | 12.9.2.10 / 9.25.0.15 / 12.9.86 / 12.9.79 | both |
| Diarization | `pyannote.audio` | **3.1.1** (Windows-viable; avoids `k2`) | both |
| ↳ its stack | `torch` / `torchaudio` / `numpy` / `huggingface_hub` / `speechbrain` | 2.2.2 / 2.2.2 / 1.26.4 / 0.23.4 / **0.5.16** | both |
| LLM (translate/summary) | **Ollama** with a GPU-fitting model | `qwen2.5:7b` (~4.7 GB, strong multilingual) | both |

**Windows GPU + diarization notes** (both tools):
- The diarization stack is version-sensitive: `pyannote.audio 3.1.1` needs the
  older `huggingface_hub 0.23.4` and **`speechbrain 0.5.16`** (speechbrain 1.x
  lazily imports `k2`, which has no Windows wheels). Pin all of these together.
- For **GPU** transcription in a venv, CTranslate2 loads `cublas64_12.dll` by name
  from its own package directory. The Python entry point registers the
  `nvidia-*-cu12` wheel DLL dirs automatically; in a separate venv you may also
  need to copy those wheels' DLLs next to `…/site-packages/ctranslate2/`.
- Diarization is **CPU** (pyannote via `torch` CPU) while Whisper uses the GPU —
  they coexist within ~8 GB. On an 8 GB GPU, keep the LLM small (qwen2.5:7b);
  larger models (gemma4 9.6 GB, 30B+) spill to CPU and crawl. `--list-models`
  shows what's installed and the translation default.
- Local LLM URLs default to **`127.0.0.1`** (not `localhost`) so detection is
  instant on Windows (PowerShell's HTTP client stalls resolving `localhost`→IPv6).

The PowerShell port runs diarization through the same `pyannote.audio` as Python
(via the bundled `powershell/pyannote_diarize.py` helper) — not
`whisper-ctranslate2`'s built-in diarization — so both honor `--diarize-model` /
`-DiarizeModel` and `--num/min/max-speakers` and produce identical speaker labels.

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
# 1. create a token: https://hf.co/settings/tokens  (Read scope; one token is enough)
# 2. accept the model terms for the pyannote version you have:
#      pyannote.audio 4.x -> https://hf.co/pyannote/speaker-diarization-community-1
#      pyannote.audio 3.x -> https://hf.co/pyannote/speaker-diarization-3.1
#                            and https://hf.co/pyannote/segmentation-3.0
# 3. expose it:
export HF_TOKEN=hf_xxx            # Windows: set HF_TOKEN=hf_xxx
```

The token is only used **once, to download the gated speaker model** — it is a
download credential, not a cloud service. Your audio never leaves the machine;
diarization (like transcription and summarization) runs entirely on your CPU/GPU.

> **Local-first by default.** Summarization/translation use your local LLM
> (Ollama/llama.cpp over localhost); the only network touch is the one-time model
> download from Hugging Face. plaude-local also **disables Hugging Face telemetry
> by default** (`HF_HUB_DISABLE_TELEMETRY`), so a normal run never phones home.
> Add `--offline` after the first download to guarantee zero network access.

**Download once, then run fully offline.** After the weights are cached (from any
first run that reached the network), add `--offline` to guarantee no network
access — it sets `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` so every model loads
from the local cache only. This applies to the Whisper transcription weights too,
not just diarization. To move to an air-gapped machine, copy your
`~/.cache/huggingface` folder across and always pass `--offline`.

**Prep a machine with no internet at all.** The helper
`tools/prepare_offline_bundle.py` (PowerShell: `tools/Prepare-OfflineBundle.ps1`)
enumerates and pre-downloads every prerequisite — FFmpeg, Python wheels, CUDA
DLLs, and models — then zips them so a recipient can install fully offline:

```bash
python tools/prepare_offline_bundle.py --list          # see what it will fetch
python tools/prepare_offline_bundle.py --include cuda   # build the zip
```

For **summarization** start a local LLM server (pick one):

```bash
# Ollama
#   install from https://ollama.com/download
ollama pull llama3.1
ollama serve            # usually already running as a service

# --- or llama.cpp ---
#   build/download from https://github.com/ggml-org/llama.cpp
llama-server -m your-model.gguf     # serves on http://127.0.0.1:8080
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

# Save every preprocessing stage (original / denoised / enhanced) to compare by ear
plaude-local lecture.m4a --denoise ffmpeg --enhance strong --keep-stages stages/

# Repair bad-hardware damage first: clipping, clicks/crackle, mains hum
plaude-local lecture.m4a --repair --denoise deepfilter --enhance strong
plaude-local hum.wav --dehum 50          # or 60, or auto

# Flag bad recordings (warn by default; transcript still written)
plaude-local maybe-empty.mp3

# Fast triage of a file without transcribing (loudness / silence check)
plaude-local suspect.wav --assess-only

# In batch jobs: exit non-zero (12) on a bad recording instead of writing
plaude-local clip.mp3 --on-bad fail

# Check your environment
plaude-local --check
```

### The HTML dashboard (default output)

By default (`--format html`) every run produces a self-contained, theme-aware
**HTML dashboard** — `output.html` — with:

- a header showing the **detected language**, rough **speech duration**, and
  **word count**;
- a **provenance line** naming the **transcription engine** and the
  **translation engine/model** actually used for the run;
- a **≤250-word summary of critical topics** (local LLM; degrades gracefully to
  a note if no Ollama/llama.cpp server is running); and
- three tabs: **Transcribed** (original), **Translated-*Language*** (target set
  with `--translate-to`, default English), and a **segment-aligned Side-by-Side**.

Translation is **accuracy-first**: `--translate-engine auto` (the default) uses
the **local LLM** whenever an Ollama/llama.cpp server is reachable — translating
line-by-line so the Side-by-Side stays aligned — and otherwise falls back to
Whisper's native (English-only) translate task. Force a specific engine with
`--translate-engine whisper|llm`, and pick a dedicated translation model with
`--translate-model` (independent of `--summarize-model`; both default to the
first model your local server reports when unset).

To see which local LLMs are installed and which one translation will use by
default, run `--list-models`:

```bash
plaude-local --list-models
# Local LLM models on Ollama (http://127.0.0.1:11434):
#   * qwen2.5:7b   <- default for translation
#     gemma4:latest
# Choose a model with --translate-model NAME (translation) or --summarize-model NAME (summary).
```

`--split-outputs` (or `--transcription-file` / `--translation-file`) also writes
the transcription and translation as plain text files (defaults
`transcription.txt` / `translation.txt`).

```bash
plaude-local interview.m4a                       # -> output.html dashboard
plaude-local rede.mp3 --translate-to en --split-outputs
plaude-local rede.mp3 --translate-engine llm --translate-model deepseek-r1  # accuracy-first LLM
plaude-local notes.wav --format txt              # plain transcript instead
```

### Output formats

`--format` accepts `html` (default, the dashboard above), `txt`, `srt`, `vtt`,
`json`. With `--diarize`, speaker labels (`Speaker 1`, `Speaker 2`, …) and
timestamps are added:

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

### Prerequisite check & auto-provisioning

Before each run, the tool verifies that **FFmpeg** (`ffmpeg` + `ffprobe`) is
available. If it isn't, it warns and offers to fix it interactively:

- **install automatically** — downloads FFmpeg from an official source
  (winget on Windows, Homebrew on macOS, or a self-contained static build) and
  adds it to the run's PATH;
- **provide a path** — you point it at an existing `ffmpeg`/`ffprobe`; or
- **abort** — the run stops.

For non-interactive/scripted use: pass `--ffmpeg-location PATH` to use an
existing install, `--install-missing` to pre-authorize the download, or
`--no-provision` to just error out. If FFmpeg can't be provided, the run aborts
(exit 3). GPU users: CUDA/cuDNN runtime DLLs are reported by `--check` but not
auto-installed (they're large and system-specific) — see the GPU notes.

```bash
# Point at an existing ffmpeg if it isn't on PATH
plaude-local note.mp3 --ffmpeg-location "C:\tools\ffmpeg\bin"

# Let it download FFmpeg automatically (e.g. first run on a fresh machine)
plaude-local note.mp3 --install-missing
```

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
| `--list-models` | — | list installed Ollama LLMs, mark the translation default, and exit |
| `--ffmpeg-location PATH` | — | folder/binary to use if ffmpeg isn't on PATH |
| `--install-missing` | off | auto-download+install missing FFmpeg (no prompt) |
| `--no-provision` | off | don't offer to install/locate; just error out |
| `--model` | `large-v3` | Whisper model size or local path |
| `--device` | `auto` | `auto` picks CUDA if present, else CPU |
| `--compute-type` | `auto` | `float16` on GPU, `int8` on CPU |
| `--language` | auto | language code, or auto-detect |
| `--format` | `html` | `html` dashboard (default), or `txt`/`srt`/`vtt`/`json` |
| `--translate-to` | `en` | translation-tab target language |
| `--translate-engine` | `auto` | `auto` (LLM if a server is up, else Whisper), `whisper`, or `llm` |
| `--translate-model` | — | model for LLM translation (defaults to the server's first model) |
| `--split-outputs` | off | also write transcription + translation as text files |
| `--transcription-file` | `transcription.txt` | path for the split transcription |
| `--translation-file` | `translation.txt` | path for the split translation |
| `--denoise` | `ffmpeg` | `ffmpeg`, `deepfilter`, or `none` |
| `--enhance` | `none` | voice enhancement: `speech`, `strong`, `resemble` |
| `--gain DB` | `0` | manual volume adjustment in dB (e.g. `6`, `-3`) |
| `--diarize` | off | speaker tagging (optional extra) |
| `--diarize-backend` | `pyannote` | `pyannote` or `whisperx` |
| `--diarize-model` | auto | pyannote pipeline repo (auto-selected by version) |
| `--offline` | off | use only cached models; never touch the network |
| `--assess-only` | off | fast audio triage (loudness/silence), then exit |
| `--on-bad` | `warn` | on a bad recording: `warn`, `skip`, or `fail` (exit 12) |
| `--summarize` | off | summarize via local LLM |
| `--summarize-backend` | `auto` | `auto`, `ollama`, or `llamacpp` |
| `--summarize-model` | — | e.g. `llama3.1` (Ollama) |
| `--summarize-timeout` | `120` | per-request LLM timeout (s); raise for big reasoning models (deepseek-r1) |
| `--keep-clean PATH` | — | also save the denoised 16 kHz wav |
| `--declip` | off | repair clipped/distorted peaks before denoise (FFmpeg `adeclip`; slow on long files) |
| `--declick` | off | remove clicks, pops and crackle/static bursts before denoise (FFmpeg `adeclick`) |
| `--dehum` | `none` | remove mains hum + 7 harmonics: `50`, `60`, or `auto` (detect from the first 5 min; skipped if none) |
| `--repair` | off | shorthand for `--declip --declick --dehum auto` (an explicit `--dehum none/50/60` is kept) |
| `--keep-stages DIR` | — | save each preprocessing stage (original / repaired / denoised / enhanced wav) + a `.stages.json` settings manifest to DIR |
| `--no-vad` | off | disable silence trimming |
| `-q`, `--quiet` | off | hide progress messages; warnings about degraded output (e.g. `warning: translation unavailable: ...`) are still printed |

---

## How the pieces work

- **Input formats.** No format allow-list. The file is handed to FFmpeg, and
  `ffprobe` confirms it contains a decodable audio stream — so support tracks
  exactly what your FFmpeg build can decode, whatever the extension. Files with
  no audio stream get a clear error before any model runs.
- **Repair.** Optional and off by default; runs *before* denoise on the
  decoded input, to undo damage from bad hardware. `--declip` rebuilds
  peaks flattened by a too-hot input (FFmpeg `adeclip`; CPU-heavy on long
  files). `--declick` removes clicks, pops and bursts of crackle/static
  (`adeclick`). `--dehum 50|60` puts narrow notch filters (Q 30, about 2 Hz
  wide) on the mains frequency and its first 7 harmonics, leaving the voice
  alone; `--dehum auto` compares the 50 Hz and 60 Hz bands with a 55 Hz
  reference over the first 5 minutes and only notches a band that is at
  least 6 dB louder, so clean files are left untouched (the detected
  frequency is logged). `--repair` turns all three on; add `--dehum none`
  to keep hum removal off. The repaired audio is kept at the source sample
  rate as 32-bit float, so peaks rebuilt above full scale are not clipped
  again and DeepFilterNet still gets full-band input. No extra dependencies.
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
- **Inspecting the stages.** `--keep-stages DIR` saves every intermediate as a
  mono wav so you can A/B them by ear or in an audio editor (16 kHz 16-bit,
  except `02-repaired`, which keeps the source rate as 32-bit float):
  `<name>.01-original.wav` (unprocessed decode), `<name>.02-repaired.wav`
  (only with `--declip`/`--declick`/`--dehum`), `<name>.03-denoised.wav`
  (omitted with `--denoise none`) and `<name>.04-enhanced.wav` (only with
  `--enhance`/`--gain`). The last stage written is exactly what Whisper
  hears. `<name>.stages.json` records the repair/denoise/enhance settings
  as requested (e.g. `"dehum": "auto"`), the detected hum frequency
  (`hum_hz`), gain, and the exact FFmpeg filter chains used. Re-running with other
  settings into the same folder overwrites that input's files, so use one
  folder per experiment.
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

### Multilingual end-to-end regression suite

`tests/regression/` holds a committed speech corpus and a runner that pushes it
through **both** implementations for real (transcribe + translate to English):

- **600 files**: 300 clean recordings plus a *damaged twin* of each, across
  **21 languages**: 16 Asian (Mandarin, Cantonese, Japanese, Korean, Vietnamese,
  Thai, Indonesian, Malay, Filipino, Khmer, Burmese, Hindi, Bengali, Tamil,
  Telugu, Urdu) plus English, Spanish, French, German and Russian - about 14
  recordings per language. (`languages.py` also lists all 73 languages both
  Whisper and edge-tts support; add codes to `CORPUS_CODES` to widen it.) Half are single-speaker monologues, half are **multi-speaker
  conversations** (2-3 voices taking turns, with speaker-turn timings recorded
  for future diarization tests). Every file is 2-10 minutes of seeded random
  content, mono Opus 16 kb/s.
- **Damaged twins** combine seeded imperfections: noise/static, 50 or 60 Hz
  mains hum, clicks/crackle, clipping or a very low level, a muffled or
  telephone-band microphone, and room echo. They run with
  `--repair --enhance strong --keep-stages`, and `--dehum auto` must detect
  exactly the hum that was injected (or none).
- **Scoring** per run: exit code, detected language, transcript character error
  rate against the source text, and English content-word recall against the
  exact English reference. Thresholds depend on how well Whisper handles the
  language (tier A/B/C in `languages.py`); `baseline.json` catches regressions.

```bash
# quick stratified subset: 24 items spread over languages and monologue/conversation,
# each run clean + damaged in both implementations (96 runs)
python tests/regression/run_regression.py --sample 24
# everything (~46 h of audio per implementation - an overnight+ GPU job)
python tests/regression/run_regression.py
# one implementation / some languages; record a new baseline
python tests/regression/run_regression.py --impl powershell --languages de,ja
python tests/regression/run_regression.py --update-baseline
```

Needs FFmpeg, faster-whisper (Python) / whisper-ctranslate2 (PowerShell), an
Ollama server with a translation model, and ideally a CUDA GPU.

**No suitable local GPU?** Run it on Google Colab: open
`tests/regression/colab_regression.ipynb` in Colab (File -> Open notebook ->
GitHub, or upload it), pick a GPU runtime and *Run all*. It installs FFmpeg,
faster-whisper, whisper-ctranslate2, PowerShell 7 and Ollama +
`translategemma:12b`, runs both tools with Whisper `large-v3`, and saves
`results.json` (after every run) plus `baseline.json` to Google Drive; after a
disconnect, *Run all* again resumes (`--resume`). Copy the resulting
`baseline.json` into `tests/regression/`. The runner uses `pwsh` automatically
where Windows PowerShell isn't available.

The corpus is regenerated with `tests/regression/generate_corpus.py` (seeded and
resumable) in two phases, **neither of which needs a GPU**:

```bash
# 1. translate the English source text into every language with Ollama, forced
#    onto the CPU (num_gpu 0; ~5 s/line with a 12B model, ~30 min/language).
#    Writes tests/regression/translations/<code>.json - commit these.
python tests/regression/generate_corpus.py --phase translate \n    --model translategemma:4b --fallback-model translategemma:12b
# 2. synthesize + assemble the audio: edge-tts (cloud TTS) + FFmpeg, no LLM.
python tests/regression/generate_corpus.py --phase audio
```

Pass `--model` to pick the translation model (recommended, e.g.
`--model translategemma:4b`; each file records it), `--ollama-device auto` to
let Ollama use a GPU, `--ollama-url` for a non-default server, `--fallback-model` for the bigger
model that takes over **whenever translation quality looks poor** - every
line is checked (wrong or mixed script, English echoes, junk/markup,
implausible length, repetition loops); lines that still fail after the main
model's retry, whole languages it garbles (a 4B model manages Thai or German
but not Burmese or Khmer), and poor lines in existing files are redone with
the fallback (the 12B), and
`--languages de,ja` to limit either phase. Only transient Ollama errors
(connection refused, HTTP 5xx) are retried. Each translation file stores a hash
of the English source lines, so editing `sentences.py` / `dialogues.py` makes
the affected files re-translate; to re-translate a language by hand, delete its
`translations/<code>.json`. An interrupted language resumes from
`<code>.partial.json`. Offline unit tests for the
harness itself run with the normal suite (`tests/test_regression_suite.py`).

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
- **Diarization on Windows (`k2` / `torchaudio.AudioMetaData` errors)** → this is
  a pyannote/PyTorch packaging problem on Windows, not a plaude-local bug.
  Transcription, GPU, and summarization are unaffected; only pyannote's runtime
  is. The two failure modes:
  - `Lazy import of speechbrain.integrations.k2_fsa failed` — pyannote **4.x**
    (the gated `speaker-diarization-community-1` model) needs `k2`, which has **no
    Windows wheels**.
  - `module 'torchaudio' has no attribute 'AudioMetaData'` — pyannote **3.x** needs
    an older torchaudio; the newer one installed alongside recent Python dropped it.

  **Recommended fix (native Windows, no k2):** use **Python 3.10 or 3.11** with an
  older, matched torch/torchaudio and pyannote **3.x** — the 3.1 pipeline never
  touches k2, and torch/torchaudio 2.1–2.2 still expose `AudioMetaData` and have
  wheels for those Python versions:

  ```bash
  py -3.11 -m venv .venv-diar
  .venv-diar\Scripts\pip install torch==2.2.2 torchaudio==2.2.2 \
      "pyannote.audio<4" faster-whisper
  set HF_TOKEN=hf_xxx
  .venv-diar\Scripts\python -m plaude_local meeting.m4a --diarize
  ```

  Recent CPython (3.13) has **no** compatible older-torch wheels, so a 3.13-only
  box can't run pyannote 3.x. **Alternative:** run under **WSL2 / Linux**, where
  both `k2` and pyannote (3.x or 4.x) install cleanly with pip.
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
