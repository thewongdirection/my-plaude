# plaude-local — PowerShell edition

A Windows-first PowerShell port of `plaude-local`, kept at **feature parity**
with the Python version. Same pipeline, same flags (in PowerShell style), same
behavior: local/offline speech-to-text with optional denoising, speaker
diarization, and local-LLM summarization, and always-UTF-8 output.

```
audio (any FFmpeg-decodable format)
  → denoise (FFmpeg / DeepFilterNet)
  → transcribe (whisper-ctranslate2 = faster-whisper / CTranslate2 engine)
  → diarize (optional)
  → summarize (optional, Ollama / llama.cpp)
  → transcript (txt/srt/vtt/json, UTF-8) + summary.md
```

## Why a separate implementation?

The Python tool uses the `faster-whisper` library directly. PowerShell can't
import a Python library, so the port drives the **same underlying engine**
through its command-line client, `whisper-ctranslate2` (CTranslate2 /
faster-whisper — identical models, devices, and compute types). Everything else
(argument handling, FFmpeg probe/denoise, output + overwrite logic, UTF-8,
summarization over HTTP, the `-Check` doctor) is implemented natively in
PowerShell. The result behaves like the Python version, not as a wrapper around
it.

## Prerequisites

| Component | Required? | For | Where to get it |
|-----------|-----------|-----|-----------------|
| **PowerShell 5.1+** (or PowerShell 7+) | ✅ | running the script | ships with Windows; PS7: https://aka.ms/powershell |
| **FFmpeg + ffprobe** | ✅ | decode any input, denoise | `winget install ffmpeg` |
| **whisper-ctranslate2** | ✅ | transcription (faster-whisper engine) | `pip install whisper-ctranslate2` |
| **NVIDIA GPU + CUDA/cuDNN** | ⭐ recommended | speed | NVIDIA driver + CUDA runtime |
| **DeepFilterNet** (`deepFilter`) | ⚪ optional | `-Denoise deepfilter` | `pip install deepfilternet` |
| **Resemble-Enhance** (`resemble-enhance`) | ⚪ optional | `-Enhance resemble` | `pip install resemble-enhance` |
| **pyannote/whisperx + HF token** | ⚪ optional | `-Diarize` | handled by whisper-ctranslate2; `pip install pyannote.audio`; set `$env:HF_TOKEN` |
| **Ollama** or **llama.cpp** server | ⚪ optional | `-Summarize` | https://ollama.com/download · https://github.com/ggml-org/llama.cpp |

Run the built-in checker any time:

```powershell
.\Plaude-Local.ps1 -Check
```

## Setup (Windows)

```powershell
# 1. FFmpeg (includes ffprobe)
winget install ffmpeg

# 2. The transcription engine (same as the Python version)
pip install whisper-ctranslate2

# 3. (optional) diarization / denoise / summary extras
pip install pyannote.audio deepfilternet
$env:HF_TOKEN = "hf_xxx"        # for -Diarize
# and/or start Ollama:  ollama pull llama3.1

# 4. Allow running the script in this session if needed
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 5. Verify
.\Plaude-Local.ps1 -Check
```

## Usage

```powershell
# Simplest: writes output.txt in the current directory (UTF-8)
.\Plaude-Local.ps1 note.mp3

# m4a / any FFmpeg-decodable format
.\Plaude-Local.ps1 voice-memo.m4a

# Specify the output file name
.\Plaude-Local.ps1 note.mp3 -Output meeting-notes.txt

# Subtitles, a specific language, force CPU
.\Plaude-Local.ps1 talk.mp3 -Format srt -Output talk.srt
.\Plaude-Local.ps1 会议.mp3 -Language zh
.\Plaude-Local.ps1 call.wav -Model small -Device cpu

# Speaker diarization
$env:HF_TOKEN = "hf_xxx"
.\Plaude-Local.ps1 interview.mp3 -Diarize

# Enhance soft or garbled voice (after denoise)
.\Plaude-Local.ps1 quiet.mp3 -Enhance speech
.\Plaude-Local.ps1 muffled.m4a -Enhance strong
.\Plaude-Local.ps1 faint.wav -Gain 8

# Flag / triage bad recordings
.\Plaude-Local.ps1 suspect.wav -AssessOnly       # fast, no transcription
.\Plaude-Local.ps1 clip.mp3 -OnBad fail          # exit 12 on a bad recording

# Summarize with a local LLM
.\Plaude-Local.ps1 meeting.mp3 -Summarize -SummarizeModel llama3.1

# Overwrite an existing output without the 10-second prompt
.\Plaude-Local.ps1 note.mp3 -Output out.txt -Yes

# Print to the console instead of a file
.\Plaude-Local.ps1 note.mp3 -Output -

# Check prerequisites
.\Plaude-Local.ps1 -Check
```

## Prerequisite check & auto-provisioning

Like the Python version, before each run the script checks for FFmpeg
(`ffmpeg` + `ffprobe`) and, if missing, warns and offers to **install it
automatically** (winget, or a self-contained download), **accept a path**
(`-FfmpegLocation`), or **abort**. Use `-InstallMissing` to pre-authorize the
download or `-NoProvision` to just error out.

```powershell
.\Plaude-Local.ps1 note.mp3 -FfmpegLocation 'C:\tools\ffmpeg\bin'
.\Plaude-Local.ps1 note.mp3 -InstallMissing
```

## Behavior parity with the Python version

| Feature | Python (`plaude-local`) | PowerShell (`Plaude-Local.ps1`) |
|---------|-------------------------|---------------------------------|
| Input formats | anything FFmpeg decodes (ffprobe-gated) | same |
| Default output | **HTML dashboard** (`output.html`) | same |
| Output format | `--format html`(default)`/txt/srt/vtt/json` | `-Format html/txt/srt/vtt/json` |
| Dashboard | summary + provenance line + Transcribed / Translated-*lang* / Side-by-Side tabs | same |
| Translation target | `--translate-to LANG` (default en) | `-TranslateTo LANG` |
| Translation engine | `--translate-engine auto/whisper/llm` (accuracy-first: LLM if a server is up) | `-TranslateEngine auto/whisper/llm` |
| Translation model | `--translate-model NAME` (defaults to server's first model) | `-TranslateModel NAME` |
| List local LLMs | `--list-models` (marks the translation default) | `-ListModels` |
| Split text outputs | `--split-outputs` / `--transcription-file` / `--translation-file` | `-SplitOutputs` / `-TranscriptionFile` / `-TranslationFile` |
| Overwrite prompt | prompt, auto-overwrite after 10 s; `--yes` skips | same (`-Yes`) |
| UTF-8 output (file + stdout) | yes | yes |
| Denoise | `--denoise ffmpeg/deepfilter/none` | `-Denoise ffmpeg/deepfilter/none` |
| Enhance | `--enhance none/speech/strong/resemble` | `-Enhance none/speech/strong/resemble` |
| Gain | `--gain DB` | `-Gain DB` |
| Quality triage | `--assess-only` | `-AssessOnly` |
| Bad-recording policy | `--on-bad warn/skip/fail` | `-OnBad warn/skip/fail` |
| Transcription engine | faster-whisper (CTranslate2) | whisper-ctranslate2 (same engine) |
| Models / device / compute | `--model/--device/--compute-type` | `-Model/-Device/-ComputeType` |
| Offline (cache-only models) | `--offline` | `-Offline` |
| Diarization | `--diarize` (pyannote, direct) | `-Diarize` (pyannote, direct — same engine) |
| Diarization model | `--diarize-model REPO` | `-DiarizeModel REPO` (honored) |
| Diarization speakers | `--num/min/max-speakers N` | `-NumSpeakers/-MinSpeakers/-MaxSpeakers N` (honored) |
| Diarization backend | `--diarize-backend pyannote/whisperx` | `-DiarizeBackend` (pyannote only; whisperx not ported) |
| Summarization | `--summarize` (Ollama/llama.cpp) | `-Summarize` (Ollama/llama.cpp) |
| LLM request timeout | `--summarize-timeout SECONDS` (default 120) | `-SummarizeTimeout SECONDS` |
| Prereq check + provision | (built-in) | (built-in) |
| Provide ffmpeg | `--ffmpeg-location` | `-FfmpegLocation` |
| Auto-install ffmpeg | `--install-missing` | `-InstallMissing` |
| Disable provisioning | `--no-provision` | `-NoProvision` |
| Doctor | `--check` | `-Check` |
| Version | `--version` | `-Version` |
| Exit codes | 2/3/4/5/6/7/8/9/10/11/12 | same meanings (incl. 7 = diarize failure) |

### Known differences

- **Transcript formatting**: the Python version renders txt/srt/vtt/json with
  its own writers; the PowerShell version uses `whisper-ctranslate2`'s output
  writers. Content is equivalent; minor spacing/label differences can occur.
- **Diarization**: **now uses the same engine as Python** — pyannote run
  *directly* (via the bundled `pyannote_diarize.py` helper) rather than through
  `whisper-ctranslate2`'s built-in diarization. This closes the earlier gaps:
  `-DiarizeModel`, `-NumSpeakers`, `-MinSpeakers`, and `-MaxSpeakers` are all
  **honored** (same as Python), speakers are merged onto segments by the shared
  overlap logic (`Merge-Turns`, parity with `diarize.merge_turns`), and a
  diarization failure returns **exit 7** (matching Python). The one remaining
  gap: `-DiarizeBackend whisperx` is **not ported** (pyannote only); it warns and
  uses pyannote. Diarization needs a Python with `pyannote.audio` on PATH and an
  HF token (`-HfToken` / `$env:HF_TOKEN`) — see the shared **tech stack** in the
  main [README](../README.md#tech-stack-that-works-for-both-versions), which is
  the exact combination verified to run both tools (incl. diarization) on
  Windows.
- **Diarization word-level split**: the Python `merge_turns` has a word-level
  branch that splits a single Whisper segment spanning two speakers using word
  timestamps. The PowerShell `Merge-Turns` implements only the segment-level
  branch, because `whisper-ctranslate2` segments carry no word timings — so a
  segment is tagged as a whole with its best-overlapping speaker. Identical
  output in practice (no word timings ever reach it); a difference only if
  word-level input were somehow supplied.
- **`-ModelDir`**: redirects the Hugging Face cache (via `$env:HF_HOME`) rather
  than mapping to faster-whisper's `download_root`; effect is equivalent
  (controls where model weights are stored/downloaded).
- **FFmpeg auto-install source**: both ports prefer the platform package
  manager (winget) and fall back to a self-contained download. The Python
  version also supports a Linux static-build download and macOS Homebrew; the
  PowerShell port targets Windows (winget / gyan.dev zip). Both accept
  `-FfmpegLocation`/`--ffmpeg-location` and abort if declined.
- **Quality report in JSON**: both ports log the verdict to stderr and honor
  `-OnBad`. In JSON output, the Python version nests the report under
  `meta.quality`; the PowerShell port adds it as a top-level `quality` field
  (because the JSON file is produced by `whisper-ctranslate2`). Same data.
- **`-Enhance resemble`**: the PowerShell port drives the `resemble-enhance`
  **CLI** (which processes a directory), whereas the Python version calls the
  Resemble-Enhance Python API directly. Same model, equivalent result. The
  `speech`/`strong` FFmpeg enhancement modes and `-Gain` are identical to Python.
- **Windows CUDA from pip wheels**: when the CUDA runtime comes from the
  `nvidia-*-cu12` wheels, both ports register those wheels' DLL directories
  (`cublas64_12.dll` / `cudnn64_9.dll` ...) before loading CUDA, so `-Device
  cuda` works out of the box (Python 3.8+ otherwise ignores `PATH` for a native
  extension's DLLs). The Python entry point does this in-process; the PowerShell
  port runs `whisper-ctranslate2` through a tiny Python bootstrap to do the same,
  so **`python` (or `py`) must be on PATH for GPU runs** — it already is wherever
  `whisper-ctranslate2` is installed. A system-wide CUDA toolkit still works too.

## Tests

Offline Pester tests (mirroring the offline Python unittest suite) cover the
pure/logic functions — no external tools or network:

```powershell
Invoke-Pester -Path ./powershell
```

> **Maintainers:** any future change to the Python tool must be mirrored here to
> preserve parity. See `../CLAUDE.md`.
