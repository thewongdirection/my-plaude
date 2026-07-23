# Project guide for plaude-local

Local, offline speech-to-text CLI. Two implementations that must stay in sync:

- **Python** — `plaude_local/` (the reference implementation), entry point
  `plaude_local.cli:main` / `python -m plaude_local`.
- **PowerShell** — `powershell/Plaude-Local.ps1` (Windows-first port).

## ⚠️ Feature-parity requirement (do not skip)

**Every change to the Python tool MUST be mirrored in the PowerShell version,
and vice versa.** When you add/modify a flag, default, exit code, output
behavior, or pipeline stage in one, make the equivalent change in the other in
the same commit, and update both READMEs (`README.md` and
`powershell/README.md`), including the parity table in the PowerShell README.

If a feature genuinely cannot be reproduced in PowerShell (or Python), document
the difference in the "Known differences" section of `powershell/README.md`
rather than silently letting them drift.

## Architecture (Python)

| Module | Responsibility |
|--------|----------------|
| `cli.py` | argument parsing, orchestration, output/overwrite, `--check` |
| `audio.py` | FFmpeg probe (`ffprobe`) + denoise; input is anything FFmpeg decodes |
| `transcribe.py` | faster-whisper (CTranslate2) wrapper; device/compute auto-resolve |
| `diarize.py` | pyannote + whisperx backends → shared `merge_turns` |
| `summarize.py` | local LLM summary via Ollama / llama.cpp over stdlib `urllib` |
| `preflight.py` | prerequisite checks with remedies (`--check`) |
| `formats.py` | txt / srt / vtt / json writers |

Design constraints: lean dependencies (core = only `faster-whisper`; extras are
optional and lazily imported), always-UTF-8 output, cross-platform (Windows +
Linux), FFmpeg is the sole arbiter of decodable input.

## Dev commands

```bash
# Python test suite (stdlib unittest, fully offline, no GPU/model downloads)
python -m unittest discover -s tests -v
# or, with pytest (pip install ".[dev]")
pytest -q

# Doctor / prerequisite check
python -m plaude_local --check
```

```powershell
# PowerShell test suite (Pester 5, offline; mirrors the offline Python tests)
Invoke-Pester -Path ./powershell
```

Keep both suites green when changing either implementation.

## Conventions

- Keep the CLI usable without optional deps installed (`--help`, `--check` must
  work). Import heavy/optional backends lazily inside the functions that use
  them.
- Add regression tests for any new logic; mock heavy/network backends so the
  suite stays offline.
- Exit codes are part of the contract; keep them stable and mirrored across both
  implementations (2 input, 3 ffmpeg, 4 diarize-token, 5 audio, 6 transcribe,
  7 diarize, 8 summarize, 9 write, 10 no-audio-stream, 11 overwrite-declined).
