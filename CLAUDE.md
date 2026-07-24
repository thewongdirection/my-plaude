# Project guide for plaude-local

Local, offline speech-to-text CLI. Two implementations that must stay in sync:

- **Python** — `plaude_local/` (the reference implementation), entry point
  `plaude_local.cli:main` / `python -m plaude_local`.
- **PowerShell** — `powershell/Plaude-Local.ps1` (Windows-first port).

## ⚠️ 100% feature-parity requirement (MANDATORY, do not skip)

**The Python tool (`plaude_local/`) and the PowerShell tool
(`powershell/Plaude-Local.ps1`) MUST remain at 100% feature parity at all
times.** Every change to one MUST be mirrored in the other **in the same
commit** — this applies to *all* future changes, enhancements, and feature
additions, without exception. There is no "Python-only" or "PowerShell-only"
change.

When you add or modify anything user-facing or behavioral, do ALL of the
following in the same commit:

1. **Implement it in both** `plaude_local/` and `powershell/Plaude-Local.ps1`
   (flags, defaults, choices/ValidateSets, exit codes, output/overwrite
   behavior, pipeline stages, error messages, `--check`/`-Check` entries).
2. **Test it in both** — add regression tests to `tests/` (Python `unittest`)
   and `powershell/Plaude-Local.Tests.ps1` (Pester), kept offline (mock heavy
   / external / network backends).
3. **Document it in both** READMEs (`README.md` and `powershell/README.md`),
   including the **parity table** in `powershell/README.md`, and update this
   file's architecture table if a module's responsibility changes.

Before committing any change, re-read the parity table in
`powershell/README.md` and confirm every row still matches.

If a feature genuinely cannot be reproduced 1:1 (e.g. it depends on a Python
library with no PowerShell-usable equivalent), you MUST still implement the
closest working equivalent AND record the exact difference in the "Known
differences" section of `powershell/README.md`. Silent drift is not allowed.

See `CONTRIBUTING.md` for the human-facing statement of this same policy.

## Architecture (Python)

| Module | Responsibility |
|--------|----------------|
| `cli.py` | argument parsing, orchestration, output/overwrite, `--check` |
| `audio.py` | FFmpeg probe (`ffprobe`) + denoise + enhance (speech/strong/resemble) + gain; input is anything FFmpeg decodes |
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
