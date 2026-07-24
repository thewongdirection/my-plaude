# Contributing to plaude-local

Thanks for contributing! This project ships **two implementations that must stay
identical in behavior**:

- **Python** — `plaude_local/` (the reference implementation)
- **PowerShell** — `powershell/Plaude-Local.ps1` (Windows-first port)

## 🔒 The one non-negotiable rule: 100% feature parity

**Any change, enhancement, or new feature MUST be made to BOTH implementations
in the same pull request / commit, so the two stay at 100% feature parity —
forever.** This is the single most important rule in this repo.

There is no such thing as a "Python-only" or "PowerShell-only" change to
user-facing or behavioral code. If you touch one, you touch the other.

### Checklist for every behavioral change

- [ ] Implemented in **Python** (`plaude_local/`)
- [ ] Implemented in **PowerShell** (`powershell/Plaude-Local.ps1`) — same flag
      names/aliases, defaults, allowed values, exit codes, output & overwrite
      behavior, and error messages
- [ ] Tests added in **both** suites:
      - Python: `tests/` (`python -m unittest discover -s tests -v`)
      - PowerShell: `powershell/Plaude-Local.Tests.ps1` (`Invoke-Pester -Path ./powershell`)
      - Both suites must stay **offline** — mock heavy/external/network backends
- [ ] Docs updated in **both** `README.md` and `powershell/README.md`
- [ ] The **parity table** in `powershell/README.md` still matches reality
- [ ] If something can't be reproduced 1:1, the closest equivalent is
      implemented **and** the exact gap is recorded under "Known differences"
      in `powershell/README.md`

### Why two implementations?

The Python version uses the `faster-whisper` library directly; the PowerShell
version drives the same engine via `whisper-ctranslate2`. Everything else
(argument handling, FFmpeg audio, output/overwrite, summarization over HTTP,
the doctor command) is implemented natively in each. Keeping them in lockstep is
what makes the PowerShell edition a true equivalent rather than a wrapper.

## Exit codes (part of the contract — keep stable and mirrored)

`2` input · `3` ffmpeg · `4` diarize-token · `5` audio · `6` transcribe ·
`7` diarize · `8` summarize · `9` write · `10` no-audio-stream ·
`11` overwrite-declined

## Running the tests

```bash
# Python (stdlib unittest, offline)
python -m unittest discover -s tests -v

# PowerShell (Pester 5, offline)
Invoke-Pester -Path ./powershell
```

See `CLAUDE.md` for the architecture overview and the same parity policy stated
for automated/agent contributors.
