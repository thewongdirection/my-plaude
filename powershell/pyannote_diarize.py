"""Standalone pyannote diarization helper for the PowerShell port.

Runs speaker diarization on an audio file and prints the result as JSON turns
(``[[start, end, "SPEAKER_00"], ...]``) to stdout. The PowerShell script parses
that and merges the turns onto the transcript segments itself (Merge-Turns),
mirroring the Python implementation's ``diarize.merge_turns``.

Why this exists: whisper-ctranslate2's own ``--hf_token`` diarization calls
``Pipeline.from_pretrained(..., token=...)`` (pyannote 4.x API only), which can't
run on Windows because pyannote 4.x needs ``k2`` (no Windows wheels). Calling
pyannote directly here — with the same ``token=`` -> ``use_auth_token=`` fallback
the Python version uses — works with the Windows-viable pyannote 3.1.x too.

Prints a line starting with ``ERROR:`` to stderr and exits non-zero on failure.
"""
import argparse
import json
import sys


# Pipeline repo differs by pyannote.audio major version (parity with
# plaude_local.diarize): 4.x -> gated community-1, 3.x -> speaker-diarization-3.1.
PYANNOTE_MODEL_V4 = "pyannote/speaker-diarization-community-1"
PYANNOTE_MODEL_V3 = "pyannote/speaker-diarization-3.1"


def default_model():
    try:
        from pyannote.audio import __version__ as ver
        major = int(str(ver).split(".")[0])
    except Exception:
        major = 3
    return PYANNOTE_MODEL_V4 if major >= 4 else PYANNOTE_MODEL_V3


def from_pretrained(Pipeline, repo, token):
    # pyannote >= 4 renamed use_auth_token -> token; older releases only accept
    # the old name. Try the new one first and fall back (parity with Python).
    try:
        return Pipeline.from_pretrained(repo, token=token)
    except TypeError:
        return Pipeline.from_pretrained(repo, use_auth_token=token)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--token", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model", default=None)
    ap.add_argument("--num-speakers", type=int, default=None)
    ap.add_argument("--min-speakers", type=int, default=None)
    ap.add_argument("--max-speakers", type=int, default=None)
    args = ap.parse_args()

    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        sys.stderr.write(
            "ERROR: pyannote.audio is not installed in this Python "
            "(pip install \"pyannote.audio>=3.1\").\n" + str(exc) + "\n")
        return 2

    repo = args.model or default_model()
    try:
        pipeline = from_pretrained(Pipeline, repo, args.token)
    except Exception as exc:
        sys.stderr.write(
            f"ERROR: could not load the pyannote pipeline {repo!r}. Accept the "
            "model terms on Hugging Face and provide a valid token. "
            f"Underlying error: {exc}\n")
        return 3
    if pipeline is None:
        sys.stderr.write(
            "ERROR: pyannote returned no pipeline - the token is likely missing "
            "or the model terms have not been accepted.\n")
        return 3

    try:
        import torch
        if args.device == "cuda" and torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
    except Exception:
        pass

    kwargs = {}
    if args.num_speakers is not None:
        kwargs["num_speakers"] = args.num_speakers
    if args.min_speakers is not None:
        kwargs["min_speakers"] = args.min_speakers
    if args.max_speakers is not None:
        kwargs["max_speakers"] = args.max_speakers

    try:
        annotation = pipeline(args.audio, **kwargs)
    except Exception as exc:
        sys.stderr.write(f"ERROR: pyannote diarization failed: {exc}\n")
        return 4

    turns = [
        [float(seg.start), float(seg.end), str(label)]
        for seg, _, label in annotation.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda t: t[0])
    sys.stdout.write(json.dumps(turns))
    return 0


if __name__ == "__main__":
    sys.exit(main())
