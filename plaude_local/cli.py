"""Command-line interface for plaude-local.

Uses only the standard-library ``argparse`` so the CLI itself adds no
dependencies. Heavy backends (faster-whisper, DeepFilterNet, pyannote) are
imported lazily by the modules they live in, so ``--help`` works even before
the optional pieces are installed.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

from . import __version__
from . import audio, formats


MODEL_CHOICES = [
    "tiny", "tiny.en", "base", "base.en", "small", "small.en",
    "medium", "medium.en", "large-v2", "large-v3",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="plaude-local",
        description="Local, offline speech-to-text. Transcribe WAV/MP3 with "
                    "faster-whisper, optional denoising and speaker diarization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", type=str, help="path to a .wav or .mp3 recording")
    p.add_argument(
        "-o", "--output", type=str, default=None,
        help="output file path (default: alongside the input, using --format's "
             "extension). Use '-' to write to stdout.",
    )
    p.add_argument(
        "-f", "--format", choices=formats.FORMATS, default="txt",
        help="output format",
    )

    # Model / hardware
    g_model = p.add_argument_group("model / hardware")
    g_model.add_argument("-m", "--model", default="large-v3", metavar="NAME",
                         help="Whisper model (one of: " + ", ".join(MODEL_CHOICES)
                              + ", or a local path / HF id)")
    g_model.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                         help="compute device")
    g_model.add_argument("--compute-type", default="auto",
                         help="CTranslate2 compute type (auto|float16|int8|"
                              "int8_float16|float32)")
    g_model.add_argument("--language", default=None, metavar="CODE",
                         help="language code (e.g. en, es, zh). Omit to "
                              "auto-detect.")
    g_model.add_argument("--beam-size", type=int, default=5,
                         help="beam search width")
    g_model.add_argument("--no-vad", action="store_true",
                         help="disable voice-activity filtering of silence")
    g_model.add_argument("--model-dir", default=None, metavar="DIR",
                         help="directory to cache/download model weights")

    # Audio preprocessing
    g_audio = p.add_argument_group("audio preprocessing")
    g_audio.add_argument("--denoise", choices=["ffmpeg", "deepfilter", "none"],
                         default="ffmpeg",
                         help="denoise strategy before transcription")
    g_audio.add_argument("--keep-clean", default=None, metavar="PATH",
                         help="also save the denoised 16kHz wav to PATH")

    # Diarization
    g_diar = p.add_argument_group("speaker diarization (optional)")
    g_diar.add_argument("--diarize", action="store_true",
                        help="tag speakers (who said what). Requires "
                             "pyannote.audio and a Hugging Face token.")
    g_diar.add_argument("--hf-token", default=None,
                        help="Hugging Face token (or set HF_TOKEN)")
    g_diar.add_argument("--num-speakers", type=int, default=None,
                        help="exact number of speakers, if known")
    g_diar.add_argument("--min-speakers", type=int, default=None)
    g_diar.add_argument("--max-speakers", type=int, default=None)

    p.add_argument("-q", "--quiet", action="store_true",
                   help="suppress progress messages on stderr")
    p.add_argument("--version", action="version",
                   version=f"%(prog)s {__version__}")
    return p


def _log(quiet: bool, *msg: object) -> None:
    if not quiet:
        print(*msg, file=sys.stderr, flush=True)


def _default_output(input_path: str, fmt: str) -> str:
    return str(Path(input_path).with_suffix("." + fmt))


def run(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    in_path = Path(args.input)
    if not in_path.is_file():
        print(f"error: input file not found: {in_path}", file=sys.stderr)
        return 2

    if not audio.have_ffmpeg():
        print("error: ffmpeg not found on PATH (needed to read audio). "
              "Install it and retry.", file=sys.stderr)
        return 3

    # Diarization needs word timestamps for the best merge.
    want_words = args.diarize
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if args.diarize and not hf_token:
        print("error: --diarize requires a Hugging Face token via --hf-token "
              "or the HF_TOKEN environment variable.", file=sys.stderr)
        return 4

    with tempfile.TemporaryDirectory(prefix="plaude-local-") as tmp:
        # 1. Preprocess / denoise
        _log(args.quiet, f"[1/3] preparing audio (denoise={args.denoise}) ...")
        try:
            prepared = audio.prepare(in_path, tmp, denoise=args.denoise)
        except audio.AudioError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 5
        if args.keep_clean:
            import shutil
            shutil.copyfile(prepared, args.keep_clean)
            _log(args.quiet, f"      saved cleaned audio -> {args.keep_clean}")

        # 2. Transcribe
        from . import transcribe  # lazy: avoids importing ctranslate2 for --help
        _log(args.quiet, f"[2/3] loading model {args.model!r} ...")
        try:
            engine = transcribe.Transcriber(
                model=args.model,
                device=args.device,
                compute_type=args.compute_type,
                download_root=args.model_dir,
            )
        except transcribe.TranscribeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 6
        _log(args.quiet, f"      device={engine.device} compute={engine.compute_type}")
        _log(args.quiet, "      transcribing (this runs the model) ...")
        try:
            segments, meta = engine.transcribe(
                str(prepared),
                language=args.language,
                word_timestamps=want_words,
                vad_filter=not args.no_vad,
                beam_size=args.beam_size,
            )
        except transcribe.TranscribeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 6
        lang = meta.get("language")
        _log(args.quiet, f"      detected language: {lang}")

        # 3. Diarize (optional)
        if args.diarize:
            from . import diarize
            _log(args.quiet, "[3/3] diarizing speakers ...")
            try:
                segments = diarize.diarize_and_merge(
                    str(prepared), segments,
                    hf_token=hf_token,
                    num_speakers=args.num_speakers,
                    min_speakers=args.min_speakers,
                    max_speakers=args.max_speakers,
                )
            except diarize.DiarizeError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 7
        else:
            _log(args.quiet, "[3/3] diarization skipped")

    # Render + write
    meta["timestamps"] = args.diarize  # show clock in txt when we have speakers
    text = formats.render(segments, args.format, meta=meta)

    if args.output == "-":
        sys.stdout.write(text)
    else:
        out_path = args.output or _default_output(str(in_path), args.format)
        Path(out_path).write_text(text, encoding="utf-8")
        _log(args.quiet, f"done -> {out_path}")

    return 0


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
