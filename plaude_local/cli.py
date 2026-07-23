"""Command-line interface for plaude-local.

Uses only the standard-library ``argparse`` so the CLI itself adds no
dependencies. Heavy backends (faster-whisper, DeepFilterNet, pyannote, whisperx)
are imported lazily by the modules they live in, so ``--help`` and ``--check``
work even before the optional pieces are installed.
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

# Input format support is delegated entirely to the FFmpeg binary: any audio
# codec/container FFmpeg can decode is accepted (and the audio track of video
# files too). We do not keep a per-format allow-list - FFmpeg (probed via
# ffprobe) is the single source of truth for what is decodable.


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="plaude-local",
        description="Local, offline speech-to-text. Transcribe audio in any "
                    "FFmpeg-decodable format (wav, mp3, m4a, flac, ogg, video "
                    "files, ...) with faster-whisper, with optional denoising, "
                    "speaker diarization, and local-LLM summarization. Output is "
                    "always UTF-8 (handles Chinese, Japanese, and other scripts).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", type=str, nargs="?", default=None,
                   help="path to an audio or video file in ANY format FFmpeg "
                        "can decode (wav, mp3, m4a, aac, flac, ogg/opus, wma, "
                        "mp4, mkv, ...)")
    p.add_argument(
        "-o", "--output", type=str, default=None,
        help="output file path (default: alongside the input, using --format's "
             "extension). Use '-' to write to stdout. Always written as UTF-8.",
    )
    p.add_argument(
        "-f", "--format", choices=formats.FORMATS, default="txt",
        help="output format",
    )
    p.add_argument("--check", "--doctor", action="store_true", dest="check",
                   help="check prerequisites (ffmpeg, models, optional extras) "
                        "and exit")

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
                        help="tag speakers (who said what). Requires the "
                             "chosen backend and a Hugging Face token.")
    g_diar.add_argument("--diarize-backend", choices=["pyannote", "whisperx"],
                        default="pyannote",
                        help="diarization backend: pyannote (lighter) or "
                             "whisperx")
    g_diar.add_argument("--hf-token", default=None,
                        help="Hugging Face token (or set HF_TOKEN)")
    g_diar.add_argument("--num-speakers", type=int, default=None,
                        help="exact number of speakers, if known")
    g_diar.add_argument("--min-speakers", type=int, default=None)
    g_diar.add_argument("--max-speakers", type=int, default=None)

    # Summarization
    g_sum = p.add_argument_group("summarization (optional, local LLM)")
    g_sum.add_argument("--summarize", action="store_true",
                       help="summarize the transcript with a local LLM "
                            "(Ollama or llama.cpp)")
    g_sum.add_argument("--summarize-backend", choices=["auto", "ollama", "llamacpp"],
                       default="auto", help="local LLM backend")
    g_sum.add_argument("--summarize-model", default=None, metavar="NAME",
                       help="model name for the Ollama backend (e.g. llama3.1); "
                            "ignored by llama.cpp, which binds its model at "
                            "server launch")
    g_sum.add_argument("--summarize-url", default=None, metavar="URL",
                       help="override the local LLM server URL")
    g_sum.add_argument("--summary-output", default=None, metavar="PATH",
                       help="where to write the summary (default: alongside the "
                            "transcript as .summary.md). Use '-' for stdout.")
    g_sum.add_argument("--summarize-max-chars", type=int, default=8000,
                       help="chunk size for long transcripts")

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


def _write_utf8(text: str, out_path: Optional[str]) -> None:
    """Write ``text`` as UTF-8 to a file, or to stdout when ``out_path`` is '-'.

    stdout is written via its binary buffer so non-Latin scripts (Chinese,
    Japanese, ...) are emitted as UTF-8 regardless of the console's code page.
    """
    if out_path == "-":
        data = text.encode("utf-8")
        buf = getattr(sys.stdout, "buffer", None)
        if buf is not None:
            buf.write(data)
            buf.flush()
        else:  # pragma: no cover - unusual stdout replacement
            sys.stdout.write(text)
    else:
        Path(out_path).write_text(text, encoding="utf-8")


def _run_check() -> int:
    from . import preflight
    checks = preflight.run_all()
    sys.stdout.write(preflight.format_report(checks))
    return 1 if preflight.missing_required(checks) else 0


def run(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.check:
        return _run_check()

    if not args.input:
        print("error: an input audio file is required (or use --check to verify "
              "your setup).", file=sys.stderr)
        return 2

    in_path = Path(args.input)
    if not in_path.is_file():
        print(f"error: input file not found: {in_path}", file=sys.stderr)
        return 2

    if not audio.have_ffmpeg():
        from . import preflight
        print("error: " + preflight.check_ffmpeg().remedy, file=sys.stderr)
        return 3

    # Let FFmpeg decide what's decodable: probe for an audio stream regardless
    # of file extension. This is what makes the tool accept anything FFmpeg
    # supports. If ffprobe isn't available we skip the check and let the decode
    # step surface any error.
    if audio.have_ffprobe():
        codec = audio.probe_audio_codec(in_path)
        if codec is None:
            print(f"error: FFmpeg found no decodable audio stream in {in_path}. "
                  "Provide an audio file (or a video file that contains audio) "
                  "in any FFmpeg-supported format.", file=sys.stderr)
            return 10
        _log(args.quiet, f"      input audio codec: {codec}")

    # Diarization needs word timestamps for the best merge.
    want_words = args.diarize
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if args.diarize and not hf_token:
        print("error: --diarize requires a Hugging Face token via --hf-token "
              "or the HF_TOKEN environment variable. Create one at "
              "https://hf.co/settings/tokens and accept the model terms at "
              "https://hf.co/pyannote/speaker-diarization-3.1", file=sys.stderr)
        return 4

    # Fail fast if summarization is requested but no local LLM is reachable.
    if args.summarize:
        from . import summarize as summ
        base = args.summarize_url
        if args.summarize_backend == "auto":
            reachable = summ.detect_backend(
                base or summ.DEFAULT_OLLAMA_URL, base or summ.DEFAULT_LLAMACPP_URL
            ) is not None
        elif args.summarize_backend == "ollama":
            reachable = summ.ollama_available(base or summ.DEFAULT_OLLAMA_URL)
        else:
            reachable = summ.llamacpp_available(base or summ.DEFAULT_LLAMACPP_URL)
        if not reachable:
            from . import preflight
            print("error: --summarize needs a local LLM server. "
                  + preflight.check_summarizer().remedy, file=sys.stderr)
            return 8

    with tempfile.TemporaryDirectory(prefix="plaude-local-") as tmp:
        # 1. Preprocess / denoise
        _log(args.quiet, f"[1/4] preparing audio (denoise={args.denoise}) ...")
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
        _log(args.quiet, f"[2/4] loading model {args.model!r} ...")
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
        _log(args.quiet, f"      detected language: {meta.get('language')}")

        # 3. Diarize (optional)
        if args.diarize:
            from . import diarize
            _log(args.quiet,
                 f"[3/4] diarizing speakers (backend={args.diarize_backend}) ...")
            try:
                segments = diarize.diarize_and_merge(
                    str(prepared), segments,
                    backend=args.diarize_backend,
                    hf_token=hf_token,
                    device=engine.device,
                    num_speakers=args.num_speakers,
                    min_speakers=args.min_speakers,
                    max_speakers=args.max_speakers,
                )
            except diarize.DiarizeError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 7
        else:
            _log(args.quiet, "[3/4] diarization skipped")

    # 4. Render + write transcript (UTF-8)
    meta["timestamps"] = args.diarize  # show clock in txt when we have speakers
    text = formats.render(segments, args.format, meta=meta)
    transcript_out = args.output if args.output else _default_output(str(in_path), args.format)
    try:
        _write_utf8(text, transcript_out)
    except OSError as exc:
        print(f"error: could not write transcript to {transcript_out}: {exc}",
              file=sys.stderr)
        return 9
    if transcript_out != "-":
        _log(args.quiet, f"[4/4] transcript -> {transcript_out}")

    # 5. Summarize (optional)
    if args.summarize:
        from . import summarize as summ
        _log(args.quiet, f"      summarizing (backend={args.summarize_backend}) ...")
        plain = formats.to_text(segments)  # language-preserving plain text
        try:
            summary = summ.summarize(
                plain,
                backend=args.summarize_backend,
                model=args.summarize_model,
                url=args.summarize_url,
                max_chars=args.summarize_max_chars,
            )
        except summ.SummarizeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 8
        body = "# Summary\n\n" + summary.rstrip() + "\n"
        to_stdout = args.summary_output == "-" or (
            transcript_out == "-" and not args.summary_output)
        summary_out = "-" if to_stdout else (
            args.summary_output or str(Path(transcript_out).with_suffix(".summary.md")))
        try:
            _write_utf8(("\n" + body) if to_stdout else body, summary_out)
        except OSError as exc:
            print(f"error: could not write summary to {summary_out}: {exc}",
                  file=sys.stderr)
            return 9
        if not to_stdout:
            _log(args.quiet, f"      summary -> {summary_out}")

    return 0


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
