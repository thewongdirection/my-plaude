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
        help="output file path. Default: output.<format> (e.g. output.txt) in "
             "the current directory. Use '-' to write to stdout. Always UTF-8.",
    )
    p.add_argument(
        "-f", "--format", choices=list(formats.FORMATS) + ["html"], default="html",
        help="output format. Default 'html' produces an interactive dashboard "
             "(summary + Transcribed / Translated / Side-by-Side tabs); "
             "txt/srt/vtt/json produce the plain transcript.",
    )
    p.add_argument(
        "--translate-to", default="en", metavar="LANG",
        help="target language for the dashboard's translation tab (default: en). "
             "English uses Whisper's native translation; other languages use the "
             "local LLM.",
    )
    p.add_argument(
        "--split-outputs", action="store_true",
        help="also write the transcription and translation to separate text "
             "files (see --transcription-file / --translation-file).",
    )
    p.add_argument(
        "--transcription-file", default=None, metavar="PATH",
        help="write the transcription here (implies --split-outputs; "
             "default: transcription.txt).",
    )
    p.add_argument(
        "--translation-file", default=None, metavar="PATH",
        help="write the translation here (implies --split-outputs; "
             "default: translation.txt).",
    )
    p.add_argument(
        "-y", "--yes", "--overwrite", action="store_true", dest="yes",
        help="overwrite the output file without asking (default: prompt, then "
             "auto-overwrite after 10s)",
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
    g_model.add_argument("--offline", action="store_true",
                         help="never reach the network: use only already-cached "
                              "models (sets HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE). "
                              "Download the weights once online, then run offline.")

    # Audio preprocessing
    g_audio = p.add_argument_group("audio preprocessing")
    g_audio.add_argument("--denoise", choices=["ffmpeg", "deepfilter", "none"],
                         default="ffmpeg",
                         help="denoise strategy before transcription")
    g_audio.add_argument("--enhance", choices=["none", "speech", "strong", "resemble"],
                         default="none",
                         help="voice enhancement after denoise: speech (fix soft "
                              "volume), strong (garbled/muffled), resemble (neural "
                              "restoration, optional extra)")
    g_audio.add_argument("--gain", type=float, default=0.0, metavar="DB",
                         help="manual volume adjustment in decibels "
                              "(e.g. 6 to boost, -3 to attenuate)")
    g_audio.add_argument("--keep-clean", default=None, metavar="PATH",
                         help="also save the denoised/enhanced 16kHz wav to PATH")

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
    g_diar.add_argument("--diarize-model", default=None, metavar="REPO",
                        help="pyannote pipeline repo to use (default: auto by "
                             "installed version — speaker-diarization-community-1 "
                             "on pyannote 4.x, speaker-diarization-3.1 on 3.x)")
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

    # Quality assessment
    g_qual = p.add_argument_group("recording quality")
    g_qual.add_argument("--assess-only", action="store_true",
                        help="fast model-free triage of the audio (loudness / "
                             "silence) - report a verdict and exit, no transcription")
    g_qual.add_argument("--on-bad", choices=["warn", "skip", "fail"], default="warn",
                        help="action when a recording is judged bad: warn (write "
                             "anyway, default), skip (write nothing, exit 0), or "
                             "fail (write nothing, exit 12)")
    g_qual.add_argument("--min-speech", type=float, default=0.15, metavar="FRAC",
                        help="min speech coverage (0-1) before flagging")
    g_qual.add_argument("--max-compression", type=float, default=2.4, metavar="R",
                        help="compression ratio above which output looks repetitive")
    g_qual.add_argument("--min-logprob", type=float, default=-1.0, metavar="LP",
                        help="avg_logprob below which confidence is too low")
    g_qual.add_argument("--max-no-speech", type=float, default=0.6, metavar="P",
                        help="no_speech_prob above which speech is unlikely")

    # Prerequisites / provisioning
    g_pre = p.add_argument_group("prerequisites")
    g_pre.add_argument("--ffmpeg-location", default=None, metavar="PATH",
                       help="folder containing ffmpeg/ffprobe (or the ffmpeg "
                            "binary) to use if they are not on PATH")
    g_pre.add_argument("--install-missing", action="store_true",
                       help="if FFmpeg is missing, download and install it "
                            "automatically (no prompt) from an official source")
    g_pre.add_argument("--no-provision", action="store_true",
                       help="do not offer to install/locate missing prerequisites; "
                            "just error out")

    p.add_argument("-q", "--quiet", action="store_true",
                   help="suppress progress messages on stderr")
    p.add_argument("--version", action="version",
                   version=f"%(prog)s {__version__}")
    return p


def _log(quiet: bool, *msg: object) -> None:
    if not quiet:
        print(*msg, file=sys.stderr, flush=True)


DEFAULT_OUTPUT_STEM = "output"
OVERWRITE_TIMEOUT_SECONDS = 10


def _default_output(fmt: str) -> str:
    """Default output path when -o is not given: output.<fmt> in the cwd."""
    return f"{DEFAULT_OUTPUT_STEM}.{fmt}"


def _prompt_yes_no_timeout(prompt: str, timeout: float, *, default: bool) -> bool:
    """Ask a yes/no question, returning ``default`` if unanswered within
    ``timeout`` seconds. Cross-platform (uses a reader thread, so it works on
    Windows where select() can't watch stdin)."""
    import threading

    box: dict = {"answer": None}

    def _reader() -> None:
        try:
            box["answer"] = input()
        except (EOFError, OSError):
            box["answer"] = ""

    sys.stderr.write(prompt)
    sys.stderr.flush()
    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    t.join(timeout)

    raw = box["answer"]
    if raw is None:  # timed out
        sys.stderr.write("\n")
        return default
    ans = raw.strip().lower()
    if ans in ("y", "yes"):
        return True
    if ans in ("n", "no"):
        return False
    return default


def _confirm_overwrite(path: str, *, assume_yes: bool, quiet: bool) -> bool:
    """Decide whether to (over)write ``path``.

    New files: always yes. Existing files: yes if --yes; otherwise prompt with a
    10-second timeout that defaults to overwrite. In a non-interactive session
    (no TTY) we can't prompt, so we overwrite after noting it.
    """
    if assume_yes or not Path(path).exists():
        return True
    if not sys.stdin.isatty():
        _log(quiet, f"note: {path} exists; overwriting (non-interactive session).")
        return True
    return _prompt_yes_no_timeout(
        f"{path} already exists. Overwrite? [Y/n] "
        f"(auto-overwrite in {OVERWRITE_TIMEOUT_SECONDS}s): ",
        OVERWRITE_TIMEOUT_SECONDS, default=True,
    )


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


def _thresholds(args) -> "object":
    from .quality import Thresholds
    return Thresholds(
        min_speech=args.min_speech,
        max_compression=args.max_compression,
        min_logprob=args.min_logprob,
        max_no_speech=args.max_no_speech,
    )


def _run_assess_only(in_path: Path, args) -> int:
    """Fast, model-free triage: measure loudness/silence and report a verdict."""
    from . import quality
    try:
        stats = audio.probe_levels(in_path)
    except audio.AudioError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 5
    report = quality.assess(audio_stats=stats, thresholds=_thresholds(args))
    print(report.summary(), file=sys.stderr)
    if report.is_bad and args.on_bad == "fail":
        return 12
    return 0


def disable_telemetry() -> None:
    """Turn off Hugging Face usage telemetry so a normal run never phones home.

    plaude-local is local-first: the only network touch is a one-time model
    download. huggingface_hub otherwise emits anonymous telemetry on model loads,
    so disable it by default (both the hub-specific and generic switches). Uses
    setdefault, so a user who explicitly opts in is still respected.
    """
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("DISABLE_TELEMETRY", "1")


def apply_offline(enabled: bool) -> None:
    """Force Hugging Face model loads to use only the local cache.

    Sets ``HF_HUB_OFFLINE`` (faster-whisper / pyannote weights) and
    ``TRANSFORMERS_OFFLINE`` (whisperx) so no network request is made — the
    "download once online, then run fully offline" flow. A no-op when disabled,
    and it never clears the variables if the user already set them.
    """
    if not enabled:
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _run_dashboard(args, segments, meta, whisper_translate_segs) -> int:
    """Build + write the HTML dashboard (default output): summary + tabs."""
    from . import dashboard
    from . import summarize as summ

    transcript_text = formats.to_text(segments)
    source_lang = (meta.get("language") or "").lower()
    target = (args.translate_to or "en").lower()

    # Translation for the "Translated-<lang>" tab, plus time-aligned Side-by-Side
    # rows (pairs). English uses Whisper's segments so rows line up; a non-English
    # LLM translation is prose, so the Side-by-Side falls back to a single block.
    pairs = None
    if target == source_lang:
        # Target language == source: no translation needed.
        translation_label = dashboard.language_name(target) if target else "Original"
        translation_text = transcript_text
        pairs = dashboard.align_segments(segments, segments)
    elif target == "en":
        translation_label = "English"
        translation_text = formats.to_text(whisper_translate_segs or [])
        pairs = dashboard.align_segments(segments, whisper_translate_segs or [])
    else:
        translation_label = dashboard.language_name(target)
        _log(args.quiet, f"      translating to {translation_label} (LLM) ...")
        try:
            translation_text = summ.translate(
                transcript_text, target_language=translation_label,
                backend=args.summarize_backend, model=args.summarize_model,
                url=args.summarize_url, max_chars=args.summarize_max_chars)
        except summ.SummarizeError as exc:
            translation_text = ""
            _log(args.quiet, f"      translation unavailable: {exc}")

    # <=250-word critical-topics summary (graceful when no LLM is reachable).
    summary_text, summary_note = None, None
    _log(args.quiet, "      summarizing critical topics ...")
    try:
        summary_text = summ.summarize(
            transcript_text, topics=True, max_words=250,
            backend=args.summarize_backend, model=args.summarize_model,
            url=args.summarize_url, max_chars=args.summarize_max_chars)
    except summ.SummarizeError as exc:
        summary_note = f"Summary unavailable ({exc})."

    speech_s = sum(max(0.0, float(s.get("end", 0)) - float(s.get("start", 0)))
                   for s in segments) or meta.get("duration")
    html_doc = dashboard.build_dashboard_html(
        title=f"{Path(args.input).stem} — transcript dashboard",
        language=meta.get("language"), speech_duration_s=speech_s,
        word_count=dashboard.count_words(transcript_text),
        summary=summary_text, summary_note=summary_note,
        transcript=transcript_text, translation=translation_text,
        translation_label=f"Translated-{translation_label}",
        cjk=source_lang in ("zh", "ja", "ko"), pairs=pairs)

    out = args.output if args.output else _default_output("html")
    if out != "-" and not _confirm_overwrite(out, assume_yes=args.yes, quiet=args.quiet):
        print(f"aborted: {out} was not overwritten.", file=sys.stderr)
        return 11
    try:
        _write_utf8(html_doc, out)
    except OSError as exc:
        print(f"error: could not write dashboard to {out}: {exc}", file=sys.stderr)
        return 9
    if out != "-":
        _log(args.quiet, f"[4/4] dashboard -> {out}")

    if args.split_outputs or args.transcription_file or args.translation_file:
        tfile = args.transcription_file or "transcription.txt"
        xfile = args.translation_file or "translation.txt"
        try:
            _write_utf8(transcript_text.rstrip("\n") + "\n", tfile)
            _write_utf8((translation_text or "").rstrip("\n") + "\n", xfile)
        except OSError as exc:
            print(f"error: could not write split outputs: {exc}", file=sys.stderr)
            return 9
        _log(args.quiet, f"      transcription -> {tfile}   translation -> {xfile}")
    return 0


def run(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    disable_telemetry()  # local-first: no HF telemetry on any run

    if args.check:
        return _run_check()

    apply_offline(args.offline)

    if not args.input:
        print("error: an input audio file is required (or use --check to verify "
              "your setup).", file=sys.stderr)
        return 2

    in_path = Path(args.input)
    if not in_path.is_file():
        print(f"error: input file not found: {in_path}", file=sys.stderr)
        return 2

    # Thorough prerequisite gate: ensure ffmpeg + ffprobe are available, and if
    # not, warn and offer to install them / accept a path, else abort.
    from . import provision
    if not provision.ensure_ffmpeg(
            ffmpeg_location=args.ffmpeg_location,
            install_missing=args.install_missing,
            no_provision=args.no_provision,
            quiet=args.quiet):
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

    # Fast, model-free triage: report a verdict and exit without transcribing.
    if args.assess_only:
        return _run_assess_only(in_path, args)

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

    whisper_translate_segs = None  # English translation segments (dashboard only)
    with tempfile.TemporaryDirectory(prefix="plaude-local-") as tmp:
        # 1. Preprocess / denoise / enhance
        _log(args.quiet,
             f"[1/4] preparing audio (denoise={args.denoise}, "
             f"enhance={args.enhance}, gain={args.gain}dB) ...")
        try:
            prepared = audio.prepare(in_path, tmp, denoise=args.denoise,
                                     enhance=args.enhance, gain_db=args.gain)
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

        # For the HTML dashboard's English tab, run Whisper's native translate
        # task (a second decode pass) unless the source is already English or a
        # non-English target was requested (that goes through the LLM later).
        if (args.format == "html" and args.translate_to == "en"
                and (meta.get("language") or "").lower() != "en"):
            _log(args.quiet, "      translating to English (Whisper) ...")
            try:
                _tr_segs, _ = engine.transcribe(
                    str(prepared), language=args.language,
                    vad_filter=not args.no_vad, beam_size=args.beam_size,
                    task="translate")
                whisper_translate_segs = _tr_segs
            except transcribe.TranscribeError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 6

        # Assess quality from the RAW segments (before diarization rebuilds
        # them and drops the per-segment metrics).
        from . import quality
        quality_report = quality.assess(
            segments=segments, duration=meta.get("duration"),
            thresholds=_thresholds(args))
        _log(args.quiet, "      " + quality_report.summary())

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
                    diarize_model=args.diarize_model,
                )
            except diarize.DiarizeError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 7
        else:
            _log(args.quiet, "[3/4] diarization skipped")

    # A bad recording triggers the --on-bad policy (default: warn and continue).
    if quality_report.is_bad:
        if args.on_bad == "skip":
            print(f"skipped: bad recording - {quality_report.summary()}",
                  file=sys.stderr)
            return 0
        if args.on_bad == "fail":
            print(f"error: bad recording - {quality_report.summary()}",
                  file=sys.stderr)
            return 12
        print(f"warning: {quality_report.summary()} (writing anyway; "
              "use --on-bad to change)", file=sys.stderr)

    # 4a. HTML dashboard (default format): summary + Transcribed / Translated /
    # Side-by-Side tabs, plus optional split text files.
    if args.format == "html":
        return _run_dashboard(args, segments, meta, whisper_translate_segs)

    # 4. Render + write transcript (UTF-8)
    # "timestamps" is an internal rendering hint for the txt writer (show a clock
    # when speakers are present). Only inject it for txt so it never leaks into
    # the user-facing JSON meta block. The quality report is surfaced in JSON.
    if args.format == "txt":
        meta["timestamps"] = args.diarize
    meta["quality"] = quality_report.as_dict()
    text = formats.render(segments, args.format, meta=meta)
    transcript_out = args.output if args.output else _default_output(args.format)
    if transcript_out != "-" and not _confirm_overwrite(
            transcript_out, assume_yes=args.yes, quiet=args.quiet):
        print(f"aborted: {transcript_out} was not overwritten.", file=sys.stderr)
        return 11
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
