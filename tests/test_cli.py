"""Regression tests for the CLI: argument parsing, error codes, orchestration."""
import json

import contextlib
import io
import os
import pathlib
import tempfile
import unittest
from unittest import mock

from plaude_local import cli, audio, transcribe, diarize, summarize, preflight
from plaude_local.preflight import Check


def _run_quiet(argv):
    """Run the CLI, swallowing its stderr diagnostics for clean test output."""
    with contextlib.redirect_stderr(io.StringIO()):
        return cli.run(argv)


class _FakeEngine:
    """Stand-in for Transcriber: returns canned segments without a model."""

    def __init__(self, segments, **_kwargs):
        self._segments = segments
        self.device = "cpu"
        self.compute_type = "int8"

    def transcribe(self, path, **_kwargs):
        meta = {"language": "en", "model": "fake", "device": "cpu"}
        # Return copies so the caller can mutate freely between tests.
        return [dict(s) for s in self._segments], meta


class TestParser(unittest.TestCase):
    def test_defaults(self):
        args = cli.build_parser().parse_args(["in.wav"])
        self.assertEqual(args.model, "large-v3")
        self.assertEqual(args.device, "auto")
        self.assertEqual(args.denoise, "ffmpeg")
        self.assertEqual(args.format, "html")
        self.assertEqual(args.translate_to, "en")
        self.assertEqual(args.diarize_backend, "pyannote")
        self.assertFalse(args.diarize)

    def test_backend_choice(self):
        args = cli.build_parser().parse_args(
            ["in.wav", "--diarize", "--diarize-backend", "whisperx"]
        )
        self.assertTrue(args.diarize)
        self.assertEqual(args.diarize_backend, "whisperx")

    def test_offline_defaults_false(self):
        self.assertFalse(cli.build_parser().parse_args(["in.wav"]).offline)
        self.assertTrue(
            cli.build_parser().parse_args(["in.wav", "--offline"]).offline
        )


class TestOffline(unittest.TestCase):
    def setUp(self):
        self._saved = {
            k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
        }
        for k in self._saved:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_disabled_sets_nothing(self):
        cli.apply_offline(False)
        self.assertNotIn("HF_HUB_OFFLINE", os.environ)
        self.assertNotIn("TRANSFORMERS_OFFLINE", os.environ)

    def test_enabled_sets_both_offline_flags(self):
        cli.apply_offline(True)
        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
        self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")


class TestCheck(unittest.TestCase):
    def test_check_all_ok_returns_zero(self):
        ok = [Check("Python", True, True), Check("FFmpeg", True, True)]
        with mock.patch.object(preflight, "run_all", return_value=ok):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.run(["--check"]), 0)

    def test_check_missing_required_returns_one(self):
        bad = [Check("FFmpeg", False, True, remedy="install it")]
        with mock.patch.object(preflight, "run_all", return_value=bad):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = cli.run(["--check"])
        self.assertEqual(rc, 1)
        self.assertIn("install it", buf.getvalue())


class TestErrorCodes(unittest.TestCase):
    def setUp(self):
        # Keep the ffprobe stream-check out of the way by default; tests that
        # exercise probing live in TestInputProbing.
        p = mock.patch.object(audio, "have_ffprobe", return_value=False)
        p.start()
        self.addCleanup(p.stop)

    def test_input_required_without_check(self):
        self.assertEqual(_run_quiet([]), 2)

    def test_input_not_found(self):
        self.assertEqual(_run_quiet(["/no/such/input.wav"]), 2)

    def test_ffmpeg_missing(self):
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "a.wav"
            f.write_bytes(b"x")
            with mock.patch.object(audio, "have_ffmpeg", return_value=False):
                self.assertEqual(_run_quiet([str(f)]), 3)

    def test_diarize_without_token(self):
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "a.wav"
            f.write_bytes(b"x")
            with mock.patch.object(audio, "have_ffmpeg", return_value=True), \
                 mock.patch.dict(os.environ, {}, clear=True):
                self.assertEqual(_run_quiet([str(f), "--diarize"]), 4)


class TestOrchestration(unittest.TestCase):
    def setUp(self):
        # Deterministic regardless of whether ffprobe is installed on the host:
        # skip the stream probe here and test it explicitly in TestInputProbing.
        p = mock.patch.object(audio, "have_ffprobe", return_value=False)
        p.start()
        self.addCleanup(p.stop)

    def _input(self, d):
        f = pathlib.Path(d) / "rec.wav"
        f.write_bytes(b"x")
        return f

    def test_enhance_and_gain_forwarded_to_prepare(self):
        segs = [{"start": 0.0, "end": 1.0, "text": "hi", "speaker": None}]
        with tempfile.TemporaryDirectory() as d:
            f = self._input(d)
            out = pathlib.Path(d) / "o.txt"
            with mock.patch.object(audio, "have_ffmpeg", return_value=True), \
                 mock.patch.object(audio, "prepare", return_value=f) as prep, \
                 mock.patch.object(
                     transcribe, "Transcriber",
                     lambda **kw: _FakeEngine(segs, **kw)):
                rc = cli.run([str(f), "-o", str(out), "--denoise", "none", "-f", "txt",
                              "--enhance", "speech", "--gain", "6", "-q"])
            self.assertEqual(rc, 0)
            self.assertEqual(prep.call_args.kwargs["enhance"], "speech")
            self.assertEqual(prep.call_args.kwargs["gain_db"], 6.0)

    def _capture_device(self, device):
        """Run the CLI in a given device mode and return (rc, device kwarg)."""
        segs = [{"start": 0.0, "end": 1.0, "text": "hi", "speaker": None}]
        captured = {}

        def _factory(**kw):
            captured.update(kw)
            return _FakeEngine(segs, **kw)

        with tempfile.TemporaryDirectory() as d:
            f = self._input(d)
            out = pathlib.Path(d) / "o.txt"
            with mock.patch.object(audio, "have_ffmpeg", return_value=True), \
                 mock.patch.object(audio, "prepare", return_value=f), \
                 mock.patch.object(transcribe, "Transcriber", _factory):
                rc = cli.run([str(f), "-o", str(out), "--denoise", "none", "-f", "txt",
                              "--device", device, "-q"])
        return rc, captured.get("device")

    def test_cpu_only_mode_forwards_device(self):
        rc, device = self._capture_device("cpu")
        self.assertEqual(rc, 0)
        self.assertEqual(device, "cpu")

    def test_gpu_only_mode_forwards_device(self):
        rc, device = self._capture_device("cuda")
        self.assertEqual(rc, 0)
        self.assertEqual(device, "cuda")

    def test_happy_path_writes_transcript(self):
        segs = [
            {"start": 0.0, "end": 1.0, "text": " Hello.", "speaker": None},
            {"start": 1.0, "end": 2.0, "text": " World.", "speaker": None},
        ]
        with tempfile.TemporaryDirectory() as d:
            f = self._input(d)
            out = pathlib.Path(d) / "out.txt"
            with mock.patch.object(audio, "have_ffmpeg", return_value=True), \
                 mock.patch.object(audio, "prepare", return_value=f), \
                 mock.patch.object(
                     transcribe, "Transcriber",
                     lambda **kw: _FakeEngine(segs, **kw)):
                rc = cli.run([str(f), "-o", str(out), "--denoise", "none", "-f", "txt", "-q"])
            self.assertEqual(rc, 0)
            self.assertEqual(out.read_text(encoding="utf-8"), "Hello.\nWorld.\n")

    def test_default_output_is_output_dot_txt_in_cwd(self):
        segs = [{"start": 0.0, "end": 1.0, "text": "hi", "speaker": None}]
        with tempfile.TemporaryDirectory() as d:
            f = self._input(d)
            cwd = os.getcwd()
            os.chdir(d)
            try:
                with mock.patch.object(audio, "have_ffmpeg", return_value=True), \
                     mock.patch.object(audio, "prepare", return_value=f), \
                     mock.patch.object(
                         transcribe, "Transcriber",
                         lambda **kw: _FakeEngine(segs, **kw)):
                    rc = cli.run([str(f), "--denoise", "none", "-f", "txt", "-q"])
            finally:
                os.chdir(cwd)
            self.assertEqual(rc, 0)
            self.assertTrue((pathlib.Path(d) / "output.txt").is_file())

    def test_default_output_matches_format(self):
        segs = [{"start": 0.0, "end": 1.0, "text": "hi", "speaker": None}]
        with tempfile.TemporaryDirectory() as d:
            f = self._input(d)
            cwd = os.getcwd()
            os.chdir(d)
            try:
                with mock.patch.object(audio, "have_ffmpeg", return_value=True), \
                     mock.patch.object(audio, "prepare", return_value=f), \
                     mock.patch.object(
                         transcribe, "Transcriber",
                         lambda **kw: _FakeEngine(segs, **kw)):
                    rc = cli.run([str(f), "--denoise", "none", "-f", "srt", "-q"])
            finally:
                os.chdir(cwd)
            self.assertEqual(rc, 0)
            self.assertTrue((pathlib.Path(d) / "output.srt").is_file())

    def test_output_is_utf8_for_non_latin_scripts(self):
        # Chinese + Japanese must round-trip as UTF-8 text.
        segs = [
            {"start": 0.0, "end": 1.0, "text": " 你好世界", "speaker": None},
            {"start": 1.0, "end": 2.0, "text": " こんにちは", "speaker": None},
        ]
        with tempfile.TemporaryDirectory() as d:
            f = self._input(d)
            out = pathlib.Path(d) / "cn.txt"
            with mock.patch.object(audio, "have_ffmpeg", return_value=True), \
                 mock.patch.object(audio, "prepare", return_value=f), \
                 mock.patch.object(
                     transcribe, "Transcriber",
                     lambda **kw: _FakeEngine(segs, **kw)):
                rc = cli.run([str(f), "-o", str(out), "--denoise", "none", "-f", "txt", "-q"])
            self.assertEqual(rc, 0)
            # Decoding as UTF-8 must succeed and preserve the characters.
            content = out.read_bytes().decode("utf-8")
            self.assertIn("你好世界", content)
            self.assertIn("こんにちは", content)

    def test_stdout_output_is_utf8_bytes(self):
        # Regression: the sys.stdout.buffer path must emit real UTF-8 bytes so
        # CJK survives on any console code page. StringIO has no .buffer, so we
        # supply a stdout stand-in that exposes a binary buffer.
        class _FakeStdout:
            def __init__(self):
                self.buffer = io.BytesIO()

        segs = [{"start": 0.0, "end": 1.0, "text": " 你好世界", "speaker": None}]
        with tempfile.TemporaryDirectory() as d:
            f = self._input(d)
            fake = _FakeStdout()
            with mock.patch.object(audio, "have_ffmpeg", return_value=True), \
                 mock.patch.object(audio, "prepare", return_value=f), \
                 mock.patch.object(
                     transcribe, "Transcriber",
                     lambda **kw: _FakeEngine(segs, **kw)), \
                 mock.patch("sys.stdout", fake):
                rc = cli.run([str(f), "-o", "-", "--denoise", "none", "-f", "txt", "-q"])
            self.assertEqual(rc, 0)
            emitted = fake.buffer.getvalue()
            self.assertIsInstance(emitted, bytes)
            self.assertIn("你好世界", emitted.decode("utf-8"))

    def test_json_output_has_no_internal_timestamps_key(self):
        # Regression: the internal "timestamps" rendering hint must not leak into
        # the user-facing JSON meta block.
        import json
        segs = [{"start": 0.0, "end": 1.0, "text": "hi", "speaker": None}]
        with tempfile.TemporaryDirectory() as d:
            f = self._input(d)
            out = pathlib.Path(d) / "o.json"
            with mock.patch.object(audio, "have_ffmpeg", return_value=True), \
                 mock.patch.object(audio, "prepare", return_value=f), \
                 mock.patch.object(
                     transcribe, "Transcriber",
                     lambda **kw: _FakeEngine(segs, **kw)):
                rc = cli.run([str(f), "-o", str(out), "-f", "json",
                              "--denoise", "none", "-q"])
            self.assertEqual(rc, 0)
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertNotIn("timestamps", data.get("meta", {}))

    def test_keep_clean_copies_prepared_audio(self):
        segs = [{"start": 0.0, "end": 1.0, "text": "hi", "speaker": None}]
        with tempfile.TemporaryDirectory() as d:
            f = self._input(d)
            out = pathlib.Path(d) / "o.txt"
            clean = pathlib.Path(d) / "cleaned.wav"
            with mock.patch.object(audio, "have_ffmpeg", return_value=True), \
                 mock.patch.object(audio, "prepare", return_value=f), \
                 mock.patch.object(
                     transcribe, "Transcriber",
                     lambda **kw: _FakeEngine(segs, **kw)):
                rc = cli.run([str(f), "-o", str(out), "--denoise", "none", "-f", "txt",
                              "--keep-clean", str(clean), "-q"])
            self.assertEqual(rc, 0)
            self.assertTrue(clean.is_file())

    def test_summary_to_stdout(self):
        # --summary-output - streams the summary to stdout instead of a file.
        segs = [{"start": 0.0, "end": 1.0, "text": "some talk", "speaker": None}]
        with tempfile.TemporaryDirectory() as d:
            f = self._input(d)
            out = pathlib.Path(d) / "o.txt"
            buf = io.StringIO()
            with mock.patch.object(audio, "have_ffmpeg", return_value=True), \
                 mock.patch.object(audio, "prepare", return_value=f), \
                 mock.patch.object(
                     transcribe, "Transcriber",
                     lambda **kw: _FakeEngine(segs, **kw)), \
                 mock.patch.object(summarize, "detect_backend", return_value="ollama"), \
                 mock.patch.object(summarize, "summarize", return_value="* key point"), \
                 contextlib.redirect_stdout(buf):
                rc = cli.run([str(f), "-o", str(out), "--denoise", "none", "-f", "txt",
                              "--summarize", "--summary-output", "-", "-q"])
            self.assertEqual(rc, 0)
            self.assertIn("* key point", buf.getvalue())
            # No summary file should be written when streaming to stdout.
            self.assertFalse((pathlib.Path(d) / "o.summary.md").exists())

    def test_bad_output_dir_fails_cleanly(self):
        segs = [{"start": 0.0, "end": 1.0, "text": "hi", "speaker": None}]
        with tempfile.TemporaryDirectory() as d:
            f = self._input(d)
            bad = pathlib.Path(d) / "no_such_dir" / "out.txt"
            with mock.patch.object(audio, "have_ffmpeg", return_value=True), \
                 mock.patch.object(audio, "prepare", return_value=f), \
                 mock.patch.object(
                     transcribe, "Transcriber",
                     lambda **kw: _FakeEngine(segs, **kw)), \
                 contextlib.redirect_stderr(io.StringIO()):
                rc = cli.run([str(f), "-o", str(bad), "--denoise", "none", "-f", "txt", "-q"])
            self.assertEqual(rc, 9)

    def test_summarize_writes_summary_file(self):
        segs = [{"start": 0.0, "end": 1.0, "text": "some talk", "speaker": None}]
        with tempfile.TemporaryDirectory() as d:
            f = self._input(d)
            out = pathlib.Path(d) / "o.txt"
            with mock.patch.object(audio, "have_ffmpeg", return_value=True), \
                 mock.patch.object(audio, "prepare", return_value=f), \
                 mock.patch.object(
                     transcribe, "Transcriber",
                     lambda **kw: _FakeEngine(segs, **kw)), \
                 mock.patch.object(summarize, "detect_backend", return_value="ollama"), \
                 mock.patch.object(summarize, "summarize",
                                   return_value="* key point") as m:
                rc = cli.run([str(f), "-o", str(out), "--denoise", "none", "-f", "txt",
                              "--summarize", "-q"])
            self.assertEqual(rc, 0)
            summary = pathlib.Path(d) / "o.summary.md"
            self.assertTrue(summary.is_file())
            body = summary.read_text(encoding="utf-8")
            self.assertIn("# Summary", body)
            self.assertIn("* key point", body)
            m.assert_called_once()

    def test_summarize_no_server_fails_fast(self):
        segs = [{"start": 0.0, "end": 1.0, "text": "x", "speaker": None}]
        with tempfile.TemporaryDirectory() as d:
            f = self._input(d)
            with mock.patch.object(audio, "have_ffmpeg", return_value=True), \
                 mock.patch.object(summarize, "detect_backend", return_value=None), \
                 contextlib.redirect_stderr(io.StringIO()):
                rc = cli.run([str(f), "--denoise", "none", "--summarize", "-q"])
            self.assertEqual(rc, 8)

    def test_diarize_path_tags_speakers(self):
        segs = [{"start": 0.0, "end": 1.0, "text": " hi", "speaker": None}]
        tagged = [{"start": 0.0, "end": 1.0, "text": "hi", "speaker": "Speaker 1"}]
        with tempfile.TemporaryDirectory() as d:
            f = self._input(d)
            out = pathlib.Path(d) / "out.txt"
            with mock.patch.object(audio, "have_ffmpeg", return_value=True), \
                 mock.patch.object(audio, "prepare", return_value=f), \
                 mock.patch.object(
                     transcribe, "Transcriber",
                     lambda **kw: _FakeEngine(segs, **kw)), \
                 mock.patch.object(diarize, "diarize_and_merge",
                                   return_value=tagged) as m:
                rc = cli.run([str(f), "--denoise", "none", "--diarize", "-f", "txt",
                              "--hf-token", "t", "--diarize-model", "pyannote/x",
                              "-o", str(out), "-q"])
            self.assertEqual(rc, 0)
            self.assertIn("Speaker 1:", out.read_text(encoding="utf-8"))
            # backend + device + model were forwarded
            self.assertEqual(m.call_args.kwargs["backend"], "pyannote")
            self.assertEqual(m.call_args.kwargs["device"], "cpu")
            self.assertEqual(m.call_args.kwargs["diarize_model"], "pyannote/x")


class TestPromptTimeout(unittest.TestCase):
    def setUp(self):
        p = contextlib.redirect_stderr(io.StringIO())
        p.__enter__()
        self.addCleanup(p.__exit__, None, None, None)

    def test_yes_answer(self):
        with mock.patch("builtins.input", return_value="y"):
            self.assertTrue(cli._prompt_yes_no_timeout("? ", 1, default=False))

    def test_no_answer(self):
        with mock.patch("builtins.input", return_value="n"):
            self.assertFalse(cli._prompt_yes_no_timeout("? ", 1, default=True))

    def test_blank_uses_default(self):
        with mock.patch("builtins.input", return_value=""):
            self.assertTrue(cli._prompt_yes_no_timeout("? ", 1, default=True))

    def test_timeout_uses_default(self):
        import time

        def _slow():
            time.sleep(1.0)
            return "n"
        # Reader blocks past the (tiny) timeout -> default is returned promptly.
        with mock.patch("builtins.input", side_effect=_slow):
            self.assertTrue(
                cli._prompt_yes_no_timeout("? ", 0.05, default=True))


class TestOverwrite(unittest.TestCase):
    def _run_with_existing_output(self, extra_args, prompt_return=None):
        segs = [{"start": 0.0, "end": 1.0, "text": "new", "speaker": None}]
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "rec.wav"
            f.write_bytes(b"x")
            out = pathlib.Path(d) / "o.txt"
            out.write_text("OLD CONTENT", encoding="utf-8")
            patches = [
                mock.patch.object(audio, "have_ffmpeg", return_value=True),
                mock.patch.object(audio, "have_ffprobe", return_value=False),
                mock.patch.object(audio, "prepare", return_value=f),
                mock.patch.object(transcribe, "Transcriber",
                                  lambda **kw: _FakeEngine(segs, **kw)),
                contextlib.redirect_stderr(io.StringIO()),
            ]
            if prompt_return is not None:
                # Force an interactive session and stub the actual prompt.
                fake_stdin = mock.Mock()
                fake_stdin.isatty.return_value = True
                patches.append(mock.patch("sys.stdin", fake_stdin))
                patches.append(mock.patch.object(
                    cli, "_prompt_yes_no_timeout", return_value=prompt_return))
            with contextlib.ExitStack() as stack:
                for p in patches:
                    stack.enter_context(p)
                rc = cli.run([str(f), "-o", str(out), "--denoise", "none", "-f", "txt", "-q"]
                             + extra_args)
            return rc, out.read_text(encoding="utf-8")

    def test_yes_flag_overwrites_without_prompt(self):
        rc, content = self._run_with_existing_output(["--yes"])
        self.assertEqual(rc, 0)
        self.assertEqual(content, "new\n")

    def test_prompt_yes_overwrites(self):
        rc, content = self._run_with_existing_output([], prompt_return=True)
        self.assertEqual(rc, 0)
        self.assertEqual(content, "new\n")

    def test_prompt_no_aborts_and_keeps_file(self):
        rc, content = self._run_with_existing_output([], prompt_return=False)
        self.assertEqual(rc, 11)
        self.assertEqual(content, "OLD CONTENT")  # untouched


class TestInputProbing(unittest.TestCase):
    """FFmpeg (via ffprobe) is the sole arbiter of what input is decodable."""

    _SEGS = [{"start": 0.0, "end": 1.0, "text": "hi", "speaker": None}]

    def _run(self, filename, *, ffprobe, codec):
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / filename
            f.write_bytes(b"x")
            out = pathlib.Path(d) / "o.txt"
            with mock.patch.object(audio, "have_ffmpeg", return_value=True), \
                 mock.patch.object(audio, "have_ffprobe", return_value=ffprobe), \
                 mock.patch.object(audio, "probe_audio_codec", return_value=codec), \
                 mock.patch.object(audio, "prepare", return_value=f), \
                 mock.patch.object(
                     transcribe, "Transcriber",
                     lambda **kw: _FakeEngine(self._SEGS, **kw)), \
                 contextlib.redirect_stderr(io.StringIO()):
                rc = cli.run([str(f), "-o", str(out), "--denoise", "none", "-f", "txt", "-q"])
            return rc, out.is_file()

    def test_any_extension_accepted_when_ffprobe_finds_audio(self):
        # Extension is irrelevant: if FFmpeg reports an audio codec, we proceed.
        for name in ("clip.m4a", "clip.opus", "movie.mkv", "weird.qqq",
                     "no_extension"):
            rc, made = self._run(name, ffprobe=True, codec="aac")
            self.assertEqual(rc, 0, name)
            self.assertTrue(made, name)

    def test_no_audio_stream_returns_error(self):
        rc, made = self._run("video-without-audio.mp4", ffprobe=True, codec=None)
        self.assertEqual(rc, 10)
        self.assertFalse(made)

    def test_missing_ffprobe_skips_probe_and_proceeds(self):
        # Without ffprobe we can't pre-check; we still attempt the decode.
        rc, made = self._run("clip.weirdext", ffprobe=False, codec=None)
        self.assertEqual(rc, 0)
        self.assertTrue(made)


class TestQualityGate(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(audio, "have_ffprobe", return_value=False)
        p.start()
        self.addCleanup(p.stop)

    def _input(self, d):
        f = pathlib.Path(d) / "rec.wav"
        f.write_bytes(b"x")
        return f

    _BAD = [{"start": 0.0, "end": 0.0, "text": "", "speaker": None,
             "no_speech_prob": 0.99, "avg_logprob": -2.0, "compression_ratio": 3.0}]

    def test_assess_only_warn_returns_zero(self):
        with tempfile.TemporaryDirectory() as d:
            f = self._input(d)
            bad_stats = {"mean_volume_db": -62.0, "max_volume_db": -40.0,
                         "silence_ratio": 0.99}
            err = io.StringIO()
            with mock.patch.object(audio, "have_ffmpeg", return_value=True), \
                 mock.patch.object(audio, "probe_levels", return_value=bad_stats), \
                 contextlib.redirect_stderr(err):
                rc = cli.run([str(f), "--assess-only", "-q"])
            self.assertEqual(rc, 0)
            self.assertIn("BAD", err.getvalue())

    def test_assess_only_fail_returns_12(self):
        with tempfile.TemporaryDirectory() as d:
            f = self._input(d)
            bad_stats = {"mean_volume_db": -62.0, "silence_ratio": 0.99}
            with mock.patch.object(audio, "have_ffmpeg", return_value=True), \
                 mock.patch.object(audio, "probe_levels", return_value=bad_stats), \
                 contextlib.redirect_stderr(io.StringIO()):
                rc = cli.run([str(f), "--assess-only", "--on-bad", "fail", "-q"])
            self.assertEqual(rc, 12)

    def _run_bad(self, extra):
        with tempfile.TemporaryDirectory() as d:
            f = self._input(d)
            out = pathlib.Path(d) / "o.txt"
            err = io.StringIO()
            with mock.patch.object(audio, "have_ffmpeg", return_value=True), \
                 mock.patch.object(audio, "prepare", return_value=f), \
                 mock.patch.object(
                     transcribe, "Transcriber",
                     lambda **kw: _FakeEngine(self._BAD, **kw)), \
                 contextlib.redirect_stderr(err):
                rc = cli.run([str(f), "-o", str(out), "--denoise", "none", "-f", "txt"]
                             + extra + ["-q"])
            return rc, out.exists(), err.getvalue()

    def test_on_bad_warn_default_still_writes(self):
        rc, exists, err = self._run_bad([])
        self.assertEqual(rc, 0)
        self.assertTrue(exists)
        self.assertIn("warning", err.lower())

    def test_on_bad_skip_writes_nothing(self):
        rc, exists, _ = self._run_bad(["--on-bad", "skip"])
        self.assertEqual(rc, 0)
        self.assertFalse(exists)

    def test_on_bad_fail_returns_12_and_writes_nothing(self):
        rc, exists, _ = self._run_bad(["--on-bad", "fail"])
        self.assertEqual(rc, 12)
        self.assertFalse(exists)

    def test_quality_in_json_meta(self):
        segs = [{"start": 0.0, "end": 9.0, "text": "clear speech here",
                 "speaker": None, "no_speech_prob": 0.03, "avg_logprob": -0.3,
                 "compression_ratio": 1.4}]
        with tempfile.TemporaryDirectory() as d:
            f = self._input(d)
            out = pathlib.Path(d) / "o.json"
            with mock.patch.object(audio, "have_ffmpeg", return_value=True), \
                 mock.patch.object(audio, "prepare", return_value=f), \
                 mock.patch.object(
                     transcribe, "Transcriber",
                     lambda **kw: _FakeEngine(segs, **kw)):
                rc = cli.run([str(f), "-o", str(out), "--denoise", "none",
                              "-f", "json", "-q"])
            self.assertEqual(rc, 0)
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertIn("quality", data["meta"])
            self.assertEqual(data["meta"]["quality"]["verdict"], "ok")


class _DashEngine:
    """Fake Transcriber for the dashboard: source language + a translate pass."""

    def __init__(self, src_segs, en_segs, language, **_kw):
        self._src, self._en, self._lang = src_segs, en_segs, language
        self.device, self.compute_type = "cpu", "int8"
        self.passes = 0

    def transcribe(self, path, *, task="transcribe", **_kw):
        self.passes += 1
        segs = self._en if task == "translate" else self._src
        return [dict(s) for s in segs], {"language": self._lang, "duration": 3.0}


class TestDashboard(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(audio, "have_ffprobe", return_value=False)
        p.start()
        self.addCleanup(p.stop)

    def _input(self, d):
        f = pathlib.Path(d) / "rec.wav"
        f.write_bytes(b"x")
        return f

    def _run(self, engine, extra, mock_summary="Key topics here.", summary_error=False):
        with tempfile.TemporaryDirectory() as d:
            f = self._input(d)
            cwd = os.getcwd()
            os.chdir(d)
            try:
                sm = (mock.patch.object(summarize, "summarize",
                                        side_effect=summarize.SummarizeError("no server"))
                      if summary_error else
                      mock.patch.object(summarize, "summarize", return_value=mock_summary))
                with mock.patch.object(audio, "have_ffmpeg", return_value=True), \
                     mock.patch.object(audio, "prepare", return_value=f), \
                     mock.patch.object(transcribe, "Transcriber", lambda **kw: engine), \
                     sm:
                    rc = cli.run([str(f), "-o", "dash.html", "--denoise", "none"] + extra + ["-q"])
                html = ""
                if (pathlib.Path(d) / "dash.html").exists():
                    html = (pathlib.Path(d) / "dash.html").read_text(encoding="utf-8")
                files = {p.name: p.read_text(encoding="utf-8")
                         for p in pathlib.Path(d).glob("*.txt")}
                return rc, html, files
            finally:
                os.chdir(cwd)

    def test_non_english_dashboard_has_transcript_translation_summary(self):
        eng = _DashEngine(
            [{"start": 0.0, "end": 3.0, "text": "你好世界", "speaker": None}],
            [{"start": 0.0, "end": 3.0, "text": "Hello world", "speaker": None}],
            "zh")
        rc, html, _ = self._run(eng, [])
        self.assertEqual(rc, 0)
        self.assertEqual(eng.passes, 2)               # transcribe + Whisper translate
        self.assertIn("你好世界", html)               # Transcribed tab
        self.assertIn("Hello world", html)            # Translated (Whisper)
        self.assertIn("Key topics here.", html)       # summary
        self.assertIn("Translated-English", html)
        self.assertIn("Side-by-Side", html)

    def test_english_source_skips_translate_pass(self):
        eng = _DashEngine(
            [{"start": 0.0, "end": 2.0, "text": "Hello there", "speaker": None}],
            [], "en")
        rc, html, _ = self._run(eng, [])
        self.assertEqual(rc, 0)
        self.assertEqual(eng.passes, 1)               # no separate translate pass
        self.assertIn("Hello there", html)

    def test_split_outputs_writes_transcription_and_translation(self):
        eng = _DashEngine(
            [{"start": 0.0, "end": 3.0, "text": "你好", "speaker": None}],
            [{"start": 0.0, "end": 3.0, "text": "Hello", "speaker": None}], "zh")
        rc, _, files = self._run(eng, ["--split-outputs"])
        self.assertEqual(rc, 0)
        self.assertEqual(files["transcription.txt"].strip(), "你好")
        self.assertEqual(files["translation.txt"].strip(), "Hello")

    def test_summary_degrades_gracefully_without_llm(self):
        eng = _DashEngine(
            [{"start": 0.0, "end": 2.0, "text": "Hello", "speaker": None}], [], "en")
        rc, html, _ = self._run(eng, [], summary_error=True)
        self.assertEqual(rc, 0)
        self.assertIn("Summary unavailable", html)


if __name__ == "__main__":
    unittest.main()
