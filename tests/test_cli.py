"""Regression tests for the CLI: argument parsing, error codes, orchestration."""

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
        self.assertEqual(args.format, "txt")
        self.assertEqual(args.diarize_backend, "pyannote")
        self.assertFalse(args.diarize)

    def test_backend_choice(self):
        args = cli.build_parser().parse_args(
            ["in.wav", "--diarize", "--diarize-backend", "whisperx"]
        )
        self.assertTrue(args.diarize)
        self.assertEqual(args.diarize_backend, "whisperx")


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
                rc = cli.run([str(f), "-o", str(out), "--denoise", "none", "-q"])
            self.assertEqual(rc, 0)
            self.assertEqual(out.read_text(encoding="utf-8"), "Hello.\nWorld.\n")

    def test_default_output_path_next_to_input(self):
        segs = [{"start": 0.0, "end": 1.0, "text": "hi", "speaker": None}]
        with tempfile.TemporaryDirectory() as d:
            f = self._input(d)
            with mock.patch.object(audio, "have_ffmpeg", return_value=True), \
                 mock.patch.object(audio, "prepare", return_value=f), \
                 mock.patch.object(
                     transcribe, "Transcriber",
                     lambda **kw: _FakeEngine(segs, **kw)):
                rc = cli.run([str(f), "--denoise", "none", "-q"])
            self.assertEqual(rc, 0)
            self.assertTrue((pathlib.Path(d) / "rec.txt").is_file())

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
                rc = cli.run([str(f), "-o", str(out), "--denoise", "none", "-q"])
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
                rc = cli.run([str(f), "-o", "-", "--denoise", "none", "-q"])
            self.assertEqual(rc, 0)
            emitted = fake.buffer.getvalue()
            self.assertIsInstance(emitted, bytes)
            self.assertIn("你好世界", emitted.decode("utf-8"))

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
                rc = cli.run([str(f), "-o", str(bad), "--denoise", "none", "-q"])
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
                rc = cli.run([str(f), "-o", str(out), "--denoise", "none",
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
                rc = cli.run([str(f), "--denoise", "none", "--diarize",
                              "--hf-token", "t", "-o", str(out), "-q"])
            self.assertEqual(rc, 0)
            self.assertIn("Speaker 1:", out.read_text(encoding="utf-8"))
            # backend + device were forwarded
            self.assertEqual(m.call_args.kwargs["backend"], "pyannote")
            self.assertEqual(m.call_args.kwargs["device"], "cpu")


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
                rc = cli.run([str(f), "-o", str(out), "--denoise", "none", "-q"])
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


if __name__ == "__main__":
    unittest.main()
