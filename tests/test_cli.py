"""Regression tests for the CLI: argument parsing, error codes, orchestration."""

import contextlib
import io
import os
import pathlib
import tempfile
import unittest
from unittest import mock

from plaude_local import cli, audio, transcribe, diarize


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


class TestErrorCodes(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
