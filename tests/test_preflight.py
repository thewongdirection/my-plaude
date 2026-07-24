"""Regression tests for the prerequisite checker (plaude_local.preflight)."""

import unittest
from unittest import mock

from plaude_local import preflight
from plaude_local.preflight import Check


class TestHasModule(unittest.TestCase):
    def test_present(self):
        self.assertTrue(preflight._has_module("os"))

    def test_absent_toplevel(self):
        self.assertFalse(preflight._has_module("definitely_not_a_module_xyz"))

    def test_absent_with_missing_parent_does_not_raise(self):
        # Regression: find_spec raises ModuleNotFoundError for a missing parent
        # package; _has_module must swallow it and return False.
        self.assertFalse(preflight._has_module("pyannote.audio"))


class TestIndividualChecks(unittest.TestCase):
    def test_python_ok(self):
        c = preflight.check_python()
        self.assertTrue(c.ok)
        self.assertTrue(c.required)

    def test_ffmpeg_found(self):
        with mock.patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            c = preflight.check_ffmpeg()
        self.assertTrue(c.ok)

    def test_ffmpeg_missing_has_remedy(self):
        with mock.patch("shutil.which", return_value=None):
            c = preflight.check_ffmpeg()
        self.assertFalse(c.ok)
        self.assertTrue(c.required)
        self.assertIn("ffmpeg", c.remedy.lower())

    def test_gpu_check_never_fails(self):
        c = preflight.check_gpu()
        self.assertTrue(c.ok)          # informational only
        self.assertFalse(c.required)

    def test_summarizer_reachable(self):
        with mock.patch("plaude_local.summarize.detect_backend", return_value="ollama"):
            c = preflight.check_summarizer()
        self.assertTrue(c.ok)
        self.assertIn("ollama", c.detail)

    def test_summarizer_unreachable_has_remedy(self):
        with mock.patch("plaude_local.summarize.detect_backend", return_value=None):
            c = preflight.check_summarizer()
        self.assertFalse(c.ok)
        self.assertIn("ollama", c.remedy.lower())


class TestReport(unittest.TestCase):
    def _checks(self, ffmpeg_ok):
        return [
            Check("Python >= 3.9", True, True, "found 3.11"),
            Check("FFmpeg", ffmpeg_ok, True, "x",
                  remedy="install FFmpeg"),
        ]

    def test_report_lists_remedy_on_failure(self):
        report = preflight.format_report(self._checks(ffmpeg_ok=False))
        self.assertIn("install FFmpeg", report)
        self.assertIn("MISSING required", report)

    def test_report_all_ok(self):
        report = preflight.format_report(self._checks(ffmpeg_ok=True))
        self.assertIn("all required prerequisites satisfied", report)

    def test_missing_required_filters(self):
        missing = preflight.missing_required(self._checks(ffmpeg_ok=False))
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0].name, "FFmpeg")

    def test_run_all_returns_checks(self):
        checks = preflight.run_all()
        names = [c.name for c in checks]
        self.assertIn("FFmpeg (audio decode/denoise)", names)
        self.assertIn("faster-whisper (ASR engine)", names)
        self.assertIn("Resemble-Enhance (--enhance resemble)", names)


if __name__ == "__main__":
    unittest.main()
