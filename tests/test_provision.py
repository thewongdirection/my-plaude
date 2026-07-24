"""Regression tests for the prerequisite provisioning gate (decision logic)."""

import io
import os
import pathlib
import tempfile
import unittest
from unittest import mock

from plaude_local import provision, audio


def _quiet(**kwargs):
    """Call ensure_ffmpeg with stderr suppressed; return its bool result."""
    import contextlib
    with contextlib.redirect_stderr(io.StringIO()):
        return provision.ensure_ffmpeg(**kwargs)


class TestPresent(unittest.TestCase):
    def test_proceeds_when_ffmpeg_and_ffprobe_present(self):
        with mock.patch.object(audio, "have_ffmpeg", return_value=True), \
             mock.patch.object(audio, "have_ffprobe", return_value=True):
            self.assertTrue(_quiet())

    def test_ffmpeg_present_ffprobe_missing_still_proceeds(self):
        with mock.patch.object(audio, "have_ffmpeg", return_value=True), \
             mock.patch.object(audio, "have_ffprobe", return_value=False):
            self.assertTrue(_quiet())


class TestMissing(unittest.TestCase):
    def setUp(self):
        self._p = [mock.patch.object(audio, "have_ffmpeg", return_value=False),
                   mock.patch.object(audio, "have_ffprobe", return_value=False)]
        for p in self._p:
            p.start()
            self.addCleanup(p.stop)

    def test_explicit_path_used_non_interactively(self):
        with mock.patch.object(provision, "register_ffmpeg_path", return_value=True):
            self.assertTrue(_quiet(ffmpeg_location="/opt/ff", interactive=False))

    def test_bad_explicit_path_then_abort_non_interactive(self):
        with mock.patch.object(provision, "register_ffmpeg_path", return_value=False):
            self.assertFalse(_quiet(ffmpeg_location="/nope", interactive=False))

    def test_install_missing_success(self):
        with mock.patch.object(provision, "install_ffmpeg", return_value=True) as inst:
            self.assertTrue(_quiet(install_missing=True))
        inst.assert_called_once()

    def test_install_missing_failure_aborts(self):
        with mock.patch.object(provision, "install_ffmpeg", return_value=False):
            self.assertFalse(_quiet(install_missing=True))

    def test_no_provision_aborts(self):
        self.assertFalse(_quiet(no_provision=True, interactive=True))

    def test_non_interactive_without_authorization_aborts(self):
        self.assertFalse(_quiet(interactive=False))

    def test_interactive_choose_install(self):
        self.assertTrue(_quiet(
            interactive=True,
            prompt_choice=lambda p, o: "i",
            installer=lambda: True))

    def test_interactive_abort(self):
        self.assertFalse(_quiet(
            interactive=True,
            prompt_choice=lambda p, o: "a"))

    def test_interactive_install_fail_then_abort(self):
        answers = iter(["i", "a"])
        self.assertFalse(_quiet(
            interactive=True,
            prompt_choice=lambda p, o: next(answers),
            installer=lambda: False))

    def test_interactive_path_uses_register(self):
        with mock.patch.object(provision, "register_ffmpeg_path", return_value=True):
            self.assertTrue(_quiet(
                interactive=True,
                prompt_choice=lambda p, o: "p",
                prompt_path=lambda p: "/opt/ff"))

    def test_interactive_path_fail_then_abort(self):
        choices = iter(["p", "a"])
        self.assertFalse(_quiet(
            interactive=True,
            prompt_choice=lambda p, o: next(choices),
            prompt_path=lambda p: "/no/such/dir/at/all"))

    def test_interactive_invalid_choice_reprompts_then_abort(self):
        # Unrecognized input must re-prompt (parity), not abort immediately.
        choices = iter(["x", "?", "a"])
        self.assertFalse(_quiet(
            interactive=True,
            prompt_choice=lambda p, o: next(choices)))


class TestRegisterPath(unittest.TestCase):
    def test_directory_with_both_binaries(self):
        with tempfile.TemporaryDirectory() as d:
            (pathlib.Path(d) / "ffmpeg").write_text("x")
            (pathlib.Path(d) / "ffprobe").write_text("x")
            with mock.patch.object(audio, "have_ffmpeg", return_value=True), \
                 mock.patch.object(audio, "have_ffprobe", return_value=True):
                ok = provision.register_ffmpeg_path(d)
            self.assertTrue(ok)
            self.assertIn(d, os.environ["PATH"])

    def test_missing_ffprobe_returns_false(self):
        with tempfile.TemporaryDirectory() as d:
            (pathlib.Path(d) / "ffmpeg").write_text("x")  # no ffprobe
            self.assertFalse(provision.register_ffmpeg_path(d))

    def test_nonexistent_dir_returns_false(self):
        self.assertFalse(provision.register_ffmpeg_path("/no/such/dir/here"))

    def test_accepts_binary_path_and_uses_parent(self):
        with tempfile.TemporaryDirectory() as d:
            binpath = pathlib.Path(d) / "ffmpeg.exe"
            binpath.write_text("x")
            (pathlib.Path(d) / "ffprobe.exe").write_text("x")
            with mock.patch.object(audio, "have_ffmpeg", return_value=True), \
                 mock.patch.object(audio, "have_ffprobe", return_value=True):
                self.assertTrue(provision.register_ffmpeg_path(str(binpath)))


class TestExtract(unittest.TestCase):
    def test_extract_and_register_finds_nested_binaries(self):
        import zipfile
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            stage = d / "stage" / "ffmpeg-1.0" / "bin"
            stage.mkdir(parents=True)
            (stage / "ffmpeg").write_text("x")
            (stage / "ffprobe").write_text("x")
            archive = d / "ff.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.write(stage / "ffmpeg", "ffmpeg-1.0/bin/ffmpeg")
                z.write(stage / "ffprobe", "ffmpeg-1.0/bin/ffprobe")
            with mock.patch.object(audio, "have_ffmpeg", return_value=True), \
                 mock.patch.object(audio, "have_ffprobe", return_value=True), \
                 mock.patch.object(provision, "_install_dir", return_value=d / "out"):
                self.assertTrue(provision._extract_and_register(archive))


if __name__ == "__main__":
    unittest.main()
