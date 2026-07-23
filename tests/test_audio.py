"""Regression tests for audio preprocessing dispatch (plaude_local.audio)."""

import unittest
from unittest import mock

from plaude_local import audio


class TestHaveFfmpeg(unittest.TestCase):
    def test_returns_bool(self):
        self.assertIsInstance(audio.have_ffmpeg(), bool)


class TestPrepareDispatch(unittest.TestCase):
    def setUp(self):
        self.tmp = mock.MagicMock()
        # A real temp dir + input file so path handling is exercised.
        import tempfile
        import pathlib
        self._dir = tempfile.mkdtemp(prefix="plaude-test-")
        self.workdir = pathlib.Path(self._dir) / "work"
        self.src = pathlib.Path(self._dir) / "in.wav"
        self.src.write_bytes(b"RIFF....")

    def tearDown(self):
        import shutil
        shutil.rmtree(self._dir, ignore_errors=True)

    def test_missing_file_raises(self):
        with self.assertRaises(audio.AudioError):
            audio.prepare("/no/such/file.wav", self.workdir)

    def test_denoise_none_uses_no_filters(self):
        with mock.patch.object(audio, "_to_wav") as to_wav:
            out = audio.prepare(self.src, self.workdir, denoise="none")
        to_wav.assert_called_once()
        self.assertIsNone(to_wav.call_args.kwargs.get("filters"))
        self.assertTrue(str(out).endswith(".prepared.wav"))

    def test_denoise_ffmpeg_uses_filter_chain(self):
        with mock.patch.object(audio, "_to_wav") as to_wav:
            audio.prepare(self.src, self.workdir, denoise="ffmpeg")
        filters = to_wav.call_args.kwargs.get("filters")
        self.assertIn("afftdn", filters)
        self.assertIn("highpass", filters)

    def test_denoise_deepfilter_dispatches(self):
        with mock.patch.object(audio, "_denoise_deepfilter") as df:
            audio.prepare(self.src, self.workdir, denoise="deepfilter")
        df.assert_called_once()

    def test_unknown_denoise_raises(self):
        with self.assertRaises(audio.AudioError):
            audio.prepare(self.src, self.workdir, denoise="magic")


class TestFfmpegErrors(unittest.TestCase):
    def test_missing_binary_message(self):
        with mock.patch("subprocess.run", side_effect=FileNotFoundError()):
            with self.assertRaises(audio.AudioError) as ctx:
                audio._run_ffmpeg(["-i", "x"])
        self.assertIn("ffmpeg", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
