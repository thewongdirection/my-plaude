"""Regression tests for audio preprocessing dispatch (plaude_local.audio)."""

import unittest
from unittest import mock

from plaude_local import audio


class TestHaveFfmpeg(unittest.TestCase):
    def test_returns_bool(self):
        self.assertIsInstance(audio.have_ffmpeg(), bool)
        self.assertIsInstance(audio.have_ffprobe(), bool)


class TestProbeAudioCodec(unittest.TestCase):
    def test_returns_none_without_ffprobe(self):
        with mock.patch.object(audio, "have_ffprobe", return_value=False):
            self.assertIsNone(audio.probe_audio_codec("x.m4a"))

    def test_parses_codec_name(self):
        completed = mock.Mock(stdout="aac\n", returncode=0)
        with mock.patch.object(audio, "have_ffprobe", return_value=True), \
             mock.patch("subprocess.run", return_value=completed) as run:
            codec = audio.probe_audio_codec("x.m4a")
        self.assertEqual(codec, "aac")
        # ffprobe is the tool invoked, selecting the first audio stream.
        argv = run.call_args.args[0]
        self.assertEqual(argv[0], "ffprobe")
        self.assertIn("a:0", argv)

    def test_empty_output_means_no_audio_stream(self):
        completed = mock.Mock(stdout="\n", returncode=0)
        with mock.patch.object(audio, "have_ffprobe", return_value=True), \
             mock.patch("subprocess.run", return_value=completed):
            self.assertIsNone(audio.probe_audio_codec("silent.mp4"))


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

    def test_enhance_speech_applies_speech_chain(self):
        with mock.patch.object(audio, "_to_wav") as to_wav:
            audio.prepare(self.src, self.workdir, denoise="none", enhance="speech")
        # denoise pass, then enhance pass.
        self.assertEqual(to_wav.call_count, 2)
        enhance_filters = to_wav.call_args_list[1].kwargs.get("filters")
        self.assertIn("speechnorm", enhance_filters)
        self.assertIn("loudnorm", enhance_filters)

    def test_enhance_strong_has_compressor_and_eq(self):
        with mock.patch.object(audio, "_to_wav") as to_wav:
            audio.prepare(self.src, self.workdir, denoise="none", enhance="strong")
        f = to_wav.call_args_list[1].kwargs.get("filters")
        self.assertIn("acompressor", f)
        self.assertIn("equalizer", f)

    def test_gain_only_applies_volume(self):
        with mock.patch.object(audio, "_to_wav") as to_wav:
            audio.prepare(self.src, self.workdir, denoise="none",
                          enhance="none", gain_db=6.0)
        self.assertEqual(to_wav.call_count, 2)
        self.assertIn("volume=6.0dB", to_wav.call_args_list[1].kwargs.get("filters"))

    def test_enhance_none_no_gain_is_single_pass(self):
        with mock.patch.object(audio, "_to_wav") as to_wav:
            out = audio.prepare(self.src, self.workdir, denoise="none",
                                enhance="none")
        to_wav.assert_called_once()
        self.assertTrue(str(out).endswith(".prepared.wav"))

    def test_enhance_resemble_dispatches(self):
        with mock.patch.object(audio, "_to_wav"), \
             mock.patch.object(audio, "_enhance_resemble") as re:
            audio.prepare(self.src, self.workdir, denoise="none", enhance="resemble")
        re.assert_called_once()

    def test_enhance_resemble_with_gain_applies_volume_then_replaces(self):
        import pathlib

        def fake_to_wav(src, dst, *, filters=None):
            pathlib.Path(dst).write_bytes(b"x")  # create output so .replace works

        with mock.patch.object(audio, "_to_wav", side_effect=fake_to_wav) as to_wav, \
             mock.patch.object(audio, "_enhance_resemble") as re:
            out = audio.prepare(self.src, self.workdir, denoise="none",
                                enhance="resemble", gain_db=6.0)
        re.assert_called_once()
        # A volume pass ran after the neural restoration.
        vol_calls = [c for c in to_wav.call_args_list
                     if (c.kwargs.get("filters") or "").startswith("volume=")]
        self.assertEqual(len(vol_calls), 1)
        self.assertIn("6.0dB", vol_calls[0].kwargs["filters"])
        self.assertTrue(str(out).endswith(".prepared.wav"))

    def test_deepfilter_denoise_then_enhance_uses_intermediate(self):
        with mock.patch.object(audio, "_denoise_deepfilter") as df, \
             mock.patch.object(audio, "_to_wav") as to_wav:
            audio.prepare(self.src, self.workdir, denoise="deepfilter",
                          enhance="speech")
        df.assert_called_once()
        # deepfilter writes the intermediate (.denoised), enhance writes .prepared;
        # the two paths must differ (no in-place ffmpeg pass).
        denoise_dst = str(df.call_args.args[1])
        enhance_src = str(to_wav.call_args_list[-1].args[0])
        enhance_dst = str(to_wav.call_args_list[-1].args[1])
        self.assertTrue(denoise_dst.endswith(".denoised.wav"))
        self.assertEqual(enhance_src, denoise_dst)
        self.assertTrue(enhance_dst.endswith(".prepared.wav"))
        self.assertNotEqual(enhance_src, enhance_dst)
        self.assertIn("speechnorm", to_wav.call_args_list[-1].kwargs.get("filters"))

    def test_unknown_enhance_raises(self):
        with self.assertRaises(audio.AudioError):
            audio.prepare(self.src, self.workdir, enhance="magic")


class TestKeepStages(unittest.TestCase):
    """--keep-stages: every preprocessing stage is saved for inspection."""

    def setUp(self):
        import tempfile
        import pathlib
        self._dir = tempfile.mkdtemp(prefix="plaude-test-")
        self.workdir = pathlib.Path(self._dir) / "work"
        self.stages = pathlib.Path(self._dir) / "stages"
        self.src = pathlib.Path(self._dir) / "lecture.m4a"
        self.src.write_bytes(b"RIFF....")

    def tearDown(self):
        import shutil
        shutil.rmtree(self._dir, ignore_errors=True)

    @staticmethod
    def _fake_to_wav(src, dst, *, filters=None):
        import pathlib
        pathlib.Path(dst).write_bytes(f"from {pathlib.Path(src).name} | {filters}".encode())

    def _names(self):
        return sorted(p.name for p in self.stages.iterdir())

    def _manifest(self):
        import json
        return json.loads((self.stages / "lecture.stages.json").read_text(encoding="utf-8"))

    def test_no_stages_dir_saves_nothing(self):
        with mock.patch.object(audio, "_to_wav", side_effect=self._fake_to_wav):
            audio.prepare(self.src, self.workdir, denoise="ffmpeg", enhance="speech")
        self.assertFalse(self.stages.exists())

    def test_denoise_and_enhance_saves_all_three_stages(self):
        with mock.patch.object(audio, "_to_wav", side_effect=self._fake_to_wav):
            final = audio.prepare(self.src, self.workdir, denoise="ffmpeg",
                                  enhance="strong", gain_db=3.0,
                                  stages_dir=self.stages)
        self.assertEqual(self._names(), [
            "lecture.01-original.wav", "lecture.02-denoised.wav",
            "lecture.03-enhanced.wav", "lecture.stages.json"])
        # The original is an unfiltered decode; the enhanced stage is the final wav.
        self.assertIn(b"| None", (self.stages / "lecture.01-original.wav").read_bytes())
        self.assertIn(b"afftdn", (self.stages / "lecture.02-denoised.wav").read_bytes())
        self.assertEqual((self.stages / "lecture.03-enhanced.wav").read_bytes(),
                         final.read_bytes())
        m = self._manifest()
        self.assertEqual(m["denoise"], "ffmpeg")
        self.assertIn("afftdn", m["denoise_filters"])
        self.assertEqual(m["enhance"], "strong")
        self.assertEqual(m["gain_db"], 3.0)
        self.assertIn("acompressor", m["enhance_filters"])
        self.assertIn("volume=3.0dB", m["enhance_filters"])
        self.assertEqual(m["sample_rate"], 16000)
        self.assertEqual(m["files"], {"original": "lecture.01-original.wav",
                                      "denoised": "lecture.02-denoised.wav",
                                      "enhanced": "lecture.03-enhanced.wav"})

    def test_denoise_only_has_no_enhanced_stage(self):
        with mock.patch.object(audio, "_to_wav", side_effect=self._fake_to_wav):
            audio.prepare(self.src, self.workdir, denoise="ffmpeg",
                          stages_dir=self.stages)
        self.assertEqual(self._names(), [
            "lecture.01-original.wav", "lecture.02-denoised.wav",
            "lecture.stages.json"])
        self.assertIsNone(self._manifest()["enhance_filters"])

    def test_denoise_none_saves_decode_as_original_only(self):
        with mock.patch.object(audio, "_to_wav", side_effect=self._fake_to_wav) as to_wav:
            audio.prepare(self.src, self.workdir, denoise="none", enhance="speech",
                          stages_dir=self.stages)
        self.assertEqual(self._names(), [
            "lecture.01-original.wav", "lecture.03-enhanced.wav",
            "lecture.stages.json"])
        # No extra decode pass: the plain decode already is the original.
        self.assertEqual(to_wav.call_count, 2)
        self.assertIsNone(self._manifest()["denoise_filters"])

    def test_resemble_manifest_names_the_neural_model(self):
        import pathlib

        def fake_resemble(src, dst):
            pathlib.Path(dst).write_bytes(b"restored")

        with mock.patch.object(audio, "_to_wav", side_effect=self._fake_to_wav),              mock.patch.object(audio, "_enhance_resemble", side_effect=fake_resemble):
            audio.prepare(self.src, self.workdir, denoise="ffmpeg",
                          enhance="resemble", stages_dir=self.stages)
        self.assertEqual(self._manifest()["enhance_filters"], "Resemble-Enhance")

    def test_unwritable_stages_dir_raises_audio_error(self):
        self.stages.write_bytes(b"not a folder")  # a file where the folder should be
        with mock.patch.object(audio, "_to_wav", side_effect=self._fake_to_wav):
            with self.assertRaises(audio.AudioError):
                audio.prepare(self.src, self.workdir, denoise="ffmpeg",
                              stages_dir=self.stages)


class TestEnhanceFilters(unittest.TestCase):
    def test_none_no_gain_is_none(self):
        self.assertIsNone(audio._enhance_filters("none", 0.0))

    def test_none_with_gain_is_volume_only(self):
        self.assertEqual(audio._enhance_filters("none", 6.0), "volume=6.0dB")

    def test_speech_chain(self):
        self.assertIn("speechnorm", audio._enhance_filters("speech", 0.0))

    def test_chain_plus_gain_appends_volume(self):
        f = audio._enhance_filters("speech", -3.0)
        self.assertTrue(f.endswith("volume=-3.0dB"))


class TestToWav(unittest.TestCase):
    """The core decode contract: everything is normalized to 16 kHz mono PCM."""

    def test_builds_16k_mono_pcm_args(self):
        import pathlib
        with mock.patch.object(audio, "_run_ffmpeg") as run:
            audio._to_wav(pathlib.Path("in.mp3"), pathlib.Path("out.wav"))
        args = run.call_args.args[0]
        self.assertIn("-ac", args)
        self.assertEqual(args[args.index("-ac") + 1], "1")        # mono
        self.assertIn("-ar", args)
        self.assertEqual(args[args.index("-ar") + 1], str(audio.TARGET_SR))
        self.assertIn("pcm_s16le", args)
        self.assertNotIn("-af", args)  # no filter chain unless requested

    def test_filters_are_passed_through(self):
        import pathlib
        with mock.patch.object(audio, "_run_ffmpeg") as run:
            audio._to_wav(pathlib.Path("in.mp3"), pathlib.Path("out.wav"),
                          filters="highpass=f=90")
        args = run.call_args.args[0]
        self.assertIn("-af", args)
        self.assertEqual(args[args.index("-af") + 1], "highpass=f=90")


class TestProbeLevels(unittest.TestCase):
    _STDERR = (
        "[Parsed_volumedetect_0 @ 0x1] mean_volume: -23.4 dB\n"
        "[Parsed_volumedetect_0 @ 0x1] max_volume: -2.1 dB\n"
        "[silencedetect @ 0x2] silence_start: 0\n"
        "[silencedetect @ 0x2] silence_end: 3 | silence_duration: 3.0\n"
        "[silencedetect @ 0x2] silence_duration: 2.0\n"
    )

    def test_parses_levels_and_silence_ratio(self):
        completed = mock.Mock(stderr=self._STDERR, returncode=0)
        with mock.patch("subprocess.run", return_value=completed), \
             mock.patch.object(audio, "probe_duration", return_value=10.0):
            stats = audio.probe_levels("x.wav")
        self.assertEqual(stats["mean_volume_db"], -23.4)
        self.assertEqual(stats["max_volume_db"], -2.1)
        self.assertAlmostEqual(stats["silence_ratio"], 0.5)  # (3+2)/10

    def test_silence_ratio_none_without_duration(self):
        completed = mock.Mock(stderr=self._STDERR, returncode=0)
        with mock.patch("subprocess.run", return_value=completed), \
             mock.patch.object(audio, "probe_duration", return_value=None):
            stats = audio.probe_levels("x.wav")
        self.assertIsNone(stats["silence_ratio"])


class TestFfmpegErrors(unittest.TestCase):
    def test_missing_binary_message(self):
        with mock.patch("subprocess.run", side_effect=FileNotFoundError()):
            with self.assertRaises(audio.AudioError) as ctx:
                audio._run_ffmpeg(["-i", "x"])
        self.assertIn("ffmpeg", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
