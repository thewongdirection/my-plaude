"""Regression tests for recording quality assessment (plaude_local.quality)."""

import unittest

from plaude_local import quality
from plaude_local.quality import assess, Thresholds, QualityReport


def _seg(start, end, text, **metrics):
    base = {"start": start, "end": end, "text": text, "speaker": None}
    base.update(metrics)
    return base


class TestTranscriptAssessment(unittest.TestCase):
    def test_good_recording_is_ok(self):
        segs = [_seg(0, 9, "hello world this is clear speech",
                     no_speech_prob=0.03, avg_logprob=-0.3, compression_ratio=1.5)]
        r = assess(segments=segs, duration=10.0)
        self.assertEqual(r.verdict, quality.VERDICT_OK)
        self.assertFalse(r.is_bad)

    def test_empty_transcript_is_bad(self):
        segs = [_seg(0, 0, "   ", no_speech_prob=0.9)]
        r = assess(segments=segs, duration=10.0)
        self.assertEqual(r.verdict, quality.VERDICT_BAD)
        self.assertTrue(any("no speech" in x for x in r.reasons))

    def test_no_speech_with_low_coverage_is_bad(self):
        segs = [_seg(0, 1, "uh", no_speech_prob=0.92, avg_logprob=-0.5,
                     compression_ratio=1.2)]
        r = assess(segments=segs, duration=30.0)  # coverage ~3%
        self.assertEqual(r.verdict, quality.VERDICT_BAD)

    def test_low_coverage_alone_is_suspect(self):
        segs = [_seg(0, 1, "hello there", no_speech_prob=0.1, avg_logprob=-0.4,
                     compression_ratio=1.3)]
        r = assess(segments=segs, duration=20.0)  # coverage 5%
        self.assertEqual(r.verdict, quality.VERDICT_SUSPECT)
        self.assertTrue(any("coverage" in x for x in r.reasons))

    def test_low_confidence_is_suspect(self):
        segs = [_seg(0, 9, "garbled maybe words", no_speech_prob=0.2,
                     avg_logprob=-1.6, compression_ratio=1.4)]
        r = assess(segments=segs, duration=10.0)
        self.assertEqual(r.verdict, quality.VERDICT_SUSPECT)
        self.assertTrue(any("confidence" in x for x in r.reasons))

    def test_high_compression_is_suspect(self):
        segs = [_seg(0, 9, "you you you you you", no_speech_prob=0.2,
                     avg_logprob=-0.5, compression_ratio=3.1)]
        r = assess(segments=segs, duration=10.0)
        self.assertEqual(r.verdict, quality.VERDICT_SUSPECT)
        self.assertTrue(any("repetitive" in x for x in r.reasons))

    def test_metrics_are_populated(self):
        segs = [_seg(0, 5, "hello", no_speech_prob=0.1, avg_logprob=-0.5,
                     compression_ratio=1.5)]
        r = assess(segments=segs, duration=10.0)
        self.assertEqual(r.metrics["num_segments"], 1)
        self.assertEqual(r.metrics["speech_coverage"], 0.5)
        self.assertIn("avg_logprob", r.metrics)

    def test_missing_metrics_do_not_crash(self):
        # Segments without any quality fields (e.g. mocked) -> ok, no crash.
        segs = [_seg(0, 5, "hello")]
        r = assess(segments=segs, duration=10.0)
        self.assertEqual(r.verdict, quality.VERDICT_OK)


class TestAudioAssessment(unittest.TestCase):
    def test_near_silent_is_bad(self):
        r = assess(audio_stats={"mean_volume_db": -62.0, "max_volume_db": -40.0,
                                "silence_ratio": 0.4})
        self.assertEqual(r.verdict, quality.VERDICT_BAD)
        self.assertTrue(any("near-silent" in x for x in r.reasons))

    def test_almost_all_silence_is_bad(self):
        r = assess(audio_stats={"mean_volume_db": -20.0, "silence_ratio": 0.99})
        self.assertEqual(r.verdict, quality.VERDICT_BAD)

    def test_mostly_silence_is_suspect(self):
        r = assess(audio_stats={"mean_volume_db": -18.0, "silence_ratio": 0.9})
        self.assertEqual(r.verdict, quality.VERDICT_SUSPECT)

    def test_healthy_levels_ok(self):
        r = assess(audio_stats={"mean_volume_db": -18.0, "silence_ratio": 0.3})
        self.assertEqual(r.verdict, quality.VERDICT_OK)


class TestThresholdsAndReport(unittest.TestCase):
    def test_custom_thresholds(self):
        segs = [_seg(0, 9, "words", no_speech_prob=0.2, avg_logprob=-0.9,
                     compression_ratio=1.4)]
        # avg_logprob -0.9 is above default -1.0 (ok) but below a stricter -0.5.
        strict = Thresholds(min_logprob=-0.5)
        self.assertEqual(assess(segments=segs, duration=10.0).verdict, quality.VERDICT_OK)
        self.assertEqual(assess(segments=segs, duration=10.0, thresholds=strict).verdict,
                         quality.VERDICT_SUSPECT)

    def test_report_summary_and_dict(self):
        r = QualityReport(verdict="bad", reasons=["no speech detected"],
                          metrics={"char_count": 0})
        self.assertIn("BAD", r.summary())
        self.assertIn("no speech detected", r.summary())
        self.assertEqual(r.as_dict()["verdict"], "bad")
        self.assertTrue(r.is_bad)


class TestLanguageAmbiguity(unittest.TestCase):
    """Hindi/Urdu are near-identical; low-confidence detection must be flagged."""

    def test_warns_for_low_confidence_hindi_or_urdu(self):
        for lang, p in (("hi", 0.50), ("ur", 0.72), ("hi", 0.89)):
            note = quality.language_ambiguity_note(lang, p)
            self.assertIsNotNone(note, (lang, p))
            self.assertIn("Hindi and Urdu", note)
            self.assertIn("--language hi or --language ur", note)
            self.assertIn(f"'{lang}' at {p:.2f}", note)

    def test_confident_detection_is_not_flagged(self):
        # Real Hindi in the corpus was always >= 0.96.
        for lang, p in (("hi", 0.98), ("ur", 0.96), ("hi", 0.90)):
            self.assertIsNone(quality.language_ambiguity_note(lang, p), (lang, p))

    def test_other_languages_and_missing_probability_are_ignored(self):
        self.assertIsNone(quality.language_ambiguity_note("en", 0.3))
        self.assertIsNone(quality.language_ambiguity_note("ta", 0.2))
        self.assertIsNone(quality.language_ambiguity_note("hi", None))
        self.assertIsNone(quality.language_ambiguity_note(None, 0.5))


if __name__ == "__main__":
    unittest.main()
