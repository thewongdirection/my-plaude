"""Regression tests for the output writers (plaude_local.formats)."""

import json
import unittest

from plaude_local import formats
from plaude_local.formats import _fmt_timestamp, _clock


def _segs():
    return [
        {"start": 0.0, "end": 3.2, "text": " Hello there.", "speaker": None},
        {"start": 3.2, "end": 6.5, "text": " General Kenobi.", "speaker": None},
    ]


class TestTimestamps(unittest.TestCase):
    def test_srt_comma_format(self):
        self.assertEqual(_fmt_timestamp(3661.5, comma=True), "01:01:01,500")

    def test_vtt_dot_format(self):
        self.assertEqual(_fmt_timestamp(3661.5, comma=False), "01:01:01.500")

    def test_zero_and_negative_clamp(self):
        self.assertEqual(_fmt_timestamp(0.0, comma=True), "00:00:00,000")
        self.assertEqual(_fmt_timestamp(-5.0, comma=True), "00:00:00,000")

    def test_millisecond_rounding(self):
        # 1.2345 s -> 1234 ms (round-half-to-even on .5 boundary not hit here)
        self.assertEqual(_fmt_timestamp(1.2345, comma=True), "00:00:01,234")

    def test_clock_short_and_long(self):
        self.assertEqual(_clock(65), "01:05")
        self.assertEqual(_clock(3725), "1:02:05")
        self.assertEqual(_clock(0), "00:00")


class TestText(unittest.TestCase):
    def test_plain(self):
        out = formats.to_text(_segs())
        self.assertEqual(out, "Hello there.\nGeneral Kenobi.\n")

    def test_with_timestamps(self):
        out = formats.to_text(_segs(), timestamps=True)
        self.assertEqual(out, "[00:00] Hello there.\n[00:03] General Kenobi.\n")

    def test_with_speakers(self):
        segs = [dict(s, speaker=f"Speaker {i + 1}") for i, s in enumerate(_segs())]
        out = formats.to_text(segs)
        self.assertEqual(out, "Speaker 1: Hello there.\nSpeaker 2: General Kenobi.\n")

    def test_skips_empty_text(self):
        segs = _segs() + [{"start": 7.0, "end": 8.0, "text": "   ", "speaker": None}]
        out = formats.to_text(segs)
        self.assertEqual(out.count("\n"), 2)  # only two non-empty lines

    def test_empty_input(self):
        self.assertEqual(formats.to_text([]), "")


class TestSrt(unittest.TestCase):
    def test_structure_and_indexing(self):
        out = formats.to_srt(_segs())
        self.assertIn("1\n00:00:00,000 --> 00:00:03,200\nHello there.", out)
        self.assertIn("2\n00:00:03,200 --> 00:00:06,500\nGeneral Kenobi.", out)

    def test_speaker_prefix(self):
        segs = [dict(s, speaker="Speaker 1") for s in _segs()]
        out = formats.to_srt(segs)
        self.assertIn("Speaker 1: Hello there.", out)

    def test_empty_segments_skipped_keep_index_contiguous(self):
        segs = [
            {"start": 0.0, "end": 1.0, "text": "", "speaker": None},
            {"start": 1.0, "end": 2.0, "text": "real", "speaker": None},
        ]
        out = formats.to_srt(segs)
        # The single real cue should be numbered 1, not 2.
        self.assertTrue(out.lstrip().startswith("1\n"))
        self.assertNotIn("2\n", out)


class TestVtt(unittest.TestCase):
    def test_header_and_voice_tag(self):
        segs = [dict(s, speaker="Speaker 1") for s in _segs()]
        out = formats.to_vtt(segs)
        self.assertTrue(out.startswith("WEBVTT"))
        self.assertIn("<v Speaker 1>Hello there.", out)
        self.assertIn("00:00:00.000 --> 00:00:03.200", out)


class TestJson(unittest.TestCase):
    def test_roundtrip_and_meta(self):
        out = formats.to_json(_segs(), meta={"model": "large-v3"})
        data = json.loads(out)
        self.assertEqual(len(data["segments"]), 2)
        self.assertEqual(data["meta"]["model"], "large-v3")

    def test_unicode_preserved(self):
        segs = [{"start": 0.0, "end": 1.0, "text": "café ünand 中文", "speaker": None}]
        out = formats.to_json(segs)
        self.assertIn("café ünand 中文", out)  # ensure_ascii=False


class TestRender(unittest.TestCase):
    def test_all_formats(self):
        for fmt in formats.FORMATS:
            self.assertIsInstance(formats.render(_segs(), fmt), str)

    def test_unknown_format_raises(self):
        with self.assertRaises(ValueError):
            formats.render(_segs(), "docx")

    def test_txt_timestamps_via_meta(self):
        out = formats.render(_segs(), "txt", meta={"timestamps": True})
        self.assertIn("[00:00]", out)


if __name__ == "__main__":
    unittest.main()
