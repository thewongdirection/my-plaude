"""Tests for the HTML dashboard renderer and its helpers (pure, offline)."""
import unittest

from plaude_local import dashboard


class TestHelpers(unittest.TestCase):
    def test_count_words_spaced(self):
        self.assertEqual(dashboard.count_words("hello there friend"), 3)
        self.assertEqual(dashboard.count_words(""), 0)

    def test_count_words_cjk_counts_each_char(self):
        # 4 CJK chars + 1 spaced word.
        self.assertEqual(dashboard.count_words("你好世界 world"), 5)

    def test_format_duration(self):
        self.assertEqual(dashboard.format_duration(5), "5s")
        self.assertEqual(dashboard.format_duration(75), "1m 15s")
        self.assertEqual(dashboard.format_duration(3661), "1h 01m 01s")
        self.assertEqual(dashboard.format_duration(None), "—")

    def test_language_name(self):
        self.assertEqual(dashboard.language_name("en"), "English")
        self.assertEqual(dashboard.language_name("zh"), "Chinese")
        self.assertEqual(dashboard.language_name("xx"), "xx")   # unknown -> code
        self.assertEqual(dashboard.language_name(None), "Unknown")

    def test_clamp_words_caps_length(self):
        text = " ".join(str(i) for i in range(300))
        out = dashboard.clamp_words(text, max_words=250)
        self.assertLessEqual(len(out.split()), 251)   # 250 + the ellipsis token
        self.assertTrue(out.endswith("…"))


class TestRender(unittest.TestCase):
    def _doc(self, **over):
        kw = dict(
            title="My Clip", language="zh", speech_duration_s=754.0,
            word_count=1234, summary="Critical topic one and two.",
            transcript="你好世界", translation="Hello world",
            translation_label="Translated-English", cjk=True,
        )
        kw.update(over)
        return dashboard.build_dashboard_html(**kw)

    def test_has_all_three_tabs(self):
        html = self._doc()
        for tab in ("tab-transcribed", "tab-translation", "tab-sbs"):
            self.assertIn(tab, html)
        self.assertIn("Side-by-Side", html)
        self.assertIn("Translated-English", html)

    def test_shows_stats_and_summary(self):
        html = self._doc()
        self.assertIn("Chinese", html)          # language name
        self.assertIn("12m 34s", html)          # duration
        self.assertIn("1,234", html)            # word count, grouped
        self.assertIn("Critical topic one", html)

    def test_escapes_html_in_transcript(self):
        html = self._doc(transcript="<script>alert(1)</script>")
        self.assertNotIn("<script>alert(1)</script>", html)
        self.assertIn("&lt;script&gt;", html)

    def test_summary_note_when_no_summary(self):
        html = self._doc(summary=None, summary_note="Summary unavailable (test).")
        self.assertIn("Summary unavailable (test).", html)

    def test_is_self_contained(self):
        html = self._doc()
        # No external resources (CSP-safe): everything inline.
        self.assertNotIn("http://", html)
        self.assertNotIn("src=", html)
        self.assertIn("<style>", html)

    def test_side_by_side_renders_aligned_rows(self):
        html = self._doc(pairs=[("你好", "Hello"), ("世界", "World")])
        self.assertIn('class="sbs-o', html)   # left (original) cells
        self.assertIn('class="sbs-x', html)   # right (translation) cells
        self.assertIn("你好", html)
        self.assertIn("World", html)
        # Two aligned rows -> two original cells.
        self.assertEqual(html.count('class="sbs-o'), 2)

    def test_side_by_side_marks_empty_translation(self):
        html = self._doc(pairs=[("solo", "")])
        self.assertIn("sbs-x empty", html)


class TestAlign(unittest.TestCase):
    def test_align_by_time_overlap(self):
        orig = [{"start": 0.0, "end": 2.0, "text": "你好"},
                {"start": 2.0, "end": 4.0, "text": "世界"}]
        trans = [{"start": 0.1, "end": 1.9, "text": "Hello"},
                 {"start": 2.1, "end": 3.9, "text": "World"}]
        self.assertEqual(dashboard.align_segments(orig, trans),
                         [("你好", "Hello"), ("世界", "World")])

    def test_align_unmatched_translation_leaves_empty(self):
        orig = [{"start": 0.0, "end": 1.0, "text": "a"},
                {"start": 5.0, "end": 6.0, "text": "b"}]
        trans = [{"start": 0.2, "end": 0.8, "text": "A"}]
        self.assertEqual(dashboard.align_segments(orig, trans),
                         [("a", "A"), ("b", "")])

    def test_align_multiple_translation_segments_join(self):
        orig = [{"start": 0.0, "end": 4.0, "text": "long"}]
        trans = [{"start": 0.5, "end": 1.5, "text": "one"},
                 {"start": 2.0, "end": 3.0, "text": "two"}]
        self.assertEqual(dashboard.align_segments(orig, trans), [("long", "one two")])


if __name__ == "__main__":
    unittest.main()
