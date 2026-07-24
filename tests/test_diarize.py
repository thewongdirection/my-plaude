"""Regression tests for diarization merge logic and backend dispatch."""

import collections
import unittest
from unittest import mock

from plaude_local import diarize
from plaude_local.diarize import (
    _best_speaker,
    _relabel,
    _annotation_to_turns,
    _dataframe_to_turns,
    merge_turns,
)


TURNS = [(0.0, 3.0, "SPEAKER_01"), (3.0, 7.0, "SPEAKER_00")]


class TestBestSpeaker(unittest.TestCase):
    def test_picks_max_overlap(self):
        self.assertEqual(_best_speaker(0.1, 2.5, TURNS), "SPEAKER_01")
        self.assertEqual(_best_speaker(3.5, 6.0, TURNS), "SPEAKER_00")

    def test_straddling_boundary_picks_larger_side(self):
        # [2.0, 5.0]: 1.0s in SPEAKER_01, 2.0s in SPEAKER_00 -> SPEAKER_00
        self.assertEqual(_best_speaker(2.0, 5.0, TURNS), "SPEAKER_00")

    def test_no_overlap_returns_none(self):
        self.assertIsNone(_best_speaker(20.0, 25.0, TURNS))


class TestRelabel(unittest.TestCase):
    def test_first_seen_by_time_order(self):
        # Even if given out of order, labels are numbered by earliest start.
        turns = [(5.0, 6.0, "B"), (0.0, 1.0, "A"), (2.0, 3.0, "B")]
        self.assertEqual(_relabel(turns), {"A": "Speaker 1", "B": "Speaker 2"})

    def test_stable_single_speaker(self):
        self.assertEqual(_relabel([(0.0, 1.0, "X")]), {"X": "Speaker 1"})


class TestMergeTurns(unittest.TestCase):
    def test_empty_turns_returns_unchanged(self):
        segs = [{"start": 0.0, "end": 1.0, "text": "hi", "speaker": None}]
        self.assertIs(merge_turns(segs, []), segs)

    def test_segment_level_tagging(self):
        segs = [
            {"start": 0.5, "end": 2.5, "text": "a", "speaker": None},
            {"start": 3.5, "end": 6.0, "text": "b", "speaker": None},
        ]
        out = merge_turns(segs, TURNS)
        self.assertEqual(out[0]["speaker"], "Speaker 1")  # SPEAKER_01 first by time
        self.assertEqual(out[1]["speaker"], "Speaker 2")

    def test_word_level_regroups_on_speaker_change(self):
        segs = [{
            "start": 0.0, "end": 7.0, "text": "one two three four", "speaker": None,
            "words": [
                {"start": 0.0, "end": 1.0, "word": " one"},
                {"start": 1.0, "end": 2.0, "word": " two"},
                {"start": 3.5, "end": 4.5, "word": " three"},
                {"start": 4.5, "end": 5.5, "word": " four"},
            ],
        }]
        out = merge_turns(segs, TURNS)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["speaker"], "Speaker 1")
        self.assertEqual(out[0]["text"], "one two")
        self.assertEqual(out[0]["start"], 0.0)
        self.assertEqual(out[0]["end"], 2.0)
        self.assertEqual(out[1]["speaker"], "Speaker 2")
        self.assertEqual(out[1]["text"], "three four")
        self.assertEqual(out[1]["end"], 5.5)

    def test_word_level_preserves_segment_without_words(self):
        # Regression: when some segments carry word timestamps and others do
        # not, the wordless segment's text must NOT be dropped.
        segs = [
            {
                "start": 0.0, "end": 2.0, "text": "one two", "speaker": None,
                "words": [
                    {"start": 0.0, "end": 1.0, "word": " one"},
                    {"start": 1.0, "end": 2.0, "word": " two"},
                ],
            },
            # No "words" key at all (e.g. VAD/silence-edge segment).
            {"start": 3.5, "end": 6.0, "text": " three four", "speaker": None},
        ]
        out = merge_turns(segs, TURNS)
        joined = " ".join(s["text"] for s in out)
        self.assertIn("one two", joined)
        self.assertIn("three four", joined)  # must not be lost
        # The wordless segment is tagged as a whole by overlap (Speaker 2).
        self.assertEqual(out[-1]["speaker"], "Speaker 2")

    def test_word_level_empty_words_list_segment_preserved(self):
        segs = [
            {
                "start": 0.0, "end": 1.0, "text": "hi", "speaker": None,
                "words": [{"start": 0.0, "end": 1.0, "word": "hi"}],
            },
            {"start": 3.5, "end": 6.0, "text": "kept", "speaker": None, "words": []},
        ]
        out = merge_turns(segs, TURNS)
        self.assertIn("kept", " ".join(s["text"] for s in out))

    def test_word_level_single_speaker_stays_one_turn(self):
        segs = [{
            "start": 0.0, "end": 2.0, "text": "one two", "speaker": None,
            "words": [
                {"start": 0.0, "end": 1.0, "word": " one"},
                {"start": 1.0, "end": 2.0, "word": " two"},
            ],
        }]
        out = merge_turns(segs, [(0.0, 3.0, "SPEAKER_01")])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["text"], "one two")
        self.assertEqual(out[0]["speaker"], "Speaker 1")


class TestAnnotationToTurns(unittest.TestCase):
    def test_flatten_and_sort(self):
        Seg = collections.namedtuple("Seg", ["start", "end"])

        class FakeAnnotation:
            def itertracks(self, yield_label=False):
                # deliberately out of order
                yield Seg(3.0, 5.0), "_", "SPEAKER_00"
                yield Seg(0.0, 2.0), "_", "SPEAKER_01"

        turns = _annotation_to_turns(FakeAnnotation())
        self.assertEqual(turns[0], (0.0, 2.0, "SPEAKER_01"))
        self.assertEqual(turns[1], (3.0, 5.0, "SPEAKER_00"))


class TestDataframeToTurns(unittest.TestCase):
    def test_flatten_from_itertuples(self):
        Row = collections.namedtuple("Row", ["start", "end", "speaker"])

        class FakeDF:
            def itertuples(self, index=False):
                yield Row(3.0, 4.0, "SPEAKER_00")
                yield Row(0.0, 1.0, "SPEAKER_01")

        turns = _dataframe_to_turns(FakeDF())
        self.assertEqual(turns[0], (0.0, 1.0, "SPEAKER_01"))
        self.assertEqual(turns[1], (3.0, 4.0, "SPEAKER_00"))


class TestDispatch(unittest.TestCase):
    def test_unknown_backend_raises(self):
        with self.assertRaises(diarize.DiarizeError):
            diarize.diarize_and_merge("x.wav", [], backend="nope")

    def test_pyannote_backend_dispatch(self):
        segs = [{"start": 0.5, "end": 2.5, "text": "a", "speaker": None}]
        with mock.patch.object(diarize, "_pyannote_turns", return_value=TURNS) as m:
            out = diarize.diarize_and_merge(
                "x.wav", segs, backend="pyannote", hf_token="t", device="cpu"
            )
        m.assert_called_once()
        self.assertEqual(out[0]["speaker"], "Speaker 1")

    def test_whisperx_backend_dispatch(self):
        segs = [{"start": 3.5, "end": 6.0, "text": "b", "speaker": None}]
        with mock.patch.object(diarize, "_whisperx_turns", return_value=TURNS) as m:
            out = diarize.diarize_and_merge(
                "x.wav", segs, backend="whisperx", hf_token="t", device="cuda"
            )
        m.assert_called_once()
        # device is threaded through to the backend
        self.assertEqual(m.call_args.args[2], "cuda")
        self.assertEqual(out[0]["speaker"], "Speaker 2")

    def test_missing_whisperx_raises_clean_error(self):
        import sys
        with mock.patch.dict(sys.modules, {"whisperx": None, "whisperx.diarize": None}):
            with self.assertRaises(diarize.DiarizeError):
                diarize._load_whisperx_pipeline("token", "cpu")


class TestDiarizeModelSelection(unittest.TestCase):
    """The pipeline model auto-selects by pyannote version; --diarize-model wins."""

    def test_v4_uses_community_model(self):
        self.assertEqual(
            diarize._model_for_pyannote_version("4.0.7"),
            "pyannote/speaker-diarization-community-1",
        )

    def test_v3_uses_3_1_model(self):
        self.assertEqual(
            diarize._model_for_pyannote_version("3.4.0"),
            "pyannote/speaker-diarization-3.1",
        )

    def test_unparseable_version_falls_back_to_3_1(self):
        self.assertEqual(
            diarize._model_for_pyannote_version("weird"),
            "pyannote/speaker-diarization-3.1",
        )

    def test_diarize_and_merge_forwards_explicit_model(self):
        segs = [{"start": 0.0, "end": 1.0, "text": "hi", "speaker": None}]
        with mock.patch.object(diarize, "_pyannote_turns", return_value=TURNS) as m:
            diarize.diarize_and_merge(
                "x.wav", segs, backend="pyannote", hf_token="t",
                diarize_model="pyannote/speaker-diarization-community-1",
            )
        self.assertEqual(
            m.call_args.kwargs["model"], "pyannote/speaker-diarization-community-1"
        )


class TestPyannoteTokenCompat(unittest.TestCase):
    """pyannote.audio 4.x renamed use_auth_token -> token; support both."""

    def test_prefers_token_kwarg(self):
        seen = []

        class FakePipeline:
            @staticmethod
            def from_pretrained(repo, **kw):
                seen.append(kw)
                return "PIPE"

        out = diarize._pyannote_from_pretrained(FakePipeline, "repo", "tok")
        self.assertEqual(out, "PIPE")
        self.assertEqual(seen, [{"token": "tok"}])

    def test_falls_back_to_use_auth_token_on_typeerror(self):
        seen = []

        class FakePipeline:
            @staticmethod
            def from_pretrained(repo, **kw):
                if "token" in kw:  # emulate the old (pyannote < 4) signature
                    raise TypeError("unexpected keyword argument 'token'")
                seen.append(kw)
                return "PIPE"

        out = diarize._pyannote_from_pretrained(FakePipeline, "repo", "tok")
        self.assertEqual(out, "PIPE")
        self.assertEqual(seen, [{"use_auth_token": "tok"}])


if __name__ == "__main__":
    unittest.main()
