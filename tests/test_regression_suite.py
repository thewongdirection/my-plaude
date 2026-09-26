"""Offline tests for the multilingual regression harness (tests/regression/).

The end-to-end suite itself needs models, a GPU and Ollama; these cover its
pure logic (scoring, damage recipes, corpus planning, pass/fail evaluation,
flag parity of the two CLI invocations) and validate the committed corpus.
"""

import json
import random
import sys
import unittest
from pathlib import Path

REG = Path(__file__).resolve().parent / "regression"
sys.path.insert(0, str(REG))

import damage  # noqa: E402
import generate_corpus as gen  # noqa: E402
import languages  # noqa: E402
import run_regression as rr  # noqa: E402
import scoring  # noqa: E402


class TestScoring(unittest.TestCase):
    def test_cer_ignores_case_space_and_punctuation(self):
        self.assertEqual(scoring.cer("Hello, World!", "hello world"), 0.0)
        self.assertEqual(scoring.cer("今天，我们讨论。", "今天我们讨论"), 0.0)

    def test_cer_counts_edits(self):
        self.assertAlmostEqual(scoring.cer("abcd", "abxd"), 0.25)
        self.assertAlmostEqual(scoring.cer("abcd", ""), 1.0)
        self.assertEqual(scoring.cer("", ""), 0.0)

    def test_cer_keeps_combining_marks(self):
        # Devanagari vowel signs are category M and must count.
        self.assertGreater(scoring.cer("किताब", "कतब"), 0.0)

    def test_cer_folds_script_variants(self):
        self.assertEqual(scoring.cer("Добро јутро", "Dobro jutro", "sr"), 0.0)
        self.assertGreater(scoring.cer("Добро јутро", "Dobro jutro"), 0.5)

    def test_levenshtein(self):
        self.assertEqual(scoring.levenshtein("kitten", "sitting"), 3)
        self.assertEqual(scoring.levenshtein("", "abc"), 3)

    def test_content_recall(self):
        ref = "The weather today is cold and windy."
        self.assertEqual(scoring.content_recall(ref, "Today the weather is cold and windy"), 1.0)
        self.assertLess(scoring.content_recall(ref, "It is sunny."), 0.5)
        self.assertEqual(scoring.content_recall("", "anything"), 1.0)

    def test_content_recall_light_stemming(self):
        self.assertEqual(scoring.content_recall("The students were learning", "a student learns"), 1.0)

    def test_detected_language_from_dashboard(self):
        html = ('<div class="stat"><p class="label">Language detected</p>'
                '<div class="value">German (de)</div></div>')
        self.assertEqual(scoring.detected_language(html), "de")
        self.assertIsNone(scoring.detected_language("<html></html>"))


class TestDamage(unittest.TestCase):
    def test_plan_is_reproducible_and_in_range(self):
        for i in range(200):
            a = damage.damage_plan(random.Random(i))
            self.assertEqual(a, damage.damage_plan(random.Random(i)))
            self.assertIn(a["hum_hz"], (None, 50, 60))
            self.assertTrue(0.02 <= a["noise"]["amplitude"] <= 0.06)
            if a["level"] == "clipped":
                self.assertGreater(a["level_db"], 0)
            elif a["level"] == "quiet":
                self.assertLess(a["level_db"], 0)
            else:
                self.assertEqual(a["level_db"], 0.0)

    def test_plans_cover_every_imperfection(self):
        plans = [damage.damage_plan(random.Random(i)) for i in range(200)]
        self.assertTrue(any(p["hum_hz"] == 50 for p in plans))
        self.assertTrue(any(p["hum_hz"] == 60 for p in plans))
        self.assertTrue(any(p["hum_hz"] is None for p in plans))
        self.assertTrue(any(p["clicks_density"] for p in plans))
        self.assertEqual({p["level"] for p in plans}, {"clipped", "quiet", "normal"})
        self.assertEqual({p["tone"] for p in plans}, {"muffled", "telephone", "none"})
        self.assertTrue(any(p["reverb"] for p in plans))

    def test_filtergraph_contains_requested_damage(self):
        plan = {"noise": {"color": "pink", "amplitude": 0.03, "seed": 7}, "hum_hz": 60,
                "hum_amplitude": 0.05, "clicks_density": 0.0005, "level": "clipped",
                "level_db": 10.0, "tone": "telephone", "lowpass_hz": None, "reverb": True}
        g = damage.damage_filtergraph(plan)
        for needle in ("anoisesrc=color=pink", "sin(2*PI*60*t)", "random(0)",
                       "highpass=f=300", "lowpass=f=3400", "aecho", "amix=inputs=4",
                       "volume=10.0dB", "[out]"):
            self.assertIn(needle, g)

    def test_filtergraph_minimal(self):
        plan = {"noise": {"color": "white", "amplitude": 0.02, "seed": 1}, "hum_hz": None,
                "hum_amplitude": None, "clicks_density": None, "level": "normal",
                "level_db": 0.0, "tone": "none", "lowpass_hz": None, "reverb": False}
        g = damage.damage_filtergraph(plan)
        self.assertIn("amix=inputs=2", g)
        self.assertNotIn("volume=", g)
        self.assertNotIn("aevalsrc", g)
        self.assertIsNone(damage.expected_hum(plan))


class TestCorpusPlan(unittest.TestCase):
    def test_plan_shape(self):
        plan = gen.plan_corpus()
        self.assertEqual(len(plan), 300)
        self.assertEqual(sum(p["kind"] == "mono" for p in plan), 150)
        self.assertEqual(sum(p["kind"] == "conv" for p in plan), 150)
        self.assertEqual(len({p["id"] for p in plan}), 300)
        self.assertEqual({p["code"] for p in plan}, {l["code"] for l in languages.LANGUAGES})
        for p in plan:
            self.assertTrue(gen.MIN_SECONDS <= p["target_s"] <= gen.MAX_TARGET)
            self.assertEqual(p["speakers"] == 1, p["kind"] == "mono")
            self.assertIn(p["speakers"], (1, 2, 3))

    def test_plan_is_reproducible(self):
        self.assertEqual(gen.plan_corpus(7), gen.plan_corpus(7))
        self.assertNotEqual(gen.plan_corpus(7), gen.plan_corpus(8))

    def test_every_language_has_both_kinds(self):
        plan = gen.plan_corpus()
        for lang in languages.LANGUAGES:
            kinds = {p["kind"] for p in plan if p["code"] == lang["code"]}
            self.assertEqual(kinds, {"mono", "conv"}, lang["code"])

    def test_translation_rejects_english_echoes_and_preambles(self):
        src = "The meeting has been moved to the large room on the second floor."
        self.assertFalse(gen.translation_ok(src, src.replace(".", "!"), "LATIN"))
        self.assertFalse(gen.translation_ok(
            src, "Here is the translation: The meeting moved to the large room.", "LATIN"))
        self.assertFalse(gen.translation_ok(
            src, "The meeting has been moved to the large room upstairs", "LATIN"))
        self.assertTrue(gen.translation_ok(
            src, "La reunión se ha trasladado a la sala grande del segundo piso.", "LATIN"))
        # Loanword-heavy Filipino / Indonesian is still a translation.
        for src_en, text in (
                ("Did you save a backup of your files?", "Nag-save ka ba ng backup ng mga files mo?"),
                ("The manager from the design department.", "Ang manager mula sa design department."),
                ("Is the professor strict about deadlines?",
                 "Mahigpit ba ang professor pagdating sa deadlines?"),
                ("Did you volunteer at the beach cleanup?", "Nag-volunteer ka ba sa beach cleanup?"),
                ("Did you save a backup of your files?",
                 "Apakah kamu sudah menyimpan backup file-file kamu?")):
            self.assertTrue(gen.translation_ok(src_en, text, "LATIN"), text)
        # Cognates alone don't make a line English.
        self.assertTrue(gen.translation_ok("The museum has a large collection.",
                                           "Das Museum hat eine große Sammlung.", "LATIN"))

    def test_clean_reply(self):
        self.assertEqual(gen.clean_reply("<think>hmm</think>\nHola."), "Hola.")
        self.assertEqual(gen.clean_reply("Here is the translation:\n\u201cHola a todos.\u201d"),
                         "Hola a todos.")
        self.assertEqual(gen.clean_reply(""), "")

    def test_synthesize_dedupes_identical_lines(self):
        seen = []

        async def fake(text, voice, pitch, sem):
            seen.append((text, voice, pitch))

        orig = gen._tts_one
        gen._tts_one = fake
        try:
            paths = gen.synthesize([("hi", "v", "+0Hz"), ("hi", "v", "+0Hz"), ("yo", "v", "+0Hz")])
        finally:
            gen._tts_one = orig
        self.assertEqual(sorted(seen), [("hi", "v", "+0Hz"), ("yo", "v", "+0Hz")])
        self.assertEqual(paths[0], paths[1])
        self.assertNotEqual(paths[0], paths[2])

    def test_script_checks(self):
        self.assertEqual(gen.script_ratio("Привет мир", "CYRILLIC"), 1.0)
        self.assertEqual(gen.script_ratio("今日はいい天気", "CJK"), 1.0)
        self.assertTrue(gen.translation_ok("Hello there.", "Bonjour à tous.", "LATIN"))
        self.assertFalse(gen.translation_ok("Hello there.", "Hello there.", "LATIN"))
        self.assertFalse(gen.translation_ok("Hello.", "Hello.", "CYRILLIC"))
        self.assertFalse(gen.translation_ok("Hello.", "", "LATIN"))

    def test_pick_speakers_reuses_voice_at_other_pitches(self):
        lang = {"code": "xx", "locales": ["xx-XX"]}
        item = {"speakers": 3}
        sp = gen.pick_speakers(item, lang, {"xx-XX": ["only-voice"]}, random.Random(1))
        self.assertEqual(len(set(sp)), 3)
        self.assertEqual({v for v, _ in sp}, {"only-voice"})

    def test_conversation_script_alternates_speakers(self):
        bank = {"sentences": [], "exchanges": [list(e) for e in gen.EXCHANGES],
                "reactions": list(gen.REACTIONS)}
        lines = gen.script_lines({"kind": "conv"}, bank, random.Random(3), ["a", "b", "c"])
        self.assertEqual({s for s, _, _ in lines}, {0, 1, 2})
        # Within an exchange the answer never comes from the asker.
        for (s1, q, _), (s2, a, _) in zip(lines, lines[1:]):
            if q in {e[0] for e in gen.EXCHANGES} and a in {e[1] for e in gen.EXCHANGES}:
                self.assertNotEqual(s1, s2)


class TestNoGpuTranslation(unittest.TestCase):
    """CPU-only Ollama translation phase and the committed translation files."""

    def test_cpu_call_sends_num_gpu_zero_and_retries(self):
        from unittest import mock
        from plaude_local import summarize

        import urllib.error

        server_error = summarize.SummarizeError("HTTP 500")
        server_error.__cause__ = urllib.error.HTTPError("u", 500, "err", {}, None)
        replies = [server_error, {"response": " Hallo "}]

        def fake_post(url, payload, timeout):
            fake_post.payload = payload
            r = replies.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        slept = []
        with mock.patch.object(summarize, "_post_json", side_effect=fake_post):
            call = gen.make_ollama_call("m", "http://x:1", cpu=True, capabilities=[],
                                        sleep=slept.append)
            self.assertEqual(call("p"), "Hallo")
        self.assertEqual(fake_post.payload["options"]["num_gpu"], 0)
        self.assertEqual(fake_post.payload["model"], "m")
        self.assertEqual(len(slept), 1)  # one backoff before the successful retry

    def test_auto_device_leaves_gpu_choice_to_ollama(self):
        from unittest import mock
        from plaude_local import summarize

        with mock.patch.object(summarize, "_post_json",
                               return_value={"response": "ok"}) as post:
            gen.make_ollama_call("m", "http://x:1", cpu=False, capabilities=[])("p")
        self.assertNotIn("num_gpu", post.call_args.args[1]["options"])

    def _error(self, cause):
        from plaude_local import summarize
        err = summarize.SummarizeError("failed")
        err.__cause__ = cause
        return err

    def test_call_gives_up_after_retries(self):
        from unittest import mock
        import urllib.error
        from plaude_local import summarize

        down = self._error(urllib.error.URLError(ConnectionRefusedError()))
        slept = []
        with mock.patch.object(summarize, "_post_json", side_effect=down) as post:
            call = gen.make_ollama_call("m", "http://x:1", retries=3, capabilities=[],
                                        sleep=slept.append)
            with self.assertRaises(summarize.SummarizeError):
                call("p")
        self.assertEqual(post.call_count, 3)
        self.assertEqual(len(slept), 2)

    def test_non_transient_errors_fail_fast(self):
        from unittest import mock
        import socket
        import urllib.error
        from plaude_local import summarize

        for cause in (urllib.error.HTTPError("u", 404, "model not found", {}, None),
                      socket.timeout("timed out"), None):
            with mock.patch.object(summarize, "_post_json",
                                   side_effect=self._error(cause)) as post:
                call = gen.make_ollama_call("m", "http://x:1", capabilities=[],
                                            sleep=lambda s: None)
                with self.assertRaises(summarize.SummarizeError):
                    call("p")
            self.assertEqual(post.call_count, 1, repr(cause))

    def test_empty_reply_fails_fast(self):
        from unittest import mock
        from plaude_local import summarize

        with mock.patch.object(summarize, "_post_json", return_value={"response": ""}) as post:
            with self.assertRaises(summarize.SummarizeError):
                gen.make_ollama_call("m", "http://x:1", capabilities=[])("p")
        self.assertEqual(post.call_count, 1)

    def test_payload_caps_output_and_disables_thinking(self):
        from unittest import mock
        from plaude_local import summarize

        with mock.patch.object(summarize, "_post_json", return_value={"response": "ok"}) as post:
            gen.make_ollama_call("m", "http://x:1", capabilities=["completion", "thinking"],
                                 temperature=0.7)("p")
        payload = post.call_args.args[1]
        self.assertIs(payload["think"], False)
        self.assertEqual(payload["options"]["num_predict"], 4096)
        self.assertEqual(payload["options"]["temperature"], 0.7)
        with mock.patch.object(summarize, "_post_json", return_value={"response": "ok"}) as post:
            gen.make_ollama_call("m", "http://x:1", capabilities=["completion"])("p")
        self.assertNotIn("think", post.call_args.args[1])

    def _in_temp_translations(self):
        import tempfile
        from unittest import mock
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        patch = mock.patch.object(gen, "TRANSLATIONS", Path(d.name))
        patch.start()
        self.addCleanup(patch.stop)
        return Path(d.name)

    def test_translation_file_tied_to_english_source(self):
        folder = self._in_temp_translations()
        gen.write_json_atomic(folder / "de.json", {"source_sha1": gen.source_sha1(), "x": 1})
        self.assertEqual(gen.load_translation("de")["x"], 1)
        gen.write_json_atomic(folder / "de.json", {"source_sha1": "stale", "x": 1})
        self.assertIsNone(gen.load_translation("de"))
        (folder / "de.json").write_text("{truncated", encoding="utf-8")
        self.assertIsNone(gen.load_translation("de"))
        self.assertEqual([p.name for p in folder.iterdir()], ["de.json"])  # no temp files left

    def test_partial_progress_resumes_and_retries_only_bad_lines(self):
        from unittest import mock
        from plaude_local import summarize

        folder = self._in_temp_translations()
        src = gen.english_lines()
        lang = {"code": "de", "name": "German", "script": "LATIN"}
        good = ["Guten Morgen, liebe Leute, willkommen hier."] * len(src)
        lines = list(good)
        lines[0] = src[0]  # an English echo that must be re-translated
        gen.write_json_atomic(folder / "de.partial.json",
                              {"source_sha1": gen.source_sha1(), "lines": lines})
        retry = mock.Mock(return_value="Guten Morgen zusammen.")
        with mock.patch.object(summarize, "translate_lines") as batch:
            data = gen.translate_language(lang, call=None, model="m", retry_call=retry)
        batch.assert_not_called()          # resumed from the partial file
        self.assertEqual(retry.call_count, 1)  # only the one bad line
        self.assertEqual(data["sentences"][0], "Guten Morgen zusammen.")
        self.assertEqual(data["source_sha1"], gen.source_sha1())
        self.assertFalse((folder / "de.partial.json").exists())
        self.assertTrue((folder / "de.json").exists())

    def test_too_many_bad_lines_keeps_progress(self):
        from unittest import mock
        from plaude_local import summarize

        folder = self._in_temp_translations()
        src = gen.english_lines()
        lang = {"code": "de", "name": "German", "script": "LATIN"}
        with mock.patch.object(summarize, "translate_lines", return_value=list(src)), \
             mock.patch.object(summarize, "translate_text", side_effect=lambda t, c, **k: t):
            with self.assertRaises(RuntimeError):
                gen.translate_language(lang, call=lambda p: p, model="m")
        self.assertTrue((folder / "de.partial.json").exists())
        self.assertFalse((folder / "de.json").exists())

    def test_garbled_partial_is_redone_in_batches(self):
        from unittest import mock
        from plaude_local import summarize

        folder = self._in_temp_translations()
        src = gen.english_lines()
        lang = {"code": "de", "name": "German", "script": "LATIN"}
        gen.write_json_atomic(folder / "de.partial.json",
                              {"source_sha1": gen.source_sha1(), "lines": list(src)})
        good = ["Guten Morgen, liebe Leute, willkommen hier."] * len(src)
        with mock.patch.object(summarize, "translate_lines", return_value=good) as batch:
            data = gen.translate_language(lang, call=lambda p: "", model="big")
        batch.assert_called_once()
        self.assertEqual(data["model"], "big")

    def test_fallback_model_redoes_a_failed_language(self):
        from unittest import mock
        from plaude_local import summarize

        folder = self._in_temp_translations()
        src = gen.english_lines()
        good = ["Guten Morgen, liebe Leute, willkommen hier."] * len(src)

        def batch(lines, call, target_language):
            return call("batch")

        made = {}

        def fake_calls(name, url, *, cpu, capabilities, temperature=0.0, **kw):
            reply = list(src) if name == "small" else good  # small echoes English
            return lambda prompt: reply

        with mock.patch.object(gen, "model_capabilities", return_value=[]), \
             mock.patch.object(gen, "make_ollama_call", side_effect=fake_calls), \
             mock.patch.object(summarize, "translate_lines", side_effect=batch), \
             mock.patch.object(summarize, "translate_text", side_effect=lambda t, c, **k: t):
            rc = gen.main(["--phase", "translate", "--languages", "de",
                           "--model", "small", "--fallback-model", "big"])
        self.assertEqual(rc, 0)
        data = gen.load_translation("de")
        self.assertEqual(data["sentences"][0], good[0])
        self.assertEqual(data["model"], "small + big")
        self.assertFalse((folder / "de.partial.json").exists())

    def test_english_needs_no_model(self):
        folder = self._in_temp_translations()
        data = gen.translate_language({"code": "en", "name": "English", "script": "LATIN"},
                                      call=None)
        self.assertEqual(data["sentences"], list(gen.SENTENCES))
        self.assertIsNone(data["model"])
        self.assertTrue((folder / "en.json").exists())

    def test_unknown_language_code_is_rejected(self):
        with self.assertRaises(SystemExit):
            gen.main(["--phase", "translate", "--languages", "xx"])

    def test_audio_phase_refuses_missing_translations(self):
        import tempfile
        from unittest import mock

        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(gen, "TRANSLATIONS", Path(d) / "none"), \
             mock.patch.object(gen, "list_voices", return_value={}), \
             mock.patch.object(gen, "assemble") as assemble:
            rc = gen.main(["--phase", "audio", "--languages", "de",
                           "--corpus", str(Path(d) / "corpus")])
        self.assertEqual(rc, 1)
        assemble.assert_not_called()

    def test_committed_translation_files_are_complete(self):
        files = sorted(f for f in (REG / "translations").glob("*.json")
                       if not f.name.endswith(".partial.json"))
        if not files:
            self.skipTest("no committed translations yet")
        by_code = languages.BY_CODE
        for f in files:
            data = json.loads(f.read_text(encoding="utf-8"))
            self.assertIn(f.stem, by_code, f.name)
            self.assertEqual(len(data["sentences"]), len(gen.SENTENCES), f.name)
            self.assertEqual(len(data["exchanges"]), len(gen.EXCHANGES), f.name)
            self.assertEqual(len(data["reactions"]), len(gen.REACTIONS), f.name)
            self.assertEqual(data.get("source_sha1"), gen.source_sha1(),
                             f"{f.name} was made from different English source lines")
            lines = data["sentences"] + [l for e in data["exchanges"] for l in e] + data["reactions"]
            usable = [l for l in lines if l]
            self.assertGreaterEqual(len(usable), 0.95 * len(lines), f.name)
            script = by_code[f.stem]["script"]
            for line in usable:
                self.assertGreaterEqual(gen.script_ratio(line, script), 0.6, (f.name, line))


class TestLanguages(unittest.TestCase):
    def test_codes_unique_and_tiers_valid(self):
        codes = [l["code"] for l in languages.ALL_LANGUAGES]
        self.assertEqual(len(codes), len(set(codes)))
        for l in languages.ALL_LANGUAGES:
            self.assertIn(l["tier"], languages.THRESHOLDS)
            self.assertTrue(l["locales"])

    def test_corpus_focuses_on_asian_languages(self):
        codes = [l["code"] for l in languages.LANGUAGES]
        self.assertEqual(len(codes), 21)
        self.assertEqual(len(set(codes)), 21)
        self.assertLessEqual(set(codes), set(languages.BY_CODE))
        self.assertEqual(set(codes[-5:]), {"en", "es", "fr", "de", "ru"})

    def test_accepted_codes_include_related_languages(self):
        self.assertEqual(languages.accepted_codes("de"), {"de"})
        self.assertIn("id", languages.accepted_codes("ms"))


class TestRunner(unittest.TestCase):
    ENTRY = {"id": "de-mono-01", "code": "de", "kind": "mono", "tier": "A",
             "damage": {"hum_hz": 60}}

    def _res(self, **kw):
        base = {"exit_code": 0, "language": "de", "cer": 0.05, "recall": 0.9, "hum_hz": 60}
        base.update(kw)
        return base

    def test_pass(self):
        self.assertEqual(rr.evaluate(self._res(), self.ENTRY, "clean", None), [])
        self.assertEqual(rr.evaluate(self._res(), self.ENTRY, "damaged", None), [])

    def test_failures(self):
        self.assertEqual(rr.evaluate(self._res(exit_code=6), self.ENTRY, "clean", None),
                         ["exit code 6"])
        self.assertTrue(rr.evaluate(self._res(language="nl"), self.ENTRY, "clean", None))
        self.assertTrue(rr.evaluate(self._res(cer=0.9), self.ENTRY, "clean", None))
        self.assertTrue(rr.evaluate(self._res(recall=0.1), self.ENTRY, "clean", None))
        self.assertTrue(rr.evaluate(self._res(hum_hz=None), self.ENTRY, "damaged", None))

    def test_related_language_accepted(self):
        entry = dict(self.ENTRY, code="ms")
        self.assertEqual(rr.evaluate(self._res(language="id"), entry, "clean", None), [])

    def test_low_resource_tier_only_checks_exit_and_baseline(self):
        entry = dict(self.ENTRY, code="lo", tier="C")
        bad = self._res(language="th", cer=0.95, recall=0.0)
        self.assertEqual(rr.evaluate(bad, entry, "clean", None), [])
        self.assertTrue(rr.evaluate(bad, entry, "clean", {"cer": 0.5, "recall": 0.4}))

    def test_baseline_regression(self):
        base = {"cer": 0.02, "recall": 0.95}
        self.assertEqual(rr.evaluate(self._res(cer=0.05, recall=0.9), self.ENTRY, "clean", base), [])
        self.assertTrue(rr.evaluate(self._res(cer=0.10), self.ENTRY, "clean", base))
        self.assertTrue(rr.evaluate(self._res(recall=0.80), self.ENTRY, "clean", base))

    def test_commands_have_flag_parity(self):
        work = Path("w")
        for variant in ("clean", "damaged"):
            py = rr.build_command("python", Path("a.opus"), work, variant=variant,
                                  model="large-v3", engine="llm", translate_model="m")
            ps = rr.build_command("powershell", Path("a.opus"), work, variant=variant,
                                  model="large-v3", engine="llm", translate_model="m")
            py_flags = [a for a in py[3:] if a.startswith("-")]  # skip "python -m plaude_local"
            ps_flags = [a for a in ps if a.startswith("-") and a not in
                        ("-NoProfile", "-ExecutionPolicy", "-File")]
            norm = lambda f: f.lstrip("-").replace("-", "").lower()  # noqa: E731
            alias = {"f": "format", "o": "output", "y": "yes", "q": "quiet", "m": "model"}
            self.assertEqual(sorted(alias.get(norm(f), norm(f)) for f in py_flags),
                             sorted(alias.get(norm(f), norm(f)) for f in ps_flags))
            self.assertEqual("--repair" in py, variant == "damaged")

    def test_timeout_kills_the_process_tree(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "run.log"
            code = rr.run_with_timeout([sys.executable, "-c", "import time; time.sleep(60)"],
                                       log, timeout=1)
            self.assertEqual(code, -1)
            self.assertIn("timeout", log.read_text(encoding="utf-8"))

    def test_stratified_sample_spreads_languages_and_kinds(self):
        entries = [{"id": f"{c}-{k}-{i}", "code": c, "kind": k}
                   for c in "abcdef" for k in ("mono", "conv") for i in range(3)]
        pick = rr.stratified_sample(entries, 12, seed=1)
        self.assertEqual(len(pick), 12)
        self.assertEqual(len({e["code"] for e in pick}), 6)
        self.assertEqual({e["kind"] for e in pick}, {"mono", "conv"})
        self.assertEqual(len({e["id"] for e in pick}), 12)


class TestCommittedCorpus(unittest.TestCase):
    """Validates tests/regression/corpus (skipped until it has been generated)."""

    @classmethod
    def setUpClass(cls):
        path = REG / "corpus" / "manifest.json"
        if not path.exists():
            raise unittest.SkipTest("corpus not generated")
        cls.items = json.loads(path.read_text(encoding="utf-8"))["items"]

    def test_items_belong_to_the_plan(self):
        planned = {p["id"] for p in gen.plan_corpus()}
        ids = [i["id"] for i in self.items]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertLessEqual(set(ids), planned)

    def test_counts(self):
        if len(self.items) < 300:
            self.skipTest(f"partial corpus ({len(self.items)} of 300 items)")
        self.assertEqual(len(self.items), 300)
        self.assertEqual(sum(i["kind"] == "mono" for i in self.items), 150)
        self.assertEqual(sum(i["kind"] == "conv" for i in self.items), 150)

    def test_files_exist_and_durations_in_range(self):
        for i in self.items:
            self.assertTrue((REG / "corpus" / i["file"]).is_file(), i["file"])
            self.assertTrue((REG / "corpus" / i["damaged_file"]).is_file(), i["damaged_file"])
            self.assertTrue(120 <= i["duration_s"] <= 600, (i["id"], i["duration_s"]))

    def test_references_and_turns(self):
        for i in self.items:
            self.assertTrue(i["text"].strip() and i["reference_en"].strip(), i["id"])
            if i["kind"] == "conv":
                self.assertGreaterEqual(len({t["speaker"] for t in i["turns"]}), 2, i["id"])
                ends = [t["end"] for t in i["turns"]]
                self.assertEqual(ends, sorted(ends), i["id"])


if __name__ == "__main__":
    unittest.main()
