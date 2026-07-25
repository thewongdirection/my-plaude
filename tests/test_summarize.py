"""Regression tests for local summarization (logic + transport seams)."""

import unittest
from unittest import mock

from plaude_local import summarize
from plaude_local.summarize import _chunk, summarize_text, build_prompt, SummarizeError


class TestChunk(unittest.TestCase):
    def test_short_text_single_chunk(self):
        self.assertEqual(_chunk("hello", 100), ["hello"])

    def test_long_text_splits(self):
        text = "\n".join(f"line {i}" for i in range(100))
        chunks = _chunk(text, 50)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 60 for c in chunks))  # ~<= max, break-aware
        # No content lost.
        self.assertEqual("".join(c.replace("\n", "") for c in chunks).count("line"),
                         text.replace("\n", "").count("line"))

    def test_pathological_max_chars_does_not_hang(self):
        # Regression: max_chars <= 1 (reachable via --summarize-max-chars) must
        # not infinite-loop. These return promptly.
        self.assertEqual("".join(_chunk("abc", 0)), "abc")  # terminates
        self.assertTrue(_chunk("\n\nab", 1))  # terminates, non-empty
        # And still loses no visible characters for a normal small value.
        joined = "".join(_chunk("hello world foo bar", 1))
        for ch in "helloworldfoobar":
            self.assertIn(ch, joined)

    def test_no_empty_chunks(self):
        text = "aaaa\n\n\n\n\n\nbbbb\n\n\n\n\n\ncccc"
        self.assertTrue(all(c.strip() for c in _chunk(text, 5)))


class TestSummarizeText(unittest.TestCase):
    def test_single_shot(self):
        calls = []
        def fake(prompt):
            calls.append(prompt)
            return "SUMMARY"
        out = summarize_text("a short transcript", fake, max_chars=1000)
        self.assertEqual(out, "SUMMARY")
        self.assertEqual(len(calls), 1)

    def test_map_reduce_for_long_text(self):
        text = "\n".join(f"sentence number {i} here" for i in range(200))
        calls = []
        def fake(prompt):
            calls.append(prompt)
            return "partial"
        out = summarize_text(text, fake, max_chars=100)
        # Several chunk calls + at least one combine call.
        self.assertGreater(len(calls), 1)
        self.assertTrue(any("partial summaries" in c.lower() for c in calls))
        self.assertIsInstance(out, str)

    def test_empty_raises(self):
        with self.assertRaises(SummarizeError):
            summarize_text("   ", lambda p: "x")


class TestPrompt(unittest.TestCase):
    def test_preserves_language_instruction(self):
        self.assertIn("Preserve the original language", build_prompt("hi"))

    def test_combine_variant(self):
        self.assertIn("partial summaries", build_prompt("x", combine=True).lower())


class TestBackendCalls(unittest.TestCase):
    def test_ollama_call_reads_response(self):
        with mock.patch.object(summarize, "_post_json",
                               return_value={"response": "  hello  "}) as m:
            out = summarize._ollama_call("p", "llama3.1", "http://h:11434", 10)
        self.assertEqual(out, "hello")
        self.assertTrue(m.call_args.args[0].endswith("/api/generate"))

    def test_llamacpp_call_reads_content(self):
        with mock.patch.object(summarize, "_post_json",
                               return_value={"content": "world"}) as m:
            out = summarize._llamacpp_call("p", None, "http://h:8080", 10)
        self.assertEqual(out, "world")
        self.assertTrue(m.call_args.args[0].endswith("/completion"))

    def test_ollama_empty_raises(self):
        with mock.patch.object(summarize, "_post_json", return_value={}):
            with self.assertRaises(SummarizeError):
                summarize._ollama_call("p", "m", "http://h", 10)


class TestPostJson(unittest.TestCase):
    class _Resp:
        status = 200

        def __init__(self, body):
            self._body = body

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def test_non_json_response_wrapped_as_summarize_error(self):
        # Regression: a reachable server returning a non-JSON 200 body must
        # raise SummarizeError, not an uncaught JSONDecodeError.
        with mock.patch("urllib.request.urlopen",
                        return_value=self._Resp(b"<html>proxy error</html>")):
            with self.assertRaises(SummarizeError):
                summarize._post_json("http://x", {"a": 1}, 5)

    def test_valid_json_parsed(self):
        with mock.patch("urllib.request.urlopen",
                        return_value=self._Resp(b'{"response":"ok"}')):
            self.assertEqual(
                summarize._post_json("http://x", {}, 5), {"response": "ok"})

    def test_unreachable_server_wrapped_as_summarize_error(self):
        # A connection failure (URLError) must surface as SummarizeError with a
        # helpful message, not an uncaught urllib exception.
        import urllib.error
        with mock.patch("urllib.request.urlopen",
                        side_effect=urllib.error.URLError("connection refused")):
            with self.assertRaises(SummarizeError) as ctx:
                summarize._post_json("http://localhost:11434", {"a": 1}, 5)
        self.assertIn("could not reach", str(ctx.exception).lower())

    def test_non_object_json_wrapped_as_summarize_error(self):
        # Regression: valid JSON that is not an object (array/string/number/null)
        # must raise SummarizeError, not an AttributeError in the callers.
        for body in (b"[1,2,3]", b'"OK"', b"42", b"null"):
            with mock.patch("urllib.request.urlopen",
                            return_value=self._Resp(body)):
                with self.assertRaises(SummarizeError):
                    summarize._post_json("http://x", {}, 5)


class TestDetect(unittest.TestCase):
    def test_detect_prefers_ollama(self):
        with mock.patch.object(summarize, "ollama_available", return_value=True), \
             mock.patch.object(summarize, "llamacpp_available", return_value=True):
            self.assertEqual(summarize.detect_backend(), "ollama")

    def test_detect_falls_back_to_llamacpp(self):
        with mock.patch.object(summarize, "ollama_available", return_value=False), \
             mock.patch.object(summarize, "llamacpp_available", return_value=True):
            self.assertEqual(summarize.detect_backend(), "llamacpp")

    def test_detect_none(self):
        with mock.patch.object(summarize, "ollama_available", return_value=False), \
             mock.patch.object(summarize, "llamacpp_available", return_value=False):
            self.assertIsNone(summarize.detect_backend())


class TestSummarizeEntryPoint(unittest.TestCase):
    def test_auto_with_no_server_raises_with_remedy(self):
        with mock.patch.object(summarize, "detect_backend", return_value=None):
            with self.assertRaises(SummarizeError) as ctx:
                summarize.summarize("text", backend="auto")
        self.assertIn("ollama", str(ctx.exception).lower())

    def test_ollama_end_to_end_mocked(self):
        with mock.patch.object(summarize, "_post_json",
                               return_value={"response": "* point one"}):
            out = summarize.summarize("hello transcript", backend="ollama")
        self.assertEqual(out, "* point one")

    def test_unknown_backend_raises(self):
        with self.assertRaises(SummarizeError):
            summarize.summarize("t", backend="bogus")


class TestTranslateLines(unittest.TestCase):
    def test_parses_numbered_lines(self):
        out = summarize.translate_lines(
            ["你好", "世界"], lambda p: "1. Hello\n2. World", target_language="English")
        self.assertEqual(out, ["Hello", "World"])

    def test_strips_reasoning_and_marks_unmatched_empty(self):
        out = summarize.translate_lines(
            ["a", "b"], lambda p: "<think>plan</think>\n1. Hi", target_language="English")
        self.assertEqual(out, ["Hi", ""])

    def test_batches_by_max_chars(self):
        calls = []

        def fake(p):
            calls.append(p)
            return "1. x"

        summarize.translate_lines(["aaaa", "bbbb", "cccc"], fake,
                                  target_language="English", max_chars=8)
        self.assertGreater(len(calls), 1)

    def test_batches_by_max_lines(self):
        # Regression: oversized batches overflow the LLM context and come back
        # empty. Lines-per-request must be capped even when max_chars is huge.
        calls = []

        def fake(p):
            calls.append(p)
            # echo back a numbered line for each input line in the prompt
            n = len([ln for ln in p.splitlines() if ln[:1].isdigit()])
            return "\n".join(f"{i + 1}. t{i}" for i in range(n))

        lines = [f"L{i}" for i in range(100)]
        out = summarize.translate_lines(lines, fake, target_language="English",
                                        max_chars=10_000, max_lines=40)
        self.assertGreaterEqual(len(calls), 3)          # 100 / 40 -> 3 batches
        self.assertEqual(len(out), 100)
        self.assertTrue(all(out))                       # nothing dropped

    def test_recovers_when_large_batch_underparses(self):
        # A batch that returns nothing until it is small enough must be split and
        # retried, never silently yielding all-empty output.
        def fake(p):
            numbered = [ln for ln in p.splitlines() if ln[:1].isdigit()]
            if len(numbered) > 3:
                return "sorry, I cannot"      # truncated/misformatted -> no match
            return "\n".join(f"{i + 1}. ok{i}" for i in range(len(numbered)))

        lines = [f"L{i}" for i in range(8)]
        out = summarize.translate_lines(lines, fake, target_language="English",
                                        max_chars=10_000, max_lines=8)
        self.assertEqual(len(out), 8)
        self.assertTrue(all(out))              # split-on-underparse filled them all


class TestOllamaCtx(unittest.TestCase):
    def test_sets_fixed_num_ctx_above_default(self):
        # A FIXED num_ctx (not sized per prompt) — a varying value would force
        # Ollama to reload the model every call. Must beat the truncating ~4k
        # default and be identical regardless of prompt length.
        seen = []

        def fake_post(url, payload, timeout):
            seen.append(payload["options"]["num_ctx"])
            return {"response": "ok"}

        with mock.patch.object(summarize, "_post_json", side_effect=fake_post):
            summarize._ollama_call("x" * 20_000, "m", "http://h", 10)
            summarize._ollama_call("short", "m", "http://h", 10)
        self.assertEqual(seen[0], seen[1])            # same for any prompt length
        self.assertGreater(seen[0], 4096)

    def test_timeout_wrapped_as_summarize_error(self):
        # Regression: a socket TimeoutError must surface as SummarizeError (so the
        # run degrades gracefully), not crash the whole dashboard.
        with mock.patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            with self.assertRaises(summarize.SummarizeError):
                summarize._post_json("http://x", {"a": 1}, 5)


class TestDefaultOllamaModel(unittest.TestCase):
    def test_returns_first_installed_model(self):
        payload = {"models": [{"name": "gemma4:latest"}, {"name": "llama3.1"}]}
        with mock.patch.object(summarize.urllib.request, "urlopen") as uo:
            uo.return_value.__enter__.return_value.read.return_value = \
                __import__("json").dumps(payload).encode()
            self.assertEqual(summarize.default_ollama_model(), "gemma4:latest")

    def test_none_when_unreachable(self):
        with mock.patch.object(summarize.urllib.request, "urlopen",
                               side_effect=OSError("down")):
            self.assertIsNone(summarize.default_ollama_model())


class TestDefaultUrls(unittest.TestCase):
    def test_defaults_use_ipv4_not_localhost(self):
        # Regression: "localhost" makes the PowerShell port's HTTP client stall on
        # IPv6 ::1 and time out backend detection. Keep both defaults on 127.0.0.1
        # so the local LLM is detected reliably (parity with the PS defaults).
        self.assertEqual(summarize.DEFAULT_OLLAMA_URL, "http://127.0.0.1:11434")
        self.assertEqual(summarize.DEFAULT_LLAMACPP_URL, "http://127.0.0.1:8080")
        self.assertNotIn("localhost", summarize.DEFAULT_OLLAMA_URL)


class TestListOllamaModels(unittest.TestCase):
    def test_lists_all_installed_names(self):
        payload = {"models": [{"name": "qwen2.5:7b"}, {"name": "gemma4:latest"}]}
        with mock.patch.object(summarize.urllib.request, "urlopen") as uo:
            uo.return_value.__enter__.return_value.read.return_value = \
                __import__("json").dumps(payload).encode()
            self.assertEqual(summarize.list_ollama_models(),
                             ["qwen2.5:7b", "gemma4:latest"])

    def test_empty_when_unreachable(self):
        with mock.patch.object(summarize.urllib.request, "urlopen",
                               side_effect=OSError("down")):
            self.assertEqual(summarize.list_ollama_models(), [])


if __name__ == "__main__":
    unittest.main()
