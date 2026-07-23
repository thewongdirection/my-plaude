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


if __name__ == "__main__":
    unittest.main()
