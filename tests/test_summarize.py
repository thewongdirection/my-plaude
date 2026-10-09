"""Regression tests for local summarization (logic + transport seams)."""

import unittest
from unittest import mock

from plaude_local import summarize
def _chat(content):
    """A llama.cpp ``/v1/chat/completions`` reply carrying ``content``."""
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


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
                               return_value=_chat("world")) as m:
            out = summarize._llamacpp_call("p", None, "http://h:8080", 10)
        self.assertEqual(out, "world")
        self.assertTrue(m.call_args.args[0].endswith("/v1/chat/completions"))

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
        # Mock the installed-model lookup too, so the test never depends on a
        # live Ollama server (it used to fail whenever none was running).
        with mock.patch.object(summarize, "_post_json",
                               return_value={"response": "* point one"}),              mock.patch.object(summarize, "default_ollama_model",
                               return_value="llama3.1"):
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


class TestLlamacppRequest(unittest.TestCase):
    """The llama.cpp ``/v1/chat/completions`` request/response contract (mocked transport)."""

    def test_payload_shape(self):
        with mock.patch.object(summarize, "_post_json",
                               return_value=_chat("x")) as m:
            summarize._llamacpp_call("the prompt", None, "http://h:8080", 7)
        url, payload, timeout = m.call_args.args
        self.assertEqual(url, "http://h:8080/v1/chat/completions")
        self.assertEqual(payload["messages"][0]["content"], "the prompt")
        self.assertIs(payload["stream"], False)
        self.assertIn("max_tokens", payload)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertIn("temperature", payload)
        self.assertEqual(timeout, 7)

    def test_payload_has_no_ollama_only_fields(self):
        # llama.cpp binds one model at launch: no model name / num_ctx options.
        with mock.patch.object(summarize, "_post_json",
                               return_value=_chat("x")) as m:
            summarize._llamacpp_call("p", "ignored-name", "http://h", 5)
        payload = m.call_args.args[1]
        self.assertNotIn("model", payload)
        self.assertNotIn("options", payload)

    def test_trailing_slash_in_url_is_normalised(self):
        with mock.patch.object(summarize, "_post_json",
                               return_value=_chat("x")) as m:
            summarize._llamacpp_call("p", None, "http://h:8080/", 5)
        self.assertEqual(m.call_args.args[0], "http://h:8080/v1/chat/completions")

    def test_empty_or_missing_content_raises(self):
        for resp in ({}, _chat(""), _chat(None), {"response": "x"}, {"choices": []},
                     _chat("<think>only reasoning</think>")):
            with mock.patch.object(summarize, "_post_json", return_value=resp):
                with self.assertRaises(SummarizeError) as ctx:
                    summarize._llamacpp_call("p", None, "http://h", 5)
            self.assertIn("llama.cpp", str(ctx.exception))

    def test_content_is_stripped(self):
        with mock.patch.object(summarize, "_post_json",
                               return_value=_chat("\n  - a\n  - b \n")):
            self.assertEqual(summarize._llamacpp_call("p", None, "http://h", 5),
                             "- a\n  - b")


class TestLlamacppOutputCleaning(unittest.TestCase):
    """Regression: raw /completion leaked chat-template/reasoning tokens."""

    def test_leaked_channel_markers_removed(self):
        raw = "Summary:\n- a\n<|channel>thought\n<channel|>Summary:\n- a"
        with mock.patch.object(summarize, "_post_json", return_value=_chat(raw)):
            out = summarize._llamacpp_call("p", None, "http://h", 5)
        self.assertNotIn("<|channel>", out)
        self.assertNotIn("<channel|>", out)
        self.assertNotIn("thought", out)
        self.assertTrue(out.endswith("- a"))

    def test_think_blocks_removed(self):
        with mock.patch.object(summarize, "_post_json",
                               return_value=_chat("<think>plan\nmore</think>\nAnswer")):
            self.assertEqual(summarize._llamacpp_call("p", None, "http://h", 5), "Answer")

    def test_clean_helper_leaves_plain_text_alone(self):
        self.assertEqual(summarize._clean_llm_text("  - a\n  - b  "), "- a\n  - b")

    def test_reasoning_content_is_never_used_as_the_answer(self):
        # A thinking model that ran out of tokens returns empty content plus
        # reasoning_content; that must be an error, not the chain of thought.
        reply = {"choices": [{"message": {"content": "",
                                          "reasoning_content": "thinking..."}}]}
        with mock.patch.object(summarize, "_post_json", return_value=reply):
            with self.assertRaises(SummarizeError):
                summarize._llamacpp_call("p", None, "http://h", 5)


def _chat_len(content):
    """A reply the server cut off at max_tokens (finish_reason == 'length')."""
    r = _chat(content)
    r["choices"][0]["finish_reason"] = "length"
    return r


class TestLlamacppTruncation(unittest.TestCase):
    """Regression: a reply cut off at max_tokens must never pass as complete."""

    def test_truncated_reply_raises_with_partial(self):
        with mock.patch.object(summarize, "_post_json",
                               return_value=_chat_len("- one\n- tw")):
            with self.assertRaises(summarize.TruncatedReply) as ctx:
                summarize._llamacpp_call("p", None, "http://h", 5)
        self.assertEqual(ctx.exception.partial, "- one\n- tw")
        self.assertIsInstance(ctx.exception, SummarizeError)   # generic handlers still work
        self.assertIn("truncated", str(ctx.exception))

    def test_normal_stop_is_not_flagged(self):
        reply = _chat("done")
        reply["choices"][0]["finish_reason"] = "stop"
        with mock.patch.object(summarize, "_post_json", return_value=reply):
            self.assertEqual(summarize._llamacpp_call("p", None, "http://h", 5), "done")

    def test_summarize_surfaces_truncation(self):
        with mock.patch.object(summarize, "_post_json", return_value=_chat_len("half")):
            with self.assertRaises(summarize.TruncatedReply):
                summarize.summarize("a transcript", backend="llamacpp")

    def test_translation_batch_is_split_and_retried(self):
        calls = []

        def fake(prompt):
            n = len([ln for ln in prompt.splitlines() if ln[:1].isdigit()])
            calls.append(n)
            if n > 2:
                raise summarize.TruncatedReply("cut", partial="1. par")
            return "\n".join(f"{i + 1}. t{i}" for i in range(n))

        out = summarize.translate_lines([f"L{i}" for i in range(8)], fake,
                                        target_language="English",
                                        max_chars=10_000, max_lines=8)
        self.assertTrue(all(out))
        self.assertEqual(len(out), 8)
        self.assertGreater(len(calls), 1)
        self.assertLessEqual(calls[-1], 2)

    def test_truncated_single_line_keeps_partial(self):
        def fake(prompt):
            raise summarize.TruncatedReply("cut", partial="1. partial text")

        out = summarize.translate_lines(["only"], fake, target_language="English")
        self.assertEqual(out, ["partial text"])

    def test_truncation_over_real_http(self):
        import json
        body = json.dumps(_chat_len("cut off")).encode()
        with _FakeLlamaServer(reply=lambda p: (200, body)) as srv:
            with self.assertRaises(summarize.TruncatedReply):
                summarize.summarize("t", backend="llamacpp", url=srv.url, timeout=5)


class TestLlamacppUnclosedReasoning(unittest.TestCase):
    def test_unclosed_think_is_dropped(self):
        self.assertEqual(summarize._clean_llm_text("Answer\n<think>The user wants..."),
                         "Answer")

    def test_unclosed_channel_is_dropped(self):
        self.assertEqual(summarize._clean_llm_text("<|channel>thought\nplanning..."), "")

    def test_stray_closers_removed(self):
        self.assertEqual(summarize._clean_llm_text("plan</think>Answer"), "planAnswer")
        self.assertEqual(summarize._clean_llm_text("<channel|>Answer"), "Answer")

    def test_balanced_block_still_removed_and_text_after_kept(self):
        self.assertEqual(summarize._clean_llm_text("<think>a</think>X<think>b</think>Y"), "XY")

    def test_only_unclosed_reasoning_is_an_error_not_the_answer(self):
        with mock.patch.object(summarize, "_post_json",
                               return_value=_chat("<think>still thinking")):
            with self.assertRaises(SummarizeError):
                summarize._llamacpp_call("p", None, "http://h", 5)


class TestLlamacppAvailability(unittest.TestCase):
    def test_health_endpoint_probed(self):
        with mock.patch.object(summarize, "_get_ok", return_value=True) as m:
            self.assertTrue(summarize.llamacpp_available("http://h:8080/"))
        self.assertEqual(m.call_args.args[0], "http://h:8080/health")

    def test_default_url_used(self):
        with mock.patch.object(summarize, "_get_ok", return_value=False) as m:
            self.assertFalse(summarize.llamacpp_available())
        self.assertEqual(m.call_args.args[0],
                         summarize.DEFAULT_LLAMACPP_URL + "/health")

    def test_loading_model_503_counts_as_unavailable(self):
        # llama-server answers /health with 503 while the model is still loading.
        import urllib.error
        err = urllib.error.HTTPError("http://h/health", 503, "Loading model", {}, None)
        with mock.patch.object(summarize.urllib.request, "urlopen", side_effect=err):
            self.assertFalse(summarize.llamacpp_available("http://h"))


class TestLlamacppResolve(unittest.TestCase):
    """Backend routing in ``_resolve_call`` for llama.cpp."""

    def test_llamacpp_routes_to_completion_with_default_url(self):
        with mock.patch.object(summarize, "_post_json",
                               return_value=_chat("ok")) as m:
            call = summarize._resolve_call("llamacpp", None, None, 9)
            self.assertEqual(call("p"), "ok")
        self.assertEqual(m.call_args.args[0],
                         summarize.DEFAULT_LLAMACPP_URL + "/v1/chat/completions")
        self.assertEqual(m.call_args.args[2], 9)

    def test_custom_url_is_honoured(self):
        with mock.patch.object(summarize, "_post_json",
                               return_value=_chat("ok")) as m:
            summarize._resolve_call("llamacpp", None, "http://gpu-box:9000", 5)("p")
        self.assertEqual(m.call_args.args[0], "http://gpu-box:9000/v1/chat/completions")

    def test_never_asks_ollama_for_a_model(self):
        # No Ollama lookups (and no model required) on the llama.cpp path.
        with mock.patch.object(summarize, "default_ollama_model") as dm, \
             mock.patch.object(summarize, "_post_json", return_value=_chat("ok")):
            summarize._resolve_call("llamacpp", None, None, 5)("p")
        dm.assert_not_called()

    def test_auto_picks_llamacpp_when_only_it_is_up(self):
        with mock.patch.object(summarize, "ollama_available", return_value=False), \
             mock.patch.object(summarize, "llamacpp_available", return_value=True), \
             mock.patch.object(summarize, "_post_json",
                               return_value=_chat("ok")) as m:
            summarize._resolve_call("auto", None, None, 5)("p")
        self.assertTrue(m.call_args.args[0].endswith("/v1/chat/completions"))

    def test_auto_prefers_ollama_when_both_are_up(self):
        with mock.patch.object(summarize, "ollama_available", return_value=True), \
             mock.patch.object(summarize, "llamacpp_available", return_value=True), \
             mock.patch.object(summarize, "default_ollama_model", return_value="m"), \
             mock.patch.object(summarize, "_post_json",
                               return_value={"response": "ok"}) as m:
            summarize._resolve_call("auto", None, None, 5)("p")
        self.assertTrue(m.call_args.args[0].endswith("/api/generate"))

    def test_auto_failure_message_mentions_llamacpp_remedy(self):
        with mock.patch.object(summarize, "detect_backend", return_value=None):
            with self.assertRaises(SummarizeError) as ctx:
                summarize._resolve_call("auto", None, None, 5)
        self.assertIn("llama-server", str(ctx.exception))

    def test_backends_constant_lists_llamacpp(self):
        self.assertIn("llamacpp", summarize.BACKENDS)


class TestLlamacppEntryPoints(unittest.TestCase):
    """summarize()/translate*() end to end over a mocked llama.cpp transport."""

    def test_summarize_end_to_end(self):
        with mock.patch.object(summarize, "_post_json",
                               return_value=_chat("* point")) as m:
            out = summarize.summarize("hello transcript", backend="llamacpp")
        self.assertEqual(out, "* point")
        self.assertIn("hello transcript", m.call_args.args[1]["messages"][0]["content"])

    def test_summarize_topics_prompt(self):
        with mock.patch.object(summarize, "_post_json",
                               return_value=_chat("t")) as m:
            summarize.summarize("x", backend="llamacpp", topics=True, max_words=50)
        self.assertIn("50 words", m.call_args.args[1]["messages"][0]["content"])

    def test_summarize_map_reduce(self):
        text = "\n".join(f"sentence number {i} here" for i in range(200))
        with mock.patch.object(summarize, "_post_json",
                               return_value=_chat("partial")) as m:
            summarize.summarize(text, backend="llamacpp", max_chars=100)
        prompts = [c.args[1]["messages"][0]["content"] for c in m.call_args_list]
        self.assertGreater(len(prompts), 2)
        self.assertTrue(any("partial summaries" in p.lower() for p in prompts))

    def test_translate_end_to_end(self):
        with mock.patch.object(summarize, "_post_json",
                               return_value=_chat(" Hola ")) as m:
            out = summarize.translate("hello", target_language="Spanish",
                                      backend="llamacpp")
        self.assertEqual(out, "Hola")
        self.assertIn("Spanish", m.call_args.args[1]["messages"][0]["content"])

    def test_translate_segments_stay_aligned(self):
        with mock.patch.object(summarize, "_post_json",
                               return_value=_chat("1. Hello\n2. World")):
            out = summarize.translate_segments(["你好", "世界"],
                                               target_language="English",
                                               backend="llamacpp")
        self.assertEqual(out, ["Hello", "World"])

    def test_server_error_surfaces_as_summarize_error(self):
        with mock.patch.object(summarize, "_post_json",
                               side_effect=SummarizeError("could not reach")):
            with self.assertRaises(SummarizeError):
                summarize.summarize("t", backend="llamacpp")


class _FakeLlamaServer:
    """A minimal in-process stand-in for ``llama-server`` on 127.0.0.1.

    Speaks just enough of the real HTTP contract (``GET /health``,
    ``POST /v1/chat/completions``) to exercise the actual urllib transport, request
    encoding and response parsing - offline, no model, no GPU.
    """

    def __init__(self, *, health_status=200, reply=None):
        import http.server
        import json
        import threading

        outer = self
        self.requests = []
        self.health_status = health_status
        # reply: callable(payload) -> (status, body_bytes)
        self.reply = reply or (lambda p: (200, json.dumps(_chat("ok")).encode()))

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):  # keep test output quiet
                pass

            def _send(self, status, body):
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                outer.requests.append(("GET", self.path, None))
                if self.path == "/health":
                    self._send(outer.health_status, b'{"status":"ok"}')
                else:
                    self._send(404, b"{}")

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                outer.requests.append(("POST", self.path, payload))
                if self.path != "/v1/chat/completions":
                    self._send(404, b"{}")
                    return
                status, body = outer.reply(payload)
                self._send(status, body)

        self._httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._httpd.server_address[1]}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *a):
        self._httpd.shutdown()
        self._httpd.server_close()
        return False


class TestLlamacppFakeServer(unittest.TestCase):
    """Real HTTP round trips against a fake llama-server (loopback only)."""

    def test_health_ok(self):
        with _FakeLlamaServer() as srv:
            self.assertTrue(summarize.llamacpp_available(srv.url))
        self.assertEqual(srv.requests[0][:2], ("GET", "/health"))

    def test_health_503_while_loading_is_unavailable(self):
        with _FakeLlamaServer(health_status=503) as srv:
            self.assertFalse(summarize.llamacpp_available(srv.url))

    def test_nothing_listening_is_unavailable(self):
        with _FakeLlamaServer() as srv:
            url = srv.url
        self.assertFalse(summarize.llamacpp_available(url))

    def test_detect_backend_finds_llamacpp(self):
        with _FakeLlamaServer() as srv:
            # Ollama URL points at a closed port; only llama.cpp answers.
            self.assertEqual(
                summarize.detect_backend("http://127.0.0.1:9", srv.url), "llamacpp")

    def test_summarize_over_real_http(self):
        import json
        with _FakeLlamaServer(
                reply=lambda p: (200, json.dumps(_chat(" * done ")).encode())
        ) as srv:
            out = summarize.summarize("a transcript", backend="llamacpp",
                                      url=srv.url, timeout=5)
        self.assertEqual(out, "* done")
        method, path, payload = srv.requests[-1]
        self.assertEqual((method, path), ("POST", "/v1/chat/completions"))
        self.assertIn("a transcript", payload["messages"][0]["content"])
        self.assertIs(payload["stream"], False)

    def test_non_ascii_round_trips_as_utf8(self):
        import json
        seen = []

        def reply(p):
            seen.append(p["messages"][0]["content"])
            return 200, json.dumps(_chat("你好，世界 ✓"), ensure_ascii=False).encode("utf-8")

        with _FakeLlamaServer(reply=reply) as srv:
            out = summarize.summarize("こんにちは 世界 Привет", backend="llamacpp",
                                      url=srv.url, timeout=5)
        self.assertEqual(out, "你好，世界 ✓")
        self.assertIn("こんにちは 世界 Привет", seen[0])

    def test_map_reduce_over_real_http(self):
        import json
        text = "\n".join(f"sentence number {i} here" for i in range(200))
        with _FakeLlamaServer(
                reply=lambda p: (200, json.dumps(_chat("partial")).encode())
        ) as srv:
            summarize.summarize(text, backend="llamacpp", url=srv.url,
                                max_chars=100, timeout=5)
        posts = [r for r in srv.requests if r[0] == "POST"]
        self.assertGreater(len(posts), 2)
        self.assertIn("partial summaries", posts[-1][2]["messages"][0]["content"].lower())

    def test_translate_segments_over_real_http(self):
        import json

        def reply(p):
            n = len([ln for ln in p["messages"][0]["content"].splitlines() if ln[:1].isdigit()])
            return 200, json.dumps(
                _chat("\n".join(f"{i + 1}. t{i}" for i in range(n)))).encode()

        with _FakeLlamaServer(reply=reply) as srv:
            out = summarize.translate_segments(
                ["a", "b", "c"], target_language="English", backend="llamacpp",
                url=srv.url, timeout=5)
        self.assertEqual(out, ["t0", "t1", "t2"])

    def test_http_500_surfaces_as_summarize_error(self):
        with _FakeLlamaServer(reply=lambda p: (500, b'{"error":"oom"}')) as srv:
            with self.assertRaises(SummarizeError) as ctx:
                summarize.summarize("t", backend="llamacpp", url=srv.url, timeout=5)
        self.assertIn("could not reach", str(ctx.exception).lower())

    def test_non_json_body_surfaces_as_summarize_error(self):
        with _FakeLlamaServer(reply=lambda p: (200, b"<html>proxy</html>")) as srv:
            with self.assertRaises(SummarizeError):
                summarize.summarize("t", backend="llamacpp", url=srv.url, timeout=5)

    def test_empty_completion_surfaces_as_summarize_error(self):
        import json
        with _FakeLlamaServer(
                reply=lambda p: (200, json.dumps(_chat("")).encode())) as srv:
            with self.assertRaises(SummarizeError) as ctx:
                summarize.summarize("t", backend="llamacpp", url=srv.url, timeout=5)
        self.assertIn("llama.cpp returned no text", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
