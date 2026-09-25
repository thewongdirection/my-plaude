"""Regression tests for device/compute resolution and backend guards."""

import os
import sys
import tempfile
import types
import unittest
from unittest import mock

from plaude_local import transcribe


class TestResolveDevice(unittest.TestCase):
    def test_explicit_passthrough(self):
        self.assertEqual(transcribe.resolve_device("cpu"), "cpu")
        self.assertEqual(transcribe.resolve_device("cuda"), "cuda")

    def test_auto_falls_back_to_cpu_without_ctranslate2(self):
        with mock.patch.dict(sys.modules, {"ctranslate2": None}):
            self.assertEqual(transcribe.resolve_device("auto"), "cpu")

    def test_auto_picks_cuda_when_gpu_present(self):
        fake = types.ModuleType("ctranslate2")
        fake.get_cuda_device_count = lambda: 1
        with mock.patch.dict(sys.modules, {"ctranslate2": fake}):
            self.assertEqual(transcribe.resolve_device("auto"), "cuda")

    def test_auto_cpu_when_zero_gpus(self):
        fake = types.ModuleType("ctranslate2")
        fake.get_cuda_device_count = lambda: 0
        with mock.patch.dict(sys.modules, {"ctranslate2": fake}):
            self.assertEqual(transcribe.resolve_device("auto"), "cpu")

    def test_cpu_only_mode_stays_cpu_without_probing(self):
        # Explicit --device cpu is a hard CPU-only mode: it must never probe for
        # a GPU nor upgrade to one even when a CUDA device is present.
        fake = types.ModuleType("ctranslate2")

        def _no_probe():
            raise AssertionError("explicit device must not probe for CUDA")

        fake.get_cuda_device_count = _no_probe
        with mock.patch.dict(sys.modules, {"ctranslate2": fake}):
            self.assertEqual(transcribe.resolve_device("cpu"), "cpu")

    def test_gpu_only_mode_stays_cuda_without_fallback(self):
        # Explicit --device cuda is a hard GPU-only mode: honored as-is, with no
        # probing and no silent fallback to CPU (a missing GPU should fail loudly
        # at load time, not be masked here).
        fake = types.ModuleType("ctranslate2")

        def _no_probe():
            raise AssertionError("explicit device must not probe for CUDA")

        fake.get_cuda_device_count = _no_probe
        with mock.patch.dict(sys.modules, {"ctranslate2": fake}):
            self.assertEqual(transcribe.resolve_device("cuda"), "cuda")


class TestResolveComputeType(unittest.TestCase):
    def test_auto_gpu_is_float16(self):
        self.assertEqual(transcribe.resolve_compute_type("auto", "cuda"), "float16")

    def test_auto_cpu_is_int8(self):
        self.assertEqual(transcribe.resolve_compute_type("auto", "cpu"), "int8")

    def test_explicit_passthrough(self):
        self.assertEqual(
            transcribe.resolve_compute_type("int8_float16", "cuda"), "int8_float16"
        )


class TestCudaDllDirs(unittest.TestCase):
    """Windows CUDA-wheel DLL discovery for out-of-the-box GPU inference."""

    def test_nvidia_dll_dirs_finds_bin_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "nvidia")
            for pkg in ("cublas", "cudnn"):
                os.makedirs(os.path.join(root, pkg, "bin"))
            os.makedirs(os.path.join(root, "metadata_only"))  # no bin -> ignored
            fake_spec = types.SimpleNamespace(submodule_search_locations=[root])
            with mock.patch("importlib.util.find_spec", return_value=fake_spec):
                dirs = transcribe.nvidia_dll_dirs()
            self.assertEqual(
                sorted(os.path.basename(os.path.dirname(d)) for d in dirs),
                ["cublas", "cudnn"],
            )
            self.assertTrue(all(d.endswith("bin") for d in dirs))

    def test_nvidia_dll_dirs_empty_without_nvidia(self):
        with mock.patch("importlib.util.find_spec", return_value=None):
            self.assertEqual(transcribe.nvidia_dll_dirs(), [])

    def test_add_cuda_dll_directories_noop_off_windows(self):
        calls = []
        with mock.patch.object(transcribe.sys, "platform", "linux"), mock.patch.object(
            transcribe.os, "add_dll_directory", calls.append, create=True
        ), mock.patch.object(transcribe, "nvidia_dll_dirs", return_value=["/x/bin"]):
            transcribe.add_cuda_dll_directories()
        self.assertEqual(calls, [])

    def test_add_cuda_dll_directories_registers_each_on_windows(self):
        calls = []
        with mock.patch.object(transcribe.sys, "platform", "win32"), mock.patch.object(
            transcribe.os, "add_dll_directory", calls.append, create=True
        ), mock.patch.object(
            transcribe, "nvidia_dll_dirs", return_value=["/a/bin", "/b/bin"]
        ):
            transcribe.add_cuda_dll_directories()
        self.assertEqual(calls, ["/a/bin", "/b/bin"])


class TestTranscriberGuard(unittest.TestCase):
    def test_missing_faster_whisper_raises_clean_error(self):
        # Force the lazy import to fail, deterministically, whether or not the
        # package is installed in the current environment.
        with mock.patch.dict(sys.modules, {"faster_whisper": None}):
            with self.assertRaises(transcribe.TranscribeError):
                transcribe.Transcriber(model="tiny", device="cpu")




class TestInferenceErrors(unittest.TestCase):
    """Runtime inference failures become TranscribeError (exit 6), not tracebacks."""

    def _engine(self, exc):
        engine = transcribe.Transcriber.__new__(transcribe.Transcriber)
        engine.device, engine.compute_type, engine.model_name = "cuda", "float16", "small"
        engine._model = mock.Mock()
        engine._model.transcribe.side_effect = exc
        return engine

    def test_missing_cublas_gets_actionable_remedy(self):
        engine = self._engine(RuntimeError("Library cublas64_12.dll is not found or cannot be loaded"))
        with self.assertRaises(transcribe.TranscribeError) as ctx:
            engine.transcribe("x.wav")
        msg = str(ctx.exception)
        self.assertIn("cublas64_12.dll", msg)
        self.assertIn("nvidia-cublas-cu12", msg)
        self.assertIn("--device cpu", msg)

    def test_other_errors_are_wrapped_without_cuda_hint(self):
        engine = self._engine(ValueError("bad audio"))
        with self.assertRaises(transcribe.TranscribeError) as ctx:
            engine.transcribe("x.wav")
        self.assertIn("bad audio", str(ctx.exception))
        self.assertNotIn("nvidia-cublas-cu12", str(ctx.exception))

    def test_errors_while_iterating_segments_are_wrapped(self):
        # faster-whisper is lazy: inference (and DLL loading) runs on iteration.
        def gen():
            raise RuntimeError("Could not load library cudnn_ops64_9.dll")
            yield  # pragma: no cover

        engine = self._engine(None)
        engine._model.transcribe.side_effect = None
        engine._model.transcribe.return_value = (gen(), mock.Mock(language="en"))
        with self.assertRaises(transcribe.TranscribeError) as ctx:
            engine.transcribe("x.wav")
        self.assertIn("nvidia-cudnn-cu12", str(ctx.exception))

    def test_explain_inference_error(self):
        explain = lambda m: transcribe.explain_inference_error(RuntimeError(m))  # noqa: E731
        # Missing libraries -> install hint.
        self.assertIn(transcribe.CUDA_LIBS_REMEDY,
                      explain("Library cublas64_12.dll is not found or cannot be loaded"))
        # Loaded but failed (GPU reset / OOM / driver) -> GPU hint, NOT the install hint.
        for m in ("CUBLAS_STATUS_NOT_INITIALIZED", "CUDA failed with error out of memory",
                  "CUDNN_STATUS_EXECUTION_FAILED", "CUDA driver version is insufficient"):
            self.assertIn(transcribe.GPU_FAILURE_REMEDY, explain(m), m)
            self.assertNotIn(transcribe.CUDA_LIBS_REMEDY, explain(m), m)
        self.assertEqual(explain("boom"), "boom")

    def test_model_load_failure_gets_remedy(self):
        fake = types.ModuleType("faster_whisper")
        fake.WhisperModel = mock.Mock(side_effect=RuntimeError(
            "Library cublas64_12.dll is not found or cannot be loaded"))
        with mock.patch.dict(sys.modules, {"faster_whisper": fake}), \
             mock.patch.object(transcribe, "add_cuda_dll_directories"):
            with self.assertRaises(transcribe.TranscribeError) as ctx:
                transcribe.Transcriber("small", device="cuda", compute_type="float16")
        self.assertIn("nvidia-cublas-cu12", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
