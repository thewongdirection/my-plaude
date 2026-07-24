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


if __name__ == "__main__":
    unittest.main()
