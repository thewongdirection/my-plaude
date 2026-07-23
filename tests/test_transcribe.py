"""Regression tests for device/compute resolution and backend guards."""

import sys
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


class TestResolveComputeType(unittest.TestCase):
    def test_auto_gpu_is_float16(self):
        self.assertEqual(transcribe.resolve_compute_type("auto", "cuda"), "float16")

    def test_auto_cpu_is_int8(self):
        self.assertEqual(transcribe.resolve_compute_type("auto", "cpu"), "int8")

    def test_explicit_passthrough(self):
        self.assertEqual(
            transcribe.resolve_compute_type("int8_float16", "cuda"), "int8_float16"
        )


class TestTranscriberGuard(unittest.TestCase):
    def test_missing_faster_whisper_raises_clean_error(self):
        # Force the lazy import to fail, deterministically, whether or not the
        # package is installed in the current environment.
        with mock.patch.dict(sys.modules, {"faster_whisper": None}):
            with self.assertRaises(transcribe.TranscribeError):
                transcribe.Transcriber(model="tiny", device="cpu")


if __name__ == "__main__":
    unittest.main()
