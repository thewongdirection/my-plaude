"""Offline tests for the offline-bundle helper (tools/prepare_offline_bundle.py).

Network fetchers are mocked; only pure logic (manifest), zipping, the install
doc, and orchestration wiring are exercised — no downloads.
"""
import importlib.util
import io
import json
import contextlib
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

_MOD_PATH = Path(__file__).resolve().parent.parent / "tools" / "prepare_offline_bundle.py"
_spec = importlib.util.spec_from_file_location("prepare_offline_bundle", _MOD_PATH)
bundle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bundle)


class TestManifest(unittest.TestCase):
    def test_default_manifest_shape(self):
        m = bundle.build_manifest(["small", "large-v3"], "windows", [], False)
        kinds = [i["kind"] for i in m]
        self.assertEqual(kinds, ["archive", "pip", "hf", "hf"])
        # Windows FFmpeg is a real downloadable archive.
        self.assertTrue(m[0]["source"].endswith(".zip"))
        # Core wheels always present.
        self.assertIn("faster-whisper", m[1]["source"])
        self.assertIn("whisper-ctranslate2", m[1]["source"])

    def test_cuda_and_diarize_expand_wheels_and_add_gated_models(self):
        m = bundle.build_manifest(["large-v3"], "windows", ["cuda", "diarize"], True)
        pip = next(i for i in m if i["kind"] == "pip")
        self.assertIn("nvidia-cublas-cu12", pip["source"])
        self.assertIn("nvidia-cudnn-cu12", pip["source"])
        self.assertIn("pyannote.audio", pip["source"])
        gated = [i for i in m if i["kind"] == "hf-gated"]
        self.assertEqual(len(gated), 2)
        self.assertTrue(all("pyannote/" in i["source"] for i in gated))

    def test_macos_ffmpeg_is_manual(self):
        m = bundle.build_manifest(["tiny"], "macos", [], False)
        self.assertEqual(m[0]["kind"], "manual")
        self.assertIn("brew", m[0]["source"])


class TestZipAndDocs(unittest.TestCase):
    def test_make_zip_contains_staged_files(self):
        with tempfile.TemporaryDirectory() as d:
            staging = Path(d) / "plaude-local-offline"
            (staging / "ffmpeg").mkdir(parents=True)
            (staging / "ffmpeg" / "ffmpeg.exe").write_bytes(b"binary")
            (staging / "INSTALL.md").write_text("hi", encoding="utf-8")
            zip_path = bundle.make_zip(staging, Path(d) / "out.zip")
            self.assertTrue(zip_path.is_file())
            with zipfile.ZipFile(zip_path) as zf:
                names = zf.namelist()
            self.assertTrue(any(n.endswith("ffmpeg.exe") for n in names))
            self.assertTrue(any(n.endswith("INSTALL.md") for n in names))

    def test_install_md_lists_items_and_steps(self):
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d)
            manifest = bundle.build_manifest(["small"], "windows", [], False)
            bundle.write_install_md(dest, manifest, "windows")
            text = (dest / "INSTALL.md").read_text(encoding="utf-8")
            self.assertIn("FFmpeg", text)
            self.assertIn("--no-index", text)
            self.assertIn("--offline", text)


class TestRunOrchestration(unittest.TestCase):
    def test_list_mode_prints_json_and_downloads_nothing(self):
        buf = io.StringIO()
        with mock.patch.object(bundle, "fetch_archive") as fa, \
             mock.patch.object(bundle, "fetch_pip") as fp, \
             contextlib.redirect_stdout(buf):
            rc = bundle.run(["--list"])
        self.assertEqual(rc, 0)
        fa.assert_not_called()
        fp.assert_not_called()
        parsed = json.loads(buf.getvalue())
        self.assertTrue(any(i["kind"] == "archive" for i in parsed))

    def test_diarize_group_requires_token(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
             contextlib.redirect_stderr(io.StringIO()):
            rc = bundle.run(["--include", "diarize", "--no-zip", "--dest",
                             tempfile.mkdtemp()])
        self.assertEqual(rc, 2)

    def test_full_run_invokes_each_fetcher_and_zips(self):
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / "stage"
            zpath = Path(d) / "bundle.zip"
            with mock.patch.object(bundle, "fetch_archive") as fa, \
                 mock.patch.object(bundle, "fetch_pip") as fp, \
                 mock.patch.object(bundle, "fetch_hf") as fh, \
                 mock.patch.object(bundle, "fetch_pip_self") as fs, \
                 contextlib.redirect_stdout(io.StringIO()):
                rc = bundle.run(["--models", "small", "--dest", str(dest),
                                 "--zip", str(zpath)])
            self.assertEqual(rc, 0)
            fa.assert_called_once()          # ffmpeg archive
            fp.assert_called_once()          # wheels
            fh.assert_called_once()          # one whisper model
            fs.assert_called_once()          # plaude-local self wheel
            self.assertTrue(zpath.is_file())
            self.assertTrue((dest / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
