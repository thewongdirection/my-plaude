#!/usr/bin/env python3
"""Prepare an offline prerequisites bundle for plaude-local.

Enumerates everything plaude-local needs (FFmpeg, Python wheels, CUDA runtime
DLLs, Whisper models, optional diarization models), downloads it in advance, and
zips it so a recipient can prep an offline / air-gapped machine by unzipping and
following the generated INSTALL.md — no internet needed at install time.

Examples
--------
    # See exactly what would be fetched (no download):
    python tools/prepare_offline_bundle.py --list

    # Build a Windows CPU bundle with the small + large-v3 models:
    python tools/prepare_offline_bundle.py --models small large-v3

    # Add GPU DLLs and diarization models (gated -> needs an HF token):
    python tools/prepare_offline_bundle.py --include cuda diarize \
        --hf-token hf_xxx --models large-v3

The PowerShell equivalent is tools/Prepare-OfflineBundle.ps1 (kept at parity).
"""
from __future__ import annotations

import argparse
import json
import os
import platform as _platform
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

# "latest"-stable download URLs so the script does not rot on a pinned version.
FFMPEG_URLS = {
    "windows": "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
    "linux": "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz",
    "macos": None,  # use Homebrew: `brew install ffmpeg` (documented in INSTALL.md)
}

# faster-whisper hosts its CTranslate2 conversions under the Systran org.
WHISPER_REPO = "Systran/faster-whisper-{model}"
DIARIZE_REPOS = ("pyannote/speaker-diarization-3.1", "pyannote/segmentation-3.0")

INCLUDE_WHEELS = {
    "cuda": ["nvidia-cublas-cu12", "nvidia-cudnn-cu12"],
    "diarize": ["pyannote.audio>=3.1"],
    "deepfilter": ["deepfilternet"],
    "resemble": ["resemble-enhance"],
}


def detect_platform() -> str:
    sysname = _platform.system().lower()
    if sysname.startswith("win"):
        return "windows"
    if sysname == "darwin":
        return "macos"
    return "linux"


def build_manifest(
    models: List[str],
    platform_name: str,
    include: List[str],
    with_diarize_models: bool,
) -> List[Dict[str, str]]:
    """Enumerate every prerequisite as a list of plain dicts (pure, no I/O).

    Each item: {name, kind, source, target, note}. This is the single source of
    truth shared by --list and the fetch step (and mirrored by the PowerShell
    helper's Get-BundleManifest).
    """
    items: List[Dict[str, str]] = []

    ff = FFMPEG_URLS.get(platform_name)
    items.append({
        "name": "FFmpeg (+ ffprobe)",
        "kind": "archive" if ff else "manual",
        "source": ff or "brew install ffmpeg",
        "target": "ffmpeg/",
        "note": "required; decodes any input and denoises",
    })

    wheels = ["faster-whisper", "whisper-ctranslate2"]
    for group in include:
        wheels += INCLUDE_WHEELS.get(group, [])
    items.append({
        "name": "Python wheels",
        "kind": "pip",
        "source": " ".join(wheels),
        "target": "wheels/",
        "note": "pip download into a local wheelhouse for --no-index installs",
    })

    for model in models:
        items.append({
            "name": f"Whisper model: {model}",
            "kind": "hf",
            "source": WHISPER_REPO.format(model=model),
            "target": "hf/hub/",
            "note": "transcription weights (faster-whisper / CTranslate2)",
        })

    if with_diarize_models:
        for repo in DIARIZE_REPOS:
            items.append({
                "name": f"Diarization model: {repo}",
                "kind": "hf-gated",
                "source": repo,
                "target": "hf/hub/",
                "note": "GATED: needs an HF token + accepted model terms",
            })

    return items


# --------------------------------------------------------------------------- #
# Fetchers (each consumes one manifest item)
# --------------------------------------------------------------------------- #

def _log(msg: str) -> None:
    print(msg, flush=True)


def fetch_archive(url: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    fname = url.split("/")[-1]
    out = dest / fname
    _log(f"  downloading {url}")
    # Some mirrors (gyan.dev, johnvansickle) reject requests without a UA.
    req = urllib.request.Request(url, headers={"User-Agent": "plaude-local-offline-bundle"})
    with urllib.request.urlopen(req) as resp, open(out, "wb") as fh:  # noqa: S310
        shutil.copyfileobj(resp, fh)
    _log(f"  saved {out.name} ({out.stat().st_size // (1024 * 1024)} MB)")


def fetch_pip(pkgs: List[str], dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "pip", "download", "-d", str(dest), *pkgs]
    _log("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def fetch_pip_self(project_root: Path, dest: Path) -> None:
    """Also build a wheel for plaude-local itself, so the bundle is complete."""
    if not (project_root / "pyproject.toml").is_file():
        _log(f"  (skip self wheel: no pyproject.toml at {project_root})")
        return
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "pip", "wheel", str(project_root), "-w", str(dest)]
    _log("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def fetch_hf(repo: str, cache_dir: Path, token: Optional[str]) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "huggingface_hub is required to fetch models. Install it with "
            "`pip install huggingface_hub` and re-run."
        ) from exc
    cache_dir.mkdir(parents=True, exist_ok=True)
    _log(f"  snapshot {repo} -> {cache_dir}")
    snapshot_download(repo_id=repo, cache_dir=str(cache_dir), token=token)


# --------------------------------------------------------------------------- #

def write_install_md(dest: Path, manifest: List[Dict[str, str]], platform_name: str) -> None:
    lines = [
        "# plaude-local — offline prerequisites bundle",
        "",
        f"Platform target: **{platform_name}**. Everything below is included so you",
        "can install without internet access.",
        "",
        "## Contents",
        "",
    ]
    for item in manifest:
        lines.append(f"- **{item['name']}** → `{item['target']}` ({item['note']})")
    lines += [
        "",
        "## Install steps",
        "",
        "1. **Unzip** this bundle somewhere permanent, e.g. `C:\\plaude-local-offline`.",
        "2. **FFmpeg** — unzip `ffmpeg/` and add its `bin/` folder to your `PATH`",
        "   (Windows), or `sudo cp ffmpeg ffprobe /usr/local/bin` (Linux static build).",
        "3. **Python packages** — install from the local wheelhouse, no internet:",
        "   ```",
        "   pip install --no-index --find-links wheels plaude-local",
        "   ```",
        "   (add `pyannote.audio` to that line if you included diarization).",
        "4. **Models** — point Hugging Face at the bundled cache and run offline:",
        "   ```",
        "   # Windows PowerShell",
        "   $env:HF_HOME = \"<bundle>\\hf\"",
        "   # bash",
        "   export HF_HOME=<bundle>/hf",
        "   ```",
        "   Then always pass `--offline` so nothing reaches the network.",
        "5. **Verify**: `plaude-local --check` (or `.\\Plaude-Local.ps1 -Check`).",
        "",
        "First real run:",
        "```",
        "plaude-local myaudio.m4a --model small --offline",
        "```",
        "",
        "### Windows note (diarization / PyTorch)",
        "",
        "Installing `pyannote.audio` pulls in PyTorch, whose deeply-nested files can",
        "exceed the legacy 260-char path limit and fail with `WinError 206`. Either",
        "enable long paths once (admin PowerShell):",
        "```",
        "New-ItemProperty -Path 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem' \\",
        "  -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force",
        "```",
        "or install into a virtual environment created at a short path (e.g. `C:\\v`).",
    ]
    (dest / "INSTALL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_zip(staging: Path, zip_path: Path) -> Path:
    if zip_path.exists():
        zip_path.unlink()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in staging.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(staging.parent))
    return zip_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download plaude-local prerequisites and zip them for offline use."
    )
    p.add_argument("--models", nargs="*", default=["small", "large-v3"],
                   help="Whisper models to bundle (default: small large-v3)")
    p.add_argument("--platform", choices=["windows", "linux", "macos"],
                   default=detect_platform(), help="target platform for FFmpeg")
    p.add_argument("--include", nargs="*", default=[],
                   choices=sorted(INCLUDE_WHEELS), metavar="GROUP",
                   help="extra wheel groups: cuda, diarize, deepfilter, resemble")
    p.add_argument("--hf-token", default=None,
                   help="HF token for gated diarization models (or set HF_TOKEN)")
    p.add_argument("--project-root", default=str(Path(__file__).resolve().parent.parent),
                   help="plaude-local repo root (to also bundle its own wheel)")
    p.add_argument("--dest", default="plaude-local-offline",
                   help="staging directory (default: ./plaude-local-offline)")
    p.add_argument("--zip", default="plaude-local-offline-bundle.zip",
                   help="output zip path (default: ./plaude-local-offline-bundle.zip)")
    p.add_argument("--no-zip", action="store_true", help="stage files but skip zipping")
    p.add_argument("--list", action="store_true",
                   help="print the prerequisite manifest as JSON and exit (no download)")
    return p


def run(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    want_diarize_models = "diarize" in args.include
    manifest = build_manifest(args.models, args.platform, args.include, want_diarize_models)

    if args.list:
        print(json.dumps(manifest, indent=2))
        return 0

    token = args.hf_token or os.environ.get("HF_TOKEN")
    if want_diarize_models and not token:
        print("error: --include diarize needs an HF token (--hf-token or HF_TOKEN) "
              "and accepted terms at https://hf.co/pyannote/speaker-diarization-3.1",
              file=sys.stderr)
        return 2

    staging = Path(args.dest).resolve()
    staging.mkdir(parents=True, exist_ok=True)
    _log(f"Staging into: {staging}")

    for item in manifest:
        _log(f"[{item['kind']}] {item['name']}")
        target = staging / item["target"].rstrip("/")
        try:
            if item["kind"] == "archive":
                fetch_archive(item["source"], target)
            elif item["kind"] == "manual":
                _log(f"  MANUAL: {item['source']} (cannot be auto-downloaded here)")
            elif item["kind"] == "pip":
                fetch_pip(item["source"].split(), target)
            elif item["kind"] in ("hf", "hf-gated"):
                fetch_hf(item["source"], target, token)
        except (subprocess.CalledProcessError, OSError, Exception) as exc:  # noqa: BLE001
            print(f"  WARNING: failed to fetch {item['name']}: {exc}", file=sys.stderr)

    fetch_pip_self(Path(args.project_root), staging / "wheels")
    write_install_md(staging, manifest, args.platform)
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.no_zip:
        _log(f"Done. Staged (not zipped) at {staging}")
        return 0

    zip_path = make_zip(staging, Path(args.zip).resolve())
    _log(f"Done. Bundle: {zip_path} ({zip_path.stat().st_size // (1024 * 1024)} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(run())
