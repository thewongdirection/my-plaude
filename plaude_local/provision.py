"""Interactive prerequisite provisioning.

Before a run, the tool needs the FFmpeg binaries (``ffmpeg`` + ``ffprobe``,
which also carry the codec libraries/DLLs it relies on). If they are missing
this module:

1. warns exactly what is missing and why it's needed, then
2. offers to **download and install** FFmpeg from an official source, or
3. lets the user **provide a path** to an existing ffmpeg/ffprobe, or
4. **aborts** the run if the user declines.

The decision logic is separated from the side effects (prompting, downloading,
installing) via injectable callables, so it is unit-tested offline. The actual
download/install runs only after explicit consent and never in a
non-interactive session unless ``--install-missing`` pre-authorizes it.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, List, Optional

from . import audio

# Official, self-contained static builds (no admin / package manager needed).
_FFMPEG_URLS = {
    "Windows": "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
    "Linux": "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz",
}


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def missing_ffmpeg_tools() -> List[str]:
    """Return which of ffmpeg/ffprobe are not on PATH."""
    missing = []
    if not audio.have_ffmpeg():
        missing.append("ffmpeg")
    if not audio.have_ffprobe():
        missing.append("ffprobe")
    return missing


def _install_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.path.join(
        os.path.expanduser("~"), ".local", "share")
    return Path(base) / "plaude-local" / "ffmpeg"


def register_ffmpeg_path(location: str) -> bool:
    """Point the tool at an existing ffmpeg/ffprobe.

    ``location`` may be the directory holding the binaries or the ffmpeg binary
    itself. On success the directory is prepended to ``PATH`` for this process
    and True is returned (only if BOTH ffmpeg and ffprobe are found there).
    """
    p = Path(os.path.expanduser(location))
    directory = p.parent if p.is_file() else p
    if not directory.is_dir():
        return False
    has_ff = any((directory / n).exists() for n in ("ffmpeg", "ffmpeg.exe"))
    has_probe = any((directory / n).exists() for n in ("ffprobe", "ffprobe.exe"))
    if not (has_ff and has_probe):
        return False
    os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
    return audio.have_ffmpeg() and audio.have_ffprobe()


def _download(url: str, dst: Path) -> None:
    import urllib.request
    with urllib.request.urlopen(url, timeout=120) as resp, open(dst, "wb") as fh:
        shutil.copyfileobj(resp, fh)


def _extract_and_register(archive: Path) -> bool:
    """Extract a downloaded ffmpeg archive and register its bin dir on PATH."""
    target = _install_dir()
    target.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        import zipfile
        with zipfile.ZipFile(archive) as z:
            z.extractall(target)
    else:  # .tar.xz / .tar.*
        import tarfile
        with tarfile.open(archive) as t:
            t.extractall(target)
    # Static builds nest the binaries in a versioned subdir; find them.
    for name in ("ffmpeg.exe", "ffmpeg"):
        for found in target.rglob(name):
            if register_ffmpeg_path(str(found.parent)):
                return True
    return False


def install_ffmpeg() -> bool:
    """Download + install FFmpeg from an official source for this platform.

    Returns True if ffmpeg/ffprobe are available afterward. Best-effort: tries
    the platform package manager first (macOS brew / Windows winget), then falls
    back to a self-contained static download.
    """
    system = platform.system()

    # Prefer a package manager when clearly present.
    if system == "Darwin" and shutil.which("brew"):
        subprocess.run(["brew", "install", "ffmpeg"], check=False)
        return audio.have_ffmpeg() and audio.have_ffprobe()
    if system == "Windows" and shutil.which("winget"):
        subprocess.run(
            ["winget", "install", "--id", "Gyan.FFmpeg", "-e", "--source", "winget",
             "--accept-package-agreements", "--accept-source-agreements"],
            check=False)
        if audio.have_ffmpeg() and audio.have_ffprobe():
            return True
        # winget installs to a versioned dir not on this process's PATH; fall
        # through to the self-contained download so the current run can proceed.

    url = _FFMPEG_URLS.get(system)
    if not url:
        _log(f"no automatic FFmpeg installer for {system}; please install it "
             "manually or pass --ffmpeg-location.")
        return False

    _log(f"downloading FFmpeg from {url} ...")
    try:
        with tempfile.TemporaryDirectory(prefix="plaude-ffmpeg-") as tmp:
            archive = Path(tmp) / os.path.basename(url)
            _download(url, archive)
            if _extract_and_register(archive):
                _log(f"installed FFmpeg to {_install_dir()}")
                return True
    except Exception as exc:  # network/extract failure - stay graceful
        _log(f"automatic FFmpeg install failed: {exc}")
    return False


def _console_choice(prompt: str, options: List[str]) -> str:
    try:
        return input(prompt).strip().lower()
    except (EOFError, OSError):
        return ""


def _console_path(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, OSError):
        return ""


def ensure_ffmpeg(
    *,
    ffmpeg_location: Optional[str] = None,
    install_missing: bool = False,
    no_provision: bool = False,
    quiet: bool = False,
    interactive: Optional[bool] = None,
    prompt_choice: Optional[Callable[[str, List[str]], str]] = None,
    prompt_path: Optional[Callable[[str], str]] = None,
    installer: Optional[Callable[[], bool]] = None,
) -> bool:
    """Ensure ffmpeg+ffprobe are available, provisioning them if needed.

    Returns True to proceed, False to abort. The prompting / install seams are
    injectable for testing. Only ``ffmpeg`` is a hard requirement (installing it
    also provides ``ffprobe``); a lone missing ``ffprobe`` is a soft note.
    """
    if audio.have_ffmpeg():
        if not audio.have_ffprobe() and not quiet:
            _log("note: ffprobe not found (it ships with FFmpeg); stream checks "
                 "and some quality metrics will be limited.")
        return True

    prompt_choice = prompt_choice or _console_choice
    prompt_path = prompt_path or _console_path
    installer = installer or install_ffmpeg
    if interactive is None:
        interactive = sys.stdin.isatty()

    _log("warning: FFmpeg (ffmpeg + ffprobe) was not found on PATH. It is "
         "required to decode audio.")

    # 1. Explicit path wins (also works non-interactively).
    if ffmpeg_location:
        if register_ffmpeg_path(ffmpeg_location):
            _log(f"using FFmpeg from {ffmpeg_location}")
            return True
        _log(f"error: no usable ffmpeg/ffprobe found at {ffmpeg_location!r}.")

    # 2. Pre-authorized auto-install (scriptable, non-interactive-safe).
    if install_missing:
        _log("attempting to download and install FFmpeg ...")
        if installer():
            return True
        _log("error: automatic FFmpeg installation did not succeed.")
        return False

    # 3. If provisioning is disabled or we can't prompt, abort with guidance.
    if no_provision or not interactive:
        _log("aborting: FFmpeg is unavailable. Install it (winget/apt/brew or "
             "https://ffmpeg.org/download.html), pass --ffmpeg-location PATH, or "
             "re-run with --install-missing to download it automatically.")
        return False

    # 4. Interactive: offer install / provide path / abort.
    while True:
        choice = prompt_choice(
            "Choose: [i]nstall automatically, provide a [p]ath, or [a]bort? ",
            ["i", "p", "a"])
        if choice in ("i", "install"):
            if installer():
                return True
            _log("automatic installation failed; try providing a path instead.")
        elif choice in ("p", "path"):
            loc = prompt_path("Enter the folder containing ffmpeg/ffprobe "
                              "(or the ffmpeg binary): ")
            if loc and register_ffmpeg_path(loc):
                _log(f"using FFmpeg from {loc}")
                return True
            _log("that path did not contain a usable ffmpeg + ffprobe.")
        elif choice in ("a", "abort", "", "n", "no"):
            _log("aborting at user request: required FFmpeg not provided.")
            return False
