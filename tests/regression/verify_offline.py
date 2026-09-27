"""Verify that BOTH tools work with no internet connection.

The regression suite runs on a cloud GPU by default purely for speed; the tool
itself is offline-first. This check proves it: it runs the Python and the
PowerShell tool on a short corpus recording, on the CPU, with ``--offline`` /
``-Offline`` (the offline environment variables are NOT pre-set, so the flags
themselves are tested) and every HTTP(S) request routed to a dead proxy, so any
attempt to reach the internet fails loudly; only 127.0.0.1 (the local Ollama
server) stays reachable. The proxy is applied to both HTTP stacks involved:
the HTTP(S)_PROXY variables for Python, whisper-ctranslate2 and PowerShell 7,
and .NET's ``DefaultWebProxy`` for Windows PowerShell 5.1 (which ignores those
variables). Control checks first confirm the internet really is unreachable
through each stack. Each tool must exit 0, print no warnings and produce a
transcript and an English translation.

(Raw socket connections would not be caught - neither tool makes any; for an
absolute guarantee, disconnect the network or block it in the firewall and run
the tools the same way.)

Prerequisites (one-time, while online): the Whisper model (``--model``,
default ``large-v3``) in the Hugging Face cache, and a local Ollama translation
model (``--translate-model``, default ``translategemma:4b``).

    python tests/regression/verify_offline.py
    python tests/regression/verify_offline.py --item ja-mono-01 --model small
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from run_regression import child_env, powershell_exe  # noqa: E402

DEAD_PROXY = "http://127.0.0.1:9"  # discard port: every proxied request is refused
LOCAL_ONLY = "127.0.0.1,localhost"
RUN_TIMEOUT = 1800  # s per tool

# Windows PowerShell 5.1's web cmdlets use .NET's DefaultWebProxy, not the
# HTTP(S)_PROXY variables: set it before the script runs (local addresses bypass).
PS_PROXY_PREAMBLE = (
    "[System.Net.WebRequest]::DefaultWebProxy = New-Object System.Net.WebProxy("
    "'" + DEAD_PROXY + "', $true, [string[]]@('127\\.0\\.0\\.1', 'localhost'))\n")
# Relay @args untouched (keeps -Switch names bound) to the script named in the env.
PS_TARGET_VAR = "PLAUDE_OFFLINE_TARGET"
PS_WRAPPER = PS_PROXY_PREAMBLE + f"& $env:{PS_TARGET_VAR} @args\nexit $LASTEXITCODE\n"
PS_CONTROL = PS_PROXY_PREAMBLE + (
    "try { Invoke-WebRequest 'https://huggingface.co' -UseBasicParsing -TimeoutSec 5 | Out-Null;"
    " exit 1 } catch { }\n"
    "try { Invoke-RestMethod 'http://127.0.0.1:11434/api/tags' -TimeoutSec 5 | Out-Null;"
    " exit 0 } catch { exit 2 }\n")


def offline_env(base: dict) -> dict:
    """``base`` with the internet made unreachable (local services still work)."""
    env = dict(base)
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env[var] = DEAD_PROXY
    env["NO_PROXY"] = env["no_proxy"] = LOCAL_ONLY
    # Deliberately NOT setting HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE: the tools'
    # own --offline / -Offline must do that.
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        env.pop(var, None)
    return env


def internet_blocked(env: dict) -> bool:
    """Control check (Python stack): under ``env`` a request to the internet must fail."""
    code = ("import urllib.request\n"
            "try:\n urllib.request.urlopen('https://huggingface.co', timeout=5)\n"
            "except Exception:\n raise SystemExit(0)\nraise SystemExit(1)\n")
    return subprocess.run([sys.executable, "-c", code], env=env,
                          stdin=subprocess.DEVNULL, timeout=60).returncode == 0


def powershell_blocked(env: dict, folder: Path) -> bool:
    """Control check (PowerShell/.NET stack): internet fails, 127.0.0.1 works."""
    script = folder / "control.ps1"
    script.write_text(PS_CONTROL, encoding="utf-8")
    proc = subprocess.run([powershell_exe(), "-NoProfile", "-ExecutionPolicy", "Bypass",
                           "-File", str(script)], env=env, stdin=subprocess.DEVNULL,
                          capture_output=True, timeout=120)
    return proc.returncode == 0


def commands(audio: Path, out: Path, model: str, translate_model: str,
             ps_wrapper: Path = Path("wrapper.ps1")) -> dict:
    py = [sys.executable, "-m", "plaude_local", str(audio), "--offline", "--no-provision",
          "--device", "cpu",
          "-m", model, "-f", "html", "-o", str(out / "py.html"),
          "--transcription-file", str(out / "py.t.txt"),
          "--translation-file", str(out / "py.x.txt"),
          "--translate-engine", "llm", "--translate-model", translate_model,
          "--summarize-timeout", "900", "-y", "-q"]
    ps = [powershell_exe(), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
          str(ps_wrapper), str(audio),  # wrapper runs $env:PLAUDE_OFFLINE_TARGET
          "-Offline", "-NoProvision", "-Device", "cpu", "-Model", model, "-Format", "html",
          "-Output", str(out / "ps.html"), "-TranscriptionFile", str(out / "ps.t.txt"),
          "-TranslationFile", str(out / "ps.x.txt"), "-TranslateEngine", "llm",
          "-TranslateModel", translate_model, "-SummarizeTimeout", "900", "-Yes", "-Quiet"]
    return {"python": py, "powershell": ps}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--item", default="zh-mono-03", help="corpus item id (default zh-mono-03)")
    p.add_argument("--model", default="large-v3", help="Whisper model (must be cached)")
    p.add_argument("--translate-model", default="translategemma:4b",
                   help="local Ollama model for the translation")
    p.add_argument("--impl", choices=["both", "python", "powershell"], default="both")
    args = p.parse_args(argv)

    manifest = json.loads((HERE / "corpus" / "manifest.json").read_text(encoding="utf-8"))
    item = next((e for e in manifest["items"] if e["id"] == args.item), None)
    if item is None:
        p.error(f"no corpus item {args.item!r}")
    impls = ["python", "powershell"] if args.impl == "both" else [args.impl]
    env = offline_env(child_env())
    env[PS_TARGET_VAR] = str(REPO / "powershell" / "Plaude-Local.ps1")
    if not internet_blocked(env):
        print("FAIL: could not block the internet for Python (proxy variables ignored?)")
        return 1
    try:
        urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=5)
    except Exception:
        print("FAIL: no local Ollama server at 127.0.0.1:11434 (needed for translation)")
        return 1

    failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        wrapper = out / "offline-wrapper.ps1"
        wrapper.write_text(PS_WRAPPER, encoding="utf-8")
        if "powershell" in impls and not powershell_blocked(env, out):
            print("FAIL: could not block the internet for PowerShell (.NET proxy not applied)")
            return 1
        print("internet: blocked for the tools (only 127.0.0.1 reachable) - "
              "Python and PowerShell stacks verified")
        cmds = commands(HERE / "corpus" / item["file"], out, args.model, args.translate_model,
                        wrapper)
        for impl in impls:
            try:
                proc = subprocess.run(cmds[impl], cwd=REPO, env=env, capture_output=True,
                                      text=True, encoding="utf-8", errors="replace",
                                      stdin=subprocess.DEVNULL, timeout=RUN_TIMEOUT)
            except subprocess.TimeoutExpired:
                failed += 1
                print(f"FAIL {impl}: no result after {RUN_TIMEOUT}s")
                continue
            tag = {"python": "py", "powershell": "ps"}[impl]
            t = (out / f"{tag}.t.txt").read_text(encoding="utf-8") if (out / f"{tag}.t.txt").exists() else ""
            x = (out / f"{tag}.x.txt").read_text(encoding="utf-8") if (out / f"{tag}.x.txt").exists() else ""
            ok = proc.returncode == 0 and t.strip() and x.strip() and "warning:" not in proc.stderr
            failed += not ok
            print(f"{'PASS' if ok else 'FAIL'} {impl}: exit {proc.returncode}, "
                  f"{len(t.split())} transcript words, {len(x.split())} translated words")
            if not ok:
                print((proc.stdout + proc.stderr)[-2000:])
            elif x.strip():
                print("   " + x.strip().splitlines()[0][:100])
    print("offline check:", "PASSED" if not failed else "FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
