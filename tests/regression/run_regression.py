"""Multilingual end-to-end regression suite for BOTH implementations.

Runs corpus files (300 clean + 300 damaged across 21 languages - mostly Asian - monologues
and multi-speaker conversations - see generate_corpus.py) through the real tool - Python (``python -m plaude_local``) and/or PowerShell
(``powershell/Plaude-Local.ps1``) - transcribing and translating to English
via the HTML dashboard's split outputs, then scores each run:

* exit code 0,
* detected language is the recording's language (or a closely related one
  Whisper is known to confuse; tiers A/B only),
* transcript character error rate (CER) against the source text,
* English translation content-word recall against the exact English reference,
* damaged files only: they run with ``--repair --enhance strong --keep-stages``
  and ``--dehum auto`` must detect exactly the hum that was injected (or none).

Absolute thresholds depend on the language's tier (``languages.THRESHOLDS``:
A well supported, B usable, C low-resource = exit code + baseline only). A run
also fails if it regresses beyond the tolerance against ``baseline.json``
(recorded with ``--update-baseline`` from a known-good run with the same
Whisper model). Needs FFmpeg, faster-whisper / whisper-ctranslate2 and an
Ollama server; a CUDA GPU is strongly recommended. The full suite is long
(~46 h of audio per implementation), so ``--sample N`` runs a seeded,
stratified subset of N items spread over languages and monologue/conversation;
each item still runs clean + damaged in every selected implementation
(``--sample 24`` = 96 runs with ``--impl both``).

    python tests/regression/run_regression.py --colab-url          # recommended: cloud GPU
    python tests/regression/run_regression.py --sample 24          # quick check (local GPU)
    python tests/regression/run_regression.py                      # everything
    python tests/regression/run_regression.py --impl python --languages de,ja
    python tests/regression/run_regression.py --update-baseline
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))  # plaude_local, when run as a script from anywhere

from languages import BY_CODE, THRESHOLDS, accepted_codes  # noqa: E402
from scoring import cer, content_recall, detected_language  # noqa: E402

CORPUS = HERE / "corpus"
BASELINE = HERE / "baseline.json"
PS_SCRIPT = REPO / "powershell" / "Plaude-Local.ps1"

# Best-first: the regression suite uses the strongest models available.
BEST_WHISPER = "large-v3"
BEST_TRANSLATORS = ("translategemma:27b", "translategemma:12b", "translategemma:4b")
OLLAMA_URL = "http://127.0.0.1:11434"
TOL_CER = 0.05       # allowed CER increase vs baseline
TOL_RECALL = 0.10    # allowed recall drop vs baseline


DEFAULT_LLM_TIMEOUT = 900  # s per LLM request: survive an 8 GB model (re)load


def build_command(impl: str, audio: Path, work: Path, *, variant: str, model: str,
                  engine: str, translate_model: Optional[str],
                  llm_timeout: int = DEFAULT_LLM_TIMEOUT) -> list:
    """The exact CLI invocation for one run (flag-for-flag parity across impls)."""
    html, tfile, xfile, stages = (work / "dashboard.html", work / "transcription.txt",
                                  work / "translation.txt", work / "stages")
    if impl == "python":
        cmd = [sys.executable, "-m", "plaude_local", str(audio), "-m", model,
               "-f", "html", "-o", str(html), "--transcription-file", str(tfile),
               "--translation-file", str(xfile), "--translate-to", "en",
               "--translate-engine", engine, "--summarize-timeout", str(llm_timeout),
               "-y", "-q"]
        if translate_model:
            cmd += ["--translate-model", translate_model]
        if variant == "damaged":
            cmd += ["--repair", "--enhance", "strong", "--keep-stages", str(stages)]
        return cmd
    cmd = [powershell_exe(), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
           str(PS_SCRIPT), str(audio), "-Model", model, "-Format", "html",
           "-Output", str(html), "-TranscriptionFile", str(tfile),
           "-TranslationFile", str(xfile), "-TranslateTo", "en",
           "-TranslateEngine", engine, "-SummarizeTimeout", str(llm_timeout),
           "-Yes", "-Quiet"]
    if translate_model:
        cmd += ["-TranslateModel", translate_model]
    if variant == "damaged":
        cmd += ["-Repair", "-Enhance", "strong", "-KeepStages", str(stages)]
    return cmd


def powershell_exe() -> str:
    """Windows PowerShell where it exists, else PowerShell 7 (``pwsh``; Linux,
    e.g. Google Colab)."""
    for name in ("powershell", "pwsh"):
        if shutil.which(name):
            return name
    return "pwsh"


def cer_comparable(expected: str, detected: Optional[str]) -> bool:
    """False when Whisper answered in an accepted related language that is
    WRITTEN in another script (e.g. Urdu audio detected as Hindi and written
    in Devanagari instead of Urdu's Arabic script): the transcript can be
    right while a character comparison against the reference is meaningless.
    Script variants the scorer folds (Chinese Traditional/Simplified, Serbian)
    stay comparable."""
    if not detected or detected == expected:
        return True
    a, b = BY_CODE.get(expected), BY_CODE.get(detected)
    return not (a and b and a["script"] != b["script"])


def evaluate(result: dict, entry: dict, variant: str,
             baseline: Optional[dict]) -> list:
    """Return the list of failure reasons for one scored run (empty = pass)."""
    failures = []
    if result["exit_code"] != 0:
        return [f"exit code {result['exit_code']}"]
    tier = entry.get("tier", "A")
    if tier in ("A", "B") and result["language"] not in accepted_codes(entry["code"]):
        failures.append(f"detected language {result['language']!r} != {entry['code']!r}")
    comparable = cer_comparable(entry["code"], result["language"])
    result["cer_comparable"] = comparable
    th = THRESHOLDS[tier][variant]
    if comparable and th["max_cer"] is not None and result["cer"] > th["max_cer"]:
        failures.append(f"CER {result['cer']:.3f} > {th['max_cer']}")
    if th["min_recall"] is not None and result["recall"] < th["min_recall"]:
        failures.append(f"EN recall {result['recall']:.3f} < {th['min_recall']}")
    if variant == "damaged":
        want = entry["damage"].get("hum_hz")
        if result.get("hum_hz") != want:
            failures.append(f"hum detected {result.get('hum_hz')} != injected {want}")
    if baseline:
        if comparable and result["cer"] > baseline["cer"] + TOL_CER:
            failures.append(f"CER regressed {baseline['cer']:.3f} -> {result['cer']:.3f}")
        if result["recall"] < baseline["recall"] - TOL_RECALL:
            failures.append(f"EN recall regressed {baseline['recall']:.3f} -> "
                            f"{result['recall']:.3f}")
    return failures


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def child_env() -> dict:
    """PATH plus the user scripts dir (Microsoft Store Python leaves it off PATH,
    which hides whisper-ctranslate2 from the PowerShell tool)."""
    env = dict(os.environ)
    extra = [sysconfig.get_path("scripts"), sysconfig.get_path("scripts", f"{os.name}_user")]
    env["PATH"] = os.pathsep.join([env.get("PATH", "")] + [p for p in extra if p and Path(p).is_dir()])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO), env.get("PYTHONPATH")]))
    return env


def run_with_timeout(cmd: list, log_path: Path, timeout: int, append: bool = False) -> int:
    """Run ``cmd`` with its output in ``log_path``; kill the whole tree on timeout.

    Output goes to a file, not a pipe: a grandchild (e.g. whisper-ctranslate2
    started by powershell.exe) inherits pipe handles, so killing only the
    direct child would leave ``communicate()`` blocked on the grandchild. The
    tree is also killed when the runner itself is interrupted (Ctrl+C, Colab
    "Interrupt"), so no tool keeps holding the GPU behind a new run's back.
    """
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    with open(log_path, "a" if append else "w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(cmd, cwd=REPO, env=child_env(), stdout=log,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                creationflags=flags, start_new_session=os.name != "nt")
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            kill_tree(proc)
            log.write(f"\n[regression] timeout after {timeout}s - process tree killed\n")
            return -1
        except BaseException:
            kill_tree(proc)
            raise
        if code < 0:  # killed by a signal: take any surviving grandchildren with it
            kill_tree(proc)
        return code


def kill_tree(proc: subprocess.Popen) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                       capture_output=True)
    else:  # pragma: no cover - POSIX
        import signal
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    proc.wait()


NOTEBOOK = "tests/regression/colab_regression.ipynb"


def colab_url(remote: str, branch: str) -> str:
    """The Colab link that opens this repo's regression notebook on ``branch``."""
    from urllib.parse import quote

    if not branch:
        raise ValueError("no branch (detached HEAD?) - check out a pushed branch first")
    m = re.search(r"github\.com(?::\d+)?[:/]+([^/]+)/([^/]+?)(?:\.git)?/?$", remote.strip())
    if not m:
        raise ValueError(f"not a GitHub remote: {remote!r}")
    return (f"https://colab.research.google.com/github/{m.group(1)}/{m.group(2)}"
            f"/blob/{quote(branch, safe='/')}/{NOTEBOOK}")


def notebook_branch() -> str:
    """The BRANCH the notebook clones (its first-cell setting)."""
    nb = json.loads((REPO / NOTEBOOK).read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        m = re.search(r'^BRANCH = "([^"]+)"', "".join(cell.get("source", [])), re.M)
        if m:
            return m.group(1)
    return ""


def git_colab_url() -> tuple:
    """(link, [warnings]) for running this branch's code on Colab."""
    def git(*a):
        return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True,
                              check=True).stdout.strip()
    branch = git("branch", "--show-current")
    url = colab_url(git("remote", "get-url", "origin"), branch)
    warnings = []
    if git("status", "--porcelain", "--untracked-files=no"):
        warnings.append("you have uncommitted changes - Colab tests the PUSHED code")
    try:
        if git("rev-list", "--count", "@{u}..HEAD") not in ("", "0"):
            warnings.append("you have unpushed commits - push first; Colab clones from GitHub")
    except subprocess.CalledProcessError:
        warnings.append(f"branch {branch!r} has no upstream - push it first")
    nb_branch = notebook_branch()
    if nb_branch and nb_branch != branch:
        warnings.append(f'the notebook clones BRANCH = "{nb_branch}"; set BRANCH = '
                        f'"{branch}" in its first cell to test this branch')
    return url, warnings


def killed_by_system(code: int) -> bool:
    """Exit codes worth one automatic retry: the process was killed by a
    signal (e.g. -9 from the out-of-memory killer on a busy Colab VM) -
    not a timeout (-1) and not a tool error (a positive code)."""
    return code < -1


def run_one(impl: str, entry: dict, variant: str, out_dir: Path, args) -> dict:
    audio = CORPUS / (entry["file"] if variant == "clean" else entry["damaged_file"])
    work = out_dir / impl / f"{entry['id']}.{variant}"
    work.mkdir(parents=True, exist_ok=True)
    cmd = build_command(impl, audio, work, variant=variant, model=args.model,
                        engine=args.translate_engine, translate_model=args.translate_model,
                        llm_timeout=args.llm_timeout)
    t0 = time.time()
    code = run_with_timeout(cmd, work / "run.log", args.timeout)
    if killed_by_system(code):
        print(f"      {impl}/{entry['id']}/{variant}: killed by the system (exit {code}); "
              "retrying once", flush=True)
        code = run_with_timeout(cmd, work / "run.log", args.timeout, append=True)

    transcript = _read(work / "transcription.txt")
    translation = _read(work / "translation.txt")
    result = {
        "impl": impl, "id": entry["id"], "code": entry["code"], "kind": entry["kind"],
        "tier": entry.get("tier"), "variant": variant,
        "exit_code": code, "seconds": round(time.time() - t0, 1),
        "audio_s": entry["duration_s"],
        "language": detected_language(_read(work / "dashboard.html")),
        "cer": round(cer(entry["text"], transcript, entry["code"]), 4),
        "recall": round(content_recall(entry["reference_en"], translation), 4),
    }
    if variant == "damaged":
        manifests = list((work / "stages").glob("*.stages.json"))
        result["hum_hz"] = (json.loads(manifests[0].read_text(encoding="utf-8")).get("hum_hz")
                            if manifests else "no-manifest")
    return result


def rescore(results_path: Path) -> int:
    """Recompute every run's failures in ``results_path`` (thresholds, script
    comparability, baseline) from the recorded scores; rewrite the file."""
    data = json.loads(results_path.read_text(encoding="utf-8"))
    items = {e["id"]: e for e in json.loads(
        (CORPUS / "manifest.json").read_text(encoding="utf-8"))["items"]}
    config = {k: data.get(k) for k in ("model", "translate_engine", "translate_model")}
    base = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else {}
    base_scores = base.get("results", {}) if base.get("config") == config else {}
    changed = 0
    for r in data["results"]:
        key = f"{r['impl']}/{r['id']}/{r['variant']}"
        new = evaluate(r, items[r["id"]], r["variant"], base_scores.get(key))
        if new != r.get("failures"):
            changed += 1
            print(f"{key}: {r.get('failures')} -> {new}")
        r["failures"] = new
    data["failed"] = sum(bool(r["failures"]) for r in data["results"])
    data["passed"] = len(data["results"]) - data["failed"]
    results_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"rescored {len(data['results'])} run(s), {changed} changed: "
          f"{data['passed']} passed, {data['failed']} failed")
    return 1 if data["failed"] else 0


def stratified_sample(entries: list, n: int, seed: int = 0) -> list:
    """``n`` items spread across languages and monologue/conversation."""
    import random

    rng = random.Random(seed)
    by_lang = {}
    for e in entries:
        by_lang.setdefault(e["code"], []).append(e)
    langs = sorted(by_lang)
    rng.shuffle(langs)
    picked, kind = [], "mono"
    while len(picked) < n and any(by_lang.values()):
        for code in langs:
            pool = by_lang[code]
            if not pool or len(picked) >= n:
                continue
            match = [e for e in pool if e["kind"] == kind] or pool
            choice = rng.choice(match)
            pool.remove(choice)
            picked.append(choice)
            kind = "conv" if kind == "mono" else "mono"
    return picked


def best_translate_model(installed: list) -> Optional[str]:
    """The strongest installed translation model (None: leave it to the tool)."""
    for name in BEST_TRANSLATORS:
        if name in installed:
            return name
    return None


def cuda_available() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--colab-url", action="store_true",
                   help="print the Google Colab link for this branch's regression notebook "
                        "(the recommended way to run the suite - no local GPU needed) and exit")
    p.add_argument("--rescore", metavar="RESULTS_JSON", default=None,
                   help="re-evaluate a finished results.json with the current pass/fail "
                        "rules (no tools are run) and rewrite it")
    p.add_argument("--impl", choices=["both", "python", "powershell"], default="both")
    p.add_argument("--variants", default="clean,damaged")
    p.add_argument("--languages", default="", help="comma-separated codes (default: all)")
    p.add_argument("--kinds", default="mono,conv", help="mono and/or conv")
    p.add_argument("--sample", type=int, default=0,
                   help="run a seeded stratified subset of N items (default: all)")
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--model", default=BEST_WHISPER,
                   help=f"Whisper model (default {BEST_WHISPER}, the most accurate)")
    p.add_argument("--translate-engine", default="llm", choices=["auto", "llm", "whisper"])
    p.add_argument("--translate-model", default="best",
                   help="Ollama model for translation; 'best' (default) picks the "
                        f"strongest installed of {', '.join(BEST_TRANSLATORS)}")
    p.add_argument("--out", default=None, help="results folder (default: results/<time>)")
    p.add_argument("--timeout", type=int, default=3600, help="per-file timeout (s)")
    p.add_argument("--llm-timeout", type=int, default=DEFAULT_LLM_TIMEOUT,
                   help="per-request LLM timeout passed to both tools (default "
                        f"{DEFAULT_LLM_TIMEOUT}s, so a model reload can't blank a translation)")
    p.add_argument("--resume", action="store_true",
                   help="skip runs that already PASSED in --out's results.json (e.g. after "
                        "a Colab disconnect); failed runs are run again")
    p.add_argument("--update-baseline", action="store_true",
                   help="record this run's scores as the new baseline")
    args = p.parse_args(argv)
    if args.rescore:
        return rescore(Path(args.rescore))
    if args.colab_url:
        url, warnings = git_colab_url()
        for w in warnings:
            print(f"warning: {w}", file=sys.stderr)
        print(url)
        return 0

    # Rule: regression runs use the GPU and the best models whenever possible.
    if args.translate_model == "best":
        from plaude_local import summarize
        installed = summarize.list_ollama_models(OLLAMA_URL)
        args.translate_model = best_translate_model(installed)
        if args.translate_engine != "whisper" and not args.translate_model:
            p.error(f"--translate-model best: no translation model found on Ollama at "
                    f"{OLLAMA_URL} (installed: {installed or 'none - is Ollama running?'}); "
                    f"pull one of {', '.join(BEST_TRANSLATORS)} or pass --translate-model")
    gpu = cuda_available()
    print(f"regression config: whisper={args.model} on {'GPU (CUDA)' if gpu else 'CPU'}, "
          f"translate={args.translate_engine}/{args.translate_model or 'tool default'}",
          flush=True)
    if not gpu:
        print("warning: no CUDA GPU - large-v3 on the CPU is very slow; pass --model small "
              "for a quick CPU check (its scores won't match a GPU baseline)", flush=True)

    manifest = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    wanted = {c.strip() for c in args.languages.split(",") if c.strip()}
    kinds = {k.strip() for k in args.kinds.split(",") if k.strip()}
    entries = [e for e in manifest["items"]
               if (not wanted or e["code"] in wanted) and e["kind"] in kinds]
    if args.sample:
        entries = stratified_sample(entries, args.sample, args.sample_seed)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    impls = ["python", "powershell"] if args.impl == "both" else [args.impl]

    out_dir = Path(args.out or HERE / "results" / datetime.now().strftime("%Y%m%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {"model": args.model, "translate_engine": args.translate_engine,
              "translate_model": args.translate_model}
    base = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else {}
    same_config = base.get("config") == config
    base_scores = base.get("results", {}) if same_config else {}
    if base and not same_config:
        print(f"note: baseline.json was recorded with {base.get('config')}; "
              f"no baseline comparison for {config}", flush=True)

    results_path = out_dir / "results.json"
    done = {}
    if args.resume and results_path.exists():
        try:
            previous = json.loads(results_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            aside = results_path.with_name(f"results.corrupt-{int(time.time())}.json")
            results_path.replace(aside)
            print(f"note: unreadable {results_path.name} ({exc}) moved to {aside.name}; "
                  "starting afresh", flush=True)
            previous = None
        if previous is not None:
            prev_config = {k: previous.get(k) for k in config}
            if prev_config != config:
                # Never overwrite results made with another config: stop instead.
                p.error(f"--resume: {results_path} was made with {prev_config}, this run "
                        f"is {config}; use another --out or the same settings")
            done = {f"{r['impl']}/{r['id']}/{r['variant']}": r
                    for r in previous.get("results", []) if r.get("failures") == []}
            print(f"resuming: {len(done)} passed run(s) kept from {results_path}", flush=True)

    def save():  # after every run, so a disconnect loses at most the current one
        summary = {**config, "passed": len(results) - failed, "failed": failed,
                   "results": results}
        tmp = results_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        tmp.replace(results_path)

    results, failed = [], 0
    total = len(impls) * len(entries) * len(variants)
    for impl in impls:
        for entry in entries:
            for variant in variants:
                key = f"{impl}/{entry['id']}/{variant}"
                if key in done:
                    r = done[key]
                else:
                    r = run_one(impl, entry, variant, out_dir, args)
                    r["failures"] = evaluate(r, entry, variant, base_scores.get(key))
                failed += bool(r["failures"])
                results.append(r)
                save()
                status = "PASS" if not r["failures"] else "FAIL"
                print(f"[{len(results):3}/{total}] {status} {key:34} lang={r['language']} "
                      f"CER={r['cer']:.3f} EN={r['recall']:.3f} "
                      f"{('hum=' + str(r.get('hum_hz')) + ' ') if variant == 'damaged' else ''}"
                      f"({r['seconds']:.0f}s for {r['audio_s']:.0f}s audio)"
                      + (f"  <- {'; '.join(r['failures'])}" if r["failures"] else ""),
                      flush=True)

    save()
    print(f"\n{len(results) - failed}/{len(results)} passed - details in {out_dir}")

    if args.update_baseline:
        merged = dict(base.get("results", {})) if same_config else {}
        merged.update({f"{r['impl']}/{r['id']}/{r['variant']}":
                       {"cer": r["cer"], "recall": r["recall"]} for r in results
                       if r["exit_code"] == 0 and r.get("failures") == []})
        BASELINE.write_text(json.dumps({"config": config, "results": dict(sorted(merged.items()))},
                                       indent=2) + "\n", encoding="utf-8")
        print(f"baseline updated -> {BASELINE}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
