"""Generate the multilingual speech corpus for the regression suite.

The corpus has 300 clean recordings plus a damaged twin of each (600 files):

* 150 single-speaker monologues and 150 multi-speaker conversations (2-3
  voices taking turns), spread round-robin over the 21 corpus languages in
  ``languages.py`` (16 Asian languages plus English, Spanish, French, German
  and Russian);
* each clean file is 2-10 minutes long (seeded, skewed towards shorter files)
  with seeded random content: a shuffled selection of ``sentences.py`` lines
  (monologue) or ``dialogues.py`` exchanges (conversation);
* each damaged twin adds a seeded mix of imperfections from ``damage.py``.

Two phases (``--phase``; the default runs both, overlapped):

* ``translate`` - translate the English banks into each language with a local
  LLM (the project's own ``summarize.translate_lines`` via Ollama), check the
  lines are in the right script and not English echoes, and save them to
  ``translations/<code>.json``. These files are COMMITTED, so this phase runs
  once. It needs no GPU: by default Ollama is told to keep the model on the CPU
  (``--ollama-device cpu`` sends ``num_gpu: 0``); transient server errors are
  retried with backoff.
* ``audio`` - needs no LLM and no GPU: synthesize each line with a Microsoft
  neural voice (``edge-tts``, a cloud service; cached per voice+line) and
  assemble files line by line - with natural pauses and exact speaker-turn
  timings - until the target length is reached, then encode with FFmpeg. Audio is mono Opus 16 kb/s to keep the committed
corpus small; ``manifest.json`` records every file's source text, exact English
reference, voices, turns and damage recipe.

This is a one-off, networked, multi-hour step (edge-tts calls Microsoft's TTS
service); the result is committed so regression runs are reproducible
offline. It resumes where it left off (translations and TTS clips are cached
outside the repo in ``~/.cache/plaude-local/regression``, so ``--force`` rebuilds files from the same cached material;
delete the cache folder to re-translate / re-synthesize). Requirements:
``pip install edge-tts``, FFmpeg on PATH, and an Ollama server with a
translation model.

    python tests/regression/generate_corpus.py                    # both phases
    python tests/regression/generate_corpus.py --phase translate  # CPU Ollama only
    python tests/regression/generate_corpus.py --phase audio      # no LLM at all
    python tests/regression/generate_corpus.py --languages de,ja --force
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import re
import time
import unicodedata
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from damage import damage_filtergraph, damage_plan  # noqa: E402
from dialogues import EXCHANGES, REACTIONS  # noqa: E402
from languages import LANGUAGES, NO_SPACE_SCRIPTS  # noqa: E402
from scoring import content_words  # noqa: E402
from sentences import SENTENCES  # noqa: E402

DEFAULT_SEED = 2026
N_MONOLOGUES = 150
N_CONVERSATIONS = 150
MIN_SECONDS, MAX_SECONDS = 120.0, 600.0
MAX_TARGET = 590.0
TTS_RATE = 24000
OPUS = ["-ac", "1", "-c:a", "libopus", "-b:a", "16k"]
PITCHES = ["+0Hz", "-45Hz", "+40Hz"]  # tell same-voice speakers apart
TTS_CONCURRENCY = 6
TRANSLATIONS = HERE / "translations"   # committed: the audio phase needs no LLM
# Not under %LOCALAPPDATA%: Microsoft Store Python silently redirects writes
# there into its package sandbox, so FFmpeg (a normal process) would not see
# the folders Python created.
CACHE = Path(os.environ.get("PLAUDE_TTS_CACHE")
             or Path.home() / ".cache" / "plaude-local" / "regression")


# --------------------------------------------------------------------------- #
# Planning (pure, unit-tested)
# --------------------------------------------------------------------------- #
def target_seconds(rng: random.Random) -> float:
    """2-10 minutes, skewed towards shorter files (mean ~4.6 min)."""
    return round(MIN_SECONDS + (MAX_TARGET - MIN_SECONDS) * rng.random() ** 2, 1)


def plan_corpus(seed: int = DEFAULT_SEED, languages=LANGUAGES,
                n_mono: int = N_MONOLOGUES, n_conv: int = N_CONVERSATIONS) -> list:
    """Every clean item to generate: language, kind, target length, speakers."""
    items, count = [], {}
    n = len(languages)
    for kind, total, offset in (("mono", n_mono, 0), ("conv", n_conv, n // 2)):
        for i in range(total):
            lang = languages[(i + offset) % n]
            key = (lang["code"], kind)
            count[key] = count.get(key, 0) + 1
            item_id = f"{lang['code']}-{kind}-{count[key]:02d}"
            rng = random.Random(f"{seed}-{item_id}")
            items.append({
                "id": item_id, "code": lang["code"], "kind": kind,
                "target_s": target_seconds(rng),
                "speakers": 1 if kind == "mono" else rng.choice([2, 2, 3]),
            })
    return items


def script_ratio(text: str, script: str) -> float:
    """Share of the letters in ``text`` written in ``script`` (CJK: Han/kana)."""
    names = {"CJK": ("CJK", "HIRAGANA", "KATAKANA")}.get(script, (script,))
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    if not letters:
        return 0.0
    ok = sum(1 for c in letters if unicodedata.name(c, "").startswith(names))
    return ok / len(letters)


_THINK = re.compile(r"(?is)<think>.*?</think>")
_PREAMBLE = re.compile(r"(?i)^(here is|here's|translation|sure[,!.])")


def clean_reply(text: str) -> str:
    """One translated line from an LLM reply: no reasoning, preamble or quotes."""
    lines = [l.strip() for l in _THINK.sub("", text or "").splitlines() if l.strip()]
    lines = [l for l in lines if not _PREAMBLE.match(l)] or lines
    return lines[0].strip(" \"'\u201c\u201d\u00ab\u00bb") if lines else ""


def translation_ok(src_en: str, text: str, script: str) -> bool:
    """A usable translation: non-empty, right script, and not (mostly) English."""
    text = (text or "").strip()
    if not text or script_ratio(text, script) < 0.6:
        return False
    if script != "LATIN":
        return True
    if _PREAMBLE.match(text) or text.casefold().strip(".!? ") == src_en.casefold().strip(".!? "):
        return False
    # An English echo keeps most of the source's content words; a real
    # translation keeps at most a few cognates (model, internet, museum ...).
    src = content_words(src_en)
    shared = src & content_words(text)
    return not (len(src) >= 3 and len(shared) >= 0.6 * len(src))


# --------------------------------------------------------------------------- #
# Translation (Ollama, cached per language)
# --------------------------------------------------------------------------- #
def english_lines() -> list:
    return (list(SENTENCES) + [line for ex in EXCHANGES for line in ex] + list(REACTIONS))


def make_ollama_call(model: str, url: str, *, cpu: bool = True, timeout: float = 1800,
                     retries: int = 5, sleep=time.sleep):
    """``prompt -> completion`` against Ollama; ``cpu`` keeps the model off the GPU.

    Retries transient failures (Ollama answers HTTP 500 while it reloads or
    runs out of memory) with exponential backoff before giving up.
    """
    from plaude_local import summarize

    options = {"num_ctx": summarize._OLLAMA_NUM_CTX, "temperature": 0}
    if cpu:
        options["num_gpu"] = 0  # offload zero layers: pure CPU inference

    def call(prompt: str) -> str:
        payload = {"model": model, "prompt": prompt, "stream": False, "options": options}
        for attempt in range(retries):
            try:
                resp = summarize._post_json(url.rstrip("/") + "/api/generate", payload, timeout)
                text = (resp.get("response") or "").strip()
                if text:
                    return text
                raise summarize.SummarizeError(f"empty response: {resp!r}")
            except summarize.SummarizeError:
                if attempt == retries - 1:
                    raise
                sleep(min(300, 10 * 2 ** attempt))
        return ""  # pragma: no cover
    return call


def translation_path(code: str) -> Path:
    return TRANSLATIONS / f"{code}.json"


def load_translation(code: str):
    path = translation_path(code)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def translate_language(lang: dict, call, model: str = "") -> dict:
    """Translated sentence / exchange / reaction banks for ``lang``.

    Reuses ``translations/<code>.json`` when present; otherwise translates
    with ``call`` and writes that file (commit it)."""
    from plaude_local import summarize

    existing = load_translation(lang["code"])
    if existing:
        return existing
    src = english_lines()
    if lang["code"] == "en":
        out = list(src)
    else:
        out = [clean_reply(line) for line in
               summarize.translate_lines(src, call, target_language=lang["name"])]
        for i, line in enumerate(out):  # retry lines the batched reply got wrong
            for _ in range(2):
                if translation_ok(src[i], out[i], lang["script"]):
                    break
                out[i] = clean_reply(summarize.translate_text(
                    src[i], call, target_language=lang["name"]))
        bad = [i for i, line in enumerate(out) if not translation_ok(src[i], line, lang["script"])]
        if len(bad) > len(src) * 0.05:
            raise RuntimeError(f"{lang['code']}: {len(bad)} of {len(src)} lines are not "
                               f"valid {lang['name']} translations")
        for i in bad:  # a handful of stubborn lines: drop them from use
            out[i] = ""
    n_s, n_e = len(SENTENCES), len(EXCHANGES)
    data = {
        "language": lang["code"],
        "model": model if lang["code"] != "en" else None,
        "sentences": out[:n_s],
        "exchanges": [[out[n_s + 2 * k], out[n_s + 2 * k + 1]] for k in range(n_e)],
        "reactions": out[n_s + 2 * n_e:],
    }
    path = translation_path(lang["code"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return data


# --------------------------------------------------------------------------- #
# Speech synthesis (edge-tts, cached per voice + pitch + text)
# --------------------------------------------------------------------------- #
async def _tts_one(text: str, voice: str, pitch: str, sem: asyncio.Semaphore) -> Path:
    import edge_tts

    wav = tts_path(text, voice, pitch)
    if wav.exists():
        return wav
    wav.parent.mkdir(parents=True, exist_ok=True)
    async with sem:
        for attempt in range(4):
            if wav.exists():  # another coroutine / run finished it meanwhile
                return wav
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    mp3 = Path(tmp) / "a.mp3"
                    await edge_tts.Communicate(text, voice, pitch=pitch).save(str(mp3))
                    part = wav.with_name(f"{wav.stem}.{uuid.uuid4().hex}.part.wav")
                    subprocess.run(
                        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(mp3),
                         "-ac", "1", "-ar", str(TTS_RATE), "-c:a", "pcm_s16le", str(part)],
                        check=True)
                    part.replace(wav)
                return wav
            except Exception:
                if attempt == 3:
                    raise
                await asyncio.sleep(2 * (attempt + 1))
    return wav


def tts_path(text: str, voice: str, pitch: str) -> Path:
    key = hashlib.sha1(f"{voice}|{pitch}|{text}".encode("utf-8")).hexdigest()
    return CACHE / "tts" / voice / f"{key}.wav"


def synthesize(lines: list) -> list:
    """TTS ``[(text, voice, pitch)]`` concurrently; returns cached wav paths.

    Identical lines are synthesized once (two coroutines must never write the
    same cache file)."""
    unique = list(dict.fromkeys(lines))

    async def go():
        sem = asyncio.Semaphore(TTS_CONCURRENCY)
        await asyncio.gather(*(_tts_one(t, v, p, sem) for t, v, p in unique))
    asyncio.run(go())
    return [tts_path(*line) for line in lines]


def list_voices() -> dict:
    """locale -> [voice short names] from edge-tts."""
    import edge_tts

    voices = asyncio.run(edge_tts.list_voices())
    by = {}
    for v in voices:
        by.setdefault(v["Locale"], []).append(v["ShortName"])
    return {k: sorted(v) for k, v in by.items()}


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def script_lines(item: dict, bank: dict, rng: random.Random, speakers: list) -> list:
    """The full seeded line order for an item: [(speaker, text, english)]."""
    if item["kind"] == "mono":
        order = list(range(len(SENTENCES)))
        rng.shuffle(order)
        return [(0, bank["sentences"][i], SENTENCES[i]) for i in order if bank["sentences"][i]]
    order = list(range(len(EXCHANGES)))
    rng.shuffle(order)
    out = []
    for i in order:
        q, a = bank["exchanges"][i]
        if not q or not a:
            continue
        asker = rng.randrange(len(speakers))
        answerer = rng.choice([s for s in range(len(speakers)) if s != asker])
        out += [(asker, q, EXCHANGES[i][0]), (answerer, a, EXCHANGES[i][1])]
        if len(speakers) == 3 and rng.random() < 0.35:
            third = ({0, 1, 2} - {asker, answerer}).pop()
            r = rng.randrange(len(REACTIONS))
            if bank["reactions"][r]:
                out.append((third, bank["reactions"][r], REACTIONS[r]))
    return out


def pick_speakers(item: dict, lang: dict, voices: dict, rng: random.Random) -> list:
    pool = sorted({v for loc in lang["locales"] for v in voices.get(loc, [])})
    if not pool:
        raise RuntimeError(f"no edge-tts voice for {lang['code']} ({lang['locales']})")
    rng.shuffle(pool)
    if len(pool) >= item["speakers"]:
        return [(v, "+0Hz") for v in pool[: item["speakers"]]]
    # Not enough distinct voices: reuse them at different pitches.
    return [(pool[i % len(pool)], PITCHES[i // len(pool) % len(PITCHES)])
            for i in range(item["speakers"])]


def _wav_frames(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        return w.readframes(w.getnframes())


def assemble(item: dict, lang: dict, bank: dict, voices: dict, corpus: Path, seed: int) -> dict:
    rng = random.Random(f"{seed}-{item['id']}-content")
    speakers = pick_speakers(item, lang, voices, rng)
    lines = script_lines(item, bank, rng, speakers)
    code = item["code"]

    chunks, turns, t = [], [], 0.0
    lead = int(0.5 * TTS_RATE) * 2
    chunks.append(b"\x00" * lead)
    t += 0.5
    pos, prev_speaker = 0, None
    while t < item["target_s"] and pos < len(lines):
        batch = lines[pos:pos + 24]
        wavs = synthesize([(text, *speakers[s]) for s, text, _ in batch])
        for (speaker, text, en), wav in zip(batch, wavs):
            if t >= item["target_s"]:
                break
            frames = _wav_frames(wav)
            dur = len(frames) / 2 / TTS_RATE
            if prev_speaker is None:
                gap = 0.0
            elif item["kind"] == "mono" or speaker == prev_speaker:
                gap = rng.uniform(0.3, 0.6)
            else:
                gap = rng.uniform(0.2, 0.8)
            if t + gap + dur > MAX_SECONDS - 1.0:
                pos = len(lines)
                break
            chunks.append(b"\x00" * (int(gap * TTS_RATE) * 2))
            t += gap
            chunks.append(frames)
            turns.append({"speaker": speaker, "start": round(t, 2),
                          "end": round(t + dur, 2), "text": text, "en": en})
            t += dur
            prev_speaker = speaker
        pos += len(batch)
    chunks.append(b"\x00" * lead)
    t += 0.5
    if not MIN_SECONDS <= t <= MAX_SECONDS:
        raise RuntimeError(f"{item['id']}: {t:.0f}s outside {MIN_SECONDS:.0f}-{MAX_SECONDS:.0f}s")

    clean = corpus / f"{item['id']}.opus"
    damaged = corpus / f"{item['id']}.damaged.opus"
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "clean.wav"
        with wave.open(str(raw), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(TTS_RATE)
            w.writeframes(b"".join(chunks))
        _ffmpeg(["-i", str(raw), *OPUS, str(clean)])
        plan = damage_plan(random.Random(f"{seed}-{item['id']}-damage"))
        dwav = Path(tmp) / "damaged.wav"
        _ffmpeg(["-i", str(raw), "-filter_complex", damage_filtergraph(plan),
                 "-map", "[out]", "-c:a", "pcm_s16le", str(dwav)])
        _ffmpeg(["-i", str(dwav), *OPUS, str(damaged)])

    sep = "" if code in NO_SPACE_SCRIPTS else " "
    entry = {
        **item,
        "name": lang["name"],
        "tier": lang["tier"],
        "file": clean.name,
        "damaged_file": damaged.name,
        "duration_s": round(t, 2),
        "voices": [{"voice": v, "pitch": p} for v, p in speakers],
        "text": sep.join(turn["text"] for turn in turns),
        "reference_en": " ".join(turn["en"] for turn in turns),
        "damage": plan,
    }
    if item["kind"] == "conv":
        entry["turns"] = [{k: turn[k] for k in ("speaker", "start", "end")} for turn in turns]
    return entry


def _ffmpeg(args: list) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--corpus", default=str(HERE / "corpus"), help="output folder")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--languages", default="", help="comma-separated codes (default: all)")
    p.add_argument("--phase", choices=["all", "translate", "audio"], default="all",
                   help="translate (Ollama -> translations/*.json), audio (edge-tts + "
                        "FFmpeg; no LLM), or all (default)")
    p.add_argument("--model", default=None,
                   help="Ollama model for translation (default: first installed)")
    p.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    p.add_argument("--ollama-device", choices=["cpu", "auto"], default="cpu",
                   help="cpu (default): keep the model off the GPU (num_gpu 0); "
                        "auto: let Ollama decide")
    p.add_argument("--force", action="store_true", help="regenerate existing items")
    args = p.parse_args(argv)
    wanted = {c.strip() for c in args.languages.split(",") if c.strip()}
    model = args.model or ""

    def translator():
        from plaude_local import summarize
        name = args.model or summarize.default_ollama_model(args.ollama_url)
        if not name:
            p.error(f"no Ollama model at {args.ollama_url}; start Ollama / pass --model")
        return name, make_ollama_call(name, args.ollama_url, cpu=args.ollama_device == "cpu")

    if args.phase == "translate":
        model, call = translator()
        failed = []
        for lang in LANGUAGES:
            if (wanted and lang["code"] not in wanted) or load_translation(lang["code"]):
                continue
            t0 = time.time()
            try:
                translate_language(lang, call, model)
                print(f"{lang['code']}: translated ({time.time() - t0:.0f}s)", flush=True)
            except Exception as exc:
                failed.append(lang["code"])
                print(f"{lang['code']}: translation FAILED - {exc}", flush=True)
        done = sum(1 for l in LANGUAGES if load_translation(l["code"]))
        print(f"done: {done}/{len(LANGUAGES)} languages translated; failed: {failed or 'none'}")
        return 1 if failed else 0

    corpus = Path(args.corpus)
    corpus.mkdir(parents=True, exist_ok=True)
    manifest_path = corpus / "manifest.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.exists() else {})
    if manifest.get("seed") not in (None, args.seed) and not args.force:
        p.error(f"the corpus was generated with --seed {manifest['seed']}; "
                f"pass --force to regenerate it with --seed {args.seed}")
    entries = {e["id"]: e for e in manifest.get("items", [])}
    voices = list_voices()

    plan = [it for it in plan_corpus(args.seed) if not wanted or it["code"] in wanted]
    langs = [lang for lang in LANGUAGES if any(it["code"] == lang["code"] for it in plan)]

    def todo(it):
        e = entries.get(it["id"])
        return (args.force or not e or not (corpus / e["file"]).exists()
                or not (corpus / e["damaged_file"]).exists())

    def save():
        order = {it["id"]: i for i, it in enumerate(plan_corpus(args.seed))}
        manifest_path.write_text(json.dumps({
            "description": "Multilingual regression corpus - see generate_corpus.py.",
            "seed": args.seed,
            "translations": "translations/<code>.json (see each file's model)",
            "tts": "edge-tts (Microsoft neural voices); mono Opus 16 kb/s",
            "items": sorted(entries.values(), key=lambda e: order.get(e["id"], 1e9)),
        }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    # Translate the next language on a worker thread while this one synthesizes.
    # On any error (or Ctrl+C) cancel the queued translations instead of
    # silently working through every remaining language first.
    need = [lang for lang in langs if any(todo(it) for it in plan if it["code"] == lang["code"])]
    call = None
    if args.phase == "all" and any(not load_translation(l["code"]) for l in need):
        model, call = translator()

    def get_bank(lang):
        bank = load_translation(lang["code"])
        if bank:
            return bank
        if call is None:
            raise RuntimeError("no translations/%s.json - run --phase translate first" % lang["code"])
        return translate_language(lang, call, model)

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        futures = {lang["code"]: pool.submit(get_bank, lang) for lang in need}
        for lang in langs:
            items = [it for it in plan if it["code"] == lang["code"] and todo(it)]
            if not items:
                continue
            try:
                bank = futures[lang["code"]].result()
            except Exception as exc:  # keep going: report the language at the end
                print(f"{lang['code']}: translation FAILED - {exc}", flush=True)
                continue
            for it in items:
                entry = assemble(it, lang, bank, voices, corpus, args.seed)
                entries[it["id"]] = entry
                save()
                print(f"{it['id']:14} {entry['duration_s']:6.1f}s  "
                      f"{len(entry['voices'])} voice(s)", flush=True)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    missing = [it["id"] for it in plan if it["id"] not in entries
               or not (corpus / entries[it["id"]]["file"]).exists()]
    print(f"done: {len(entries)} items ({2 * len(entries)} files); missing: {missing or 'none'}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
