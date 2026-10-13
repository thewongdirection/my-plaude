# Regression results

Recorded outcomes of the multilingual end-to-end regression suite (see the
[main README](../../README.md#multilingual-end-to-end-regression-suite) for what it
tests and how it scores). Newest first. Raw per-run data is committed next to this
file under [`reports/`](reports/).

## October 2026 - full Python run: 598 of 600 passed

**Scope.** All 600 Python-tool runs: 300 corpus items (150 monologues, 150
multi-speaker conversations, 21 languages), each as a clean and a damaged variant
(`--repair --enhance strong`). The 600 PowerShell runs were **not** run in full (see
*Not covered*); PowerShell parity rests on the Pester suite and the 48-run
sample below.

**Configuration.** Whisper `large-v3`, float16 on an NVIDIA RTX 5080 (16 GB).
English translation by a local LLM through **llama.cpp**: `llama-server` serving
`gemma-4-26B-A4B-it-qat` (UD-Q4_K_XL, experts partly offloaded to the CPU with
`--n-cpu-moe 24`, 16k context), selected with `--translate-backend llamacpp`, so
translation ran entirely on the local machine (no cloud LLM). A 16 GB card cannot hold a ~14 GB LLM and
Whisper together, which is why a mixture-of-experts model with CPU-offloaded
experts was used.

**Result.** 598 passed, 2 failed.

| Lang | runs | pass | mean CER clean | mean CER damaged | mean EN recall | detected-lang mismatches |
|---|---|---|---|---|---|---|
| bn | 30 | 29 | 0.289 | 0.259 | 0.619 | 0 |
| de | 28 | 28 | 0.011 | 0.024 | 0.898 | 0 |
| en | 28 | 28 | 0.011 | 0.023 | 0.995 | 0 |
| es | 28 | 28 | 0.005 | 0.009 | 0.910 | 0 |
| fr | 28 | 28 | 0.004 | 0.028 | 0.902 | 0 |
| hi | 30 | 30 | 0.051 | 0.099 | 0.848 | 0 |
| id | 28 | 28 | 0.008 | 0.010 | 0.891 | 0 |
| ja | 30 | 30 | 0.032 | 0.029 | 0.821 | 0 |
| km | 28 | 28 | 0.895 | 0.839 | 0.032 | 6 |
| ko | 28 | 28 | 0.050 | 0.046 | 0.818 | 0 |
| ms | 28 | 28 | 0.016 | 0.047 | 0.851 | 0 |
| my | 30 | 30 | 0.947 | 0.954 | 0.026 | 0 |
| ru | 28 | 28 | 0.003 | 0.010 | 0.892 | 0 |
| ta | 28 | 27 | 0.145 | 0.307 | 0.582 | 0 |
| te | 28 | 28 | 0.522 | 0.520 | 0.438 | 0 |
| th | 28 | 28 | 0.033 | 0.093 | 0.764 | 0 |
| tl | 28 | 28 | 0.024 | 0.026 | 0.855 | 0 |
| ur | 28 | 28 | 0.451 | 0.538 | 0.707 | 13 |
| vi | 28 | 28 | 0.013 | 0.032 | 0.866 | 0 |
| yue | 30 | 30 | 0.265 | 0.271 | 0.763 | 0 |
| zh | 30 | 30 | 0.012 | 0.095 | 0.827 | 0 |

CER = character error rate against the source text (0 = perfect, lower is better).
EN recall = share of the English reference's content words found in the English
translation (higher is better). "Detected-lang mismatches" counts runs where
Whisper answered with a different language than the recording's (mostly Urdu heard
as Hindi; for Khmer, other languages such as Vietnamese).

**Reading the table**

- **Strong (CER <= 0.1 on clean audio):** English, Spanish, French, German, Russian,
  Indonesian, Vietnamese, Malay, Tagalog, Korean, Japanese, Chinese, Thai and Hindi.
- **Usable but weaker:** Bengali (~0.29), Cantonese (~0.27), Tamil (0.15 clean /
  0.31 damaged) and Telugu (~0.52).
- **Not usable with `large-v3`:** Khmer (CER ~0.9) and Burmese (~0.95). Both are
  tier C, where the suite only checks the exit code, so they "pass" with
  essentially unusable transcripts. Telugu is close behind.
- **Urdu** is heard as Hindi in 13 of 28 runs, almost entirely conversations (see
  *Findings*). The transcript is then written in Devanagari, so its CER (~0.45-0.54)
  is not comparable; the English recall (0.71) shows the content came through.
- Damaged variants cost little on well-supported languages (e.g. Spanish 0.005 to
  0.009) and more on weaker ones (Tamil 0.145 to 0.307, Chinese 0.012 to 0.095).
  `--dehum auto` detected the injected 50/60 Hz hum correctly where applicable.

**Failures (2)**

| Run | Result | Limit | Note |
|---|---|---|---|
| `python/ta-mono-01/damaged` | CER 0.844, EN recall 0.111 | CER <= 0.8, EN >= 0.2 | Unstable: scored CER 0.930, 0.772 and 0.844 on three attempts. `--batched` scores 0.261. |
| `python/bn-mono-02/clean` | CER 0.637 | CER <= 0.6 | Consistent near miss on every attempt. |

## Findings and the changes they led to

The run found real defects, fixed in the same period (all merged):

1. **GPU transcription failed on newer CTranslate2** with "Library cublas64_12.dll
   is not found": it loads cuBLAS with a plain `LoadLibrary`, which ignores
   `os.add_dll_directory`. The `nvidia-*-cu12` directories are now also put on
   `PATH` (Python tool and the PowerShell CUDA bootstrap).
2. **llama.cpp output was garbled** (leaked `<|channel>thought` tokens, endless
   repeated "Summary:" lines): the tool sent raw prompts to `/completion`. Both
   tools now use `/v1/chat/completions` with thinking disabled, strip leaked
   reasoning blocks (including an unclosed one), and raise an error instead of
   returning a reply that was cut off at the token cap.
3. **Whisper repetition loops** (Bengali, Tamil) came from carrying text across
   windows. `condition_on_previous_text` is now off by default in both tools
   (`--condition-on-previous` / `-ConditionOnPrevious` restores it). Measured on the
   corpus: Bengali clean 0.856 to 0.309 CER with repeated lines 33% to 0%; Chinese
   `zh-mono-01` clean 0.323 to 0.015; Malay 0.206 to ~0.004.
4. **Opt-in batched decoding** (`--batched` / `-Batched`, 15 s chunks, 8 in
   parallel). See the comparison below. Default behavior is unchanged.
5. **Hindi/Urdu ambiguity:** averaging the language probabilities over the whole
   recording did **not** help (25/29 correct against 26/29 for the first window);
   Urdu conversations are about 50/50 Hindi/Urdu in every window while real Hindi
   was always >= 0.96. Silently forcing a language would mislabel real speakers, so
   both tools now **warn** when auto-detection picks `hi`/`ur` below 0.9
   confidence and advise `--language hi` or `--language ur`.
6. **Regression runner:** `--translate-backend {auto,ollama,llamacpp}` and
   `--llm-url` (llama.cpp as the translation server) and `--batched` pass-through.

### Batched vs default decoding (24-item sample, 48 runs each way)

| | default | `--batched` |
|---|---|---|
| passed | 48/48 | 48/48 |
| mean CER | 0.168 | 0.162 |
| mean EN recall | 0.721 | 0.715 |
| total time | 3,933 s | 2,280 s |

Per file (CER change over +-0.03): 9 better, 32 about the same, 6 worse. It helps
**Tamil** (mean CER -0.13; conversation 7: 0.16/0.25 to 0.06/0.09), Cantonese
damaged, Malay, Korean and Hindi, and it detected one Urdu conversation as Urdu that
default decoding heard as Hindi. It hurts **Bengali** (+0.13), Telugu damaged
(0.56 to 0.84), Cantonese clean (0.03 to 0.17) and, very slightly, one clean Chinese file (0.004 to 0.043). Single attempts
vary by about 0.1 on their own (see `ta-mono-01/damaged` above), so only the large
swings are reliable. Net: no gain on average, mixed by language, hence **opt-in**.
Data: [`reports/2026-10-python/batched-vs-default-sample.txt`](reports/2026-10-python/batched-vs-default-sample.txt).

On Tamil's damaged recording specifically, with both tools end to end: Python
`--batched` and PowerShell `-Batched` both score CER 0.261 with identical segments
(defaults 0.706 and 0.780).

## Earlier sample: both implementations (96 runs)

A 24-item stratified sample (`--sample 24`) run through **both** tools before the
fixes above: 96/96 passed (48 Python, 48 PowerShell). The tools' CERs agree closely
(mean absolute difference 0.021, largest 0.154 on a Tamil conversation) and detect the
same hum frequencies.

## Not covered / caveats

- **PowerShell was not run on the full corpus** (the 48-run sample and the Pester
  suite cover it; the two tools share the corpus, scoring and flags).
- **No `baseline.json` was recorded** (`--update-baseline` was not run), so there is
  no regression comparison yet for this model and translation setup.
- **Mixed code versions:** the first 104 results came from the code before the
  Whisper no-condition fix; passed runs were kept on resume and the 5 early failures
  were re-run with the fix. Failures re-run on every resume.
- **Decoding is not deterministic** run to run; differences below ~0.1 CER on one
  file are noise.
- Translation quality depends on the local model; a different LLM will shift the
  English-recall numbers (not CER).

## Reproduce

```powershell
# start llama-server with a translation-capable model (own GPU memory!), then:
python tests/regression/run_regression.py --impl python --translate-backend llamacpp --out <dir> --resume
python tests/regression/run_regression.py --impl python --sample 24 --batched --translate-backend llamacpp --out <dir2>
```

`--resume` continues an interrupted run (passed runs are kept, failed ones re-run).
The runner stops by itself after the selected runs. Use `--impl python` to skip the
PowerShell half.
