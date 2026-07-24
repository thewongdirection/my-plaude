"""Render a self-contained HTML dashboard for a transcription.

The dashboard is the default output format (``--format html``): a header with
the detected language, rough speech duration and word count, a <=250-word summary
of critical topics, then three tabs — Transcribed (original), the translation
(English by default), and a Side-by-Side view.

Pure/offline: takes already-computed text + metadata and returns an HTML string.
No network, no heavy imports — so it is fully unit-testable.
"""
from __future__ import annotations

import html as _html
import re
from typing import Optional

# CJK ideographs + Japanese kana + Hangul: languages without spaces between words.
_CJK = re.compile(r"[㐀-鿿豈-﫿぀-ヿ가-힯]")

# Minimal ISO-639-1 -> display name map (extended as needed; falls back to code).
_LANG_NAMES = {
    "en": "English", "zh": "Chinese", "ms": "Malay", "es": "Spanish",
    "fr": "French", "de": "German", "ja": "Japanese", "ko": "Korean",
    "id": "Indonesian", "hi": "Hindi", "ar": "Arabic", "pt": "Portuguese",
    "ru": "Russian", "it": "Italian", "vi": "Vietnamese", "th": "Thai",
    "nl": "Dutch", "tr": "Turkish", "pl": "Polish", "uk": "Ukrainian",
}


def language_name(code: Optional[str]) -> str:
    if not code:
        return "Unknown"
    return _LANG_NAMES.get(code.lower(), code)


def count_words(text: str) -> int:
    """Word count that is sensible for both spaced and CJK scripts.

    Counts each CJK/kana/Hangul character as one word, plus whitespace-delimited
    tokens for the rest of the text.
    """
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    spaced = len(_CJK.sub(" ", text).split())
    return cjk + spaced


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "—"  # em dash
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {sec:02d}s"
    if m:
        return f"{m}m {sec:02d}s"
    return f"{sec}s"


def clamp_words(text: str, max_words: int = 250) -> str:
    """Defensive cap so the summary never exceeds ``max_words`` words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(".,;:") + " …"


_CSS = """
:root{--bg:#f6f7f9;--surface:#fff;--text:#1a1d23;--muted:#5b6270;--accent:#0e7c86;
--accent-ink:#0a5b62;--accent-soft:#e2f1f1;--border:#e4e7eb;
--shadow:0 1px 2px rgba(20,25,35,.04),0 8px 24px rgba(20,25,35,.06);
--font-sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
--font-read:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,"Songti SC","Noto Serif CJK SC","Microsoft YaHei",serif;
--font-mono:ui-monospace,"Cascadia Code","SF Mono",Menlo,Consolas,monospace;}
@media (prefers-color-scheme:dark){:root{--bg:#0f1216;--surface:#171b21;--text:#e8eaed;
--muted:#9aa2ad;--accent:#43c9c0;--accent-ink:#7ee0d8;--accent-soft:#12332f;--border:#262b33;
--shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);}}
:root[data-theme="light"]{--bg:#f6f7f9;--surface:#fff;--text:#1a1d23;--muted:#5b6270;--accent:#0e7c86;
--accent-ink:#0a5b62;--accent-soft:#e2f1f1;--border:#e4e7eb;
--shadow:0 1px 2px rgba(20,25,35,.04),0 8px 24px rgba(20,25,35,.06);}
:root[data-theme="dark"]{--bg:#0f1216;--surface:#171b21;--text:#e8eaed;--muted:#9aa2ad;--accent:#43c9c0;
--accent-ink:#7ee0d8;--accent-soft:#12332f;--border:#262b33;
--shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);}
*{box-sizing:border-box;}
body{margin:0;background:var(--bg);color:var(--text);font-family:var(--font-sans);line-height:1.5;-webkit-font-smoothing:antialiased;}
.wrap{max-width:860px;margin:0 auto;padding:clamp(20px,4vw,56px) clamp(16px,4vw,28px) 72px;}
.kicker{font-family:var(--font-mono);font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);margin:0 0 12px;display:flex;align-items:center;gap:10px;}
.kicker::before{content:"";width:26px;height:2px;background:var(--accent);border-radius:2px;}
h1{font-family:var(--font-read);font-weight:600;letter-spacing:-.01em;font-size:clamp(1.7rem,4vw,2.4rem);line-height:1.1;margin:0 0 20px;text-wrap:balance;}
.stats{display:flex;gap:12px;flex-wrap:wrap;margin:0 0 26px;}
.stat{flex:1 1 150px;background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px 16px;box-shadow:var(--shadow);}
.stat .label{font-family:var(--font-mono);font-size:.66rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin:0 0 6px;}
.stat .value{font-size:1.5rem;font-weight:600;font-variant-numeric:tabular-nums;line-height:1;}
.summary{background:var(--surface);border:1px solid var(--border);border-left:3px solid var(--accent);border-radius:12px;padding:18px 22px;box-shadow:var(--shadow);margin:0 0 8px;}
.summary h2{font-family:var(--font-read);font-size:1.15rem;margin:0 0 10px;}
.summary .body{color:var(--text);}
.summary .body h1,.summary .body h2,.summary .body h3{font-size:1rem;margin:.8em 0 .3em;}
.summary .body ul{margin:.3em 0;padding-left:1.2em;}
.summary .note{color:var(--muted);font-style:italic;}
.tabs{display:flex;gap:4px;margin:28px 0 0;border-bottom:1px solid var(--border);flex-wrap:wrap;}
.tab{appearance:none;border:0;background:transparent;cursor:pointer;font-family:var(--font-sans);font-size:.94rem;color:var(--muted);padding:11px 15px;border-bottom:2px solid transparent;margin-bottom:-1px;transition:color .15s;}
.tab:hover{color:var(--text);}
.tab[aria-selected="true"]{color:var(--text);border-bottom-color:var(--accent);font-weight:600;}
.tab:focus-visible{outline:2px solid var(--accent);outline-offset:3px;border-radius:6px;}
.panel{margin-top:22px;}
.panel[hidden]{display:none;}
.doc{background:var(--surface);border:1px solid var(--border);border-radius:14px;box-shadow:var(--shadow);padding:clamp(20px,3.5vw,34px);overflow-x:auto;}
.doc pre{margin:0;font-family:var(--font-read);font-size:1.03rem;line-height:1.72;white-space:pre-wrap;word-break:break-word;}
.doc.cjk pre{line-height:1.95;font-size:1.08rem;}
.sbs{display:grid;grid-template-columns:1fr 1fr;gap:14px;}
@media (max-width:640px){.sbs{grid-template-columns:1fr;}}
.sbs .col h3{font-family:var(--font-mono);font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;color:var(--accent);margin:0 0 8px;}
footer{margin-top:36px;padding-top:18px;border-top:1px solid var(--border);color:var(--muted);font-size:.8rem;font-family:var(--font-mono);}
@media (prefers-reduced-motion:reduce){*{transition:none!important;}}
"""

_JS = """
const IDS=["transcribed","translation","sbs"];
function sel(id){IDS.forEach(x=>{const on=x===id;
document.getElementById("tab-"+x).setAttribute("aria-selected",on);
document.getElementById("panel-"+x).hidden=!on;});}
IDS.forEach(x=>document.getElementById("tab-"+x).addEventListener("click",()=>sel(x)));
"""


def _summary_html(summary: Optional[str], note: Optional[str]) -> str:
    if summary:
        body = _html.escape(clamp_words(summary)).replace("\n", "<br>")
        return f'<div class="body">{body}</div>'
    msg = note or "Summary unavailable (no local LLM server reachable)."
    return f'<div class="note">{_html.escape(msg)}</div>'


def build_dashboard_html(
    *,
    title: str,
    language: Optional[str],
    speech_duration_s: Optional[float],
    word_count: int,
    summary: Optional[str],
    transcript: str,
    translation: str,
    translation_label: str = "English",
    summary_note: Optional[str] = None,
    cjk: bool = False,
) -> str:
    lang_disp = language_name(language)
    lang_code = f" ({language})" if language else ""
    cjk_cls = " cjk" if cjk else ""
    t_esc = _html.escape(transcript)
    x_esc = _html.escape(translation)
    tr_tab = _html.escape(translation_label)
    return (
        f"<title>{_html.escape(title)}</title>\n<style>{_CSS}</style>\n"
        '<div class="wrap">\n'
        '  <p class="kicker">plaude-local · speech-to-text</p>\n'
        f"  <h1>{_html.escape(title)}</h1>\n"
        '  <div class="stats">\n'
        f'    <div class="stat"><p class="label">Language detected</p><div class="value">{_html.escape(lang_disp)}{_html.escape(lang_code)}</div></div>\n'
        f'    <div class="stat"><p class="label">Speech duration</p><div class="value">{_html.escape(format_duration(speech_duration_s))}</div></div>\n'
        f'    <div class="stat"><p class="label">Words</p><div class="value">{word_count:,}</div></div>\n'
        "  </div>\n"
        '  <section class="summary"><h2>Critical topics</h2>'
        f"{_summary_html(summary, summary_note)}</section>\n"
        '  <div class="tabs" role="tablist" aria-label="Views">\n'
        '    <button class="tab" role="tab" id="tab-transcribed" aria-controls="panel-transcribed" aria-selected="true">Transcribed</button>\n'
        f'    <button class="tab" role="tab" id="tab-translation" aria-controls="panel-translation" aria-selected="false">{tr_tab}</button>\n'
        '    <button class="tab" role="tab" id="tab-sbs" aria-controls="panel-sbs" aria-selected="false">Side-by-Side</button>\n'
        "  </div>\n"
        f'  <section class="panel" id="panel-transcribed" role="tabpanel" aria-labelledby="tab-transcribed"><div class="doc{cjk_cls}"><pre>{t_esc}</pre></div></section>\n'
        f'  <section class="panel" id="panel-translation" role="tabpanel" aria-labelledby="tab-translation" hidden><div class="doc"><pre>{x_esc}</pre></div></section>\n'
        '  <section class="panel" id="panel-sbs" role="tabpanel" aria-labelledby="tab-sbs" hidden><div class="sbs">'
        f'<div class="col"><h3>Transcribed · {_html.escape(lang_disp)}</h3><div class="doc{cjk_cls}"><pre>{t_esc}</pre></div></div>'
        f'<div class="col"><h3>{tr_tab}</h3><div class="doc"><pre>{x_esc}</pre></div></div>'
        "</div></section>\n"
        f"  <footer>plaude-local · {word_count:,} words · {_html.escape(format_duration(speech_duration_s))}</footer>\n"
        f"</div>\n<script>{_JS}</script>\n"
    )
