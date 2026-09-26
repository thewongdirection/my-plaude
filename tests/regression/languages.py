"""Languages for the regression corpus.

``ALL_LANGUAGES`` lists every language that both Whisper (large-v3) and the
edge-tts neural voices support (73). The corpus itself focuses on
``CORPUS_CODES``: 16 Asian languages plus a few European ones (21 in total);
``LANGUAGES`` is that subset, in corpus order. To widen the corpus, add codes
to ``CORPUS_CODES`` and rerun ``generate_corpus.py``.

Fields:

* ``code``    - the Whisper language code the tool is expected to detect.
* ``name``    - the name given to the translation model (script noted where a
                language has several).
* ``locales`` - edge-tts locales whose voices may speak this language.
* ``script``  - Unicode script the translated text must be written in (a
                sanity check on the generated translations).
* ``tier``    - how well Whisper handles the language, which sets the
                absolute pass thresholds: ``A`` well supported, ``B`` usable,
                ``C`` low-resource (only exit code + baseline are enforced).
* ``alt``     - other detected codes accepted for closely related languages
                Whisper is known to confuse (e.g. Malay/Indonesian).
"""

ALL_LANGUAGES = [
    {"code": "af", "name": "Afrikaans", "locales": ["af-ZA"], "script": "LATIN", "tier": "B"},
    {"code": "am", "name": "Amharic", "locales": ["am-ET"], "script": "ETHIOPIC", "tier": "C"},
    {"code": "ar", "name": "Arabic (Modern Standard)", "locales": ["ar-SA", "ar-EG", "ar-AE", "ar-JO"], "script": "ARABIC", "tier": "A"},
    {"code": "az", "name": "Azerbaijani (Latin script)", "locales": ["az-AZ"], "script": "LATIN", "tier": "B", "alt": ["tr"]},
    {"code": "bg", "name": "Bulgarian", "locales": ["bg-BG"], "script": "CYRILLIC", "tier": "A", "alt": ["mk", "ru"]},
    {"code": "bn", "name": "Bengali", "locales": ["bn-BD", "bn-IN"], "script": "BENGALI", "tier": "B", "alt": ["as"]},
    {"code": "bs", "name": "Bosnian (Latin script)", "locales": ["bs-BA"], "script": "LATIN", "tier": "B", "alt": ["hr", "sr", "sl"]},
    {"code": "ca", "name": "Catalan", "locales": ["ca-ES"], "script": "LATIN", "tier": "A", "alt": ["es"]},
    {"code": "cs", "name": "Czech", "locales": ["cs-CZ"], "script": "LATIN", "tier": "A", "alt": ["sk"]},
    {"code": "cy", "name": "Welsh", "locales": ["cy-GB"], "script": "LATIN", "tier": "B"},
    {"code": "da", "name": "Danish", "locales": ["da-DK"], "script": "LATIN", "tier": "A", "alt": ["no", "nn", "sv"]},
    {"code": "de", "name": "German", "locales": ["de-DE", "de-AT", "de-CH"], "script": "LATIN", "tier": "A"},
    {"code": "el", "name": "Greek", "locales": ["el-GR"], "script": "GREEK", "tier": "A"},
    {"code": "en", "name": "English", "locales": ["en-US", "en-GB", "en-AU", "en-IN", "en-CA"], "script": "LATIN", "tier": "A"},
    {"code": "es", "name": "Spanish", "locales": ["es-ES", "es-MX", "es-AR", "es-CO", "es-US"], "script": "LATIN", "tier": "A", "alt": ["gl", "ca"]},
    {"code": "et", "name": "Estonian", "locales": ["et-EE"], "script": "LATIN", "tier": "B", "alt": ["fi"]},
    {"code": "fa", "name": "Persian", "locales": ["fa-IR"], "script": "ARABIC", "tier": "B"},
    {"code": "fi", "name": "Finnish", "locales": ["fi-FI"], "script": "LATIN", "tier": "A"},
    {"code": "tl", "name": "Filipino (Tagalog)", "locales": ["fil-PH"], "script": "LATIN", "tier": "B"},
    {"code": "fr", "name": "French", "locales": ["fr-FR", "fr-CA", "fr-BE", "fr-CH"], "script": "LATIN", "tier": "A"},
    {"code": "gl", "name": "Galician", "locales": ["gl-ES"], "script": "LATIN", "tier": "B", "alt": ["pt", "es"]},
    {"code": "gu", "name": "Gujarati", "locales": ["gu-IN"], "script": "GUJARATI", "tier": "C"},
    {"code": "he", "name": "Hebrew", "locales": ["he-IL"], "script": "HEBREW", "tier": "A"},
    {"code": "hi", "name": "Hindi", "locales": ["hi-IN"], "script": "DEVANAGARI", "tier": "A", "alt": ["ur", "mr", "ne"]},
    {"code": "hr", "name": "Croatian", "locales": ["hr-HR"], "script": "LATIN", "tier": "A", "alt": ["bs", "sr", "sl"]},
    {"code": "hu", "name": "Hungarian", "locales": ["hu-HU"], "script": "LATIN", "tier": "A"},
    {"code": "id", "name": "Indonesian", "locales": ["id-ID"], "script": "LATIN", "tier": "A", "alt": ["ms", "jw", "su"]},
    {"code": "is", "name": "Icelandic", "locales": ["is-IS"], "script": "LATIN", "tier": "B", "alt": ["fo", "no"]},
    {"code": "it", "name": "Italian", "locales": ["it-IT"], "script": "LATIN", "tier": "A"},
    {"code": "ja", "name": "Japanese", "locales": ["ja-JP"], "script": "CJK", "tier": "A"},
    {"code": "jw", "name": "Javanese (Latin script)", "locales": ["jv-ID"], "script": "LATIN", "tier": "C", "alt": ["id", "su", "ms"]},
    {"code": "ka", "name": "Georgian", "locales": ["ka-GE"], "script": "GEORGIAN", "tier": "C"},
    {"code": "kk", "name": "Kazakh (Cyrillic script)", "locales": ["kk-KZ"], "script": "CYRILLIC", "tier": "B", "alt": ["ru", "ky"]},
    {"code": "km", "name": "Khmer", "locales": ["km-KH"], "script": "KHMER", "tier": "C"},
    {"code": "kn", "name": "Kannada", "locales": ["kn-IN"], "script": "KANNADA", "tier": "C"},
    {"code": "ko", "name": "Korean", "locales": ["ko-KR"], "script": "HANGUL", "tier": "A"},
    {"code": "lo", "name": "Lao", "locales": ["lo-LA"], "script": "LAO", "tier": "C", "alt": ["th"]},
    {"code": "lt", "name": "Lithuanian", "locales": ["lt-LT"], "script": "LATIN", "tier": "B", "alt": ["lv"]},
    {"code": "lv", "name": "Latvian", "locales": ["lv-LV"], "script": "LATIN", "tier": "B", "alt": ["lt"]},
    {"code": "mk", "name": "Macedonian", "locales": ["mk-MK"], "script": "CYRILLIC", "tier": "B", "alt": ["bg", "sr"]},
    {"code": "ml", "name": "Malayalam", "locales": ["ml-IN"], "script": "MALAYALAM", "tier": "C"},
    {"code": "mn", "name": "Mongolian (Cyrillic script)", "locales": ["mn-MN"], "script": "CYRILLIC", "tier": "C", "alt": ["ru"]},
    {"code": "mr", "name": "Marathi", "locales": ["mr-IN"], "script": "DEVANAGARI", "tier": "B", "alt": ["hi", "ne"]},
    {"code": "ms", "name": "Malay", "locales": ["ms-MY"], "script": "LATIN", "tier": "A", "alt": ["id", "jw", "su"]},
    {"code": "mt", "name": "Maltese", "locales": ["mt-MT"], "script": "LATIN", "tier": "C", "alt": ["it", "ar"]},
    {"code": "my", "name": "Burmese", "locales": ["my-MM"], "script": "MYANMAR", "tier": "C"},
    {"code": "no", "name": "Norwegian (Bokmal)", "locales": ["nb-NO"], "script": "LATIN", "tier": "A", "alt": ["nn", "da", "sv"]},
    {"code": "ne", "name": "Nepali", "locales": ["ne-NP"], "script": "DEVANAGARI", "tier": "C", "alt": ["hi", "mr"]},
    {"code": "nl", "name": "Dutch", "locales": ["nl-NL", "nl-BE"], "script": "LATIN", "tier": "A", "alt": ["af"]},
    {"code": "pl", "name": "Polish", "locales": ["pl-PL"], "script": "LATIN", "tier": "A"},
    {"code": "ps", "name": "Pashto", "locales": ["ps-AF"], "script": "ARABIC", "tier": "C", "alt": ["fa", "ur", "ar"]},
    {"code": "pt", "name": "Portuguese", "locales": ["pt-BR", "pt-PT"], "script": "LATIN", "tier": "A", "alt": ["gl", "es"]},
    {"code": "ro", "name": "Romanian", "locales": ["ro-RO"], "script": "LATIN", "tier": "A"},
    {"code": "ru", "name": "Russian", "locales": ["ru-RU"], "script": "CYRILLIC", "tier": "A", "alt": ["uk", "be"]},
    {"code": "si", "name": "Sinhala", "locales": ["si-LK"], "script": "SINHALA", "tier": "C"},
    {"code": "sk", "name": "Slovak", "locales": ["sk-SK"], "script": "LATIN", "tier": "A", "alt": ["cs"]},
    {"code": "sl", "name": "Slovenian", "locales": ["sl-SI"], "script": "LATIN", "tier": "B", "alt": ["hr", "bs", "sr"]},
    {"code": "so", "name": "Somali", "locales": ["so-SO"], "script": "LATIN", "tier": "C"},
    {"code": "sq", "name": "Albanian", "locales": ["sq-AL"], "script": "LATIN", "tier": "C"},
    {"code": "sr", "name": "Serbian (Cyrillic script)", "locales": ["sr-RS"], "script": "CYRILLIC", "tier": "B", "alt": ["hr", "bs", "mk", "ru"]},
    {"code": "su", "name": "Sundanese", "locales": ["su-ID"], "script": "LATIN", "tier": "C", "alt": ["id", "jw", "ms"]},
    {"code": "sv", "name": "Swedish", "locales": ["sv-SE"], "script": "LATIN", "tier": "A", "alt": ["no", "da"]},
    {"code": "sw", "name": "Swahili", "locales": ["sw-KE", "sw-TZ"], "script": "LATIN", "tier": "B"},
    {"code": "ta", "name": "Tamil", "locales": ["ta-IN", "ta-LK", "ta-MY", "ta-SG"], "script": "TAMIL", "tier": "B"},
    {"code": "te", "name": "Telugu", "locales": ["te-IN"], "script": "TELUGU", "tier": "C"},
    {"code": "th", "name": "Thai", "locales": ["th-TH"], "script": "THAI", "tier": "A", "alt": ["lo"]},
    {"code": "tr", "name": "Turkish", "locales": ["tr-TR"], "script": "LATIN", "tier": "A", "alt": ["az"]},
    {"code": "uk", "name": "Ukrainian", "locales": ["uk-UA"], "script": "CYRILLIC", "tier": "A", "alt": ["ru", "be"]},
    {"code": "ur", "name": "Urdu", "locales": ["ur-PK", "ur-IN"], "script": "ARABIC", "tier": "B", "alt": ["hi"]},
    {"code": "uz", "name": "Uzbek (Latin script)", "locales": ["uz-UZ"], "script": "LATIN", "tier": "C", "alt": ["tr", "az"]},
    {"code": "vi", "name": "Vietnamese", "locales": ["vi-VN"], "script": "LATIN", "tier": "A"},
    {"code": "zh", "name": "Chinese (Simplified, Mandarin)", "locales": ["zh-CN", "zh-TW"], "script": "CJK", "tier": "A", "alt": ["yue"]},
    {"code": "yue", "name": "Cantonese (Traditional Chinese characters, spoken Cantonese)", "locales": ["zh-HK"], "script": "CJK", "tier": "B", "alt": ["zh"]},
]

BY_CODE = {lang["code"]: lang for lang in ALL_LANGUAGES}

# The corpus: Asian languages first (East, Southeast, South Asia - including a
# few low-resource ones), then a handful of European languages.
CORPUS_CODES = [
    "zh", "yue", "ja", "ko",                          # East Asia
    "vi", "th", "id", "ms", "tl", "km", "my",         # Southeast Asia
    "hi", "bn", "ta", "te", "ur",                     # South Asia
    "en", "es", "fr", "de", "ru",                     # Europe
]
LANGUAGES = [BY_CODE[code] for code in CORPUS_CODES]

# Scripts written without spaces between words: lines are joined with no
# separator and scored on characters only.
NO_SPACE_SCRIPTS = {"zh", "yue", "ja", "th", "lo", "km", "my"}

# Absolute pass thresholds per tier and variant (None = not enforced; the
# baseline still catches regressions). Language detection is enforced for
# tiers A and B only.
THRESHOLDS = {
    "A": {"clean": {"max_cer": 0.30, "min_recall": 0.50},
          "damaged": {"max_cer": 0.50, "min_recall": 0.35}},
    "B": {"clean": {"max_cer": 0.60, "min_recall": 0.30},
          "damaged": {"max_cer": 0.80, "min_recall": 0.20}},
    "C": {"clean": {"max_cer": None, "min_recall": None},
          "damaged": {"max_cer": None, "min_recall": None}},
}


def accepted_codes(code: str) -> set:
    """Detected language codes that count as correct for ``code``."""
    return {code, *BY_CODE.get(code, {}).get("alt", [])}
