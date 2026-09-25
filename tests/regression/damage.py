"""Seeded "bad recording" recipes for the damaged half of the corpus (pure, unit-tested).

Each damaged file combines several imperfections that the tool's features are
meant to handle:

* background noise / static (pink, white or brown)   -> denoise
* mains hum at 50 or 60 Hz with harmonics (half)     -> --dehum auto (detection is checked)
* clicks / crackle (half)                            -> --declick
* clipping from a too-hot input, or a very quiet one -> --declip / --enhance
* muffled microphone or telephone band (most)        -> --enhance strong
* room echo (some)                                   -> (robustness only)

``damage_plan`` draws a recipe from an RNG; ``damage_filtergraph`` turns it
into an FFmpeg ``-filter_complex`` graph over input ``[0:a]``.
"""

from __future__ import annotations

import random
from typing import Optional

SAMPLE_RATE = 22050


def damage_plan(rng: random.Random) -> dict:
    """Draw a reproducible combination of imperfections."""
    plan: dict = {
        "noise": {"color": rng.choice(["pink", "white", "brown"]),
                  "amplitude": round(rng.uniform(0.02, 0.06), 3),
                  "seed": rng.randrange(1, 2**31)},
        "hum_hz": None,
        "hum_amplitude": None,
        "clicks_density": None,
        "level": rng.choice(["clipped", "quiet", "normal"]),
        "level_db": 0.0,
        "tone": rng.choice(["muffled", "telephone", "muffled", "none"]),
        "lowpass_hz": None,
        "reverb": rng.random() < 0.35,
    }
    if rng.random() < 0.5:
        plan["hum_hz"] = rng.choice([50, 60])
        plan["hum_amplitude"] = round(rng.uniform(0.04, 0.09), 3)
    if rng.random() < 0.5:
        plan["clicks_density"] = round(rng.uniform(0.0002, 0.0008), 5)
    if plan["level"] == "clipped":
        plan["level_db"] = round(rng.uniform(8.0, 13.0), 1)
    elif plan["level"] == "quiet":
        plan["level_db"] = -round(rng.uniform(16.0, 22.0), 1)
    if plan["tone"] == "muffled":
        plan["lowpass_hz"] = rng.randrange(1800, 3001, 100)
    return plan


def damage_filtergraph(plan: dict, sr: int = SAMPLE_RATE) -> str:
    """FFmpeg filter graph applying ``plan`` to ``[0:a]``; output label ``[out]``."""
    voice = [f"aresample={sr}", "aformat=channel_layouts=mono"]
    if plan["tone"] == "muffled":
        voice.append(f"lowpass=f={plan['lowpass_hz']}")
    elif plan["tone"] == "telephone":
        voice += ["highpass=f=300", "lowpass=f=3400"]
    if plan["reverb"]:
        voice.append("aecho=0.8:0.6:50|110:0.3|0.2")
    parts = ["[0:a]" + ",".join(voice) + "[v]"]
    mix = ["[v]"]

    n = plan["noise"]
    parts.append(f"anoisesrc=color={n['color']}:amplitude={n['amplitude']}:"
                 f"sample_rate={sr}:seed={n['seed']},aformat=channel_layouts=mono[n]")
    mix.append("[n]")

    if plan["hum_hz"]:
        f, a = plan["hum_hz"], plan["hum_amplitude"]
        parts.append(f"aevalsrc='{a}*sin(2*PI*{f}*t)+{a / 2:.4f}*sin(2*PI*{2 * f}*t)"
                     f"+{a / 3:.4f}*sin(2*PI*{3 * f}*t)':s={sr}[h]")
        mix.append("[h]")
    if plan["clicks_density"]:
        parts.append(f"aevalsrc='if(lt(random(0),{plan['clicks_density']}),"
                     f"0.9*(2*random(1)-1),0)':s={sr}[c]")
        mix.append("[c]")

    tail = f"amix=inputs={len(mix)}:duration=first:normalize=0"
    if plan["level_db"]:
        # Clipping happens when the boosted float mix is written as 16-bit PCM.
        tail += f",volume={plan['level_db']}dB"
    parts.append("".join(mix) + tail + "[out]")
    return ";".join(parts)


def expected_hum(plan: dict) -> Optional[int]:
    """The hum frequency ``--dehum auto`` should detect (None if none added)."""
    return plan.get("hum_hz")
