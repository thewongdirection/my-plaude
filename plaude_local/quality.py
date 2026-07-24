"""Recording quality assessment.

Flags "bad" recordings - no speech, near-silence, or noise that produces
low-confidence / hallucinated transcripts - using signals that are essentially
free:

* From the transcript (faster-whisper reports these per segment):
  - ``no_speech_prob``      high     -> no speech
  - ``avg_logprob``         low      -> uncertain (garbled/noisy)
  - ``compression_ratio``   high     -> repetitive/looping (noise hallucination)
  - speech coverage (segment time / duration) low -> mostly silence/noise

* From the audio (FFmpeg ``volumedetect`` / ``silencedetect``), used by the
  fast ``--assess-only`` triage that runs *without* transcription:
  - ``mean_volume_db`` very low     -> near-silent
  - ``silence_ratio``  very high    -> mostly silence

Everything here is pure/standard-library; the caller supplies the metrics.
Verdicts are heuristics, so the CLI default is to *warn*, never to silently
drop a transcript.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

VERDICT_OK = "ok"
VERDICT_SUSPECT = "suspect"
VERDICT_BAD = "bad"


@dataclass
class Thresholds:
    min_speech: float = 0.15          # min speech coverage (fraction) before "suspect"
    max_compression: float = 2.4      # above this -> repetitive/hallucinated
    min_logprob: float = -1.0         # below this -> low confidence
    max_no_speech: float = 0.6        # above this -> likely no speech
    near_silent_db: float = -50.0     # mean volume below this -> near-silent
    mostly_silence: float = 0.85      # silence ratio above this -> suspect


@dataclass
class QualityReport:
    verdict: str
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_bad(self) -> bool:
        return self.verdict == VERDICT_BAD

    def summary(self) -> str:
        head = f"quality: {self.verdict.upper()}"
        if self.reasons:
            return f"{head} - {'; '.join(self.reasons)}"
        return head

    def as_dict(self) -> Dict[str, Any]:
        return {"verdict": self.verdict, "reasons": self.reasons,
                "metrics": self.metrics}


def _mean(values: Sequence[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def assess(
    *,
    segments: Optional[List[Dict[str, Any]]] = None,
    duration: Optional[float] = None,
    audio_stats: Optional[Dict[str, Any]] = None,
    thresholds: Optional[Thresholds] = None,
) -> QualityReport:
    """Assess recording quality from transcript segments and/or audio stats.

    Provide ``segments`` (+ ``duration``) for the full, transcript-based
    assessment, or ``audio_stats`` alone for the model-free ``--assess-only``
    triage. Either or both may be given.
    """
    t = thresholds or Thresholds()
    reasons: List[str] = []
    metrics: Dict[str, Any] = {}
    bad = False
    suspect = False

    # ---- transcript-based signals -----------------------------------------
    if segments is not None:
        text = "".join(s.get("text", "") for s in segments).strip()
        speech_time = sum(max(0.0, float(s.get("end", 0)) - float(s.get("start", 0)))
                          for s in segments)
        coverage = (speech_time / duration) if (duration and duration > 0) else None
        avg_no_speech = _mean([s.get("no_speech_prob") for s in segments])
        avg_logprob = _mean([s.get("avg_logprob") for s in segments])
        comps = [s.get("compression_ratio") for s in segments
                 if s.get("compression_ratio") is not None]
        max_comp = max(comps) if comps else None

        metrics.update({
            "num_segments": len(segments),
            "char_count": len(text),
            "speech_coverage": round(coverage, 4) if coverage is not None else None,
            "avg_no_speech_prob": round(avg_no_speech, 4) if avg_no_speech is not None else None,
            "avg_logprob": round(avg_logprob, 4) if avg_logprob is not None else None,
            "max_compression_ratio": round(max_comp, 4) if max_comp is not None else None,
        })

        if not text or len(segments) == 0:
            bad = True
            reasons.append("no speech detected (empty transcript)")
        else:
            strong_no_speech = (avg_no_speech is not None
                                and avg_no_speech >= t.max_no_speech)
            almost_no_coverage = coverage is not None and coverage < 0.02
            if almost_no_coverage or (strong_no_speech and coverage is not None
                                      and coverage < t.min_speech):
                bad = True
                reasons.append(
                    "little or no speech "
                    + (f"(coverage {coverage:.0%}"
                       if coverage is not None else "(unknown coverage")
                    + (f", no-speech {avg_no_speech:.2f})" if avg_no_speech is not None else ")"))
            else:
                if coverage is not None and coverage < t.min_speech:
                    suspect = True
                    reasons.append(f"low speech coverage ({coverage:.0%})")
                if strong_no_speech:
                    suspect = True
                    reasons.append(f"high no-speech probability ({avg_no_speech:.2f})")
            if avg_logprob is not None and avg_logprob < t.min_logprob:
                suspect = True
                reasons.append(f"low transcription confidence (avg_logprob {avg_logprob:.2f})")
            if max_comp is not None and max_comp > t.max_compression:
                suspect = True
                reasons.append(
                    f"repetitive output (compression ratio {max_comp:.2f}) - possible noise")

    # ---- audio-level signals (triage / supplementary) ---------------------
    if audio_stats:
        mean_db = audio_stats.get("mean_volume_db")
        silence_ratio = audio_stats.get("silence_ratio")
        metrics.update({
            "mean_volume_db": mean_db,
            "max_volume_db": audio_stats.get("max_volume_db"),
            "silence_ratio": round(silence_ratio, 4) if silence_ratio is not None else None,
        })
        if mean_db is not None and mean_db <= t.near_silent_db:
            bad = True
            reasons.append(f"near-silent audio (mean volume {mean_db:.0f} dB)")
        if silence_ratio is not None:
            if silence_ratio >= 0.98:
                bad = True
                reasons.append(f"almost entirely silence ({silence_ratio:.0%})")
            elif silence_ratio >= t.mostly_silence:
                suspect = True
                reasons.append(f"mostly silence ({silence_ratio:.0%})")

    if bad:
        verdict = VERDICT_BAD
    elif suspect:
        verdict = VERDICT_SUSPECT
    else:
        verdict = VERDICT_OK
    return QualityReport(verdict=verdict, reasons=reasons, metrics=metrics)
