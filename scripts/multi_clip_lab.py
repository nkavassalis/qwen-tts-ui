#!/usr/bin/env python3
# coding=utf-8
"""Multi-clip clone lab: is a clone built from several clips better than your one clip?

    Stop the web UI first (one model owner at a time), then:
    .venv/bin/python scripts/multi_clip_lab.py                  # web UI on 127.0.0.1:7861
    .venv/bin/python scripts/multi_clip_lab.py --clips DIR      # pre-load DIR/*.wav (+ .txt)
    .venv/bin/python scripts/multi_clip_lab.py --clips DIR --single NAME --render-only
    .venv/bin/python scripts/multi_clip_lab.py --listen-only    # A/B an old run, no GPU

An experiment, not a feature: nothing here touches saved voices or the main UI.

**The question, as an A/B.** You pick the one clip you would use in the Clone tab today.
Every other version is compared against a take from *that clip*, on the same sentence and
the same seed, one pair at a time: you hear the real person (a clip neither take was built
from), then A and B in random order, and say which sounds more like them — or that you
can't tell. That forced choice is what ears are good at; scoring six takes 1–5 at once is
not (the scale drifts, and it never asks "better than what I have?").

**The versions.** Qwen3-TTS conditions on ONE reference: one clip's codec codes (+ its
transcript) and one speaker embedding. Several clips can only help through those slots:

  single       your clip, as the Clone tab would use it            (the baseline)
  single_mean  your clip's codes and transcript, with the embedding swapped for the
               average of all the clips — only the embedding differs from `single`, so
               this pair isolates the effect of averaging
  concat       all the clips joined into one longer reference (≤20s)
  cleanest     the clip that does best on the reference checks (noise, coverage, edges,
               clipping, transcript fit) — "would a better single clip have done it?"
  xvec_mean    the averaged embedding alone (x-vector mode: no codes, no transcript)

When `cleanest` *is* your clip, its pair plays the same take twice. Those pairs are kept as a
check on the listener: the honest answer is "can't tell", and how often you pick a side
anyway is shown next to the results.

**Scoring.** Each round holds one clip out (never your clip); the takes are built from the
rest. Results give, per version: how often you preferred it over your clip, a sign test on
that (so 7 of 10 is not mistaken for a finding), and the same paired comparison on SIM
against the held-out clip. SIM against the average of the build clips is *not* used: the
encoder that grades it made the average, so it favours the averaging versions by design.

Output: tests/output/multi_clip/<run>/ (gitignored) — run.json, ab.json, wavs.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "webui"))

import numpy as np
import soundfile as sf

import app as webui

OUT_ROOT = os.path.join(ROOT, "tests", "output", "multi_clip")
SR = webui.REF_SAMPLE_RATE
RUN_FORMAT = 2                # run.json layout; older runs are refused, not misread

ARMS: List[Tuple[str, str]] = [
    ("single", "Your clip (baseline)"),
    ("single_mean", "Your clip + averaged voice fingerprint"),
    ("concat", "All clips joined"),
    ("cleanest", "Cleanest clip"),
    ("xvec_mean", "Averaged fingerprint only"),
]
ARM_KEYS = [k for k, _ in ARMS]
ARM_LABELS = dict(ARMS)
BASELINE = "single"
CHALLENGERS = [k for k in ARM_KEYS if k != BASELINE]
AUTO_SINGLE = "(most typical clip)"
MIN_CLIPS = 3                 # one held out + at least two to build from
MIN_DECIDED = 10              # fewer decided pairs than this: no verdict either way
SIGNIFICANCE = 0.05
CONCAT_MAX_SECONDS = 20.0     # past REF_MAX_SECONDS on purpose: length is what this arm tests
CONCAT_GAP_SECONDS = 0.3
LISTEN_RMS_DB = -23.0
MAX_LONG_CLIPS = 24

DEFAULT_SENTENCES = (
    "I left the keys on the kitchen table, right next to the letter from your aunt.\n"
    "Honestly, I did not expect the meeting to run that long, but here we are.\n"
    "Could you call me back when you get a minute? It is nothing urgent."
)


# ---------------------------------------------------------------------------
# Pure helpers (CPU only; tested in tests/test_app_logic.py)

def make_folds(n: int, k: int, keep: Optional[int] = None) -> List[Tuple[int, List[int]]]:
    """Rounds of (held-out index, build indices): at most `k`, spread evenly, and the clip
    `keep` (your baseline clip) is never the one held out."""
    if n < MIN_CLIPS:
        raise ValueError(f"Need at least {MIN_CLIPS} clips (one held out, two to build from); "
                         f"got {n}.")
    if keep is not None and not 0 <= keep < n:
        raise ValueError(f"baseline clip index {keep} out of range")
    candidates = [i for i in range(n) if i != keep]
    k = max(1, min(int(k), len(candidates)))
    picks = sorted({int(round(j * len(candidates) / float(k))) % len(candidates)
                    for j in range(k)})
    return [(candidates[p], [i for i in range(n) if i != candidates[p]]) for p in picks]


def _unit_rows(embs: Sequence[np.ndarray]) -> np.ndarray:
    e = np.stack([np.asarray(x, dtype=np.float64).reshape(-1) for x in embs])
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)


def mean_embedding(embs: Sequence[np.ndarray]) -> np.ndarray:
    """Average direction, rescaled to the clips' mean norm.

    Averaging unit vectors (not raw ones) keeps one loud-norm clip from dominating; the
    rescale keeps the result in the range the model was trained on — a plain mean of
    vectors that disagree is shorter than any of them.
    """
    if not len(embs):
        raise ValueError("mean_embedding needs at least one embedding")
    raw = np.stack([np.asarray(x, dtype=np.float64).reshape(-1) for x in embs])
    direction = _unit_rows(embs).mean(axis=0)
    direction /= np.linalg.norm(direction) + 1e-12
    return (direction * float(np.linalg.norm(raw, axis=1).mean())).astype(np.float32)


def medoid_index(embs: Sequence[np.ndarray]) -> int:
    """Index of the embedding closest (cosine) to the set's mean direction."""
    u = _unit_rows(embs)
    centre = u.mean(axis=0)
    return int(np.argmax(u @ (centre / (np.linalg.norm(centre) + 1e-12))))


def reference_quality(wav: np.ndarray, sr: int) -> Dict[str, Any]:
    """The Clone tab's reference checks as numbers: SNR, speech coverage, edges, clipping."""
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    clipped = float(np.mean(np.abs(wav) >= 0.999)) if wav.size else 0.0
    db, _ = webui._frame_levels(wav, sr)
    active, floor, level = webui._activity(db)
    if active is None:
        return {"snr": 0.0, "coverage": 0.0, "edges_ok": False, "clipped": clipped}
    edge = max(1, int(0.06 / webui.FRAME_SECONDS))
    return {"snr": round(float(level - floor), 2), "coverage": round(float(np.mean(active)), 3),
            "edges_ok": bool(not active[:edge].all() and not active[-edge:].all()),
            "clipped": clipped}


def clean_score(quality: Dict[str, Any], text_confidence: Optional[float] = None) -> float:
    """Higher is cleaner. Weighted in the order the checks were measured to cost:
    a transcript that does not fit, then noise, then clipping, coverage and edges.
    SNR is capped at 40dB so a gated-to-digital-silence floor does not run away with it."""
    score = min(float(quality.get("snr", 0.0)), 40.0)
    score += 10.0 * min(float(quality.get("coverage", 0.0)), 0.8)
    if not quality.get("edges_ok"):
        score -= 3.0
    if float(quality.get("clipped", 0.0)) > webui.CLIP_FRACTION:
        score -= 10.0
    if text_confidence is not None and text_confidence < webui.REF_TEXT_CONF_WARN:
        score -= 15.0
    return round(score, 3)


def screen_clips(embs: Sequence[np.ndarray]) -> List[Dict[str, Any]]:
    """Each clip's mean similarity to the others; flag ones that sit well apart.

    An outlier is a clip that is likely a different mic, room, mood — or person. The
    margin is the larger of 0.05 and 3x the median absolute deviation, so a tight set does
    not flag its least-typical member just for being last.
    """
    u = _unit_rows(embs)
    n = len(u)
    sims = u @ u.T
    typical = [(float(sims[i].sum()) - 1.0) / max(1, n - 1) for i in range(n)]
    med = float(np.median(typical))
    mad = float(np.median(np.abs(np.asarray(typical) - med)))
    margin = max(0.05, 3.0 * mad)
    return [{"typicality": round(t, 4), "outlier": bool(n >= 3 and t < med - margin)}
            for t in typical]


def concat_clips(wavs: Sequence[np.ndarray], texts: Sequence[str], sr: int,
                 max_seconds: float = CONCAT_MAX_SECONDS,
                 gap: float = CONCAT_GAP_SECONDS) -> Tuple[np.ndarray, str, int]:
    """Join clips (with a short silence) up to `max_seconds`; always keeps the first."""
    pieces: List[np.ndarray] = []
    words: List[str] = []
    silence = np.zeros(int(gap * sr), dtype=np.float32)
    total = 0
    for wav, text in zip(wavs, texts):
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        extra = wav.size + (silence.size if pieces else 0)
        if pieces and (total + extra) / float(sr) > max_seconds:
            break
        if pieces:
            pieces.append(silence)
        pieces.append(wav)
        words.append((text or "").strip())
        total += extra
    return np.concatenate(pieces), " ".join(w for w in words if w), len(words)


def high_band_milli(wav: np.ndarray, sr: int, cutoff: float = 9000.0) -> Optional[float]:
    """Share of spectral energy above `cutoff`, in thousandths (the docs' "2.67m")."""
    wav = np.asarray(wav, dtype=np.float64).reshape(-1)
    if wav.size < 32 or sr / 2.0 <= cutoff:
        return None
    power = np.abs(np.fft.rfft(wav)) ** 2
    freqs = np.fft.rfftfreq(wav.size, 1.0 / sr)
    total = float(power.sum())
    return None if total <= 0 else round(1000.0 * float(power[freqs >= cutoff].sum()) / total, 3)


def loudness_match(wav: np.ndarray, rms_db: float = LISTEN_RMS_DB, peak: float = 0.95) -> np.ndarray:
    """Same RMS for everything a listener compares, peak-limited by scaling (no clipping),
    so "louder" cannot pass for "more like them"."""
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    rms = float(np.sqrt(np.mean(np.square(wav)))) if wav.size else 0.0
    if rms <= 1e-6:
        return wav
    gain = (10.0 ** (rms_db / 20.0)) / rms
    top = float(np.max(np.abs(wav))) * gain
    if top > peak:
        gain *= peak / top
    return (wav * gain).astype(np.float32)


def build_trials(folds: int, sentences: int, seeds: Sequence[int]) -> List[Dict[str, Any]]:
    """One trial per (round, sentence, seed): every version renders it with that seed."""
    return [{"id": f"f{f}_s{s}_r{seed}", "fold": f, "sentence": s, "seed": int(seed)}
            for f in range(folds) for s in range(sentences) for seed in seeds]


def build_pairs(trials: Sequence[Dict[str, Any]], challengers: Sequence[str],
                rng: random.Random) -> List[Dict[str, Any]]:
    """One A/B pair per (trial, challenger); your clip lands on A or B at random, and the
    pairs are shuffled so fatigue does not fall on one version, round or sentence."""
    pairs = [{"id": f"{t['id']}__{c}", "trial": t["id"], "challenger": c,
              "single_side": rng.choice("AB")} for t in trials for c in challengers]
    rng.shuffle(pairs)
    return pairs


def sign_test_p(wins: int, losses: int) -> float:
    """Two-sided exact sign test: how surprising is this split if neither side is better?"""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n)


def verdict(wins: int, losses: int) -> str:
    decided = wins + losses
    if decided < MIN_DECIDED:
        return f"too few to tell ({decided} decided, need {MIN_DECIDED})"
    p = sign_test_p(wins, losses)
    if p < SIGNIFICANCE:
        return f"**{'better' if wins > losses else 'worse'} than your clip** (p={p:.3f})"
    return f"no clear difference (p={p:.2f})"


def _stats(values: Sequence[Optional[float]]) -> Tuple[Optional[float], Optional[float], int]:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return None, None, 0
    return float(np.mean(xs)), (float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0), len(xs)


def summarise(run: Dict[str, Any], answers: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Per challenger: your A/B tally against your clip, and the paired SIM comparison."""
    takes = {(t["trial"], t["arm"]): t for t in run["takes"]}
    rows = []
    identical = {"pairs": 0, "cant_tell": 0}
    side_a = side_total = 0
    for pair in run["pairs"]:
        a = answers.get(pair["id"])
        if a and a.get("picked") in ("A", "B"):
            side_total += 1
            side_a += a["picked"] == "A"
        if a and pair.get("identical"):
            identical["pairs"] += 1
            identical["cant_tell"] += a["choice"] == "tie"
    for arm in CHALLENGERS:
        mine = [p for p in run["pairs"] if p["challenger"] == arm]
        if not mine:
            continue
        real = [p for p in mine if not p.get("identical")]
        got = [answers[p["id"]] for p in real if p["id"] in answers]
        wins = sum(a["choice"] == "challenger" for a in got)
        losses = sum(a["choice"] == "single" for a in got)
        ties = sum(a["choice"] == "tie" for a in got)
        sim_w = sim_l = 0
        deltas, wers = [], []
        for t in run["trials"]:
            c, s = takes.get((t["id"], arm)), takes.get((t["id"], BASELINE))
            if not c or not s or c.get("same_as") == BASELINE:
                continue
            wers.append(c.get("wer"))
            if c.get("sim_heldout") is not None and s.get("sim_heldout") is not None:
                d = float(c["sim_heldout"]) - float(s["sim_heldout"])
                deltas.append(d)
                sim_w += d > 1e-9
                sim_l += d < -1e-9
        rows.append({"arm": arm, "label": ARM_LABELS[arm], "pairs": len(real),
                     "answered": len(got), "wins": wins, "losses": losses, "ties": ties,
                     "verdict": verdict(wins, losses) if got else "not listened to yet",
                     "sim_delta": _stats(deltas), "sim_wl": (sim_w, sim_l),
                     "sim_p": sign_test_p(sim_w, sim_l), "wer": _stats(wers),
                     "identical": len(mine) - len(real)})
    base_wer = _stats([t.get("wer") for t in run["takes"] if t["arm"] == BASELINE])
    return {"rows": rows, "identical": identical, "base_wer": base_wer,
            "side_bias": (side_a, side_total), "answered": len(answers),
            "pairs": len(run["pairs"])}


def summary_markdown(s: Dict[str, Any]) -> str:
    def ms(stat, digits=3, signed=False):
        mean, sd, n = stat
        if mean is None:
            return "—"
        return f"{mean:+.{digits}f} ± {sd:.{digits}f}" if signed else f"{mean:.{digits}f}"

    lines = [f"**{s['answered']} of {s['pairs']}** pairs answered.", "",
             "| version | you: preferred it / your clip / can't tell | verdict | "
             "SIM vs held-out, minus your clip's | SIM wins/losses | WER |",
             "|---|---|---|---|---|---|"]
    for r in s["rows"]:
        lines.append(f"| **{r['label']}** (`{r['arm']}`) | {r['wins']} / {r['losses']} / "
                     f"{r['ties']} of {r['pairs']} | {r['verdict']} | "
                     f"{ms(r['sim_delta'], signed=True)} | {r['sim_wl'][0]}/{r['sim_wl'][1]} "
                     f"(p={r['sim_p']:.2f}) | {ms(r['wer'], 2)} |")
    lines.append(f"| *your clip* | | | | | {ms(s['base_wer'], 2)} |")
    ident = s["identical"]
    a, total = s["side_bias"]
    lines += [""]
    if ident["pairs"]:
        lines.append(f"**Check on the listener:** {ident['pairs']} pair(s) played the same take "
                     f"twice (the cleanest clip *was* your clip); you said *can't tell* on "
                     f"{ident['cant_tell']}. Picking a side often there means the other verdicts "
                     f"carry some guessing.")
    if total >= 10:
        lines.append(f"**Side bias:** you picked A {a} of {total} times "
                     f"(p={sign_test_p(a, total - a):.2f} against 50/50).")
    lines += ["", "How to read it: a version is worth building into the app only if *you* "
              "prefer it over your clip with a verdict of **better**, held-out SIM does not "
              "disagree, and its WER is no worse. \"No clear difference\" after 20+ pairs is a "
              "real answer too: averaging is not worth the complexity."]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Clips in

def read_transcript(path: str) -> str:
    stem = os.path.splitext(path)[0]
    for candidate in (stem + ".txt", path + ".txt"):
        if os.path.isfile(candidate):
            with open(candidate, encoding="utf-8") as handle:
                return handle.read().strip()
    return ""


def segment_long(wav: np.ndarray, sr: int, target: float = webui.REF_TARGET_SECONDS,
                 max_clips: int = MAX_LONG_CLIPS) -> List[Tuple[float, float]]:
    """Cut one long recording into non-overlapping ~`target`s windows with edges in pauses.

    Caveat, shown in the UI: clips from one session share a mic and room, so a held-out
    clip from the same file is a kinder test than a clip recorded another day.
    """
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    duration = len(wav) / float(sr)
    spans: List[Tuple[float, float]] = []
    position = 0.0
    while position + webui.REF_MIN_SECONDS <= duration and len(spans) < max_clips:
        start, end = webui.snap_to_pauses(wav, sr, position, min(duration, position + target))
        start = max(start, spans[-1][1] if spans else 0.0)
        if end - start < webui.REF_MIN_SECONDS:
            break
        spans.append((round(start, 3), round(end, 3)))
        position = end + 0.05
    return spans


def clips_from_paths(paths: Sequence[str], trusted: bool) -> List[Dict[str, Any]]:
    """`trusted` (a CLI directory) skips the browser-path confinement of load_media."""
    loader = webui._load_media if trusted else webui.load_media
    clips = []
    for path in paths:
        wav, sr = loader(path, SR)
        clips.append({"name": os.path.basename(path), "wav": np.asarray(wav, dtype=np.float32),
                      "sr": int(sr), "text": read_transcript(path) if trusted else ""})
    return clips


def clips_from_long(path: str, trusted: bool) -> List[Dict[str, Any]]:
    loader = webui._load_media if trusted else webui.load_media
    wav, sr = loader(path, SR)
    base = os.path.splitext(os.path.basename(path))[0]
    return [{"name": f"{base}@{a:.1f}-{b:.1f}s", "wav": wav[int(a * sr):int(b * sr)].copy(),
             "sr": int(sr), "text": ""} for a, b in segment_long(wav, sr)]


def unique_names(clips: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Two uploads called take.wav must stay two clips you can tell apart."""
    seen: Dict[str, int] = {}
    for clip in clips:
        name = clip["name"]
        if name in seen:
            seen[name] += 1
            clip["name"] = f"{name} ({seen[name]})"
        else:
            seen[name] = 1
    return clips


# ---------------------------------------------------------------------------
# The GPU part

class Lab:
    def __init__(self, hub: Optional[Any], asr: Optional[Any], out_root: str = OUT_ROOT):
        self.hub = hub
        self.asr = asr
        self.out_root = out_root

    # -- screening -----------------------------------------------------------
    def screen(self, clips: List[Dict[str, Any]], language: str,
               progress: Any = None) -> List[Dict[str, Any]]:
        """Fill missing transcripts, embed every clip, flag outliers and bad references."""
        if self.hub is None:
            raise RuntimeError("Started with --listen-only: no model to screen with.")
        code_language = None if language in (None, "", "Auto") else language

        def say(fraction: float, what: str) -> None:
            print(f"[lab] {what}", flush=True)
            if progress is not None:
                progress(fraction, desc=what)

        for i, clip in enumerate(clips):
            if not clip["text"].strip():
                if self.asr is None or not self.asr.enabled:
                    raise RuntimeError(f"{clip['name']} has no transcript and Whisper is off; "
                                       f"type one in the table.")
                say(0.6 * i / len(clips), f"Transcribing {clip['name']} ({i + 1}/{len(clips)})"
                    + (" — the first loads Whisper, ~1 min" if i == 0 else ""))
                clip["text"] = self.asr.transcribe_detailed(clip["wav"], clip["sr"],
                                                            code_language)["text"]
        say(0.65, "Loading the voice model and embedding clips (first time ~1 min)")
        with self.hub.lock:
            tts = self.hub.acquire("voice_clone")
            embs = [self._embed(tts, c["wav"], c["sr"]) for c in clips]
        say(0.9, "Checking each reference")
        for clip, emb, verdict_ in zip(clips, embs, screen_clips(embs)):
            clip["embedding"] = emb
            clip.update(verdict_)
            check = None
            if self.asr is not None and self.asr.enabled:
                check = webui.check_reference_transcript(clip["wav"], clip["sr"], clip["text"],
                                                         may_load=True)
            clip["quality"] = reference_quality(clip["wav"], clip["sr"])
            clip["clean_score"] = clean_score(clip["quality"],
                                              None if not check else check.get("confidence"))
            clip["report"] = webui.reference_report(clip["wav"], clip["sr"], clip["text"])
            clip["warnings"] = [line[2:].strip() for line in clip["report"].split("\n\n")
                                if line.startswith("⚠️")]
        return clips

    @staticmethod
    def _embed(tts: Any, wav: np.ndarray, sr: int) -> np.ndarray:
        emb = tts.model.extract_speaker_embedding(audio=webui._to_24k(wav, sr), sr=24000)
        return emb.float().cpu().numpy().reshape(-1)

    # -- rendering -----------------------------------------------------------
    def render(self, clips: List[Dict[str, Any]], sentences: List[str], seeds: List[int],
               folds: int, language: str, single: Optional[int] = None, clean: bool = True,
               progress: Any = None) -> str:
        """Render every version for every trial. `single` is the index of your clip; None
        means "the most typical clip of each round's build set"."""
        import torch
        from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

        if self.hub is None:
            raise RuntimeError("Started with --listen-only: no model to render with.")
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            raise ValueError("Give at least one target sentence.")
        if any(not c["text"].strip() for c in clips):
            raise ValueError("Every clip needs a transcript — run Screen first.")
        plan = make_folds(len(clips), folds, keep=single)
        stamp, n = time.strftime("%Y%m%d-%H%M%S"), 1
        run_id = stamp
        os.makedirs(self.out_root, exist_ok=True)
        while True:                # a second render in the same second must not overwrite
            run_dir = os.path.join(self.out_root, run_id)
            try:
                os.mkdir(run_dir)
                break
            except FileExistsError:
                n += 1
                run_id = f"{stamp}-{n}"
        for sub in ("takes", "listen", "heldout"):
            os.makedirs(os.path.join(run_dir, sub))
        rng = random.Random(run_id)
        trials = build_trials(len(plan), len(sentences), seeds)
        total = len(trials) * len(ARM_KEYS)
        asr_language = None if language in (None, "", "Auto") else language
        takes: List[Dict[str, Any]] = []
        done = 0
        started = time.time()
        sr = clips[0]["sr"]

        def say(what: str) -> None:
            print(f"[lab] {what}", flush=True)
            if progress is not None:
                progress(done / total, desc=what)

        say(f"Encoding {len(clips)} references, then {total} takes")
        refs = [webui.prepare_reference(c["wav"], c["sr"]) if clean else c["wav"] for c in clips]
        with self.hub.lock:
            tts = self.hub.acquire("voice_clone")
            items = [tts.create_voice_clone_prompt(ref_audio=(w, c["sr"]), ref_text=c["text"],
                                                   x_vector_only_mode=False)[0]
                     for w, c in zip(refs, clips)]
            embs = [it.ref_spk_embedding.float().cpu().numpy().reshape(-1) for it in items]
            fold_meta = []
            for f, (held, build) in enumerate(plan):
                b_embs = [embs[i] for i in build]
                s = single if single is not None else build[medoid_index(b_embs)]
                c = max(build, key=lambda i: (clips[i].get("clean_score", 0.0), i == s))
                like = items[s].ref_spk_embedding
                mean_t = torch.from_numpy(mean_embedding(b_embs)).reshape(like.shape).to(
                    device=like.device, dtype=like.dtype)
                joined, joined_text, used = concat_clips([refs[i] for i in build],
                                                         [clips[i]["text"] for i in build], sr)
                prompts = {
                    "single": items[s],
                    "single_mean": VoiceClonePromptItem(
                        ref_code=items[s].ref_code, ref_spk_embedding=mean_t,
                        x_vector_only_mode=False, icl_mode=True, ref_text=clips[s]["text"]),
                    "concat": tts.create_voice_clone_prompt(
                        ref_audio=(joined, sr), ref_text=joined_text,
                        x_vector_only_mode=False)[0],
                    "cleanest": items[c],
                    "xvec_mean": VoiceClonePromptItem(
                        ref_code=None, ref_spk_embedding=mean_t, x_vector_only_mode=True,
                        icl_mode=False, ref_text=None),
                }
                held_emb = self._embed(tts, clips[held]["wav"], clips[held]["sr"])
                held_path = os.path.join(run_dir, "heldout", f"fold{f}.wav")
                sf.write(held_path, loudness_match(clips[held]["wav"]), clips[held]["sr"])
                fold_meta.append({"fold": f, "heldout": clips[held]["name"],
                                  "build": [clips[i]["name"] for i in build],
                                  "single": clips[s]["name"], "cleanest": clips[c]["name"],
                                  "concat_clips": used, "concat_seconds": round(len(joined) / sr, 2),
                                  "heldout_wav": os.path.relpath(held_path, run_dir)})
                for trial in [t for t in trials if t["fold"] == f]:
                    text = sentences[trial["sentence"]]
                    for arm in ARM_KEYS:
                        if arm == "cleanest" and c == s:
                            # Same clip, same seed: the take would be your clip's again.
                            twin = dict(next(t for t in takes if t["trial"] == trial["id"]
                                             and t["arm"] == BASELINE))
                            twin.update(arm=arm, same_as=BASELINE)
                            takes.append(twin)
                            done += 1
                            continue
                        torch.manual_seed(trial["seed"])
                        wavs, rate = tts.generate_voice_clone(
                            text=text, language=language, voice_clone_prompt=[prompts[arm]],
                            max_new_tokens=webui._budget(text),
                            **webui._sampling(**webui.SAMPLING_DEFAULTS))
                        wav = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
                        takes.append(self._score(tts, wav, int(rate), text, asr_language,
                                                 held_emb, trial, arm, run_dir))
                        done += 1
                        left = (time.time() - started) / done * (total - done)
                        say(f"Take {done}/{total} ({ARM_LABELS[arm]}) · ~{left / 60:.0f} min left")

        pairs = build_pairs(trials, CHALLENGERS, rng)
        twins = {(t["trial"], t["arm"]) for t in takes if t.get("same_as") == BASELINE}
        for p in pairs:
            p["identical"] = (p["trial"], p["challenger"]) in twins
        clip_meta = [{k: c.get(k) for k in ("name", "text", "typicality", "outlier", "warnings",
                                            "quality", "clean_score")}
                     | {"seconds": round(len(c["wav"]) / c["sr"], 2)} for c in clips]
        run = {"format": RUN_FORMAT, "id": run_id, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "language": language, "sentences": sentences, "seeds": seeds,
               "single": None if single is None else clips[single]["name"],
               "clean_reference": clean, "sampling": webui.SAMPLING_DEFAULTS, "arms": ARMS,
               "clips": clip_meta, "folds": fold_meta, "trials": trials, "takes": takes,
               "pairs": pairs}
        with open(os.path.join(run_dir, "run.json"), "w", encoding="utf-8") as handle:
            json.dump(run, handle, indent=1, default=float)
        print(f"[lab] done: {run_dir}", flush=True)
        return run_dir

    def _score(self, tts, wav, sr, text, asr_language, held_emb, trial, arm, run_dir):
        name = f"{trial['id']}_{arm}.wav"
        sf.write(os.path.join(run_dir, "takes", name), wav, sr)
        sf.write(os.path.join(run_dir, "listen", name), loudness_match(wav), sr)
        take: Dict[str, Any] = {"trial": trial["id"], "arm": arm, "file": name,
                                "seconds": round(len(wav) / float(sr), 3),
                                "hf_milli": high_band_milli(wav, sr),
                                "penalty": webui.defect_report(wav, sr, text).get("penalty")}
        try:
            take["sim_heldout"] = round(webui._cosine(held_emb, self._embed(tts, wav, sr)), 4)
        except Exception:  # noqa: BLE001 - a missing score must not sink the run
            take["sim_heldout"] = None
        take["wer"] = take["confidence"] = None
        if self.asr is not None and self.asr.enabled:
            try:
                heard, confidence = self.asr.assess(wav, sr, text, asr_language)
                take.update(heard=heard, wer=round(webui.word_error_rate(heard, text), 3),
                            confidence=None if confidence is None else round(confidence, 3))
            except Exception:  # noqa: BLE001
                pass
        return take


# ---------------------------------------------------------------------------
# Runs on disk

def list_runs(out_root: str = OUT_ROOT) -> List[str]:
    return sorted((os.path.basename(os.path.dirname(p))
                   for p in glob.glob(os.path.join(out_root, "*", "run.json"))), reverse=True)


def load_run(run_id: str, out_root: str = OUT_ROOT) -> Dict[str, Any]:
    if not run_id or os.path.basename(run_id) != run_id or run_id.startswith("."):
        raise ValueError("Pick a run.")
    with open(os.path.join(out_root, run_id, "run.json"), encoding="utf-8") as handle:
        run = json.load(handle)
    if run.get("format") != RUN_FORMAT:
        raise ValueError(f"Run {run_id} was made by an older version of the lab (no A/B pairs). "
                         f"Render it again.")
    return run


def load_answers(run_id: str, out_root: str = OUT_ROOT) -> Dict[str, Dict[str, Any]]:
    path = os.path.join(out_root, run_id, "ab.json")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _write_answers(run_id: str, answers: Dict[str, Any], out_root: str) -> None:
    path = os.path.join(out_root, run_id, "ab.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(answers, handle, indent=1)
    os.replace(tmp, path)          # a crash mid-write must not lose earlier answers


def record_answer(run: Dict[str, Any], pair_id: str, picked: str,
                  out_root: str = OUT_ROOT) -> Dict[str, Any]:
    """`picked` is what the listener clicked: "A", "B" or "tie". Stored as what it means."""
    pair = next(p for p in run["pairs"] if p["id"] == pair_id)
    if picked == "tie":
        choice = "tie"
    elif picked in ("A", "B"):
        choice = "single" if picked == pair["single_side"] else "challenger"
    else:
        raise ValueError(f"unknown answer {picked!r}")
    answers = load_answers(run["id"], out_root)
    answers[pair_id] = {"picked": picked, "choice": choice,
                        "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "n": len(answers)}
    _write_answers(run["id"], answers, out_root)
    return answers


def undo_answer(run_id: str, out_root: str = OUT_ROOT) -> Optional[str]:
    answers = load_answers(run_id, out_root)
    if not answers:
        return None
    last = max(answers, key=lambda k: answers[k].get("n", 0))
    answers.pop(last)
    _write_answers(run_id, answers, out_root)
    return last


def next_pair(run: Dict[str, Any], answers: Dict[str, Any],
              arms: Optional[Sequence[str]] = None) -> Optional[Dict[str, Any]]:
    for pair in run["pairs"]:
        if pair["id"] not in answers and (not arms or pair["challenger"] in arms):
            return pair
    return None


def results_markdown(run_id: str, out_root: str = OUT_ROOT) -> str:
    run = load_run(run_id, out_root)
    text = summary_markdown(summarise(run, load_answers(run_id, out_root)))
    rounds = "\n".join(
        f"- round {f['fold']}: held out *{f['heldout']}* · your clip *{f['single']}* · "
        f"cleanest *{f['cleanest']}*" + (" (= your clip)" if f["cleanest"] == f["single"] else "")
        + f" · joined {f['concat_clips']} clips / {f['concat_seconds']}s" for f in run["folds"])
    flagged = [c for c in run["clips"] if c.get("outlier") or c.get("warnings")]
    notes = ""
    if flagged:
        notes = "\n\n**Clips flagged at screening** (still used):\n" + "\n".join(
            f"- {c['name']}: " + "; ".join((["outlier — sits apart from the others"]
                                             if c.get("outlier") else []) + (c.get("warnings") or []))
            for c in flagged)
    single = run.get("single") or "the most typical clip of each round"
    return f"### Run {run_id} — baseline: {single}\n\n{text}\n\n**Rounds**\n{rounds}{notes}"


# ---------------------------------------------------------------------------
# Web UI

def _clip_rows(clips: List[Dict[str, Any]]) -> List[List[Any]]:
    rows = []
    for c in clips:
        flag = ("outlier · " if c.get("outlier") else "") + " · ".join(
            w.split(".")[0] for w in c.get("warnings", []))
        rows.append([c["name"], round(len(c["wav"]) / c["sr"], 1), c["text"],
                     c.get("typicality", ""), c.get("clean_score", ""), flag or "ok"])
    return rows


def build_lab_ui(lab: Lab, preloaded: List[Dict[str, Any]]) -> Any:
    import gradio as gr

    def names(clips):
        return [c["name"] for c in clips or []]

    def clip_views(clips, msg, single=None):
        """Everything that shows the clip list, refreshed together."""
        choices = [AUTO_SINGLE] + names(clips)
        keep = single if single in choices else AUTO_SINGLE
        return (clips, _clip_rows(clips), msg, gr.update(choices=choices, value=keep),
                gr.update(choices=names(clips), value=None), estimate(clips, *_last))

    _last = [DEFAULT_SENTENCES, 1, 4]      # sentences, seeds, rounds — for the estimate line

    def load_uploads(files, long_file, clips, single):
        clips = list(clips or [])
        try:
            for f in files or []:
                clips += clips_from_paths([f], trusted=False)
            if long_file:
                clips += clips_from_long(long_file, trusted=False)
        except Exception as e:  # noqa: BLE001
            return clip_views(clips, f"Error: {e}", single)
        unique_names(clips)
        return clip_views(clips, f"{len(clips)} clip(s). Next: pick **your clip**, then "
                                 f"**Screen**.", single)

    def remove_clip(clips, which, single):
        clips = [c for c in clips or [] if c["name"] != which]
        return clip_views(clips, f"Removed {which}." if which else "Pick a clip to remove.",
                          single)

    def apply_table(clips, table):
        values = table.values.tolist() if hasattr(table, "values") else (table or [])
        by_name = {row[0]: row for row in values if row}
        for clip in clips:
            if clip["name"] in by_name:
                clip["text"] = str(by_name[clip["name"]][2] or "").strip()
        return clips

    def do_screen(clips, table, language, single, progress=gr.Progress()):
        if not clips:
            return clip_views([], "Add some clips first.", single)
        clips = apply_table(clips, table)
        try:
            clips = lab.screen(clips, language, progress)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            return clip_views(clips, f"Error: {e}", single)
        bad = [c["name"] for c in clips if c.get("outlier")]
        msg = (f"Screened {len(clips)} clips. Read each transcript against what you hear and fix "
               f"it in the table if it is wrong.")
        if bad:
            msg += (f"\n\n⚠️ **Outliers:** {', '.join(bad)} — they sit apart from the rest (other "
                    f"mic, room, mood, or person?). Remove them, unless you want to see what an "
                    f"outlier does.")
        return clip_views(clips, msg, single)

    def estimate(clips, sentences, seeds, folds):
        n = len(clips or [])
        if n < MIN_CLIPS:
            return f"Need at least {MIN_CLIPS} clips; have {n}."
        rounds = min(int(folds or 1), n - 1)
        s = len([x for x in (sentences or "").splitlines() if x.strip()])
        trials = rounds * s * int(seeds or 1)
        takes = trials * len(ARM_KEYS)
        return (f"{rounds} round(s) × {s} sentence(s) × {int(seeds or 1)} seed(s): "
                f"**{takes} takes** to render (~{takes * 8 / 60:.0f}–{takes * 15 / 60:.0f} min), "
                f"then **{trials * len(CHALLENGERS)} A/B pairs** to listen to "
                f"(~{trials * len(CHALLENGERS) * 20 / 60:.0f} min). Untick versions in Listen to "
                f"do fewer.")

    def on_settings(clips, sentences, seeds, folds):
        _last[:] = [sentences, seeds, folds]
        return estimate(clips, sentences, seeds, folds)

    def do_render(clips, table, sentences, seeds, folds, language, single, clean,
                  progress=gr.Progress()):
        clips = apply_table(clips or [], table)
        try:
            if len(clips) < MIN_CLIPS:
                raise ValueError(f"Need at least {MIN_CLIPS} clips — add some first.")
            if any("embedding" not in c for c in clips):
                clips = lab.screen(clips, language, progress)
            index = None if single in (None, "", AUTO_SINGLE) else names(clips).index(single)
            run_dir = lab.render(clips, sentences.splitlines(),
                                 list(range(1234, 1234 + int(seeds or 1))), int(folds or 1),
                                 language, index, bool(clean), progress)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            return f"Error: {e}", gr.update(), gr.update()
        run_id = os.path.basename(run_dir)
        runs = list_runs(lab.out_root)
        return (f"✅ Rendered run **{run_id}**. Open **2 · Listen** and press **Start**.",
                gr.update(choices=runs, value=run_id), gr.update(choices=runs, value=run_id))

    def pair_view(run_id, run, pair, header):
        trial = next(t for t in run["trials"] if t["id"] == pair["trial"])
        takes = {t["arm"]: t for t in run["takes"] if t["trial"] == trial["id"]}
        base = os.path.join(lab.out_root, run_id)
        mine = os.path.join(base, "listen", takes[BASELINE]["file"])
        theirs = os.path.join(base, "listen", takes[pair["challenger"]]["file"])
        a, b = (mine, theirs) if pair["single_side"] == "A" else (theirs, mine)
        held = os.path.join(base, run["folds"][trial["fold"]]["heldout_wav"])
        header += (f" · *“{run['sentences'][trial['sentence']]}”*\n\n"
                   f"Play **the real person**, then **A** and **B**. Which sounds more like them?")
        return [pair["id"], header, held, a, b]

    def show_pair(run_id, arms, notice=""):
        lead = notice + "\n\n" if notice else ""
        try:
            run = load_run(run_id, lab.out_root)
        except Exception as e:  # noqa: BLE001
            return [None, f"{e}", None, None, None]
        answers = load_answers(run_id, lab.out_root)
        todo = [p for p in run["pairs"] if not arms or p["challenger"] in arms]
        left = sum(p["id"] not in answers for p in todo)
        pair = next_pair(run, answers, arms)
        if pair is None:
            return [None, lead + f"✅ All {len(todo)} pairs answered. See **3 · Results**.",
                    None, None, None]
        return pair_view(run_id, run, pair, lead + f"**Pair {len(todo) - left + 1} of "
                                                   f"{len(todo)}**")

    def answer(run_id, arms, pair_id, picked):
        if pair_id:
            try:
                record_answer(load_run(run_id, lab.out_root), pair_id, picked, lab.out_root)
            except Exception as e:  # noqa: BLE001
                return show_pair(run_id, arms, f"Error: {e}")
        return show_pair(run_id, arms)

    def undo(run_id, arms):
        try:
            gone = undo_answer(run_id, lab.out_root)
            if gone is None:
                return show_pair(run_id, arms, "Nothing to undo.")
            run = load_run(run_id, lab.out_root)
            pair = next(p for p in run["pairs"] if p["id"] == gone)
            return pair_view(run_id, run, pair, "↩️ **Undone — answer this pair again**")
        except Exception as e:  # noqa: BLE001
            return show_pair(run_id, arms, f"Error: {e}")

    def show_results(run_id):
        try:
            return results_markdown(run_id, lab.out_root)
        except Exception as e:  # noqa: BLE001
            return f"{e}"

    runs = list_runs(lab.out_root)
    challenger_choices = [(ARM_LABELS[k], k) for k in CHALLENGERS]
    with gr.Blocks(title="Multi-clip clone lab") as demo:
        gr.Markdown("# Multi-clip clone lab\nIs a clone built from several clips better than "
                    "**your one clip**? Render the versions, then A/B each one against your "
                    "clip, blind, with the real person as the reference.\n\n" + webui.ETHICS_NOTE)
        clips_state = gr.State(preloaded)
        with gr.Tab("1 · Clips & render"):
            gr.Markdown("Add **3 or more clips of one person**, ideally from different days or "
                        "rooms (4–8 clips of 5–10s is plenty), or one long recording, which is cut "
                        "into ~6s clips. A long recording's clips share a mic and room, so its "
                        "results are kinder than clips from different sessions would give.")
            with gr.Row():
                uploads = gr.File(label="Clips (several files)", file_count="multiple",
                                  type="filepath")
                long_file = gr.File(label="…or one long recording", file_count="single",
                                    type="filepath")
            add_btn = gr.Button("Add clips")
            table = gr.Dataframe(headers=["clip", "sec", "transcript (editable)", "typicality",
                                          "clean", "flags"], value=_clip_rows(preloaded),
                                 interactive=True, wrap=True)
            with gr.Row():
                single = gr.Dropdown(label="Your clip — the one you'd use in the Clone tab "
                                           "(the baseline; never held out)",
                                     choices=[AUTO_SINGLE] + names(preloaded),
                                     value=AUTO_SINGLE, scale=3)
                language = gr.Dropdown(label="Language", choices=webui.LANGUAGE_CHOICES,
                                       value="English", scale=1)
            with gr.Row():
                screen_btn = gr.Button("Screen (transcribe + check)", variant="secondary")
                remove_dd = gr.Dropdown(label="Remove a clip", choices=names(preloaded),
                                        value=None, scale=2)
                remove_btn = gr.Button("Remove", scale=1)
                clear_btn = gr.Button("Clear all", scale=1)
            screen_md = gr.Markdown("Add clips, pick yours, then **Screen**.", min_height=40)
            sentences = gr.Textbox(label="Target sentences (one per line)", lines=4,
                                   value=DEFAULT_SENTENCES)
            with gr.Row():
                seeds = gr.Number(label="Seeds per sentence", value=1, precision=0, minimum=1)
                folds = gr.Number(label="Rounds (a different clip held out each)", value=4,
                                  precision=0, minimum=1)
                clean = gr.Checkbox(label="Clean references (as the Clone tab does)", value=True)
            estimate_md = gr.Markdown(estimate(preloaded, DEFAULT_SENTENCES, 1, 4))
            render_btn = gr.Button("Render all versions", variant="primary")
            render_md = gr.Markdown("Progress shows here, and per take in the terminal.",
                                    min_height=80)

        with gr.Tab("2 · Listen"):
            with gr.Row():
                run_dd = gr.Dropdown(label="Run", choices=runs, value=runs[0] if runs else None,
                                     scale=2)
                arms_cb = gr.CheckboxGroup(challenger_choices, value=list(CHALLENGERS),
                                           label="Versions to A/B against your clip", scale=3)
                start_btn = gr.Button("Start / resume", variant="primary", scale=1)
            pair_state = gr.State(None)
            pair_md = gr.Markdown("Pick a run and press **Start**.", min_height=60)
            real = gr.Audio(label="The real person (held out: neither take was built from it)",
                            type="filepath", interactive=False)
            with gr.Row():
                take_a = gr.Audio(label="A", type="filepath", interactive=False)
                take_b = gr.Audio(label="B", type="filepath", interactive=False)
            with gr.Row():
                pick_a = gr.Button("A sounds more like them", variant="primary")
                pick_tie = gr.Button("Can't tell")
                pick_b = gr.Button("B sounds more like them", variant="primary")
            undo_btn = gr.Button("↩ Undo last answer", size="sm")

        with gr.Tab("3 · Results"):
            with gr.Row():
                res_dd = gr.Dropdown(label="Run", choices=runs, value=runs[0] if runs else None)
                res_btn = gr.Button("Refresh")
            res_md = gr.Markdown()

        clip_outputs = [clips_state, table, screen_md, single, remove_dd, estimate_md]
        add_btn.click(load_uploads, [uploads, long_file, clips_state, single], clip_outputs)
        remove_btn.click(remove_clip, [clips_state, remove_dd, single], clip_outputs)
        clear_btn.click(lambda: clip_views([], "Cleared."), None, clip_outputs)
        screen_btn.click(do_screen, [clips_state, table, language, single], clip_outputs,
                         show_progress="full")
        for box in (sentences, seeds, folds):
            box.change(on_settings, [clips_state, sentences, seeds, folds], estimate_md)
        render_btn.click(lambda: gr.update(interactive=False), None, render_btn).then(
            do_render, [clips_state, table, sentences, seeds, folds, language, single, clean],
            [render_md, run_dd, res_dd], show_progress="full").then(
            lambda: gr.update(interactive=True), None, render_btn)
        pair_outputs = [pair_state, pair_md, real, take_a, take_b]
        start_btn.click(show_pair, [run_dd, arms_cb], pair_outputs)
        run_dd.change(show_pair, [run_dd, arms_cb], pair_outputs)
        for button, picked in ((pick_a, "A"), (pick_b, "B"), (pick_tie, "tie")):
            button.click(lambda r, a, p, picked=picked: answer(r, a, p, picked),
                         [run_dd, arms_cb, pair_state], pair_outputs)
        undo_btn.click(undo, [run_dd, arms_cb], pair_outputs)
        res_btn.click(show_results, res_dd, res_md)
        res_dd.change(show_results, res_dd, res_md)
    demo.lab_handlers = {"load_uploads": load_uploads, "remove_clip": remove_clip,
                         "show_pair": show_pair, "answer": answer, "undo": undo,
                         "estimate": estimate}
    return demo


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    import torch

    parser = argparse.ArgumentParser(description="A/B a several-clip clone against your one clip.")
    parser.add_argument("--clips", help="Directory of audio clips (optional name.txt transcripts).")
    parser.add_argument("--long", help="One long recording to cut into ~6s clips.")
    parser.add_argument("--single", help="With --render-only: the file name of your clip "
                                         "(default: the most typical clip of each round).")
    parser.add_argument("--language", default="English")
    parser.add_argument("--sentences", help="File with one target sentence per line.")
    parser.add_argument("--seeds", type=int, default=1, help="Seeds per sentence.")
    parser.add_argument("--folds", type=int, default=4, help="Rounds (held-out clips).")
    parser.add_argument("--no-clean", action="store_true", help="Skip prepare_reference.")
    parser.add_argument("--render-only", action="store_true", help="Render and exit, no UI.")
    parser.add_argument("--listen-only", action="store_true",
                        help="A/B existing runs; loads no model and takes no lock.")
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--no-asr", action="store_true", help="No Whisper: no WER, typed transcripts.")
    parser.add_argument("--username", default=None,
                        help="Require a login (same as the web UI; also QWEN_TTS_USERNAME).")
    parser.add_argument("--password-file", default=None,
                        help="File holding the password (chmod 600); or QWEN_TTS_PASSWORD.")
    args = parser.parse_args(argv)
    args.password = None          # resolve_credentials reads it; never taken on argv here

    preloaded: List[Dict[str, Any]] = []
    if args.clips:
        paths = sorted(p for p in glob.glob(os.path.join(args.clips, "*"))
                       if os.path.splitext(p)[1].lower() in (".wav", ".flac", ".mp3", ".ogg", ".m4a"))
        preloaded = clips_from_paths(paths, trusted=True)
    if args.long:
        preloaded += clips_from_long(args.long, trusted=True)
    unique_names(preloaded)

    def launch(lab: Lab) -> int:
        if args.render_only:
            if len(preloaded) < MIN_CLIPS:
                print(f"error: --render-only needs --clips/--long with ≥{MIN_CLIPS} clips",
                      file=sys.stderr)
                return 2
            index = None
            if args.single:
                found = [i for i, c in enumerate(preloaded) if c["name"] == args.single]
                if not found:
                    print(f"error: --single {args.single!r} is not one of: "
                          f"{', '.join(c['name'] for c in preloaded)}", file=sys.stderr)
                    return 2
                index = found[0]
            sentences = DEFAULT_SENTENCES.splitlines()
            if args.sentences:
                with open(args.sentences, encoding="utf-8") as handle:
                    sentences = handle.read().splitlines()
            clips = lab.screen(preloaded, args.language)
            run_dir = lab.render(clips, sentences, list(range(1234, 1234 + args.seeds)),
                                 args.folds, args.language, index, not args.no_clean)
            print(results_markdown(os.path.basename(run_dir), lab.out_root))
            print(f"\nwrote {run_dir} — A/B it with: scripts/multi_clip_lab.py --listen-only")
            return 0
        credentials = webui.resolve_credentials(args)
        if args.ip not in ("127.0.0.1", "localhost", "::1") and credentials is None:
            print("[lab] WARNING: serving with no login on "
                  f"{args.ip}:{args.port}. Anyone who can reach it can upload audio and run "
                  "the model. Pass --username (and --password-file).", flush=True)
        demo = build_lab_ui(lab, preloaded)
        demo.queue(default_concurrency_limit=1).launch(
            server_name=args.ip, server_port=args.port, inbrowser=False,
            auth=[credentials] if credentials else None,
            theme=webui.UI_THEME, allowed_paths=[lab.out_root])
        return 0

    if args.listen_only:
        return launch(Lab(hub=None, asr=None))

    # Same rule as the web UI, the smoke test and make_voice: one model owner at a time.
    owner = contextlib.ExitStack()
    try:
        owner.enter_context(webui.exclusive_owner("multi_clip_lab"))
    except RuntimeError as e:
        print(f"error: {e}\n(or pass --listen-only to A/B an existing run)", file=sys.stderr)
        return 2
    with owner:
        hub = webui.ModelHub(webui.MODELS_DIR, torch.bfloat16, "sdpa", 1, max_cpu_resident=1)
        webui.HUB = hub
        asr = webui.Transcriber(model_id="none" if args.no_asr else webui.ASR_MODEL_ID)
        webui.ASR = asr
        return launch(Lab(hub=hub, asr=asr))


if __name__ == "__main__":
    sys.exit(main())
