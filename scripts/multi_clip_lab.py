#!/usr/bin/env python3
# coding=utf-8
"""Multi-clip clone lab: does building a clone from several good clips beat one?

    ./run.sh is NOT running (one model owner at a time), then:
    .venv/bin/python scripts/multi_clip_lab.py                  # web UI on 127.0.0.1:7861
    .venv/bin/python scripts/multi_clip_lab.py --clips DIR      # pre-load DIR/*.wav (+ .txt)
    .venv/bin/python scripts/multi_clip_lab.py --clips DIR --render-only   # headless
    .venv/bin/python scripts/multi_clip_lab.py --listen-only    # rate an old run, no GPU

It is an experiment, not a feature: nothing here touches saved voices or the main UI.

**What it compares.** Qwen3-TTS conditions every generation on ONE reference: the codec
codes of one clip (plus its transcript) and one speaker embedding. "Several clips" can only
enter through those two slots, so the arms are the honest ways of filling them:

  other        one build clip that is *not* the most typical — "any one good clip"
  medoid       the most typical build clip (embedding closest to the build-set mean)
  medoid_mean  the medoid's codes and transcript + the *averaged* embedding of all build clips
  concat       all build clips joined into one longer reference, transcripts joined
  xvec_mean    the averaged embedding alone (x-vector mode, no codes, no transcript)

`medoid` is the baseline: the question is whether averaging beats picking the best clip,
not whether it beats a random one.

**Why held-out scoring, and why your ears.** The similarity score (SIM) uses the same
speaker encoder that produced the averaged embedding, so an arm conditioned on the build
centroid is graded by its own ruler when scored against that centroid. That number is
shown, labelled *biased*. The primary objective score is SIM against a clip none of the
arms were built from (leave-one-out folds). Better still is a listener: the Listen tab
plays the real held-out clip, then every arm's take of the same sentence and seed, blind
and shuffled and loudness-matched (so "louder" cannot pass for "more like them").

Same seed across arms: each trial is a paired comparison, which is what makes a few dozen
ratings enough to see a real difference.

Output: tests/output/multi_clip/<run>/ (gitignored) — run.json, ratings.json, wavs.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import json
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

ARMS: List[Tuple[str, str]] = [
    ("other", "One good clip (not the most typical)"),
    ("medoid", "Most typical clip"),
    ("medoid_mean", "Most typical clip + averaged embedding"),
    ("concat", "All build clips joined"),
    ("xvec_mean", "Averaged embedding only (x-vector)"),
]
ARM_KEYS = [k for k, _ in ARMS]
ARM_LABELS = dict(ARMS)
BASELINE = "medoid"
LETTERS = "ABCDEFGH"
MIN_CLIPS = 3                 # one held out + at least two to build from
CONCAT_MAX_SECONDS = 20.0     # past REF_MAX_SECONDS on purpose: length is what this arm tests
CONCAT_GAP_SECONDS = 0.3
LISTEN_RMS_DB = -23.0
RATING_CHOICES = ["1", "2", "3", "4", "5"]

DEFAULT_SENTENCES = (
    "I left the keys on the kitchen table, right next to the letter from your aunt.\n"
    "Honestly, I did not expect the meeting to run that long, but here we are.\n"
    "Could you call me back when you get a minute? It is nothing urgent."
)


# ---------------------------------------------------------------------------
# Pure helpers (CPU only; tested in tests/test_app_logic.py)

def make_folds(n: int, k: int) -> List[Tuple[int, List[int]]]:
    """Leave-one-out folds: (held-out index, build indices), at most `k`, spread evenly."""
    if n < MIN_CLIPS:
        raise ValueError(f"Need at least {MIN_CLIPS} clips (one held out, two to build from); "
                         f"got {n}.")
    k = max(1, min(int(k), n))
    held = sorted({int(round(i * n / float(k))) % n for i in range(k)})
    return [(h, [i for i in range(n) if i != h]) for h in held]


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


def other_index(n: int, medoid: int) -> int:
    """The 'one good clip' arm: the first build clip that is not the medoid."""
    if n < 2:
        raise ValueError("need two build clips")
    return 1 if medoid == 0 else 0


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
    """Same RMS for every take a listener compares, peak-limited by scaling (no clipping)."""
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    rms = float(np.sqrt(np.mean(np.square(wav)))) if wav.size else 0.0
    if rms <= 1e-6:
        return wav
    gain = (10.0 ** (rms_db / 20.0)) / rms
    top = float(np.max(np.abs(wav))) * gain
    if top > peak:
        gain *= peak / top
    return (wav * gain).astype(np.float32)


def build_trials(folds: int, sentences: int, seeds: Sequence[int], arms: Sequence[str],
                 rng: random.Random) -> List[Dict[str, Any]]:
    """One trial per (fold, sentence, seed); the arms appear in a shuffled order."""
    trials = []
    for f in range(folds):
        for s in range(sentences):
            for seed in seeds:
                order = list(arms)
                rng.shuffle(order)
                trials.append({"id": f"f{f}_s{s}_r{seed}", "fold": f, "sentence": s,
                               "seed": int(seed), "order": order})
    rng.shuffle(trials)           # so fatigue does not land on one fold or sentence
    return trials


def _stats(values: Sequence[Optional[float]]) -> Tuple[Optional[float], Optional[float], int]:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return None, None, 0
    return float(np.mean(xs)), (float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0), len(xs)


def _paired(by_trial: Dict[str, Dict[str, float]], arm: str, base: str) -> Tuple[int, int, int]:
    wins = ties = losses = 0
    for scores in by_trial.values():
        if arm in scores and base in scores and scores[arm] is not None and scores[base] is not None:
            d = scores[arm] - scores[base]
            if abs(d) < 1e-9:
                ties += 1
            elif d > 0:
                wins += 1
            else:
                losses += 1
    return wins, ties, losses


def summarise(takes: Sequence[Dict[str, Any]], ratings: Dict[str, Dict[str, Any]],
              baseline: str = BASELINE) -> List[Dict[str, Any]]:
    """Per arm: listener and objective means, plus paired W/T/L against the baseline."""
    listen: Dict[str, Dict[str, float]] = {}
    best: Dict[str, int] = {}
    for trial_id, r in ratings.items():
        listen[trial_id] = {a: float(v) for a, v in (r.get("scores") or {}).items()}
        if r.get("best"):
            best[r["best"]] = best.get(r["best"], 0) + 1
    rated = sum(1 for r in ratings.values() if r.get("best"))
    sim_by_trial: Dict[str, Dict[str, float]] = {}
    for t in takes:
        if t.get("sim_heldout") is not None:
            sim_by_trial.setdefault(t["trial"], {})[t["arm"]] = float(t["sim_heldout"])
    rows = []
    for arm in ARM_KEYS:
        mine = [t for t in takes if t["arm"] == arm]
        if not mine:
            continue
        row: Dict[str, Any] = {"arm": arm, "label": ARM_LABELS[arm], "takes": len(mine)}
        row["listen"] = _stats([s.get(arm) for s in listen.values()])
        row["best_share"] = (best.get(arm, 0) / rated) if rated else None
        for key in ("sim_heldout", "sim_build", "wer", "confidence", "hf_milli", "seconds"):
            row[key] = _stats([t.get(key) for t in mine])
        if arm != baseline:
            row["listen_wtl"] = _paired(listen, arm, baseline)
            row["sim_wtl"] = _paired(sim_by_trial, arm, baseline)
        rows.append(row)
    return rows


def summary_markdown(rows: Sequence[Dict[str, Any]], n_rated: int) -> str:
    def ms(stat, digits=3):
        mean, sd, n = stat
        return "—" if mean is None else f"{mean:.{digits}f} ± {sd:.{digits}f}"

    def wtl(v):
        return "baseline" if v is None else f"{v[0]}/{v[1]}/{v[2]}"

    lines = [
        f"Listener ratings: **{n_rated}** trial(s). W/T/L = paired wins/ties/losses against "
        f"`{BASELINE}` on the same sentence and seed.",
        "",
        "| arm | you (1–5) | picked best | W/T/L (you) | SIM held-out | W/T/L (SIM) | "
        "SIM build *(biased)* | WER | conf | >9kHz (m) | sec |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        best = "—" if r["best_share"] is None else f"{r['best_share']:.0%}"
        lines.append(
            f"| **{r['arm']}** — {r['label']} | {ms(r['listen'], 2)} | {best} | "
            f"{wtl(r.get('listen_wtl'))} | {ms(r['sim_heldout'])} | {wtl(r.get('sim_wtl'))} | "
            f"{ms(r['sim_build'])} | {ms(r['wer'], 2)} | {ms(r['confidence'], 2)} | "
            f"{ms(r['hf_milli'], 2)} | {ms(r['seconds'], 1)} |")
    lines += ["", "How to read it: an arm is worth building only if it wins on *your* ratings "
              "and on held-out SIM, by more than the spread, without losing on WER. A lead on "
              "the biased column alone means nothing — that arm is being graded by its own "
              "ruler."]
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
                 max_clips: int = 12) -> List[Tuple[float, float]]:
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


# ---------------------------------------------------------------------------
# The GPU part

class Lab:
    def __init__(self, hub: Optional[Any], asr: Optional[Any], out_root: str = OUT_ROOT):
        self.hub = hub
        self.asr = asr
        self.out_root = out_root

    # -- screening -----------------------------------------------------------
    def screen(self, clips: List[Dict[str, Any]], language: str) -> List[Dict[str, Any]]:
        """Fill missing transcripts, embed every clip, flag outliers and bad references."""
        if self.hub is None:
            raise RuntimeError("Started with --listen-only: no model to screen with.")
        code_language = None if language in (None, "", "Auto") else language
        for clip in clips:
            if not clip["text"].strip():
                if self.asr is None or not self.asr.enabled:
                    raise RuntimeError(f"{clip['name']} has no transcript and Whisper is off; "
                                       f"type one in the table.")
                clip["text"] = self.asr.transcribe_detailed(clip["wav"], clip["sr"],
                                                            code_language)["text"]
        with self.hub.lock:
            tts = self.hub.acquire("voice_clone")
            embs = [self._embed(tts, c["wav"], c["sr"]) for c in clips]
        for clip, emb, verdict in zip(clips, embs, screen_clips(embs)):
            clip["embedding"] = emb
            clip.update(verdict)
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
               folds: int, language: str, clean: bool = True,
               progress: Any = None) -> str:
        import torch
        from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

        if self.hub is None:
            raise RuntimeError("Started with --listen-only: no model to render with.")
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            raise ValueError("Give at least one target sentence.")
        if any(not c["text"].strip() for c in clips):
            raise ValueError("Every clip needs a transcript — run Screen first.")
        plan = make_folds(len(clips), folds)
        run_id = time.strftime("%Y%m%d-%H%M%S")
        run_dir = os.path.join(self.out_root, run_id)
        os.makedirs(os.path.join(run_dir, "takes"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "listen"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "heldout"), exist_ok=True)
        rng = random.Random(run_id)
        trials = build_trials(len(plan), len(sentences), seeds, ARM_KEYS, rng)
        total = len(trials) * len(ARM_KEYS)
        asr_language = None if language in (None, "", "Auto") else language
        takes: List[Dict[str, Any]] = []
        done = 0
        started = time.time()

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
                m = build[medoid_index(b_embs)]
                o = build[other_index(len(build), build.index(m))]
                mean_np = mean_embedding(b_embs)
                like = items[m].ref_spk_embedding
                mean_t = torch.from_numpy(mean_np).reshape(like.shape).to(device=like.device,
                                                                         dtype=like.dtype)
                joined, joined_text, used = concat_clips([refs[i] for i in build],
                                                         [clips[i]["text"] for i in build],
                                                         clips[0]["sr"])
                prompts = {
                    "other": items[o],
                    "medoid": items[m],
                    "medoid_mean": VoiceClonePromptItem(
                        ref_code=items[m].ref_code, ref_spk_embedding=mean_t,
                        x_vector_only_mode=False, icl_mode=True, ref_text=clips[m]["text"]),
                    "concat": tts.create_voice_clone_prompt(
                        ref_audio=(joined, clips[0]["sr"]), ref_text=joined_text,
                        x_vector_only_mode=False)[0],
                    "xvec_mean": VoiceClonePromptItem(
                        ref_code=None, ref_spk_embedding=mean_t, x_vector_only_mode=True,
                        icl_mode=False, ref_text=None),
                }
                held_emb = self._embed(tts, clips[held]["wav"], clips[held]["sr"])
                centre = mean_embedding(b_embs)
                held_path = os.path.join(run_dir, "heldout", f"fold{f}.wav")
                sf.write(held_path, loudness_match(clips[held]["wav"]), clips[held]["sr"])
                fold_meta.append({"fold": f, "heldout": clips[held]["name"],
                                  "build": [clips[i]["name"] for i in build],
                                  "medoid": clips[m]["name"], "other": clips[o]["name"],
                                  "concat_clips": used,
                                  "concat_seconds": round(len(joined) / clips[0]["sr"], 2),
                                  "heldout_wav": os.path.relpath(held_path, run_dir)})
                for trial in [t for t in trials if t["fold"] == f]:
                    text = sentences[trial["sentence"]]
                    for arm in ARM_KEYS:
                        torch.manual_seed(trial["seed"])
                        wavs, rate = tts.generate_voice_clone(
                            text=text, language=language, voice_clone_prompt=[prompts[arm]],
                            max_new_tokens=webui._budget(text),
                            **webui._sampling(**webui.SAMPLING_DEFAULTS))
                        wav = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
                        takes.append(self._score(tts, wav, int(rate), text, asr_language,
                                                 held_emb, centre, trial, arm, run_dir))
                        done += 1
                        if progress is not None:
                            left = (time.time() - started) / done * (total - done)
                            progress(done / total, desc=f"take {done}/{total} · ~{left/60:.0f} min left")
                        print(f"[lab] {done}/{total} {trial['id']} {arm} "
                              f"sim={takes[-1]['sim_heldout']}", flush=True)

        clip_meta = [{k: c.get(k) for k in ("name", "text", "typicality", "outlier", "warnings")}
                     | {"seconds": round(len(c["wav"]) / c["sr"], 2)} for c in clips]
        run = {"id": run_id, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "language": language,
               "sentences": sentences, "seeds": seeds, "clean_reference": clean,
               "sampling": webui.SAMPLING_DEFAULTS, "arms": ARMS, "baseline": BASELINE,
               "clips": clip_meta, "folds": fold_meta, "trials": trials, "takes": takes}
        with open(os.path.join(run_dir, "run.json"), "w", encoding="utf-8") as handle:
            json.dump(run, handle, indent=1, default=float)
        return run_dir

    def _score(self, tts, wav, sr, text, asr_language, held_emb, centre, trial, arm, run_dir):
        name = f"{trial['id']}_{arm}.wav"
        sf.write(os.path.join(run_dir, "takes", name), wav, sr)
        sf.write(os.path.join(run_dir, "listen", name), loudness_match(wav), sr)
        take: Dict[str, Any] = {"trial": trial["id"], "arm": arm, "file": name,
                                "seconds": round(len(wav) / float(sr), 3),
                                "hf_milli": high_band_milli(wav, sr),
                                "penalty": webui.defect_report(wav, sr, text).get("penalty")}
        try:
            emb = self._embed(tts, wav, sr)
            take["sim_heldout"] = round(webui._cosine(held_emb, emb), 4)
            take["sim_build"] = round(webui._cosine(centre, emb), 4)
        except Exception:  # noqa: BLE001 - a missing score must not sink the run
            take["sim_heldout"] = take["sim_build"] = None
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
        return json.load(handle)


def load_ratings(run_id: str, out_root: str = OUT_ROOT) -> Dict[str, Dict[str, Any]]:
    path = os.path.join(out_root, run_id, "ratings.json")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def save_rating(run_id: str, trial_id: str, rating: Dict[str, Any], out_root: str = OUT_ROOT) -> None:
    ratings = load_ratings(run_id, out_root)
    ratings[trial_id] = rating
    path = os.path.join(out_root, run_id, "ratings.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(ratings, handle, indent=1)
    os.replace(tmp, path)          # a crash mid-write must not lose earlier ratings


def next_trial(run: Dict[str, Any], ratings: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for trial in run["trials"]:
        if trial["id"] not in ratings:
            return trial
    return None


def results_markdown(run_id: str, out_root: str = OUT_ROOT) -> str:
    run = load_run(run_id, out_root)
    ratings = load_ratings(run_id, out_root)
    text = summary_markdown(summarise(run["takes"], ratings), len(ratings))
    flagged = [c for c in run["clips"] if c.get("outlier") or c.get("warnings")]
    folds = "\n".join(f"- fold {f['fold']}: held out *{f['heldout']}*, medoid *{f['medoid']}*, "
                      f"other *{f['other']}*, concat {f['concat_clips']} clips / "
                      f"{f['concat_seconds']}s" for f in run["folds"])
    notes = ""
    if flagged:
        notes = "\n\n**Clips flagged at screening** (they were still used):\n" + "\n".join(
            f"- {c['name']}: " + "; ".join((["outlier — sits apart from the others"]
                                             if c.get("outlier") else []) + (c.get("warnings") or []))
            for c in flagged)
    return f"### Run {run_id}\n\n{text}\n\n**Folds**\n{folds}{notes}"


# ---------------------------------------------------------------------------
# Web UI

def _clip_rows(clips: List[Dict[str, Any]]) -> List[List[Any]]:
    rows = []
    for c in clips:
        flag = ("outlier · " if c.get("outlier") else "") + " · ".join(
            w.split(".")[0] for w in c.get("warnings", []))
        rows.append([c["name"], round(len(c["wav"]) / c["sr"], 1), c["text"],
                     c.get("typicality", ""), flag or "ok"])
    return rows


def build_lab_ui(lab: Lab, preloaded: List[Dict[str, Any]]) -> Any:
    import gradio as gr

    n_slots = len(ARM_KEYS)
    letters = list(LETTERS[:n_slots])

    def load_uploads(files, long_file, clips):
        clips = list(clips or [])
        try:
            for f in files or []:
                clips += clips_from_paths([f], trusted=False)
            if long_file:
                clips += clips_from_long(long_file, trusted=False)
        except Exception as e:  # noqa: BLE001
            return clips, _clip_rows(clips), f"Error: {e}"
        return clips, _clip_rows(clips), f"{len(clips)} clip(s) loaded. Next: **Screen**."

    def apply_table(clips, table):
        values = table.values.tolist() if hasattr(table, "values") else (table or [])
        for clip, row in zip(clips, values):
            clip["text"] = str(row[2] or "").strip()
        return clips

    def do_screen(clips, table, language, progress=gr.Progress()):
        if not clips:
            return clips, [], "Load some clips first."
        clips = apply_table(clips, table)
        try:
            progress(0.1, desc="transcribing and embedding")
            clips = lab.screen(clips, language)
        except Exception as e:  # noqa: BLE001
            return clips, _clip_rows(clips), f"Error: {e}"
        bad = [c["name"] for c in clips if c.get("outlier")]
        msg = (f"Screened {len(clips)} clips. Read each transcript against what you hear — "
               f"fix it in the table if it is wrong.")
        if bad:
            msg += (f"\n\n⚠️ **Outliers:** {', '.join(bad)} — sit apart from the rest (other mic, "
                    f"room, mood, or person?). Remove them with **Clear** and reload, or keep "
                    f"them to see what an outlier does to each arm.")
        return clips, _clip_rows(clips), msg

    def estimate(clips, sentences, seeds, folds):
        n = len(clips or [])
        if n < MIN_CLIPS:
            return f"Need at least {MIN_CLIPS} clips; have {n}."
        k = min(int(folds or 1), n)
        s = len([x for x in (sentences or "").splitlines() if x.strip()])
        trials = k * s * int(seeds or 1)
        return (f"{k} fold(s) × {s} sentence(s) × {int(seeds or 1)} seed(s) = **{trials} "
                f"trials** to rate, **{trials * n_slots} takes** to render "
                f"(roughly {trials * n_slots * 8 / 60:.0f}–{trials * n_slots * 15 / 60:.0f} min).")

    def do_render(clips, table, sentences, seeds, folds, language, clean,
                  progress=gr.Progress()):
        clips = apply_table(clips or [], table)
        try:
            if any("embedding" not in c for c in clips):
                clips = lab.screen(clips, language)
            run_dir = lab.render(clips, sentences.splitlines(),
                                 list(range(1234, 1234 + int(seeds or 1))), int(folds or 1),
                                 language, bool(clean), progress)
        except Exception as e:  # noqa: BLE001
            return f"Error: {e}", gr.update(), gr.update()
        run_id = os.path.basename(run_dir)
        runs = list_runs(lab.out_root)
        return (f"Rendered run **{run_id}**. Go to **Listen**.",
                gr.update(choices=runs, value=run_id), gr.update(choices=runs, value=run_id))

    def show_trial(run_id):
        empty = [gr.update(value=None)] * n_slots + [gr.update(value=None)] * n_slots
        try:
            run = load_run(run_id, lab.out_root)
        except Exception as e:  # noqa: BLE001
            return [None, f"{e}", None, gr.update(value=None), ""] + empty
        ratings = load_ratings(run_id, lab.out_root)
        trial = next_trial(run, ratings)
        if trial is None:
            return ([None, f"All {len(run['trials'])} trials rated — see **Results**.", None,
                     gr.update(value=None), ""] + empty)
        fold = run["folds"][trial["fold"]]
        held = os.path.join(lab.out_root, run_id, fold["heldout_wav"])
        by_arm = {t["arm"]: t for t in run["takes"] if t["trial"] == trial["id"]}
        audios = [os.path.join(lab.out_root, run_id, "listen", by_arm[a]["file"])
                  for a in trial["order"]]
        header = (f"**Trial {len(ratings) + 1} of {len(run['trials'])}**  \n"
                  f"Sentence: *“{run['sentences'][trial['sentence']]}”*  \n"
                  f"Listen to the real person first, then rate each take: **how much does it "
                  f"sound like the same person?** (1 = someone else, 5 = can't tell). Then "
                  f"pick the one you'd keep.")
        return ([trial["id"], header, held, gr.update(value=None), ""]
                + [gr.update(value=a) for a in audios]
                + [gr.update(value=None) for _ in range(n_slots)])

    def save_and_next(run_id, trial_id, best, note, *scores):
        if not trial_id:
            return show_trial(run_id)
        if any(s in (None, "") for s in scores) or not best:
            out = show_trial(run_id)
            out[1] = "⚠️ Rate every take and pick a favourite before moving on.\n\n" + str(out[1])
            # keep what the listener already entered
            out[3] = gr.update(value=best)
            out[4] = note
            out[5:5 + n_slots] = [gr.update() for _ in range(n_slots)]
            out[5 + n_slots:] = [gr.update(value=s) for s in scores]
            return out
        run = load_run(run_id, lab.out_root)
        trial = next(t for t in run["trials"] if t["id"] == trial_id)
        order = trial["order"]
        save_rating(run_id, trial_id, {
            "scores": {order[i]: int(scores[i]) for i in range(n_slots)},
            "best": order[letters.index(best)],
            "note": (note or "").strip(),
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, lab.out_root)
        return show_trial(run_id)

    def show_results(run_id):
        try:
            return results_markdown(run_id, lab.out_root)
        except Exception as e:  # noqa: BLE001
            return f"{e}"

    runs = list_runs(lab.out_root)
    with gr.Blocks(title="Multi-clip clone lab") as demo:
        gr.Markdown("# Multi-clip clone lab\nDoes a clone built from several good clips beat "
                    "one? Render the arms, then be the ear: rate them blind against a real clip "
                    "none of them were built from.\n\n" + webui.ETHICS_NOTE)
        clips_state = gr.State(preloaded)
        with gr.Tab("1 · Clips & render"):
            gr.Markdown("Three or more clips of **one person**, ideally from different days or "
                        "rooms — 4–8 clips of 5–10s each is plenty. Transcripts come from "
                        "Whisper unless you started with `--clips DIR` and DIR has `name.txt` "
                        "next to `name.wav`.")
            with gr.Row():
                uploads = gr.File(label="Clips (several files)", file_count="multiple",
                                  type="filepath")
                long_file = gr.File(label="…or one long recording, cut into ~6s clips",
                                    file_count="single", type="filepath")
            with gr.Row():
                add_btn = gr.Button("Add clips")
                clear_btn = gr.Button("Clear")
                language = gr.Dropdown(label="Language", choices=webui.LANGUAGE_CHOICES,
                                       value="English")
                screen_btn = gr.Button("Screen", variant="secondary")
            table = gr.Dataframe(headers=["clip", "sec", "transcript (editable)", "typicality",
                                          "flags"], value=_clip_rows(preloaded),
                                 interactive=True, wrap=True)
            screen_md = gr.Markdown()
            gr.Markdown("A long recording's clips share one mic and room, so held-out scores "
                        "from it are kinder than a clip from another day would be.")
            sentences = gr.Textbox(label="Target sentences (one per line)", lines=4,
                                   value=DEFAULT_SENTENCES)
            with gr.Row():
                seeds = gr.Number(label="Seeds per sentence", value=1, precision=0, minimum=1)
                folds = gr.Number(label="Folds (held-out clips)", value=4, precision=0, minimum=1)
                clean = gr.Checkbox(label="Clean references (as the Clone tab does)", value=True)
            estimate_md = gr.Markdown(estimate(preloaded, DEFAULT_SENTENCES, 1, 4))
            render_btn = gr.Button("Render all arms", variant="primary")
            render_md = gr.Markdown()

        with gr.Tab("2 · Listen"):
            with gr.Row():
                run_dd = gr.Dropdown(label="Run", choices=runs, value=runs[0] if runs else None)
                start_btn = gr.Button("Start / resume", variant="primary")
            trial_state = gr.State(None)
            trial_md = gr.Markdown()
            real = gr.Audio(label="The real person (held out — no take was built from it)",
                            type="filepath", interactive=False)
            slot_audio, slot_score = [], []
            with gr.Row():
                for letter in letters:
                    with gr.Column(min_width=180):
                        slot_audio.append(gr.Audio(label=f"Take {letter}", type="filepath",
                                                   interactive=False))
                        slot_score.append(gr.Radio(RATING_CHOICES, label=f"{letter}: same person?",
                                                   value=None))
            best = gr.Radio(letters, label="Which one would you keep?", value=None)
            note = gr.Textbox(label="Anything you noticed (optional)", lines=1)
            save_btn = gr.Button("Save and next", variant="primary")

        with gr.Tab("3 · Results"):
            with gr.Row():
                res_dd = gr.Dropdown(label="Run", choices=runs, value=runs[0] if runs else None)
                res_btn = gr.Button("Refresh")
            res_md = gr.Markdown()

        add_btn.click(load_uploads, [uploads, long_file, clips_state],
                      [clips_state, table, screen_md])
        clear_btn.click(lambda: ([], [], "Cleared."), None, [clips_state, table, screen_md])
        screen_btn.click(do_screen, [clips_state, table, language],
                         [clips_state, table, screen_md])
        for box in (sentences, seeds, folds):
            box.change(estimate, [clips_state, sentences, seeds, folds], estimate_md)
        table.change(estimate, [clips_state, sentences, seeds, folds], estimate_md)
        render_btn.click(do_render, [clips_state, table, sentences, seeds, folds, language, clean],
                         [render_md, run_dd, res_dd])
        trial_outputs = [trial_state, trial_md, real, best, note] + slot_audio + slot_score
        start_btn.click(show_trial, run_dd, trial_outputs)
        run_dd.change(show_trial, run_dd, trial_outputs)
        save_btn.click(save_and_next, [run_dd, trial_state, best, note] + slot_score,
                       trial_outputs)
        res_btn.click(show_results, res_dd, res_md)
        res_dd.change(show_results, res_dd, res_md)
    return demo


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    import torch

    parser = argparse.ArgumentParser(description="Compare one-clip and multi-clip clones.")
    parser.add_argument("--clips", help="Directory of *.wav (optional name.txt transcripts).")
    parser.add_argument("--long", help="One long recording to cut into ~6s clips instead.")
    parser.add_argument("--language", default="English")
    parser.add_argument("--sentences", help="File with one target sentence per line.")
    parser.add_argument("--seeds", type=int, default=1, help="Seeds per sentence.")
    parser.add_argument("--folds", type=int, default=4, help="How many clips to hold out in turn.")
    parser.add_argument("--no-clean", action="store_true", help="Skip prepare_reference.")
    parser.add_argument("--render-only", action="store_true", help="Render and exit, no UI.")
    parser.add_argument("--listen-only", action="store_true",
                        help="Rate existing runs; loads no model and takes no lock.")
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

    def launch(lab: Lab) -> int:
        if args.render_only:
            if len(preloaded) < MIN_CLIPS:
                print(f"error: --render-only needs --clips/--long with ≥{MIN_CLIPS} clips",
                      file=sys.stderr)
                return 2
            sentences = DEFAULT_SENTENCES.splitlines()
            if args.sentences:
                with open(args.sentences, encoding="utf-8") as handle:
                    sentences = handle.read().splitlines()
            clips = lab.screen(preloaded, args.language)
            run_dir = lab.render(clips, sentences, list(range(1234, 1234 + args.seeds)),
                                 args.folds, args.language, not args.no_clean)
            print(results_markdown(os.path.basename(run_dir), lab.out_root))
            print(f"\nwrote {run_dir} — rate it with: scripts/multi_clip_lab.py --listen-only")
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
        print(f"error: {e}\n(or pass --listen-only to rate an existing run)", file=sys.stderr)
        return 2
    with owner:
        hub = webui.ModelHub(webui.MODELS_DIR, torch.bfloat16, "sdpa", 1, max_cpu_resident=1)
        webui.HUB = hub
        asr = webui.Transcriber(model_id="none" if args.no_asr else webui.ASR_MODEL_ID)
        webui.ASR = asr
        return launch(Lab(hub=hub, asr=asr))


if __name__ == "__main__":
    sys.exit(main())
