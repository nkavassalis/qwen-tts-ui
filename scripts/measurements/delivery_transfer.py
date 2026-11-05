"""Does a clip's *delivery* transfer through cloning — or only its identity?

Nick's idea: keep several clips per voice, ranked by how they were delivered (brisk, calm,
animated, flat), and let the user pick which read to condition on per line. That is the only
style channel this checkpoint plausibly has — the ICL prefix is built from the reference's
codec codes, which carry how it was spoken — and the `instruct` route was just measured to be
a control that does something but not what you asked. But the whole feature is decoration
unless delivery actually carries over. So measure that first, the same way the instruct box
was measured: before anyone builds a UI for it.

Method. Cut one long recording into clips, measure each clip's delivery (speaking rate from
its Whisper transcript, pitch centre and pitch variability from F0, level and its dynamics,
pause structure). Pick the clips at the extremes of rate and of pitch variability. Then clone
the SAME target sentence from each, across several seeds, and measure the outputs with the
same instruments. The question is not "do the outputs differ" — they always do. It is whether
the outputs keep the reference ranking by more than the seed-to-seed noise floor, which is
reported next to every number.
"""
# Prerequisite: a long single-speaker recording to cut clips from, passed as argv[1].
# The verdict is closed — speaking rate does not transfer through cloning, so a
# brisk/calm clip picker would be inert. See docs/quality.md, "Does a clip's
# *delivery* transfer through cloning? Mostly no".
import importlib.util
import json
import os
import sys
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(ROOT, "webui")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "webui"))
os.chdir(ROOT)

import numpy as np
import soundfile as sf
import torch

import app as webui

spec = importlib.util.spec_from_file_location("mcl", os.path.join(ROOT, "scripts",
                                                                   "multi_clip_lab.py"))
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)

# The long recording to cut clips from: pass its path as argv[1]. Not committed —
# it is someone's voice, and only measurements/ is in git.
SOURCE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    ROOT, "tmp", "delivery", "source.mov")
OUT = os.path.join(ROOT, "tmp", "delivery")
os.makedirs(OUT, exist_ok=True)
TARGET = "その話は今朝聞いて、みんな少し驚いていました。"
SEEDS = [1234, 1235, 1236, 1237, 1238]
LANGUAGE = "Japanese"

import librosa  # noqa: E402


def delivery(wav: np.ndarray, sr: int, text: str) -> Dict[str, float]:
    """How a clip was delivered. All relative measures — no absolute thresholds anywhere."""
    dur = len(wav) / float(sr)
    win = int(0.02 * sr)
    frames = webui._rms_windows(wav, sr, win)
    active = frames[frames > 0.01] if frames.size else np.array([0.0])
    f0, _, _ = librosa.pyin(wav, fmin=60, fmax=400, sr=sr)
    voiced = f0[~np.isnan(f0)] if f0 is not None and len(f0) else np.array([0.0])
    words = len("".join(text.split()))          # CJK-aware: characters, not space-tokens
    return {
        "seconds": round(dur, 2),
        "rate": round(words / max(dur, 1e-6), 2),                     # brisk vs unhurried
        "f0_mean": round(float(np.mean(voiced)), 1),
        "f0_std": round(float(np.std(voiced)), 1),                    # animated vs flat
        "f0_range": round(float(np.percentile(voiced, 90) - np.percentile(voiced, 10)), 1),
        "level_db": round(float(20 * np.log10(float(np.mean(active)) + 1e-9)), 1),
        "dynamics_db": round(float(20 * np.log10(float(np.std(active)) + 1e-9)), 1),
        "voiced_frac": round(float(len(voiced) / max(len(f0), 1)), 2) if f0 is not None else 0.0,
    }


hub = webui.ModelHub("models", torch.bfloat16, "sdpa", 1, 1, 4096, 1024)
webui.HUB = hub
asr = webui.Transcriber(model_id=webui.ASR_MODEL_ID)
webui.ASR = asr

with hub.lock:
    tts = hub.acquire("voice_clone")

    clips = lab.clips_from_long(SOURCE, True)
    clips = [c for c in clips if len(c["wav"]) / c["sr"] >= webui.REF_MIN_SECONDS][:24]
    print(f"cut {len(clips)} clips from the long recording")
    for c in clips:
        c["text"] = asr.transcribe_detailed(c["wav"], c["sr"], None)["text"]
        c["clean"] = webui.prepare_reference(c["wav"], c["sr"])
        c["d"] = delivery(c["clean"], c["sr"], c["text"])
    print(f"{'clip':>5s} {'sec':>5s} {'rate':>6s} {'f0':>7s} {'f0sd':>6s} {'rng':>6s} "
          f"{'lvl':>6s} {'dyn':>6s} {'voiced':>6s}")
    for i, c in enumerate(clips):
        d = c["d"]
        print(f"{i:5d} {d['seconds']:5.2f} {d['rate']:6.2f} {d['f0_mean']:7.1f} "
              f"{d['f0_std']:6.1f} {d['f0_range']:6.1f} {d['level_db']:6.1f} "
              f"{d['dynamics_db']:6.1f} {d['voiced_frac']:6.2f}")

    order_rate = sorted(range(len(clips)), key=lambda i: clips[i]["d"]["rate"])
    order_f0 = sorted(range(len(clips)), key=lambda i: clips[i]["d"]["f0_std"])
    picks = {"slowest": order_rate[0], "fastest": order_rate[-1],
             "flattest": order_f0[0], "most animated": order_f0[-1]}
    # Distinct clips only: if one clip wins two axes, take the runner-up for the other.
    seen: List[int] = []
    chosen: Dict[str, int] = {}
    for label, idx in picks.items():
        while idx in seen:
            idx = order_rate.pop(0) if label in ("slowest", "fastest") else order_f0.pop(0)
        seen.append(idx)
        chosen[label] = idx
    print("\nselected references: " + ", ".join(f"{k}=clip{v}" for k, v in chosen.items()))

    results: List[Dict[str, Any]] = []
    for label, idx in chosen.items():
        c = clips[idx]
        prompt = [tts.create_voice_clone_prompt(ref_audio=(c["clean"], c["sr"]),
                                                ref_text=c["text"], x_vector_only_mode=False)[0]]
        for seed in SEEDS:
            torch.manual_seed(seed)
            wavs, sr = tts.generate_voice_clone(
                text=TARGET, language=LANGUAGE, voice_clone_prompt=prompt,
                max_new_tokens=webui._budget(TARGET),
                **webui._sampling(**webui.SAMPLING_DEFAULTS))
            wav = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
            sf.write(os.path.join(OUT, f"take_{label.replace(' ', '_')}_{seed}.wav"), wav, sr)
            m = delivery(wav, sr, TARGET)
            m.update({"reference": label, "clip": idx, "seed": seed})
            results.append(m)

json.dump({"clips": [{"index": i, **c["d"]} for i, c in enumerate(clips)],
           "chosen": chosen, "takes": results},
          open(os.path.join(OUT, "delivery_transfer.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

print(f"\n=== outputs: same target sentence, {len(SEEDS)} seeds each ===")
metrics = ("rate", "f0_mean", "f0_std", "f0_range", "level_db", "dynamics_db", "voiced_frac")
print(f"{'reference':15s}{'ref rate':>9s}{'ref f0sd':>9s}" +
      "".join(f"{m:>11s}" for m in metrics))
for label in chosen:
    rows = [r for r in results if r["reference"] == label]
    ref = clips[chosen[label]]["d"]
    cells = "".join(f"{np.mean([r[m] for r in rows]):7.2f}±{np.std([r[m] for r in rows]):3.1f}"
                    for m in metrics)
    print(f"{label:15s}{ref['rate']:9.2f}{ref['f0_std']:9.1f}" + cells)

print("\n=== does the ranking survive? ===")
for axis, ref_key, out_key in (("speaking rate", "rate", "rate"),
                               ("pitch variability", "f0_std", "f0_std"),
                               ("pitch range", "f0_range", "f0_range")):
    lo = min(chosen, key=lambda k: clips[chosen[k]][ref_key])
    hi = max(chosen, key=lambda k: clips[chosen[k]][ref_key])
    lo_out = np.mean([r[out_key] for r in results if r["reference"] == lo])
    hi_out = np.mean([r[out_key] for r in results if r["reference"] == hi])
    noise = max(np.std([r[out_key] for r in results if r["reference"] == lo]),
                np.std([r[out_key] for r in results if r["reference"] == hi]))
    gap = hi_out - lo_out
    verdict = ("FOLLOWS the reference" if abs(gap) > 2 * noise and
               np.sign(gap) == np.sign(clips[chosen[hi]][ref_key] - clips[chosen[lo]][ref_key])
               else "does NOT follow (inside seed noise)")
    print(f"  {axis:18s} {lo} {clips[chosen[lo]][ref_key]:7.2f} → out {lo_out:7.2f} | "
          f"{hi} {clips[chosen[hi]][ref_key]:7.2f} → out {hi_out:7.2f} | "
          f"gap {gap:+6.2f} vs noise ±{noise:.2f} → {verdict}")
print("\n  listening: tmp/delivery/take_<reference>_<seed>.wav")
