"""Do S2-Pro's inline [tags] do what their words say?

This is the same question we asked Qwen3-TTS and answered "no": there, an `instruct` field
reached the model and changed the audio, but not in the direction the words described — a
control that appears to respond. S2-Pro is a different bet, and the code says so: only
`<|speaker:N|>` is parsed. `[whisper]`, `[screaming]`, `[sad]` are *not* special-cased
anywhere in the inference engine — they go into the prompt as ordinary text. So if they work,
they work because the model was trained to condition on them, which is the same reason an LLM
follows a word. That is a real mechanism rather than a bolted-on embedding.

Instrument, same as before: one reference (fox, a generated recipe voice — no personal audio),
one fixed sentence, fixed sampling, two seeds per condition. The decisive pair is [whisper]
against [screaming]: the predicted difference in level is enormous, so if *that* pair does not
separate by far more than the seed-to-seed noise floor, the tags are decoration and we say so.
[sad] vs [excited] is the second pair, on pitch and pace.
"""
# Prerequisite, and the reason nobody runs this any more: it needs an 11GB
# fishaudio/s2-pro download plus a SEPARATE venv (their transformers==5.19 cannot
# coexist with our 4.57.3), with their api_server.py on :8080. The comparison is
# closed — ours sounded more accurate to the ear. See docs/quality.md,
# "Fish Audio S2-Pro: evaluated on this box, and not better", before rerunning.
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import ormsgpack
import requests
import soundfile as sf

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
while not (ROOT / "webui").is_dir():
    ROOT = ROOT.parent
OUT = ROOT / "tmp" / "fish-out"
OUT.mkdir(exist_ok=True)
URL = "http://127.0.0.1:8080/v1/tts"

TARGET = "The results came back this morning."
CONDITIONS = [
    ("baseline", TARGET),
    ("whisper", f"[whisper] {TARGET}"),
    ("screaming", f"[screaming] {TARGET}"),
    ("sad", f"[sad] {TARGET}"),
    ("excited", f"[excited] {TARGET}"),
]
SEEDS = [1234, 1235]

ref_path = ROOT / "voices" / "fox.wav"
ref_text = json.loads((ROOT / "voices" / "fox.json").read_text())["text"]
reference = {"audio": ref_path.read_bytes(), "text": ref_text}
print(f"reference: fox.wav | target: {TARGET!r} | {len(CONDITIONS)} conditions × {len(SEEDS)} seeds")

import librosa  # noqa: E402


def measure(path):
    wav, sr = sf.read(path, dtype="float32")
    wav = np.asarray(wav).reshape(-1)
    dur = len(wav) / float(sr)
    rms = float(np.sqrt(np.mean(wav ** 2))) if wav.size else 0.0
    f0, _, _ = librosa.pyin(wav, fmin=60, fmax=600, sr=sr)
    v = f0[~np.isnan(f0)] if f0 is not None and len(f0) else np.array([0.0])
    spec = np.abs(np.fft.rfft(wav * np.hanning(wav.size)))
    freqs = np.fft.rfftfreq(wav.size, 1.0 / sr)
    return {
        "seconds": round(dur, 2), "rate": round(len(TARGET) / max(dur, 1e-6), 1),
        "rms": round(rms, 4), "level_db": round(20 * np.log10(rms + 1e-9), 1),
        "peak": round(float(np.max(np.abs(wav))) if wav.size else 0.0, 3),
        "f0_mean": round(float(np.mean(v)), 1), "f0_std": round(float(np.std(v)), 1),
        "centroid_hz": int((spec * freqs).sum() / (spec.sum() + 1e-12)),
    }


rows = []
for label, text in CONDITIONS:
    for seed in SEEDS:
        payload = {"text": text, "references": [reference], "format": "wav", "seed": seed}
        t0 = time.time()
        r = requests.post(URL, data=ormsgpack.packb(payload),
                          headers={"Content-Type": "application/msgpack"}, timeout=900)
        if r.status_code != 200:
            print(f"  {label} seed {seed}: HTTP {r.status_code} {r.text[:200]}")
            continue
        p = OUT / f"tag_{label}_{seed}.wav"
        p.write_bytes(r.content)
        m = measure(p)
        m.update({"condition": label, "seed": seed, "secs_wall": round(time.time() - t0, 1)})
        rows.append(m)
        print(f"  {label:10s} seed {seed}: {m['seconds']:5.2f}s audio in {m['secs_wall']:5.1f}s | "
              f"level {m['level_db']:6.1f}dB peak {m['peak']:.2f} "
              f"f0 {m['f0_mean']:6.1f}±{m['f0_std']:4.1f} centroid {m['centroid_hz']}Hz")

json.dump(rows, open(OUT / "tag_results.json", "w"), indent=1)

print("\n=== per condition (mean over seeds) ===")
cols = ("seconds", "rate", "level_db", "peak", "f0_mean", "f0_std", "centroid_hz")
print(f"{'condition':11s}" + "".join(f"{c:>13s}" for c in cols))
means = {}
for label, _ in CONDITIONS:
    sel = [r for r in rows if r["condition"] == label]
    if not sel:
        continue
    means[label] = {c: float(np.mean([r[c] for r in sel])) for c in cols}
    spread = {c: float(np.std([r[c] for r in sel])) for c in cols}
    print(f"{label:11s}" + "".join(f"{means[label][c]:9.2f}±{spread[c]:3.1f}" for c in cols))

print("\n=== the two pairs that decide it ===")
for lo, hi, key, expect in (("whisper", "screaming", "level_db", "louder"),
                            ("whisper", "screaming", "peak", "louder"),
                            ("sad", "excited", "f0_mean", "higher"),
                            ("sad", "excited", "rate", "faster")):
    if lo not in means or hi not in means:
        continue
    noise = max(float(np.std([r[key] for r in rows if r["condition"] == lo])),
                float(np.std([r[key] for r in rows if r["condition"] == hi])))
    gap = means[hi][key] - means[lo][key]
    ok = abs(gap) > 2 * noise and abs(gap) > 0.05 * max(abs(means[hi][key]), 1e-6)
    print(f"  {key:11s} {lo}→{hi}: {gap:+7.2f} (noise ±{noise:.2f}) "
          f"{'MOVES AS SAID' if ok else 'inside noise — tag inert'}")
print("\n  listening: tmp/fish-out/tag_<condition>_<seed>.wav")
