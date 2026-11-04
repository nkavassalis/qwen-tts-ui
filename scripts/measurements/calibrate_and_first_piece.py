"""Calibration + first-piece probe for the render-cache / quick-start / transcript-check commit.

Two questions the GPU tier prints one data point for:

1. `media_pipeline` prints the reference-transcript confidence for one right and one wrong
   transcript. One pair cannot place a blocking threshold, so this sweeps four clips against
   four degrees of wrongness each — exact, one word off, half wrong, unrelated. The number
   that matters is not the gap between right and wrong (huge), it is where *mostly right*
   lands: that is what a blocking threshold must not catch.
2. Chat's first piece. Whether quick_start actually cut anything for this text, how long the
   first piece is, and whether it ends in a way that will sound clipped (energy right up to
   the last sample) or flat (trailing silence, punctuation at the cut).
"""
import glob
import json
import os
import shutil
import sys
import time

# Walk up to the project root instead of assuming a depth, so these stay runnable
# wherever they live under it.
ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(ROOT, "webui")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "webui"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
os.chdir(ROOT)

import numpy as np
import soundfile as sf
import torch

import app as webui
import smoke_test as st

OUT = os.path.join(ROOT, "tests", "output")
os.makedirs(OUT, exist_ok=True)

hub = webui.ModelHub("models", torch.bfloat16, "sdpa", 1, 2, 4096, 1024)
webui.HUB = hub
webui.ASR = webui.Transcriber()

# ---------------------------------------------------------------- 1. transcript check sweep
print("\n=== reference transcript confidence: right vs wrong ===")
ref_dir = webui.media_dir()
best_ref = os.path.join(ref_dir, "best_of_n_ref.wav")
if not os.path.exists(best_ref):
    import urllib.request
    os.makedirs(ref_dir, exist_ok=True)
    urllib.request.urlretrieve(st.CASES["best_of_n"]["ref_audio"], best_ref)

cases = []
if os.path.exists(best_ref):
    cases.append(("clone_ref", best_ref, st.CASES["best_of_n"]["ref_text"]))
# Every saved voice that has a clip and a transcript of its own, whatever they happen to be.
# The published numbers were four rows: this upstream reference plus three saved voices, one
# of them a Japanese clip (a script with no spaces, which is what exposes the CJK caveat in
# the printout below). That clip is someone's private recording and is not in the repo, so add
# your own to reproduce the fourth row.
for record in sorted(glob.glob("voices/*.json")):
    wav_path = record[:-5] + ".wav"
    try:
        text = json.load(open(record, encoding="utf-8")).get("text", "")
    except (OSError, ValueError):
        continue
    if text.strip() and os.path.exists(wav_path):
        cases.append((os.path.basename(record)[:-5], wav_path, text))

UNRELATED = ("The quarterly report was filed before the audit, and nobody read the "
             "appendix at all.")


def variants(text):
    words = text.split()
    one = list(words)
    if len(one) > 3:
        one[len(one) // 2] = "zebra"
    half = words[:len(words) // 2] + UNRELATED.split()[:len(words) - len(words) // 2]
    return [("exact", text), ("one word off", " ".join(one)),
            ("half wrong", " ".join(half)), ("unrelated", UNRELATED)]


rows = []
for name, path, text in cases:
    wav, sr = sf.read(path, dtype="float32")
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    line = []
    for label, guess in variants(text):
        result = webui.check_reference_transcript(wav, sr, guess, may_load=True)
        conf = None if not result else result.get("confidence")
        wer = None if not result else result.get("wer")
        line.append(f"{label}={conf:+.2f}/wer{wer:.2f}" if conf is not None else f"{label}=n/a")
        rows.append((name, label, conf, wer))
    print(f"  {name:10s} ({len(wav)/sr:.1f}s) " + "  ".join(line))

good = [c for n, l, c, w in rows if l == "exact" and c is not None]
mostly = [c for n, l, c, w in rows if l == "one word off" and c is not None]
bad = [c for n, l, c, w in rows if l in ("half wrong", "unrelated") and c is not None]
print(f"  exact:      min {min(good):+.2f}  max {max(good):+.2f}")
print(f"  one off:    min {min(mostly):+.2f}  max {max(mostly):+.2f}")
print(f"  wrong:      min {min(bad):+.2f}  max {max(bad):+.2f}")
print(f"  current REF_TEXT_CONF_WARN = {webui.REF_TEXT_CONF_WARN}")

# ---------------------------------------------------------------- 2. Chat's first piece
print("\n=== chat first piece ===")
text = ("The build finished cleanly and the test suite is green. "
        "I rewrote the chunker so long text is spoken at a natural pace instead of rushing. "
        "The seam between pieces uses a short equal-power crossfade, so there is no click. "
        "Tell me if it still sounds hurried anywhere and I will widen the band.")
long_first = ("I have finished the whole thing you asked me about this morning, including the "
              "tests, the documentation and the deployment, and everything passed on the first "
              "run without any complaints at all. The cache is warm now.")

for label, body in (("the smoke reply", text), ("a long first sentence", long_first)):
    plain = webui.split_for_tts(body, "English")
    quick = webui.split_for_tts(body, "English", quick_start=True)
    print(f"  [{label}] pieces plain={len(plain)} quick={len(quick)} | "
          f"first plain={len(plain[0].split())}w quick={len(quick[0].split())}w")
    print(f"      quick first piece: “{quick[0][:78]}”")

# render the smoke reply cold, and the long-first-sentence one, and look at piece 1
ref_wav, ref_sr = sf.read(best_ref, dtype="float32")
ref_wav = np.asarray(ref_wav, dtype=np.float32).reshape(-1)
webui.RENDER_CACHE_DIR = os.path.join(OUT, "probe_render_cache")
shutil.rmtree(webui.RENDER_CACHE_DIR, ignore_errors=True)
tmp_voices = os.path.join(OUT, "probe_voices")
shutil.rmtree(tmp_voices, ignore_errors=True)
os.makedirs(tmp_voices, exist_ok=True)
original_voices, webui.VOICES_DIR = webui.VOICES_DIR, tmp_voices
webui.save_voice(name="Probe Narrator", wav=ref_wav, sr=ref_sr,
                 text=st.CASES["best_of_n"]["ref_text"], instruct="", language="English")


def edges(wav, sr):
    """Silence at each end, and the RMS of the last 80ms (a mid-word cut stays loud)."""
    win = int(0.02 * sr)
    frames = webui._rms_windows(wav, sr, win)
    tail = float(np.sqrt(np.mean(wav[-int(0.08 * sr):] ** 2))) if wav.size > win else 0.0
    lead = float(np.sqrt(np.mean(wav[:int(0.08 * sr)] ** 2))) if wav.size > win else 0.0
    hf = 0.0
    if wav.size:
        spec = np.abs(np.fft.rfft(wav * np.hanning(wav.size)))
        freqs = np.fft.rfftfreq(wav.size, 1.0 / sr)
        hf = float(spec[freqs > 9000].sum() / (spec.sum() + 1e-12))
    return lead, tail, hf, len(frames)


for label, body in (("smoke reply", text), ("long first sentence", long_first)):
    synth, mode = webui._voice_synth("Probe Narrator", "English", 0.9, 1.0, 50, 1.05)
    started = time.time()
    stamps = []
    for piece in webui.iter_chunked(body, "English", synth, None,
                                    cache=getattr(synth, "render_cache", None), quick_start=True):
        stamps.append((time.time() - started, piece))
    print(f"  [{label}] first audio {stamps[0][0]:.2f}s | {len(stamps)} pieces | "
          f"total {stamps[-1][0]:.1f}s")
    for index, (at, piece) in enumerate(stamps):
        wav, sr = piece["wav"], piece["sr"]
        lead, tail, hf, _ = edges(wav, sr)
        path = os.path.join(OUT, f"first_piece_{index}_{label.split()[0]}.wav")
        sf.write(path, wav, sr)
        print(f"      #{index + 1} ready {at:5.2f}s | {len(wav)/sr:4.2f}s | "
              f"{len(piece['text'].split()):2d}w | lead {lead:.3f} tail80ms {tail:.3f} "
              f"HF>9k {hf:.4f} | cached={piece.get('cached')} | “{piece['text'][:44]}”")

webui.VOICES_DIR = original_voices
print(f"\n  pieces written to {OUT}/first_piece_*.wav")
