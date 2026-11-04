"""Two things the first probe could not answer.

1. Every confidence above was measured on *clean* clips. A blocking threshold set from clean
   data can refuse a legitimate hand-recorded reference, so re-measure the same transcripts
   with broadband noise at SNR 10 dB and 5 dB — the condition the README already says costs
   the most similarity — and see how far the "exact" band moves down.
2. The tier printed Chat's first audio at 2.2s; a cold-hub probe printed 7.1s for the same
   piece. Attribute it: same process, hub already on the GPU, render cache disabled, and run
   the same reply with quick_start on and off. Then the only difference is the first piece's
   length.
"""
import json
import os
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

hub = webui.ModelHub("models", torch.bfloat16, "sdpa", 1, 2, 4096, 1024)
webui.HUB = hub
webui.ASR = webui.Transcriber()

rng = np.random.default_rng(7)


def noisy(wav, sr, snr_db):
    signal = float(np.sqrt(np.mean(wav ** 2))) + 1e-9
    noise = rng.standard_normal(wav.shape) * (signal / (10 ** (snr_db / 20.0)))
    return np.clip(wav + noise, -1.0, 1.0).astype(np.float32)


# ------------------------------------------------------------------ 1. noisy references
print("\n=== exact-transcript confidence as the reference gets noisier ===")
targets = [("fox", "voices/fox.wav", json.load(open("voices/fox.json"))["text"]),
           ("test", "voices/test.wav", json.load(open("voices/test.json"))["text"])]
best_ref = os.path.join(webui.media_dir(), "best_of_n_ref.wav")
targets.append(("clone_ref", best_ref, st.CASES["best_of_n"]["ref_text"]))

for name, path, text in targets:
    if not os.path.exists(path):
        continue
    wav, sr = sf.read(path, dtype="float32")
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    line = []
    for snr in (None, 10, 5):
        clip = wav if snr is None else noisy(wav, sr, snr)
        result = webui.check_reference_transcript(clip, sr, text, may_load=True)
        conf = None if not result else result.get("confidence")
        line.append(f"{'clean' if snr is None else f'SNR{snr}'}={conf:+.2f}"
                    if conf is not None else f"{'clean' if snr is None else f'SNR{snr}'}=n/a")
    print(f"  {name:10s} " + "  ".join(line))

# ------------------------------------------------------------------ 2. quick_start, isolated
print("\n=== Chat first audio: quick_start on vs off, warm hub, cache off ===")
text = ("The build finished cleanly and the test suite is green. "
        "I rewrote the chunker so long text is spoken at a natural pace instead of rushing. "
        "The seam between pieces uses a short equal-power crossfade, so there is no click. "
        "Tell me if it still sounds hurried anywhere and I will widen the band.")

webui.RENDER_CACHE_MB = 0                      # measure generation, not replay
ref_wav, ref_sr = sf.read(best_ref, dtype="float32")
ref_wav = np.asarray(ref_wav, dtype=np.float32).reshape(-1)
tmp_voices = os.path.join(ROOT, "tests", "output", "probe_voices2")
import shutil
shutil.rmtree(tmp_voices, ignore_errors=True)
os.makedirs(tmp_voices, exist_ok=True)
original_voices, webui.VOICES_DIR = webui.VOICES_DIR, tmp_voices
webui.save_voice(name="Probe Narrator", wav=ref_wav, sr=ref_sr,
                 text=st.CASES["best_of_n"]["ref_text"], instruct="", language="English")

with hub.lock:                                  # pay the promotion once, before timing
    hub.acquire("voice_clone")

for quick in (True, False):
    synth, mode = webui._voice_synth("Probe Narrator", "English", 0.9, 1.0, 50, 1.05)
    started = time.time()
    first = None
    pieces = []
    for piece in webui.iter_chunked(text, "English", synth, None,
                                    cache=getattr(synth, "render_cache", None),
                                    quick_start=quick):
        if first is None:
            first = time.time() - started
        pieces.append((len(piece["wav"]) / piece["sr"], len(piece["text"].split())))
    total = time.time() - started
    print(f"  quick_start={str(quick):5s} | first audio {first:5.2f}s | total {total:5.1f}s | "
          f"{len(pieces)} pieces | first piece {pieces[0][1]}w / {pieces[0][0]:.2f}s audio")

webui.VOICES_DIR = original_voices
