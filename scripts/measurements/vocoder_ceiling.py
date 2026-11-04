"""What is the 12Hz codec's own ceiling?

Feed the speech tokenizer something with full, flat top-end — white noise — and see what
comes back out the other side. Whatever survives that is the most any model output can have,
because every path (preset, design, clone) ends in this decoder.
"""
import os
import sys

# Walk up to the project root instead of assuming a depth, so these stay runnable
# wherever they live under it.
ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(ROOT, "webui")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "webui"))
os.chdir(ROOT)

import numpy as np
import torch
from scipy.signal import welch

import app as webui

SR = 24000


def spectrum(wav, sr, label):
    x = np.asarray(wav, dtype=np.float32).reshape(-1)
    freqs, psd = welch(x, fs=sr, nperseg=min(2048, x.size))
    band = lambda lo, hi: float(np.sqrt(np.sum(psd[(freqs >= lo) & (freqs < hi)])))
    full = float(np.sqrt(np.sum(psd)))
    top = freqs[psd >= psd.max() * 10 ** (-60 / 20.0)]
    print(f"  {label:34s} hf(9-12k) {band(9000,12000)*1000:6.2f}m  "
          f"mid(3-6k) {band(3000,6000)*1000:6.2f}m  low(0-300) {band(0,300)*1000:6.2f}m  "
          f"top(-60dB) {(float(top.max()) if top.size else 0)/1000:4.1f}k  rms {full*1000:5.2f}m")


hub = webui.ModelHub("models", torch.bfloat16, "sdpa", 1, 2, 4096, 1024)
webui.HUB = hub

with hub.lock:
    model = hub.acquire("custom_voice")
    tok = model.model.speech_tokenizer

    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(SR * 2) * 0.15).astype(np.float32)
    chirp = np.sin(2 * np.pi * (200 * np.arange(SR) / SR + (5000 - 200) * np.arange(SR) ** 2
                                / (2 * SR * 1.0)) * 0.3).astype(np.float32)

    print("in → out of the codec (this is the ceiling every path shares):")
    spectrum(noise, SR, "white noise, before")
    back, fs = tok.decode([{"audio_codes": tok.encode([noise], sr=SR).audio_codes[0]}])
    spectrum(back[0], int(fs), "white noise, after codec")

    spectrum(chirp, SR, "200Hz→5kHz sweep, before")
    back, fs = tok.decode([{"audio_codes": tok.encode([chirp], sr=SR).audio_codes[0]}])
    spectrum(back[0], int(fs), "sweep, after codec")

    torch.manual_seed(11)
    wavs, sr = model.generate_custom_voice(
        text="The build finished cleanly and every test in the suite passed.",
        language="English", speaker="ryan", instruct=None, non_streaming_mode=True,
        max_new_tokens=webui._budget("The build finished cleanly and every test in the suite passed."),
        **webui._sampling(0.9, 1.0, 50, 1.05))
    preset = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
    spectrum(preset, int(sr), "preset speech, from the model")
    back, fs = tok.decode([{"audio_codes": tok.encode([preset], sr=int(sr)).audio_codes[0]}])
    spectrum(back[0], int(fs), "preset speech, codec round-trip")
