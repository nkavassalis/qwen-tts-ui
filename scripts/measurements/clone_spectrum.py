"""Is the clone path really missing high frequencies, or is my ratio metric lying?

Measures, per source, on the raw 24kHz model decode:
  hf_rms      absolute RMS of 9-12kHz   (a real loss of energy, not a relative effect)
  lf_rms      absolute RMS below 300Hz  (a louder bottom would fake a low hf *ratio*)
  centroid    energy-weighted mean frequency
  bw60        highest frequency still within 60dB of the PSD peak — the usable top end
And measures the reference clips themselves, because if cloning tracks the reference's
spectrum, the reference's own top end should predict the clone's.
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
import soundfile as sf
import torch
from scipy.signal import butter, sosfilt, welch

import app as webui

LINE = "The build finished cleanly and every test in the suite passed on the first run."
REF_TEXT = ("Okay. Yeah. I resent you. I love you. I respect you. "
            "But you know what? You blew it! And thanks to you.")
TAKE_TEXT = ("The quick brown fox jumps over the lazy dog near the riverbank, then "
             "walks home along the shore.")
DESIGN = ("A calm adult female voice, mid thirties, warm and slightly husky, unhurried "
          "delivery, neutral American accent, a faint smile in the tone.")


def spectrum(wav, sr):
    x = np.asarray(wav, dtype=np.float32).reshape(-1)
    if x.size < 256:
        return dict(hf=0.0, lf=0.0, centroid=0.0, bw60=0.0, rms=0.0)
    freqs, psd = welch(x, fs=sr, nperseg=min(2048, x.size))
    band = lambda lo, hi: float(np.sqrt(np.sum(psd[(freqs >= lo) & (freqs < hi)])))
    hf, lf, full = band(9000, 12000), band(0, 300), float(np.sqrt(np.sum(psd)))
    centroid = float(np.sum(freqs * psd) / np.sum(psd)) if psd.sum() > 0 else 0.0
    top = freqs[psd >= psd.max() * 10 ** (-60 / 20.0)]
    return dict(hf=hf, lf=lf, centroid=centroid, bw60=float(top.max()) if top.size else 0.0,
                rms=full)


def show(name, wav, sr):
    s = spectrum(wav, sr)
    ratio = s["hf"] / s["rms"] if s["rms"] else 0.0
    print(f"  {name:26s} hf_rms {s['hf']*1000:6.2f}m  lf_rms {s['lf']*1000:6.2f}m  "
          f"hf/rms {ratio:.3f}  centroid {s['centroid']/1000:4.2f}k  "
          f"top(-60dB) {s['bw60']/1000:4.1f}k  rms {s['rms']*1000:5.2f}m")
    return s


hub = webui.ModelHub("models", torch.bfloat16, "sdpa", 1, 3, 4096, 1024)
webui.HUB = hub

ref_wav, ref_sr = sf.read("tmp/media/best_of_n_ref.wav", dtype="float32")
if ref_wav.ndim > 1:
    ref_wav = ref_wav.mean(axis=1)

print("the sources themselves:")
show("real reference clip", ref_wav, ref_sr)

with hub.lock:
    base = hub.acquire("voice_clone")
    prompt_ref = base.create_voice_clone_prompt(ref_audio=(ref_wav, ref_sr), ref_text=REF_TEXT,
                                                x_vector_only_mode=False)
    torch.manual_seed(33)
    wavs, take_sr = base.generate_voice_clone(
        text=TAKE_TEXT, language="English", voice_clone_prompt=prompt_ref,
        max_new_tokens=webui._budget(TAKE_TEXT), **webui._sampling(0.9, 1.0, 50, 1.05))
    take = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
    prompt_take = base.create_voice_clone_prompt(ref_audio=(take, take_sr), ref_text=TAKE_TEXT,
                                                 x_vector_only_mode=False)
show("generated take (ref B)", take, take_sr)


def gen(capability, fn):
    out = []
    for seed in (11, 22):
        torch.manual_seed(seed)
        with hub.lock:
            model = hub.acquire(capability)
            wavs, sr = fn(model, seed)
        out.append((np.asarray(wavs[0], dtype=np.float32).reshape(-1), int(sr)))
    return out


def average(name, clips):
    rows = [spectrum(w, s) for w, s in clips]
    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    ratio = mean["hf"] / mean["rms"] if mean["rms"] else 0.0
    print(f"  {name:26s} hf_rms {mean['hf']*1000:6.2f}m  lf_rms {mean['lf']*1000:6.2f}m  "
          f"hf/rms {ratio:.3f}  centroid {mean['centroid']/1000:4.2f}k  "
          f"top(-60dB) {mean['bw60']/1000:4.1f}k  rms {mean['rms']*1000:5.2f}m")


print("\nthe four ways to produce speech (2 seeds each, raw 24kHz decode):")
average("preset (CustomVoice)", gen("custom_voice", lambda m, s: m.generate_custom_voice(
    text=LINE, language="English", speaker="ryan", instruct=None, non_streaming_mode=True,
    max_new_tokens=webui._budget(LINE), **webui._sampling(0.9, 1.0, 50, 1.05))))

average("design (VoiceDesign)", gen("voice_design", lambda m, s: m.generate_voice_design(
    text=LINE, language="English", instruct=DESIGN, max_new_tokens=webui._budget(LINE),
    **webui._sampling(0.9, 1.0, 50, 1.05))))

average("clone from real clip", gen("voice_clone", lambda m, s: m.generate_voice_clone(
    text=LINE, language="English", voice_clone_prompt=prompt_ref,
    max_new_tokens=webui._budget(LINE), **webui._sampling(0.9, 1.0, 50, 1.05))))

average("clone from generated take", gen("voice_clone", lambda m, s: m.generate_voice_clone(
    text=LINE, language="English", voice_clone_prompt=prompt_take,
    max_new_tokens=webui._budget(LINE), **webui._sampling(0.9, 1.0, 50, 1.05))))

# The same words spoken by the clone path, but decoded from the *preset* conditioning:
# does the vocoder itself cap the top end, independent of how the tokens were produced?
print("\ncodec round-trip (encode then decode a preset output — the vocoder's own ceiling):")
with hub.lock:
    model = hub.acquire("custom_voice")
    preset, preset_sr = model.generate_custom_voice(
        text=LINE, language="English", speaker="ryan", instruct=None, non_streaming_mode=True,
        max_new_tokens=webui._budget(LINE), **webui._sampling(0.9, 1.0, 50, 1.05))
    preset = np.asarray(preset[0], dtype=np.float32).reshape(-1)
show("preset, straight from model", preset, preset_sr)
try:
    tokenizer = model.speech_tokenizer
    encoded = tokenizer.encode((preset, preset_sr))
    back_list, back_sr = tokenizer.decode(encoded)
    back = np.asarray(back_list[0], dtype=np.float32).reshape(-1)
    show("preset, codec round-trip", back, int(back_sr))
except Exception as e:  # noqa: BLE001 - diagnostic only
    print(f"  (round-trip unavailable: {type(e).__name__}: {e})")
