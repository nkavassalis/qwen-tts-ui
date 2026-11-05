"""Does the Base (voice-clone) checkpoint actually obey a style instruction?

The hybrid Nick wants is: clone the identity from a clip, and *direct the reading* per line.
Mechanically nothing stops it — generate_voice_clone forwards **kwargs to model.generate(),
which handles `voice_clone_prompt` (speaker embeds, line 2072) and `instruct_ids` (style
embeds, line 2076) as independent additions to the same talker input. The question is whether
the Base checkpoint was ever *trained* to listen to the second one. If it wasn't, the feature
would accept "read it skeptically" and return identical audio — a UI that lies, which is the
failure shape this project has been bitten by before (inert sampling sliders, tokenizer left
on CPU).

So: same reference, same target sentence, same seed, same sampling. Only the instruction
changes. A style-free control ("Read the sentence.") is what makes it an instrument — without
it, any difference between "no instruction" and "some instruction" could be the mere presence
of extra embeds rather than their meaning. The real test is whether two *contrasting*
instructions move the audio in the directions their words describe, and by more than the
control moves it.
"""
# Prerequisite: any voice in voices/ (it discovers one; none is named here). The
# verdict is closed — a style instruction on the clone path changes the audio without
# carrying any meaning. See docs/quality.md, "A style instruction on the clone path:
# accepted, obeyed by nothing".
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(ROOT, "webui")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "webui"))
os.chdir(ROOT)

import numpy as np
import soundfile as sf
import torch

import app as webui

TARGET = "The results came back this morning, and they were not what anyone expected."
SEEDS = [1234, 1235, 1236]
CONDITIONS = [
    ("none", None),
    ("neutral", "Read the sentence."),
    ("skeptical", "Read it in a skeptical, impatient tone, as if you have heard this "
                  "excuse before."),
    ("warm", "Read it in a warm, gentle, unhurried tone, as if comforting a friend."),
]

record = json.load(open("voices/fox.json", encoding="utf-8"))
ref_wav, ref_sr = sf.read("voices/fox.wav", dtype="float32")
ref_wav = np.asarray(ref_wav, dtype=np.float32).reshape(-1)
print(f"reference: fox (designed recipe), {len(ref_wav)/ref_sr:.2f}s | target: {TARGET}")

hub = webui.ModelHub("models", torch.bfloat16, "sdpa", 1, 1, 4096, 1024)
webui.HUB = hub

import librosa  # noqa: E402


def measures(wav, sr):
    dur = len(wav) / float(sr)
    rms = float(np.sqrt(np.mean(wav ** 2))) if wav.size else 0.0
    spec = np.abs(np.fft.rfft(wav * np.hanning(wav.size)))
    freqs = np.fft.rfftfreq(wav.size, 1.0 / sr)
    centroid = float((spec * freqs).sum() / (spec.sum() + 1e-12))
    hf = float(spec[freqs > 9000].sum() / (spec.sum() + 1e-12))
    f0, voiced, _ = librosa.pyin(wav, fmin=70, fmax=400, sr=sr)
    f0v = f0[~np.isnan(f0)] if f0 is not None and len(f0) else np.array([0.0])
    mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=13).mean(axis=1)
    return {"seconds": round(dur, 2), "rate": round(len(TARGET) / max(dur, 1e-6), 1),
            "rms": round(rms, 4), "centroid_hz": round(centroid), "hf_milli": round(hf * 1000, 2),
            "f0_mean": round(float(f0v.mean()), 1), "f0_std": round(float(f0v.std()), 1),
            "voiced_frac": round(float(np.mean(~np.isnan(f0))) if f0 is not None else 0.0, 2),
            "_mfcc": mfcc}


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


rows = {}
with hub.lock:
    tts = hub.acquire("voice_clone")
    prompt = [tts.create_voice_clone_prompt(ref_audio=(ref_wav, ref_sr),
                                            ref_text=record["text"],
                                            x_vector_only_mode=False)[0]]
    def ids_for(text):
        if text is None:
            return None
        return [tts._tokenize_texts([tts._build_instruct_text(text)])[0]]

    for name, instruction in CONDITIONS:
        ids = ids_for(instruction)
        for seed in SEEDS:
            torch.manual_seed(seed)
            kwargs = dict(text=TARGET, language="English", voice_clone_prompt=prompt,
                          max_new_tokens=webui._budget(TARGET),
                          **webui._sampling(**webui.SAMPLING_DEFAULTS))
            if ids is not None:
                kwargs["instruct_ids"] = ids
            wavs, sr = tts.generate_voice_clone(**kwargs)
            wav = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
            sf.write(os.path.join(ROOT, "tmp", f"instruct_{name}_{seed}.wav"), wav, sr)
            rows[(name, seed)] = measures(wav, sr)
            print(f"  {name:10s} seed {seed}: {rows[(name, seed)]['seconds']:5.2f}s "
                  f"rate {rows[(name, seed)]['rate']:5.1f} rms {rows[(name, seed)]['rms']:.4f} "
                  f"f0 {rows[(name, seed)]['f0_mean']:6.1f}±{rows[(name, seed)]['f0_std']:5.1f} "
                  f"centroid {rows[(name, seed)]['centroid_hz']:5d}Hz")

print("\n=== per condition, mean over seeds ===")
cols = ("seconds", "rate", "rms", "f0_mean", "f0_std", "centroid_hz", "hf_milli", "voiced_frac")
print(f"{'condition':11s}" + "".join(f"{c:>13s}" for c in cols))
means = {}
for name, _ in CONDITIONS:
    vals = {c: float(np.mean([rows[(name, s)][c] for s in SEEDS])) for c in cols}
    means[name] = vals
    print(f"{name:11s}" + "".join(f"{vals[c]:13.3f}" for c in cols))

print("\n=== the instrument's actual question ===")
base_mfcc = np.mean([rows[("none", s)]["_mfcc"] for s in SEEDS], axis=0)
for name, _ in CONDITIONS[1:]:
    m = np.mean([rows[(name, s)]["_mfcc"] for s in SEEDS], axis=0)
    print(f"  MFCC cosine distance of “{name}” from no-instruction: {1 - cos(base_mfcc, m):.4f}")
sk = np.mean([rows[("skeptical", s)]["_mfcc"] for s in SEEDS], axis=0)
wm = np.mean([rows[("warm", s)]["_mfcc"] for s in SEEDS], axis=0)
print(f"  MFCC cosine distance skeptical vs warm (the two that SHOULD differ): "
      f"{1 - cos(sk, wm):.4f}")
print(f"  seed-to-seed spread within 'none' (the noise floor): "
      f"{1 - cos(rows[('none', SEEDS[0])]['_mfcc'], rows[('none', SEEDS[1])]['_mfcc']):.4f}")
print("\n  listening files: tmp/instruct_<condition>_<seed>.wav")
