"""What does the CustomVoice capability actually buy?

Three ways to get "a voice" for the same three lines, three seeds each:

  preset   CustomVoice, one speaker from the checkpoint's spk_id table (no clip, no description)
  design   VoiceDesign, one fixed text description of a voice
  clone    Clone from a generated take (the 'ship fox as a generated voice' route)

Two questions:
  1. identity stability — mean pairwise cosine between outputs of the *same* source.
     If the same description does not give the same voice, reuse means re-rolling.
  2. bandwidth — energy above 9kHz. A voice that has already been through the vocoder
     once (clone-from-take) should be thinner than one that has not (preset).
"""
import itertools
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

import app as webui
from scipy.signal import butter, sosfilt

LINES = [
    "The build finished cleanly and every test in the suite passed on the first run.",
    "I left the notes on the kitchen table, along with the receipt and the spare key.",
    "She said the harbour was quietest just before dawn, when even the gulls are still.",
]
SEEDS = (11, 22, 33)
DESIGN = ("A calm adult female voice, mid thirties, warm and slightly husky, unhurried "
          "delivery, neutral American accent, a faint smile in the tone.")
TAKE_TEXT = ("The quick brown fox jumps over the lazy dog near the riverbank, then "
             "walks home along the shore.")


def hf_ratio(wav, sr, cutoff=9000):
    x = np.asarray(wav, dtype=np.float32)
    if x.size < 64:
        return 0.0
    sos = butter(4, cutoff / (sr / 2.0), btype="high", output="sos")
    high = float(np.sqrt(np.mean(sosfilt(sos, x) ** 2)))
    full = float(np.sqrt(np.mean(x ** 2)))
    return high / full if full > 0 else 0.0


hub = webui.ModelHub("models", torch.bfloat16, "sdpa", 1, 3, 4096, 1024)
webui.HUB = hub
webui.ASR = webui.Transcriber()

results = {}


def emb(wav, sr):
    """The speaker encoder only exists on the Base checkpoint, so this goes through it."""
    with hub.lock:
        model = hub.acquire("voice_clone").model
        return model.extract_speaker_embedding(
            audio=webui._to_24k(np.asarray(wav, dtype=np.float32), int(sr)),
            sr=24000).float().cpu().numpy()


with hub.lock:
    cv = hub.acquire("custom_voice")

    def sample(capability, gen):
        out = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            with hub.lock:
                model = hub.acquire(capability)
                wav, sr = gen(model, seed)
            out.append((np.asarray(wav, dtype=np.float32).reshape(-1), int(sr)))
        return out

    def report(name, clips):
        embeddings = [emb(w, s) for w, s in clips]
        pairs = [webui._cosine(a, b) for a, b in itertools.combinations(embeddings, 2)]
        wers, hfs = [], []
        for (w, s), line in zip(clips, LINES):
            wers.append(webui.word_error_rate(webui.ASR.transcribe(w, s, "English"), line))
            hfs.append(hf_ratio(w, s))
        results[name] = {"identity": float(np.mean(pairs)), "worst_pair": float(np.min(pairs)),
                         "hf": float(np.mean(hfs)), "wer": float(np.mean(wers)),
                         "seconds": float(np.mean([len(w) / s for w, s in clips]))}
        print(f"  {name:14s} identity {np.mean(pairs):.3f} (worst pair {np.min(pairs):.3f}) | "
              f"hf>9k {np.mean(hfs):.3f} | wer {np.mean(wers):.2f}")

    print("preset (CustomVoice / ryan):")
    report("preset", sample("custom_voice", lambda model, seed: model.generate_custom_voice(
        text=LINES[SEEDS.index(seed)], language="English", speaker="ryan", instruct=None,
        non_streaming_mode=True, max_new_tokens=webui._budget(LINES[SEEDS.index(seed)]),
        **webui._sampling(0.9, 1.0, 50, 1.05))))

    print("design (VoiceDesign / one description):")
    report("design", sample("voice_design", lambda model, seed: model.generate_voice_design(
        text=LINES[SEEDS.index(seed)], language="English", instruct=DESIGN,
        max_new_tokens=webui._budget(LINES[SEEDS.index(seed)]),
        **webui._sampling(0.9, 1.0, 50, 1.05))))

print("clone (from a generated take — the 'ship fox generated' route):")
ref_wav, ref_sr = sf.read("tmp/media/best_of_n_ref.wav", dtype="float32")
if ref_wav.ndim > 1:
    ref_wav = ref_wav.mean(axis=1)
REF_TEXT = ("Okay. Yeah. I resent you. I love you. I respect you. "
            "But you know what? You blew it! And thanks to you.")
with hub.lock:
    base = hub.acquire("voice_clone")
    prompt_ref = base.create_voice_clone_prompt(ref_audio=(ref_wav, ref_sr), ref_text=REF_TEXT,
                                                x_vector_only_mode=False)
    torch.manual_seed(33)
    take, take_sr = base.generate_voice_clone(
        text=TAKE_TEXT, language="English", voice_clone_prompt=prompt_ref,
        max_new_tokens=webui._budget(TAKE_TEXT), **webui._sampling(0.9, 1.0, 50, 1.05))
    take = np.asarray(take[0], dtype=np.float32).reshape(-1)
    prompt_take = base.create_voice_clone_prompt(ref_audio=(take, take_sr), ref_text=TAKE_TEXT,
                                                 x_vector_only_mode=False)

    def clone_gen(model, seed, prompt):
        torch.manual_seed(seed)
        wavs, rate = model.generate_voice_clone(
            text=LINES[SEEDS.index(seed)], language="English", voice_clone_prompt=prompt,
            max_new_tokens=webui._budget(LINES[SEEDS.index(seed)]),
            **webui._sampling(0.9, 1.0, 50, 1.05))
        return wavs[0], rate

    report("clone-take", sample("voice_clone", lambda model, seed: clone_gen(model, seed, prompt_take)))

print("\n=== which voice source is reusable? ===")
for name, r in results.items():
    print(f"{name:12s} same-source identity {r['identity']:.3f} | above-9kHz energy "
          f"{r['hf']:.3f} | wer {r['wer']:.2f} | {r['seconds']:.1f}s avg")
