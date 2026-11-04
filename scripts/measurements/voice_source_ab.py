"""Which makes the better voice to keep: a real reference clip, or a generated take?

Same identity, same target lines, same seeds. The only variable is what sits in the
reference slot:

  A  the real 8s clip + its transcript            (what 'Save to voice library' keeps)
  B a take the model produced from A, kept raw    (what 'Save selected take' keeps)

Scored on identity (cosine to the original clip), content (WER vs the intended line),
decodability (Whisper's log-prob of the intended text), defects, and a bandwidth proxy —
energy above 9kHz, which is where a clone-of-a-clone would be expected to thin out,
because the vocoder's own imaging gets cloned along with the voice.
"""
import os
import sys
import time

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
from scipy.signal import butter, sosfilt

import app as webui

REF_TEXT = ("Okay. Yeah. I resent you. I love you. I respect you. "
            "But you know what? You blew it! And thanks to you.")
TARGETS = [
    "The build finished cleanly and every test in the suite passed on the first run.",
    "I left the notes on the kitchen table, along with the receipt and the spare key.",
    "She said the harbour was quietest just before dawn, when even the gulls are still.",
]
SEEDS = (1234, 77)


def hf_ratio(wav, sr, cutoff=9000):
    """Energy above `cutoff` relative to total — a thin-sounding take scores lower."""
    x = np.asarray(wav, dtype=np.float32)
    if x.size < 64:
        return 0.0
    sos = butter(4, cutoff / (sr / 2.0), btype="high", output="sos")
    high = float(np.sqrt(np.mean(sosfilt(sos, x) ** 2)))
    full = float(np.sqrt(np.mean(x ** 2)))
    return high / full if full > 0 else 0.0


hub = webui.ModelHub("models", torch.bfloat16, "sdpa", 1, 2, 4096, 1024)
webui.HUB = hub
webui.ASR = webui.Transcriber()

ref_wav, ref_sr = sf.read("tmp/media/best_of_n_ref.wav", dtype="float32")
if ref_wav.ndim > 1:
    ref_wav = ref_wav.mean(axis=1)

with hub.lock:
    tts = hub.acquire("voice_clone")

    def embedding(wav, sr):
        return tts.model.extract_speaker_embedding(
            audio=webui._to_24k(np.asarray(wav, dtype=np.float32), int(sr)),
            sr=24000).float().cpu().numpy()

    anchor = embedding(ref_wav, ref_sr)

    def prompt_for(wav, sr, text):
        return tts.create_voice_clone_prompt(ref_audio=(wav, sr), ref_text=text,
                                             x_vector_only_mode=False)

    def clone(prompt, text, seed):
        torch.manual_seed(seed)
        wavs, rate = tts.generate_voice_clone(
            text=text, language="English", voice_clone_prompt=prompt,
            max_new_tokens=webui._budget(text),
            **webui._sampling(0.9, 1.0, 50, 1.05))
        return np.asarray(wavs[0], dtype=np.float32).reshape(-1), int(rate)

    # --- build B: the best of three takes of one neutral line, kept as the model made it ---
    prompt_a = prompt_for(ref_wav, ref_sr, REF_TEXT)
    best, best_penalty, best_wav, best_sr = None, None, None, None
    for seed in (11, 22, 33):
        wav, sr = clone(prompt_a, "The quick brown fox jumps over the lazy dog near the "
                                  "riverbank, then walks home along the shore.", seed)
        wer = webui.word_error_rate(webui.ASR.transcribe(wav, sr, "English"),
                                    "The quick brown fox jumps over the lazy dog near the "
                                    "riverbank, then walks home along the shore.")
        conf = webui.ASR.score_transcript(wav, sr,
                                          "The quick brown fox jumps over the lazy dog near "
                                          "the riverbank, then walks home along the shore.",
                                          "English")
        penalty = webui.candidate_penalty(wer, conf, webui.defect_report(wav, sr, "The quick brown fox jumps over the lazy dog near the riverbank, then walks home along the shore."))
        print(f"  building the take: seed {seed} -> {len(wav)/sr:.1f}s wer {wer:.2f} "
              f"conf {conf:+.2f} pen {penalty:.2f}")
        if best_penalty is None or penalty < best_penalty:
            best, best_penalty, best_wav, best_sr = seed, penalty, wav, sr
    take_text = ("The quick brown fox jumps over the lazy dog near the riverbank, then "
                 "walks home along the shore.")
    print(f"  kept take: seed {best}, {len(best_wav)/best_sr:.2f}s at {best_sr}Hz\n")
    prompt_b = prompt_for(best_wav, best_sr, take_text)

    rows = []
    for label, wav, sr, text in (("A real clip", ref_wav, ref_sr, REF_TEXT),
                                 ("B kept take", best_wav, best_sr, take_text)):
        prompt = prompt_for(wav, sr, text)
        for target in TARGETS:
            for seed in SEEDS:
                started = time.time()
                out, out_sr = clone(prompt, target, seed)
                hypothesis = webui.ASR.transcribe(out, out_sr, "English")
                wer = webui.word_error_rate(hypothesis, target)
                conf = webui.ASR.score_transcript(out, out_sr, target, "English")
                defects = webui.defect_report(out, out_sr, target)
                rows.append({
                    "source": label, "seed": seed,
                    "seconds": len(out) / out_sr,
                    "sim": webui._cosine(anchor, embedding(out, out_sr)),
                    "wer": wer, "conf": conf,
                    "hf": hf_ratio(out, out_sr),
                    "flags": ",".join(webui.defect_flags(defects)) or "-",
                    "wall": time.time() - started,
                })
                r = rows[-1]
                print(f"  {label} seed {seed} -> {r['seconds']:4.1f}s sim {r['sim']:.3f} "
                      f"wer {r['wer']:.2f} conf {r['conf']:+.2f} hf {r['hf']:.3f} "
                      f"[{r['flags']}]")

print("\n=== summary (mean ± spread across 6 generations each) ===")
for label in ("A real clip", "B kept take"):
    sel = [r for r in rows if r["source"] == label]
    def stat(key):
        values = np.array([r[key] for r in sel], dtype=np.float64)
        return f"{values.mean():.3f} (max-min {values.max() - values.min():.3f})"
    flagged = sum(1 for r in sel if r["flags"] != "-")
    print(f"{label}: sim {stat('sim')} | wer {stat('wer')} | conf {stat('conf')} | "
          f"hf>9k {stat('hf')} | flagged {flagged}/{len(sel)}")
