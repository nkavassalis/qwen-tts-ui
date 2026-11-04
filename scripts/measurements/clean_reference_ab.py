"""A/B the new reference cleanup on the GPU box, which is where it was never measured.

`prepare_reference` (DC removal, 60Hz high-pass, edge trim to 0.2s) ships behind a
`clean_reference` switch and the commit says plainly it was designed to be neutral-or-better,
not measured. This measures it: same reference, same target text, same seeds, three takes per
arm, identity scored by the model's own speaker encoder against the *raw* reference (so both
arms are compared to the same identity), content by Whisper against the requested line.
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

import app as webui

REF = os.path.join(webui.media_dir(), "best_of_n_ref.wav")
REF_TEXT = ("Okay. Yeah. I resent you. I love you. I respect you. But you know what? "
            "You blew it! And thanks to you.")
TARGET = "Cloning a voice should still pronounce every word clearly, even on a second pass."
N = 3

if not os.path.exists(REF):
    raise SystemExit(f"missing {REF} — run the GPU tier once to fetch it")

ref_wav, ref_sr = sf.read(REF, dtype="float32")
ref_wav = np.asarray(ref_wav, dtype=np.float32).reshape(-1)

hub = webui.ModelHub("models", torch.bfloat16, "sdpa", 1, 2, 4096, 1024)
webui.HUB = hub
webui.ASR = webui.Transcriber()

with hub.lock:
    tts = hub.acquire("voice_clone")
    # identity target: the raw reference, identical for both arms
    anchor = tts.model.extract_speaker_embedding(audio=webui._to_24k(ref_wav, ref_sr),
                                                 sr=24000).float().cpu().numpy()

    for label, clean in (("raw", False), ("cleaned", True)):
        wav = webui.prepare_reference(ref_wav, ref_sr) if clean else ref_wav
        prompt = webui._clone_prompt(tts, wav, ref_sr, REF_TEXT, False)
        ref_emb = tts.model.extract_speaker_embedding(audio=webui._to_24k(wav, ref_sr),
                                                      sr=24000).float().cpu().numpy()
        sims, wers, hf, lengths = [], [], [], []
        for i in range(N):
            torch.manual_seed(1000 + i)
            wavs, sr = tts.generate_voice_clone(text=TARGET, language="English",
                                                voice_clone_prompt=prompt,
                                                max_new_tokens=webui._budget(TARGET),
                                                **webui._sampling(0.9, 1.0, 50, 1.05))
            out = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
            cand = tts.model.extract_speaker_embedding(audio=webui._to_24k(out, int(sr)),
                                                       sr=24000).float().cpu().numpy()
            sims.append(float(np.dot(anchor.ravel(), cand.ravel()) /
                              (np.linalg.norm(anchor) * np.linalg.norm(cand) + 1e-9)))
            guess = webui.ASR.transcribe_detailed(out, int(sr), "English")
            wers.append(webui.word_error_rate(guess["text"], TARGET))
            spec = np.abs(np.fft.rfft(out * np.hanning(len(out))))
            freqs = np.fft.rfftfreq(len(out), 1.0 / int(sr))
            hf.append(float(spec[freqs > 9000].sum() / (spec.sum() + 1e-12)))
            lengths.append(len(out) / int(sr))
        print(f"{label:8s} SIM(vs raw) {np.mean(sims):.4f} ± {np.std(sims):.4f} | "
              f"WER {np.mean(wers):.3f} | HF>9k {np.mean(hf):.4f} | "
              f"{np.mean(lengths):.2f}s | ref SIM(vs raw) "
              f"{float(np.dot(anchor.ravel(), ref_emb.ravel()) / (np.linalg.norm(anchor) * np.linalg.norm(ref_emb) + 1e-9)):.4f} "
              f"| ref {len(wav) / ref_sr:.2f}s")
