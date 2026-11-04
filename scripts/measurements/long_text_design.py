"""Re-baseline long_text for the voice that actually ships.

The pacing/WER bounds in smoke_test.run_long_text were measured on the CustomVoice 'ryan'
preset, which this app no longer ships. Same instrument (Whisper over hard 20s windows on a
sentence repeated ten times), same text, single pass vs the chunked path — so the bound can
be set from what the shipped route does, not from what the removed one used to do.
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
sys.path.insert(0, os.path.join(ROOT, "tests"))
os.chdir(ROOT)

import numpy as np
import soundfile as sf
import torch

import app as webui
import smoke_test as st

DESC = "A clear, steady adult voice, neutral American English, unhurried."
cfg = st.CASES["long_text"]
text = " ".join([cfg["sentence"]] * cfg["repeat"])
words = len(text.split())

hub = webui.ModelHub("models", torch.bfloat16, "sdpa", 1, 2, 4096, 1024)
webui.HUB = hub
webui.ASR = webui.Transcriber()


def wer_of(wav, sr):
    step = int(20 * sr)
    hypotheses = []
    for start in range(0, len(wav), step):
        clip = wav[start:start + step]
        if len(clip) < sr // 2:
            continue
        hypotheses.append(webui.ASR.transcribe(clip, sr, "English"))
    return webui.word_error_rate(" ".join(hypotheses), text)


results = {}

# single pass: one call over the whole 300 words, the way the pre-chunking app did it
with hub.lock:
    tts = hub.acquire("voice_design")
    torch.manual_seed(42)
    wavs, sr = tts.generate_voice_design(text=text, language="English", instruct=DESC,
                                         non_streaming_mode=True, max_new_tokens=2048,
                                         **webui._sampling(0.9, 1.0, 50, 1.05))
single = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
results["single pass"] = (single, int(sr))

# chunked: the callback, which splits and stitches automatically
audio, status = webui.generate_voice_design(text, "English", DESC, 0.9, 1.0, 50, 1.05,
                                            postproc="off")
chunked = np.asarray(audio[1], dtype=np.float32).reshape(-1)
results["chunked"] = (chunked, int(audio[0]))

for name, (wav, sr) in results.items():
    seconds = len(wav) / sr
    print(f"{name:12s} {seconds:5.1f}s audio | {words / seconds:4.2f} words/s | "
          f"WER {wer_of(wav, sr):.2f}")
print("status said:", status.splitlines()[-1] if status else "")
