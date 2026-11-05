"""Transcribe the S2-Pro tagged takes with our own Whisper.

Why this is not optional: [whisper] produced a 9 dB drop, a 57% lower peak, a 62 Hz lower
pitch centre, and a *63% brighter* spectrum. Brighter is what turbulent, breathy airflow does
to a recording — which is exactly right for a whisper, but it is also exactly what noise looks
like. So the flattering reading of those numbers is unproven until someone confirms the clip
still says the sentence. A model that produces convincing breathiness over garbled words would
score beautifully on every metric above and be useless.
"""
# Prerequisite, and the reason nobody runs this any more: it needs an 11GB
# fishaudio/s2-pro download plus a SEPARATE venv (their transformers==5.19 cannot
# coexist with our 4.57.3), with their api_server.py on :8080. The comparison is
# closed — ours sounded more accurate to the ear. See docs/quality.md,
# "Fish Audio S2-Pro: evaluated on this box, and not better", before rerunning.
import glob
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(ROOT, "webui")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "webui"))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

import app as webui  # noqa: E402

TARGET = "The results came back this morning."
hub = webui.ModelHub("models", torch.bfloat16, "sdpa", 1, 1, 4096, 1024)
webui.HUB = hub
asr = webui.Transcriber(model_id=webui.ASR_MODEL_ID)
webui.ASR = asr

print(f"target: {TARGET!r}\n")
for p in sorted(glob.glob("tmp/fish-out/tag_*.wav")):
    wav, sr = sf.read(p, dtype="float32")
    d = asr.transcribe_detailed(np.asarray(wav, dtype=np.float32).reshape(-1), sr, None)
    text = (d.get("text") or "").strip()
    wer = webui.word_error_rate(text, TARGET)
    name = os.path.basename(p).replace("tag_", "").replace(".wav", "")
    flag = "OK" if wer <= 0.15 else "WRONG WORDS"
    print(f"  {name:20s} WER {wer:.2f} conf {d.get('confidence', float('nan')):+.2f} "
          f"[{flag}]  “{text}”")
