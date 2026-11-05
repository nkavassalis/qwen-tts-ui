"""Build the two references both models will be cloned from.

The comparison is only fair if both models are handed exactly the same conditioning, so the
references are cut once, transcribed once, and written to disk as WAVs that both generators
read. Two lengths on purpose: ~6 s, which is what this app has always asked for
(REF_TARGET_SECONDS), and ~20 s, which is what Fish says their model actually wants (10-30 s).
If the long reference is clearly better, that is a finding about our own default, not just
about their model.
"""
# Prerequisite, and the reason nobody runs this any more: it needs an 11GB
# fishaudio/s2-pro download plus a SEPARATE venv (their transformers==5.19 cannot
# coexist with our 4.57.3), with their api_server.py on :8080. The comparison is
# closed — ours sounded more accurate to the ear. See docs/quality.md,
# "Fish Audio S2-Pro: evaluated on this box, and not better", before rerunning.
import importlib.util
import json
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

spec = importlib.util.spec_from_file_location("mcl", os.path.join(ROOT, "scripts", "multi_clip_lab.py"))
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)

SOURCE = os.path.join(ROOT, "tmp", "delivery", "source.mov")
OUT = os.path.join(ROOT, "tmp", "clone_ab")
os.makedirs(OUT, exist_ok=True)

hub = webui.ModelHub("models", torch.bfloat16, "sdpa", 1, 1, 4096, 1024)
webui.HUB = hub
asr = webui.Transcriber(model_id=webui.ASR_MODEL_ID)
webui.ASR = asr

clips = lab.clips_from_long(SOURCE, True)
clips = [c for c in clips if len(c["wav"]) / c["sr"] >= webui.REF_MIN_SECONDS]
print(f"{len(clips)} candidate segments from the long recording")

short = clips[0]
n_long = max(3, int(20.0 / max(len(short["wav"]) / short["sr"], 1e-6)))
# concat_clips returns (wav, joined_text, clips_kept) — the third value is a COUNT, not a
# sample rate. Passing it as `sr` once produced a 149,760-second WAV and Whisper's feature
# extractor segfaulted on it rather than raising, so the length is asserted before anything
# downstream ever sees the array.
long_wav, _long_text, clips_kept = lab.concat_clips(
    [c["wav"] for c in clips[:n_long]], [""] * n_long, clips[0]["sr"])
sr = clips[0]["sr"]
print(f"long reference joined {clips_kept} segments")

refs = {}
for tag, wav in (("short", short["wav"]), ("long", long_wav)):
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    seconds = len(wav) / float(sr)
    if not (0.5 <= seconds <= 30.0) or not np.isfinite(wav).all():
        raise RuntimeError(f"ref_{tag} is nonsense: {seconds}s, finite={np.isfinite(wav).all()} "
                           f"(sr={sr}) — refusing to hand that to Whisper")
    d = asr.transcribe_detailed(wav, sr, None)
    text = (d.get("text") or "").strip()
    p = os.path.join(OUT, f"ref_{tag}.wav")
    sf.write(p, wav, sr)
    refs[tag] = {"path": p, "seconds": round(seconds, 2), "sr": int(sr),
                 "text": text, "asr_confidence": d.get("confidence")}
    print(f"  ref_{tag}: {seconds:5.2f}s  asr conf {d.get('confidence', float('nan')):+.2f}")
    print(f"           “{text[:110]}{'…' if len(text) > 110 else ''}”")

json.dump(refs, open(os.path.join(OUT, "refs.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(f"\nwrote {OUT}/ref_short.wav, ref_long.wav, refs.json")
print("note: the ~20s reference is several consecutive segments of one speaker joined with "
      "0.3s gaps — same method the lab's concat arm used.")
