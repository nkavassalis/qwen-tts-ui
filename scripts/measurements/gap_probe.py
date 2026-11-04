"""Where is the long quiet run: at a join, or inside a piece? (GPU probe)"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "webui"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import app as webui
import smoke_test as st
import torch

hub = webui.ModelHub("models", torch.bfloat16, "sdpa", 1, 1, 4096, 1024)
webui.HUB = hub
cfg = st.CASES["long_text"]
text = " ".join([cfg["sentence"]] * cfg["repeat"])

with hub.lock:
    tts = hub.acquire("custom_voice")

    def one(chunk):
        wavs, rate = tts.generate_custom_voice(
            text=chunk, language="English", speaker=webui.default_speaker(),
            instruct=None, non_streaming_mode=True, max_new_tokens=webui._budget(chunk),
            **webui._sampling(0.9, 1.0, 50, 1.05))
        return wavs[0], rate

    pieces, texts = [], []
    for piece in webui.iter_chunked(text, "English", one):
        pieces.append(piece["wav"]); texts.append(piece["text"])
        print(f"  piece {piece['index']+1}: {len(piece['wav'])/piece['sr']:.2f}s")
sr = 24000

# mirror stitch_clips, recording where each seam lands
trimmed = [webui.trim_edges(p, sr) for p in pieces]
fade = int(webui.CROSSFADE_MS / 1000.0 * sr)
out, joins = trimmed[0], []
for nxt in trimmed[1:]:
    overlap = min(fade, out.size // 2, nxt.size // 2)
    if overlap <= 0:
        out = np.concatenate([out, nxt]); continue
    ramp = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
    joined = out[-overlap:] * np.sqrt(1.0 - ramp) + nxt[:overlap] * np.sqrt(ramp)
    out = np.concatenate([out[:-overlap], joined, nxt[overlap:]])
    joins.append((out.size - nxt.size + overlap / 2) / sr)

windows = webui._rms_windows(out, sr, int(0.02 * sr))
best, gap, prev, best_start, start = 0, 0, None, 0, 0
for i in np.flatnonzero(windows < webui.SILENCE_RMS):
    if prev == i - 1:
        gap += 1
    else:
        gap, start = 1, i
    if gap > best:
        best, best_start = gap, start
    prev = i

at = best_start * 0.02
near = min(joins, key=lambda j: abs(j - at)) if joins else None
print(f"\naudio {out.size/sr:.1f}s | pieces {len(pieces)} | joins {len(joins)}")
print(f"longest quiet run: {best*0.02:.2f}s starting at {at:.2f}s")
print(f"nearest join: {near:.2f}s (distance {abs(near-at):.2f}s)")
print("join offsets:", [round(j, 2) for j in joins])
print("quiet runs > 0.5s:")
gap, prev, start = 0, None, 0
for i in np.flatnonzero(windows < webui.SILENCE_RMS):
    gap = gap + 1 if prev == i - 1 else 1
    if prev is not None and i != prev + 1 and gap * 0.02 > 0.5:
        print(f"   {start*0.02:.2f}s .. {i*0.02:.2f}s  ({gap*0.02:.2f}s)")
    if i == np.flatnonzero(windows < webui.SILENCE_RMS)[-1] and gap * 0.02 > 0.5:
        print(f"   {start*0.02:.2f}s .. end  ({gap*0.02:.2f}s)")
    prev = i
