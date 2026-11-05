"""S2-Pro's side of the clone comparison.

Same reference WAVs and same target sentences the Qwen path will get, so the only difference
between the two columns is the model. Two reference lengths because Fish advertises 10-30 s and
this app has always asked for ~6 s — if the long reference is clearly better, that reflects on
our default as much as on their model.
"""
# Prerequisite, and the reason nobody runs this any more: it needs an 11GB
# fishaudio/s2-pro download plus a SEPARATE venv (their transformers==5.19 cannot
# coexist with our 4.57.3), with their api_server.py on :8080. The comparison is
# closed — ours sounded more accurate to the ear. See docs/quality.md,
# "Fish Audio S2-Pro: evaluated on this box, and not better", before rerunning.
import json
import os
import time
from pathlib import Path

import numpy as np
import ormsgpack
import requests
import soundfile as sf

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
while not (ROOT / "webui").is_dir():
    ROOT = ROOT.parent
AB = ROOT / "tmp" / "clone_ab"
URL = "http://127.0.0.1:8080/v1/tts"

SENTENCES = [("ja", "その話は今朝聞いて、みんな少し驚いていました。"),
             ("en", "The results came back this morning.")]
SEEDS = [1234, 1235]

refs = json.loads((AB / "refs.json").read_text(encoding="utf-8"))
rows = []
for rname in ("short", "long"):
    meta = refs[rname]
    reference = {"audio": Path(meta["path"]).read_bytes(), "text": meta["text"]}
    print(f"ref_{rname}: {meta['seconds']}s, transcript {len(meta['text'])} chars")
    for slang, text in SENTENCES:
        for seed in SEEDS:
            payload = {"text": text, "references": [reference], "format": "wav", "seed": seed}
            t0 = time.time()
            r = requests.post(URL, data=ormsgpack.packb(payload),
                              headers={"Content-Type": "application/msgpack"}, timeout=900)
            if r.status_code != 200:
                print(f"  {rname}/{slang}/seed{seed}: HTTP {r.status_code} {r.text[:200]}")
                continue
            p = AB / f"fish_{rname}_{slang}_{seed}.wav"
            p.write_bytes(r.content)
            wav, sr = sf.read(p, dtype="float32")
            wav = np.asarray(wav).reshape(-1)
            rows.append({"model": "s2-pro", "ref": rname, "lang": slang, "seed": seed,
                         "file": p.name, "seconds": round(len(wav) / sr, 2), "sr": int(sr),
                         "wall": round(time.time() - t0, 1)})
            print(f"  {rname}/{slang}/seed{seed}: {rows[-1]['seconds']:5.2f}s audio "
                  f"in {rows[-1]['wall']:5.1f}s wall")

json.dump(rows, open(AB / "fish_takes.json", "w", encoding="utf-8"), indent=1)
print(f"\n{len(rows)} takes written to tmp/clone_ab/")
