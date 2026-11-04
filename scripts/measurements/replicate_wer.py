"""Replicate the multi-clip WER finding, and separate audio quality from transcript accuracy.

The lab's first real run found the baseline (most typical) clip failed to say the sentence in
12 of 16 takes while the cleanest clip failed in 0 of 16. But "cleanest" is chosen by a score
that *includes transcript fit*, so the two candidate explanations were never separated:

    clean audio gets the words right        vs        an accurate transcript gets the words right

They imply opposite fixes (pick quieter clips vs fix the transcript), so this runs the 2x2:
each clip's audio with its own transcript and with the other clip's. Plus the three
several-clip versions, on three sentences instead of one, so the finding either replicates
outside the original sentence or dies. No listening; WER is scored by Whisper.
"""
import importlib.util
import json
import os
import shutil
import sys
import time
from typing import Any, Dict, List

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

spec = importlib.util.spec_from_file_location("mcl", os.path.join(ROOT, "scripts",
                                                                   "multi_clip_lab.py"))
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)
VoiceClonePromptItem = None  # imported from upstream below

SR = webui.REF_SAMPLE_RATE
OUT = os.path.join(ROOT, "tmp", "replication")
CLIPS = os.path.join(OUT, "clips")
TAKES = os.path.join(OUT, "takes")
shutil.rmtree(TAKES, ignore_errors=True)
os.makedirs(TAKES, exist_ok=True)

SENTENCES = [
    "ハッピーハロウィーン最後の投稿ですねハロウィンの時期ということも",       # the original
    "あけましておめでとうございます。今年もよろしくお願いします。",         # short, familiar
    ("昨日は雨が降っていたので、会議の資料を家に忘れてしまったことに気づかず、"
     "会社についてから慌ててメールを送りました。"),                          # long, complex
]
SEEDS = [1234, 1235, 1236, 1237]
LANGUAGE = "Japanese"

roles = json.load(open(os.path.join(OUT, "roles.json"), encoding="utf-8"))
names = sorted(f[:-4] for f in os.listdir(CLIPS) if f.endswith(".wav"))
clips: List[Dict[str, Any]] = []
for n in names:
    wav, sr = sf.read(os.path.join(CLIPS, n + ".wav"), dtype="float32")
    text = open(os.path.join(CLIPS, n + ".txt"), encoding="utf-8").read().strip()
    clips.append({"name": n, "wav": np.asarray(wav, dtype=np.float32).reshape(-1),
                  "sr": int(sr), "text": text})
base = clips[names.index(roles["baseline"])]
clean = clips[names.index(roles["cleanest"])]
print(f"clips: {len(clips)} | baseline {base['name']} ({len(base['wav'])/SR:.1f}s) | "
      f"cleanest {clean['name']} ({len(clean['wav'])/SR:.1f}s)")

hub = webui.ModelHub("models", torch.bfloat16, "sdpa", 1, 1, 4096, 1024)
webui.HUB = hub
asr = webui.Transcriber(model_id=webui.ASR_MODEL_ID)
webui.ASR = asr

from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem  # noqa: E402

results: List[Dict[str, Any]] = []
started = time.time()

with hub.lock:
    tts = hub.acquire("voice_clone")
    refs = {c["name"]: webui.prepare_reference(c["wav"], c["sr"]) for c in clips}
    items = {c["name"]: tts.create_voice_clone_prompt(
        ref_audio=(refs[c["name"]], SR), ref_text=c["text"], x_vector_only_mode=False)[0]
        for c in clips}
    by_name = {c["name"]: c for c in clips}

    # The build set exactly as the lab builds it: every clip but the held-out one, your clip
    # always included. Fold 0, because WER does not depend on which clip is held out.
    held, build = lab.make_folds(len(clips), 2, keep=names.index(roles["baseline"]))[0]
    embs = [items[clips[i]["name"]].ref_spk_embedding for i in build]
    mean_emb = torch.from_numpy(lab.mean_embedding(
        [e.float().cpu().numpy().reshape(-1) for e in embs])).reshape(
        embs[0].shape).to(device=embs[0].device, dtype=embs[0].dtype)
    joined, joined_text, used = lab.concat_clips([refs[clips[i]["name"]] for i in build],
                                                 [clips[i]["text"] for i in build], SR)
    print(f"build set: {len(build)} clips | joined {used} clips / "
          f"{len(joined)/SR:.1f}s | held out {clips[held]['name']}")

    conditions = {
        "single (baseline audio, own transcript)":
            lambda: items[base["name"]],
        "cleanest (clean audio, own transcript)":
            lambda: items[clean["name"]],
        "CROSSOVER baseline audio + CLEAN transcript":
            lambda: VoiceClonePromptItem(
                ref_code=items[base["name"]].ref_code,
                ref_spk_embedding=items[base["name"]].ref_spk_embedding,
                x_vector_only_mode=False, icl_mode=True, ref_text=clean["text"]),
        "CROSSOVER clean audio + BASELINE transcript":
            lambda: VoiceClonePromptItem(
                ref_code=items[clean["name"]].ref_code,
                ref_spk_embedding=items[clean["name"]].ref_spk_embedding,
                x_vector_only_mode=False, icl_mode=True, ref_text=base["text"]),
        "single_mean (baseline codes, averaged embedding)":
            lambda: VoiceClonePromptItem(
                ref_code=items[base["name"]].ref_code, ref_spk_embedding=mean_emb,
                x_vector_only_mode=False, icl_mode=True, ref_text=base["text"]),
        "concat (all clips joined)":
            lambda: tts.create_voice_clone_prompt(ref_audio=(joined, SR),
                                                  ref_text=joined_text,
                                                  x_vector_only_mode=False)[0],
        "xvec_mean (averaged embedding only)":
            lambda: VoiceClonePromptItem(ref_code=None, ref_spk_embedding=mean_emb,
                                         x_vector_only_mode=True, icl_mode=False,
                                         ref_text=None),
    }

    total = len(conditions) * len(SENTENCES) * len(SEEDS)
    done = 0
    for label, make in conditions.items():
        prompt = [make()]
        for s_index, text in enumerate(SENTENCES):
            for seed in SEEDS:
                torch.manual_seed(seed)
                wavs, rate = tts.generate_voice_clone(
                    text=text, language=LANGUAGE, voice_clone_prompt=prompt,
                    max_new_tokens=webui._budget(text),
                    **webui._sampling(**webui.SAMPLING_DEFAULTS))
                wav = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
                heard, conf = asr.assess(wav, int(rate), text, None)
                wer = float(webui.word_error_rate(heard, text))
                name = f"{done:03d}_{s_index}_{seed}.wav"
                sf.write(os.path.join(TAKES, name), wav, int(rate))
                results.append({"condition": label, "sentence": s_index, "seed": seed,
                                "wer": round(wer, 3), "seconds": round(len(wav) / rate, 2),
                                "heard": heard,
                                "hf_milli": lab.high_band_milli(wav, int(rate))})
                done += 1
        print(f"  [{done}/{total}] {label}  ({(time.time()-started)/60:.1f} min)")

json.dump(results, open(os.path.join(OUT, "results.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

import statistics as st

print("\n=== does the WER finding replicate on three sentences? ===")
print(f"{'condition':46s} {'n':>3s} {'bad':>4s} {'WER':>6s} {'conf':>6s} {'hf_m':>5s} {'sec':>5s}")
summary = {}
for label in conditions:
    rows = [r for r in results if r["condition"] == label]
    bad = sum(1 for r in rows if r["wer"] > 0.05)
    summary[label] = bad
    print(f"{label:46s} {len(rows):3d} {bad:4d} "
          f"{st.mean(r['wer'] for r in rows):6.3f} "
          f"{st.mean(r['confidence'] for r in rows if r['confidence'] is not None):6.2f} "
          f"{st.mean(r['hf_milli'] for r in rows):5.2f} "
          f"{st.mean(r['seconds'] for r in rows):5.1f}")

print("\n=== per sentence (bad takes out of 4 seeds) ===")
for s_index, text in enumerate(SENTENCES):
    line = []
    for label in conditions:
        bad = sum(1 for r in results if r["condition"] == label and r["sentence"] == s_index
                  and r["wer"] > 0.05)
        line.append(f"{label.split(' (')[0].split('CROSSOVER ')[-1][:14]}={bad}/4")
    print(f"  sentence {s_index}: " + "  ".join(line))
