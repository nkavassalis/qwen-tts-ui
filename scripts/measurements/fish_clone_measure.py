"""Qwen3-TTS's side of the clone comparison, then both sides measured alike.

Same reference WAVs, same sentences, same seeds as the S2-Pro run. Every take from both models
gets the same instruments: Whisper WER and teacher-forced confidence against the INTENDED text,
duration and speaking rate, level, the energy above 9 kHz (the bandwidth proxy this project has
been using), a noise-floor SNR, and cosine similarity to the reference from the Qwen speaker
encoder.

Two honest caveats printed with the result. The similarity encoder is Qwen's own, so it can be
expected to favour Qwen outputs — treat it as a floor on their side, not a neutral judge. And
the two models do not share a sample rate (S2-Pro decodes at 44.1 kHz, Qwen3-TTS at 24 kHz), so
"energy above 9 kHz" is comparable but the models do not have the same ceiling; above 11 kHz the
Qwen output has nothing left to give.
"""
# Prerequisite, and the reason nobody runs this any more: it needs an 11GB
# fishaudio/s2-pro download plus a SEPARATE venv (their transformers==5.19 cannot
# coexist with our 4.57.3), with their api_server.py on :8080. The comparison is
# closed — ours sounded more accurate to the ear. See docs/quality.md,
# "Fish Audio S2-Pro: evaluated on this box, and not better", before rerunning.
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

AB = os.path.join(ROOT, "tmp", "clone_ab")
SENTENCES = [("ja", "その話は今朝聞いて、みんな少し驚いていました。", "Japanese"),
             ("en", "The results came back this morning.", "English")]
SEEDS = [1234, 1235]

hub = webui.ModelHub("models", torch.bfloat16, "sdpa", 1, 1, 4096, 1024)
webui.HUB = hub
asr = webui.Transcriber(model_id=webui.ASR_MODEL_ID)
webui.ASR = asr
refs = json.load(open(os.path.join(AB, "refs.json"), encoding="utf-8"))


def band(wav, sr, lo):
    if not wav.size:
        return 0.0
    spec = np.abs(np.fft.rfft(wav * np.hanning(wav.size)))
    freqs = np.fft.rfftfreq(wav.size, 1.0 / sr)
    return float(spec[freqs > lo].sum() / (spec.sum() + 1e-12))


def snr_db(wav, sr):
    frames = webui._rms_windows(wav, sr, int(0.02 * sr))
    if not frames.size:
        return 0.0
    floor = float(np.percentile(frames, 10)) + 1e-9
    top = float(np.percentile(frames, 90)) + 1e-9
    return 20 * np.log10(top / floor)


def measure(path, intended):
    wav, sr = sf.read(path, dtype="float32")
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    d = asr.transcribe_detailed(wav, sr, None)
    text = (d.get("text") or "").strip()
    rms = float(np.sqrt(np.mean(wav ** 2))) if wav.size else 0.0
    return {"seconds": round(len(wav) / sr, 2), "sr": int(sr),
            "rate": round(len(intended) / max(len(wav) / sr, 1e-6), 1),
            "level_db": round(20 * np.log10(rms + 1e-9), 1),
            "hf_9k_milli": round(band(wav, sr, 9000) * 1000, 2),
            "hf_11k_milli": round(band(wav, sr, 11000) * 1000, 2),
            "snr_db": round(snr_db(wav, sr), 1),
            "wer": round(webui.word_error_rate(text, intended), 3),
            "conf": round(float(d.get("confidence") or 0.0), 2), "heard": text}


with hub.lock:
    tts = hub.acquire("voice_clone")

    # Qwen side (skipped if already rendered, so a re-run of the measurement is cheap)
    for rname in ("short", "long"):
        meta = refs[rname]
        wav, sr = sf.read(meta["path"], dtype="float32")
        ref = (np.asarray(wav, dtype=np.float32).reshape(-1), int(sr))
        for slang, text, lang in SENTENCES:
            for seed in SEEDS:
                p = os.path.join(AB, f"qwen_{rname}_{slang}_{seed}.wav")
                if os.path.exists(p):
                    continue
                torch.manual_seed(seed)
                wavs, out_sr = tts.generate_voice_clone(
                    text=text, language=lang, ref_audio=ref, ref_text=meta["text"],
                    x_vector_only_mode=False, max_new_tokens=webui._budget(text),
                    **webui._sampling(**webui.SAMPLING_DEFAULTS))
                out = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
                sf.write(p, out, out_sr)
                print(f"  qwen {rname}/{slang}/seed{seed}: {len(out)/out_sr:5.2f}s")

    # Similarity: one encoder, both models, so the comparison is at least symmetric.
    def embed(wav, sr):
        emb = tts.model.extract_speaker_embedding(audio=webui._to_24k(wav, sr), sr=24000)
        return emb.float().cpu().numpy().reshape(-1)

    ref_emb = {}
    for rname in ("short", "long"):
        wav, sr = sf.read(refs[rname]["path"], dtype="float32")
        ref_emb[rname] = embed(np.asarray(wav, dtype=np.float32).reshape(-1), sr)

rows = []
# The Fish client wrote its takes as fish_*.wav; the model is S2-Pro either way.
PREFIX = {"s2-pro": "fish", "qwen": "qwen"}
for model, prefix in PREFIX.items():
    for rname in ("short", "long"):
        for slang, text, _lang in SENTENCES:
            for seed in SEEDS:
                p = os.path.join(AB, f"{prefix}_{rname}_{slang}_{seed}.wav")
                if not os.path.exists(p):
                    print(f"  missing {os.path.basename(p)}")
                    continue
                m = measure(p, text)
                wav, sr = sf.read(p, dtype="float32")
                m.update({"model": model, "ref": rname, "lang": slang, "seed": seed,
                          "sim": round(webui._cosine(
                              ref_emb[rname],
                              embed(np.asarray(wav, dtype=np.float32).reshape(-1), sr)), 4)})
                rows.append(m)

json.dump(rows, open(os.path.join(AB, "measured.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

cols = ("seconds", "rate", "level_db", "hf_9k_milli", "hf_11k_milli", "snr_db", "wer", "conf", "sim")
print("\n=== every take ===")
print(f"{'model':8s}{'ref':7s}{'lang':5s}{'seed':6s}" + "".join(f"{c:>12s}" for c in cols))
for r in rows:
    print(f"{r['model']:8s}{r['ref']:7s}{r['lang']:5s}{r['seed']:<6d}"
          + "".join(f"{r[c]:12.3f}" if isinstance(r[c], float) else f"{r[c]:12d}" for c in cols))

print("\n=== means: does the longer reference help, and which model wins per axis? ===")
print(f"{'':16s}" + "".join(f"{c:>12s}" for c in cols))
for model in ("s2-pro", "qwen"):
    for rname in ("short", "long"):
        sel = [r for r in rows if r["model"] == model and r["ref"] == rname]
        if not sel:
            continue
        print(f"{model+'/'+rname:16s}" +
              "".join(f"{np.mean([r[c] for r in sel]):12.3f}" for c in cols))
print("\ncaveats: SIM uses Qwen's own speaker encoder (favours Qwen). S2-Pro decodes at "
      "44.1kHz, Qwen3-TTS at 24kHz — above 11kHz the 24kHz output has nothing to give.")
