#!/usr/bin/env python3
# coding=utf-8
"""Create a shipped "described" voice — a description Speak and Chat can talk from.

    scripts/make_voice.py --name Fox \
        --description "A bright, expressive adult female voice..." \
        --line "Hey, I got your message. Look, I'm not upset."

Two decisions are baked into this file, so they are written down where someone will find them:

**Why generate instead of ship a recording.** A bundled voice would be a real person's
voice, and the README's ethical-use section says this project does not distribute those.
A description is text; the model renders it on your GPU, on your terms.

**Why the description is the primary form, with the clip as an extra.** Measured on the same
three lines: speaking a description directly keeps 2.67m of energy above 9kHz, while cloning
a saved take keeps 0.57m — cloning retains only ~60% of whatever reference it is handed, so
a voice that has already been through the vocoder once comes out duller the second time.
The frozen clip is there for the Clone tab and for anyone who prefers the steadier identity
(pairwise 0.993 for a fixed clip against 0.974 when the description is re-rolled).
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "webui"))

import numpy as np
import torch

import app as webui


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Create a described voice in the voice library.")
    parser.add_argument("--name", required=True, help="Display name; becomes voices/<slug>.json")
    parser.add_argument("--description", required=True,
                        help="What the voice sounds like — timbre, age, pace, accent, mood.")
    parser.add_argument("--line", default="Hey, I got your message. Look, I am not upset, I "
                                          "just wish you had told me earlier, that is all.",
                        help="A sample line, rendered into the frozen clip the Clone tab can use.")
    parser.add_argument("--language", default="English")
    parser.add_argument("--seed", type=int, default=1234, help="Same seed, same frozen clip.")
    parser.add_argument("--no-clip", action="store_true",
                        help="Description only; do not render a frozen clip (no GPU needed).")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing voice.")
    args = parser.parse_args(argv)

    slug = webui._slug(args.name)
    wav_path, meta_path = webui._voice_paths(slug)
    if os.path.exists(meta_path) and not args.force:
        print(f"{meta_path} already exists — pass --force to overwrite.")
        return 0

    description = args.description.strip()
    if not description:
        print("error: --description is what makes this a voice; it cannot be empty", file=sys.stderr)
        return 2

    wav, sr = None, 0
    if not args.no_clip:
        # Loading a checkpoint while the web UI (or the smoke test) also holds one is how
        # this machine was hard-locked once; take the same single-owner lock they do.
        owner = contextlib.ExitStack()
        try:
            owner.enter_context(webui.exclusive_owner("make_voice"))
        except RuntimeError as e:
            print(f"error: {e}\n(or pass --no-clip to save the description without a GPU)",
                  file=sys.stderr)
            return 2
        with owner:
            hub = webui.ModelHub(webui.MODELS_DIR, torch.bfloat16, "sdpa", 1, 1)
            webui.HUB = hub
            torch.manual_seed(args.seed)
            with hub.lock:
                tts = hub.acquire("voice_design")
                wavs, rate = tts.generate_voice_design(
                    text=args.line, language=args.language, instruct=description,
                    max_new_tokens=webui._budget(args.line),
                    **webui._sampling(0.9, 1.0, 50, 1.05))
        wav = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        sr = int(rate)
        print(f"rendered {len(wav) / sr:.1f}s of {args.name} at {sr}Hz")

    record = webui.save_voice(name=args.name, wav=wav, sr=sr, text=args.line.strip(),
                              instruct=description, language=args.language,
                              description=description, kind="design")
    print(f"wrote {meta_path}" + (f" and {wav_path}" if wav is not None else ""))
    print(f"  slug {record['slug']} | kind {record['kind']} | speaks from the description")
    return 0


if __name__ == "__main__":
    sys.exit(main())
