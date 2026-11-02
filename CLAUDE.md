# CLAUDE.md — qwen-tts (local Qwen3-TTS deployment)

A local ML tooling repo: no cloud accounts, no personal data, no deployed services. Treat it as a local
ML tooling repo. The house rules that still apply: done = code + tests + docs in one
change, and never leave finished work uncommitted.

## What this is

A local install of QwenLM/Qwen3-TTS (1.7B checkpoints) with a Gradio UI, tuned for one
RTX 5070 Ti (16GB) inside **WSL2** with dynamic memory (VM ceiling = host RAM, 32GB).
Read `README.md` for user-facing docs and design notes.

## Ground rules for changes

- **Never run two model-owning processes at once.** The server and `tests/smoke_test.py`
  each load ~4.3GB per checkpoint; two copies pinned the VM into Windows' own memory and
  hard-locked the machine once. The `flock` single-owner lock in `webui.exclusive_owner()`
  enforces this — do not weaken it, and never use `--force` in a script or CI.
- **Never load a checkpoint without going through `ModelHub._get_cpu()` / `acquire()`.**
  They enforce `max_cpu_resident` and the RAM/VRAM floors. Ad-hoc
  `Qwen3TTSModel.from_pretrained(...)` in a script bypasses every guard.
- **Keep the WSL2 ceiling in mind when raising any cap.** `free -m` before you bump
  `--max-cpu-resident`; `MemAvailable` here is not headroom, it is the host's RAM.

- **Never move model-loading logic out of `ModelHub._move()`.** It is the single place
  that knows `model.speech_tokenizer` is a non-Module wrapper that must be moved to CUDA
  by hand. Bypassing it costs ~25x inference speed and produces no error, only slowness.
  If you refactor loading, keep `tests/test_app_logic.py::test_move_reaches_the_speech_tokenizer`
  honest.
- **Keep `sdpa` as the default attention implementation.** flash-attn 2 has no sm_120
  wheels; do not add it as a dependency.
- **Do not add dialect values to the language dropdown.** `beijing_dialect` /
  `sichuan_dialect` exist in `codec_language_id` but upstream excludes them from
  `get_supported_languages()`; dialect comes from the *speaker* (`spk_is_dialect`).
  `tests/test_app_logic.py::test_language_choices_match_checkpoint` enforces this.
- **Pin the CUDA story.** This box is Blackwell (sm_120): torch must be >= 2.7 built with
  cu128. `setup.sh` pins `torch==2.8.0` / `torchaudio==2.8.0`. Don't let a plain
  `pip install -U torch` drift the venv into a non-CUDA or older-CUDA wheel.
- **Bound generation length.** Any new generation path must set a `max_new_tokens`
  derived from prompt length (`_budget()`); the checkpoint default of 8192 lets a
  non-stopping decode run for minutes.
- **Gradio is v6** (`Blocks(css=...)` moved to `launch(css=...)`, `launch(show_api=...)`
  is gone). Check the installed version before copying snippets from the Space or from
  older blog posts.
- Weights in `models/` and generated audio in `tests/output/` are gitignored — never
  commit them.

## Verifying a change

```bash
.venv/bin/python tests/test_app_logic.py   # fast, CPU-only, runs on every change
.venv/bin/python tests/smoke_test.py       # GPU end-to-end, ~60s; required before "done"
.venv/bin/python tests/smoke_test.py --only voice_clone
```

The smoke test must run with the web server **stopped** (the lock enforces this).

There is no linter/formatter configured; match the existing style (4 spaces, type hints
on public functions).

## Things that are deliberately NOT here

- No flash-attn, no vLLM/vLLM-Omni (upstream supports it; unnecessary at 1.7B on this GPU).
- No 0.6B checkpoints (1.7B is the quality target and RAM/VRAM budget).
- No auth / public exposure. `--ip 0.0.0.0` plus `--share` exist for convenience; both
  expose an unauthenticated model endpoint — keep them off by default.
- No fine-tuning. Upstream `repo/finetuning/` is present but untouched.
