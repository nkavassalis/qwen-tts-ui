# Changelog

All notable changes to this project are documented here. Newest first.

## 2026-11-02 — pre-public hardening

Security / exposure:

- Client-named paths are now confined to `allowed_input_roots()` (Gradio's upload and
  component-cache dirs plus `tmp/media/`) before anything opens them —
  `resolve_input_path()`. Gradio 6 already refuses a path no user uploaded, but the File
  and Audio components do no checking of their own, so this is the guard that survives a
  Gradio upgrade. Read-only audio inputs moved to `type="filepath"` so we open them
  ourselves instead of Gradio doing it before our code runs.
- `--username/--password` (Gradio `auth=`), plus a loud startup warning whenever the app
  is bound beyond localhost or `--share`d without them.
- Extracted reference clips are pruned at startup and on import once older than
  `QWEN_TTS_MEDIA_TTL_SECONDS` (default 6h) — `tmp/media/` no longer accumulates people's
  voices forever.
- The clone-prompt cache is keyed on a SHA-256 of the reference PCM instead of
  `(length, float sum)`, which collided for same-length, same-sum clips.
- "Save as voice" takes the clip, text, description and language as inputs instead of
  reading a process-global `_LAST_DESIGN`, which leaked one user's clip into another
  user's save on a shared LAN.

Bugs:

- `--device cpu` was unusable: the VRAM budget ran unconditionally and `vram_available_mb()`
  reports 0 with no CUDA device, so every load raised `MemoryError`.
- `mem_available_mb()` returned `0.0` when `/proc/meminfo` is unreadable, which the RAM
  floor read as "no memory" — every load failed off-Linux. It now returns `+inf` and the
  status line says `unknown`.
- `./run.sh --ssl-self-signed` crashed at launch: Gradio's startup self-check does an
  https request to its own port and dies on `CERTIFICATE_VERIFY_FAILED`. `launch()` now
  gets `ssl_verify=False` via the new `--ssl-self-signed` flag.
- `speaker_choices()` default was hard-coded to `ryan`, which silently broke the dropdown
  (and "Speaker is required") whenever the checkpoint's `spk_id` table differed. Now
  `default_speaker()`.
- `setup.sh` hard-asserted `compute capability >= (12, 0)`, aborting the install on every
  pre-Blackwell GPU despite its own message. Now a warning.
- `setup.sh` re-downloaded the 12Hz tokenizer on every run (the skip test looked for
  `model.safetensors`, which that directory does not have).
- `run.sh` only checked `certs/key.pem`, so a half-written cert pair reached Gradio as a
  launch error; it also needs `openssl` on PATH now, and reports it clearly.
- The ffmpeg scratch file is written under `tmp/media/` and deleted after import instead
  of being left in a system temp dir.

Docs / reproducibility:

- README claimed Whisper is fetched with `--exclude "*fp32*" "*.bin" "*.msgpack"` — that
  command existed nowhere. `setup.sh` now does it (plus `*.h5`, `*.ot`).
- torch is installed from the explicit cu128 index rather than relying on whatever CUDA
  build PyPI's default `torch==2.8.0` wheel happens to be.
- Added `LICENSE` (Apache-2.0, matching upstream), `requirements.txt`, `AGENTS.md` (the
  agent doc) and this `CHANGELOG.md`; `CLAUDE.md` is now a pointer to `AGENTS.md`.
- Removed another project's name from `CLAUDE.md`; scrubbed the machine's exact RAM figure
  from the WSL2 note.
- Corrected the upstream line reference for the dialect filter to `modeling_qwen3_tts.py:
  1831-1834`; `--preload` is documented as bounded by `--max-cpu-resident` (it always was).

Tests (CPU tier 20 → 28 cases):

- `test_client_paths_are_confined_to_the_upload_dirs` (absolute path, symlink escape, URL,
  audio box), `test_gradio_roundtrip_paths_are_accepted` (our own outputs stay readable),
  `test_prune_media`, `test_build_ui_constructs` (the Blocks tree builds against the
  installed Gradio), `test_acquire_works_without_cuda`, `test_default_speaker_is_offered`,
  `test_setup_sh_knows_every_capability`, a same-length/same-sum clone-cache collision
  case, and a `test_save_designed_voice_uses_the_audio_box` replacement for the old
  `_LAST_DESIGN` test.
- The runner reports `ERROR` + traceback instead of aborting the whole run on a non-
  assertion exception; `test_memory_guards` no longer uses a deleted temp dir.
- `smoke_test.py`'s `media_pipeline` fixture mp4 is now built inside `MEDIA_DIR`, which the
  path guard requires; `--only media_pipeline` on a box without ffmpeg printed SKIPPED and
  still exited 0 — skipped cases are now recorded and fail the run unless `--allow-skip`
  (`test_smoke_test_skips_are_not_passes`). The dead `finally` branch in
  `run_voice_library` is gone.
