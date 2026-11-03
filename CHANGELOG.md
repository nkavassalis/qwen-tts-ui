# Changelog

All notable changes to this project are documented here. Newest first.

## 2026-11-03 — clone quality: measured, then two guards added

Measured first (SIM = the model's own speaker-encoder cosine, WER = Whisper large-v3
against the requested text). Temperature / top-p / `non_streaming_mode` moved both metrics
inside noise, so nothing was changed there. What did matter: a transcript that does not
match the clip (WER 1.000 and 17.4s of audio for a 5s sentence) and reference noise
(−0.020 SIM). Reference length pays up to ~5s; recording level does not matter at all
(0.991 at 10% peak).

- **Fit check** (`transcript_fit`, `clip_fit_status`): a live, non-blocking line on the
  Voice Clone tab plus a WARNING appended to the generate status, catching a transcript
  that cannot be what the selection says. Coarse on purpose (factor-of-2 words-per-second,
  CJK counted per character so Chinese/Japanese/Korean clips are not falsely flagged).
- **Best of N** (`Candidates` slider + `Seed` field): generates up to 4 takes and keeps the
  best, ranked by WER then speaker similarity, with every candidate's score reported so the
  choice is auditable. Falls back to similarity-only when transcription is disabled, and
  scoring failures never sink a generation. Same seed + same inputs reproduces bit-identical
  audio (asserted in the GPU tier).
- README gains *Getting the best quality* with both sweep tables and the ranked advice.
- Tests: CPU tier 31 → 35 (fit boundaries, CJK counting, WER semantics, candidate ranking,
  UI wiring drift check). GPU tier gains a `best_of_n` case that drives the real callback.

## 2026-11-03 — save a cloned reference; targets 16GB Blackwell, not one SKU

- The Voice Clone tab could *load* and *delete* saved voices but had no Save, so a voice
  you built from a video or an upload — trimmed waveform, Whisper transcript and all —
  could not be kept. `save_reference_voice()` + a *Save to voice library* button store the
  reference clip and its transcript in the same `voices/` format a designed voice uses.
  A sub-minimum clip is refused; a transcript-less save is allowed but says it will need
  *x-vector only*. Generated output is unchanged: download it from its player.
- README gains a *Cloning a voice, start to finish* walkthrough, which is what the missing
  button was hiding.
- Positioning: this targets the **16GB Blackwell class** (RTX 5070 Ti, RTX 5080, …), not
  the 5070 Ti specifically — that is just the SKU it is developed and tested on. Title,
  UI header, module docstring and AGENTS.md updated; the dated test records still name the
  5070 Ti, because that is what ran them.
- Tests: CPU tier 30 → 31.

## 2026-11-03 — argv-free credentials, GPU tier verified

- `resolve_credentials()` takes the UI password from `--password-file` (contents never
  reach argv or the logs), then `QWEN_TTS_PASSWORD`, then an interactive hidden prompt.
  `--password` still works but warns: a command line is world-readable through `ps`/`/proc`
  (demonstrated during review — the test run's password was visible in `pgrep`).
  `QWEN_TTS_USERNAME` is read as well; a password without a username fails closed to
  unauthenticated, as does an empty or unreadable secret file. A world-readable secret
  file is accepted but called out for `chmod 600`.
- Tests: CPU tier 28 → 30 (credential source priority, fail-closed paths, prompt
  labelling, "the secret never appears in output"). `test_smoke_test_skips_are_not_passes`
  now stubs `exclusive_owner` instead of depending on the webui server being stopped.
- GPU tier executed on the RTX 5070 Ti box: 6/6 smoke cases, auth 401s unauthenticated
  API calls, and the path guard refuses `/etc/passwd` and friends over HTTP.
- Recorded: all three checkpoints ship `do_sample: true`, so the sampling sliders are
  effective (an open question from the CPU-only run).

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
