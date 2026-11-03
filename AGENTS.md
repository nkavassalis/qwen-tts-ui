# AGENTS.md — working on qwen-tts

## Project mode: REAL

GitHub `nkavassalis/qwen-tts`, **private** until Nick says otherwise; default branch
`main`. Verify `gh auth status` shows `nkavassalis` before any push. Both test tiers
green before any push or handoff (the GPU tier cannot run on a box without a GPU — see
*Testing limitations*). Secret scan before every commit.

## Purpose

A local, offline deployment of [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
(1.7B checkpoints) behind a Gradio UI, tuned for **one** 16GB Blackwell consumer GPU —
RTX 5070 Ti, RTX 5080, anything in that class (the 5070 Ti is the SKU it is developed and
tested on, not the only one it targets). Three modes:
CustomVoice (9 preset speakers), Voice Design (voice from a text description), Voice
Clone (from a ~3-10s reference clip). Nick uses it at his desk; it is not a service.

**Spirit: one GPU, one process, no cloud.** Everything is sized around "a 16GB card and
one box's RAM". Every feature should be checkable against that — if it needs a second
process, a second GPU, or a network call at inference time, argue for it explicitly.

## Layout

```
webui/app.py             the whole app: Gradio UI, ModelHub (CPU-RAM + LRU GPU slot),
                         MediaError/path guards, ffmpeg ingest, Whisper, voice library
setup.sh               reproducible env (uv, py3.12, torch cu128, upstream qwen-tts) +
                       weight and Whisper downloads
run.sh                 launcher; mints self-signed certs for --ssl-self-signed
requirements.txt       direct deps only (torch/torchaudio and transformers/accelerate are
                       pinned elsewhere — read the file header before adding to it)
tests/test_app_logic.py  CPU tier: 37 checks, no weights, ~10s with deps installed
tests/smoke_test.py      GPU tier: 6 end-to-end flows, writes tests/output/*.wav
LICENSE                  Apache-2.0 (matches upstream; app.py derives from their Space)
README.md              human-facing docs, troubleshooting table
CHANGELOG.md           one entry per change, newest first

gitignored, created on demand:
.venv/  models/ (~14GB weights)  repo/ (upstream checkout)  voices/  tmp/media/
certs/  logs/  tests/output/  .qwen-tts.lock  .gradio/
```

## How to run

```bash
./setup.sh                       # uv venv + torch cu128 + upstream + 14GB weights + Whisper
./run.sh                         # -> http://127.0.0.1:7860, "Runtime:" line shows GPU/RAM
./run.sh --ip 0.0.0.0 --username u --password-file ~/tts.pw   # LAN, authenticated
./run.sh --ssl-self-signed --ip 0.0.0.0 --username u --password-file ~/tts.pw   # browser mic needs HTTPS
./run.sh --device cpu            # CPU inference (slow; works, no CUDA needed)
```

You should see `[webui] transcription: … | ffmpeg: …`, then a `Runtime:` line, then
`* Running on local URL`. If the second one never appears, check `nvidia-smi` (VRAM) and
`free -m` (RAM) — the hub refuses to over-commit rather than swapping.

## How to run the tests

```bash
.venv/bin/python tests/test_app_logic.py   # CPU tier — 37 checks, ~10s
.venv/bin/python tests/smoke_test.py       # GPU tier — 6 flows, ~90s, needs weights
.venv/bin/python tests/smoke_test.py --only media_pipeline
```

Current state (2026-11-03): **CPU tier 37/37 passing**, run on Python 3.12 with
torch 2.8.0 (CPU wheel), gradio 6.17.3, transformers 4.57.3. **GPU tier NOT run in that
session** — the box had no GPU; see *Testing limitations*.

The CPU tier imports `torch`, `gradio`, `qwen_tts`, `soundfile`, `librosa` and `PIL`, so
it needs the venv — it just needs no weights and no CUDA. The GPU tier refuses to start
while the web server holds `.qwen-tts.lock`; stop the server rather than using `--force`.

## Design decisions (and why — don't undo casually)

- **CPU-resident weights + LRU GPU slot** (`ModelHub`). The official Space keeps five
  checkpoints on the GPU (~20GB); a 16GB card cannot. Each 1.7B model lives in host RAM
  and an LRU promotes one at a time to the GPU. Switching tabs costs ~2s.
- **`_move()` is the only place that moves models.** `model.speech_tokenizer` is a
  `Qwen3TTSTokenizer` *wrapper*, not an `nn.Module` (verified against upstream
  `inference/qwen3_tts_tokenizer.py`: plain class with `.model` and `.device`), so
  `model.to("cuda")` leaves the 12Hz vocoder on CPU. Symptom: generation ~25x slower
  (141s vs 6s), GPU pinned near 38%, **no error**. Enforced by
  `test_move_reaches_the_speech_tokenizer`.
- **Three memory guardrails, all of them load-bearing.** This box is WSL2 with dynamic
  memory: the VM balloons toward the host's whole RAM, so over-committing does not just
  OOM the distro, it can hard-lock Windows. It did once (`--preload` server + concurrent
  smoke test ≈ 26GB + page cache). Hence: the `flock` single-owner lock, `max_cpu_resident`
  LRU cap, and the `MemAvailable`/VRAM floors. Do not weaken any of them; verified live
  2026-11-02 that a second server refuses to start and names the holder.
- **`sdpa`, not flash-attn.** flash-attn 2 ships no sm_120 (Blackwell) wheels.
- **Dialect is a speaker trait, not a language.** The checkpoint carries
  `talker_config.spk_is_dialect` (`dylan → beijing_dialect`, `eric → sichuan_dialect`) and
  upstream filters dialects out of `supported_languages`
  (`modeling_qwen3_tts.py:1831-1834`), so `language="beijing_dialect"` raises
  `ValueError: Unsupported languages`. Pick the speaker instead. Enforced by
  `test_language_choices_match_checkpoint`. Italian *is* selectable — the hosted Space
  omits it by oversight.
- **Output post-processing is listening-only** (`postprocess_output`). The vocoder is fixed
  at 24kHz, so the "digital" character is near-Nyquist imaging (measured 4x excess), not
  missing bandwidth — nothing here can add air above 12kHz, and anything that appears to is
  inventing it. Modes: off / clean (soxr HQ 48k) / warm (48k + lowpass 11.5k + small
  high-shelf + mild tanh). It runs *after* `_report()` and after candidate scoring, so
  durations, WER and SIM all describe the model's own decode; it falls back to the raw
  signal on any failure, including missing soxr/scipy, because polish must not be able to
  break a generation. `Off` is the mode to use when saving a Voice Design clip into the
  library, since that clip becomes a clone reference.
- **Quality lives in the reference, not the sliders** (measured — see README *Getting the
  best quality*). Decode knobs moved SIM/WER inside noise; a transcript that does not match
  the clip cost WER 1.000 and 3x the output length. Hence `transcript_fit()` (a coarse
  words-per-second guard, deliberately not blocking) and best-of-N scored by WER then
  speaker similarity. Do not "optimise" by removing either to save a few seconds.
- **`max_new_tokens` is bounded by prompt length** (`_budget()`, cap 2048). The checkpoint
  default is 8192 and ICL cloning sometimes never emits EOS: one run produced 163s of
  audio for one sentence.
- **Whisper via transformers, not faster-whisper.** Reuses the torch/CUDA stack
  (ctranslate2 kernels may not cover sm_120). Loaded only on the first *Transcribe* click.
  The explicit padding mask is required: Whisper pads every clip to 30s and its pad token
  *is* eos, so without `attention_mask` you get warnings plus hallucinated text over the
  silence.
- **The waveform is drawn with PIL.** No matplotlib; pixel geometry is deterministic, so
  `test_render_waveform` can assert the shaded selection actually changes the image.
- **`gr.State` holds JSON only** (clip path + duration), never a numpy array — it must
  survive the browser round-trip. `test_media_state_is_json_serialisable` guards it.
- **"Save as voice" takes the clip as an input** (2026-11-02). It used to read a
  module-level `_LAST_DESIGN` global, so with two people on the LAN one person's Save
  button could persist the other's clip. Same class of bug: the clone-prompt cache is now
  keyed on a SHA-256 of the PCM bytes, because `(length, float sum)` collided
  (`test_clone_cache_key`).
- **Client-named paths are confined** (`resolve_input_path`). Gradio 6's
  `preprocess_data` already refuses a path no user uploaded (verified live: posting
  `{"path": "/etc/hostname.wav"}` to `/gradio_api/call/import_media` dies with
  "…because it was not uploaded by a user"), **but the File/Audio components themselves
  do no such check** — `components/file.py::_process_single_file` uses `f.path` straight
  off the payload. Ours is the belt that keeps working if a Gradio upgrade loosens
  theirs. Read-only inputs are therefore `type="filepath"` (with `type="numpy"` Gradio
  opens the client's path before our code runs).
- **`--ssl-self-signed` sets `ssl_verify=False`** (2026-11-02). `launch()` does an https
  self-check against its own port and dies with `CERTIFICATE_VERIFY_FAILED` on a
  self-signed cert — the documented mic-on-LAN flow was broken until that flag existed.
  It only affects Gradio dialling itself; browsers still see the cert.
- **`mem_available_mb()` returns `+inf`, never 0, when unknown** (2026-11-02). 0 reads as
  "no RAM" to the floor check and made every load raise `MemoryError` off-Linux.
  `status()` renders it as `unknown`.
- **The VRAM budget only applies to a CUDA target** (2026-11-02). It used to run
  unconditionally, so `--device cpu` refused every load (`vram_available_mb()` is 0 with
  no CUDA). `test_acquire_works_without_cuda`.
- **`setup.sh` warns instead of asserting on `cap >= (12, 0)`** (2026-11-02). The old
  assert killed the install for every pre-Blackwell GPU while its own message claimed
  otherwise.

## Invariants (test-enforced unless noted)

1. One model-owning process at a time (`flock`); `--force` is the only override and must
   never appear in a script or CI. `test_exclusive_owner_lock`.
2. Models load only through `ModelHub._get_cpu()`/`acquire()` and Whisper only through
   `Transcriber.load()`. Ad-hoc `from_pretrained` bypasses every floor. (not automated)
3. `ModelHub._move()` moves the talker *and* every non-Module wrapper holding an
   `nn.Module`. `test_move_reaches_the_speech_tokenizer`.
4. Language dropdown values ⊆ what the checkpoint accepts, dialects excluded, Italian
   present. `test_language_choices_match_checkpoint`.
5. Speakers come from the checkpoint's `spk_id` table, and the preselected default is one
   of them. `test_speaker_choices_come_from_the_checkpoint`, `test_default_speaker_is_offered`.
6. Every generation path sets `max_new_tokens` from `_budget()`. `test_budget`, `test_report`.
7. `gr.State` values are JSON-serialisable. `test_media_state_is_json_serialisable`.
8. A browser-named path is only ever opened inside `allowed_input_roots()`.
   `test_client_paths_are_confined_to_the_upload_dirs`,
   `test_gradio_roundtrip_paths_are_accepted` (the counterweight: our own outputs must
   still be readable).
9. Whisper always gets an explicit padding mask. (not automated — GPU tier only)
10. Video ingest stays ffmpeg-only; no `av`/`imageio-ffmpeg`/ctranslate2/matplotlib.
    (not automated)
11. `setup.sh`'s download list matches `CAPABILITIES`. `test_setup_sh_knows_every_capability`.
12. The Blocks tree constructs against the installed Gradio. `test_build_ui_constructs`.
13. A skipped GPU case is **not** a pass: `smoke_test.py` exits non-zero unless
    `--allow-skip`. `test_smoke_test_skips_are_not_passes`.
14. Extracted clips are pruned (`MEDIA_TTL_SECONDS`, default 6h) so `tmp/media/` cannot
    grow forever and hold people's voices. `test_prune_media`.

## Conventions

- 4 spaces, type hints on public functions, no linter/formatter configured — match the
  existing style.
- User-fixable problems raise `MediaError` and surface in a status box; never let an
  exception kill a request. Generation callbacks catch broadly on purpose so the UI keeps
  working — but they echo `type(e).__name__: e`, so do not put anything in an exception
  message you would not want a LAN user to read.
- New UI helpers go in `webui/app.py` next to their siblings and get a CPU-tier test; the
  tier is fast and it is what catches Gradio API drift.
- Prefer inputs over module globals for anything per-user (see `_LAST_DESIGN` above).
- Keep Whisper out of the import path; never import its classes at module scope.
- Commit style: imperative subject, bullet body explaining the *why* (see `git log`).

## Known issues / TODOs

- No CI. The CPU tier needs torch+gradio (~2GB of wheels) but no GPU; a GitHub Actions
  job would catch the Gradio-drift class of failure that pinning only partly covers.
- `requirements.txt` is ranges, not a lock, because the pins were frozen on a box without
  its venv available. Freeze the venv that ran the green GPU tier and commit
  `requirements.lock.txt`.
- Credentials are one shared credential (Gradio's `auth=`), not users. Fine for a house
  LAN; do not put this on the internet. `resolve_credentials()` takes the secret from
  `--password-file`, then `QWEN_TTS_PASSWORD`, then an interactive prompt; `--password`
  remains only as a warned-about fallback because argv is world-readable via `ps`/`/proc`.
- `Transcriber` keeps Whisper (~3.1GB bf16) on the GPU once loaded and is never evicted
  by `ModelHub`; `unload()` exists but only the smoke test calls it. Fits next to one
  resident 1.7B model on 16GB, but it does eat into the VRAM margin.
- `default_concurrency_limit=1` serialises everything: a Whisper load blocks a queued
  generation behind it. Deliberate (one GPU), but surprising if two people share it.
- The clone-prompt cache holds exactly one entry (`_CLONE_PROMPTS.clear()` before each
  insert) — right for retakes, wasteful if you alternate references.
- `smoke_test.py` fetches its clone reference over the network — the example clip URL from
  the upstream Qwen3-TTS README (Alibaba's own public asset host, flagged by the secret
  scan by design). The GPU tier therefore needs network for that one case; there is no
  offline mode for it yet.

## Testing limitations / unverified

Run on 2026-11-02 on a **CPU-only Linux box with no GPU and no weights**:

- **CPU tier: 27/27 passing** (`tests/test_app_logic.py`), torch 2.8.0+cpu,
  gradio 6.17.3, transformers 4.57.3, Python 3.12.
- **GPU tier: never run.** `tests/smoke_test.py` needs CUDA + the 14GB weights. It was
  edited in this session (its `media_pipeline` fixture mp4 now lives inside `MEDIA_DIR`,
  which the path guard requires) and is therefore **unverified** — run it before pushing.
- **Verified by launching the real server** (CPU target, no weights): UI builds and
  serves, `--username/--password` gates the app (anon gets the login shell, authed gets
  the 143KB app), `--ssl-self-signed` serves HTTPS (this needed the `ssl_verify=False`
  fix — it crashed before), the single-owner lock refuses a second server by name, and a
  forged `{"path": "/etc/hostname.wav"}` payload to `/gradio_api/call/import_media` is
  rejected.
- **Not verified, needs the GPU box:** actual audio quality, VRAM/RAM behaviour under
  real loads, `--preload`, `--share`, the unauthenticated-bind warning (it is a `print`
  gated on `--ip`/`--share`; binding 0.0.0.0 to test it was not appropriate on a shared
  box), Whisper transcription accuracy, mic capture over HTTPS, dialect speakers, and
  whether the checkpoint's `generation_config.json` leaves `do_sample` on — if it does
  not, the sampling sliders are inert. `_load()` now prints the merged generate defaults
  at startup; check that line once on the real box.
- Upstream API claims in this file were checked against a `main` checkout of
  QwenLM/Qwen3-TTS (2026-11-02), not against the pinned `repo/` Nick actually runs.

Run on 2026-11-03 on the **real GPU box** (RTX 5070 Ti, 16GB, CUDA, 14GB of weights) — the
tier the hardening pass could not execute:

- **CPU tier: 31/31 passing**, including the two new `resolve_credentials` checks and
  `test_save_reference_voice_from_the_clone_tab`. The
  `smoke_test` bookkeeping check was made hermetic (it stubs `exclusive_owner`, so it no
  longer fails just because the webui server is running).
- **GPU tier: 6/6 passing** (`tests/smoke_test.py`): custom_voice, voice_design,
  voice_clone, dialect, voice_library, media_pipeline. Loads 1.5–3.0s, generation
  1.4–9.7s, every clip inside the RMS/peak/duration bounds.
- **Whisper verified**: `media_pipeline` transcribed the mp4-derived clip as
  "The quick brown fox jumps over the lazy dog near the riverbank." (3.2s).
- **Auth verified live**: unauthenticated `/gradio_api/info` → 401, `gradio_client`
  refuses without credentials and succeeds with them.
- **Path guard verified live**: `/etc/passwd`, `../../etc/passwd`, `file:///etc/passwd`
  and `~/.config/gh/hosts.yml` are all refused by `/import_media`.
- **Sampling sliders are live, not inert**: `generation_config.json` in all three
  checkpoints sets `do_sample: true` (top_k 50, temperature 0.9), which closes the open
  question from the CPU-only run.
- **Still unverified:** `--preload`, `--share`, HTTPS mic capture, and the
  unauthenticated-bind warning (needs a real `--ip 0.0.0.0` bind, which was not done on a
  networked box).

## Do-not-touch list

- **The three memory guardrails** (lock, `max_cpu_resident`, RAM/VRAM floors) — they
  exist because the machine hard-locked. Nick's OK required.
- **`ModelHub._move()`** — bypassing it costs 25x inference and produces no error.
- **The dialect/language split** — re-adding `*_dialect` to the dropdown raises at
  generate time.
- **`UPSTREAM_REF` in `setup.sh`** — moving it off the SHA Nick tested changes the model
  code under a "working" install; freeze deliberately, not casually.
- **`tests/smoke_test.py`'s audio assertions** (rms/peak/duration) — they are what catch
  a model quietly left on CPU. Weakening one to make it pass is a red flag.
- **`voices/`, `tmp/media/`, `certs/`, `.qwen-tts.lock`** — real user audio and key
  material; gitignored forever.

## Guardrails in effect

- Visibility: **private** on `nkavassalis/qwen-tts` until Nick explicitly flips it.
- Branch: `main`; `feat/*`/`fix/*` for work, never force-push `main`.
- Before any commit: scan `git diff --cached` (`secret-scan` skill / `scripts/secret-scan.sh`).
  Before any push: both test tiers, the secret scan, and `gh auth status` = `nkavassalis`.
- Nothing internal in any commit message, README, AGENTS.md or log: no hostnames, LAN
  addresses, home paths, account ids, or live user audio.
- Model weights and generated audio stay gitignored; never commit `models/`,
  `tests/output/`, `voices/` or `tmp/`.
