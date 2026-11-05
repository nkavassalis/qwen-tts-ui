# Changelog

All notable changes to this project are documented here. Newest first.

## [Unreleased] — the server can now survive a reboot

### Added

- **`scripts/install_service.sh`** — `install | uninstall | status | restart | logs`. macOS gets a
  LaunchAgent with `KeepAlive`; Linux gets a systemd user unit with `Restart=always`, enabled at
  boot with lingering on. Per-user, no sudo, and `uninstall` removes exactly what `install` wrote.
  Before this, a reboot (or simply closing the terminal you started it from) left the box silent
  until someone noticed.
- README section *Keep it running*, covering the three things that surprise people: the lock still
  arbitrates against a manual `./run.sh`, `--password` is refused, and a service needs its own
  `PATH` because it never reads your shell profile.

### Fixed

- **`install_service.sh install` failed on a Mac that had never had a LaunchAgent**, because
  `cat >` does not create `~/Library/LaunchAgents`. Caught by the CPU tier, not by a user.

### Notes

- Both were installed here and proved by killing the process: launchd respawned the Mac's server
  (pid 7329 → 7352) and systemd respawned this one, both serving 200 afterwards, with ffmpeg
  visible to the service.
- The service runs `run.sh` rather than `webui/app.py`, so the venv and cert minting stay in one
  place. `--password` is refused outright; only `--password-file` reaches a unit file.

## [Unreleased] — the server can speak out loud (`/say_aloud`)

### Added

- **`/say_aloud`, `say.py --server-play`, and a "Speak aloud on this machine" button on the Speak
  tab.** `/speak` returns audio, so it plays on the machine that asked. This plays on the machine
  that *answered* — the box under the TV, the Mac on the desk — and sends back only a status line:
  `Speak (described) OK — 2.1s of audio · playing on this machine via ffplay`. Same voice
  resolution, chunking, render cache and post-processing as `/speak`, so a line the house already
  heard replays instantly. `--volume` (0–2) for a speaker in the next room.
- Documented the plain-`curl` route, which is three requests because Gradio 6 has no GET endpoints
  and the call API needs the session cookie from `/login` (verified: `GET /speak?…` is 404, and
  `-u user:pass` on `/gradio_api/call/…` is 401 — basic auth is not what that endpoint wants).

### Notes

- The player is spawned **detached**, with stdin/stdout/stderr on `DEVNULL`.
  `default_concurrency_limit=1` serialises the app, so a synchronous player would freeze the UI for
  the length of playback, and a player whose pipe filled up blocks the writer — which is us.
- `play_on_server()` refuses anything outside `tmp/media/`. The endpoint takes no path parameter at
  all, and a test fails if one is ever added: a player that accepts an arbitrary filename is a file
  reader wearing a speaker costume, on a box reachable from the LAN.
- Gain is applied to the samples rather than delegated to the player, because `afplay`/`paplay`/
  `ffplay` spell volume three different ways and only the sample version is testable here.

## [Unreleased] — `say.py --language` became usable

### Added

- **`--list-languages`**, read from the server's own dropdown rather than a second copy in the
  script, so the two cannot drift from the list invariant 4 keeps honest against the checkpoint.

### Fixed

- **`--language` existed but was unusable.** It had no help text, was not in the README, and was
  **case-sensitive**: `--language english` printed
  `speak failed: AppError: Value: english is not in the list of choices: ['Auto', 'Chinese', ...]`
  — the server's internals, in someone's terminal, for a capital letter. It is now
  case-insensitive, documented in `--help` and the README, and an unknown value is refused
  locally (exit 2) naming what the server speaks, **before** a render is paid for.
  `test_say_py_normalises_the_language_flag`.
- The language list is read from `api_info_as_input.enum`, which sits on the component as a
  **sibling of `props`**, not inside it — looking only under `props` finds nothing and silently
  turns the check off, which from the shell looks exactly like "the flag is broken".
  `test_say_py_reads_the_language_list_from_the_server` pins both shapes.

## [Unreleased] — `scripts/say.py` now finds the right interpreter itself

### Fixed

- **`scripts/say.py --voice … --text …` failed with a message that blamed the server.** Running
  it by name gave
  `cannot talk to http://127.0.0.1:7860: TypeError: argument of type 'bool' is not iterable`.
  The server was healthy. The script is executable and its shebang is `/usr/bin/env python3`,
  so a bare invocation used the **system** interpreter, which on this box carries
  **gradio_client 1.4.3** — and 1.x cannot parse the API schema gradio 6.17 emits, dying inside
  its own `get_type(schema)`. The venv's 2.5.0 works. The script now re-execs itself under
  `./.venv/bin/python` when one exists, so the documented one-liner works as documented;
  `QWEN_TTS_NO_REEXEC=1` opts out, and if the client is still unusable the error names the
  version and the fix instead of mimicking a server fault.
  `test_say_py_finds_the_project_venv`, `test_say_py_refuses_an_unusable_gradio_client`.

### Notes

- The guard compares realpaths before exec'ing: already-in-the-venv must not re-exec, or the
  script loops forever.

## [Unreleased] — the first and last word were being trimmed away

### Fixed

- **`trim_edges()` deleted audible speech.** It called anything under an absolute 20 ms RMS of
  **0.005** silence and kept only **60 ms** of air — less than one 12Hz codec frame (83 ms). One
  measured take had a last window at RMS **0.0034**: clearly audible against a 0.002 floor,
  invisible to the bar, and **180 ms of it was deleted**. That is "it cuts off the last word".
  The cut point is now relative to the take's own level (`TRIM_RELATIVE`, floored by
  `TRIM_FLOOR_RMS`) and **clamped to never exceed the old bar**, so this can only keep more audio
  than before, never less; the kept margin went 60 → 150 ms. The five measured takes that lost
  160/0/80/180/180 ms now lose **0/0/0/0/30 ms**. `test_trim_edges_*` (4 new checks).
- **The render cache could replay audio the current code would not produce.** `_model_tag()`
  ignored the engine and the quantisation, so `--quant 8bit` and `bf16` shared cached renders —
  which makes a working lever look dead — and nothing at all invalidated when the *model-input*
  path changed. Keys now carry `v{RENDER_IDENTITY}`, bumped on such changes. This is the
  mechanism that made the fixed clone still sound generic: the cache is keyed on the caller's
  inputs, so a bug in what the model *received* left the wrong audio cached under the right key.
  `test_render_cache_identity_covers_what_changes_the_audio`.

### Notes

- Ruled out by measurement, not assumption: the decode is clean (`start_jump`/`end_jump` 0.000
  on every take, first/last 20 ms at zero), and post-processing is innocent — `off`/`clean`/`warm`
  all change the length by **0.0 ms**, and `warm` actually *lifts* the fricative tail
  (0.0034 → 0.0055). A single-chunk Speak returns "the model's own decode, untouched", so this
  bit the chunked path — Speak and Chat on multi-sentence text, at every seam.

## [Unreleased] — fix: the Mac cloned the wrong person

### Fixed

- **A second voice was cloned from the first voice's clip.** `mlx_engine` kept its reference
  temp file in one slot keyed on the literal string `"path"`, so every reference after the first
  reused the first file. Measured: prompt #2's file was byte-identical to fox's and scored
  **SIM 1.000 against fox, 0.924 against the voice actually requested**. WER and confidence were
  perfect throughout, because both measure the words — which is why this reached a real listening
  session instead of a test failure. The cache is now keyed on a SHA-256 of the PCM bytes plus the
  sample rate, bounded at `REF_CACHE_SIZE`, and evicted oldest-first.
  `test_mlx_reference_cache_is_keyed_on_content`.
- **The clone path ignored the reference the UI actually sends.** The app calls
  `generate_voice_clone(text=, language=, voice_clone_prompt=)` and nothing else; the shim read
  `ref_audio`/`ref_text`, found neither, dropped the prompt, and mlx-audio took its **non-ICL**
  branch — a fluent, correctly-worded rendering in the checkpoint's default speaker. That is both
  of Nick's complaints ("sounded very little like the original", "a generic male voice") from one
  cause. `voice_clone_prompt` is now the payload it always was: it carries the audio, the
  transcript and the mode. A clone that reaches the backend with no reference **refuses** rather
  than generating a stranger. `test_mlx_clone_uses_the_prompt_the_app_actually_sends`.
- **The speaker-similarity tie-break was dead on the Mac.** mlx's `extract_speaker_embedding`
  wants an `mx.array`; handing it numpy raises inside `mx.concatenate`, and best-of-N calls it
  under `except Exception` — so candidates ranked on penalties alone and nothing said so. Now
  converted, and resampled to the 24kHz that encoder demands.
- **A shadowed method made one fix inert.** The rewritten `create_voice_clone_prompt` sat above
  the original, so the original won and its prompt item never carried the audio — the "rewrite the
  temp file if macOS reaped it" fallback could not fire. `test_mlx_engine_has_no_shadowed_methods`
  now fails on any duplicate `def` in a class in that file.

### Notes

- The first probe passed ref_audio/ref_text by hand and reported SIM 0.996, which is why this
  survived the earlier verification: **the probe tested a call shape the app does not use.** The
  regression tests now call it the way the UI does.

## [Unreleased] — Apple Silicon is a first-party platform (MLX)

### Added

- **`webui/mlx_engine.py`**: the same app on an M-series Mac, running on
  [mlx-audio](https://github.com/Blaizzy/mlx-audio) instead of CUDA. It is a shim over the four
  places the app touches a model (`generate_voice_clone`, `generate_voice_design`,
  `create_voice_clone_prompt`, the speaker embedding), selected inside `ModelHub` — which is
  already the only place a model is loaded (invariant 2). The CUDA path is untouched.
- **`--engine {auto,torch,mlx}`** and **`--quant {bf16,8bit}`**. `auto` picks MLX on Apple
  Silicon, torch elsewhere (`default_engine()`, mirrored in `setup.sh`). Both quantisations of
  both checkpoints are downloaded by `setup.sh` so `--quant` is a restart, not a surprise 2GB
  fetch. **bf16 is the default** because every number in `docs/quality.md` was measured on it.
- **`setup.sh` branches on `uname`**: Linux gets torch 2.8.0+cu128 and the pinned upstream
  checkout; macOS gets CPU torch (Whisper and the audio helpers), `mlx-audio`, and the four
  `mlx-community` repos into `models-mlx/`. The Mac path never clones upstream `qwen-tts`, and
  `from qwen_tts import Qwen3TTSModel` is now optional so it does not have to.
- **`resolve_device()`**: `--device cuda` on a box with no CUDA now prints a fallback at startup
  instead of failing later inside Whisper's `.to("cuda")`.

### Measured

- MLX 8-bit on an M4 Pro against bf16 on the 5070 Ti, same reference and sentences:
  **WER 0.00 on both**, teacher-forced confidence matching take for take (−0.02…−0.50 vs
  −0.04…−0.47), top-end energy in the same band, both 24kHz. **2.5–3.0s** per take on the Mac.
  Warm best-of-N (3 candidates + scoring) over the LAN: **5.3s**. See
  [docs/quality.md](docs/quality.md#apple-silicon-does-mlx-leave-anything-on-the-table).

### Fixed

- **Whisper silently did nothing on the Mac.** `Transcriber` inherited `--device`'s default of
  `cuda`; on a Mac that raises inside Whisper's module walk, which the best-of-N scorer's broad
  `except` reported as "scored by waveform defects (transcription off)" while everything kept
  working. Now `resolve_device()` falls back to CPU and says so at startup. `test_resolve_device_falls_back_without_cuda`.
- **`hf download --exclude` was being ignored** by recent `hf` CLI versions ("Ignoring
  `--exclude` since filenames have been explicitly set"), so Whisper dragged its fp32/flax
  duplicates down too. Left as-is (it still completes) but noted in AGENTS.md as a known drift.

### Tests

- CPU tier 93 → **101 checks**: engine-by-platform, quantisation/engine validation, MLX repo
  list ↔ `setup.sh` sync, `LANGUAGE_VALUES` ↔ mlx `lang_code` coverage, `max_new_tokens` →
  `max_tokens` mapping, every generated segment surviving the generator, `_move()` skipping only
  the MLX marker (and still reaching the speech tokenizer for torch), and the device fallback.

## [Unreleased] — a mismatched reference transcript is now refused, not warned about

### Changed

- **`generate_voice_clone` refuses a reference whose transcript isn't the clip's own words.**
  Below **`REF_TEXT_CONF_BLOCK = -3.5`** it returns a status naming the confidence and printing
  what Whisper heard, instead of spending a minute of GPU on audio that is very likely wrong.
  The warn line stays at **−2.0** and still generates: an imperfect transcript is a thing to
  edit, not an error to obey, and blocking there would be a coin flip — the half-wrong class
  straddles −2.0 (−2.79 on one clip, −3.18 on another).

  The threshold was previously recommended on statistical grounds. It now has a **cost**: with a
  mismatched transcript, **12 of 12** takes failed to say the target line and render time
  doubled (5.6s of speech became 9.5–11.1s, worst 18.2s) — and that happened to a *clean* clip
  too, so a good recording is no defence. Accurate transcripts in the same test read −0.40 and
  −0.43; the mismatched pairs −3.95 and −3.99, leaving the populations 3.4 apart at their
  closest. `scripts/measurements/replicate_wer.py` reproduces it.

  Honest limit, documented rather than papered over: the check runs when Whisper is resident
  (best-of-N, Transcribe, or the checklist load it). A single-candidate clone with Whisper not
  yet loaded gets **no verdict, and therefore no block** — absence of a check is not a pass.

### Added

- **`scripts/measurements/` — the scripts behind every number in the docs.** They lived in
  gitignored `tmp/` while `AGENTS.md` and `docs/quality.md` cited them by path, so on a fresh
  clone the evidence for those numbers did not exist. Now tracked, and each walks up to the
  project root so it runs from anywhere. `calibrate_and_first_piece.py` no longer names a
  private Japanese voice — it discovers whatever is in `voices/`, and its docstring says which
  row of the published table needs a clip you supply yourself.

### Tests

- CPU tier **93/93**. `test_reference_transcript_check` asserted the old behaviour ("it warns,
  it does not block"); it now asserts the refusal, that the refusal names the number and what
  Whisper heard, that the **warn band still generates** (proving the line is at −3.5, not −2.0),
  and that no verdict means no block.
- GPU tier: see below.

## [Unreleased] — Snappier Speak/Chat; reference transcript check

### Added

- **`scripts/multi_clip_lab.py` — is a clone from several clips better than your one clip?**
  An experiment with its own web UI (port 7861, the same login options and single-owner lock
  as the app). Add 3+ clips of one person, or one long recording, cut into ~6s clips. Pick
  the clip you'd use in the Clone tab, which becomes the baseline and is never held out.
  **Screen** runs Whisper for transcripts you can correct, flags outliers and reference
  problems, and gives each clip a cleanliness score. **Render** makes, on the same sentences
  and seeds:
  - your clip;
  - your clip's codes with the averaged speaker embedding, so only the embedding differs;
  - all clips joined;
  - the cleanest clip;
  - the averaged embedding alone.

  **Listen** is a blind A/B. You hear the real person (a held-out clip), then your clip's take
  and one other version, in random order and loudness-matched, and click A, B or *can't
  tell*. Undo is there for a misclick. **Results** gives each version's tally against your
  clip with an exact sign test, and the same paired comparison on held-out SIM. It also
  checks the listener: identical pairs, and a bias toward one side. `--render-only` runs
  headless and `--listen-only` A/Bs an old run without a GPU. Nothing in the app or saved
  voices changes yet.

- **A render cache for Speak and Chat.** Every sentence they render is kept per voice. The
  same sentence in the same voice, language, sampling settings and weights replays from disk
  instead of costing 7–11s of generation. "Tests passed." is instant the second time, and a
  reply that shares sentences with an earlier one is partly instant. The key is content, so a
  hit is never stale. Post-processing runs after it. Takes that failed the pace or clipping
  check are never stored. Deleting a voice deletes its renders. Capped by
  `QWEN_TTS_RENDER_CACHE_MB` (256, `0` disables) and `QWEN_TTS_RENDER_CACHE_TTL_SECONDS`
  (7 days since last use). The Clone and Design tabs never use it. Trade-off: a cached line
  sounds identical every time.
- **Chat's first piece is short** (4–10 words, cut at a sentence end or punctuation), so the
  first audio arrives sooner. It measured 11.7s before; the new figure is not yet measured.
- **Whisper checks the reference transcript.** Generate grades the transcript against the
  reference with the same scorer used for takes. A transcript of the right length with the
  wrong words, which the length check misses, produces a warning showing what Whisper heard,
  both in the status and in the checklist. It runs only when Whisper is loaded or best-of-N
  is about to load it, and the result is cached per clip and transcript. The threshold
  (`REF_TEXT_CONF_WARN = -2.0`) warns and never blocks; it is calibrated below.

### Fixed — the multi-clip lab was reading noise as signal

The first real run of the lab (one speaker, 24 clips from a single long recording, Japanese,
16 takes per version, 51 of 64 pairs rated by ear) produced a table that argued against the
listener, and the two causes were both in the scoring layer:

- **SIM wins/losses counted any delta above `1e-9`.** Re-rendering one version with only the
  seed changed moves held-out SIM by SD **0.0055**, while the whole spread between five
  versions was **0.0027** — so a +0.0001 wobble was tallied as a win, and the table printed
  `13/3 (p=0.02)` for a difference no one could hear. Its own advice ("only if held-out SIM
  does not disagree") then told the reader the ears were wrong. There is a `SIM_DEADBAND` of
  0.0025 now; the same run reads 6/1 (p=0.12), which agrees with the listener. A sign test on
  a continuous measure needs a dead band, or it manufactures significance against any baseline.
- **The column that mattered was hidden.** Mean WER showed `0.10` against `0.00` and looked
  like a rounding story, when in fact the baseline clip failed to say the sentence in **12 of
  16** takes and the cleanest clip in **0 of 16**. The table now counts failed takes and tests
  them paired (McNemar's exact on the pairs where the two disagree): p=0.003, 0.002, <0.001.

It also prints the take-to-take noise next to the deltas, so a reader can see whether a delta
is bigger than a re-render, and says **"you can stop rating"** once every version is flat on
both measures — 64 pairs was never going to resolve a zero effect, and someone gives up an
hour to rate them.

And the Run dropdowns are re-listed on every page load. They were populated once, when the UI
was built, so a run rendered after the server started vanished from both tabs the moment
anyone reloaded — Listen and Results could not reach it even though every WAV was on disk, and
it looked like the render had never happened.

The experiment's own answer, recorded in [docs/quality.md](docs/quality.md): **averaging the
speaker embedding does nothing you can hear** (every ear verdict "no clear difference", every
Δ SIM inside one take's noise), but **which clip you pick matters a lot for getting the words
right** — the most *typical* clip mis-said the line three times in four, the *cleanest* never
did. Typicality does not predict intelligibility; cleanliness does.

CPU tier 93/93 (`test_multi_clip_results_do_not_read_noise_as_signal`,
`test_multi_clip_lab_lists_runs_rendered_after_startup`); GPU tier 8/8, no skips.

### Tests

- CPU tier 88/88 (four new tests: quick start, cache replay/invalidation/deletion, cache
  safety and bounds, the reference check end to end).
- GPU tier **run on the 5070 Ti: 8/8** — see the next subsection. `smoke_test.py` now checks that repeats replay faster in `speak_api` and `chat_stream`, and prints right-vs-wrong transcript confidences in `media_pipeline` for calibrating the threshold.

### Measured on the GPU box (RTX 5070 Ti), and one hardening fix

The tier this was pushed without has been run: **8/8**, and the two new assertions hold —
`speak_api` repeats a line in `0.01s` against 3.7s for the first call, `chat_stream` repeats a
reply with `first audio 0.01s, all 0.04s` and `4 from the render cache`. The tier's private
cold cache left the real `tmp/render_cache` untouched (0 files afterwards).

- **`REF_TEXT_CONF_WARN` calibrated.** The tier prints one pair (-0.04 right, -7.02 wrong);
  one pair cannot place a threshold, so four references were run against four degrees of
  wrongness, plus noise: exact **−0.12..−0.51** (broadband noise to SNR 5 dB moves it <0.05),
  one word wrong −1.11..−1.34, half wrong −2.79..−3.18, unrelated −4.59..−5.09. **−2.0 stays**
  — it clears a single wrong word by 0.66 and fires around 40–50% wrong, which is what a warn
  line should do. It is **not** a safe block line: the half-wrong class straddles it (-2.79 on
  one clip, -3.18 on another), so blocking on it would be a coin flip at the boundary. If it
  ever blocks, **−3.5** — 3.0 below the worst correct transcript measured, above every
  half-wrong case, so it fires only on "that is not this clip at all". Left to warn-only, as
  you asked. Bands recorded at the constant and in `docs/quality.md`.
- **Quick start measured, and attributed.** Same reply, warm hub, cache off, only the first
  piece's length differing: first audio **11.49s → 5.30s** (26 words → 10), whole reply ~2s
  *slower* because a fourth piece is a fourth model call. The tier's 2.2s and a cold hub's
  7.14s are the same render at different points in a session — promotion (~2–3s) and building
  the clone prompt (~2–3s) are inside "time to first audio" the first time you ask.
- **The short first piece, as far as measuring goes.** It ends in silence (last 80ms RMS
  0.000 — not cut mid-word) and keeps its top end (energy above 9kHz 0.051 against
  0.052/0.033/0.042 for the later pieces). **That is not listening** — the pieces are written
  to `tests/output/first_piece_*.wav` for that. Also found by measuring: quick start leaves a
  15-word first clause alone when the first comma is at 15 words, so "4–10 words" is a target,
  not a promise.
- **Hardening:** `RenderCache` now re-slugs the name it builds filenames from. The slug comes
  from the voice record, and `list_voices()` reads that field from the sidecar JSON — disk-
  supplied text on a **write** path, so a hand-edited `voices/<x>.json` with
  `"slug": "../../escape"` could have written outside `tmp/render_cache/`. Now pinned by a test
  alongside the damaged-entry and LRU checks.
- One probe artifact worth knowing: the Japanese clip's "one word off" row read identical to
  exact. That was the probe (it built the variant by splitting on spaces, a no-op in
  Japanese), not the app: `word_error_rate` and `transcript_fit` count CJK per character, now
  pinned by `test_cjk_is_scored_per_character`. An earlier note here blamed the app; corrected.

## [Unreleased] — UI reorganised; clone quality

### Changed

- **The UI is reorganised around one Status box per tab.** The Clone tab had seven status
  areas and two save paths. Now it is three numbered steps in two columns: reference on the
  left (Upload / Record / From library, waveform, selection, a read-only *Reference clip*
  player, transcript), say and save on the right. The GPU/RAM line moved to a sidebar on a 5s
  timer instead of being appended to every status. Settings (post-processing, sampling) live
  in the same sidebar. The Clone tab is now the first tab.
- **One Save per tab, saying what it saves.** On the Clone tab a radio offers *the reference*
  or *take #k*; one button saves it. **Saving never overwrites** an existing voice: it used to
  replace it silently. Library writes are atomic.
- **A new Voices tab** lists the library in a table: preview, choose the Speak/Chat route per
  voice (`prefer`: auto / clip / description), load a voice back as a clone reference, and a
  two-step delete. Every voice list on every tab refreshes after any library change.
- **Best of 3 is the Clone tab default** (was 1).

### Added — clone quality

- **Reference cleanup** (`prepare_reference`, on by default, switchable): DC removal, 60Hz
  high-pass, edge trim to ~0.2s. No denoise and no level change. Used by Speak/Chat too.
- **Auto-pick scores speech coverage, not loudness**, penalises clipping, and snaps both edges
  into pauses so the reference doesn't start or end mid-word. Uploads start on that pick.
- **A live reference checklist**: length, SNR estimate, speech coverage, mid-word edges,
  clipping and transcript plausibility, recomputed whenever the reference or transcript changes.
- **Transcribe sets the target language** from what Whisper heard when it was on Auto.

### Fixed

- **Whisper confidence was scored with a doubled decoder prefix.** The text was tokenised with
  special tokens and then prefixed again, so every take paid for two impossible tokens.
  `CONF_GOOD` moves from -2.60 to a provisional -1.00 pending re-calibration on the GPU box.
- `generate_voice_clone`'s validation exits returned 2 values to 9 outputs (an error toast
  instead of a status).
- Deleting a voice cleared the Clone tab's reference.
- `/voices_available` listed a described voice as `Nones`.
- `scripts/make_voice.py` loaded models without the single-owner lock.
- Media decode is capped (`MAX_MEDIA_SECONDS`, ffmpeg `-t`) and ffmpeg runs under a timeout.
- The clone-prompt cache was duplicated and held one entry; it is one content-keyed LRU of 8.

### API

- `/generate_voice_clone` gains a trailing `clean_reference` input. `/speak`,
  `/voices_available` and `scripts/say.py` are unchanged.
- `delete_saved_voice` returns 2 values; `voice_library_updates` returns 6.

### Verified on the GPU box (RTX 5070 Ti)

The tier that was pushed unrun has been run here, and the claims that needed real audio now
have it.

- **GPU tier 8/8.** Loads 1.5–3.0s, generation 7–11s, chat first audio 11.7s, `long_text` 2.97
  words/s at WER 0.19, every clip inside the RMS/peak/duration bounds.
- **The confidence fix is confirmed**: clean takes read `−0.02`/`−0.06` where the double-prefix
  scorer read ~−1.9; a real Japanese take reads `−0.51`. `CONF_GOOD = −1.00` is no longer
  provisional, and `docs/quality.md` carries the new table.
- **Reference cleanup measured** — same reference, same line, same seeds, identity scored
  against the raw clip: SIM 0.9870 → 0.9886, WER 0.000 in both arms, energy above 9kHz 0.0472 →
  0.0470, the reference's own embedding 0.9991. Neutral-to-marginally-better with no top-end
  cost, which is the claim; the "not yet measured" note is replaced by the table.
- **Language detection works on real audio** (`ja` on the Japanese clip, `en` on fox's clip),
  and **`make_voice.py` refuses to start while the server holds the lock** — exit 2, names the
  holder, suggests `--no-clip` — while `--no-clip` runs unlocked as intended.
- **Deployed and probed over HTTP**: page serves with all five tabs, `say.py --list` shows the
  `described` column, `/speak` returns audio for `fox`, forged paths to `/import_media` refused.

Still needing a browser, not a GPU: microphone recording into the Record sub-tab, the
Voices-tab row-click, the two-step delete, and the speech-coverage auto-pick on a genuinely
messy long upload (cleanup is measured; *which window* gets picked is not).

### Fixed (review pass)

- `scipy` is now a direct requirement. `prepare_reference` imports it for the 60Hz high-pass and
  falls back silently when it is missing — so without this, a venv that lost librosa would drop
  the filter with no message.
- The single-owner lock's error still said "two copies of ~13GB" from the three-checkpoint era;
  it now says what a second process can actually cost: up to two ~4.3GB checkpoints plus
  Whisper, ~9GB or more (`--max-cpu-resident` defaults to 2).

### Fixed — a saved designed voice keeps the transcript it was generated from

Design's Save read the transcript, description and language from the text boxes at the moment
you clicked Save, not from the Generate that produced the clip. Generate, type your next line,
then save the take you liked: the voice was stored with a transcript that does not match its
audio — the worst clone failure measured (17s of audio for a 5s line, WER 1.0) the next time
it is used as a reference. The take-state now also records the description, and
`save_designed_voice(name, take_state)` takes no box inputs at all; the status quotes the
transcript it saved. The Clone tab's take save already worked this way.

### Fixed — what a saved voice actually contains

**Saving a designed voice stored the post-processed listening copy.** `save_designed_voice`
read the audio box, and the box holds the copy meant for ears: 48kHz, de-imaged, high-shelf,
soft-limited. So every voice saved from the Design tab went into `voices/` already through an
EQ curve — and since a saved voice is usually a future clone reference, it then got cloned from
that, through the codec a second time, on a path that already keeps only ~60% of whatever it is
handed. A voice saved from the browser measured `sample_rate: 48000`; that was the tell.

`generate_voice_design` now writes the raw decode with `_write_take` and hands back
`{path, text, language}` as a third output, and Save reads that path through `gr.State` — the
same shape the Clone tab already uses for its takes, so "what the last generation produced" is
one pattern rather than two. If the write fails the audio still plays, the status says why, and
Save is disabled instead of quietly saving something stale.

- **No API break.** The third output is a `gr.State`, and Gradio omits `gr.State` from
  `/gradio_api/info` — verified against the running server, `/generate_voice_design` still
  advertises 8 inputs and 2 returns, so scripts are unaffected and the server-side path never
  leaves the box. (The Clone tab's take-state has always worked this way.)
- `test_saving_a_designed_voice_keeps_the_raw_decode` replaces the test that pinned the old
  behaviour (its name was the bug: *uses the audio box*). It stubs the hub, generates with
  `postproc="Warm"`, and asserts the player gets 48kHz while the saved file is 24kHz and its
  samples are the model's — plus the confinement check, because that state round-trips through
  the browser and whatever arrives is untrusted.
- `voice_library` (GPU tier) now walks the real callbacks rather than calling `save_voice`
  directly, and asserts `saved @ 24000Hz` against `player copy @ 48000Hz` on real audio.
  Tier 8/8.

## [Unreleased] — CustomVoice removed; voices can ship as descriptions

### Removed

- **The Preset voices tab, and the CustomVoice checkpoint with it.** Nine speakers of someone
  else's, a second 4.3GB download, and a path around the thing this app is for. The product is
  your voices: designed, cloned, saved, spoken. `CAPABILITIES` is now two entries, `SPEAKERS` /
  `SPEAKER_INFO` / `speaker_choices()` / `default_speaker()` / `generate_custom_voice()` are
  gone, `setup.sh` downloads ~9.3GB instead of ~14GB, and the smoke test drops `custom_voice`
  and `dialect` (8 cases). The language dropdown keeps its dialect guard — re-adding a
  `*_dialect` value still raises at generate time — and
  `test_language_choices_match_checkpoint` now reads the VoiceDesign config, which is the
  checkpoint that ships.
- The header tagline. The tab names say what the app does.

### Added

- **A saved voice can be a description rather than a clip.** `voice_kind()` routes it: a
  record with a description is spoken by Voice Design, one with a clip is cloned, and the
  dropdown labels them `Fox (described)` vs `Booth (3.9s)`. Speak and Chat prefer the
  description because it is measurably the fuller sound — energy above 9kHz on identical
  lines: designed **2.67m**, clone of a real clip 0.89m, clone of a generated take **0.57m**.
  The vocoder is not the cause (white noise crosses it at 95%, an output round-trips at 90%);
  `generate_icl_prompt` prefixes the reference's own codec codes, so a clone inherits the
  reference's spectrum, quantisation included. The clip is still kept and still what the Clone
  tab loads, because a frozen clip is the steadier identity (pairwise 0.993 vs 0.974).
- **`fox` ships as a paragraph, not a recording.** `scripts/make_voice.py` renders her at
  `setup.sh` time (`--voices` re-runs it; `--no-clip` needs no GPU). No one's voice in the
  repository, and the route that measures best.
- `load_saved_voice` refuses a clip-less recipe and names the alternative instead of failing
  with "missing its audio".
- `speak_api` (GPU tier) also speaks a described voice, asserting it acquires `voice_design`
  and never touches the clone path — the route `fox` depends on.

### Tests

- CPU tier 73/73 (two speaker tests went with the checkpoint, one recipe-routing case added);
  GPU tier 8/8.
- `long_text` re-baselined, and its docstring carries the reason: its pacing/WER constants came
  from the `ryan` preset, and one pass with the shipped voice measured 3.39 words/s / WER 0.62
  on one run and 1.40 words/s on another — without chunking this checkpoint either rushes or
  rambles, so an in-run comparison against one pass failed by *beating* a degenerate single
  pass. Absolute bands on the chunked output now carry it, with the measurements recorded.

## 2026-11-03 — the auto-pick now says which take it picked, and a saved take reaches every tab

Two things Nick noticed by using it.

**"Which take am I listening to?"** With N candidates, four players are hidden and one clip is
playing, and only a buried line in the takes note answered it. The kept take is now named in
three places: the status opens with `VoiceClone · kept take #2 of 3`, the takes note leads with
**Auto-pick: take #2 of 3**, and the picker's label reads "the auto-pick was take #2; select
another to overrule it". The per-player `(kept)` marker stays.

**"Save selected take said it worked but nothing happened."** It had worked — the saved clip
was on disk — but the confirmation went into a text box the user had already looked away from,
and the voice appeared only in the Clone tab's dropdown. **Speak and Chat never learned the
library had changed**, so a voice saved on one tab was invisible on the others until a reload.

- `voice_library_updates()` refreshes all three dropdowns (Clone, Speak, Chat) plus the
  machine-readable list, and is chained with `.then()` onto every action that mutates the
  library: save reference, save take, save designed voice, delete, refresh. (`.then()` is on
  the *event*, not the `Button` — the first attempt raised `AttributeError` at UI build.)
- The Chat tab gained the **Refresh voices** button it never had.
- The confirmation now says where the file went and where to find it: `Saved take #1 as
  'myvoice' (3.9s, voices/myvoice.wav) — … in the Saved voices list here and in the Speak and
  Chat tabs now.`
- The CPU tier caught a regression from this work: replacing the Refresh chain had quietly
  un-wired `voices_available`, which is the endpoint `scripts/say.py --list` calls. The Chat
  refresh button now owns that chain, and the endpoint test is what noticed.
- Also removed a duplicated `pick_update = gr.update(...)` block left behind in the clone
  callback.

CPU 72 -> 74 checks (auto-pick naming, library refresh fan-out), GPU 10/10.

## 2026-11-03 — Chat audio stopped autoplaying after the first few pieces

Nick hit it while testing: the first three pieces played, then the chat went silent.

The cause is in Gradio's own player, not in the streaming path. Reading the compiled Audio
component, it keeps **one** `<audio>` element and on a value update swaps the `src`; `play()` is
called on the first source load, guarded by a flag that stays set — so every later piece
arrives in the same element and never plays again. That is exactly "plays a few, then quiet"
on a streamed reply, where each yield changes the value.

- The chat player is now **our own element, one per piece**: `_audio_html()` renders a fresh
  `<audio controls autoplay playsinline>` with the wav embedded as a `data:` URI. A new element
  is what the browser needs to honour `autoplay` again, and embedding the audio removes the
  dependency on Gradio's temp-file serving and its cleanup from the playback path entirely.
- `controls` stays on deliberately: if a browser blocks autoplay outright (Safari is the strict
  one), each piece is one click away instead of silently lost. A troubleshooting row says so.
- Verified at the wire, not just in process: an SSE call to `/gradio_api/call/chat_say` delivered
  **3 separate `data:audio/wav;base64` elements, each with `autoplay` and `controls`**, one per
  piece. `chat_stream` (GPU) still asserts time-to-first-audio — 13.6s of a 31.3s reply.
- Tests updated to assert the payload the browser actually receives: the CPU chat test now
  **decodes the base64** and checks rate, peak and length, instead of trusting a tuple that
  never crossed a network.

## 2026-11-03 — named qwen3-tts-ui, polished the UI, added an ethical-use statement, QA pass

The header read `Qwen3-TTS — local (1.7B, 16GB Blackwell)`, which is a commit message, not a
title. The product is **qwen3-tts-ui** and now the repo, the browser tab, the page header and
the README H1 all say so. Tab labels went from checkpoint names to what they do — **Preset
voices**, **Design a voice**, **Clone a voice**, **Speak**, **Chat** — with the checkpoint names
kept in the README table where they are useful. The dev detail (1.7B, Blackwell, LRU swapping)
moved out of the header into the Runtime line, which already carried it. Softer theme, a
readable container width instead of full-bleed, and user-facing strings now name tabs the way
the UI labels them.

**Ethical use**, because voice cloning is the harmful-use surface and offline is not a defence:
a README section (consent, no impersonation, label synthetic audio, the documented harms, and
that `voices/` holds real people's voices), a note on the Clone tab itself — where you are
about to upload someone's voice — and a pointer in the footer.

The QA pass found and fixed:

- **The footer told users to do something the app now does for them**: "Long texts are capped at
  `max_new_tokens=2048`; split very long scripts into chunks" — stale since the chunker landed.
- **AGENTS.md named a repo that no longer exists** (`nkavassalis/qwen-tts`) in two places.
- **Six environment variables were undocumented** (`QWEN_TTS_MODELS_DIR`, `QWEN_TTS_RAM_FLOOR_MB`,
  `QWEN_TTS_VRAM_MARGIN_MB`, `QWEN_TTS_LOCK`, `QWEN_TTS_URL`, `QWEN_TTS_USERNAME`). There is now
  a table, and a test that fails if a `QWEN_TTS_*` knob exists without a README entry.
- **`gradio_client` was a direct dependency of `scripts/say.py` and missing from
  `requirements.txt`** — it only worked because Gradio pulls it transitively.
- The README described three tabs when there are five, and `run.sh`'s own header omitted
  `--no-asr` / `--asr-model`.
- Three new doc-drift tests (72 CPU checks): one name everywhere, every `gr.Tab` label present
  in the README, every env var documented, the stale footer claim cannot come back, and the
  ethical-use language present in both the README and the Clone tab.

Checked and found fine, so it is on the record: no unused top-level functions, no TODO/FIXME
debt, `.gitignore` still covers every user-audio and key path, `--max-resident` and
`--max-cpu-resident` are both live (GPU slot vs RAM cap), and the speaker/language counts in
the README (9 speakers, 10 languages + Auto) match the checkpoint.

## 2026-11-03 — Chat tab: type it, hear it, starting before the reply is finished

Asked whether there could be a chat-style interface with semi-real-time playback. The honest
answer shaped the feature: the model has **no partial-audio API** — `non_streaming_mode` in the
model code is about how text goes *in*, not how audio comes out — so the only streamable
granularity is the sentence. The Chat tab streams at that.

- **`chat_say` is a generator** over the chunker: each piece is post-processed and handed to an
  `autoplay=True` player the moment it exists, while the rest is still decoding. Measured on a
  60-word reply: **first audio 12.6s, whole reply 30.2s**, 3 pieces, 17.0s of speech.
- **`generate_chunked` became a thin consumer of a new `iter_chunked`**, so the chat path and
  the three tabs cannot grow two different chunkers. A single-chunk reply still returns the
  model's own decode *without* edge trimming — trimming is for joins.
- Two bugs caught while building it, both now pinned by tests:
  - the hub lock was being held across `yield`s. A client that walks away mid-reply closes the
    generator, and a stranded lock stops the entire app. It is now taken **per chunk inside
    `one()`** — an LRU check per piece is a cheap price for not being able to deadlock the box.
  - every yield handed out the *same* mutated message list, so all updates render the final
    state. Yields now carry copies.
- Tests: CPU 65 -> 68 (progressive hand-offs, log text exactly equal to the input, the reply
  filling in over time, failures reported in the bubble instead of killing the chat, per-chunk
  locking, Enter submits, autoplay really set). GPU 9 -> 10 flows with `chat_stream`, which
  asserts time-to-first-audio is well before the end rather than trusting the claim.
- Documented rather than oversold: at ~1.4x realtime this is "read this to me now", not a
  conversation partner.

## 2026-11-03 — assistant API: one call to make a saved voice speak, plus `scripts/say.py`

The Clone tab's flow (resolve a voice, upload its wav, supply the transcript) is three calls
and a lot of client-side knowledge — fine for a person, wrong for a coding assistant that
just wants to say something. So there is now one endpoint and one command:

```bash
scripts/say.py --list
scripts/say.py --voice fox_reference --text "Build finished. Three tests failed."
echo "The migration is done." | scripts/say.py --voice ryan
```

- **`speak(voice, text, language, temperature, top_p, top_k, repetition_penalty, postproc)`**
  and **`voices_available()`**, both exposed as a visible **Speak (assistant API)** tab so they
  are named endpoints rather than hidden plumbing.
- **The caller names a voice; it never names a path.** `_resolve_voice` reads the clip and the
  transcript from `voices/<slug>.json` server-side, slugging the name to `[a-z0-9_]` first, so
  there is nothing to traverse — `say.py` has no `--ref-audio` on purpose, and a test asserts it
  stays that way. Falls back to x-vector when the sidecar has no transcript instead of failing,
  and the status reports which path it took.
- It reuses everything: chunked long text, post-processing, the memory floors, the single-owner
  lock. Consequences are documented rather than discovered: with `default_concurrency_limit=1`
  a long line from an assistant queues a UI generation behind it.
- `say.py` mirrors the server's credential precedence (`--password-file` > `QWEN_TTS_PASSWORD`
  > `--password`, warned), supports `--url`/`--insecure`/`--out`/`--no-play`, and picks the
  first available of paplay/ffplay/aplay/afplay.
- Tests: CPU 61 -> 65 (validation before the GPU is touched, hostile voice names refused,
  display-name and slug resolution, `--list` content, both functions really being endpoints,
  and the CLI staying a thin client of those two endpoints). GPU 8 -> 9 flows with `speak_api`
  covering ICL, x-vector fallback and the unknown-voice error. Verified live over HTTP: 3.68s @
  48kHz, one sample in 176k grazing full scale (not clipping), RMS 0.144.

## 2026-11-03 — long text is chunked and stitched, because tempo is the real ceiling

Started as a question: is `max_new_tokens=2048` costing us quality, and how far could this box
actually go? Measured single-pass on the 1.7B checkpoint (full table in the README):

| words | audio | tokens | wall | peak VRAM | words/s | WER |
|---|---|---|---|---|---|---|
| 18 | 5.9s | 74 | 8.7s | 4.36 GB | 3.05 | 0.00 |
| 300 | 74.3s | 929 | 100.7s | 7.58 GB | **4.04** | **0.68** |

The cap is not the limit and neither is VRAM: 300 words needs 929 of 2048 tokens, and raising
the cap to 4096 produced *identical* audio because EOS arrives on its own. VRAM sat at 7.5-7.6 GB
across 221–929 tokens (~112 KB/token of KV cache). What degrades is **pacing** — 3.05 → 4.04
words/s — and intelligibility with it. Usable ceiling: ~30-45s of audio per pass.

- **`split_for_tts`**: real sentence boundaries only, after protecting the dots that are not full
  stops (`Mr.`, `3.5`, `J. R. R.`, `...`, URLs, emails, `1.` list markers), then the measured
  band — 12–26 words or 20–42 CJK characters — merging fragments and cutting over-long sentences
  at the clause boundary nearest their middle, separator kept with the left piece so intonation
  stays non-final. Lossless by construction; deterministic, so the seed promise survives.
- **`stitch_clips`**: trim each piece's edge padding, join with a 15 ms equal-power crossfade.
  Naive concatenation stutters and clicks.
- **`generate_chunked`** wraps all three capabilities, re-rolls a chunk whose speaking rate lands
  out of band, reports the piece list in the status, and refuses >40 chunks with a `MediaError`
  instead of running for an hour. `max_new_tokens` is now budgeted per chunk.
- Same 300 words after: **2.48 words/s, WER 0.42** (from 4.04 and 0.68), 92.6s of audio in 16
  pieces, 131s wall against 101s single-pass.
- Two bugs found on the way. `_rms_windows` padded with a **negative** width for any length that
  was not an exact multiple of the window — latent because every earlier caller passed round
  numbers. And ideographic terminators had to be written by escape: the typed `。` had come out as
  an ASCII `.`, so Chinese never split at all.
- Tests: CPU tier 49 -> 61 (losslessness over a corpus of traps, false-boundary protection, band
  limits, CJK by codepoint, determinism, the 40-chunk refusal, trim/crossfade/click behaviour on
  phase-shifted tones, re-roll logic, all three callbacks wired). GPU tier 6 -> 8 flows with a
  `long_text` case asserting pacing, gaps and levels. Clicks are asserted in the CPU tier, where
  the seams are knowable — a global max-diff bound on real speech is meaningless.

## 2026-11-03 — keep the take you preferred, not just the one the scorer picked

The voice library held only *references* — the clip a voice was taken from. So the take you
actually preferred, once you had listened to all four, had exactly one way out: the browser
download icon. Turning it into a reusable voice meant uploading it back, re-trimming it and
transcribing it.

- Each candidate is now written to `tmp/media/` at generation time; its **path** travels
  through `gr.State` (`build_take_state`), never a module-level "last generation" global —
  that global is the bug class that once let one LAN user's Save button persist another
  person's clip. Files are covered by the existing `MEDIA_TTL_SECONDS` prune.
- Step 3 gains *Which take to keep* (a radio carrying the same score labels as the players,
  defaulting to the auto-pick) and **Save selected take**, which writes
  `voices/<slug>.wav` + `.json` with the requested line as the transcript.
- The saved copy is the **raw 24kHz model decode**, not the 48kHz post-processed listening
  copy: a saved voice is usually a future clone reference, and a reference is better coming
  straight from the model than through an EQ curve. Asserted in the GPU tier.
- A forged state payload is still confined: the path goes through `resolve_input_path`, so a
  client cannot point the Save button at `/etc/passwd`.
- Tests: CPU tier 45 -> 49 (state is JSON-safe and flags the winner, name/selection/empty
  validation, real save round-trip incl. transcript and refusal of an out-of-tree path,
  picker wiring). The GPU tier now saves a non-preferred take end-to-end and checks the
  stored wav is 24kHz — 7/7.

## 2026-11-03 — candidate scoring gets two more signals, both calibrated by measurement

WER alone was a weak judge: a hard 0/1 per word, blind to a take that is half dead air and
indifferent between four takes that all say the words right.

- **Waveform defects** (`defect_report`, CPU-only so it survives `--no-asr`): speaking rate
  against text length (CJK counted per character), leading/trailing silence, clipping.
  Capped at 0.3 so they can never outrank content. On the measured four-seed run this is
  what changed the pick: all four takes score WER 0.00 and similarity prefers #1, but #1
  carries 0.9s of dead air, so #4 is kept.
- **Whisper decoder confidence** (`Transcriber.score_transcript`): mean log-probability of
  the *intended* text given the candidate audio, from a teacher-forced forward pass.
- Ranking is now `candidate_penalty` = WER ×2.0 + confidence ×0.4 (only below -2.60) +
  defects (≤0.3), with similarity as a tie-break. The players and the status line show every
  number, and the winner is marked `(kept)`.
- Three findings that only measuring could give, all in the README:
  * confidence reads **~-1.9 on a clean take**, so the guessed threshold of -0.35 would
    have penalised every candidate and swamped WER. It is a disaster detector (pure silence
    -6.3), not a quality meter: tail silence, clipping and half-speed pacing move it <0.15.
  * scoring Whisper's own transcript instead of the requested text **misses** degradation —
    on a noisy take: -1.91 (missed) vs -2.07 (caught). That is self-consistency, not audio.
  * an off-by-`prefix_len` in the log-prob gather reads **-13.5, worse than uniform**, and
    raises nothing. Now pinned by `test_mean_token_logprob_alignment`, since the GPU tier
    cannot tell a plausible-looking wrong number from a right one.
- Also: `output_scores` is silently dropped by Whisper's generation-config handling (only
  `logprob_threshold` / token timestamps force it) and `sequences_scores` is None — hence
  the second forward pass.
- Tests: CPU tier 40 -> 45 (defect report, penalty ordering and caps, log-prob alignment,
  take labels with confidence/defects). GPU tier unchanged at 7/7, still asserting three
  labelled visible takes with exactly one `(kept)`.

## 2026-11-03 — Clone tab reads as three steps; every candidate is auditionable

Two usability problems, both real: *Save this reference as a voice* sat between the
reference section and the synthesis section (so people looked for it in the wrong place),
and *Target text* named nothing (it is the line to synthesise, and it sat under a box that
is also a text box about the clip).

- The tab is now three numbered sections in the order the workflow actually runs: **1 · The
  reference** (upload, waveform, trim, transcribe, transcript, x-vector, fit check) → **2 ·
  What it should say** (renamed *Say this in the cloned voice*, language, candidates, seed,
  generate, output) → **3 · Keep it** (save + the saved-voices list together, where the
  save button can be seen next to what it feeds). The two text boxes now say which is which.
- **Best-of-N no longer throws away the losers.** The callback returns one audio player per
  take, each labelled with its scores (`take_label`) and marked `(kept)`, alongside
  **Generated audio (the auto-pick)**. Every player has its own download icon, so the scorer
  is a suggestion rather than a gate. With one candidate the players and the explanation row
  stay hidden.
- Documented what the scorer is and is not good at, from the measurements: it rejects dud
  takes (WER 0.133 vs 0.000 on identical inputs), but among correctly-pronounced takes SIM
  varies by ~0.001 and neither metric measures naturalness - so ties are for the human.
- Tests: CPU tier 37 -> 40 (reading order + box labels, take labels, player wiring). The
  GPU tier asserts 3 visible labelled takes with exactly one `(kept)`, and none when N=1.
  Gradio 6 gotchas caught by the CPU tier on the way: `gr.File` and `gr.Audio` accept no
  `info=`.

## 2026-11-03 — output post-processing: the "digital" edge is imaging, not sample rate

The vocoder is fixed at 24kHz (`output_sample_rate: 24000`), so no post-filter can add
bandwidth above 12kHz. Measured instead: the raw decode carries 0.155% of its energy in
the top 5% of the band against 0.036% for the same signal resampled properly — a 4x excess
of near-Nyquist imaging from the decoder's transposed-convolution upsampling, which is
what listeners call digital/harsh. A cheap linear resampler (what a browser may do for
you) cost 20% of the high-band energy, so the 24->48kHz step is done here with soxr HQ.

- `postprocess_output()` + an **Output post-processing** dropdown on all three tabs:
  Off (raw 24kHz) / Clean (HQ 48kHz) / Warm (48kHz + lowpass 11.5k + small high-shelf +
  mild tanh), default Warm. Warm cuts near-Nyquist energy ~6x and ends up brighter than
  raw (centroid 2164 vs 2044 Hz), because the lowpass is followed by the shelf.
- Likeness is not the price: speaker-encoder SIM went 0.9900 (raw) -> 0.9893 (clean) ->
  0.9879 (warm), inside the 0.001 seed-to-seed spread. Both metrics are in the README.
- Runs after `_report()` and after candidate scoring, so durations/WER/SIM still describe
  the model's own decode; falls back to the raw signal on any failure (missing soxr/scipy,
  a clip too short for the filter padding) because polish must never break a generation.
- Tests: CPU tier 35 -> 37 (mode fallbacks, duration, full scale, and a functional
  assertion that Warm really reduces near-Nyquist energy). GPU tier asserts 48kHz out and
  24kHz when Off. The new tests caught a real bug: a sub-50-sample clip raised inside
  `sosfiltfilt`, now guarded.

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
