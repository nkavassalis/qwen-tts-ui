# AGENTS.md — working on qwen-tts

## Project mode: REAL

GitHub `nkavassalis/qwen-tts-ui`, **private** until Nick says otherwise; default branch
`main`. Verify `gh auth status` shows `nkavassalis` before any push. Both test tiers
green before any push or handoff (the GPU tier cannot run on a box without a GPU — see
*Testing limitations*). Secret scan before every commit.

## Purpose

A local, offline deployment of [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
(1.7B checkpoints) behind a Gradio UI, tuned for **one** 16GB Blackwell consumer GPU —
RTX 5070 Ti, RTX 5080, anything in that class (the 5070 Ti is the SKU it is developed and
tested on, not the only one it targets). The product name is **qwen3-tts-ui** (the repo, the
browser tab title and the README H1 all agree). Five tabs: **Clone a voice** (from a ~3-10s
reference clip), **Design a voice** (from a text description), **Speak** (the one-call
endpoint `scripts/say.py` uses), **Chat** (streams a reply sentence by sentence) and
**Voices** (the library). Settings and the runtime line live in a sidebar. Nick uses it at
his desk; it is not a service.

Two checkpoints ship (`VoiceDesign` + `Base` + the tokenizer, ~9.3GB). The third upstream
capability, CustomVoice with its nine preset speakers, was removed: the product is your
voices, not Alibaba's, and every preset was a path around the thing the app is for.

**Voice cloning is the harmful-use surface.** The README's *Ethical use* section and the Clone
tab's own note are part of the product, not decoration: keep them, and keep any new cloning
affordance (voice sharing, export, batch synthesis) next to that language. Offline does not
make impersonation acceptable, and it is not a defence.

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
scripts/say.py             CLI over the assistant API: /speak + /voices_available, plays
                           through paplay/ffplay/aplay/afplay
scripts/make_voice.py      create a described voice (takes the single-owner lock to render)
requirements.txt       direct deps only (torch/torchaudio and transformers/accelerate are
                       pinned elsewhere — read the file header before adding to it)
tests/test_app_logic.py  CPU tier: 88 checks, no weights, ~10s with deps installed
tests/smoke_test.py      GPU tier: 8 end-to-end flows, writes tests/output/*.wav
docs/quality.md          every quality measurement and what the app does about it
docs/design.md           the why behind the code shape, including the UI layout rules
LICENSE                  Apache-2.0 (matches upstream; app.py derives from their Space)
README.md              user guide: workflow, safety, configuration, troubleshooting
CHANGELOG.md           one entry per change, newest first

gitignored, created on demand:
.venv/  models/ (~9.3GB weights)  repo/ (upstream checkout)  voices/  tmp/media/
certs/  logs/  tests/output/  .qwen-tts.lock  .gradio/
```

## How to run

```bash
./setup.sh                       # uv venv + torch cu128 + upstream + 9.3GB weights + Whisper + fox
./run.sh                         # -> http://127.0.0.1:7860; GPU/RAM status is in the sidebar
./run.sh --ip 0.0.0.0 --username u --password-file ~/tts.pw   # LAN, authenticated
./run.sh --ssl-self-signed --ip 0.0.0.0 --username u --password-file ~/tts.pw   # browser mic needs HTTPS
./run.sh --device cpu            # CPU inference (slow; works, no CUDA needed)
```

You should see `[webui] transcription: … | ffmpeg: …`, then a `GPU: … | RAM: …` line, then
`* Running on local URL`. If the second one never appears, check `nvidia-smi` (VRAM) and
`free -m` (RAM) — the hub refuses to over-commit rather than swapping.

## How to run the tests

```bash
.venv/bin/python tests/test_app_logic.py   # CPU tier — 88 checks, ~10s
.venv/bin/python tests/smoke_test.py       # GPU tier — 8 flows, ~2 min, needs weights
.venv/bin/python tests/smoke_test.py --only media_pipeline
```

Current state (2026-11-04, render cache + quick start + reference transcript check):
**CPU tier 88/88 and GPU tier 8/8 on the 5070 Ti** — pushed unrun from a CPU-only box and
executed here; see *Testing limitations* for what that run measured. Before them (2026-11-03):
**CPU tier 84/84**, Python 3.12,
torch 2.8.0 (CPU wheel), gradio 6.17.3, transformers 4.57.3, `qwen-tts` from PyPI. **GPU tier
8/8 on the 5070 Ti** — it was pushed unrun from a CPU-only box and was executed here; see
*Testing limitations* for what that run confirmed.

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
- **Cloning keeps ~60% of its reference's top end, so a described voice is the higher-fidelity
  form.** Measured on the same lines, raw 24kHz decode, energy above 9kHz: designed 2.67m,
  clone of a real clip 0.89m, clone of a generated take 0.57m. The codec is not the cause —
  white noise crosses it at 95% and an output round-trips at ~90% — the top end is missing in
  the tokens the talker chose. `generate_icl_prompt` prefixes the reference's own codec codes,
  so the clone inherits the reference's spectrum, quantisation included. Hence a voice record
  carries a clip, a description, or both, and Speak/Chat prefer the description
  (`voice_kind`, `_voice_synth`). The counterweight: a frozen clip is the steadier identity
  (pairwise 0.993 vs 0.974), which is why the clip is kept and what the Clone tab loads.
- **`fox` ships as words, not as a recording.** `setup.sh` creates her with
  `scripts/make_voice.py`; the repository carries a paragraph, and your GPU renders the voice.
  That keeps anyone's voice out of a public repo (see README → Ethical use) and takes the route
  that measures best. The description is written before any GPU work so the voice exists even
  where the clip cannot be rendered.
- **Dialect is a speaker trait, not a language — and no longer exists here.** CustomVoice
  carried `talker_config.spk_is_dialect` (`dylan → beijing_dialect`) and upstream filters those
  out of `supported_languages` (`modeling_qwen3_tts.py:1831-1834`), so
  `language="beijing_dialect"` raises `ValueError: Unsupported languages`. The capability is
  removed; the guard stays in `test_language_choices_match_checkpoint` so the dropdown cannot
  quietly grow a dialect back. Italian *is* selectable — the hosted Space omits it by oversight.
- **The chat tab streams *sentences*, not samples — and says so.** There is no partial-audio
  API upstream; `non_streaming_mode` in the model code is about text going in, not audio coming
  out. `chat_say` is a **generator** over `iter_chunked` that yields each piece as it is
  generated into an `autoplay=True` player. Measured: first audio 12.6s into a 30.2s reply.
  Two invariants that only bite in production, both tested: the hub lock is taken **per chunk
  inside `one()`**, never across a `yield` (a client that walks away mid-reply closes the
  generator, and a lock held across yields could strand the whole app); and each yield hands
  out a **copy** of the message list, because yielding one mutated list makes every update
  show the final state.
- **`generate_chunked` is a thin consumer of `iter_chunked`** so the chat tab and the three
  tabs cannot drift into two different chunkers. A single-chunk reply returns the model's own
  decode **without** edge trimming — trimming is for joins, and trimming a one-shot answer
  would silently shave the model's own silence handling.
- **The assistant API takes a voice *name*, never a path** (`speak` + `scripts/say.py`). The
  reference clip and its transcript are read from `voices/<slug>.json` server-side by
  `_resolve_voice`, which slugs the name to `[a-z0-9_]` before touching the filesystem — so
  there is no argument to point at `/etc/passwd` and `say.py` deliberately has no
  `--ref-audio`. No transcript in the sidecar falls back to x-vector instead of failing, and
  the status says which (`Speak (ICL)` / `Speak (x-vector)`). It shares the single GPU and
  `default_concurrency_limit=1`, so a long line from an assistant queues UI generations —
  by design, but worth knowing before wiring it into a hook that fires often.
- **Long text is chunked because *tempo*, not tokens, is the limit.** Measured single-pass on
  this checkpoint: 3.05 words/s at 18 words rising to 4.04 at 300, WER 0.00 → 0.68, while peak
  VRAM stayed at 7.5-7.6 GB and 300 words used only 929 of the 2048 tokens (raising the cap to
  4096 produced identical audio — EOS arrives on its own). So `MAX_NEW_TOKENS_CAP` is a runaway
  guard and nothing more; do not raise it hoping for quality, and do not remove it — ICL
  cloning sometimes never stops. The real ceiling is ~30-45s of audio per pass, which is why
  `split_for_tts` + `stitch_clips` run automatically above `CHUNK_THRESHOLD` units.
- **The splitter's one unforgivable failure is losing words**, so `split_for_tts` is lossless
  by construction (whitespace-insensitive equality is asserted over a corpus). Dots that are
  not full stops are placeholdered before scanning — abbreviations, decimals, initials,
  ellipses, URLs, emails, list markers — because a false boundary is spoken with sentence-final
  intonation, which is a quality bug, not a cosmetic one. Ideographic terminators are matched
  by **escape** (`\u3002\uff01\uff1f`), not literal: a typed `。` silently came out as an ASCII
  `.` and Chinese collapsed into one giant chunk.
- **Stitching trims then crossfades** (`trim_edges` + `stitch_clips`, 15 ms equal-power). Click
  behaviour is asserted in the **CPU** tier where the seam locations and the expected waveform
  are known; a global max-diff threshold on real speech is meaningless (measured 0.76 on clean
  stitched audio), so the GPU tier asserts gaps and levels instead.
- **`_rms_windows` pads with `ceil`, not `floor`** — with `floor` the pad width went negative
  for any length that was not an exact multiple of the window and numpy raised. It hid because
  every earlier caller passed round-number clip lengths.
- **A preferred take can be kept as a voice** (`save_generated_voice`, reached through the
  one Save control via `save_clone_voice`). The library used to
  hold only *references*, so the take you actually liked could only leave the browser as a
  download. Each candidate is written to `tmp/media/` at generation time and its **path**
  travels through `gr.State` (`build_take_state`) — never a module-level "last generation"
  global, which is the bug class that once let one LAN user's Save persist another's clip.
  The saved copy is the **raw 24kHz decode**, not the post-processed listening copy: a saved
  voice is usually a future clone reference, and a reference is better straight from the
  model than through an EQ curve. Files fall under `MEDIA_TTL_SECONDS` pruning.
- **The Design tab's Save follows the same rule, and did not used to.** `save_designed_voice`
  read the audio box, which holds the listening copy — 48kHz, de-imaged, high-shelf,
  soft-limited — so every designed voice was stored through an EQ curve and then cloned from
  that, through the codec a second time. A voice saved from the browser measured
  `sample_rate: 48000`, which is how it surfaced. `generate_voice_design` now also writes the
  raw decode with `_write_take` and returns `{path, text, language, description}` as a
  **`gr.State`** output, and Save reads the clip *and its metadata* from state — never from
  the text boxes, which may have been edited since Generate (a transcript that does not match
  its clip is the worst clone failure measured) — same shape as the Clone tab's `build_take_state`, so
  there is one pattern for "what the last generation produced", not two. Being a `gr.State`,
  that output is browser-only: it is absent from the public API, so the server-side path never
  leaves the box and scripts see the same two outputs as before. If the write fails the audio
  still plays, the status says so, and Save is disabled rather than saving something stale.
- **Candidate ranking is a penalty** (`candidate_penalty`). WER ×2.0 dominates; Whisper
  decoder confidence ×0.4 but only below `CONF_GOOD`; waveform defects capped at 0.3; SIM is
  a tie-break only. What was learned, on the real checkpoint:
  * **The confidence scorer had a double-prefix bug until 2026-11-03.** `_confidence`
    tokenised the text with `tokenizer(text)`, which *adds* `<|startoftranscript|>
    <|notimestamps|>` … `<|endoftext|>`, then prepended the decoder prefix again. Every take
    paid for two impossible tokens and its words were scored in a corrupted context — the
    real reason "a clean take reads ~-1.9", and why short lines read worse than long ones.
    Now `_text_ids` uses `add_special_tokens=False` and `_prefix_ids` builds
    `<|startoftranscript|><|lang|><|transcribe|><|notimestamps|>` explicitly, with the
    language Whisper detected. `test_whisper_confidence_scores_only_the_words` pins it.
  * **`CONF_GOOD` is -1.00 (was -2.60), measured on the corrected scale**: clean takes read
    -0.02 to -0.06, a real Japanese take -0.51 (`docs/quality.md`). Silence has not been
    re-measured on this scale. Every confidence figure recorded before 2026-11-03 belongs to
    the buggy scale — do not reuse the old -1.9/-6.3 figures. To re-check, read the per-take
    lines `smoke_test.py --only best_of_n` prints.
  * **Score the text we asked for** (`Transcriber.assess`), never Whisper's own transcript:
    on a noise-degraded take, grading the returned transcript missed it, grading the
    intended text caught it. Whisper is confident about whatever it decided to hear.
    `assess` is one beam search plus one teacher-forced pass per take (it used to be two
    beam searches).
  * **`_mean_token_logprob`'s `offset` is `prefix_len - 1`** — `logits[t]` predicts
    `fed[t+1]`. One token off reads worse than uniform with no error raised.
    `test_mean_token_logprob_alignment` pins it; the GPU tier cannot.
  Confidence comes from a teacher-forced forward pass, not from beam bookkeeping:
  `output_scores` is silently dropped by Whisper's own generation-config juggling.
- **The UI follows four layout rules** (2026-11-03, `docs/design.md`): one **Status** box
  per tab; the runtime line in the **sidebar** on a `gr.Timer` (`concurrency_limit=None`, so
  it never queues behind a generation) instead of appended to every status; one **Save** per
  tab that says what it saves; the library managed on the **Voices** tab, with every voice
  list refreshed by `voice_library_updates` after any change and on page load. The Clone
  tab is three numbered steps in two columns — reference (left), say + save (right) —
  pinned by `test_clone_tab_reads_top_to_bottom`. The reference arrives by Upload (imports on
  `.upload`), Record (`.stop_recording`) or Library; all three land in a **read-only**
  reference player, and the checklist reads that player, so it is right whichever way the
  clip came in. `gr.File` and `gr.Audio` take no `info=` in Gradio 6.
- **Saving never overwrites** (`voice_exists`/`_taken_message`). Two people naming a voice
  the same thing used to silently replace the first one's clip. Writes are atomic
  (temp + `os.replace`). Deleting is two clicks on the Voices tab, and no longer clears the
  Clone tab's reference box.
- **Reference handling is where clone quality is won** (measured: transcript mismatch, noise,
  then length). So: `auto_pick_window` scores speech coverage (adaptive threshold between
  the 5th- and 95th-percentile frame level), penalises clipping, and `snap_to_pauses` moves
  each edge into a pause; `import_media` starts on that pick instead of the first 6s;
  `reference_report` is a live checklist (length, SNR, coverage, mid-word edges, clipping,
  transcript fit); `prepare_reference` (DC, 60Hz high-pass, edge trim to 0.2s, no denoise,
  no level change — level was measured irrelevant) runs before encoding on both the Clone
  tab (switchable) and Speak/Chat. Transcribe sets an Auto target language to what Whisper
  heard. **None of this has been A/B-measured on the GPU yet** — see *Testing limitations*.
- **Best of 3 is the Clone tab default** (was 1). Sampling variance was the one quality
  loss no knob fixed (WER 0.000 on one seed, 0.133 on three); it costs two extra generations.
- **Speak and Chat replay sentences they have already said** (`RenderCache`, 2026-11-04).
  Chat's first audio measured 11.7s and every sentence costs 7–11s, while the lines an
  assistant repeats ("Tests passed.") are the same every time. Per chunk, keyed by an
  *identity* that covers everything that shapes the raw decode: the voice's content (cleaned
  clip PCM hash + transcript + mode via `_clone_cache_key`, or the description), language,
  sampling, the weights (`_model_tag`: folder + config mtime) and `RENDER_CACHE_VERSION`.
  So a hit is never stale and nothing needs invalidating, except on delete, where
  `forget_voice_renders` removes `<slug>__*.wav` because renders of someone's voice must not
  outlive the voice. Raw decodes are stored and post-processing runs after. Only takes that
  pass the rate and clipping checks are stored, since a cached take replays forever. The
  Clone and Design tabs never use it: Generate there must re-roll. **Bump
  `RENDER_CACHE_VERSION` if anything that shapes a raw decode changes** (`_budget`, chunk
  generation, `prepare_reference`), or old takes will replay.
- **Chat cuts its first piece short** (`split_for_tts(quick_start=True)` → `_quick_first`,
  4–10 words). Generation time tracks audio length, so the first piece is the wait you
  notice. Only at a sentence end or punctuation (never " and " and never mid-phrase), and
  only when both halves are ≥4 words. Speak is unchanged. Unmeasured for prosody.
- **The reference transcript is graded by Whisper** (`check_reference_transcript`,
  2026-11-04). The length check (`transcript_fit`) cannot catch the right length with the
  wrong words, and a mismatched transcript is the worst failure measured. The take scorer is
  reused on the reference (`ASR.assess`). It runs in `generate_voice_clone` only when Whisper
  is loaded or best-of-N will load it anyway, so there is never a surprise 3GB load. Results
  are cached per (clip, transcript), and the checklist shows the cached verdict for exactly
  that pair. **`REF_TEXT_CONF_WARN = -2.0` warns and never blocks**, and the scale is now
  measured (`tmp/calibrate_and_first_piece.py`, `tmp/calibrate_2.py`): exact -0.12..-0.51
  (noise to SNR 5 dB moves it <0.05), one word wrong -1.11..-1.34, half wrong -2.79..-3.18,
  unrelated -4.59..-5.09. -2.0 clears a single wrong word by 0.66 and fires around 40-50%
  wrong, so it is right as a warn line. It is **not** a safe block line: the half-wrong class
  straddles it. If it ever blocks, block at **-3.5** — 3.0 below the worst correct transcript
  measured, above every half-wrong case, so it fires only on "that is not this clip".
  `smoke_test.py --only media_pipeline` prints the right-vs-wrong pair.
- **Each voice can choose its Speak/Chat route** (`prefer`: auto/clip/description,
  `voice_route`). Auto keeps the measured default (description when present); a frozen clip
  is the steadier identity across a multi-sentence reply because a description is re-rolled
  per chunk.
- **Every best-of-N candidate stays auditionable.** The callback returns one audio player
  per take (labelled with its WER/SIM, winner marked) alongside the auto-pick, because
  `choose_best_candidate` is only defensible as a *suggestion*: it rejects dud takes well,
  but among correctly-pronounced takes SIM varies by ~0.001 and nothing measures
  naturalness. Do not turn it into a gate that discards the losers.
- **Output post-processing is listening-only** (`postprocess_output`). The vocoder is fixed
  at 24kHz, so the "digital" character is near-Nyquist imaging (measured 4x excess), not
  missing bandwidth — nothing here can add air above 12kHz, and anything that appears to is
  inventing it. Modes: off / clean (soxr HQ 48k) / warm (48k + lowpass 11.5k + small
  high-shelf + mild tanh). It runs *after* `_report()` and after candidate scoring, so
  durations, WER and SIM all describe the model's own decode; it falls back to the raw
  signal on any failure, including missing soxr/scipy, because polish must not be able to
  break a generation. `Off` is the mode to use when saving a Design-a-voice clip into the
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
- **Every `generate_voice_clone` exit returns `CLONE_OUTPUTS` values** (`_clone_failure`).
  The validation exits used to return 2 values to 9 wired outputs, which Gradio turns into
  an error toast instead of a status line.
- **The clone-prompt cache is a content-keyed LRU of 8** (`_clone_prompt`), shared by the
  Clone tab and Speak/Chat (there used to be two copies of the logic, each holding one entry).
- **Media decode is bounded**: `MAX_MEDIA_SECONDS` (600) via `ffmpeg -t` / `sf.read(frames=)`,
  and ffmpeg runs under `FFMPEG_TIMEOUT_SECONDS`. A two-hour upload used to be decoded whole.
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
4. Language dropdown values ⊆ what the checkpoint accepts, no dialect values selectable,
   Italian present. `test_language_choices_match_checkpoint`.
5. A saved voice is spoken by the route its record implies: a description goes through Voice
   Design, a clip through the clone path, and the dropdown labels which is which.
   `test_a_described_voice_is_spoken_from_its_description`.
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
15. `/speak` keeps its 8 positional inputs and `/voices_available` stays public — that is
    the `scripts/say.py` contract. `test_build_ui_constructs`.
16. One Status box per tab; no generation status carries the runtime line.
    `test_build_ui_constructs`, `test_runtime_status_lives_in_the_sidebar`.
17. Saving never replaces an existing voice. `test_saving_never_overwrites`.
18. Every model-loading entry point takes the single-owner lock — the web UI, the smoke
    test and `make_voice.py` (without `--no-clip`). (make_voice: not automated)
19. No Save path may read an audio box or a text box: a saved voice is the raw decode, and its
    transcript/description/language are the ones it was generated from, all reached through
    `gr.State`. `test_build_ui_constructs` pins Design's Save to (name, state); `test_saving_a_designed_voice_keeps_the_raw_decode`, and the GPU tier's
    `voice_library`, which asserts saved @ 24kHz while the player copy is 48kHz.
20. A render-cache hit is only ever a take the exact same configuration produced (voice
    content, language, sampling, weights); a take that failed the rate/clipping checks is
    never stored; deleting a voice deletes its renders.
    `test_render_cache_replays_a_line_already_said`,
    `test_render_cache_never_stores_a_bad_take_and_stays_bounded`.
21. Chat's quick start only ever cuts the first piece, losslessly, at punctuation.
    `test_quick_start_cuts_only_the_first_piece_short`.

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
- **`CONF_GOOD` is calibrated, but thinly.** Measured on the GPU box after the scorer fix:
  clean takes `−0.02` to `−0.06`, a real Japanese take `−0.51`, so `−1.00` is a sane disaster
  line. That is a handful of takes, not a sweep — if best-of-N ever ranks two clearly-different
  takes wrongly, this is the first constant to re-measure.
- **`generate_voice_clone` gained a `clean_reference` input** at the end of its argument
  list. The UI is wired for it; a script calling `/generate_voice_clone` positionally with
  the old 12 arguments needs updating (`/speak` is unchanged).
- **`generate_voice_design` gained a third output**, the take-state `gr.State`. It is **not**
  part of the public API — Gradio omits `gr.State` from `/gradio_api/info`, so
  `/generate_voice_design` still advertises 8 inputs and 2 returns and scripts are unaffected
  (verified against the running server). Same for the Clone tab's `cl_take_state`.
- **Warm post-processing's `tanh` stage runs on every sample**, not only near full scale —
  `tanh(1.25x)/tanh(1.25)` is ~2.3 dB of compression at 0.5. Unmeasured colouring; consider
  a peak-only limiter and A/B it.
- **Designed voices drift across a long reply**: Speak/Chat re-roll the description per
  chunk (pairwise SIM 0.974). The `prefer: clip` route is the workaround; a better fix is to
  render the first chunk by description and clone it for the rest — untried. The render
  cache helps for *repeated* sentences only (each replays the same take).
- **A cache hit still waits in the GPU queue.** `default_concurrency_limit=1` applies to
  the whole event, so a replayed `/speak` queues behind a running generation even though it
  needs no GPU. Fixing that means a separate, unqueued fast path; not done.
- **Ideas not yet tried for clone quality** (from the 2026-11-04 review): best-of-N for
  Speak/Chat clone voices (pairs naturally with the render cache: spend the extra takes
  once); measuring whether x-vector-only clones keep more top end than ICL (ICL continues
  the reference's codec codes, x-vector does not); an opt-in denoiser A/B on noisy
  references (noise was the biggest SIM loss, −0.020); a `say.py --warm` to pre-render
  common lines.
- **`webui/app.py` is ~3,800 lines.** Natural seams for a split: `hub.py` (ModelHub,
  memory guards, lock), `media.py` (ingest, path guards, reference analysis/cleanup),
  `asr.py` (Transcriber), `scoring.py` (penalty, defects, labels), `chunking.py`,
  `library.py`, `ui.py`. Not done yet because the tests import `app` as one module.
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

Run on 2026-11-03 on the same RTX 5070 Ti, covering the length study and chunking:

- **CPU tier 73/73**, **GPU tier 8/8** (CustomVoice and the dialect case went with the
  checkpoint; `speak_api` now also speaks a described voice).
- **Chat streaming measured**: 60-word reply → **first audio at 12.6s**, whole reply 30.2s, 3
  pieces, 17.0s of speech, log text identical to the input.
- **One GPU assertion was corrected, and it is on the record here because it touches the
  do-not-touch list.** `long_text`'s gap check asserted `longest quiet run < 1.0s` with the
  message "joins were not trimmed". It failed at 1.04s. A probe (`tmp/gap_probe.py`, generates
  the same 16 pieces and mirrors `stitch_clips` to record seam offsets) showed the longest quiet
  run on a clean run was **0.64s and 1.76s away from the nearest join** — the model pausing
  inside a sentence. The bound was measuring prosody and blaming stitching. It is now 1.6s with
  a message that names both real causes (an empty piece, or `trim_edges` stopped trimming), and
  join behaviour remains pinned where seams are knowable: the CPU tier's
  `test_stitch_trims_and_crossfades`. The pacing, duration, level and WER assertions were not
  touched.
- **`long_text`'s baselines were voice-dependent, and the voice changed.** Its pacing and WER
  bounds were measured on the CustomVoice `ryan` preset (4.04 words/s, WER 0.68 one pass). On
  the designed voice that now ships, one pass over the same 300 words measured 3.39 words/s /
  WER 0.62 on one run and **1.40 words/s** on another: without chunking this checkpoint either
  rushes or rambles, so an in-run "chunked must be slower than one pass" assertion failed by
  beating a degenerate single pass. The case now asserts absolute bands on the chunked output
  (0.6–3.6 words/s, WER < 0.85) and prints the one-pass figures, which are recorded in the
  docstring. The WER instrument itself wanders ±0.15 on this corpus (one sentence × 10, hard
  20s windows, Whisper hallucinating repeats — it measured 0.42, 0.47 and 0.77 for the same
  route), which is why it carries a catastrophe bound and not the claim.
- **Assistant API verified live**: `scripts/say.py --list` and a real `/speak` call over HTTP
  returned 3.68s @ 48kHz, peak 0.9999 with `clipped_fraction` 5.7e-6 (one sample in 176k — a
  graze, not clipping), RMS 0.144, no defect flags.
- **Length study** (single pass, same speaker, in the README): 18/60/150/300 words → 5.9/17.7/
  45.7/74.3s of audio; peak VRAM 4.36/7.52/7.58/7.58 GB; 1.32–1.50× realtime; words/s
  3.05/3.39/3.28/4.04; chunked-Whisper WER 0.00/0.62/0.55/0.68. Raising the cap from 2048 to
  4096 on the 300-word case produced **identical** audio (929 tokens, EOS reached), which is how
  we know the cap is not the constraint.
- **Chunking verified**: same 300 words → 16 pieces, 92.6s audio, **2.48 words/s**, WER **0.42**,
  no internal gap over 1s, no gap-induced stutter. Wall time 131s vs 101s single pass.
- **Instrument caveat, recorded so nobody over-reads it**: the long-text WER comes from hard
  20s Whisper windows over a text that is one sentence repeated ten times. That floor is why
  the GPU bound is 0.55 rather than something tight; the pacing assertion is the strong one.
- **Latent bug found and fixed while building this**: `_rms_windows` padded with a negative
  width for any clip length that was not an exact multiple of the window (numpy raised). It had
  survived every earlier test because they all used round-number lengths.

Run on 2026-11-03 (UI and clone-quality review) on a **CPU-only box, no GPU, no weights**:

- **CPU tier 84/84 passing.** Rewritten: the layout tests (they pinned the old widgets),
  `test_candidate_penalty` (now relative to `CONF_GOOD`), `test_auto_pick_window` (edges
  snap). `test_media_helpers` now stubs `shutil.which`, so it no longer needs ffmpeg.
  New: scorer token layout, pauses/snapping, reference checklist, reference cleanup, no-
  overwrite saves, voice routes and the Voices-tab helpers, library loading, clone
  validation outputs, the prompt LRU, transcribe→language, the sidebar status.
- **Verified by launching the real server** (CPU, `--no-asr`): the UI builds and serves; all
  five tabs render (headless-Chromium screenshots of Clone, Speak and Voices checked by eye);
  `scripts/say.py --list` works against it; `/speak` reaches the model loader and fails only
  on the missing weights, as expected. Found and fixed there: a described voice listed as
  `Nones` in `/voices_available`.
- **GPU tier: RUN on the 5070 Ti after the rework — 8/8 passing** (`voice_design`,
  `voice_clone`, `voice_library`, `speak_api`, `chat_stream`, `long_text`, `best_of_n`,
  `media_pipeline`). Loads 1.5–3.0s, generation 7–11s, chat first audio 11.7s, `long_text`
  2.97 words/s at WER 0.19, every clip inside the RMS/peak/duration bounds.
- **The corrected confidence scale is confirmed on real audio**: clean English takes read
  `conf −0.02`/`−0.06` (the buggy scale read ~−1.9), a saved Japanese take `−0.51`, fox's
  shipped clip `−0.03`. `CONF_GOOD = −1.00` sits clear of every real measurement and is no
  longer provisional — see `docs/quality.md`.
- **Reference cleanup measured** (`tmp/clean_reference_ab.py`, three seeds per arm, identity
  vs the raw reference): SIM 0.9870 → 0.9886, WER 0.000 both arms, energy above 9kHz 0.0472 →
  0.0470, the reference's own embedding 0.9991. Neutral-to-marginally-better, and specifically
  no top-end cost — the claim in `docs/quality.md`, which now carries the table.
- **`transcribe_detailed` detects language on real audio**: the Japanese clip → `ja`/Japanese,
  fox → `en`/English.
- **`make_voice.py` under the lock verified live**: with the web server running it exits 2 and
  names the holder (`webui pid=… started=…`) and suggests `--no-clip`; `--no-clip` itself
  succeeds without the lock, as intended.
- **Deployed and checked over HTTP**: page serves (226KB, five tab labels present),
  `scripts/say.py --list` lists described voices with the new `described` column, `/speak`
  returns audio for `fox`, and forged paths to `/gradio_api/call/import_media` are refused.
  Only the five documented endpoints now carry an `api_name` (`generate_voice_clone`,
  `generate_voice_design`, `speak`, `chat_say`, `voices_available`); the internal UI handlers
  are no longer in the public API at all — a tightening, but it means `curl` probes of
  `/gradio_api/call/<handler>` are no longer a way to exercise them.
- **Still not verified, needs a browser:** microphone recording into the new Record sub-tab,
  the Voices-tab row-click (`gr.SelectData.row_value`), the two-step delete, and the
  speech-coverage auto-pick / pause snapping on a real messy upload (cleanup is measured; the
  *window choice* is not — it needs a long upload with several candidate stretches).

Changed on 2026-11-04 on the CPU-only box (render cache, Chat quick start, reference
transcript check). **CPU tier 88/88.**

Run on 2026-11-04 on the **RTX 5070 Ti** — the tier it was pushed without:

- **GPU tier 8/8** (`voice_design`, `voice_clone`, `voice_library`, `speak_api`, `chat_stream`,
  `long_text`, `best_of_n`, `media_pipeline`). The edited assertions hold: `speak_api` repeats
  a line and gets `repeat from cache 0.01s` against 3.7s for the first call; `chat_stream`
  repeats a reply with `first audio 0.01s, all 0.04s` and `4 from the render cache`. The tier
  used its own cold `tests/output/render_cache` and left the real `tmp/render_cache` empty —
  checked afterwards, 0 files.
- **`REF_TEXT_CONF_WARN` calibrated** — see *Design decisions*. The tier's pair (-0.04 right,
  -7.02 wrong) is a fair summary; the sweep behind it is four references x four degrees of
  wrongness plus noise, and it is what says -2.0 is fine to warn with and unsafe to block with.
- **Quick start measured**: first audio 11.49s → 5.30s for the same reply (warm hub, cache
  off), at the cost of ~2s on the whole reply. Cold hub 7.14s; tier sequence with the
  clone-prompt cache already warm 2.2s. First piece ends in silence and keeps its top end —
  measured, not listened to.
- **Still not verified, needs a browser**: the Voices tab row-click, mic recording, the
  two-step delete, and whether a cached replay is indistinguishable from a fresh take when you
  listen (the cache writes PCM_16, so a replay differs from the first play by ~3e-5).

## Do-not-touch list

- **The three memory guardrails** (lock, `max_cpu_resident`, RAM/VRAM floors) — they
  exist because the machine hard-locked. Nick's OK required.
- **`ModelHub._move()`** — bypassing it costs 25x inference and produces no error.
- **The dialect/language split** — CustomVoice is gone, but re-adding `*_dialect` to the
  language dropdown raises at generate time (upstream filters them out of
  `supported_languages`).
- **`UPSTREAM_REF` in `setup.sh`** — moving it off the SHA Nick tested changes the model
  code under a "working" install; freeze deliberately, not casually.
- **`tests/smoke_test.py`'s audio assertions** (rms/peak/duration) — they are what catch
  a model quietly left on CPU. Weakening one to make it pass is a red flag.
- **`voices/`, `tmp/media/`, `certs/`, `.qwen-tts.lock`** — real user audio and key
  material; gitignored forever.

## Guardrails in effect

- Visibility: **private** on `nkavassalis/qwen-tts-ui` until Nick explicitly flips it.
- Branch: `main`; `feat/*`/`fix/*` for work, never force-push `main`.
- Before any commit: scan `git diff --cached` (`secret-scan` skill / `scripts/secret-scan.sh`).
  Before any push: both test tiers, the secret scan, and `gh auth status` = `nkavassalis`.
- Nothing internal in any commit message, README, AGENTS.md or log: no hostnames, LAN
  addresses, home paths, account ids, or live user audio.
- Model weights and generated audio stay gitignored; never commit `models/`,
  `tests/output/`, `voices/` or `tmp/`.
