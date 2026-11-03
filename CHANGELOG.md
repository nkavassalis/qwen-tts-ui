# Changelog

All notable changes to this project are documented here. Newest first.

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

### Not verified

No GPU in this session. The CPU tier passes (84/84); the GPU tier was not run, and its
`smoke_test.py` was edited. Audio-quality effects of cleanup and snapping are unmeasured.

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

**"Save selected take said it worked but nothing happened."** It had worked — `voices/myvoice.wav`
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
