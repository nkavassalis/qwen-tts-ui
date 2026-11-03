# Qwen3-TTS — local deployment (16GB Blackwell: RTX 5070 Ti / 5080)

[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) running fully offline on this machine,
behind a Gradio web UI at **http://127.0.0.1:7860**.

**Target hardware is any single 16GB Blackwell card** — RTX 5070 Ti, RTX 5080, anything in
that class. The VRAM budget, the LRU GPU slot and the memory guards are all written for
that card class, not for one specific SKU; the 5070 Ti is simply what it is developed and
tested on (under WSL2).

Three modes, all on the **1.7B** checkpoints (the best quality tier Qwen has released —
the 0.6B variants score noticeably worse on WER/speaker-similarity and VoiceDesign is
1.7B-only):

| Tab | Checkpoint | What it does |
|---|---|---|
| TTS (CustomVoice) | `Qwen3-TTS-12Hz-1.7B-CustomVoice` | 9 preset speakers (labels show each one's native language/dialect), 10 languages, natural-language style instructions ("用愤怒的语气说", "very happy") |
| Voice Design | `Qwen3-TTS-12Hz-1.7B-VoiceDesign` | Invent a brand-new voice from a text description of timbre/age/emotion/accent |
| Voice Clone (Base) | `Qwen3-TTS-12Hz-1.7B-Base` | Clone a voice from a ~3s reference clip (upload or microphone) |

**Voice library**: both generation tabs can save a voice, and the Voice Clone tab loads
them back as a reusable reference — Voice Design's *Save as clone voice* keeps the clip it
generated, Voice Clone's *Save to voice library* keeps the trimmed reference clip plus its
transcript. Each save is `voices/<slug>.wav` + a sidecar `.json`, so a voice survives
requests and restarts — the hosted demo cannot, since it keeps nothing between sessions.
Under the hood the saved clip + transcript feed `create_voice_clone_prompt`, so it clones
with full ICL quality.

**Reference-clip workshop** (Voice Clone tab) — the three things the hosted box makes you
do in a text editor:

- **Video uploads work.** Drop in an `.mp4`/`.mov`/`.mkv`/`.webm`/… and ffmpeg strips the
  audio track; audio files (`.wav`/`.mp3`/`.flac`/`.m4a`/…) load natively.
- **Visual trimming.** A hand-drawn waveform with a second grid shows the whole clip; two
  sliders move the selection, the shaded region is exactly what gets cloned, and
  *Auto-pick best clip* jumps to the loudest 6s (drops silence and room tone).
- **Transcription you can edit.** *Transcribe selected clip* runs **Whisper large-v3** on
  the selected window only and drops the text into an editable box — because ICL cloning
  needs the transcript to match the clip, not the whole upload.

### Cloning a voice, start to finish

The tab is three numbered steps, top to bottom, and they are the whole workflow.

**1 · The reference** — the clip the voice is taken from

1. *Upload video or audio*. Video is fine; only its audio track is used.
2. **Extract audio & show waveform**. The whole upload is drawn; the first ~6s lands in
   *Reference clip*.
3. Drag **Start (s)** / **End (s)** so the shaded region covers the cleanest stretch
   (3–10s, one speaker, no music) — or press **Auto-pick best clip** for the loudest 6s.
4. **Transcribe selected clip**, then fix anything Whisper got wrong in *The reference clip
   says…*. This matters: ICL cloning conditions on that transcript matching the clip, and
   the *Fit check* line warns when it obviously does not. Skip it only with *x-vector only*
   (lower similarity).

**2 · What it should say**

5. Type **Say this in the cloned voice** — the new line, not what the reference says — set
   **Language**, and press **Clone & generate**. Raise **Candidates (best of N)** for
   several takes; see below.

**3 · Keep it**

6. Three different things you might mean by "keep", three different controls:
   - the **audio file**: the download icon on its player. Files under `/tmp/gradio/` are
     ephemeral, so download rather than pathing to them.
   - the **reference voice** you built: name it and press **Save to voice library** →
     `voices/<slug>.wav` + `.json`, listed in *Saved voices* on every later run. **Load**
     restores clip + transcript; **Delete** removes both files.
   - a **generated take you preferred**: pick it under *Which take to keep* (defaults to the
     auto-pick), name it, press **Save selected take**. It stores the model's own decode —
     not the post-processed listening copy — with the line it was asked to say as its
     transcript, so you can clone from it later without re-uploading. This is how you
     overrule the scorer and still keep the result as a reusable voice.

#### Auditioning candidates

With *Candidates (best of N)* above 1 you get **one player per take**, labelled with its
numbers, plus **Generated audio (the auto-pick)** above them. The pick ranks by a penalty:

| signal | what it catches | weight |
|---|---|---|
| **WER** — Whisper vs the text you asked for | wrong, missing or extra words | ×2.0, dominant |
| **confidence** — Whisper's decoder log-prob of the *intended* text, given the audio | a take that is barely decodable at all | ×0.4 below −2.6 |
| **defects** — speaking rate vs text length, edge silence, clipping | dead air, machine-gun pacing, blown peaks | capped at 0.3 |
| **SIM** — speaker encoder vs your reference | near-identical takes | tie-break only |

Listen to them all and keep whichever you like: every player has its own download icon, so
the scorer is a suggestion, not a gate.

Measured on four seeds of the same line — the pick below is reproducible with `Seed: 7`,
`Candidates: 4`:

| take | WER | conf | SIM | defect flags | penalty |
|---|---|---|---|---|---|
| #1 | 0.00 | −2.01 | 0.988 | 0.9s silence | 0.12 |
| #2 | 0.00 | −1.91 | 0.988 | 0.8s silence | 0.10 |
| #3 | 0.00 | −1.99 | 0.987 | 0.5s silence | 0.01 |
| **#4 (kept)** | 0.00 | −2.09 | 0.985 | — | **0.00** |

All four pronounce the line perfectly, so WER cannot separate them and similarity prefers
#1. The defect term is what promotes #4 — #1 is carrying 0.9 seconds of dead air.

**What confidence is, and what it is not.** It is Whisper's own decoder log-probability,
which on this checkpoint measures about **−1.9 for a clean take**, so the threshold sits at
−2.6: it fires on disaster, not on taste. Deliberately degraded takes read −1.87 to −2.02
for added silence, clipping and half-speed pacing (no signal at all), and −6.3 for pure
silence. It is scored against **the text you asked for**, never against the transcript
Whisper produced — on a noise-degraded take, grading Whisper's own answer read −1.91 and
missed it, while grading the intended text read −2.07 and caught it. Whisper is confident
about whatever it decided to hear; that is self-consistency, not audio quality.

And nothing here measures naturalness, prosody or emotion. When the numbers read as a tie,
your ears are the better judge — which is exactly why the takes stay on the page.

## Quick start

```bash
./run.sh                      # -> http://127.0.0.1:7860
./run.sh --ip 0.0.0.0 --username you --password-file ~/tts.pw   # expose on the LAN, with a login
./run.sh --preload            # warm checkpoints into RAM at startup (bounded by --max-cpu-resident)
./run.sh --max-cpu-resident 1 # safest: pin only one 4.3GB copy at a time
./run.sh --ssl-self-signed --ip 0.0.0.0 --username you --password-file ~/tts.pw   # HTTPS; needed for browser mic on non-localhost
./run.sh --no-asr                # drop Whisper (frees ~3GB VRAM, disables the transcribe button)
./run.sh --asr-model openai/whisper-large-v3-turbo   # swap the transcription model
./run.sh --help                  # all flags
```

Nothing binds beyond localhost unless you pass `--ip` or `--share`, and the server prints a
loud warning if you do either without credentials. Give the password through
`--password-file` or `QWEN_TTS_PASSWORD` (or let the app prompt for it) — `--password` still
works but warns, because a command line is world-readable via `ps`. See
[Exposure and safety](#exposure-and-safety) before sharing it.

First request after a model switch pays ~2s of GPU upload; after that generation runs
faster than realtime (~13s for 6s of audio, including queueing).

Rebuild the environment from scratch with `./setup.sh` — it creates `.venv` on Python 3.12
(needs [uv](https://docs.astral.sh/uv/)), installs `torch==2.8.0` + `torchaudio` from the
**cu128** index (the wheel line that covers sm_120 / RTX 50xx), installs
`requirements.txt`, installs upstream `qwen-tts` editable from `./repo` (which pins
`transformers==4.57.3`), downloads the checkpoints into `./models/`, and fetches Whisper
into the Hugging Face cache. `./setup.sh --models` re-downloads weights only, `--asr` only
Whisper. Pin the upstream checkout with `QWEN_TTS_UPSTREAM_REF=<sha-or-tag>`.

## Layout

```
setup.sh            reproducible env + weight/Whisper download
run.sh              launcher (handles self-signed certs for HTTPS)
requirements.txt    direct deps; torch and transformers are pinned elsewhere on purpose
webui/app.py        the Gradio UI + VRAM/RAM-aware model manager + voice library
scripts/say.py      CLI for the assistant API: render text in a saved voice and play it
tests/test_app_logic.py   CPU-only unit tests (65 cases) for UI helpers and guards
tests/smoke_test.py       GPU end-to-end: 9 flows, generates audio and validates the wav
AGENTS.md           the agent doc: design rationale, invariants, do-not-touch list
CLAUDE.md           pointer to AGENTS.md
CHANGELOG.md        one entry per change
LICENSE             Apache-2.0 (upstream Qwen3-TTS is Apache-2.0 too)
repo/               upstream QwenLM/Qwen3-TTS checkout (installed editable, gitignored)
models/             1.7B checkpoints + Qwen3-TTS-Tokenizer-12Hz (~14GB, gitignored)
voices/             saved designed voices: <slug>.wav + <slug>.json (gitignored)
tmp/media/          extracted reference clips, pruned after 6h (gitignored)
certs/              self-signed HTTPS key pair (gitignored)
.qwen-tts.lock      single-owner advisory lock (gitignored)
```

Beyond Python: **ffmpeg** on PATH (video ingest; checked and reported at startup) and
~2.9GB for Whisper large-v3 in the Hugging Face cache, fetched on first use.

**Reference clips want to be short and clean.** The model card advertises a "3-second rapid
voice clone"; in practice 3–10s clones best, so the UI defaults the selection to the loudest
6s (`REF_TARGET_SECONDS`) and clamps any selection into 2–15s (`trim_window`). Longer
references mostly cost time and raise the chance of a rambling ICL run.

**Whisper large-v3 via transformers, not faster-whisper.** It reuses the torch/CUDA stack
already installed (no ctranslate2, whose kernels may not cover sm_120), transcribes a 6s
clip in ~3s on this GPU, and only loads when the user presses the button. `setup.sh` fetches
it with `--exclude "*fp32*" "*.bin" "*.msgpack" "*.h5" "*.ot"` — the repo otherwise carries
~24GB of flax/tf/fp32 duplicates of the one 2.9GB safetensors file needed. `--asr-model` swaps it (e.g. turbo), `--no-asr`
skips it. The raw padding mask is passed explicitly, which both silences the
pad-token-equals-eos warning and stops Whisper hallucinating over the 30s of zero padding a
short clip gets.

**The waveform is drawn with PIL, not a plotting library.** No matplotlib dependency, pixel
geometry is deterministic, and `test_render_waveform` can assert the shaded selection
actually changes the image.

## Memory safety (read this first)

This box is **WSL2 with dynamic memory** (`hv_balloon` is set to essentially the host's
whole RAM).
The VM is allowed to balloon to essentially your whole 32GB of host RAM, so over-committing
inside WSL does not just OOM the distro — it can starve Windows and hard-lock the machine.
That happened once: the server was started with `--preload` (three checkpoints, ~13GB) while
`smoke_test.py` was running with its own three copies, ~26GB plus page cache.

Three guardrails now make that impossible to repeat:

1. **Single-owner lock** (`.qwen-tts.lock`, `flock`). The server and the smoke test compete
   for it and the loser refuses to start, naming the holder. `--force` overrides, deliberately.
2. **Bounded CPU residency**: at most `--max-cpu-resident` (default **2**) checkpoints are
   pinned in host RAM; beyond that the LRU one is dropped and re-read from disk (~1.5s).
3. **Explicit floors**: a checkpoint is only loaded if `MemAvailable` stays above
   `--ram-floor-mb` (default 4096), and only promoted to the GPU if VRAM free exceeds its
   size plus a margin. Otherwise you get a `MemoryError` with a diagnosis, not a swap spiral.

Rule of thumb: **one Qwen3-TTS process at a time**, and check `free -m` before raising the
caps. If Windows itself feels tight, lower `--max-cpu-resident` to 1 or set a hard cap in
`C:\\Users\\<you>\\.wslconfig` (`memory=20GB`).

## Exposure and safety

This is a desk tool that runs an unauthenticated model endpoint by design, so the
boundaries are worth stating:

- **It binds to 127.0.0.1 unless you say otherwise.** `--ip 0.0.0.0` and `--share` (a
  public gradio.app tunnel) are conveniences, not defaults, and the server prints a warning
  if you use either without credentials. Those are one shared
  credential (Gradio's `auth=`), not per-user accounts — fine for a house LAN, not for the
  internet. Supply them via `--password-file` (chmod 600) or `QWEN_TTS_PASSWORD`; the app
  prompts for the secret if you gave a username but no password, and never prints it. The
  legacy `--password` flag is visible to every local user through `ps`/`/proc` and warns.
- **Uploads and reference clips are confined to the upload directories.** Anything the
  browser names is resolved and checked against `allowed_input_roots()` (Gradio's upload and
  component-cache dirs plus `tmp/media/`) before it is opened, so a client cannot point the
  app at an arbitrary `.wav`/`.mp4` on this machine. Gradio 6 checks this too; ours is the
  second check, and the File/Audio components themselves do neither. To serve a curated
  folder of clips, add it to `QWEN_TTS_ALLOWED_MEDIA_DIRS` (path-separator separated).
- **Nothing is kept forever.** Extracted reference clips live in `tmp/media/` and are pruned
  at startup and on each import once older than `QWEN_TTS_MEDIA_TTL_SECONDS` (default 6
  hours). Saved voices in `voices/` are yours to keep — and gitignored.
- **Nothing is logged.** Status lines echo exception text into the browser, so treat the
  status boxes as user-visible output.
- `certs/` holds a private key when you use `--ssl-self-signed`; it is gitignored.

## Talking to it from elsewhere (assistant API)

The **Speak (assistant API)** tab is one call: name a saved voice, type what to say. It is
also what external tools hit, and `scripts/say.py` wraps it so the whole thing is a shell
command — which is the shape a coding assistant actually wants:

```bash
scripts/say.py --list
scripts/say.py --voice fox_reference --text "Build finished. Three tests failed."
echo "The migration is done." | scripts/say.py --voice ryan
scripts/say.py --voice ryan --text "Coffee." --out kitchen.wav --no-play
```

(`.venv/bin/python scripts/say.py …` if you are outside the venv. `--url` points it at
another box; `--username/--password-file` or `QWEN_TTS_PASSWORD` match the server's own
precedence; `--insecure` is for `--ssl-self-signed`.)

Under the hood it is the `/speak` endpoint, taking `(voice, text, language, temperature,
top_p, top_k, repetition_penalty, postproc)` and returning audio plus a status line, with
`/voices_available` behind `--list`. Three properties worth knowing if you wire it into
something:

- **A caller names a voice, never a path.** The reference clip and its transcript are read
  from `voices/<slug>.json` on the server, and the name is slugged to `[a-z0-9_]` before it
  touches the filesystem. There is no argument to point at `/etc/passwd`, which is why the
  CLI has no `--ref-audio`.
- **No transcript in the sidecar → x-vector fallback** rather than a failure, and the status
  says which path it took (`Speak (ICL)` / `Speak (x-vector)`).
- **It shares the one GPU.** With `default_concurrency_limit=1`, a long line from the
  assistant queues a UI generation behind it — by design (one process, one card), but
  surprising the first time. Expect ~1.4× realtime, first audio after ~6–9s on long text.

For an assistant, the useful pattern is fire-and-forget on things worth hearing: a build
finishing, tests failing, a long job ending. Not narration — at 1.4× realtime, having it
read everything aloud is a way to make your terminal slower.

## Tests

```bash
.venv/bin/python tests/test_app_logic.py   # ~10s, no weights needed, 65 cases
.venv/bin/python tests/smoke_test.py       # ~90s, needs the GPU + downloaded weights
.venv/bin/python tests/smoke_test.py --only media_pipeline
```

The CPU tier still needs the venv (`torch`, `gradio`, `qwen_tts`, `soundfile`, `librosa`,
`PIL`) — it imports the app module — it just needs no CUDA and no weights. It also builds
the whole Gradio UI (`test_build_ui_constructs`), which is what catches Gradio API drift
before a human clicks it.

The smoke test covers six flows: CustomVoice, VoiceDesign, Base clone, a dialect speaker
(Dylan → Beijing dialect), the design→save→load→clone voice-library flow, and the media
pipeline (synth speech → wrap in a real mp4 → ffmpeg import → auto-trim → Whisper → clone).
It refuses to run while the web server holds the lock — stop the server first rather than
using `--force`. A case that never ran (no ffmpeg, `--no-asr`) is reported as `SKIPPED` and
exits non-zero, because a skip is not a pass; pass `--allow-skip` when the skip is intended.

`smoke_test.py` writes the audio it produces to `tests/output/` and asserts each clip is
non-silent, non-clipping and of plausible length, so a silently-broken pipeline (e.g. a
model left on CPU) fails the test rather than just feeling slow.

## Getting the best quality

Measured on the 5070 Ti box rather than guessed, because the obvious knobs turned out not
to matter. Both metrics are the ones upstream benchmarks with: **SIM** = cosine similarity
of the model's own speaker encoder between reference and output; **WER** = Whisper
large-v3 on the output against the text that was requested.

**The decode sliders are not the lever.** Same reference, same text, two seeds each:

| config | SIM | WER |
|---|---|---|
| current defaults (t=0.90, top-p 1.0, top-k 50, rep 1.05) | 0.9912 | 0.000 |
| `non_streaming_mode=True`, t=0.90 | 0.9910 | 0.067 |
| `non_streaming_mode=True`, t=0.70, top-p 0.95 | 0.9913 | 0.000 |
| current defaults, t=0.70, top-p 0.95 | 0.9916 | 0.000 |

All inside noise — leave the sliders alone unless a take rambles, and then lower
temperature or raise the repetition penalty.

**The reference is the lever.** Same target text, one speaker, varying the reference:

| reference | SIM | WER | output length |
|---|---|---|---|
| 1.8s clean | 0.987 | 0.000 | 5.0s |
| 4.5s clean | 0.992 | 0.000 | 5.5s |
| 9s clean | 0.993 | 0.133 | 5.6s |
| 9s + broadband noise (SNR 10 dB) | **0.973** | 0.133 | 5.4s |
| 9s at 10% peak level | 0.991 | 0.133 | 5.7s |
| 9s, x-vector only | 0.991 | 0.133 | 4.9s |
| 9s, **transcript does not match the clip** | 0.978 | **1.000** | **17.4s** |

In order of how much each one actually costs you:

1. **The transcript must match the clip.** Not a polish issue: with a mismatched
   transcript the model produced 17.4s of audio for a 5s sentence at WER 1.000. The
   *Fit check* line warns before you generate, and the status line repeats it after — it is
   a plausibility guard (words per second against the selection's length), which is how
   mismatches almost always present: the transcript of the whole upload, or a stub left in
   the box. It cannot tell that a right-length transcript has the wrong words, so still
   read what Whisper wrote. If you cannot get it right, tick **x-vector only** — 0.991 SIM
   vs 0.993 for ICL on a clean reference, and immune to this failure entirely.
2. **Use a clean reference.** Noise was the largest similarity loss measured (−0.020).
   Quiet room, one speaker, no music, no reverb.
3. **Length 4–10s.** 1.8s → 9s bought +0.006 SIM, most of it in the first seconds; past
   ~5s it flattens, and long references make ICL runs more likely to ramble.
4. **Recording level does not matter.** A reference peaking at 0.1 scored 0.991 — don't
   bother normalising.
5. **Set the language explicitly** instead of Auto, and on the CustomVoice tab pick a
   speaker whose native language matches the text (upstream's own recommendation).
6. **Shape the text.** Prosody follows punctuation; spell out numbers; split long scripts
   into sentences, which also keeps you inside the 2048-token budget.

**Sampling variance is real, which is what *Candidates (best of N)* is for.** Four seeds on
identical inputs: SIM spread only 0.001, but WER was 0.000 on one seed and 0.133 on the
others — the model occasionally serves a dud take and no slider fixes that. Generating
three and keeping the best does. Set a **Seed** to reproduce a take you liked.

### Why it sounds a little "digital", and what *Output post-processing* does about it

The vocoder is fixed at 24kHz (tokenizer config: `output_sample_rate: 24000`,
`decode_upsample_rate: 1920`), so **nothing can add real bandwidth above 12kHz** — the
"air" simply isn't in the signal and any filter that claims to put it back is inventing it.
What *is* fixable is the fizz. The raw decode measured 0.155% of its energy in the top 5%
of the band (11.4–12kHz) where a properly band-limited signal has 0.036% — a **4× excess of
near-Nyquist imaging** from the decoder's transposed-convolution upsampling, which is
exactly what people describe as digital/harsh. Doing the 24→48kHz step here with soxr HQ
also stops your browser guessing at it: a cheap linear resampler cost 20% of the high-band
energy in the same measurement.

| Output post-processing | SIM | centroid | HF energy | near-Nyquist |
|---|---|---|---|---|
| Off — raw 24kHz from the vocoder | 0.9900 | 2044 Hz | 3.26% | 0.155% |
| Clean — soxr HQ → 48kHz | 0.9893 | 2029 Hz | 3.11% | 0.036% |
| *(a cheap linear resampler, for scale)* | 0.9884 | 1898 Hz | 2.17% | 0.038% |
| **Warm** — 48kHz + de-image + gentle air (default) | 0.9879 | 2164 Hz | 3.83% | 0.027% |

SIM is the model's own speaker encoder against the reference, so the right-hand columns
cannot be paid for with likeness: 0.9900 → 0.9879 is inside the 0.001 seed-to-seed spread
measured above. **Warm** cuts the near-Nyquist energy ~6× and ends up *brighter* than raw
(centroid 2164 vs 2044) because the lowpass is followed by a small high-shelf, which is
what makes it read as open rather than dull.

Pick **Off** to hear the untouched decode — useful for A/B, and worth using when you save a
Voice Design clip into the library, since that clip becomes the clone reference and is
better coming straight from the model than from an EQ curve.

Method: one English target sentence, single speaker per sweep, n=2–4 per row, so anything
under ~0.005 SIM should be read as noise. The two findings above are far outside it.

### Long text: the 2048-token cap is not the limit — tempo is

Every generation is bounded by `max_new_tokens`, scaled to the text (`_budget()`, capped at
2048). That cap is a **runaway guard**, not a quality limit, and it is worth being precise
about why, because both obvious guesses are wrong.

Measured on the 1.7B checkpoint, single pass, same speaker:

| words | audio | tokens | wall time | peak VRAM | words/sec | Whisper WER |
|---|---|---|---|---|---|---|
| 18 | 5.9s | 74 | 8.7s | 4.36 GB | 3.05 | 0.00 |
| 60 | 17.7s | 221 | 23.3s | 7.52 GB | 3.39 | 0.62 |
| 150 | 45.7s | 571 | 60.5s | 7.58 GB | 3.28 | 0.55 |
| 300 | 74.3s | 929 | 100.7s | 7.58 GB | **4.04** | **0.68** |
| 300, cap raised to 4096 | 74.3s | 929 | 111.4s | 7.58 GB | 4.04 | 0.68 |

- **Not VRAM.** 7.52 GB at 221 tokens, 7.58 GB at 929. The talker's KV cache is ~112 KB per
  token (28 layers × 8 KV heads × 128 head_dim × K and V × bf16), so even the checkpoint's own
  8192 default fits comfortably on a 16 GB card.
- **Not the token budget.** 300 words needs 929 of the 2048 tokens allowed, and raising the cap
  to 4096 produced *identical* audio — EOS arrives on its own. The cap only matters when the
  model fails to stop, which ICL cloning does occasionally.
- **It is tempo.** The model accelerates as the sequence grows — 3.05 words/s at 18 words, 4.04
  at 300 — and intelligibility goes with it. The usable ceiling on this checkpoint is roughly
  **30–45 seconds of audio per pass**, whether the cap says 2048 or 8192.

So long text is split and stitched automatically (above ~25 words): `split_for_tts` then
`stitch_clips`.

- **Split on real sentence boundaries only**, after protecting the dots that are not full
  stops: `Mr.` `Dr.` `St.`, decimals (`3.5`), initials (`J. R. R. Tolkien`), ellipses, URLs,
  emails, list markers (`1.`). A false boundary is not cosmetic — the model reads each one with
  sentence-final intonation.
- **Then fit the measured band** (12–26 words, or 20–42 CJK characters): fragments are merged so
  nothing arrives clipped and toneless, over-long sentences are cut at the clause boundary
  nearest their middle, and the separator stays with the left piece so the model does not use
  falling intonation in the middle of a thought.
- **Lossless.** Concatenating the chunks reproduces the input with whitespace ignored. That is
  the invariant the tests hammer, because the failure mode is silently dropped words.
- **Stitched, not appended.** Each piece's leading/trailing padding is trimmed and the joins get
  a 15 ms equal-power crossfade; naive concatenation gives you a stutter of gaps and a click at
  every seam.
- A piece that comes out at an implausible rate is **re-rolled once** — about 10s, against
  re-running the whole document.

The same 300 words after chunking: **92.6s of audio at 2.48 words/s, WER 0.42** — from 4.04 and
0.68. The cost is 16 separate decodes: 131s instead of 101s, still ~1.4× realtime.

Past ~40 chunks a request is refused with a `MediaError` rather than quietly running for an
hour. One GPU, one process.

## Design notes

**Why CPU-resident weights with an LRU GPU slot.** The official Hugging Face Space keeps
five checkpoints (both 0.6B and both 1.7B sizes, plus VoiceDesign) permanently on the
GPU — roughly 20GB. A 16GB Blackwell card (5070 Ti, 5080) can't hold that. Instead each 1.7B model (~4.3GB)
lives in host RAM and a small LRU ("GPU slot", `--max-resident`) promotes the one you're
actually using, evicting the least recently used model to RAM otherwise. Switching tabs
costs ~2s; nothing is re-read from disk. With `--max-resident 2` two models stay on the
GPU (fits, but leaves little headroom for long prompts).

**`attn_implementation="sdpa"`, not flash-attn.** `flash-attn` 2 ships no sm_120
(Blackwell / RTX 50xx) wheels and building it from source is slow and fragile. The
official Space uses `sdpa` too, and torch's bundled flash kernels still get used where
applicable. `--attn-implementation flash_attention_2` is available if you ever install
a Blackwell build.

**The speech tokenizer must be moved to the GPU explicitly.** `model.speech_tokenizer`
is a `Qwen3TTSTokenizer` *wrapper*, not an `nn.Module`, so `model.to("cuda")` leaves the
12Hz vocoder on CPU. The symptom is dramatic and silent: generation is ~25x slower
(141s vs 6s for a few seconds of audio) with GPU utilisation stuck near 38%.
`ModelHub._move()` walks the model's attributes and moves any wrapper holding an
`nn.Module`, and `tests/test_app_logic.py::test_move_reaches_the_speech_tokenizer`
guards the regression.

**Dialects are a speaker trait, not a language.** The checkpoint carries
`talker_config.spk_is_dialect` (`dylan → beijing_dialect`, `eric → sichuan_dialect`) and
upstream deliberately filters the dialect entries out of `get_supported_languages()`
(modeling_qwen3_tts.py:1831-1834), so passing `language="beijing_dialect"` raises
`ValueError: Unsupported languages`. You get an accent by picking the speaker; the UI labels
the speaker dropdown accordingly and keeps dialects out of the language dropdown. Italian *is*
selectable — the hosted Space omits it by oversight.

**Output length is bounded by prompt length.** The checkpoint ships
`max_new_tokens=8192` and sampling defaults of `temperature=0.9, top_k=50,
repetition_penalty=1.05`. ICL voice cloning occasionally fails to emit EOS; one such run
produced 163s of audio for a one-sentence prompt. The UI therefore caps generation at
`min(2048, 4 × len(text))` tokens and exposes temperature / top-p / top-k / repetition
penalty under "Sampling settings (advanced)". If output rambles, raise the repetition
penalty (1.1–1.2) or lower temperature.

**Clone prompts are cached.** The reference-audio encoder runs once per distinct
(reference audio, transcript, x-vector mode) triple, so retaking a line with the same
reference doesn't re-encode it.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ffmpeg could not read audio from …` | The upload has no audio stream (or is not really a video). Check with `ffmpeg -i yourfile`, or pass `--ffmpeg-bin` |
| `Transcription came back empty` | The selected window is music or silence — move the sliders onto speech, or type the transcript yourself |
| Transcription wrong on names/accents | Expected; the transcript box is editable on purpose — fix it before cloning, ICL quality depends on it |
| `Another Qwen3-TTS process is already holding the weights` | The server and the smoke test cannot run together (two copies of ~13GB of weights risk locking WSL2). Stop the server, or pass `--force` if you know what you are doing |
| `MemoryError: ... would leave less than the ...MB floor` | RAM is genuinely tight: close GPU/LLM processes, lower `--max-cpu-resident`, or cap WSL in `.wslconfig` |
| Whole machine hard-locks under load | WSL2 ballooning into host RAM — see *Memory safety* above |
| Generation takes minutes, GPU ~35% | Speech tokenizer stuck on CPU — you bypassed `ModelHub._move()` (see design note above) |
| `Torch has no CUDA` / `no kernel image available` | torch wheel predates sm_120; need `torch>=2.7` built with cu128 (`setup.sh` pins 2.8.0+cu128) |
| Mic record button does nothing on a LAN URL | Browsers only allow mic on localhost or HTTPS — use `./run.sh --ssl-self-signed --ip 0.0.0.0` |
| Clone rambles / never stops | Hit the token cap; raise repetition penalty, lower temperature, or use a cleaner reference clip |
| `Model not found at models/...` | `./setup.sh --models` (or `.venv/bin/hf download Qwen/<name> --local-dir models/<name>`) |
| Two people at once | Requests are queued (`default_concurrency_limit=1`); raise `--max-resident` and the queue limit only if VRAM allows |
| `CERTIFICATE_VERIFY_FAILED` while starting with HTTPS | Pass `--ssl-self-signed` (run.sh adds it for you): Gradio's own startup self-check dials its own https port and dies on a self-signed cert unless `ssl_verify=False` |
| `Refusing to read a media file outside the upload directory` | Something named a path the app may not open. Re-upload it through the file box; to serve a curated folder, add it to `QWEN_TTS_ALLOWED_MEDIA_DIRS` |
| Gradio: `Cannot move … to the gradio cache dir because it was not uploaded by a user` | The same rule from Gradio's side — a path that did not come from a real upload. Not something to work around |

`sox` prints a warning at import time; it is only used by the 25Hz tokenizer path, which
none of these 12Hz models touch. Safe to ignore (or `apt install sox`).

## License

Apache-2.0 — see [LICENSE](LICENSE). `webui/app.py` is adapted from the official
Qwen3-TTS Hugging Face Space, and the whole thing is built on
[QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (Apache-2.0, Alibaba Qwen). The
model checkpoints and Whisper weights are downloaded separately under their own licenses;
nothing here redistributes them.
