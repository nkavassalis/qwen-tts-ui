# Qwen3-TTS — local deployment (RTX 5070 Ti)

[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) running fully offline on this machine,
behind a Gradio web UI at **http://127.0.0.1:7860**.

Three modes, all on the **1.7B** checkpoints (the best quality tier Qwen has released —
the 0.6B variants score noticeably worse on WER/speaker-similarity and VoiceDesign is
1.7B-only):

| Tab | Checkpoint | What it does |
|---|---|---|
| TTS (CustomVoice) | `Qwen3-TTS-12Hz-1.7B-CustomVoice` | 9 preset speakers (labels show each one's native language/dialect), 10 languages, natural-language style instructions ("用愤怒的语气说", "very happy") |
| Voice Design | `Qwen3-TTS-12Hz-1.7B-VoiceDesign` | Invent a brand-new voice from a text description of timbre/age/emotion/accent |
| Voice Clone (Base) | `Qwen3-TTS-12Hz-1.7B-Base` | Clone a voice from a ~3s reference clip (upload or microphone) |

**Voice library**: the Voice Design tab can *save* a designed voice (`voices/*.wav` + a
sidecar with its transcript), and the Voice Clone tab can load it back as a reusable
reference. That makes a designed voice persistent across requests and restarts — the
hosted demo cannot, since it keeps nothing between sessions. Under the hood the saved
clip + transcript feed `create_voice_clone_prompt`, so it clones with full ICL quality.

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
tests/test_app_logic.py   CPU-only unit tests (30 cases) for UI helpers and guards
tests/smoke_test.py       GPU end-to-end: 6 flows, generates audio and validates the wav
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

## Tests

```bash
.venv/bin/python tests/test_app_logic.py   # ~10s, no weights needed, 30 cases
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

## Design notes

**Why CPU-resident weights with an LRU GPU slot.** The official Hugging Face Space keeps
five checkpoints (both 0.6B and both 1.7B sizes, plus VoiceDesign) permanently on the
GPU — roughly 20GB. A 16GB 5070 Ti can't hold that. Instead each 1.7B model (~4.3GB)
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
