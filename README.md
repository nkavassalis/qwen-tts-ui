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

## Quick start

```bash
./run.sh                      # -> http://127.0.0.1:7860
./run.sh --ip 0.0.0.0         # expose on the LAN
./run.sh --preload            # warm checkpoints into RAM at startup (bounded by --max-cpu-resident)
./run.sh --max-cpu-resident 1 # safest: pin only one 4.3GB copy at a time
./run.sh --ssl-self-signed --ip 0.0.0.0   # HTTPS; needed for browser mic on non-localhost
./run.sh --help               # all flags
```

First request after a model switch pays ~2s of GPU upload; after that generation runs
faster than realtime (~13s for 6s of audio, including queueing).

Rebuild the environment from scratch with `./setup.sh` (creates `.venv` on Python 3.12,
installs `torch==2.8.0+cu128`, installs the pinned upstream `qwen-tts` from `./repo`,
downloads the checkpoints into `./models/`).

## Layout

```
setup.sh            reproducible env + weight download
run.sh              launcher (handles self-signed certs for HTTPS)
webui/app.py        the Gradio UI + VRAM/RAM-aware model manager + voice library
repo/               upstream QwenLM/Qwen3-TTS checkout (installed editable)
models/             1.7B checkpoints + Qwen3-TTS-Tokenizer-12Hz (~14GB, gitignored)
voices/             saved designed voices: <slug>.wav + <slug>.json (gitignored)
.qwen-tts.lock      single-owner advisory lock (gitignored)
tests/test_app_logic.py   CPU-only unit tests for the UI helpers and memory guards
tests/smoke_test.py       GPU end-to-end: loads each model, generates, validates the wav
```

## Memory safety (read this first)

This box is **WSL2 with dynamic memory** (`hv_balloon` is set to essentially the host's whole RAM).
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

## Tests

```bash
.venv/bin/python tests/test_app_logic.py   # ~1s, no weights needed, 12 cases
.venv/bin/python tests/smoke_test.py       # ~60s, needs the GPU + downloaded weights
.venv/bin/python tests/smoke_test.py --only voice_clone
```

The smoke test covers five flows: CustomVoice, VoiceDesign, Base clone, a dialect speaker
(Dylan → Beijing dialect), and the design→save→load→clone voice-library flow. It refuses to
run while the web server holds the lock — stop the server first rather than using `--force`.

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
(modeling_qwen3_tts.py:1832), so passing `language="beijing_dialect"` raises
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
| `Another Qwen3-TTS process is already holding the weights` | The server and the smoke test cannot run together (two copies of ~13GB of weights risk locking WSL2). Stop the server, or pass `--force` if you know what you are doing |
| `MemoryError: ... would leave less than the ...MB floor` | RAM is genuinely tight: close GPU/LLM processes, lower `--max-cpu-resident`, or cap WSL in `.wslconfig` |
| Whole machine hard-locks under load | WSL2 ballooning into host RAM — see *Memory safety* above |
| Generation takes minutes, GPU ~35% | Speech tokenizer stuck on CPU — you bypassed `ModelHub._move()` (see design note above) |
| `Torch has no CUDA` / `no kernel image available` | torch wheel predates sm_120; need `torch>=2.7` built with cu128 (`setup.sh` pins 2.8.0+cu128) |
| Mic record button does nothing on a LAN URL | Browsers only allow mic on localhost or HTTPS — use `./run.sh --ssl-self-signed --ip 0.0.0.0` |
| Clone rambles / never stops | Hit the token cap; raise repetition penalty, lower temperature, or use a cleaner reference clip |
| `Model not found at models/...` | `./setup.sh --models` (or `.venv/bin/hf download Qwen/<name> --local-dir models/<name>`) |
| Two people at once | Requests are queued (`default_concurrency_limit=1`); raise `--max-resident` and the queue limit only if VRAM allows |

`sox` prints a warning at import time; it is only used by the 25Hz tokenizer path, which
none of these 12Hz models touch. Safe to ignore (or `apt install sox`).
