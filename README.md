# qwen3-tts-ui

Text to speech that runs entirely on your own machine. Clone a voice from a few seconds of
audio, or invent one from a description, then make it say anything — from a browser at
**http://127.0.0.1:7860**, from a shell script, or from a coding assistant. No cloud calls, no
API keys, nothing leaves the box.

Built on [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (Alibaba Qwen), **1.7B** checkpoints
— the best quality tier released; voice design is 1.7B-only. Ten languages: Auto, Chinese,
English, Japanese, Korean, French, German, Spanish, Portuguese, Russian, Italian.

**Hardware:** one 16GB NVIDIA card (developed on an RTX 5070 Ti under WSL2). The two
checkpoints are ~9.3GB on disk, Whisper large-v3 another ~2.9GB.

| Tab | What it is for |
|---|---|
| **Clone a voice** | Copy a voice from a short clip — upload (video works), record, or load a saved one; trim it on a waveform; transcribe it; generate the best of several takes |
| **Design a voice** | Invent a voice from a description of timbre, age, accent and mood |
| **Speak** | Pick a saved voice, type a line. Also the endpoint `scripts/say.py` calls |
| **Chat** | Type a line, hear it sentence by sentence while the rest is still generating |
| **Voices** | The voice library: preview, inspect, choose how each voice is spoken, delete |

Settings (sampling, output post-processing) and the live GPU/RAM status are in the
**sidebar** (the › at the top left). Every tab has exactly one **Status** box.

More depth lives in [`docs/`](docs/): [quality.md](docs/quality.md) has the measurements
behind every quality claim here, [design.md](docs/design.md) explains why the code is shaped
the way it is. [AGENTS.md](AGENTS.md) is the guide for changing it.

## Quick start

```bash
./setup.sh                    # once: venv, torch cu128, weights, Whisper, the `fox` voice
./run.sh                      # -> http://127.0.0.1:7860
./run.sh --help               # every flag
```

Common variations:

```bash
./run.sh --ip 0.0.0.0 --username you --password-file ~/tts.pw   # on the LAN, with a login
./run.sh --ssl-self-signed --ip 0.0.0.0 --username you --password-file ~/tts.pw   # HTTPS (browser mic off-localhost)
./run.sh --max-cpu-resident 1    # safest on a tight machine: one 4.3GB checkpoint in RAM
./run.sh --preload               # warm checkpoints into RAM at startup
./run.sh --no-asr                # no Whisper (frees ~3GB VRAM; you type transcripts)
```

`setup.sh` needs [uv](https://docs.astral.sh/uv/). It creates `.venv` on Python 3.12, installs
`torch==2.8.0` from the **cu128** index (the wheels that cover sm_120 / RTX 50xx), installs
upstream `qwen-tts` from a pinned checkout in `./repo`, downloads the checkpoints into
`./models/`, fetches Whisper into the Hugging Face cache and creates the shipped `fox` voice.
`--models`, `--asr` and `--voices` redo one part. **ffmpeg** must be on PATH for video uploads.

The first request after switching between Clone and Design pays ~2s moving weights to the GPU;
after that a 6s line takes ~8s.

## Cloning a voice

The Clone tab is three numbered steps. The left column is the reference, the right is what you
generate and save.

**1 · Reference — the voice to copy**

1. Bring in audio one of three ways: **Upload** (any audio or video file — only its audio
   track is used), **Record** from the microphone, or **From library** to reuse a saved voice.
   It loads straight away; there is no separate import button.
2. The waveform shows the whole file, and the shaded part is what gets cloned. The app starts on
   the stretch with the most speech and lands its edges in pauses. Drag **Start** / **End** to
   change it, or press **Auto-pick the best stretch** to get that choice back.
3. Press **Transcribe the clip** (Whisper large-v3, on the selection only). Then **read the
   transcript against what you hear** and fix it. If the target language is still Auto, it is
   set to the language Whisper heard.
4. Read the **checklist** under the transcript. It is recomputed for every selection, and
   each line flags something that was measured to hurt a clone: a transcript that doesn't fit
   the clip, a noisy background, a clip that is too short or too long, an edge that cuts a word
   in half, clipped peaks.

**Reference options** holds two switches:

- **Clean up the reference**, on by default. It removes DC offset, rumble below 60Hz and
  long silence at the edges. It does no denoising and no level change, because the codec
  reproduces a denoiser's artefacts.
- **x-vector only**. No transcript is needed, at the cost of a little likeness.

**2 · What it should say**

Type the new line, check **Language**, press **Clone & generate**. **Takes (best of N)**
defaults to 3. Each take is a full generation, and the best is picked by Whisper's word error
rate, then its confidence, then waveform defects. Speaker similarity only breaks ties. Every
take stays on the page with its scores, so you can overrule the pick by ear. The Status box
says which take you are hearing and why. Set a **Seed** to reproduce a take.

**3 · Save it to the voice library**

One control. **What to save** offers two kinds of thing:

- **the reference clip + transcript.** This is the default, and the best thing to clone from
  again.
- **any generated take.** This saves the model's own 24kHz decode, never the post-processed
  copy, with the line it said as its transcript.

Give it a name and press **Save voice**. Saving never overwrites: a name that is already taken
is refused, so delete the old voice on the Voices tab first if you mean to replace it. To keep
just the audio file, use the download icon on any player.

### Getting good clones

These are in order of how much they matter. The numbers are in
[docs/quality.md](docs/quality.md).

1. **The transcript must match the clip, word for word.** A mismatched transcript gave 17s
   of rambling for a 5s line. If you can't get it right, use x-vector only.
2. **Use a clean reference.** One speaker, a quiet room, no music, no reverb. Noise was the
   biggest likeness loss measured.
3. **Use 4–10 seconds of it.** Short is fine; long mostly costs time and raises the chance
   of a rambling take.
4. **Set the language explicitly.** Transcribe does this for you.
5. **Keep best-of-3 on.** The model sometimes serves a dud take, and no slider fixes that.
   Generating several and keeping the best does.
6. **Leave the sampling sliders alone** unless a take rambles. If one does, lower the
   temperature or raise the repetition penalty.
7. **Shape the text.** Prosody follows punctuation, so spell out numbers. Long text is split
   at sentence boundaries and stitched automatically.

## Designing a voice

Describe the voice (timbre, age, gender, accent, emotion, pace), give it a line, press
**Generate**. **Save voice** keeps both the description and the clip. Speak and Chat talk
from the description, which is the brighter sound. The clip can be cloned like any reference,
and that is the steadier identity across a long reply. The Voices tab lets you choose per
voice.

`scripts/make_voice.py` does the same thing from a shell:

```bash
scripts/make_voice.py --name "Marlow" --no-clip \
  --description "A low, unhurried male voice in his fifties, dry, a cigarette in every vowel."
```

Drop `--no-clip` (and pass `--line "…"`) to also render a clip. That needs the GPU, so stop
the web UI first. The script takes the same single-owner lock and will tell you if you forgot.

The shipped voice, **fox**, is made this way at setup time: an expressive adult female voice
defined by a paragraph of text, so the repository contains words, not anyone's recording.

## The voice library (Voices tab)

A saved voice is `voices/<slug>.wav` + `voices/<slug>.json`: a clip and its transcript, a
description, or both. The **Voices** tab lists them all. Click a row to preview the clip, see
the transcript or description, and choose what **Speak and Chat** speak it from:

| Setting | Means |
|---|---|
| Auto | the description if the voice has one, otherwise the clip |
| Its clip | clone the saved clip — the steadiest identity across sentences |
| Its description | Voice Design from the words — the fullest, brightest sound |

**Use as clone reference** loads the voice into step 1 of the Clone tab. **Delete…** asks
once more before it removes the files. Every voice list on every tab refreshes when the
library changes, and when the page is reloaded.

## Speak, Chat and scripts

**Speak** is one call: a voice name and a line. It is also the API that external tools use.
`scripts/say.py` wraps it, so a coding assistant can tell you a build finished:

```bash
scripts/say.py --list
scripts/say.py --voice fox --text "Build finished. Three tests failed."
echo "The migration is done." | scripts/say.py --voice fox
scripts/say.py --voice fox --text "Coffee." --out kitchen.wav --no-play
```

`--url` points it at another machine. `--username` / `--password-file` or `QWEN_TTS_PASSWORD`
log in, and `--insecure` accepts a self-signed certificate.

The endpoints are:

- `/speak` takes `(voice, text, language, temperature, top_p, top_k, repetition_penalty,
  postproc)` and returns audio plus a status line.
- `/voices_available` lists the voices.
- `/generate_voice_clone`, `/generate_voice_design` and `/chat_say` are public as well.

A caller always names a voice, never a path. The name is slugged to `[a-z0-9_]` before it
touches the filesystem.

**Chat** speaks each sentence as soon as it exists, so the first one plays after a few seconds
instead of after the whole reply. The model can't stream samples; sentences are the finest
granularity there is. Overall it runs at about 1.4× realtime. If your browser blocks autoplay,
each sentence has its own play button.

Everything shares one GPU, and requests run one at a time. A long line from a script queues a
UI request behind it.

## Memory safety (read this first)

On WSL2 with dynamic memory, the VM can balloon into nearly all of the host's RAM. So
over-committing doesn't just OOM Linux: it can starve Windows and hard-lock the machine. That
happened once, with two processes each holding ~13GB of weights. Three guardrails stop it
happening again:

1. **A single-owner lock** (`.qwen-tts.lock`, `flock`). The web UI, the smoke test and
   `make_voice.py` compete for it, and the loser refuses to start and names the holder.
   `--force` exists for humans, never for scripts.
2. **Bounded RAM residency.** At most `--max-cpu-resident` (default 2) checkpoints are held
   in RAM. Past that, the least recently used one is dropped and re-read from disk.
3. **Floors.** A checkpoint is only loaded if `MemAvailable` stays above `--ram-floor-mb`
   (default 4096), and only moved to the GPU if it fits with a margin. Otherwise you get a
   `MemoryError` that says why.

Run **one Qwen3-TTS process at a time**. If Windows feels tight, use `--max-cpu-resident 1`,
or cap WSL in `.wslconfig` (`memory=20GB`).

## Exposure and safety

- **It binds to 127.0.0.1** unless you pass `--ip` or `--share`. Doing either without
  credentials prints a warning. Credentials are one shared login (Gradio `auth=`): fine for a
  home LAN, not for the internet. Pass the password with `--password-file` (chmod 600) or
  `QWEN_TTS_PASSWORD`, or let the app prompt for it. `--password` works but warns, because
  any local user can read it in `ps`.
- **Uploads are confined.** Any path the browser names is checked against
  `allowed_input_roots()`: Gradio's upload and cache dirs plus `tmp/media/`. A client cannot
  point the app at an arbitrary file. To allow a curated folder, add it to
  `QWEN_TTS_ALLOWED_MEDIA_DIRS`.
- **Media is bounded.** Uploads are decoded up to 10 minutes, and ffmpeg runs under a
  timeout. Extracted clips in `tmp/media/` are pruned after `QWEN_TTS_MEDIA_TTL_SECONDS`
  (default 6 hours). `voices/` is yours, and it is gitignored.
- **Nothing is logged or uploaded.** Status boxes can echo error text, so treat them as
  visible to whoever is using the page.
- `certs/` holds a private key when you use `--ssl-self-signed`. It is gitignored.

## Ethical use

Voice cloning is the part of this tool that can do harm. Running it offline doesn't change what
the audio is once it leaves the machine. So, plainly:

- **Clone voices you have permission to use.** That means your own, or someone else's with
  their knowledge and consent. Consent to record a clip is not consent to put words in
  someone's mouth.
- **Never impersonate a real person to deceive.** Passing synthetic audio off as someone's
  real voice — to a family member, an employer, a bank, a court — is fraud, and in a growing
  number of places a crime. The software being local is not a defence.
- **Say when audio is synthetic.** Label it in anything you publish or send.
- **No harassment, no fabrication, no vote suppression, no sexualised deepfakes.** These are
  the documented harms of this technology, not edge cases.
- **Saved voices are real people's voices.** That is why `voices/` is gitignored and
  `tmp/media/` is pruned. On a shared LAN, other people's clips sit on your disk.

Nothing in this repo logs, uploads or reports what you synthesise. That is the point of a local
tool, and it is also why the judgement is yours rather than the software's.

## Configuration

Flags are listed by `./run.sh --help`. The environment variables are all optional:

| Variable | Default | What it does |
|---|---|---|
| `QWEN_TTS_MODELS_DIR` | `./models` | Where the `Qwen3-TTS-12Hz-1.7B-*` checkpoints live (`--models-dir` wins) |
| `QWEN_TTS_ASR_MODEL` | `openai/whisper-large-v3` | Whisper model for Transcribe |
| `QWEN_TTS_MEDIA_TTL_SECONDS` | 6 hours | Age after which clips in `tmp/media/` are pruned |
| `QWEN_TTS_ALLOWED_MEDIA_DIRS` | — | Extra directories the path guard accepts, path-separator separated |
| `QWEN_TTS_RAM_FLOOR_MB` | 4096 | Refuse to load another model below this much available RAM |
| `QWEN_TTS_VRAM_MARGIN_MB` | 1024 | VRAM kept free for activations and the vocoder |
| `QWEN_TTS_LOCK` | `./.qwen-tts.lock` | The single-owner lock file |
| `QWEN_TTS_UPSTREAM_REF` | a pinned SHA | Which upstream commit `setup.sh` installs — move it deliberately |
| `QWEN_TTS_PASSWORD` | — | Shared credential for `--username` (also `--password-file`) |
| `QWEN_TTS_USERNAME` | — | Read by `scripts/say.py` for the server login |
| `QWEN_TTS_URL` | `http://127.0.0.1:7860` | Read by `scripts/say.py` (also `--url`) |

**Output post-processing** (sidebar) is set to *Warm* by default. It resamples the 24kHz
vocoder output to 48kHz properly, and trims the near-Nyquist imaging that sounds "digital".
*Off* gives the raw decode. Saved takes are always raw. See
[docs/quality.md](docs/quality.md#output-post-processing).

## Layout

```
setup.sh, run.sh       environment setup and launcher
requirements.txt       direct deps (torch and transformers are pinned in setup.sh on purpose)
webui/app.py           the Gradio UI, model manager, scoring, media and voice library
scripts/say.py         speak a line through the running server (the /speak API)
scripts/make_voice.py  create a described voice from the shell
tests/test_app_logic.py  CPU tier: no GPU or weights needed
tests/smoke_test.py      GPU tier: end-to-end flows that generate and check real audio
docs/                  quality measurements and design rationale
AGENTS.md              the guide for changing this code (CLAUDE.md points to it)
CHANGELOG.md           one entry per change
repo/ models/ voices/ tmp/ certs/   upstream checkout, weights, your voices, scratch, keys (all gitignored)
```

## Tests

```bash
.venv/bin/python tests/test_app_logic.py   # ~15s, CPU only, no weights
.venv/bin/python tests/smoke_test.py       # ~2 min, needs the GPU + weights; stop the server first
.venv/bin/python tests/smoke_test.py --only best_of_n
```

The CPU tier still imports torch and gradio. It builds the whole UI, which is what catches
Gradio API drift before anyone clicks. The GPU tier generates real audio for every flow
(design, clone, design→save→clone, `/speak`, chat, long text, best-of-N, and
video → ffmpeg → trim → Whisper → clone). It checks that each clip is non-silent,
non-clipping and of plausible length. A skipped case counts as a failure unless you pass
`--allow-skip`.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Another Qwen3-TTS process is already holding the weights` | Only one process may own the weights. Stop the web UI before the smoke test or `make_voice.py` |
| `MemoryError: … would leave less than the …MB floor` | RAM is genuinely tight: close other GPU/LLM processes or use `--max-cpu-resident 1` |
| The whole machine hard-locks | WSL2 ballooning — see *Memory safety* |
| Generation takes minutes, GPU ~35% busy | The speech tokenizer is on the CPU; something bypassed `ModelHub._move()` ([design.md](docs/design.md)) |
| A clone rambles or never stops | Almost always a transcript that doesn't match the clip — check the reference checklist. Otherwise lower the temperature or raise the repetition penalty |
| A clone sounds unlike the speaker | Noisy or reverberant reference — pick a cleaner stretch; the checklist flags noise |
| `Transcription came back empty` | The selection is music or silence — move it onto speech, or type the transcript |
| Transcript wrong on names or accents | Expected; the box is editable on purpose. Fix it before cloning |
| `ffmpeg could not read audio from …` | The file has no audio stream; check with `ffmpeg -i file`, or pass `--ffmpeg-bin` |
| Mic button does nothing on a LAN URL | Browsers allow the mic only on localhost or HTTPS: `./run.sh --ssl-self-signed --ip 0.0.0.0` |
| Chat plays a sentence or two, then stops | The browser blocked autoplay; click the next player |
| `no kernel image available` | The torch wheel predates sm_120; `setup.sh` pins 2.8.0+cu128 |
| `CERTIFICATE_VERIFY_FAILED` on HTTPS start | Use `--ssl-self-signed` (run.sh adds it): Gradio's startup self-check dials its own port |
| `Refusing to read a media file outside the upload directory` | A path that didn't come from an upload. Re-upload it, or add the folder to `QWEN_TTS_ALLOWED_MEDIA_DIRS` |
| `Model not found at models/...` | `./setup.sh --models` |

`sox` prints a warning at import time. Only the 25Hz tokenizer uses it, and none of these models
do, so it is safe to ignore.

## License

Apache-2.0 — see [LICENSE](LICENSE). `webui/app.py` began as an adaptation of the official
Qwen3-TTS Hugging Face Space, and everything is built on
[QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (Apache-2.0, Alibaba Qwen). Model
checkpoints and Whisper weights are downloaded separately under their own licenses; nothing
here redistributes them.
