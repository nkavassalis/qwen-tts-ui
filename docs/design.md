# Design notes

Why the code is shaped the way it is. The rules that must not be undone, and the tests that
enforce them, are in [AGENTS.md](../AGENTS.md). This file is the background.

## Model residency: CPU-resident weights, an LRU GPU slot

The official Hugging Face Space keeps five checkpoints on the GPU permanently, roughly 20GB.
A 16GB card can't hold that. Here, each 1.7B model (~4.3GB) lives in host RAM, and a small LRU
"GPU slot" (`--max-resident`, default 1) promotes the one in use. Switching between Clone and
Design costs ~2s, and nothing is re-read from disk unless `--max-cpu-resident` forced an
eviction.

The memory guardrails (the single-owner lock, bounded CPU residency, the RAM and VRAM floors)
are explained in the README's *Memory safety* section. They exist because this machine
hard-locked once.

## The speech tokenizer must be moved explicitly

`model.speech_tokenizer` is a `Qwen3TTSTokenizer` *wrapper*, not an `nn.Module`, so
`model.to("cuda")` leaves the 12Hz vocoder on the CPU. The symptom is dramatic and silent:
generation runs ~25× slower (141s vs 6s) with the GPU ~38% busy. `ModelHub._move()` walks the
model's attributes and moves any wrapper that holds an `nn.Module`.
`test_move_reaches_the_speech_tokenizer` guards it.

## `sdpa`, not flash-attn

`flash-attn` 2 ships no sm_120 (Blackwell) wheels. The official Space uses `sdpa` too, and
torch's bundled flash kernels are still used where they apply.

## Bounded generation

The checkpoint ships `max_new_tokens=8192`. ICL cloning occasionally fails to emit EOS; one run
produced 163s of audio for a single sentence. Generation is capped at `_budget(text)`, which is
`min(2048, 4 × len(text))` tokens. That cap is a runaway guard, not a quality limit: see
[quality.md](quality.md#long-text-tempo-is-the-limit-not-the-token-cap).

## Clone prompts are cached by content

Encoding a reference (codec codes plus speaker embedding) is the expensive half of a clone.
`_clone_prompt` keys it by a SHA-256 of the PCM bytes plus the transcript and the mode, and
keeps the last 8 in an LRU. The Clone tab and Speak/Chat share it, so alternating between two
references doesn't re-encode either one.

An earlier key, (length, float sum), let two different clips share a prompt. On a shared LAN
that meant answering in someone else's voice.

## Speak and Chat replay what they have already said

`RenderCache` keeps the raw decode of every sentence Speak and Chat render, under
`tmp/render_cache/<slug>__<hash>.wav`. The hash covers an *identity* plus the sentence. The
identity is everything that shapes a raw decode:

- the voice's content: the cleaned clip's PCM hash, transcript and mode, or the description;
- the language;
- the sampling settings;
- the weights (checkpoint folder and `config.json` mtime);
- `RENDER_CACHE_VERSION`.

Because the key is content, nothing ever has to be invalidated. Edit the transcript, change
`prefer`, move a slider, swap the weights: the identity changes and the old entries age out.
Deletion is the exception. Renders of a person's voice must not outlive the saved voice, so
deleting it removes its `<slug>__*` files.

What is stored is the raw decode, and post-processing runs after the cache, so Warm, Clean and
Off share entries. A take is stored only if it passed the pace and clipping checks, because a
cached take replays forever. The Clone and Design tabs bypass the cache entirely: a Generate
button that returned the same take twice would be broken.

The cost is variety. A cached sentence sounds identical every time. For notification lines
that's a feature, and it also keeps a designed voice from drifting on repeated sentences.

## Whisper large-v3 through transformers

Whisper runs through transformers rather than faster-whisper. That reuses the torch/CUDA
stack (ctranslate2 kernels may not cover sm_120), it transcribes 6s in ~3s, and it loads only
on first use. The padding mask is passed explicitly, so Whisper doesn't hallucinate over the
30s of zero-padding a short clip gets.

Whisper is used four ways:

- **Transcribe**: beam 4, `no_repeat_ngram_size=4`, on the selection only. It also reports
  the language it detected, which the Clone tab uses to set the target language.
- **WER** of each best-of-N take against the requested text.
- **The reference check**: the confidence scorer below, run on the *reference* and its
  transcript, to catch a transcript of the right length with the wrong words.
- **Confidence**: one teacher-forced forward pass of the *requested* text. The decoder
  prefix (`<|startoftranscript|><|lang|><|transcribe|><|notimestamps|>`) is built
  explicitly, and the text is tokenised without special tokens. Before this release it was
  tokenised *with* them and prefixed again (see [quality.md](quality.md#best-of-n-how-a-take-is-picked)).

## State that crosses the browser is JSON

`gr.State` round-trips through the client, so the media state and the take state hold paths
and numbers, never arrays or tensors. The audio a later request needs (the uploaded clip, each
generated take) is written under `tmp/media/` and named by path. Every path the browser sends
back goes through `resolve_input_path`. A module-level "last generation" global is exactly what
once let one LAN user's Save button keep another user's clip, so there is none.

## The UI layout rules

The UI was reorganised around four rules, and `tests/test_app_logic.py` pins each one:

1. **One Status box per tab.** Every action on that tab reports there. The old Clone tab had
   seven (Status, Media, Selected clip, Fit check, Save status, Take status, Voice library) and nobody
   could tell where to look.
2. **Diagnostics live in the sidebar.** The VRAM/RAM line used to be appended to every
   status. Now it is a sidebar panel on a 5s timer, outside the GPU queue
   (`concurrency_limit=None`).
3. **One Save per tab, saying what it saves.** The Clone tab had two name boxes and two
   buttons, one for the reference and one for a take. Now one radio offers *the reference*
   or *take #k*, and one button saves it. Saving never overwrites.
4. **The library is managed in one place.** The Voices tab lists, previews, deletes and sets
   the Speak/Chat route. Every action that changes the library refreshes every voice list on
   every tab, as does a page load.

The reference arrives through Upload, Record or Library. All three land in the same place: a
waveform, a selection, and a read-only *Reference clip* player that holds exactly what will be
cloned. The checklist under it is recomputed from that player, so it is correct however the
reference arrived.

## The waveform is drawn with PIL

There is no matplotlib dependency, the pixel geometry is deterministic, and
`test_render_waveform` can assert that the shaded selection really changes the image.

## Saved voices: clip, description, or both

A record is `voices/<slug>.json` plus an optional `voices/<slug>.wav`. `voice_route()` decides
how Speak and Chat say it. `prefer` is set on the Voices tab, and *auto* falls back to
`voice_kind()`: the description if there is one, otherwise the clip. The trade-off behind that
default is in [quality.md](quality.md#where-the-thinness-comes-from-cloning-keeps-60-of-the-top-end).
Writes are atomic (temp file + `os.replace`), so a crash can't leave a half-written sidecar.

## Dialects

The (removed) CustomVoice checkpoint had dialect speakers, and upstream rejects
`language="beijing_dialect"`: the accent came from the speaker, never from the language
field. `test_language_choices_match_checkpoint` keeps dialects out of the dropdown if that
capability ever returns. Italian *is* selectable; the hosted Space omits it by oversight.
