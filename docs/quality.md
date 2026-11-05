# Quality: what was measured, and what it means

Everything here was measured on the development box (RTX 5070 Ti, 1.7B checkpoints), not
guessed, because the obvious knobs turned out not to matter. Two metrics, the same ones
upstream benchmarks with:

- **SIM**: cosine similarity between reference and output, using the model's own speaker
  encoder.
- **WER**: Whisper large-v3's transcript of the output, against the text that was requested.

Method: one English target sentence and a single speaker per sweep, n=2–4 per row. Read any
SIM difference under ~0.005 as noise.

## The reference is the lever, not the sliders

**Decode settings.** Same reference, same text, two seeds each:

| config | SIM | WER |
|---|---|---|
| defaults (t=0.90, top-p 1.0, top-k 50, rep 1.05) | 0.9912 | 0.000 |
| `non_streaming_mode=True`, t=0.90 | 0.9910 | 0.067 |
| `non_streaming_mode=True`, t=0.70, top-p 0.95 | 0.9913 | 0.000 |
| defaults, t=0.70, top-p 0.95 | 0.9916 | 0.000 |

All four are inside the noise.

**The reference.** Same target text and speaker, varying only the reference:

| reference | SIM | WER | output length |
|---|---|---|---|
| 1.8s clean | 0.987 | 0.000 | 5.0s |
| 4.5s clean | 0.992 | 0.000 | 5.5s |
| 9s clean | 0.993 | 0.133 | 5.6s |
| 9s + broadband noise (SNR 10 dB) | **0.973** | 0.133 | 5.4s |
| 9s at 10% peak level | 0.991 | 0.133 | 5.7s |
| 9s, x-vector only | 0.991 | 0.133 | 4.9s |
| 9s, **transcript does not match the clip** | 0.978 | **1.000** | **17.4s** |

Here is what follows from that, and what the app does about each finding:

1. **A mismatched transcript is the worst failure.** It produced 17.4s of audio for a 5s
   sentence, at WER 1.0. The Clone tab checklist runs a plausibility check (words per second
   against the selection length) as you type. That check catches the usual causes: the
   transcript of the whole upload, or a stub left in the box. It can't catch a transcript of
   the right length with the wrong words. For that, Generate also asks Whisper how likely
   your transcript is given the reference audio. This is the same teacher-forced scorer that
   grades takes, pointed at the reference instead. A matching transcript should read near 0.
   Below `REF_TEXT_CONF_WARN` (−2.0) the status and checklist warn and show what Whisper
   heard. **Measured on the GPU box** — four references against four degrees of wrongness:

   | transcript vs the clip | confidence |
   |---|---|
   | exact | **−0.12 .. −0.51** (adding broadband noise to SNR 5 dB moves it <0.05) |
   | one word wrong (~5% WER) | −1.11 .. −1.34 |
   | half wrong (~50%) | −2.79 .. −3.18 |
   | unrelated (>100%) | −4.59 .. −5.09 |

   −2.0 is a good warn line: it clears a single wrong word by 0.66 and fires around 40–50%
   wrong. It is **not** a safe block line — the half-wrong class straddles it, so blocking
   there would be a coin flip at the boundary. Generation therefore refuses at
   **`REF_TEXT_CONF_BLOCK = −3.5`**: 3.0 below the worst correct transcript measured, 2.2
   below the worst one-word-wrong, and above every half-wrong case, so it fires only on "that
   is not this clip at all". Between the two lines it warns and generates, which matters — an
   imperfect Whisper transcript is a thing to edit, not an error to obey.

   The threshold is not a guess about how bad "bad" is. A reference whose transcript is not
   its own words failed to say the target line in **12 of 12** takes and took **twice** as
   long to render (5.6s of speech became 9.5–11.1s, worst 18.2s) — and it did that to a
   *clean* clip too, so a good recording is no defence. Accurate transcripts in that same test
   read −0.40 and −0.43; the mismatched pairs read −3.95 and −3.99, leaving the two
   populations 3.4 apart at their closest. `scripts/measurements/replicate_wer.py` reproduces
   it. The refusal names the confidence and prints what Whisper heard, so the fix is obvious
   from the status box: correct the transcript in step 1, or press Transcribe and edit it.

   One honest limit: the check runs when Whisper is resident (best-of-N loads it, Transcribe
   loads it, and the checklist loads it on request). On a single-candidate clone with Whisper
   not yet loaded there is **no verdict, and so no block** — absence of a check is not a pass.

   CJK works on every signal: confidence (the Japanese clip reads −0.51 exact), and the WER
   and length checks, which count kana, Han and Hangul per character. (One calibration row
   read identical to exact because the probe made its "one word wrong" variant by splitting
   on spaces, a no-op in Japanese. That was the probe, not the app.)
   `smoke_test.py --only media_pipeline` prints the right-vs-wrong pair. It runs only when
   Whisper is already loaded or best-of-N is about to load it.
2. **Noise is the biggest likeness loss** (−0.020). The checklist estimates the SNR from the
   level distribution, and warns below 20 dB.
3. **Length matters, but only so much.** 1.8s → 9s bought +0.006 SIM, most of it in the
   first few seconds. Selections are clamped to 2–15s, and the target is 6s.
4. **Level doesn't matter.** A clip peaking at 0.1 scored 0.991. That is why *Clean up the
   reference* does not normalise.
5. **Sampling variance is real.** Over four seeds on identical inputs, SIM spread only 0.001,
   but WER was 0.000 on one seed and 0.133 on the others. That is what best-of-N is for, and
   why the Clone tab now defaults to 3 takes.

## What *Clean up the reference* does

It is conservative on purpose, because nothing in it may change the voice:

- **DC removal and a 2nd-order 60Hz high-pass.** This removes handling noise, HVAC rumble
  and mains hum: energy the codec and the speaker encoder would otherwise spend on things
  that aren't the voice.
- **Edge trim to ~0.2s.** ICL cloning *continues* the reference's codec codes, so a clip that
  ends in two seconds of room tone hands the model room tone to continue. The trim keeps a
  short natural pause instead.
- **No denoiser.** A denoiser's artefacts are exactly the kind of thing the codec reproduces
  faithfully.

If cleanup would leave less than the clone minimum, or fails for any reason, the original
clip is used.

**Measured on the GPU box, 2026-11-03** (`scripts/measurements/clean_reference_ab.py`: same reference, same
target line, same three seeds per arm, identity scored by the model's own speaker encoder
against the *raw* reference so both arms face the same target):

| reference | SIM of clones vs raw | WER | energy >9kHz | clone length |
|---|---|---|---|---|
| raw | 0.9870 ± 0.0008 | 0.000 | 0.0472 | 4.69s |
| cleaned | 0.9886 ± 0.0007 | 0.000 | 0.0470 | 4.80s |

Neutral to marginally better, and specifically **not** a cost: the top end is unchanged
(0.0472 → 0.0470), so the 60Hz high-pass takes nothing away where the thinness problem lives,
and the reference's own embedding barely moves (0.9991 against 1.0) — it is not changing the
voice. The SIM gain is +0.0016, which is about two standard deviations of the spread within an
arm: real, but small enough that you should keep the switch on for the edge trim and the rumble
removal rather than expecting a leap. On this reference the edge trim was a no-op (no silence
longer than 0.2s at either end), so the clip is the same length in both arms.

## Choosing the reference stretch

*Auto-pick* used to choose the loudest 6s, which favours shouting, music beds and clipped
peaks. It now scores every 6s window on three things:

- **speech coverage**: the share of 20ms frames above an adaptive threshold set between
  the noise floor (5th percentile level) and the speech level (95th);
- **a heavy penalty for clipped samples**;
- **level**, as a tie-break only.

It then moves each edge into the nearest pause within 0.35s, so the reference doesn't start
or end mid-word. Snapping never shrinks a selection below the 2s minimum. A recording with
no pauses at all (music, a tone, constant noise) leaves the selection alone, and the
checklist says why.

## Best-of-N: how a take is picked

Each candidate gets a penalty, and the lowest penalty wins:

| signal | what it catches | weight |
|---|---|---|
| **WER**: Whisper vs the text you asked for | wrong, missing or extra words | ×2.0, dominant |
| **confidence**: Whisper's mean per-token log-prob of the *intended* text, given the audio | a take that is barely decodable | ×0.4 below `CONF_GOOD` |
| **defects**: speaking rate vs text length, edge silence, clipping | dead air, machine-gun pacing, blown peaks | capped at 0.3 |
| **SIM**: speaker encoder vs your reference | near-identical takes | tie-break only |

Confidence is scored against **the text you asked for**, never against Whisper's own
transcript. Measured on a noise-degraded take, grading Whisper's own answer missed the damage,
while grading the intended text caught it. Whisper is confident about whatever it decided to
hear.

**The confidence scale changed in this release, and the new scale is now measured.** The old
scorer tokenised the text *with* Whisper's special tokens and then prepended the decoder prefix
again. Every take was charged for two impossible tokens and scored in a corrupted context. That
is why the old docs said "a clean take measures about −1.9" and set the threshold at −2.6. With
the fix, on the GPU box:

| take | WER | conf | SIM | defect flags |
|---|---|---|---|---|
| best_of_n #1 | 0.00 | **−0.02** | 0.984 | — |
| best_of_n #2 | 0.00 | **−0.06** | 0.987 | 0.9s silence |
| best_of_n #3 | 0.00 | **−0.02** | 0.984 | — |
| fox's shipped clip (English) | — | −0.03 | — | — |
| a saved Japanese take | — | −0.51 | — | — |

Clean speech sits at −0.02 to −0.06, exactly the "near zero" the fix predicted, and a real
Japanese take sits at −0.51 — below clean English, nowhere near a disaster. `CONF_GOOD = −1.00`
sits clear of every real measurement, so it is kept (silence was not re-measured on the new
scale, so how far below −1 it lands is not yet known);
it is no longer provisional, though it is calibrated from a handful of takes rather than a
sweep. The confidence term is still a disaster detector, not a quality meter: #1 and #3 were
identical to two decimal places and the ranking still came from defects and SIM.

The old measurements below are kept for their shape, not their absolute values:

| take (old scorer, seed 7, 4 candidates) | WER | conf | SIM | defect flags | penalty |
|---|---|---|---|---|---|
| #1 | 0.00 | −2.01 | 0.988 | 0.9s silence | 0.12 |
| #2 | 0.00 | −1.91 | 0.988 | 0.8s silence | 0.10 |
| #3 | 0.00 | −1.99 | 0.987 | 0.5s silence | 0.01 |
| **#4 (kept)** | 0.00 | −2.09 | 0.985 | — | **0.00** |

All four takes said the line perfectly, so WER couldn't separate them, and SIM preferred #1.
The defect term picked #4, because #1 was carrying 0.9s of dead air.

Nothing here measures naturalness, prosody or emotion. When the numbers read as a tie, trust
your ears. That is why every take stays on the page.

## Where the thinness comes from: cloning keeps ~60% of the top end

Same three lines, raw 24kHz decode, energy above 9kHz:

| source | energy >9kHz | pairwise SIM (identity) |
|---|---|---|
| real reference clip (a broadcast sample) | 1.51m | — |
| a take cloned from it | 0.89m | — |
| a designed voice | **2.67m** | 0.974 |
| clone of a generated take | 0.57m | **0.993** |

- **The vocoder is not the culprit.** White noise crosses the 12Hz codec at 95% and a
  generated output round-trips at ~90%. The top end is missing *before* decoding, in the
  tokens the talker chose, so post-processing cannot restore it.
- **Cloning tracks the reference's spectrum.** `generate_icl_prompt` gives the talker the
  reference's own codec codes as a prefix, so the clone inherits the reference's spectral
  character as well as its timbre. Cloning a clone loses more each generation.
- **The two routes trade off.** A frozen clip is the steadier identity (0.993 vs 0.974 when
  a description is re-rolled for every sentence). A description is the fuller sound. That is
  why a saved voice can keep both, and why the Voices tab lets you choose per voice.

The practical advice is to clone from a **real recording** where you can. If you save a
generated take, keep it for Speak and Chat, not as the next clone's reference.

And a warning about the metric itself: **energy above 9kHz is a diagnostic of the codec, not a
measure of accuracy.** A competing model measured 1.8× ours in that band and sounded *worse* —
see *Fish Audio S2-Pro* at the end of this file. Do not chase that number.

## Several clips: an experiment, and its first result

Qwen3-TTS conditions on one reference: one clip's codec codes plus transcript, and one
speaker embedding. More clips can only help by filling those two slots better.
`scripts/multi_clip_lab.py` tests each candidate against **the clip you would use today**:

| version | codes + transcript from | embedding from |
|---|---|---|
| `single` (baseline) | your clip | your clip |
| `single_mean` | your clip | average of all build clips |
| `concat` | all build clips joined (≤20s) | the joined clip |
| `cleanest` | the clip that does best on the reference checks | that clip |
| `xvec_mean` | nothing (x-vector mode) | average of all build clips |

`single_mean` differs from `single` only in the embedding, so its pair isolates the effect of
averaging. `cleanest` asks a different question: would picking a better single clip have
done as well?

**How it is judged.** Each round holds out one clip, never yours, and builds every version
from the rest. The listener gets one forced choice per pair. They hear the held-out real
clip, then A and B: your clip's take and one other version, on the same sentence and seed,
in random order and loudness-matched. The answer is A, B or *can't tell*. A forced choice
against a reference is what listeners are reliable at. Rating several takes at once on a
1–5 scale was not: the scale drifts, and it never asks "better than what I have?"

Each version gets a tally against your clip and an exact two-sided sign test. There is no
verdict under 10 decided pairs; 7 of 10 is p≈0.34, so not a finding. Held-out SIM gets the
same paired comparison. SIM against the build-clip average is deliberately not reported: the
encoder that made the average would be grading it.

Two checks on the listener are built in:
- **Identical pairs.** When the cleanest clip *is* your clip, its pair plays the same take
  twice, and the honest answer is *can't tell*.
- **Side bias.** The results show how often A was chosen.

**When it is worth building.** A version earns a place in the app only if:
- the listener verdict is **better**;
- held-out SIM doesn't disagree;
- WER is no worse.

"No clear difference" after 20+ pairs is an answer too.

**Caveat.** Clips cut from one long recording share a mic and room, so they are a kinder
test than clips from different days.

### What it found

Measured on the 5070 Ti (2026-11-04): one speaker, 24 clips cut from a single long recording,
Japanese, one sentence, 4 seeds × 4 rounds = 16 takes per version, **51 of 64 blind pairs
answered by ear**.

| version | you preferred it / your clip / can't tell | Δ SIM vs your clip | mis-said the sentence |
|---|---|---|---|
| `single` (your clip) | — | — | **12/16** |
| `single_mean` | 5 / 5 / 5 | +0.002 | 11/16 |
| `concat` | 2 / 8 / 2 | −0.001 | 2/16 |
| `cleanest` | 4 / 6 / 2 | +0.002 | **0/16** |
| `xvec_mean` | 5 / 6 / 1 | +0.002 | 1/16 |

Side bias 21/41 (p=1.00), so the randomisation held and the instrument worked.

**Averaging the speaker embedding does nothing you can hear.** All five versions sit between
held-out SIM 0.9737 and 0.9764, while re-rendering *one* version with only the seed changed
moves SIM by **±0.0055** — the entire spread between versions is half the noise of a single
take. Every ear verdict is "no clear difference", and `single_mean` — the pair that isolates
averaging — was called *can't tell* 5 times in 16. **Not worth the complexity.**

**What did move is which clip you pick, and it moved on intelligibility, not identity.** The
baseline, chosen as the most *typical* clip by embedding, failed to say the sentence in 12 of
16 takes. The clip chosen by the technical reference checks never did, and `xvec_mean` and
`concat` failed once or twice (paired exact p=0.003, 0.002, <0.001 against the baseline's 12).
Typicality does not predict getting the words right; **cleanliness does**. That is a
reference-selection lesson, not an averaging one — and it is the actual finding, sitting in
the table's last column.

**Caveats on the result.** One speaker, one language, one sentence, all clips from one
session. The *ranking* is paired on that sentence and stands; the absolute rates do not
generalise until a second sentence and a second speaker agree.

**Two instrument bugs this run exposed, now fixed** in `summarise` / `summary_markdown`,
pinned by `test_multi_clip_results_do_not_read_noise_as_signal`:

- SIM wins/losses counted any delta above `1e-9`, so a +0.0001 wobble was a "win" and the
  table printed `13/3 (p=0.02)` for a difference well inside one take's own noise — then
  advised itself that held-out SIM disagreed with the ears. There is a `SIM_DEADBAND` of
  0.0025 now; the same data reads 6/1 (p=0.12), which agrees with the listener.
- The table showed mean WER only, which hid a 12/16-vs-0/16 failure rate behind two decimals
  that looked equal. It now counts failed takes and tests them paired (McNemar's exact, on the
  pairs where the two versions disagree).

It also prints the take-to-take noise beside the deltas, and says **"you can stop rating"**
once every version is flat on both measures: 64 pairs was never going to resolve a zero
effect, and rating them is real work someone has to give up an hour for.

## Output post-processing

The vocoder is fixed at 24kHz, so nothing can add real bandwidth above 12kHz. What *can* be
fixed is the fizz. The raw decode had 0.155% of its energy in the top 5% of the band, where a
properly band-limited signal has 0.036%. That 4× excess of near-Nyquist imaging comes from
transposed-convolution upsampling, and it is what people describe as "digital".

| setting | SIM | centroid | HF energy | near-Nyquist |
|---|---|---|---|---|
| Off — raw 24kHz | 0.9900 | 2044 Hz | 3.26% | 0.155% |
| Clean — soxr HQ → 48kHz | 0.9893 | 2029 Hz | 3.11% | 0.036% |
| *(a cheap linear resampler, for scale)* | 0.9884 | 1898 Hz | 2.17% | 0.038% |
| **Warm** — 48kHz + de-image + gentle air (default) | 0.9879 | 2164 Hz | 3.83% | 0.027% |

The SIM change, 0.9900 → 0.9879, is inside the seed-to-seed spread. *Warm* ends up brighter
than raw because the low-pass is followed by a small high-shelf. Its final soft limiter is a
`tanh` that runs on every sample, not only on peaks. That is slight colouring at normal
levels, but it is unmeasured; see AGENTS.md. Saved takes always bypass post-processing.

## Chat: the first piece is the wait you notice

Generation time tracks the length of the audio being generated, and Chat streams one piece at
a time. So the wait before anything plays is the time to render the first piece. Chat now cuts
that first piece to 4–10 words, at a sentence end or a comma (never mid-phrase, never on
"and"). The rest of the reply is chunked as before. Speak is unchanged.

Measured on the 5070 Ti, same reply, hub already on the GPU, render cache disabled — so the
only difference is the first piece's length:

| | first audio | first piece | whole reply |
|---|---|---|---|
| quick start off | **11.49s** | 26 words / 7.92s | 22.6s (3 pieces) |
| quick start on | **5.30s** | 10 words / 3.20s | 24.5s (4 pieces) |

Halves the wait. The whole reply gets ~2s **slower**, because a fourth piece means a fourth
model call — that is the trade, and it is the right one when someone is waiting to hear
anything.

Two things that number hides, both measured:

- **Where the rest of the wait is.** Cold hub (first chat after startup) the same piece lands
  at 7.14s, and the tier — hub *and* clone-prompt cache already warm from earlier cases —
  prints 2.2s. Promotion (~2–3s) and building the clone prompt (~2–3s) are part of "time to
  first audio" the first time you ask, and invisible afterwards. Quote the context or the
  number is meaningless.
- **It only helps when there is punctuation early.** A reply whose first clause runs 15 words
  to the first comma is left at 15 words: the cut is lossless and refuses to split
  mid-clause. So "4–10 words" is a target, not a promise.

Does the short first piece sound worse? On measurements, no: it ends in silence (RMS of the
last 80ms is 0.000, so it is not cut mid-word) and carries as much top end as the later
pieces (energy above 9kHz 0.051 against 0.052/0.033/0.042). **That is measurement, not
listening** — the pieces are in `tests/output/first_piece_*.wav` for your ears.

Repeated sentences skip generation entirely: see *render cache* in the README and
[design.md](design.md#speak-and-chat-replay-what-they-have-already-said).

## Long text: tempo is the limit, not the token cap

| words | audio | tokens | wall time | peak VRAM | words/s | WER |
|---|---|---|---|---|---|---|
| 18 | 5.9s | 74 | 8.7s | 4.36 GB | 3.05 | 0.00 |
| 60 | 17.7s | 221 | 23.3s | 7.52 GB | 3.39 | 0.62 |
| 150 | 45.7s | 571 | 60.5s | 7.58 GB | 3.28 | 0.55 |
| 300 | 74.3s | 929 | 100.7s | 7.58 GB | **4.04** | **0.68** |
| 300, cap raised to 4096 | 74.3s | 929 | 111.4s | 7.58 GB | 4.04 | 0.68 |

VRAM is not the limit, and neither is the 2048-token cap: EOS arrives on its own. The model
speeds up as the sequence grows, and intelligibility drops with it. So long text is split at
real sentence boundaries into chunks of 12–26 words (20–42 CJK characters). Abbreviations,
decimals, initials, URLs and list markers are protected. The chunks are rendered separately
and stitched with trimmed padding and a 15ms equal-power crossfade. A chunk that comes out at
an implausible rate is re-rolled once.

The same 300 words after chunking came out as 92.6s of audio at 2.48 words/s, WER 0.42. It
took 131s instead of 101s, which is still ~1.4× realtime. Requests over ~40 chunks are
refused.

## A style instruction on the clone path: accepted, obeyed by nothing

The hybrid someone will ask for next: clone the identity from a clip and *direct the reading*
per line — "read it skeptically", "warm and unhurried". Mechanically it is about ten lines.
`generate_voice_clone` forwards `**kwargs` straight into `model.generate()`, which handles
`voice_clone_prompt` (speaker embeddings) and `instruct_ids` (style embeddings) as independent
additions to the same talker input, on different code paths. Nothing stops it.

Measured 2026-11-04 with `scripts/measurements/instruct_probe.py`: one reference, one sentence,
one seed, one sampling configuration, four conditions × three seeds — including a **style-free
control** ("Read the sentence."), without which any difference could be the mere presence of
extra embeddings rather than their meaning.

**The instruction is not ignored, and that is the bad outcome.** At the same seed the takes are
not identical — lengths and waveforms differ — so the embeddings reach the model and perturb
generation. But they do not do what the words say:

- *"warm, gentle, unhurried"* produced the **fastest** speech of any condition (20.4 chars/s
  against 19.0 for no instruction).
- *"skeptical, impatient"* was indistinguishable from saying nothing (18.9 chars/s, F0 233.5
  against 240.3).
- MFCC distance from no-instruction: control 0.0003, skeptical 0.0001, warm 0.0007. **The seed
  noise floor is 0.0009.** Every instruction effect sits underneath the noise from merely
  changing the seed, including between the two conditions that should differ most (0.0003).

An input that is silently dropped is inert and obvious. This one changes the audio while
carrying no meaning — a control that looks alive. It is the same failure family as the tokenizer
left on CPU and the sampling sliders that turned out to be live only because
`generation_config.json` says `do_sample: true`, and the reason the Clone tab has no
style-instruction box. If it is ever revisited, it should be with a model that was *trained* on
instructions, and it should be measured the same way before any UI exists.

## Does a clip's *delivery* transfer through cloning? Mostly no

The idea being tested: keep several clips per voice, ranked by how they were delivered (brisk,
calm, animated, flat), and let the user pick which read to condition on per line. Measured
2026-11-04 with `scripts/measurements/delivery_transfer.py`, because building it untested would
have meant a clip-suggestion UI that changes nothing.

24 clips cut from one recording of one speaker, ranked by measured delivery, then the same
Japanese sentence cloned from the four extremes across five seeds (20 takes).

**Speaking rate does not transfer at all.** A reference spoken at 2.40 characters/s and one
spoken at 7.63 — a **3.2× difference** — produced outputs at 6.30 and 6.76, a **7% difference**.
The target sentence sets the pacing; the clip's pace is discarded. A "brisk read vs calm read"
picker would therefore be pure decoration: detectable, rankable, inert.

**Pitch variability does transfer**, in two tiers about 2× apart, and it survived re-running on
IQR and MAD rather than standard deviation — which matters, because `voiced_frac` differed
between the groups and unvoiced frames are exactly where F0 estimation produces octave errors
that inflate a σ.

| reference | its F0 σ | output F0 σ | output IQR | output MAD |
|---|---|---|---|---|
| flattest | 15.6 | 19.0 ± 3.6 | 28.3 | 14.8 |
| fastest | 22.5 | 27.5 ± 12.5 | 30.9 | 14.5 |
| most animated | 91.0 | 46.3 ± 11.7 | 52.6 | 25.3 |
| slowest | 77.7 | 46.5 ± 13.6 | 62.1 | 31.3 |

Three honest limits. The transfer is **compressed** — references spanning 15.6→91.0 (5.8×) come
out spanning 28→62 (2.2×), so you get *more* animated, never *as* animated. Voiced fraction
covaries with the tiers (0.80/0.67 low, 0.57/0.53 high), so "varies in pitch more" is not fully
separable from "breathier" — fine when you are picking a clip, wrong when you are labelling one.
And pitch **centre** does not transfer (166–291 Hz in, 197–223 Hz out), which is the point: that
is identity, and it holds.

Fish Audio reached the same conclusion about their own products from the other side: the drama-3
README advertises that laughter, crying and whispers are "**no longer flattened by the
reference**". In these models the reference flattens delivery. That is what the row above says,
independently, in the vendor's own words.

## Fish Audio S2-Pro: evaluated on this box, and not better

Evaluated 2026-11-04 and **rejected**. Recorded so nobody spends an afternoon and 18GB of disk
rediscovering this. The scripts that produced these numbers are in `scripts/measurements/`
(`fish_clone_ab_refs.py`, `fish_clone_s2pro.py`, `fish_clone_measure.py`, `fish_tag_probe.py`,
`fish_transcribe_check.py`); they need an 11GB download and a second venv, and there is no good
reason to run them again.

**What it is.** `fishaudio/s2-pro`: a 4B dual-autoregressive TTS (slow AR 4B on the time axis,
fast AR 400M for 9 residual codebooks) on a Qwen3-Omni backbone, RVQ codec at ~21Hz, decoding at
44.1kHz. 11GB (9.1 weights + 1.87 codec). Fish Audio Research License — non-commercial only, and
redistribution must display "Built with Fish Audio".

**The test.** Identical conditioning into both models: two references cut from one recording of
one speaker (5.76s and 18.72s), two target sentences (Japanese and English), two seeds — eight
takes each. Same instruments on both sides.

| model | reference | >9kHz | >11kHz | SIM→reference | WER | SNR | seconds/take |
|---|---|---|---|---|---|---|---|
| S2-Pro | 5.8s | **104.2** | **21.1** | 0.961 | 0.458 | 24.1 | 10–19s |
| S2-Pro | 18.7s | **97.4** | **19.7** | 0.980 | 0.000 | 24.1 | 33–176s |
| Qwen3-TTS | 5.8s | 60.2 | 5.4 | 0.981 | 0.000 | 21.2 | ~2–4s |
| Qwen3-TTS | 18.7s | 53.4 | 4.5 | 0.982 | 0.000 | 24.7 | ~2–4s |

**Verdict: ours sounds more accurate, without fail.** That was a listening judgement, and it is
the one that decided it — but the interesting part is *which* metric disagreed with it. Every
number that could favour S2-Pro favoured it: 1.8× the energy above 9kHz, 4× above 11kHz, where
the separation was clean (worst Fish take 14.2, best ours 6.3). None of that survived the ear.
Above 11kHz the comparison is partly our own 24kHz Nyquist talking — there is nothing above 12kHz
to measure — and at 9kHz the ranges overlap slightly (Fish min 67.4, ours max 72.1). A fuller
top end is not a more accurate voice. This is the second time this session a spectral proxy
pointed away from what listening said; the first was best-of-N scoring.

**The one thing worth keeping: reference length.** S2-Pro's identity keeps improving with more
reference (SIM 0.961 → 0.980 from 5.8s to 18.7s). Ours is flat (0.981 → 0.982) — it saturates
well before 6s. So `REF_TARGET_SECONDS = 6.0` is now a measured default rather than a guess, and
asking the user for a longer clip would buy us nothing. Their 10–30s claim is true *of their
model*, which is the honest way to read a vendor number.

**A structural gap, not a bug.** One of their eight takes was asked for English over a Japanese
reference and **produced Japanese** ("The results came back this morning" → 結果は午前に戻りました,
WER 1.83, confidence −1.26). Their request schema has **no language field at all** — 18 fields,
none of them language — so the language is inferred from text with the reference in context, and
nothing pins it. Our explicit language dropdown, with its checkpoint-checked values
(invariant 4), is load-bearing in a way this experiment only just demonstrated. 1/8 vs 0/8 is not
significant on its own (Fisher p ≈ 1.0); the claim is that the failure mode exists and is
architectural.

**Why integration was never realistic anyway:** ~10× slower per take, and worse with the long
reference their model prefers (33–176s for two seconds of audio). It reports `GPU Memory used:
20.33 GB` on a 15.9GB card and has **no VRAM check anywhere** — the WSL2 driver pages the overflow
to host RAM and the first run crawled at 1.7–2.5 *seconds per token*, silently, with no error.
That is our tokenizer-left-on-CPU bug arriving from the other direction, and it is exactly what
`ModelHub`'s floors exist to prevent. `transformers==5.19.0` cannot share a venv with our 4.57.3,
so it needs a second process, which is what the `flock` single-owner lock exists to stop.

**Also measured, same model, same session: its advertised inline tags.** S2-Pro claims 15,000+
`[tag]` controls. Four were tested against baseline with two seeds each, and **one of four
worked**. `[whisper]` is real and physically correct — −9.3dB, −62Hz pitch centre, −23 pitch σ,
and *brighter* (+1654Hz centroid), which is what breathy turbulent airflow does. `[screaming]`,
`[sad]` and `[excited]` were inside the seed-to-seed noise floor; `[screaming]` never rose above
baseline at all. The whisper take was then transcribed to make sure "brighter" had not meant
"noise over garbled words": WER 0.00, confidence −0.03, so it is genuinely whispered speech.
No tag moved the speaking rate — the third independent confirmation in this project that pacing
comes from the text, not from a direction.

**Two incidental facts worth recording.** Their stack runs on Blackwell consumer: `torch
2.8.0+cu128` sees sm_120 fine, so the flash-attn class of problem did not recur. And Whisper's
feature extractor **segfaults instead of raising** when handed an absurd-length input (a probe
bug wrote a WAV at sr=3, i.e. 149,760 seconds) — which the app never hits because clip length is
bounded upstream, but any new probe script should assert the length before transcribing.

## Apple Silicon: does MLX leave anything on the table?

The question was whether running the same checkpoints through
[mlx-audio](https://github.com/Blaizzy/mlx-audio) on a Mac is a downgrade you can hear, or a
downgrade you can only see in a config file. Quantised weights are exactly the kind of change
that produces numbers that look fine and audio that isn't, so this was measured before it was
adopted, and the ear test is Nick's.

**Setup.** Same reference (`fox`), same three sentences, same sampling, same instruments as every
other comparison here. MLX side: `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit` on an M4 Pro
(64GB unified). GPU side: the local bf16 checkpoint on the 5070 Ti.

| take | seconds | rate | level | >9 kHz | WER | conf |
|---|---|---|---|---|---|---|
| MLX en | 2.24 | 15.6 | −23.5 | 61.6 | **0.00** | −0.07 |
| MLX en2 | 3.76 | 19.9 | −22.9 | 40.0 | **0.00** | −0.02 |
| MLX ja | 4.24 | 5.4 | −22.4 | 70.1 | **0.00** | −0.50 |
| GPU en (bf16) | 1.76 | 19.9 | −22.0 | 80.5 | 0.00 | −0.04 |
| GPU en2 | 3.76 | 19.9 | −22.1 | 25.9 | 0.00 | −0.02 |
| GPU ja | 4.40 | 5.2 | −23.5 | 71.2 | 0.00 | −0.47 |

**Nothing was left on the table.** Word error rate 0 on both, confidence matching take for take,
top-end energy in the same band (and the spread within each column is wider than the spread
between them — this is the take-to-take variance measured elsewhere on this page, not a
quantisation effect), and both at 24kHz so the bandwidth comparison is honest. **8-bit is not the
artifact here.**

**Speed** surprised in the good direction: 2.5–3.0s per take warm on the M4 Pro against ~5s on
the 5070 Ti for the same sentence, and a full best-of-N (3 candidates, Whisper scoring, speaker
similarity) requested over the LAN came back in **5.3s**. The decode is only a few dozen tokens
for a sentence like these, so the 64GB unified pool is not the bottleneck this workload thinks it
is. Do not generalise that to an 8GB machine.

Two things worth recording about how this was measured, because both nearly produced a wrong
answer:

- **My first pass reported WER 1.00 for every MLX take.** That was the measurement script
  keying results by filename after stripping a `_000` suffix; the transcripts had been
  character-perfect all along. A metric that disagrees with what you can hear is a bug until
  proven otherwise.
- **Confidence is Whisper's, not the TTS model's.** It comes from a teacher-forced pass in
  `Transcriber`, so best-of-N keeps WER, confidence, defect penalties and the SIM tie-break on
  the Mac with no changes. The initial assumption that an MLX port would lose it — because
  mlx-audio keeps only the last position's logits — was wrong, and would have shipped a
  documented "limitation" that didn't exist.

**Still open, and platform-independent.** Every take that goes through the app peaks at exactly
**1.000** and sits ~6dB hotter (−15.5 to −17 dB) than the same model's raw decode (peak ~0.5,
−22 dB). It does this identically on Linux and on the Mac, so it is not an MLX effect and was not
introduced here — but "peak exactly 1.000" is also what clipping looks like, and nobody has
established which it is. It is a question for the output path, not for this port.
