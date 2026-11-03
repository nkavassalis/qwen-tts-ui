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
   the right length with the wrong words, so Transcribe runs on the selection only, and the
   box stays editable.
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

**Measured on the GPU box, 2026-11-03** (`tmp/clean_reference_ab.py`: same reference, same
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
