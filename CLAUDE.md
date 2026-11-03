# CLAUDE.md — qwen-tts

The agent doc for this repo is **[AGENTS.md](AGENTS.md)**. Read it and follow it — it has
the mode, the run/test commands, the design rationale, the invariants (each with the test
that enforces it), the do-not-touch list and the list of what was never verified.

It lives in `AGENTS.md` so that every agent tool reads the same document; this file is a
pointer, not a second source of truth. If you are about to duplicate a rule here, put it
in `AGENTS.md` instead.

The three rules that have cost the most to learn, repeated because they are the easy ones
to "tidy" away:

1. **Never run two model-owning processes at once.** The `flock` single-owner lock
   enforces it; `--force` is for humans, never for a script or CI.
2. **Never move model-loading logic out of `ModelHub._move()`.** `model.speech_tokenizer`
   is a non-`nn.Module` wrapper; missing it costs ~25x inference and raises nothing.
3. **Keep the WSL2 ceiling in mind before raising any cap.** `MemAvailable` here is the
   host's RAM, not headroom; over-committing has hard-locked the machine once.
