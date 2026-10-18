# Plans — CodeSwitch-TTS "paper-critical + trust" bundle

Twelve self-contained implementation plans produced by the 2026-10-18 `improve` audit of this
repository. Every plan was written for a zero-context executor agent: exact anchors, exact
old/new strings, exact commands with expected outputs, STOP conditions, done criteria.
**Read the plan; follow it literally; do not improvise.**

Evidence base: `plans/research/findings-2026-10-18.md` (all six audit reports + subagent reports
verbatim, plus 2026-10-19 addenda) · index: `plans/research/INDEX.md`.

## Ground rules (apply to every plan)

- Repo: `/home/tripathiji/projects/all_work`. One plan → one branch (`git checkout -b
  advisor/NNN-slug`), commits in the plan's own rhythm, **never `git push`**. The maintainer
  merges; note Kaggle's notebook cell 6 hard-resets the repo to `origin/main`, so a branch
  that is not merged+pushed never reaches a Kaggle run.
- Interpreter: `movio/.venv/bin/python` for everything (the venv has **no `pip`**; the
  `movio/.venv/bin/pytest` console script has a broken shebang — **never** use it).
- Tests: `movio/.venv/bin/python -m pytest tests/ -q -p no:cacheprovider` from the repo root.
  Script-style `pytest tests/` fails collection (`ModuleNotFoundError: run_full_evaluation`).
- Executors: run the plan's drift check first; honor every STOP condition; when done, flip this
  plan's status row to `DONE` (unless the dispatching reviewer maintains the table).
- Run plans **strictly sequentially**. They share files heavily: `KAGGLE_RUN.md` is touched by
  all twelve; `CodeSwitch_TTS_Kaggle.ipynb` by 003/004/005/007/011/012; `run_full_evaluation.py`
  by 001/005/006/007/008/011/012; the test files by 001–005/007–009/012; `push_to_wandb.py` by
  005/008. Never run two plans concurrently.

## Baseline amendment (2026-10-19)

The plans were written against `d69b399`. Since then one compatible commit landed on `main`
(user-approved) and was pushed:

- **`f689736`** — "Create the results directory before writing the provenance file": one added
  line at `run_full_evaluation.py:3196`, fixing a fresh-checkout crash in `check_provenance`
  (it wrote `eval_results/provenance.json` before the later `mkdir`). Root cause + repro +
  post-fix verification are recorded in the findings log's 2026-10-19 entry.

All plans remain valid — every anchor is content-based. Line numbers quoted for
`run_full_evaluation.py` **after `:3196` shift by +1** (e.g. 012's gc block `:3412–3417` is at
`:3413–3418` on `f689736`). Expected pre-state: `git log --oneline -3` lists `f689736` (or a
descendant) and `d69b399`.

`plans/` is tracked on `main` as of the bundle commit; drift checks have a stable reference.

## Status table

Statuses: `TODO` · `IN PROGRESS` · `DONE` · `BLOCKED` · `REJECTED`.

| # | Plan | Priority | Effort | Δ tests | Branch | Status |
|---|---|---|---|---|---|---|
| 001 | [paper-table-stats-fixes](001-paper-table-stats-fixes.md) | P1 | S | +1 | `advisor/001-paper-table-stats` | TODO |
| 002 | [residual-window-coverage](002-residual-window-coverage.md) | P1 | S | +1 | `advisor/002-residual-window-coverage` | TODO |
| 003 | [demo-verifier-honesty](003-demo-verifier-honesty.md) | P1 | M | +3 | `advisor/003-demo-verifier-honesty` | TODO |
| 004 | [honest-engine-telemetry](004-honest-engine-telemetry.md) | P1 | S–M | +1 | `advisor/004-honest-telemetry` | TODO |
| 005 | [secrets-hygiene-wandb](005-secrets-hygiene-wandb.md) | P0 | S–M | +3 | `advisor/005-secrets-hygiene-wandb` | TODO |
| 006 | [supply-chain-pinning](006-supply-chain-pinning.md) | High | ~1–2 h | ±0 | `advisor/006-supply-chain-pinning` | TODO |
| 007 | [dependency-declaration-unification](007-dependency-declaration-unification.md) | P2 | S–M | +2 | `advisor/007-dependency-declaration-unification` | TODO |
| 008 | [training-eval-contract-tests](008-training-eval-contract-tests.md) | P2 | M | +2 | `advisor/008-training-eval-contract-tests` | TODO |
| 009 | [normalizer-and-cli-tests](009-normalizer-and-cli-tests.md) | P2 | S–M | +9 | `advisor/009-normalizer-and-cli-tests` | TODO |
| 010 | [dx-onboarding](010-dx-onboarding.md) | P2 | S–M | ±0 | `advisor/010-dx-onboarding` | TODO |
| 011 | [docs-truth-pass](011-docs-truth-pass.md) | Medium | ~1 h | ±0 | `advisor/011-docs-truth-pass` | TODO |
| 012 | [perf-quick-wins](012-perf-quick-wins.md) | Medium-High | ~2–3 h | +3 | `advisor/012-perf-quick-wins` | TODO |

## Execution order & hard gates

Locked order: **001 → 002 → 003 → 004 → 005 → 006 → 007 → 008 → 009 → 010 → 011 → 012.**
The gates below are enforced by the dependent plans' own drift checks; they exist so dispatch
scheduling cannot break them:

- 003 → **004** (004 edits the code 003 introduces).
- 006 → **007** (torch floor in `requirements.txt`), **008** (`weights_only=True` literal gate),
  **010** (pins it documents).
- 007 → **010** (`requirements-dev.txt` it documents).
- 009 → **011** (corpus-stats key gate).

## Test-count convention (authoritative)

`KAGGLE_RUN.md:427` is a hand-maintained total; **the pytest summary is the only source of
truth**. Every test-adding plan records the pre-count, verifies the delta from the table below,
and writes the number pytest actually printed.

| Plan | 001 | 002 | 003 | 004 | 005 | 006 | 007 | 008 | 009 | 010 | 011 | 012 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Δ tests | +1 | +1 | +3 | +1 | +3 | ±0 | +2 | +2 | +9 | ±0 | ±0 | +3 |

If every delta lands exactly, in the locked order, the final count is **98 + 25 = 123** — a
sanity check, never an assertion. Cumulative figures written inside a plan's text (e.g. "107 on
the bare 98-baseline tree") are as-of-that-plan illustrations; when earlier plans have already
landed, the real number is higher. Plans 002 and 003 give the only in-text chain examples
("99 → 100 if 001 already landed"; "103 if 001–003 landed in order") and they are consistent
with the table above.

## Finding coverage

Finding ids are from `plans/research/findings-2026-10-18.md`. Findings not listed here are
outside this bundle — see "Not selected".

| Plan | Findings addressed |
|---|---|
| 001 | CORRECTNESS-04 (Table V ±0.0), CORRECTNESS-05 (W&B cold TTFA silently dropped), CORRECTNESS-07 (switch-distance sentinel), CORRECTNESS-08 (Table I gate-threshold pooling) |
| 002 | CORRECTNESS-03 (residual shards' trailing windows dropped) |
| 003 | CORRECTNESS-01 (demo verifier fallback to `randn` target) |
| 004 | CORRECTNESS-02 (Mode A simulated TTFA), DOCS-01, DOCS-07, DEBT-02 (smaller half) |
| 005 | SECURITY-01 … SECURITY-04 (W&B key persistence/upload; whole-directory artifact uploads) |
| 006 | supply-chain report A1–A3, A5–A7 (immutable revision pins; `weights_only=True`; A4's phantom premise corrected) |
| 007 | DEPS-02, DEPS-03, DEPS-04, DEBT-07, SECURITY-03 (dependency declarations) |
| 008 | TESTS-01, TESTS-02, TESTS-03 (dry-run wiring, row schema, checkpoint round-trip; also reconciles the hand-shaped W&B fixture flagged under CORRECTNESS-05) |
| 009 | TESTS-05, TESTS-06, DOCS-06 (+ DOCS-09 context) |
| 010 | DX-01 … DX-06, DOCS-08, DOCS-09 (as scoped in the plan) |
| 011 | DOCS-02, DOCS-03, DOCS-04, DOCS-05, DOCS-10 |
| 012 | PERF-02, PERF-03, PERF-04, PERF-05 (+ the ASR-batching addendum from the 2026-10-19 deep-researcher run) |

## Not selected (so this bundle is not mistaken for the full audit)

Deferred **by the plans themselves** (they point here):

- **CORRECTNESS-06** — high-pass guard passes 12–15-sample inputs to `sosfiltfilt`; deferred
  by 001.
- **CORRECTNESS-09** — crossfade applied to already-contiguous chunks (≈5.8 ms deleted per
  join + comb); deferred by 001 — bug-vs-intent needs a product decision.
- **DEBT-03** — engine prefix-stop vs eval per-frame rejection sampling (changes acceptance
  semantics); deferred by 004.
- **DEBT-02 (larger half)** — unify the demo chunk loop with
  `EvalSynthesizer._chunked_synthesis`; deferred by 004.
- **parler-tts pip git-install pinning** — out of scope in 006 (010 documents the gap);
  SpeechMOS weight-content and ASR-model revision pins recorded as follow-ups in 006.
- **`--no-utmos` help-text size inaccuracy** (~50 MB vs the real 411 MB asset) — recorded as a
  follow-up by 011.

Outside the selected set (no plan covers; candidates for a future bundle):

- **PERF-01** — teacher-forced verification recomputes the growing prefix every chunk
  (quadratic in utterance length).
- **TESTS-04** — the model-backed `SpeculativeEngine` methods and the full Mode A path stay
  Kaggle-only (004 pins the chunk-telemetry half via `test_mode_a_chunk_telemetry_is_honest`).
- **DEBT-01 / DEBT-04 / DEBT-05 / DEBT-06** — god-module split, metric-spec triplication,
  boilerplate + no logging layer, dead code (refactor tier).
- **DEPS-01** — `transformers==4.46.1` advisory, blocked by parler-tts's `<=4.46.1` upper pin;
  **DEPS-05** — staleness audit, informational.
- **DIRECTION-01 … DIRECTION-06** — product/roadmap directions from the direction audit.

## Verification provenance

- `research/findings-2026-10-18.md` — all six audit reports + every subagent report verbatim,
  plus 2026-10-19 addenda: anchor verification for plans 006/008–012; the supply-chain /
  ASR-batching deep-researcher run; the fresh-run crash reproduction + fix (`f689736`); and the
  plan-012 probes (ASR batch equivalence on tiny-whisper at `batch_size` 2/3/4; notebook-edit
  dry-run; memory-guard trial).
- `research/INDEX.md` — index of the log files.
