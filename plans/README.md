# Plans — CodeSwitch-TTS "paper-critical + trust" bundle

Twelve self-contained implementation plans produced by the 2026-10-18 `improve` audit of this
repository, plus plan 013 (the 2026-10-20 B/C noise fix). Every plan was written for a zero-context
executor agent: exact anchors, exact old/new strings, exact commands with expected outputs, STOP
conditions, done criteria.
**Read the plan; follow it literally; do not improvise.**

Evidence base: `plans/research/findings-2026-10-18.md` (all six audit reports + subagent reports
verbatim, plus 2026-10-19 addenda) · index: `plans/research/INDEX.md`.

Detailed 013 status, exact metrics, and run commands: `plans/013-bc-noise-status.md`.

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

## Baseline amendment #2 (2026-10-19, post-bundle)

The bundle was followed by three companion commits and then plans 001–002 (all on `main`):

- **`b3167a6`** — `--min-epochs` floor before early stopping in both trainers
  (`train_draft_distill.py`; +2 tests in `tests/test_corpus_builder.py`).
- **`0ee2652`** — ASR generation capped by audio length (`run_full_evaluation.py`; +1 test).
- **`47d8a01`** — `KAGGLE_RUN.md` count sync. **Pre-plan baseline is 101 tests, not 98.**
- **`70577dd`…`9f1256c`** (plan 001, 5 commits) and **`9cd0f40`** (plan 002, 1 commit) —
  executed on their branches and merged fast-forward (branches kept). Current count: **103**;
  terminal sanity check from here: 103 + 23 = 126 (never an assertion).

Both plans' drift checks re-verified every excerpt against the live files **by content** — all
matched; only line numbers moved. For 003+ executors: expect `train_draft_distill.py`,
`run_full_evaluation.py`, `tests/*`, `KAGGLE_RUN.md` to show as changed in
`git diff --stat d69b399..HEAD`; verify by content, never by line number.

**Notebook amendment (2026-10-19, pre-run).** `CodeSwitch_TTS_Kaggle.ipynb` was re-shaped for
the 2000-line re-run on the same day: the distillation cell is split into phase 1 (teacher) and
phase 2 (train, now `--min-epochs 300`); new cells add the CB0-only diagnostic and a listening
step; the evaluation cell uses `--asr-model openai/whisper-large-v3` with fresh `*_2k` output
dirs; the artifact cell zips those. Cells that were `>= 19` in the pre-big-run layout shift by
**+4** (plans quoting "markdown cells 21/23" find them at 25/27; cells `< 19` stay put). A new
standalone `CodeSwitch_BigRun_Kaggle.ipynb` mirrors the flow without the demo cells. Plans
003/004/005/007/011/012 touch this notebook — their anchors are content strings and still
match; only the quoted cell indices move.

**Command fix (2026-10-19, pre-run).** The `train_draft_distill.py` invocations in both
notebooks and the launch pack now pass `--num-sentences 2000`. Previously none passed it —
and the trainer silently caps any external corpus at its 300-sentence default
(`train_draft_distill.py:1182`), so runs that followed the old commands trained on 300
sentences, not the full 2 000. (`KAGGLE_RUN.md` §2b still shows the flag-less commands —
that file is owned by plans 003-012 and was left untouched.) Plans quoting those commands
should drift-check by content.

**Hotfix (2026-10-19, post-plan-002).** Plan 002's tail-coverage change kept *ragged
trailing* windows (`t mod 86` frames, e.g. 75), and `_batches` stacks raw windows — the
first big run crashed at the first residual batch (`RuntimeError: stack expects each tensor
to be equal size, but got [86] ... and [75]`, `train_draft_distill.py:945`) after both the
teacher pass (2 000/2 000 shards, 5 h 36 m) and the draft stage (best val 5.1146, CB0 top-1
7.45% ≈ 76× chance) had succeeded. `build_windows` now right-aligns the trailing window
(every window exactly `WINDOW` = 86 frames; full frame coverage; shards shorter than
`WINDOW` excluded and logged) and `_batches` raises a clear `ValueError` if window lengths
ever mix again. The plan-002 window test now checks the uniform-length contract.

**W&B artifact backups (2026-10-19, post-hotfix).** `push_artifacts_to_wandb.py` (new) snapshots
the run's expensive outputs to W&B as artifacts so a dead Kaggle session can no longer cost the
5.5 h teacher pass: `--group teacher` → `codeswitch-teacher-cache` (`teacher_cache/` + corpus +
stats), `--group student` → `codeswitch-student` (`codeswitch_ckpt/`), `--group bundle` →
`codeswitch-artifacts` (final zip). Entry names are fixed, so `--download <group> --dest .`
(`/kaggle/working` for the bundle) restores the exact repo-relative layout; `:latest` always
resolves to the newest push. The teacher / train / zip cells of both notebooks now end with the
push (`!pip install -q wandb` first; a missing key or flaky upload warns and continues), and the
stale "Save Version" advice was replaced — Quick Save is not a reliable `/kaggle/working`
backup. Tests: `tests/test_push_artifacts.py` (17 cases; suite 103 → 120).

## Status table

Statuses: `TODO` · `IN PROGRESS` · `DONE` · `BLOCKED` · `REJECTED`.

| # | Plan | Priority | Effort | Δ tests | Branch | Status |
|---|---|---|---|---|---|---|
| 001 | [paper-table-stats-fixes](001-paper-table-stats-fixes.md) | P1 | S | +1 | `advisor/001-paper-table-stats` | DONE |
| 002 | [residual-window-coverage](002-residual-window-coverage.md) | P1 | S | +1 | `advisor/002-residual-window-coverage` | DONE |
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
| 013 | [bc-noise-fix](013-bc-noise-fix.md) | P0 | L | +5 | `fix/013-bc-noise` | IN PROGRESS |

Plan 013 is the 2026-10-20 follow-up to the B/C noise diagnosis. Its evidence base is
`research/findings-2026-10-20.md` (repo-root `research/`, not `plans/research/`). It
conflicts with plan 006 on `codeswitch_tts/utils/audio.py::DACWrapper._load_dac`;
execute 013 first and rebase 006 by content (or drop 006's DAC-loader section).
Code/tests for Phases 1–4 have landed on `fix/013-bc-noise`; the Kaggle retrain,
argmax evaluation, and quality gates are still pending.

**2026-10-20 residual-only amendment (post-v2 listening).** The matched v2
hybrids isolate the residual as the primary bottleneck: `teacher_full` is clear,
`student_cb0_argmax_teacher_res` is muffled but understandable, and
`teacher_cb0_student_res_argmax` is robotic. The v2 residual also selected its
best state on **teacher** CB0 (`best_epoch=11`) and stopped at 3.3 % draft
scheduled sampling, so the saved state never saw the inference distribution.
The next run is residual-only from the v2 draft, not another combined run:

```bash
python train_draft_distill.py --skip-teacher --skip-draft \
  --draft-ckpt codeswitch_ckpt_v2 \
  --residual-epochs 150 --residual-ramp-epochs 100 --residual-sched 1.0 \
  --residual-d-model 512 --residual-layers 6 \
  --min-epochs 150 --max-minutes-residual 90 \
  --ckpt-dir codeswitch_ckpt_v3
```

`train_draft_distill.py` now loads the real draft under `--skip-draft` (fails
closed without `--draft-ckpt` for a residual run), copies it into the output
directory, validates on **draft CB0**, and deploys the best state found during
the **full-draft hold** (not an early pre-ramp state). It records
`best_sched_p`/`best_val_sched_p`/`best_epoch_any`/teacher-vs-draft top-1,
supports a 6-block residual (`_RESIDUAL_DILATIONS=(1,2,4,8,16,32)`), and holds
the full draft for the post-ramp epochs. The Kaggle notebook defaults to
`DO_TRAIN=False`, `DO_RESIDUAL_RETRAIN=True`, `CKPT_V3`, and uses `EVAL_CKPT`
in the push/diagnostic/smoke/eval/bundle cells. Tests: `131 passed`
(`movio/.venv/bin/python -m pytest tests/ -q -p no:cacheprovider`). The 5 %
`rt_a`/`rd_a` gate is unchanged; do not run the 60-utterance suite until the v3
diagnostic passes and the one-sentence argmax WAV is confirmed intelligible.

**2026-10-21 v3 diagnostic result (gate failed).** The residual-only run
completed in 31m14s and selected the full-draft state at epoch 139
(`best_sched_p=1.0`, `selected_full_draft=true`, `cb_top1_teacher=2.03 %`,
`cb_top1_draft=1.56 %`). On the unchanged 4-shard diagnostic the teacher path
cleared the gate but the draft path did not:

| metric | v2 | v3 | gate |
|---|---:|---:|---:|
| `cb0_s` / `cb0_a` | — | 2.364 / 9.942 % | — |
| `rt_s` / `rt_a` | 3.195 / — | 1.773 / **5.594 %** | `rt_a >= 5 %` PASS |
| `rd_s` / `rd_a` | 1.640 / — | 0.094 / **1.834 %** | `rd_a >= 5 %` FAIL |

This is not a ramp or selection miss (`best_epoch_any=149`,
`best_sched_p_any=1.0`). It is the draft-CB0 bottleneck the plan predicted:
the residual works on teacher CB0 but not on the draft CB0 it sees at
inference. The v3 `student_full_argmax` WAV is also near-silent
(`rms≈0.0098`, `flat≈0.622`, `zcr≈0.199`) while `student_full_sample` and
`teacher_cb0_student_res_argmax` are normal. `diagnose_audio_bug.py` now writes
`student_cb0_argmax_teacher_res`, `student_cb0_argmax_res_sample`,
`student_cb0_sample_res_argmax`, and `student_cb0_only_argmax`, and records
CB0 uniqueness/top-token/run-length/entropy for draft sample vs argmax.

**2026-10-21 argmax isolation result (draft mode collapse).** The new hybrid
paths are conclusive. On all 4 shards, draft sample and draft argmax are not
the same sequence:

| stat | sample | argmax |
|---|---:|---:|
| unique tokens | 152.25 | 1.50 |
| top-token fraction | 8.7 % | 99.9 % |
| longest run | 2.5 frames | 382.75 frames |
| CB0 entropy | 4.51 nats | 0.01 nats |

The argmax path emits token `568` for the whole utterance. That is why
`student_cb0_argmax_teacher_res` is muffled (constant coarse code, teacher
fine codes) and `student_full_argmax` is silent (constant coarse code, v3
residual trained on the same constant code). It also means the residual-only
v3 scheduled-sampling input was degenerate: `precompute_draft_cb0` uses
`sample=False`, so the residual never saw a varied draft-CB0 sequence.
**Do not run another residual-only v3-style run.** The next step is
`tune_draft_prior.py`, a no-retrain sweep that subtracts
`alpha * log p_teacher(token)` from the draft logits and reports CB0 top-1,
sequence diversity, residual top-1, and teacher-residual WAVs. If no alpha
produces a varied, intelligible CB0, retrain the draft with a sequence-level
diversity/regularization objective before touching the residual. Do not run the
one-sentence smoke or the 60-utterance suite until the draft path is fixed.
Test count is now `134`.

**2026-10-21 prior sweep rejected; true per-sample draft repair implemented.**
`tune_draft_prior.py` completed on the cached shards. Subtracting
`alpha * log p_teacher(token)` from the draft logits did not rescue the path:

| alpha | cb0 top-1 | `rd_a` | argmax unique | top frac | longest run |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 10.019 % | 1.834 % | 1.5 | 99.9 % | 383 |
| 0.25 | 11.240 % | 1.827 % | 1.5 | 99.9 % | 383 |
| 0.50 | 11.433 % | 1.797 % | 1.5 | 99.9 % | 383 |
| 0.75 | 7.927 % | 1.398 % | 2.0 | 99.5 % | 300 |
| 1.00 | 0.663 % | 0.225 % | 20.0 | 25.6 % | 60 |

Only `alpha=1.0` diversifies CB0, and it does so by moving to rare tokens while
CB0 accuracy falls to chance. The listener confirmed that every
prior-adjusted WAV is muffled/weird. **Prior correction is rejected; do not use
it as the fix.**

The actual identified defect was only partially fixed by plan 013 Phase 3.
`length_bucketed_order` reduced the batch-max padding, but every sample was
still interpolated to `max_t`, so short sentences were trained on a stretched
text→frame alignment. This is now fixed:

- `TemporalLengthProjector` accepts a per-sample `frame_lengths` tensor. Each
  sample is interpolated to its own DAC frame count; padded tails are masked
  after each convolution/norm so a batch's valid tail is identical to a
  one-sample forward.
- `DraftTransformer.forward(..., frame_lengths=...)` returns logits padded to
  the batch maximum; `generate_chunk(..., num_frames=...)` is unchanged.
- `DraftTrainer._collate` returns per-sample lengths, and both `_run_epoch` and
  `evaluate_topk` pass them to the model.
- Draft validation now records `val_argmax_unique`, `val_argmax_top_frac`,
  `val_argmax_longest_run`, and `val_argmax_entropy_nats` in `meta.json`, so a
  draft-only retrain can be rejected before spending residual time.

Tests: `137 passed` with
`movio/.venv/bin/python -m pytest tests/ -q -p no:cacheprovider` for the
per-sample slice; the full suite is `140 passed` after the v5 decoder tests.

**2026-10-22 v4 draft-only result (triage failed).** The draft-only run completed
in 23m54s and used the existing 2 000-shard cache (`--skip-teacher
--skip-residual`). Best validation loss was `5.0542 @ epoch 12`; the final draft
block was:

| metric | v2 | v4 | triage |
|---|---:|---:|---:|
| `val_top1` | 8.2483 % | 8.26 % | `< 10 %` FAIL |
| `val_top5` | — | 23.07 % | — |
| `val_argmax_unique` | — | 2.1 | `< 16` FAIL |
| `val_argmax_top_frac` | — | 98.8 % | `> 40 %` FAIL |
| `val_argmax_longest_run` | — | 332 | `> 100` FAIL |
| `val_argmax_entropy_nats` | — | 0.05 | — |

Per-sample frame lengths were necessary but not sufficient: the argmax path is
still the teacher prior, not text conditioning. **Do not train the residual from
`codeswitch_ckpt_v4_draft`.**

**2026-10-22 v5 draft repair: text-conditioned frame decoder.** The next lever
is architectural, not another residual or loss-only tweak. `DraftModelConfig`
now has `frame_decoder_layers: int = 0`; nonzero creates a `FrameDecoder` with
learned frame queries, sinusoidal frame positions, a whole-utterance length
embedding, frame self-attention, and cross-attention to the text encoder memory.
Per-sample frame lengths use `tgt_key_padding_mask`; text padding uses
`memory_key_padding_mask`. The default `0` keeps the legacy
`TemporalLengthProjector` path, so v2/v3/v4 checkpoints load without new keys.
The CLI flag is `--draft-frame-decoder-layers`.

Run the v5 draft only, on the existing cache, into a new directory:

```bash
python train_draft_distill.py \
  --corpus-json codeswitch_corpus.txt \
  --num-sentences 2000 \
  --skip-teacher --skip-residual \
  --draft-epochs 300 --draft-batch 24 \
  --draft-frame-decoder-layers 2 \
  --min-epochs 300 --max-minutes 60 \
  --teacher-cache teacher_cache \
  --ckpt-dir codeswitch_ckpt_v5_draft
```

Inspect `codeswitch_ckpt_v5_draft/meta.json` with the same triage:

- if `val_argmax_unique` is still ~1–2 and `val_argmax_longest_run` is still
  near the utterance length, stop. Cross-attention did not repair conditioning;
  the next levers are class-balanced/focal loss or multi-sample teacher targets,
  not another residual run.
- if the argmax is materially diverse and `val_top1` improves on v2's
  `0.082483`, train the residual from the repaired draft:

```bash
python train_draft_distill.py \
  --corpus-json codeswitch_corpus.txt \
  --num-sentences 2000 \
  --skip-teacher --skip-draft \
  --draft-ckpt codeswitch_ckpt_v5_draft \
  --residual-epochs 150 --residual-ramp-epochs 100 \
  --residual-sched 1.0 \
  --residual-d-model 512 --residual-layers 6 \
  --min-epochs 150 --max-minutes-residual 90 \
  --teacher-cache teacher_cache \
  --ckpt-dir codeswitch_ckpt_v5
```

Then run the unchanged 4-shard diagnostic. The 5 % `rt_a`/`rd_a` gate is
unchanged: do not run the one-sentence smoke or the 60-utterance suite until
both pass and the v5 argmax WAV is intelligible.

Before the 60-utterance run, `CodeSwitch_013_RunAll_Kaggle.ipynb` now synthesizes
one held-out sentence (`ta-01`) in `argmax` mode, prints draft CB0 / residual
top-1 against a fresh Parler teacher target, and saves
`output_audio/test_fixed_argmax.wav`. The full evaluation flags default to
`False`; flip them only after that WAV is confirmed intelligible. The standalone
driver is `smoke_test_argmax.py`.

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
