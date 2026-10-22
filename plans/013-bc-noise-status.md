# Plan 013 — CodeSwitch-TTS B/C Noise Fix: Detailed Status

**Snapshot:** 2026-10-22  
**Branch:** `fix/013-bc-noise`  
**Last code commit at this snapshot:** `086c9ea` (`feat(draft): add opt-in text-conditioned frame decoder`)  
**Current status:** **blocked on v5 draft training**; no B/C quality claim.

This document is the detailed, self-contained record of what Plan 013 is trying to
fix, what has been tried, what the exact measured results were, what changed in
the code, and what happens next. The shorter running log is
`plans/README.md`; the original implementation plan is
`plans/013-bc-noise-fix.md`.

---

## 1. One-paragraph summary

Configs B and C still fail the paper-quality code gate because the
non-autoregressive draft model's **argmax** Codebook-0 (CB0) path collapses to a
near-constant token (`568`), even though its **sampled** path is diverse. The
residual decoder works on teacher CB0 (`rt_a = 5.594 %`) but fails when fed the
draft's CB0 (`rd_a = 1.834 %`, gate is `>= 5 %`). A residual-only v3 run was
therefore not enough. The next hypothesis was that per-sample frame alignment
was wrong; that was fixed in v4, but the draft still produced only
`val_top1 = 8.26 %`, `argmax unique = 2.1`, `top_frac = 98.8 %`, and
`longest_run = 332`. Per-sample lengths were necessary but not sufficient.
The current code adds an **opt-in text-conditioned frame decoder** with frame
queries, frame self-attention, and cross-attention to the text encoder. It is
pushed as v5 and is ready to be trained as a draft-only run. The 5 % gate and
the smoke/full-evaluation ordering are unchanged.

---

## 2. Goal and non-negotiable constraints

### 2.1 Goal

Make CodeSwitch-TTS Configs B and C intelligible and defensible **without
losing the existing latency story**, using a real Parler/DAC provenance and the
unchanged code-level gate.

### 2.2 Hard gate (do not lower)

The paper-quality code gate is:

- `rt_a >= 5 %` — residual top-1 with **teacher CB0** in argmax mode.
- `rd_a >= 5 %` — residual top-1 with **draft CB0** in argmax mode.

Both are measured by `diagnose_audio_bug.py` on the same 4-shard diagnostic with
a real DAC backend (`hf_pretrained` or `dac_official`). A failure in either one
blocks any B/C quality claim.

There is also a separate, earlier **draft triage** used only to save Kaggle
time. It is not the paper gate:

- `val_top1 >= 10 %`
- `val_argmax_unique >= 16`
- `val_argmax_top_frac <= 40 %`
- `val_argmax_longest_run <= 100`

If the draft triage fails, the residual must not be trained from that draft.

### 2.3 Ordering constraints

1. Do not run the one-sentence argmax smoke test until the argmax path is
   confirmed intelligible.
2. Do not run the 60-utterance suite until the smoke WAV is intelligible and
   the 5 % code gate passes.
3. Do not lower the gate, do not reinterpret a failed gate as a pass, and do
   not report B/C as fixed on sampled audio alone.
4. Keep the real teacher/cache/checkpoint provenance.
5. Do not rerun the ~6-hour teacher pass if the 2 000-shard cache exists; use
   `--skip-teacher`.
6. Do not overwrite `codeswitch_ckpt_v2`, `codeswitch_ckpt_v3`, or
   `teacher_cache`.
7. Do not run `--dry-run` with the default checkpoint directory.
8. Do not edit `KAGGLE_RUN.md`; amendments belong in `plans/README.md` or this
   status file.
9. Test with:
   `movio/.venv/bin/python -m pytest tests/ -q -p no:cacheprovider`.
10. Push code changes; the Kaggle notebook must remain directly runnable.

---

## 3. Vocabulary and system map

| Term | Meaning |
|---|---|
| CB0 | DAC codebook 0, the coarse acoustic token sequence predicted by the draft. |
| CB1–8 | The eight residual DAC codebooks predicted by the residual decoder. |
| Draft | `DraftTransformer`, a non-autoregressive text-to-CB0 model. |
| Residual | `ResidualDecoder`, maps CB0 (teacher or draft) to CB1–8. |
| `cb0_s` / `cb0_a` | Draft CB0 top-1 against teacher CB0, sample / argmax. |
| `rt_s` / `rt_a` | Residual top-1 with teacher CB0, sample / argmax. |
| `rd_s` / `rd_a` | Residual top-1 with draft CB0, sample / argmax. |
| Argmax path | `sample=False`; used by the evaluation and by `precompute_draft_cb0`. |
| Sample path | `sample=True`; diverse but not reliable. |

Data flow:

```text
text
  -> CharacterTokenizer (vocab 4096)
  -> text embedding + positional encoding
  -> 4-layer Transformer encoder
  -> memory (batch, text_len, d_model)
  -> legacy TemporalLengthProjector
       OR opt-in FrameDecoder (frame queries + self-attn + cross-attn to memory)
  -> cb0_head
  -> CB0 logits / probabilities / tokens
  -> ResidualDecoder
  -> CB1-8 logits / tokens
  -> DAC
```

The failure is specifically in the **draft CB0 argmax** path. The residual is
not the first defect: it works on teacher CB0; it fails on the constant draft
CB0 it is given at inference.

---

## 4. Status at a glance

| Item | State |
|---|---|
| v2 draft `val_top1` | `0.082483` = 8.2483 % |
| v3 residual-only | trained 150 epochs; W&B run `edmjhmnm` |
| v3 `rt_a` | `5.594 %` — **PASS** |
| v3 `rd_a` | `1.834 %` — **FAIL** (`< 5 %`) |
| v3 draft argmax | near-constant token `568` |
| v4 per-sample draft | trained 300 epochs in 23m54s; **triage failed** |
| v4 `val_top1` | `8.26 %` (essentially v2) |
| v4 argmax unique | `2.1` |
| v4 argmax top fraction | `98.8 %` |
| v4 longest run | `332` frames |
| v5 cross-attention decoder | implemented, tested, pushed as `086c9ea` |
| v5 draft training | **not yet run** |
| v5 residual training | **not yet run; requires v5 draft triage pass** |
| Current test count | **140 passed, 3 warnings** |
| B/C quality claim | **none** |

---

## 5. Experiment history and exact results

### 5.1 Baseline / v2

- The v2 draft was the last usable draft before the repair attempts.
- Recorded v2 draft `val_top1`: **`0.082483` = 8.2483 %**.
- v2 diagnostic (from the v3 comparison table):
  - `rt_s = 3.195 %`
  - `rd_s = 1.640 %`
- v2 was the source for the v3 residual-only run.
- The v2 draft was not itself producing an intelligible B/C argmax path; the
  later isolation showed the draft argmax was the limiting factor.

### 5.2 v3 residual-only run

Purpose: test whether the residual could be improved independently of the
draft, using the v2 draft as the source.

Configuration/contract:

- `--skip-teacher --skip-draft`
- draft source: `codeswitch_ckpt_v2`
- residual capacity: `d_model=512`, `num_layers=6`
- `residual_epochs=150`, `residual_ramp_epochs=100`, `residual_sched=1.0`
- `min_epochs=150`, `max_minutes_residual=90`
- output: `codeswitch_ckpt_v3`
- W&B: `https://wandb.ai/tripathiji1312-vit/codeswitch-tts/runs/edmjhmnm`

Command shape:

```bash
python train_draft_distill.py --skip-teacher --skip-draft \
  --draft-ckpt codeswitch_ckpt_v2 \
  --residual-epochs 150 --residual-ramp-epochs 100 --residual-sched 1.0 \
  --residual-d-model 512 --residual-layers 6 \
  --min-epochs 150 --max-minutes-residual 90 \
  --ckpt-dir codeswitch_ckpt_v3
```

Result:

- Run time: **31m14s**
- Selected state: `best_epoch=139`
- `best_sched_p=1.0`
- `selected_full_draft=true`
- `best_epoch_any=149`
- `best_sched_p_any=1.0`
- `cb_top1_teacher_mean=0.0202847` = 2.02847 %
- `cb_top1_draft_mean=0.0155954` = 1.55954 %

This was a valid residual run: it reached the full-draft schedule and selected
the full-draft state. It still failed the end-to-end gate.

### 5.3 v3 diagnostic: teacher path passes, draft path fails

The unchanged 4-shard diagnostic with real DAC produced:

| Metric | v2 | v3 | Gate | Result |
|---|---:|---:|---:|---|
| `cb0_s` | — | 2.364 % | — | — |
| `cb0_a` | — | 9.942 % | — | — |
| `rt_s` | 3.195 % | 1.773 % | — | — |
| `rt_a` | — | **5.594 %** | `>= 5 %` | **PASS** |
| `rd_s` | 1.640 % | 0.094 % | — | — |
| `rd_a` | — | **1.834 %** | `>= 5 %` | **FAIL** |

Interpretation:

- The residual can use teacher CB0 well enough to clear `rt_a`.
- The same residual collapses when it receives draft CB0.
- `cb0_a = 9.942 %` is misleading by itself: a near-constant token can score
  around the teacher prior while producing a degenerate acoustic sequence.
- `rd_a = 1.834 %` is the decisive failure.

### 5.4 Audio isolation: the draft argmax emits token 568 constantly

`diagnose_audio_bug.py` was extended to write hybrid paths:

- `student_cb0_argmax_teacher_res`
- `student_cb0_argmax_res_sample`
- `student_cb0_sample_res_argmax`
- `student_cb0_only_argmax`

On the 4 diagnostic shards:

| Stat | Draft sample | Draft argmax |
|---|---:|---:|
| Unique tokens | 152.25 | 1.50 |
| Top-token fraction | 8.7 % | 99.9 % |
| Longest run | 2.5 frames | 382.75 frames |
| CB0 entropy | 4.51 nats | 0.01 nats |

The teacher CB0 prior over the cached data is:

- Most common token: **568**
- Frequency: **8.09 %** over **712,756** frames

The argmax path emits token `568` for essentially the whole utterance. The
sample path is diverse but not accurate. This explains the audio:

- `student_cb0_argmax_teacher_res` is muffled: constant coarse code plus
  teacher fine codes.
- `student_full_argmax` is near-silent: constant coarse code plus a residual
  trained on the same constant code.
- `student_full_sample` and `teacher_cb0_student_res_argmax` are normal.
- The v3 `student_full_argmax` WAV metrics were approximately
  `rms ≈ 0.0098`, `flatness ≈ 0.622`, `zcr ≈ 0.199`.

Critical detail: `precompute_draft_cb0` uses `sample=False`. Therefore the v3
residual was trained on the constant argmax token `568`, not on a varied draft
sequence. Another residual-only run is not justified until the draft argmax is
repaired.

### 5.5 Prior-subtraction sweep (rejected)

Hypothesis: subtract `alpha * log p_teacher(token)` from the draft logits to
suppress the prior and diversify the argmax without retraining.

Output: `/kaggle/working/codeSwitch/audio_diag_v3_prior/` and
`prior_sweep.json`.

| Alpha | CB0 top-1 | `rd_a` | Argmax unique | Top fraction | Longest run |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 10.019 % | 1.834 % | 1.5 | 99.9 % | 383 |
| 0.25 | 11.240 % | 1.827 % | 1.5 | 99.9 % | 383 |
| 0.50 | 11.433 % | 1.797 % | 1.5 | 99.9 % | 383 |
| 0.75 | 7.927 % | 1.398 % | 2.0 | 99.5 % | 300 |
| 1.00 | 0.663 % | 0.225 % | 20.0 | 25.6 % | 60 |

Conclusion: **rejected**. Only `alpha = 1.0` diversifies CB0, and it does so by
falling to chance accuracy. The listener confirmed every prior-adjusted WAV was
muffled/weird. Do not use this as the fix or rerun the sweep.

### 5.6 Per-sample frame lengths (v4, necessary but not sufficient)

The old draft trainer bucketed similar lengths but still interpolated every
sample to the batch maximum. Short sentences were therefore trained on a
stretched text-to-frame alignment.

What changed:

- `TemporalLengthProjector` accepts a per-sample `frame_lengths` tensor.
- Each sample is interpolated to its own DAC frame count.
- Padded tails are masked after each convolution and norm, so a batch's valid
  tail matches a one-sample forward.
- `DraftTransformer.forward(..., frame_lengths=...)` pads output to the batch
  maximum.
- `DraftTrainer._collate` returns per-sample lengths; `_run_epoch` and
  `evaluate_topk` pass them.
- `sequence_diversity_stats` computes unique tokens, top fraction, longest run,
  and entropy.
- Meta keys added: `val_argmax_unique`, `val_argmax_top_frac`,
  `val_argmax_longest_run`, `val_argmax_entropy_nats`.

Commit: `2cfaf4b` (`fix: train draft on per-sample frame lengths`)  
Tests at that point: **137 passed**  
Worktree: clean and pushed.

### 5.7 v4 draft-only run (triage failed)

Command shape:

```bash
python train_draft_distill.py \
  --corpus-json codeswitch_corpus.txt \
  --num-sentences 2000 \
  --skip-teacher --skip-residual \
  --draft-epochs 300 --draft-batch 24 \
  --min-epochs 300 --max-minutes 60 \
  --teacher-cache teacher_cache \
  --ckpt-dir codeswitch_ckpt_v4_draft
```

It used the existing 2 000-shard cache and did **not** rerun the teacher pass.

Result:

- Run time: **23m54s**
- Epochs: 300 (early stopping fired at epoch 300 with patience 12)
- Best validation loss: **5.0542 at epoch 12**
- Final log:

```text
[Draft] done | best val 5.0542 @ epoch 12 | CB0 top-1 8.26% |
top-5 23.07% | argmax unique 2.1, top 98.8%, run 332, H 0.05
```

Triage:

| Metric | v2 | v4 | Threshold | Result |
|---|---:|---:|---:|---|
| `val_top1` | 8.2483 % | 8.26 % | `>= 10 %` | **FAIL** |
| `val_top5` | — | 23.07 % | — | — |
| `val_argmax_unique` | — | 2.1 | `>= 16` | **FAIL** |
| `val_argmax_top_frac` | — | 98.8 % | `<= 40 %` | **FAIL** |
| `val_argmax_longest_run` | — | 332 | `<= 100` | **FAIL** |
| `val_argmax_entropy_nats` | — | 0.05 | — | — |

Conclusion: per-sample frame lengths removed a real bug, but the model still
learned the marginal teacher prior rather than text conditioning. The v4 draft
is not usable for residual training.

**Do not use `codeswitch_ckpt_v4_draft` for a residual run.**

Cosmetic caveat: the v4 draft-only final print lists a
`residual_cb1_8.pt` path even though `--skip-residual` was used. No residual file
was written; treat the run as draft-only.

### 5.8 v5 cross-attention frame decoder (implemented, not yet trained)

Hypothesis: the legacy projector's interpolation/convolution cannot choose the
right text position for every acoustic frame, so the model falls back to the
marginal prior. A frame decoder gives every frame its own query and lets it
cross-attend to the text encoder.

Implemented in commit `086c9ea`:

- `DraftModelConfig.frame_decoder_layers: int = 0`
- `DraftModelConfig.max_frames: int = 2048`
- New `FrameDecoder`:
  - learned frame queries `(1, max_frames, d_model)`;
  - sinusoidal frame position encoding;
  - whole-utterance length embedding;
  - `nn.TransformerDecoderLayer` with GELU, batch-first, pre-norm;
  - frame self-attention;
  - cross-attention to the text encoder memory;
  - `tgt_key_padding_mask` for per-sample frame counts;
  - `memory_key_padding_mask` for text padding.
- `DraftTransformer.forward` routes to `FrameDecoder` when
  `frame_decoder_layers > 0`; otherwise it keeps the legacy
  `TemporalLengthProjector` path byte-for-byte.
- CLI: `--draft-frame-decoder-layers` (0–6; default 0).
- The v5 Kaggle run uses `--draft-frame-decoder-layers 2`.

Parameter counts with the default 512-wide model:

| Path | Parameters |
|---|---:|
| Legacy projector | 16,807,936 |
| 2-layer frame decoder | 27,314,688 |

The new path is larger and will be slower; v5 latency has **not** been measured
yet. Existing latency claims are not automatically valid for v5.

Tests at this point: **140 passed, 3 warnings** with
`movio/.venv/bin/python -m pytest tests/ -q -p no:cacheprovider`.

---

## 6. What changed in the repository for v5

### 6.1 `codeswitch_tts/config.py`

Added to `DraftModelConfig`:

```python
frame_decoder_layers: int = 0
max_frames: int = 2048
```

`0` is the legacy behavior. Old checkpoint config dictionaries do not contain
these keys; dataclass defaults fill them in.

### 6.2 `codeswitch_tts/models/draft_transformer.py`

- Added `FrameDecoder`.
- `DraftTransformer.__init__` creates `self.frame_decoder = None` by default and
  only instantiates `FrameDecoder` when `frame_decoder_layers > 0`.
- `DraftTransformer.forward`:
  - builds `lengths` from `frame_lengths` or `num_frames`;
  - calls the frame decoder with text memory and the text padding mask;
  - otherwise calls the unchanged legacy `TemporalLengthProjector`.
- The `DraftOutput` contract is unchanged: logits, probabilities, tokens.
- `generate_chunk` is unchanged; it passes `num_frames` through the same path.

### 6.3 `train_draft_distill.py`

- Added `--draft-frame-decoder-layers` parser option.
- Validates `0 <= layers <= 6`.
- Sets `config.draft.frame_decoder_layers` before model construction.
- The existing `--skip-draft --draft-ckpt` path loads the config from the v5
  checkpoint, so the residual sees the matching decoder architecture.

### 6.4 Tests

`tests/test_codeswitch_tts.py`:

- `test_frame_decoder_forward_and_per_sample_masking`
  - cross-attention output shape;
  - deterministic argmax;
  - batched `[4, 7]` first sample matches a single `num_frames=4` forward.
- `test_legacy_draft_checkpoint_has_no_frame_decoder_keys`
  - default config has no `frame_decoder.*` state keys;
  - strict state-dict reload succeeds.

`tests/test_corpus_builder.py`:

- `test_draft_frame_decoder_layers_flag_is_wired`
  - parser default and plumbing into `config.draft.frame_decoder_layers`.

Existing regression tests retained:

- `test_draft_forward_uses_per_sample_frame_lengths`
- `test_draft_collate_carries_per_sample_frame_lengths`
- `test_sequence_diversity_stats_flags_constant_and_varied_argmax`
- `test_length_bucketed_order_keeps_lengths_together`

### 6.5 Kaggle notebook

`CodeSwitch_013_RunAll_Kaggle.ipynb` is now the **v5** run sheet:

- v5 is the primary path.
- v3 cells are off; do not rerun the v3 residual or prior sweep.
- v4 stage cells were repurposed to v5; v4 results remain in the docs and the
  `codeswitch_ckpt_v4_draft` directory is untouched.
- v5 draft cell writes `codeswitch_ckpt_v5_draft`.
- v5 residual cell writes `codeswitch_ckpt_v5`.
- v5 post-retrain diagnostic writes `audio_diag_v5`.
- v5 smoke WAV is `output_audio/test_v5_argmax.wav`.
- `DO_V5_DRAFT_REPAIR=True`
- `DO_V5_RESIDUAL=True` (but the cell re-checks the draft gate)
- `DO_V5_PUSH=True`
- `DO_V5_POST_DIAG=True`
- `DO_V5_SMOKE=False`
- `DO_ARGMAN_EVAL=False`, `DO_SAMPLE_EVAL=False`, `DO_GATE=False`,
  `DO_BUNDLE=False`
- `V5_DRAFT_GATE=True`
- `OVERWRITE_V5_DRAFT=False`, `OVERWRITE_V5_RESIDUAL=False`

The sync cell uses `git checkout -B REPO_REF FETCH_HEAD`, so the notebook picks
up the pushed branch commit.

### 6.6 Documentation

- `plans/README.md`: v4 result and v5 amendment.
- `plans/013-bc-noise-fix.md`: appended amendment with the v5 design and gates.
- `KAGGLE_RUN.md`: **untouched**.

---

## 7. Verification performed locally

### 7.1 Passing checks

- Full test suite:

```text
movio/.venv/bin/python -m pytest tests/ -q -p no:cacheprovider
140 passed, 3 warnings
```

- `git diff --check`: clean.
- Notebook JSON parses; all 23 code cells compile.
- Legacy config check: an old config dict without the new fields instantiates
  with `frame_decoder_layers=0`; legacy state dict has no `frame_decoder.*`
  keys.
- Cross-attn check: `(2, 7, 1024)` output, deterministic argmax, per-sample
  masking matches a single-sample forward.
- Tiny CPU overfit sanity check: a 2-text/16-token synthetic model reached
  `acc=1.000` by optimizer step 50. This only proves gradients/masks work; it
  is not evidence about real speech.
- Tiny `DraftTrainer` run with fake shards and a frame decoder produced the
  expected meta keys (`val_top1`, `val_argmax_*`, `best_epoch`, etc.).

### 7.2 Not completed / not evidence

- A full CPU `--dry-run` was attempted with a non-default checkpoint directory,
  but it exceeded the 180 s tool timeout before writing checkpoints. It is not
  used as evidence.
- No local GPU/Kaggle training was run; the user controls the Kaggle instance.
- The independent `verifier` subagent could not run because its free-usage
  limit was reached. Review was completed locally instead.
- No v5 diagnostic, smoke, or full evaluation has been run.

---

## 8. What we are going to do next

### 8.1 Step 1 — sync the Kaggle repo

In the notebook:

1. Run the configuration cell (defines `REPO_DIR`, `REPO_REF`, `CKPT_V5_DRAFT`,
   `V5_DRAFT_FRAME_DECODER_LAYERS`, etc.).
2. Run the sync cell.
3. Run the preflight cell.
4. Confirm the commit:

```bash
%%bash
cd /kaggle/working/codeSwitch
git log -1 --oneline
# expect: 086c9ea feat(draft): add opt-in text-conditioned frame decoder
```

The existing 2 000-shard `teacher_cache` and the v2/v3/v4 checkpoint
directories must remain untouched.

### 8.2 Step 2 — v5 draft-only run

Run **only** the notebook cell:

**“10c. Draft repair (v5)”**

Manual equivalent:

```bash
%%bash
cd /kaggle/working/codeSwitch
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

If `codeswitch_corpus.txt` is missing, omit `--corpus-json`; with
`--skip-teacher`, the cached shards are the training data and the corpus file
is only used for the disjointness check.

Expected behavior:

- 2 000 cached shards loaded.
- 1 800 train / 200 validation (every tenth shard is validation).
- Draft only; no teacher pass, no residual.
- Output directory: `codeswitch_ckpt_v5_draft/`.
- The run may take longer than v4 because the 2-layer decoder adds
  frame self-attention. The cap is 60 minutes.
- If the run reaches the cap, the best validation state is still saved and can
  be triaged.

Do not use `codeswitch_ckpt_v4_draft` as a fallback. Do not overwrite an
existing `codeswitch_ckpt_v5_draft` unless intentional
(`OVERWRITE_V5_DRAFT=True`).

### 8.3 Step 3 — inspect the v5 draft meta

Read:

```text
codeswitch_ckpt_v5_draft/meta.json
```

Focus on the `draft` block:

- `best_epoch`
- `best_val_loss`
- `val_top1`
- `val_top5`
- `val_argmax_unique`
- `val_argmax_top_frac`
- `val_argmax_longest_run`
- `val_argmax_entropy_nats`

Decision:

```text
val_top1 >= 10 %
AND val_argmax_unique >= 16
AND val_argmax_top_frac <= 40 %
AND val_argmax_longest_run <= 100
```

If **any** condition fails:

- stop;
- do not train the residual;
- send the draft block and the training log;
- next lever is class-balanced/focal loss or multi-sample teacher targets, not
  another residual-only run.

If all pass:

- continue to v5 residual training.

### 8.4 Step 4 — v5 residual-only run (only after triage pass)

Notebook cell: **“10d. Residual-only v5”**.

Manual equivalent:

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

The cell must verify:

- `residual_only=true`
- `residual.epochs_run >= 150`
- `residual.final_sched_p >= 0.95`
- `residual.selected_full_draft=true`
- `residual.best_sched_p=1.0`
- copied `draft_cb0.pt` is byte-identical to the v5 draft source
- `residual_cb1_8.pt` exists
- `cb_top1_teacher_mean` and `cb_top1_draft_mean` are recorded

### 8.5 Step 5 — unchanged 4-shard diagnostic

Notebook cell: **“10f. v5 post-retrain diagnostic + code gate”**.

Manual equivalent:

```bash
python diagnose_audio_bug.py \
  --ckpt codeswitch_ckpt_v5 \
  --teacher-cache teacher_cache \
  --out audio_diag_v5 \
  --shards 4 \
  --audio-frames 172
```

Gate:

- `dac_backend in {"hf_pretrained", "dac_official"}`
- `rt_a >= 5 %`
- `rd_a >= 5 %`

If either code metric fails:

- no B/C quality claim;
- do not run smoke or the 60-utterance suite;
- inspect `diagnostic_report.json` and the isolation WAVs.

### 8.6 Step 6 — listen, then smoke

If and only if both code metrics pass:

1. Open notebook cell **“10g. Listen to the v5 isolation WAVs”**.
2. Start with `student_full_argmax`.
3. It must be intelligible.
4. Then set `DO_V5_SMOKE=True` and run the one-sentence smoke
   (`ta-01`, Config C, argmax, output
   `output_audio/test_v5_argmax.wav`).
5. Listen to the smoke WAV.

If the smoke WAV is not intelligible, stop. Do not run the 60-utterance suite.

### 8.7 Step 7 — full evaluation

Only after the code gate passes and the smoke WAV is intelligible:

- `DO_ARGMAN_EVAL=True`: 60 utterances × A/B/C, argmax.
- `DO_SAMPLE_EVAL=True`: matched B/C sample baseline.
- `DO_GATE=True`: CER/UTMOS/ASR/latency gate.
- `DO_BUNDLE=True`: zip and W&B bundle.

The existing full-evaluation gate includes:

- B CER p50 `<=` A CER p50 + 10 percentage points;
- C CER p50 `<=` B CER p50 + 10 percentage points;
- B UTMOS `>= 0.75 * A UTMOS`;
- no ASR loops;
- B/C TTFA and RTF within ±25 % of the matched sample run.

Do not loosen these thresholds.

---

## 9. Decision tree

```text
v5 draft-only
  |
  +-- triage FAIL
  |     -> stop; no residual
  |     -> next: class-balanced/focal loss OR multi-sample teacher targets
  |
  +-- triage PASS
        -> v5 residual-only
             |
             +-- meta checks FAIL
             |     -> fix ramp/selection/capacity; no claim
             |
             +-- meta checks PASS
                   -> 4-shard diagnostic
                        |
                        +-- code gate FAIL
                        |     -> no claim; inspect draft CB0 / residual
                        |
                        +-- code gate PASS
                              -> listen to student_full_argmax
                                   |
                                   +-- not intelligible
                                   |     -> stop; diagnose before smoke
                                   |
                                   +-- intelligible
                                         -> one-sentence smoke
                                              |
                                              +-- fail/not intelligible
                                              |     -> stop
                                              |
                                              +-- pass
                                                    -> 60-utterance argmax + sample
                                                    -> CER/UTMOS/ASR/latency gate
                                                    -> bundle
```

---

## 10. Why each previous attempt was rejected

| Attempt | What it did | Result | Decision |
|---|---|---|---|
| v3 residual-only | 150 epochs, 512/6 residual, 100 % draft-CB0 ramp | `rt_a=5.594 %` pass; `rd_a=1.834 %` fail | Rejected as incomplete; draft is bottleneck |
| Prior subtraction | `alpha * log p_teacher` logit correction | Only alpha=1.0 diversified; accuracy fell to chance; WAVs bad | Rejected |
| Per-sample lengths | Each sample projected to own frame count | v4 top1 8.26 %, unique 2.1 | Necessary but insufficient |
| Another residual-only | — | Would train on constant token 568 again | Rejected |
| Lower gate | — | Would hide the defect | Forbidden |
| Cross-attention | Frame queries + text memory | Not yet trained | Current v5 experiment |

---

## 11. Artifacts and paths

### 11.1 Kaggle paths (existing instance, do not overwrite)

| Path | Status | Use |
|---|---|---|
| `/kaggle/working/codeSwitch/teacher_cache/` | 2 000 shards | reuse with `--skip-teacher` |
| `codeswitch_ckpt_v2/` | exists | v2 draft source |
| `codeswitch_ckpt_v3/` | exists | failed residual; provenance only |
| `codeswitch_ckpt_v4_draft/` | exists | failed draft triage; do not residual |
| `audio_diag_v3_prior/` | exists | rejected prior sweep |
| `codeswitch_ckpt_v5_draft/` | to create | v5 draft-only output |
| `codeswitch_ckpt_v5/` | to create | v5 residual output |
| `audio_diag_v5/` | to create | v5 diagnostic |
| `output_audio/test_v5_argmax.wav` | to create | smoke listening check |
| `/kaggle/working/codeswitch_013_artifacts.zip` | eventual | bundle |

### 11.2 Repository files

| File | Role |
|---|---|
| `codeswitch_tts/config.py` | `DraftModelConfig` |
| `codeswitch_tts/models/draft_transformer.py` | draft architecture, legacy projector, `FrameDecoder` |
| `train_draft_distill.py` | teacher cache, draft/residual training, meta |
| `diagnose_audio_bug.py` | unchanged 4-shard code gate + isolation WAVs |
| `smoke_test_argmax.py` | one-sentence listening check |
| `tune_draft_prior.py` | rejected prior sweep; do not rerun |
| `CodeSwitch_013_RunAll_Kaggle.ipynb` | v5 run sheet |
| `plans/013-bc-noise-fix.md` | original plan + v5 amendment |
| `plans/README.md` | running log |
| `plans/013-bc-noise-status.md` | this detailed snapshot |

---

## 12. Risks and open questions

### 12.1 Open technical questions

1. **Will cross-attention break the prior collapse?** Unknown until v5 draft
   training completes. Frame-level cross-attention is the right architectural
   lever, but the teacher's CB0 distribution may still be dominated by a small
   set of common tokens.
2. **If v5 still fails, which fallback first?** Class-balanced/focal loss is
   cheap and targets the prior dominance. Multi-sample teacher targets require
   additional teacher generation and are more expensive.
3. **If the draft passes but the residual fails, what next?** Residual capacity
   or soft targets; do not return to a residual-only run on a collapsed draft.
4. **Will the 5 % gate be achievable?** The gate is fixed; the result may be
   negative. A negative result is acceptable; a false B/C quality claim is not.
5. **Latency:** the 2-layer cross-attention decoder is larger. TTFA/RTF must be
   measured in the matched full evaluation. Do not reuse v4/v3 latency numbers
   for v5.

### 12.2 Known risks

- Cross-attention may overfit the 1 800 training sentences before learning
  alignment.
- Frame self-attention is O(T²) in memory; very long utterances could cause a
  T4 OOM. If that happens, reduce `V5_DRAFT_BATCH` to 16 and keep the 60-minute
  cap; do not change the gate.
- The length model used at inference is approximate; training uses exact
  teacher frame counts. This mismatch is unchanged from earlier drafts.
- The v4 final print is cosmetically misleading about the residual file; treat
  v4 as draft-only.
- The independent verifier was unavailable; local tests and manual checks are
  the evidence.
- No v5 result exists yet. Do not write v5 numbers into the paper or claim
  success.

### 12.3 Stop rules

Stop and report, do not improvise, if:

- the draft triage fails;
- the residual meta checks fail;
- the code gate fails;
- the argmax WAV is unintelligible;
- the one-sentence smoke fails;
- the full evaluation gate fails;
- a command would overwrite v2/v3/v4/teacher cache.

---

## 13. Exact commands cheat sheet

### 13.1 Sync and verify

```bash
%%bash
cd /kaggle/working/codeSwitch
git fetch origin fix/013-bc-noise
git checkout -B fix/013-bc-noise FETCH_HEAD
git log -1 --oneline
# expect 086c9ea
find teacher_cache -maxdepth 1 -name 'shard_*.npz' | wc -l
# expect 2000
ls -d codeswitch_ckpt_v2 codeswitch_ckpt_v3 codeswitch_ckpt_v4_draft 2>/dev/null
```

### 13.2 v5 draft-only

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

### 13.3 Inspect draft meta

```bash
python - <<'PY'
import json
from pathlib import Path
d = json.loads(Path("codeswitch_ckpt_v5_draft/meta.json").read_text())["draft"]
for k in ("best_epoch", "best_val_loss", "val_top1", "val_top5",
          "val_argmax_unique", "val_argmax_top_frac",
          "val_argmax_longest_run", "val_argmax_entropy_nats"):
    print(f"{k:24s} {d.get(k)}")
PY
```

### 13.4 v5 residual-only

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

### 13.5 v5 diagnostic

```bash
python diagnose_audio_bug.py \
  --ckpt codeswitch_ckpt_v5 \
  --teacher-cache teacher_cache \
  --out audio_diag_v5 \
  --shards 4 \
  --audio-frames 172
```

### 13.6 Tests

```bash
movio/.venv/bin/python -m pytest tests/ -q -p no:cacheprovider
```

---

## 14. What to send back after the v5 draft run

At minimum:

1. `codeswitch_ckpt_v5_draft/meta.json` — especially the `draft` block.
2. The final training log lines:
   - best val / epoch;
   - `CB0 top-1`;
   - `top-5`;
   - `argmax unique`;
   - `top`;
   - `run`;
   - `H`;
   - total minutes.
3. Whether the run hit the 60-minute cap or early-stopped.
4. Any Python traceback or CUDA OOM.

If the triage fails, that is enough; do not start the residual. If it passes,
send the same meta before starting the residual so the gate can be checked
independently.

---

## 15. Decision log

| Date | Decision | Reason |
|---|---|---|
| 2026-10-21 | Keep 5 % `rt_a`/`rd_a` gate | Paper-quality contract; lowering hides the defect |
| 2026-10-21 | Reject prior subtraction | Only alpha=1.0 diversifies; accuracy falls to chance; audio bad |
| 2026-10-21 | Do not run another v3-style residual | Residual would train on constant draft CB0 token 568 |
| 2026-10-21 | Implement per-sample frame lengths | Removes batch-max text stretching; necessary fix |
| 2026-10-22 | Do not residual from v4 | v4 triage failed on all four criteria |
| 2026-10-22 | Implement opt-in frame decoder | Text-conditioned frame queries + cross-attention is the next architectural lever |
| 2026-10-22 | Keep legacy path default | Old v2/v3/v4 checkpoints must load unchanged |
| 2026-10-22 | Keep smoke/full eval off | Argmax path is not yet confirmed intelligible |
| 2026-10-22 | v5 draft-only next | Isolate draft repair before spending residual time |

---

## 16. Appendix — exact v4 log excerpt

```text
[Draft] epoch 300 | train 4.8547 | val 5.1337 | lr 1.83e-08 | 4s/ep | total 23m49s
[Draft] early stopping at epoch 300 (patience 12)
[Draft] done | best val 5.0542 @ epoch 12 | CB0 top-1 8.26% | top-5 23.07% |
argmax unique 2.1, top 98.8%, run 332, H 0.05
[Save] codeswitch_ckpt_v4_draft/draft_cb0.pt (16.81M params)
==============================================================================
[Done] total 23m54s
  checkpoints : codeswitch_ckpt_v4_draft/draft_cb0.pt, codeswitch_ckpt_v4_draft/residual_cb1_8.pt
  meta        : codeswitch_ckpt_v4_draft/meta.json
```

Note: the residual path in the final print is cosmetic; the run used
`--skip-residual`.

---

## 17. Appendix — v3 diagnostic values

| Metric | Value |
|---|---:|
| `cb0_s` | 2.364 % |
| `cb0_a` | 9.942 % |
| `rt_s` | 1.773 % |
| `rt_a` | 5.594 % |
| `rd_s` | 0.094 % |
| `rd_a` | 1.834 % |

`rt_a` passes; `rd_a` fails. No B/C quality claim is allowed.

---

## 18. Appendix — current code/test status

- Branch: `fix/013-bc-noise`
- Code commit: `086c9ea`
- Previous commit: `2cfaf4b`
- Worktree at code snapshot: clean
- Tests: `140 passed, 3 warnings`
- Notebook: 48 cells, 23 code cells, all compile
- No new dependencies
- `KAGGLE_RUN.md`: untouched
- Teacher cache: 2 000 shards (existing instance)
- v5 draft: not yet trained
- v5 residual: not yet trained
- v5 diagnostic: not yet run
- B/C quality: not claimed
