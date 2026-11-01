# CodeSwitch-TTS v6 Real-Data Run — Consolidated Context

| Field | Value |
|---|---|
| Last updated | 2026-10-31 |
| Repo HEAD | `e9c6234` (v7 text/confidence residual, diagnostics, notebook) |
| Status | **BLOCKED** — unchanged 5% gate fails on `rd_a` |
| Scope | Primary: real Rasa Tamil v6. Appendix A: pre-v6 synthetic-cache lineage. Appendix B: plans 001–012 |
| Quality claim | **None** — 60-utterance evaluation not run |

**Bottom line:** the unchanged 5% gate fails. The 60-utterance evaluation has not run. There is no valid B/C quality claim.

---

## 1. One-page summary

- Real Rasa Tamil speech was prepared, DAC-encoded, and used to train the CTC-duration-conditioned v6 draft and residual.
- Best full run: **6.0001 h**, 4,015 shards, draft `val_top1 = 5.8416%`, residual selected at epoch 149.
- Gate result: `rt_a = 8.8726%` **passes**, `rd_a = 2.2487%` **fails**.
- The LR-floor rerun improved `rt_a` to **10.6244%** but left `rd_a` at **2.2340%**. The gate still fails.
- The binding constraint is draft CB0 accuracy: `p = 7.9383%`. The residual reaches `c = 10.6244%` when CB0 is correct, but only `w = 1.5105%` when CB0 is wrong.
- At current `c` and `w`, the draft would need `p ≈ 38.3%` to reach `rd_a = 5%`. The 6 h numbers imply `p ≈ 46.2%`.
- Rejected levers: LR floor, duration-loss weight 0.5, WIKI-only data, 20 h + residual projections, token corruption/soft top-k.
- Remaining paths: a fundamentally stronger draft/residual, an explicit gate re-scope with user sign-off, or one cheap draft-capacity diagnostic.
- v7 intervention (code complete, not yet trained): text/confidence-conditioned residual to raise `w`, with a standalone Kaggle notebook. See Appendix C.

---

## 2. Objective and non-negotiables

**Objective:** make real Rasa/IndicVoices-R speech drive DAC CB0–8 shards and a CTC-duration-conditioned v6 draft/residual, then pass the unchanged 5% gate with a valid 60-utterance evaluation.

**Non-negotiables:**

- Gate remains `rt_a >= 5%` **and** `rd_a >= 5%`.
- Do not lower, reinterpret, or hide a failure.
- Do not claim B/C quality until the gate passes and the 60-utterance evaluation is valid.
- Do not overwrite `codeswitch_ckpt_v2`, `codeswitch_ckpt_v3`, `codeswitch_ckpt_v4_draft`, or `teacher_cache`.
- Preserve each checkpoint before a new run; use a new `CKPT_DIR` per experiment.
- Preserve backward-compatible teacher-cache/shard loading.
- No Kaggle/GPU jobs locally; deliver code, tests, docs, and notebook.
- Read `HF_TOKEN`, `WANDB_API_KEY`, and `GITHUB_TOKEN` only from Kaggle Secrets/environment; never log or hardcode them.
- One variable at a time; do not bundle scheduler, selection, and architecture changes.
- Only run the 60-utterance eval after both gate halves pass.

---

## 3. Pipeline and metric meanings

1. **Data prep:** Rasa Tamil rows → resampled audio → DAC encode → one `SpeechShard` per utterance with text, CB0–8 codes, duration, and metadata.
2. **Draft:** text + duration → CB0 token sequence.
3. **Residual:** draft CB0 → CB1–8 tokens.
4. **Gate:** `diagnose_audio_bug.py`, 4 shards sampled by `np.linspace`, unchanged thresholds.
5. **60-utterance eval:** `run_full_evaluation.py --configs B,C --sample-mode argmax`; blocked unless the gate passes.

Metric definitions:

- `cb0_s` / `cb0_a`: draft CB0 top-1 under sampled/all evaluation.
- `rt_s` / `rt_a`: residual top-1 with **teacher CB0**.
- `rd_s` / `rd_a`: residual top-1 with **draft CB0**.
- Gate pass condition uses the `_a` values: `rt_a >= 5` and `rd_a >= 5`.
- `_s` values are diagnostics only.
- `rt_a` is effectively the residual ceiling when CB0 is correct: `c`.
- `rd_a` is the actual deployed gate metric.

---

## 4. Exact run ledger

### 4.1 FLEURS Tamil smoke (plumbing only)

- 10 shards.
- 10/10 duration-covered.
- v6 draft trained.
- W&B synced.
- This is a plumbing smoke, **not** a quality result.

### 4.2 200-row Rasa pilot — FAIL

| Item | Value |
|---|---:|
| Rows seen / written | 201 / 200 |
| Shards | 200 |
| Audio | **0.2927674659863946 h** |
| Duration shards | 200 / 200 |
| Mean duration coverage | 0.9728298477194176 |
| Draft best val | 5.2843 @ epoch 17 |
| Draft `val_top1` | 6.9244% |
| Draft `val_top5` | 20.2883% |
| Residual selected | 6.9280 @ epoch 100 |
| Residual schedule | 1.00 |
| `selected_full_draft` | true |
| Gate `cb0_s` | 3.2945736434% |
| Gate `cb0_a` | 11.9186046512% |
| Gate `rt_s` | 1.0023013566% |
| Gate `rt_a` | 2.6132509690% |
| Gate `rd_s` | 0.1665455426% |
| Gate `rd_a` | **1.7593265504% — FAIL** |
| W&B | data `gd4ea2go`, train `ucsyx8e0`, gate `qg3pnhk5` |

The pilot is **17.6 minutes**, not 6 hours. The row cap, not `--max-hours`, was the binding limit. It is a plumbing/data-scale result, not an architecture verdict.

### 4.3 6 h mixed Rasa run — FAIL

**Data prep**

| Item | Value |
|---|---:|
| Rows seen / written | 4,024 / 4,015 |
| Shards | 4,015 |
| Audio | **6.000136616906021 h** |
| Mean duration coverage | 0.9724745706084863 |
| Errors | 0 |
| Prep elapsed | 917.825023938 s |
| W&B | `ck8i90t6` |

**Draft**

| Item | Value |
|---|---:|
| Epochs | 150 |
| Best val | 4.8525 @ epoch 20 |
| `val_top1` | **5.8416%** |
| `val_top5` | **17.9546%** |
| Oracle-duration top1 | 9.8324% |
| Oracle-duration top5 | 27.3165% |
| Train time | 2,433.75 s |
| W&B | `92gb8bnm` |

**Residual**

| Item | Value |
|---|---:|
| Windows | 23,564 total |
| Train / val windows | 21,288 / 2,276 |
| Epochs | 150 |
| Selected val | 6.7345 @ epoch 149 |
| Schedule | 1.00 |
| Full-draft hold | true |
| Train time | 4,994.77 s |
| Total training | 2h 04m 25s |

**Gate**

| Metric | Value |
|---|---:|
| `cb0_s` | 3.5834078712% |
| `cb0_a` | **7.9382826476%** |
| `rt_s` | 2.8343023256% |
| `rt_a` | **8.8725682021% — PASS** |
| `rd_s` | 0.2124329159% |
| `rd_a` | **2.2487142218% — FAIL** |
| W&B | `c3w0vxxy` |

Preserved as `codeswitch_ckpt_v6_6h`, `eval_results_v6_6h_gate`, and `meta_6h.json`.

### 4.4 LR-floor residual rerun — FAIL

Only change: residual scheduler `min_lr = 1e-5`.

| Item | Value |
|---|---:|
| Residual epochs | 150 |
| Train time | 5,087.23 s |
| Total | 1h 25m 25s |
| Selected val | 6.7347 @ epoch 100 |
| Best val any | 6.73399 @ epoch 87 |
| Best teacher val | 6.480997 |
| `rt_s` | 3.6798412343% |
| `rt_a` | **10.6244409660% — PASS** |
| `rd_s` | 0.2033486136% |
| `rd_a` | **2.2340395796% — FAIL** |
| `cb0` | same as 6 h |

**Interpretation:** the LR floor improved residual performance under teacher CB0 (`rt_a 8.8726 → 10.6244`) but did **not** improve the deployed draft-CB0 gate (`rd_a 2.2487 → 2.2340`). The full-draft validation worsened after epoch 100 (`6.7347 → 6.7652`). The LR-floor hypothesis is rejected.

The patch exists only in `/kaggle/working/codeSwitch/train_draft_distill.py`; it is not in the local repo.

### 4.5 Duration-loss weight 0.5 — rejected

| Item | Value |
|---|---:|
| Best val | 4.8846 @ epoch 26 |
| `val_top1` | 5.69593052136881% |
| `val_top5` | 16.819320953545097% |
| `val_duration_loss` | 0.12004632646074662 |
| Oracle-duration top1 | 10.022716743970574% |
| Oracle-duration top5 | 27.655263642604777% |
| Train time | 11m 19s |
| Verdict | **rejected** |

Compared with the 6 h draft:

- `val_top1` fell **5.8416 → 5.6959%**.
- `val_top5` fell **17.9546 → 16.8193%**.
- The oracle gap widened **3.99 → 4.33 points**.

Duration loss went down, but token accuracy got worse. Do not train the residual on `dur05`.

### 4.6 WIKI-only style isolation — rejected at triage

**Prep**

- `real_cache/rasa_ta_wiki`
- 2,715 shards
- 6.001 h
- 100% duration coverage
- All Female
- 2,717 rows seen
- 1 long row, 1 frame filter, 0 errors

**Draft**

| Item | Value |
|---|---:|
| Best val | 4.8101 @ epoch 34 |
| `val_top1` | **4.570192940488253%** |
| `val_top5` | 14.357718141837897% |
| `val_duration_loss` | 0.1106291338801384 |
| Oracle-duration top1 | 10.122324478475697% |
| Oracle-duration top5 | 27.77580461511141% |
| Diversity: unique | 67.3 |
| Diversity: top token | 21.6% |
| Diversity: longest run | 19 |
| Diversity: entropy | 3.18 nats |
| Train time | 11m 30s |
| Residual / gate | **not run** |
| Verdict | **rejected** |

The WIKI model had **better val loss, better oracle accuracy, and better diversity** than the mixed 6 h model, but **lower top-1** (`4.5702% < 5.8416%`). Style isolation is not the lever. The gate rewards argmax CB0 accuracy, and the evidence is consistent with it rewarding a sharper/more mode-collapsed draft.

### 4.7 CB0 corruption sweep

- 401 validation shards.
- 2,276 windows.
- Draft CB0 per-token match: **5.842%** (10,466 / 179,163 frames).
- Recorded sweep means:

| Corruption level | Mean |
|---:|---:|
| 0.00 | 2.603% |
| 0.10 | 2.191% |
| 0.25 | 1.631% |
| 0.50 | 1.104% |
| 0.75 | 1.188% |
| 1.00 | 1.212% |

No corruption setting reaches 5%. Corrupting CB0 does not rescue the residual. Token corruption, soft CB0, and top-k CB0 were ruled out.

---

## 5. Gate arithmetic — why `rd_a` is stuck

The deployed gate decomposes as:

```text
rd_a = p * c + (1 - p) * w
```

where:

- `p = cb0_a` = draft CB0 top-1.
- `c = rt_a` = residual top-1 when CB0 is correct.
- `w` = residual top-1 when CB0 is wrong.

Using the LR-fix numbers:

```text
p = 7.9383%
c = 10.6244%
w = 1.5105%

rd_a = 0.079383 * 0.106244 + 0.920617 * 0.015105
     = 0.008434 + 0.013906
     = 0.022340 = 2.2340%
```

Using the 6 h numbers:

```text
p = 7.9383%
c = 8.8726%
w = 1.6771%

rd_a = 2.2487%
```

**Required draft accuracy at current `c` and `w`:**

```text
p_required = (5% - w) / (c - w)
           = (5 - 1.5105) / (10.6244 - 1.5105)
           = 38.3%
```

At the 6 h `c` and `w`, the requirement is **46.2%**.

**Projection at current `c = 10.6244%`, `w = 1.5105%`:**

| Draft CB0 `p` | Projected `rd_a` |
|---:|---:|
| 12% | 2.60% |
| 20% | 3.33% |
| 30% | 4.24% |
| 38.3% | 5.00% |

**Context:**

- Chance is `0.0977%` (1/1024).
- `rd_a = 2.234%` is about **22.9× chance**.
- 5% is about **51.2× chance**.
- The earlier top-5 oracle projection was **≈3.15%**, still below 5%.
- To pass by improving wrong-CB0 recovery alone, `w` would need to be **4.52%**, about **3×** the observed 1.51%.

**Conclusion:** the gate is dominated by draft CB0 accuracy. The residual cannot recover wrong CB0 frames. At the current residual ceiling, even a perfect CB0 would only produce ~10.6%, so the draft does not need to be perfect — but it needs to be roughly **5× better** than it is now (`7.94% → 38.3%`) unless the residual ceiling also improves.

---

## 6. Rejected levers and reasoning

| Lever | Result | Why rejected |
|---|---|---|
| **LR floor `1e-5`** | `rd_a 2.2487 → 2.2340` | Improved `c`, not `p`; full-draft val worsened after epoch 100; selection stayed at first full-draft epoch. |
| **Duration-loss weight 0.5** | top1 `5.8416 → 5.6959`, top5 `17.9546 → 16.8193` | Lower duration loss, worse token accuracy; oracle gap widened. Duration conditioning is not the bottleneck. |
| **WIKI-only data** | top1 `4.5702%` | Better loss/oracle/diversity but lower argmax top-1; style is not the lever; gate rewards mode collapse. |
| **20 h + residual** | projected `2.60 / 3.33 / 4.24%` at `p = 12 / 20 / 30%` | Still fails; unfiltered 20 h adds male/style variance and is not a clean data-scale test. |
| **Token corruption / soft / top-k** | top-5 oracle ceiling ≈3.15% | Cannot reach 5%; wrong-CB0 recovery would need `w = 4.52%`. |
| **Cross-cache gate comparison** | invalid | Different 4-shard samples: 200-shard cache picks `0,66,133,199`; 4,015-shard cache picks `0,1338,2676,4014`. |

The pilot’s `cb0_a = 11.9186%` is **not** comparable to the 6 h `cb0_a = 7.9383%` for this reason.

---

## 7. Data audit and why the 6 h cache is biased

**Rasa Tamil schema:**

- 33,005 examples.
- 19.99 GB.
- Features: `filename, text, language, gender, style, duration, wav_path, audio`.
- No `speaker_id`.
- Two speakers via `gender`:
  - Female: **29.94 h / 19,871 rows**.
  - Male: **26.18 h / 16,790 rows**.

**First 5,000 rows are all Female.** Style counts:

| Style | Count |
|---|---:|
| WIKI | 1,089 |
| CONV | 922 |
| ANGER | 476 |
| HAPPY | 414 |
| SURPRISE | 370 |
| SAD | 352 |
| BOOK | 309 |
| DISGUST | 295 |
| FEAR | 286 |
| NEWS | 212 |
| ALEXA | 194 |
| INDIC | 49 |
| UMANG | 26 |
| BB | 6 |

The 6 h cache is the first ~4,015 rows: **female-only, multi-style**, with WIKI ≈22% of the first 5,000 rows. It is not a random sample.

**Metadata gap:**

- The 6 h cache `speaker` field is empty for all 4,015 shards.
- `_SPEAKER_COLUMNS` lacks `gender`, so auto-detection misses it.
- The 6 h prep did not request `--speaker-column gender`.
- `style` is parsed but **not stored** in `SpeechShard`.
- `real_speech.py:568` records speaker but not style.
- Speaker/style conditioning is therefore not implemented and would require a schema change plus a re-prep.

WIKI-only was a clean style isolation test: 6.001 h, all Female, but it failed the 10% triage.

---

## 8. Issues and validity caveats

1. **Gate fails.** `rd_a = 2.2340% < 5%`; the 60-utterance eval remains blocked.
2. **Only 4 gate shards.** The gate samples 4 shards via `np.linspace`; cross-cache comparisons are confounded. No confidence interval or seed-variance study was run.
3. **Pilot artifacts overwritten.** The pilot weights were replaced at `codeswitch_ckpt_v6`; pilot meta/prep time is lost. Only metrics survive in `plans/014` and W&B `qg3pnhk5`.
4. **No training resume.** Each run starts from scratch. The LR patch exists only in the Kaggle working copy.
5. **Notebook still points at `codeswitch_ckpt_v6`.** It must be changed per experiment or it will overwrite the pilot again.
6. **Cache ordering bias.** The 6 h cache is the first rows of Rasa Tamil, female-only and style-skewed. It is not representative of the full 56 h corpus.
7. **Speaker/style metadata missing.** No clean conditioning experiment is possible until re-prep captures `gender` and `style`.
8. **6 h draft is below the 10% triage.** `5.8416% < 10%`; WIKI is also below (`4.5702%`). The current draft is not close to the gate.
9. **Cross-cache gate numbers are invalid for comparison.** Only same-cache comparisons are meaningful.
10. **The gate measures draft CB0, not audio quality.** WIKI’s better loss/diversity but lower top-1 is evidence of this mismatch.

---

## 9. Local repo state

- HEAD: `decefc38c6547fc985244639a133266bbb6d1de3` (`decefc3`).
- Commits `0bff727` and `decefc3` pushed.
- Worktree clean at the time of the last check.
- Local test suite: **172 passed, 3 warnings in 10.65s**.
- Notebook: **21 cells, 11 code cells**, all compile.
- Shell-command-in-Python-cell error fixed with:
  ```python
  subprocess.run([...], cwd="/kaggle/working/codeSwitch", check=True)
  ```
- `plans/014-real-data-duration.md` still needs the WIKI and `dur05` numbers added.
- The notebook’s `CKPT_DIR` is still `codeswitch_ckpt_v6`; this is a real overwrite risk.

**Preserved checkpoints:**

- `codeswitch_ckpt_v6_6h` — 6 h draft/residual/meta.
- `codeswitch_ckpt_v6_lrfix` — LR-floor checkpoint/meta.
- `codeswitch_ckpt_v6_dur05` — duration-weight checkpoint/meta.
- `codeswitch_ckpt_v6_wiki` — WIKI draft checkpoint/meta.
- `codeswitch_ckpt_v2`, `codeswitch_ckpt_v3`, `codeswitch_ckpt_v4_draft`, `teacher_cache` — untouched.

---

## 10. Next-move options

### Option A — Stronger draft/residual research path

Sequence, not a bundle:

1. **Draft-only:** iterative/AR CB0 refinement, speaker/style conditioning, or 20 h female mixed data.
2. Target draft `p ≈ 15–20%`.
3. **Residual-only:** improve `c` toward ≈20%.
4. Joint example: `p = 20%`, `c = 20%`, `w = 1.5%` → `rd_a = 5.2%`.

If only the draft changes, the target remains `p ≈ 38.3%` at current `c/w`. That is a ~5× improvement and is unlikely from data scale alone.

This path is research-heavy and requires schema changes for speaker/style conditioning. It should be staged one variable at a time.

### Option B — Explicit gate re-scope

Only with explicit user sign-off. This conflicts with the standing requirement and would break comparability with all prior runs. It is **not** recommended as a silent fix, and it must be documented as a scope change, not a model pass.

### Option C — Cheap draft-capacity diagnostic

Run the mixed 6 h cache with **only** the frame-decoder depth changed:

```bash
python train_draft_distill.py \
  --skip-teacher --teacher-cache real_cache/rasa_ta \
  --ckpt-dir codeswitch_ckpt_v6_diag2 \
  --device cuda:0 --dtype fp16 \
  --draft-frame-decoder-layers 2 \
  --draft-duration-conditioning --draft-duration-loss-weight 0.1 \
  --draft-epochs 300 --draft-batch 16 --min-epochs 150 --max-minutes 90 \
  --skip-residual
```

Compare against:

```text
val_top1 5.8416% | val_top5 17.9546% | oracle 9.8324% / 27.3165%
```

Stop rule: if `val_top1 < 7%`, capacity is not the lever; stop. If it clears ~10–15%, capacity helps but is still far from the gate. Do **not** train a residual or run the gate unless the projection reaches 5%.

**Recommendation:** run **C first**. It is one variable, cheap, and falsifiable. If it fails the 7% floor, do not spend on 20 h + residual. If it shows a real jump, use that signal to decide whether A is worth the larger draft/residual work.

---

## 11. Reproduction commands

### 11.1 6 h mixed prep

```bash
python prepare_real_speech.py \
  --out real_cache/rasa_ta \
  --hf-dataset ai4bharat/Rasa --hf-config Tamil --hf-split train \
  --language ta --durations ctc --limit 5000 --max-hours 6 --device cuda
```

### 11.2 6 h draft + residual

Use a new `CKPT_DIR`, not `codeswitch_ckpt_v6`:

```bash
python train_draft_distill.py \
  --skip-teacher --teacher-cache real_cache/rasa_ta \
  --ckpt-dir codeswitch_ckpt_v6_6h \
  --device cuda:0 --dtype fp16 --seed 42 \
  --draft-frame-decoder-layers 1 \
  --draft-duration-conditioning --draft-duration-loss-weight 0.1 \
  --draft-epochs 300 --draft-batch 16 --min-epochs 150 --max-minutes 90 \
  --residual-epochs 150 --residual-ramp-epochs 100 --residual-sched 1.0 \
  --residual-d-model 512 --residual-layers 6 --max-minutes-residual 90
```

### 11.3 Gate

```bash
python diagnose_audio_bug.py \
  --ckpt codeswitch_ckpt_v6_6h \
  --teacher-cache real_cache/rasa_ta \
  --shards 4 --out eval_results_v6_6h_gate --device cuda:0
```

Pass requires both `rt_a >= 5.0` and `rd_a >= 5.0`.

### 11.4 WIKI-only prep and draft

```bash
python prepare_real_speech.py \
  --out real_cache/rasa_ta_wiki \
  --hf-dataset ai4bharat/Rasa --hf-config Tamil --hf-split train \
  --language ta --durations ctc --limit 10000 --max-hours 6 \
  --style-filter WIKI --speaker-column gender --device cuda
```

```bash
python train_draft_distill.py \
  --skip-teacher --teacher-cache real_cache/rasa_ta_wiki \
  --ckpt-dir codeswitch_ckpt_v6_wiki \
  --device cuda:0 --dtype fp16 \
  --draft-frame-decoder-layers 1 --draft-duration-conditioning \
  --draft-epochs 300 --draft-batch 16 --min-epochs 40 --max-minutes 90 \
  --skip-residual
```

### 11.5 Notebook variables to change per experiment

```python
CKPT_DIR = REPO_DIR / "codeswitch_ckpt_v6_6h"   # never leave at codeswitch_ckpt_v6
GATE_DIR = REPO_DIR / "eval_results_v6_6h_gate"
DO_EVAL = False
ALLOW_EVAL_WITHOUT_GATE = False
```

---

## 12. Open questions

- Does a 2-layer frame decoder materially improve draft CB0 top-1?
- Can the residual ceiling `c` be raised from 10.6% toward 20% with a better residual architecture or conditioning?
- Does 20 h of **female-only** mixed-style data improve `p`, or does the 6 h cache already saturate the draft?
- Is the 4-shard gate sample representative? No confidence interval or seed-variance study exists.
- Does the gate’s preference for sharp/mode-collapsed drafts make the current objective misaligned with audio quality?
- Would a random or style-balanced 6 h sample change `p` materially compared with the first-rows cache?

---

## 13. Decision needed

**C first (recommended), A, or B.** If C is chosen, the next concrete action is a draft-only run with `--draft-frame-decoder-layers 2` into a new checkpoint directory; no residual and no gate unless the projection clears 5%.

---

# Appendix A — Pre-v6 experiment lineage (synthetic 2,000-shard teacher cache)

The main body above starts at the v6 real-data phase. This appendix records the earlier model-quality work that led to v6. These runs used the **2,000-shard synthetic teacher cache**, not Rasa speech. They are not directly comparable to the v6 gate numbers because the cache and 4-shard samples differ.

## A.1 Project context

- CodeSwitch-TTS has three paper operating points, Configs A/B/C.
- The paper-quality code gate applies to Configs **B and C** (draft + residual).
- The gate is `rt_a >= 5%` and `rd_a >= 5%`, measured by `diagnose_audio_bug.py`.
- The original failure was not the residual's teacher-CB0 path; it was the draft's **argmax CB0** path, which collapsed to a near-constant token.
- Plan 013 attempted to fix that collapse. Plan 014 then switched to real transcribed speech plus CTC durations (v6).

## A.2 v1 / v2 baseline

| Item | Value |
|---|---:|
| v1 `draft.val_top1` (run-context reference, not a committed artifact) | 7.4269% |
| v2 `val_top1` | **8.2483%** (`0.082483`) |
| v2 `rt_s` | 3.195% |
| v2 `rd_s` | 1.640% |
| v2 residual selection | best epoch 11 on **teacher** CB0 |
| v2 draft scheduled-sampling exposure at selection | 3.3% |

The v2 residual selected a pre-ramp state that had barely seen draft CB0. That was the first selection bug: validation used teacher CB0, so the saved state was not the inference distribution.

## A.3 v3 residual-only run

Purpose: test whether the residual could be improved independently of the draft.

Configuration:

```bash
python train_draft_distill.py --skip-teacher --skip-draft \
  --draft-ckpt codeswitch_ckpt_v2 \
  --residual-epochs 150 --residual-ramp-epochs 100 --residual-sched 1.0 \
  --residual-d-model 512 --residual-layers 6 \
  --min-epochs 150 --max-minutes-residual 90 \
  --ckpt-dir codeswitch_ckpt_v3
```

Result:

| Item | Value |
|---|---:|
| Run time | 31m 14s |
| `best_epoch` | 139 |
| `best_sched_p` | 1.0 |
| `selected_full_draft` | true |
| `best_epoch_any` | 149 |
| `best_sched_p_any` | 1.0 |
| `cb_top1_teacher_mean` | 2.02847% |
| `cb_top1_draft_mean` | 1.55954% |
| W&B | `edmjhmnm` |

v3 diagnostic:

| Metric | v2 | v3 | Gate | Result |
|---|---:|---:|---:|---|
| `cb0_s` | — | 2.364% | — | — |
| `cb0_a` | — | 9.942% | — | — |
| `rt_s` | 3.195% | 1.773% | — | — |
| `rt_a` | — | **5.594%** | `>= 5%` | **PASS** |
| `rd_s` | 1.640% | 0.094% | — | — |
| `rd_a` | — | **1.834%** | `>= 5%` | **FAIL** |

The residual could use teacher CB0 well enough to clear `rt_a`, but collapsed on draft CB0. `cb0_a = 9.942%` was misleading: a near-constant token can score near the teacher prior while producing degenerate audio.

## A.4 v3 argmax isolation

On the 4 diagnostic shards:

| Stat | Draft sample | Draft argmax |
|---|---:|---:|
| Unique tokens | 152.25 | 1.50 |
| Top-token fraction | 8.7% | 99.9% |
| Longest run | 2.5 frames | 382.75 frames |
| CB0 entropy | 4.51 nats | 0.01 nats |

Teacher CB0 prior over the cached data:

- Most common token: **568**
- Frequency: **8.09%** over **712,756** frames.

The argmax path emitted token `568` for essentially the whole utterance. The sample path was diverse but inaccurate. The v3 `student_full_argmax` WAV was near-silent:

- `rms ≈ 0.0098`
- `flatness ≈ 0.622`
- `zcr ≈ 0.199`

## A.5 Prior-subtraction sweep — rejected

Hypothesis: subtract `alpha * log p_teacher(token)` from draft logits.

| Alpha | CB0 top-1 | `rd_a` | Argmax unique | Top fraction | Longest run |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 10.019% | 1.834% | 1.5 | 99.9% | 383 |
| 0.25 | 11.240% | 1.827% | 1.5 | 99.9% | 383 |
| 0.50 | 11.433% | 1.797% | 1.5 | 99.9% | 383 |
| 0.75 | 7.927% | 1.398% | 2.0 | 99.5% | 300 |
| 1.00 | 0.663% | 0.225% | 20.0 | 25.6% | 60 |

Only `alpha = 1.0` diversified CB0, and it did so by falling to chance accuracy. The listener confirmed every prior-adjusted WAV was muffled/weird. **Rejected.**

## A.6 v4 per-sample frame lengths — triage failed

The identified defect: every sample was interpolated to the batch maximum, so short sentences were trained on a stretched text→frame alignment. v4 fixed per-sample lengths and masked padded tails.

Run:

- 300 epochs.
- 23m 54s.
- Existing 2,000-shard cache, no teacher rerun.
- Best validation loss: **5.0542 @ epoch 12**.

Result:

| Metric | v2 | v4 | Triage | Result |
|---|---:|---:|---:|---|
| `val_top1` | 8.2483% | **8.26%** | `>= 10%` | **FAIL** |
| `val_top5` | — | 23.07% | — | — |
| `val_argmax_unique` | — | 2.1 | `>= 16` | **FAIL** |
| `val_argmax_top_frac` | — | 98.8% | `<= 40%` | **FAIL** |
| `val_argmax_longest_run` | — | 332 | `<= 100` | **FAIL** |
| `val_argmax_entropy_nats` | — | 0.05 | — | — |

Per-sample lengths were necessary but not sufficient. **Do not use `codeswitch_ckpt_v4_draft` for a residual run.**

## A.7 v5 text-conditioned frame decoder — implemented, not trained

v5 added an opt-in `FrameDecoder`:

- learned frame queries;
- sinusoidal frame positions;
- whole-utterance length embedding;
- frame self-attention;
- cross-attention to the text encoder memory;
- per-sample frame padding masks;
- legacy `TemporalLengthProjector` remains the default when
  `frame_decoder_layers = 0`.

Parameter counts:

| Path | Parameters |
|---|---:|
| Legacy projector | 16,807,936 |
| 2-layer frame decoder | 27,314,688 |

Local tests at that point: **140 passed, 3 warnings**.

No v5 draft, v5 residual, v5 diagnostic, or v5 60-utterance evaluation was run in the recorded history. v6 is the first trained duration-conditioned frame-decoder model, using **1** frame-decoder layer on real speech.

## A.8 Original 2,000-sentence baseline — reference only

The original 2026-10-20 Kaggle run produced these approximate B/C reference numbers:

- B/C CER ≈ **93–94%**
- UTMOS ≈ **1.25**
- MCD ≈ **103–105**
- TTFA ≈ **36 / 80 ms**
- RTF ≈ **0.05 / 0.08**

These are **run-context reference points**, not committed artifacts. They are not a matched evaluation and must not be used as a paper claim.

## A.9 Test-count evolution

| Stage | Tests |
|---|---:|
| Early runbook | 101 passed |
| v3 residual selection | 131 passed |
| Prior-sweep stage | 134 passed |
| v4 per-sample lengths | 137 passed |
| v5 frame decoder | 140 passed |
| Current local HEAD | **179 passed, 3 warnings** |

---

# Appendix B — Plans 001–012 (code/infrastructure work)

These are not model-quality runs and produced no `rt_a`/`rd_a` numbers. They are part of the project history but outside the v6 experiment.

| Plan | Focus |
|---|---|
| 001 | Paper-table statistics defects: cold TTFA, Table V dispersion, switch-distance sentinel, gate-threshold pooling |
| 002 | Residual window coverage: shard tails and dropped frames |
| 003 | Demo verifier honesty: real teacher-forced CB0 probabilities or explicit failure |
| 004 | Honest Mode A telemetry: remove fabricated sub-latencies, fix docstrings |
| 005 | Secrets hygiene: stop persisting/uploading W&B API key; explicit artifact allow-list |
| 006 | Supply-chain pinning |
| 007 | Dependency declaration unification |
| 008 | Training/eval contract tests |
| 009 | Normalizer and CLI tests |
| 010 | Developer-experience onboarding |
| 011 | Documentation truth pass |
| 012 | Performance quick wins: batched ASR, memory guard, unused residual logits |

The plan index is `plans/README.md`. Plans 001–012 are code-quality work; their exact diffs and test details live in their individual plan files, not in this experiment log.

---

# Appendix C — v7 text/confidence residual (implemented; Kaggle run pending)

**Date:** 2026-10-31. **Status:** code complete and locally tested; **not yet trained on
the 6 h cache, so there is no v7 gate number.** The unchanged gate remains failed as of
the last recorded run (`rd_a = 2.2340%`).

## Why this intervention

The gate decomposes exactly as `rd_a = p*c + (1-p)*w`. At the best measured
`p = 7.9383%`, `c = 10.6244%`, `w = 1.5105%`, the binding term is `w`: about 92% of
deployed frames have a wrong draft CB0, and the CB0-only residual only reaches 1.51%
on them. At fixed `p/c`, passing requires `w ≈ 4.52%`. A CB0-only residual cannot
recover a wrong CB0; it has no text signal.

## What changed (one intervention)

`ResidualDecoder` now optionally consumes:

- the draft's frame-level text features (`DraftOutput.frame_features`, from the
  existing frame decoder) through a 512→512 projection;
- the draft CB0 argmax confidence, embedded into 16 bins;
- a learned fusion gate over `(CB0 embed, text embed, confidence embed)` plus a
  learned null-CB0 embedding for dropout/corruption training.

Legacy CB0-only checkpoints keep their exact state-dict keys and load unchanged.
`DraftTransformer.generate_sequence` chunks utterances longer than `max_frames`
(30 s × 86.13 Hz can exceed 2048), so the gate no longer drops long shards.
The residual trainer precomputes draft CB0, confidence, and frame features once,
then trains with the same scheduled-sampling ramp and full-draft checkpoint
selection as v6. New CLI flags: `--residual-text-conditioning`,
`--residual-confidence-conditioning`, `--residual-cb0-dropout`,
`--residual-cb0-corruption`. Artificial dropout/corruption defaults to 0.

Two diagnostic fixes matter for truth:

- `p` is now averaged over the same overlapping 86-frame windows as `c` and
  `w`. The earlier version used raw shard frames, so on a local 80-shard check
  the identity drifted by 0.0375 points. A unit test covers non-divisible shard
  lengths and asserts `decomposition_error < 1e-9`.
- The official gate averages per-shard window means. The diagnostic now reports
  that `rd_a_gate_style` next to the frame-weighted identity, plus the
  gate-style identity error. On the local 4-shard smoke it gave
  `rd_a_gate_style=0.7397%` versus the gate's `0.740%`, while the frame-weighted
  identity was `0.7367%`; the conditional identity is not exact under
  equal-shard weighting because `p/c/w` means are nonlinear.
- `diagnose_gate_ceiling.py --skip-ngram` skips the n-gram baselines for
  gate-only decomposition; `--ngram-shards 0` still means *all* shards.

## Local FLEURS calibration (80 shards, 0.230 h, 80 Tamil utterances)

CPU-only, random tiny draft/residual; only the n-gram baselines are meaningful.

| Baseline | Mean CB1–8 top-1 | Coverage |
|---|---:|---:|
| character/position text n-gram | **1.03%** | 100% |
| character trigram text n-gram | 0.97% | 100% |
| CB0-only n-gram | 1.21% | 100% |
| text + CB0 n-gram | 1.10% | 100% |
| CB0 AR n-gram | **8.94%** | 100% |

**Interpretation:** on real Tamil speech, a text-only character n-gram explains about
1% exact per-codebook identity, while the target `w` is 4.52%. A neural
text-conditioned residual must therefore exceed this empirical floor by ~4.4×; the
n-gram result is a floor, not a ceiling. The CB0 AR n-gram floor (8.94%) is consistent
with the measured NAR draft `p = 7.94%`, so a causal model can already match/beat the
n-gram estimate. FLEURS is not Rasa and this is not a gate result.

## Artifacts

- `CodeSwitch_TextResidual_Kaggle.ipynb`: standalone Kaggle runbook. Locates or pulls
  the preserved 6 h draft/cache, reproduces the base gate and decomposition, trains a
  residual-only text/confidence model into `codeswitch_ckpt_v7_textres`, runs the
  unchanged gate, reports exact truth, and only runs the 60-utterance evaluation if
  the gate passes.
- `diagnose_gate_ceiling.py`: p/c/w decomposition, per-codebook table, top-k coverage,
  confidence deciles, n-gram baselines, and required-p / required-w projections.
- Local suite: **179 passed, 3 warnings** (added text/confidence, chunking, and trainer
  plumbing tests).

## Truth still unchanged

No v7 number exists yet. The prior four gate results remain the evidence:
v3 `1.834%`, pilot `1.7593%`, 6 h `2.2487%`, LR floor `2.2340%` — all **FAIL**.
A v7 failure would be the fifth failure, not a reason to re-scope. A v7 pass would
still require the 60-utterance evaluation before any B/C quality claim.

---

# Appendix D — What this document still does not contain

This is a consolidated **experiment and decision** record, not a byte-for-byte archive of every artifact.

It does not contain:

- The 60-utterance evaluation results — **they were never run** because the gate failed.
- A v5 training/diagnostic result — **v5 was never trained** in the recorded history.
- Full per-epoch W&B curves, raw logs, or every checkpoint tensor.
- Raw data shards, audio, or the `teacher_cache` contents.
- Every individual diff in plans 001–012.
- The uncommitted 2026-10-20 baseline's raw `raw_results.jsonl` / `run_meta.json`; only the approximate reference numbers survive in the plan documents.
- Any B/C audio-quality claim. The only valid claim is the negative gate result.

For those, use the checkpoint directories, W&B runs, `plans/013-bc-noise-status.md`, `plans/014-real-data-duration.md`, and the individual plan files.
