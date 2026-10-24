# Plan 014 — Real speech → DAC shards + CTC durations + duration-conditioned draft (v6)

**Snapshot:** 2026-10-24
**Branch:** `fix/013-bc-noise`
**HEAD at snapshot:** `ed9b78f`
**Status:** **IN PROGRESS** — code/tests are complete and the public-speech
smoke is verified. The gated Rasa/IndicVoices-R corpus, full-scale v6
training, the unchanged 5% gate, and the 60-utterance evaluation have **not**
been run.

This plan is the implementation companion to the 2026-10-24 data-scale
decision: stop waiting for a large synthetic teacher cache, encode transcribed
human speech directly into the existing DAC CB0–8 shard contract, and give the
draft an explicit text-to-frame duration signal. It is Move 1 + Move 2 from
that discussion; Move 3 (acceptance/switch instrumentation) is already in
`run_full_evaluation.py`.

---

## 1. What problem this solves

Configs B and C still fail the paper-quality code gate at
`rt_a = 5.594%` / `rd_a = 1.834%` (gate: both `>= 5%`). The draft's CB0 argmax
collapses on the small teacher corpus, so the residual is fed a poor CB0 and
`rd_a` fails. More synthetic teacher data is not the only lever:

1. **Real speech is already transcribed.** Rasa / IndicVoices-R (and public
   FLEURS as a smoke) can be encoded directly with the same DAC codec. That
   produces the same 9-codebook shards the trainer already consumes, without a
   teacher pass.
2. **The draft has no explicit duration/length layout.** A FastSpeech-2-style
   duration head predicts how many DAC frames each character occupies. CTC
   forced alignment supplies the supervision, so inference needs no external
   aligner.

This plan does **not** claim that v6 passes the gate. It makes the experiment
possible and records exactly what was verified.

---

## 2. Hard constraints (unchanged)

- The code gate remains `rt_a >= 5%` **and** `rd_a >= 5%`. Do not lower,
  reinterpret, or hide it.
- Preserve all Plan 013 stop rules: no 60-utterance suite before the gate and
  an intelligible argmax smoke.
- Do not overwrite `codeswitch_ckpt_v2`, `codeswitch_ckpt_v3`,
  `codeswitch_ckpt_v4_draft`, or the existing `teacher_cache`.
- Old teacher shards must load unchanged. The new on-disk keys are additive.
- Never put `HF_TOKEN`, `WANDB_API_KEY`, `GITHUB_TOKEN`, or other secrets in
  code, notebooks, commands, or logs. Read them from the environment or
  Kaggle Secrets.
- Do not run Kaggle/GPU training from the local machine. The deliverable is
  code + tests + a ready-to-run notebook; the user runs it on Kaggle.
- Do not commit or push unless the user explicitly asks.

---

## 3. What changed

### 3.1 Shared shard contract

New package: `codeswitch_tts/data/`

| File | Responsibility |
|---|---|
| `shards.py` | `SpeechShard` (old `TeacherShard` is now an alias), atomic `.npz` save/load, optional duration metadata, legacy-key fallback, `(9, T)` validation |
| `audio.py` | WAV/bytes/path/`(array, sr)` and `torchcodec.AudioDecoder` decoding; mono conversion; resampling |
| `duration.py` | Pure-NumPy CTC Viterbi, character→CTC label mapping, exact seconds→DAC-frame conversion, lazy HF CTC adapter |
| `real_speech.py` | manifest/parquet/HF row adapters, language normalization, DAC encoder, filtering/resume/dedup pipeline, `meta.json`/`manifest.jsonl` |
| `__init__.py` | public exports |

`SpeechShard` fields kept from the old teacher cache:

`index, raw_text, norm_text, n_chars, frames, codes, audio_sec,
target_source, gen_ms`

Additive fields (all optional on load):

`char_durations, duration_source, alignment_score, alignment_coverage,
source, speaker, language, sample_rate`

`char_durations` is one `int32` per normalized **Python character**, in DAC
frames, and must sum to `frames`. BOS/EOS are not stored; the trainer inserts
zero-length sentinels.

### 3.2 Real-speech CLI

New `prepare_real_speech.py`:

- sources: JSONL/CSV manifest, recursive parquet directory, or HF streaming;
- `--durations ctc|none`;
- CTC models: Tamil
  `Harveenchadha/vakyansh-wav2vec2-tamil-tam-250` (default), English
  `facebook/wav2vec2-large-960h-lv60-self` (default), optional code-mixed
  model;
- filters: `--min-duration`, `--max-duration`, `--min-chars`,
  `--max-frames`, `--style-filter`, `--limit`, `--max-hours`;
- resume/dedup by content-addressed shard key; `--overwrite` to redo;
- `--require-durations` fails if any row lacks an alignment;
- `HF_TOKEN` only from the environment; gated datasets require prior terms
  acceptance.

The exact DAC frame rate is `44100 / 512 = 86.1328125`. Do not use the rounded
`87` from the DAC config.

### 3.3 Duration-conditioned draft

`codeswitch_tts/models/draft_transformer.py` adds:

- `DurationPredictor`: 2-layer MLP over text encoder states, predicts
  `log1p(per-character frames)`;
- `normalize_token_durations`: positive per-token durations → integer frame
  counts summing exactly to the acoustic length (largest remainder);
- `expand_text_by_durations`: repeats each text state by its duration to
  produce frame-level features;
- `FrameDecoder.duration_proj`: optional projection added to frame queries;
- `DraftOutput.duration_logits` and `DraftOutput.frame_durations`;
- `DraftTransformer.forward(..., token_durations=...)`: training-time oracle
  layout; when omitted, the predicted durations build the layout.

`DraftModelConfig` adds `duration_conditioning: bool = False` and
`duration_hidden: int = 256`. Defaults are false, so v2–v5 state dicts remain
compatible.

`DraftTrainer` adds:

- duration collation in tokenizer coordinates (BOS/EOS zero);
- masked smooth-L1 duration loss on rows that have CTC durations;
- uniform layout for legacy shards, with the duration loss masked out;
- `--draft-duration-conditioning`, `--draft-duration-loss-weight`,
  `--require-durations`;
- `val_duration_loss`, `duration_shards`, `duration_shard_fraction`,
  `val_duration_shards`, plus an **oracle-duration** top-1/top-5 ablation;
- the primary `val_top1`/`val_top5` remain inference-honest (predicted
  durations, no external aligner).

`TeacherShard` is now `TeacherShard = SpeechShard`, not a second dataclass. Old
teacher caches load through the additive fallback in `SpeechShard.load`.

### 3.4 Dependencies

`requirements.txt` adds the optional data-pipeline packages:
`datasets>=2.19.0`, `pandas>=1.5.0`, `pyarrow>=14.0.0`.

`requirements-kaggle.txt` is the notebook's narrow install set. It omits Torch
and torchaudio (Kaggle owns the matched CUDA pair) and omits
`descript-audio-codec` (its `descript-audiotools` dependency pins
`protobuf<3.20`). It uses only lower bounds for the scientific stack: Kaggle's
Python 3.13 image ships NumPy 2.x with a pandas built against the same C ABI,
so pinning `numpy<2` causes `numpy.dtype size changed`. `datasets==5.0.1` is
selected because it is the last release accepting `huggingface-hub<2`, keeping
Kaggle's preinstalled Gradio/Diffusers hub constraint intact. The notebook
checks imports and skips pip entirely when the data-path packages are present.

### 3.5 Kaggle notebook

New `CodeSwitch_DataScale_Kaggle.ipynb` (21 cells, 11 code cells):

- setup checklist (GPU, Internet, Secrets);
- clone/pull GitHub branch and print commit SHA;
- check imports, then install only missing packages from
  `requirements-kaggle.txt` (no Torch/NumPy downgrade, no conflicting
  `descript-audio-codec`; W&B included);
- verify the new feature flags exist in the checkout;
- public FLEURS Tamil smoke by default; Rasa/IndicVoices-R when `HF_TOKEN` is
  set and terms are accepted;
- W&B data-prep run, `meta.json` artifact;
- v6 draft/residual training, W&B config/git commit/metrics/model artifact;
- optional unchanged 5% gate via `diagnose_audio_bug.py`, with
  `rt_a >= 5%` and `rd_a >= 5%`;
- optional 60-utterance evaluation, blocked unless the gate ran and passed
  (or `ALLOW_EVAL_WITHOUT_GATE=True` is explicitly set);
- zip of checkpoint/meta/reports; raw shards excluded by default.

The notebook does **not** run locally and does **not** contain secrets.

---

## 4. Commands

### 4.1 Public real-speech smoke (verified)

```bash
export HF_TOKEN=...   # only for gated sources; not needed for FLEURS
movio/.venv/bin/python prepare_real_speech.py \
  --out /tmp/opencode/cs_fleurs_ta2/cache \
  --hf-dataset google/fleurs --hf-config ta_in --hf-split validation \
  --language ta --durations ctc --device cpu --limit 2 --min-duration 0.3
```

### 4.2 Real corpus (gated; Kaggle)

```bash
# HF_TOKEN from Kaggle Secrets, never inline
python prepare_real_speech.py \
  --out real_cache/rasa_ta \
  --hf-dataset ai4bharat/Rasa --hf-config Tamil --hf-split train \
  --language ta --durations ctc --limit 200 --max-hours 2 --device cuda
```

For IndicVoices-R:

```bash
python prepare_real_speech.py \
  --out real_cache/indicvoices_ta \
  --hf-dataset ai4bharat/indicvoices_r --hf-config Tamil --hf-split train \
  --language ta --durations ctc --limit 200 --max-hours 2 --device cuda
```

If some rows are code-mixed and have no Tamil CTC alignment, omit
`--require-durations`; those shards still train the CB0 head, and the duration
loss is masked to aligned rows.

### 4.3 v6 draft smoke (verified, CPU)

```bash
python train_draft_distill.py \
  --skip-teacher --teacher-cache /tmp/opencode/cs_fleurs_ta2/cache \
  --skip-residual --ckpt-dir /tmp/opencode/cs_v6_smoke2 \
  --device cpu --dtype fp32 --draft-epochs 1 --max-minutes 2 \
  --draft-batch 1 --draft-frame-decoder-layers 1 \
  --draft-duration-conditioning --draft-duration-loss-weight 0.1 \
  --require-durations
```

### 4.4 Full v6 run (not run)

Draft + residual, new checkpoint directory, no old checkpoint overwritten:

```bash
python train_draft_distill.py \
  --skip-teacher --teacher-cache real_cache/rasa_ta \
  --ckpt-dir codeswitch_ckpt_v6 \
  --device cuda:0 --dtype fp16 \
  --draft-frame-decoder-layers 1 \
  --draft-duration-conditioning --draft-duration-loss-weight 0.1 \
  --draft-epochs 300 --draft-batch 16 --min-epochs 100 --max-minutes 90 \
  --residual-epochs 150 --residual-ramp-epochs 100 --residual-sched 1.0 \
  --residual-d-model 512 --residual-layers 6 --max-minutes-residual 90
```

For a draft-only smoke, add `--skip-residual`. For a strict corpus, add
`--require-durations`; the full command above deliberately leaves it off so a
few unaligned code-mixed rows do not discard an otherwise usable corpus.

The gate remains separate and unchanged:

```bash
python diagnose_audio_bug.py \
  --ckpt codeswitch_ckpt_v6 \
  --teacher-cache real_cache/rasa_ta \
  --shards 4 --out eval_results_v6_gate --device cuda:0
```

It must print `rt_a >= 5.0` and `rd_a >= 5.0` before the 60-utterance suite.
The Kaggle notebook wraps this and marks FAIL otherwise.

---

## 5. Verification evidence (actual runs)

### 5.1 Tests

```text
$ .venv/bin/python -m pytest tests -q -p no:cacheprovider
171 passed, 3 warnings in 20.72s
```

New coverage:

- `tests/test_data_pipeline.py` — 23 tests: shard round-trip/legacy load,
  `(9,T)` validation, CTC Viterbi (clean/repeated/too-many-labels), label
  mapping, exact frame sums, manifest/parquet adapters, raw HF audio (no
  torchcodec requirement), language normalization, CLI guard, injected-encoder
  pipeline, resume/dedup;
- `tests/test_codeswitch_tts.py` — 7 duration-model tests (normalization sums,
  expansion, log1p inversion regression, head/projection gradients, config
  gating, output contract, legacy keys);
- `tests/test_corpus_builder.py` — duration collation/loss/oracle evaluation
  integration.

The 3 warnings are pre-existing `tight_layout` warnings from
`run_full_evaluation.py`, not caused by this plan.

A local legacy checkpoint (`codeswitch_ckpt_dryrun_smoke/draft_cb0.pt`)
loaded unchanged through `load_draft_checkpoint`: `frame_decoder_layers=0`,
`duration_conditioning=False`, 16.81M params, no duration-head keys. This is
the compatibility check for the v2–v5 shape.

### 5.2 Kaggle install resolver

`requirements.txt` is intentionally **not** the Kaggle install path. Its
optional `descript-audio-codec` dependency pulls `descript-audiotools`, which
pins `protobuf<3.20`, while the repo requires `protobuf>=5.29.5`. Asking pip to
solve the full manifest therefore backtracks until Kaggle reports
`resolution-too-deep`.

Kaggle's current image is Python 3.13 with a NumPy 2.x / pandas pair built
against the same C ABI. The first version of this fix pinned `numpy<2` and
produced `ValueError: numpy.dtype size changed` when `datasets` imported pandas.
The install set now has no upper bounds on NumPy, pandas, pyarrow, Transformers,
tokenizers, or huggingface-hub, and the notebook checks imports before running
pip at all.

A clean Python 3.13.15 pip dry run of `requirements-kaggle.txt` succeeds with:

```text
numpy 2.5.4, pandas 3.1.0, pyarrow 26.0.0, datasets 5.0.1,
huggingface-hub 1.33.0, transformers 5.20.2, soundfile 0.14.0, scipy 1.18.1
```

No torch/torchaudio, torchcodec, or descript package is pulled into that
isolated environment (Kaggle supplies Torch separately). The same 23 data
pipeline tests pass under Python 3.13.15, NumPy 2.5.4, pandas 3.1.0, datasets
5.0.1, and torch 2.14.1 CPU. A real `datasets` 5.0.1 check confirmed
`Audio(decode=False)` returns `{bytes, path}` for both map-style and iterable
datasets, so the HF adapter does not require a torchcodec wheel matching
Kaggle's Torch ABI.

### 5.3 Real Tamil speech → real DAC → Tamil CTC

Source: public `google/fleurs`, config `ta_in`, split `validation`, 2 rows.

| Shard | Frames | Audio | Chars | Duration sum | Coverage |
|---|---:|---:|---:|---:|---:|
| `shard_2cdcbb91866ebd28.npz` | 1214 | 14.1 s | 211 | 1214 | 0.995 |
| `shard_d3596b071b9238cb.npz` | 956 | 11.1 s | 159 | 956 | 0.962 |

Both `char_durations` sums exactly equal `frames`; both are `language=ta` with
`duration_source=ctc:Harveenchadha/vakyansh-wav2vec2-tamil-tam-250`. This
verifies the `torchcodec` audio path, real DAC encoding, real Tamil CTC
alignment, exact frame scaling, and the shard writer.

The Tamil CTC adapter was also loaded directly on synthetic audio: 50 Hz,
`blank_id=1`, 99 CTC frames for 1.98 s, coverage 1.0. That verifies the model
contract without claiming speech quality.

A separate one-off public English LibriSpeech smoke verified the
`torchcodec.AudioDecoder` path with real speech: 5.855 s → 504 DAC frames,
duration sum 504, coverage 0.989. That cache was in `/tmp` and was cleaned by a
session restart; the reproducible command is the same CLI with
`--hf-dataset hf-internal-testing/librispeech_asr_dummy --hf-config clean
--hf-split validation --language en --ctc-model-en facebook/wav2vec2-base-960h`.

### 5.4 Duration-conditioned v6 smoke

Trained on the two FLEURS shards for 1 epoch on CPU
(`/tmp/opencode/cs_v6_smoke2`):

- 23.83M params;
- duration predictor and `frame_decoder.duration_proj` keys present;
- `duration_shards = 2/2`, `duration_shard_fraction = 1.0`;
- `val_duration_loss = 0.7071959972381592` (finite);
- `val_top1 = 1.67%`, `val_top5 = 11.09%` on a 1-utterance val split — a
  plumbing smoke, **not** a quality result;
- oracle-duration and predicted-duration top-1/top-5 are finite and printed.

This verifies the duration loss, masked collation, duration-conditioned frame
decoder, checkpoint config, and meta recording. It does not verify convergence,
generalization, or the gate.

---

## 6. Not verified / blockers

1. **Rasa and IndicVoices-R were not downloaded.** Both are gated; this
   environment has no user `HF_TOKEN`. Their exact column names, config names,
   and text punctuation are therefore unconfirmed. The generic adapters cover
   the expected `audio`/`text`/`speaker_id`/`language`/`style` schemas, and the
   notebook discovers the Tamil config, but the first real run must inspect
   the first few shards and `meta.json`.
2. **No full-scale v6 training.** Only 2 shards / 1 epoch / CPU. No B/C quality,
   intelligibility, or gate claim.
3. **No 5% gate run.** `rt_a`/`rd_a` remain unmeasured for v6. Do not run the
   60-utterance suite until the gate passes and the argmax smoke is
   intelligible.
4. **Code-mixed alignment is partial by construction.** A Tamil CTC vocabulary
   cannot represent Latin characters; those characters get zero duration and
   lower `alignment_coverage`. The trainer masks by positive duration, so this
   is visible but not silently treated as full supervision.
5. **No external duration baseline.** The CTC model's alignment is used as
   supervision, not validated against human phone boundaries. The oracle vs
   predicted ablation is a relative diagnostic, not proof of phonetic
   accuracy.
6. **The English CTC path was a one-off /tmp smoke**, not retained as an
   artifact. Reproduce with the command in §5.3.

---

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Duration head gets no gradient on a mixed corpus | `duration_shards` / `val_duration_shards` are recorded; watch the fraction. Use `--require-durations` only on a fully aligned corpus. |
| CTC coverage silently low | `alignment_coverage` is stored per shard and averaged in `meta.json`; inspect before training. |
| Wrong frame rate stretches every layout | Use `44100 / 512 = 86.1328125`; tested by exact duration sums. |
| `datasets` version returns a new audio object | `decode_audio` handles the `torchcodec.AudioDecoder` API; a focused test covers it. The notebook attempts an optional `torchcodec` install and warns if the wheel is unavailable. |
| Old checkpoint incompatibility | New config defaults are false; v2–v5 keys are untouched; `TeacherShard` is an alias with legacy load fallback. Tested by the existing checkpoint/legacy-shard tests. |
| Gated token leakage | CLI reads `HF_TOKEN` from the environment only; notebook reads Kaggle Secrets and never prints it. |
| Full run silently overwrites v2–v4 | The runbook uses a new `codeswitch_ckpt_v6`; the trainer already refuses `--ckpt-dir` equal to a loaded draft source. |
| W&B missing during training | The trainer does not create a run; the notebook creates data/train/gate runs and logs config, git commit, metrics, and artifacts. Offline mode prints `wandb sync` instructions. |

---

## 8. Kaggle runbook

1. Push `fix/013-bc-noise` (or merge it) before running the notebook. The
   notebook clones the branch; uncommitted local work is invisible.
2. Open `CodeSwitch_DataScale_Kaggle.ipynb` in Kaggle.
3. Set **Accelerator → GPU** and **Internet → On**.
4. Add Secrets: `WANDB_API_KEY`; `HF_TOKEN` only for gated Rasa/IndicVoices-R;
   `GITHUB_TOKEN` only if the repo is private.
5. Default config is the public FLEURS Tamil smoke (`SMOKE=True`,
   `DATASET_CHOICE="fleurs_ta"`). Run All to verify plumbing.
6. For the real corpus set `SMOKE=False`, `DATASET_CHOICE="rasa_ta"` (or
   `"indicvoices_ta"`), and increase `PREP_LIMIT` / `PREP_MAX_HOURS` as
   needed.
7. For a full run set `DO_RESIDUAL=True`; keep `DO_GATE=True` before
   `DO_EVAL=True`.
8. The notebook prints the commit SHA, checkpoint path, W&B URLs, and a zip
   path under `/kaggle/working`.

---

## 9. Rollback

- Code: this work is uncommitted; `git restore` the modified files and remove
  the new `codeswitch_tts/data/`, `prepare_real_speech.py`,
  `tests/test_data_pipeline.py`, and `CodeSwitch_DataScale_Kaggle.ipynb`.
- Data/checkpoints: delete `real_cache/` and `codeswitch_ckpt_v6/`. Do **not**
  delete `teacher_cache/`, `codeswitch_ckpt_v2`, `_v3`, or `_v4_draft`.
- Format: new shard keys are additive; old caches remain readable, so no
  migration/rollback of existing shards is required.

---

## 10. Acceptance checklist

- [x] `SpeechShard` additive contract; old teacher caches load unchanged.
- [x] Real manifest/parquet/HF adapters and `torchcodec` audio decode.
- [x] CTC Viterbi + exact DAC-frame duration conversion.
- [x] Real Tamil FLEURS → real DAC shards with exact duration sums.
- [x] Duration predictor + duration-conditioned frame decoder.
- [x] Trainer duration collation/loss/oracle ablation and flags.
- [x] 171 tests pass; no new warnings attributable to this plan.
- [x] Self-contained Kaggle notebook statically validated (JSON parses; all
      11 code cells compile).
- [ ] Rasa / IndicVoices-R real corpus prepared (gated; needs user `HF_TOKEN`).
- [ ] Full-scale v6 draft/residual trained on the real corpus.
- [ ] Unchanged 5% `rt_a`/`rd_a` gate passes.
- [ ] 60-utterance evaluation and intelligibility check.
- [ ] B/C quality claim (blocked until the gate passes).

Do not mark this plan `DONE` until the three unchecked run items above have
real evidence. Code completion is not quality.
