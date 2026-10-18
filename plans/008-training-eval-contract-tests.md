# Plan 008: Pin the training→evaluation checkpoint contract in tests (dry-run round-trip, 36-key row schema, real W&B train keys)

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**: `git diff --stat d69b399..HEAD -- run_full_evaluation.py train_draft_distill.py tests/test_evaluation_suite.py tests/test_wandb_push.py KAGGLE_RUN.md`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; line
> numbers may have shifted (plan 006 adds a CLI flag in
> `run_full_evaluation.py`), but the quoted code snippets must still exist —
> locate them by content. If a quoted snippet is gone or changed, treat it as
> a STOP condition.
>
> **Dependency gate (plan 006)**: this plan's new tests assume the evaluator
> loads checkpoints with `weights_only=True` (plan 006). Run:
> `grep -n "weights_only=True" run_full_evaluation.py`
> - Two lines must print (the draft and residual `torch.load` calls; at
>   `d69b399` they are lines 3145 and 3157 with `weights_only=False`).
> - If instead you see `weights_only=False` twice, plan 006 has NOT landed →
>   STOP and run `plans/006-supply-chain-pinning.md` first.

## Status

- **Priority**: P2
- **Effort**: M — ~60 lines of test code, but test A executes the real
  dry-run trainer (fabricated shards + 3 draft epochs + 3 residual epochs on
  8 sentences; the dry-run is documented as "no GPU, ~30 s" at
  `KAGGLE_RUN.md:190`, so expect tens of seconds, not milliseconds)
- **Risk**: LOW–MED — test-only changes, but test A exercises real training
  code paths and write-to-`tmp_path` I/O; if it turns out too slow or flaky,
  the STOP condition below applies rather than deleting the assertions
- **Depends on**: `plans/006-supply-chain-pinning.md` (hard gate above)
- **Category**: tests (contract pinning)
- **Planned at**: commit `d69b399`, 2026-10-19

## Why this matters

Three interfaces between the two halves of the pipeline have **no test at
all**, and one test pins a contract that does not exist:

1. **No test loads a trainer-written checkpoint through the evaluator's
   loader.** The repository's own direction audit recorded this:
   "no test loads a checkpoint into the engine"
   (`plans/research/findings-2026-10-18.md`, direction report, Other
   discoveries). Test A closes the loop: the dry-run trainer writes
   `draft_cb0.pt`, `residual_cb1_8.pt`, `meta.json`
   (`train_draft_distill.py:1091-1156`), and the evaluator's
   `_load_checkpoints` (`run_full_evaluation.py:3134-3181`) must read them
   back. This is exactly the path that plan 006 makes stricter
   (`weights_only=True`): if a future producer stores a numpy scalar in a
   checkpoint payload, this test fails loudly at the checkpoint-loading
   boundary instead of during a 40-minute GPU run.
2. **The per-utterance JSONL row schema is a 36-key unwritten contract.**
   `_chunked_synthesis` builds config-B/C rows from `_base_row`
   (11 keys, `:752-767`) + `row.update` (24 keys, `:1055-1085`) +
   `_attach_zone_stats` (1 key, `:1089-1099`); every table, figure and the
   W&B flatten read these keys. Nothing pins the set. A silent key rename
   breaks Fig. 1–5 caption code, `gate_sweep`, and `push_to_wandb.py`
   without any test noticing.
3. **The W&B flatten test validates a fiction.** `tests/test_wandb_push.py`
   writes a `meta.json` fixture whose keys (`draft.top1_pct`,
   `residual.top1_cb1_8_pct`, `draft.best_val`, `length_model` without
   `min_frames`/`max_frames`/`n_samples`) exist **nowhere in the producer** —
   a repo-wide grep finds them only in the test file. The real producer
   writes `best_val_loss`/`val_top1`/`val_top5`/`cb_top1`/`cb_top1_mean`
   (`train_draft_distill.py:839-846`, `:1058-1064`, `LengthModel.to_dict`
   `:167-168`). So the test proves the flattener handles invented keys and
   would keep passing if the real keys changed. Rewriting the fixture to the
   real payload makes the test meaningful (and it already passes with the
   real keys — `push_to_wandb.py:216-247` flattens any dict generically via
   `flatten_numeric`, so this is a test fix, not a code fix).

Evidence: `plans/research/findings-2026-10-18.md`, entry
"2026-10-19 — Anchor verification for plans 006, 008–012", claims 1, 14–17,
22, 24; and the audit's tests report (same file, 2026-10-18 entries).

## Current state

Files involved:

- `tests/test_evaluation_suite.py` — 55 tests, imports at `:13-24`:
  ```python
  import codeswitch_benchmark_data as data          # :22
  import run_full_evaluation as R                   # :23
  from train_draft_distill import LengthModel, _CodeCaptureStreamer  # :24
  ```
  `torch` is **not** imported at module level; the new tests import it (and
  the normalizer / config) locally so the module import graph is unchanged.
- `tests/test_wandb_push.py` — 12 tests; **no `@pytest.fixture` anywhere in
  the suite**; `_workspace(tmp_path)` is a plain helper at `:107`.
- `train_draft_distill.py` — `run_dry_run` at `:1091-1156`; `save_checkpoint`
  at `:1077-1084`; stats blocks at `:839-846` / `:1058-1064`; `LengthModel`
  at `:136-173`.
- `run_full_evaluation.py` — `_load_checkpoints` at `:3134-3181` (returns
  `(draft_model, residual_model, length_model, draft_trained,
  residual_trained)`); `EvalSynthesizer` at `:695`; `_base_row` `:752-767`;
  `_chunked_synthesis` `:897-1087`; `row.update` `:1055-1085`;
  `_attach_zone_stats` `:1089-1099`; `_frames_for` fallback formula
  `:726-729`; `_postprocess` `:731-739` (the `peak > 1e-6` guard that makes
  all-zero audio safe lives in `codeswitch_tts/utils/audio.py:66`).
- `push_to_wandb.py` — `headline_metrics` at `:150-250`; the train-side
  flatten is `:240-247`:
  ```python
  for group in ("draft", "residual", "length_model"):
      block = ckpt_meta.get(group)
      if isinstance(block, dict):
          flatten_numeric(block, f"train/{sanitize_key(group)}", metrics)
  ...
  for key in ("corpus_size", "residual_scheduled_sampling"):
      _put(metrics, f"train/{sanitize_key(key)}", ckpt_meta.get(key))
  ```
- `KAGGLE_RUN.md:427` — the hand-maintained test count:
  `* Local test suite: \`python -m pytest tests/ -q\` (98 tests, no GPU and no network needed).`

Key producer contracts (quoted from the anchor-verification entry, claims
14–18):

- Dry-run `meta.json` (written at `train_draft_distill.py:1142-1155`):
  top-level keys `mode, length_model, draft, residual, versions` where
  - `draft` = exactly `{best_val_loss, best_epoch, epochs_run, val_top1, val_top5, train_seconds}`
  - `residual` = exactly `{best_val_loss, epochs_run, cb_top1, cb_top1_mean, train_seconds}`
  - `length_model` = exactly `{a, b, min_frames, max_frames, n_samples, mae_frames}`
- `save_checkpoint` payload = `{"state_dict": {k: CPU tensor}, "config":
  dataclasses.asdict(config), "meta": Dict}` and the dry-run `.pt` metas add
  `dry_run: True` + `param_count`.
- `run_dry_run` reads exactly `args.seed, args.num_sentences,
  args.teacher_cache, args.ckpt_dir, args.residual_sched`.

## Commands you will need

| Purpose | Command | Expected |
|---|---|---|
| Full suite | `movio/.venv/bin/python -m pytest tests/ -q -p no:cacheprovider` | previous count + 2 passed |
| Only the new tests | `movio/.venv/bin/python -m pytest tests/test_evaluation_suite.py -q -p no:cacheprovider -k "dry_run_checkpoints or chunked_synthesis_row_schema"` | `2 passed` |
| Only W&B tests | `movio/.venv/bin/python -m pytest tests/test_wandb_push.py -q -p no:cacheprovider` | `12 passed` |
| Producer keys sanity | `grep -n "top1_pct\|top5_pct\|best_val\"" tests/ push_to_wandb.py train_draft_distill.py` | after the fix, hits only in `train_draft_distill.py` (`best_val` local variable) — no `_pct` keys anywhere |

Never use `movio/.venv/bin/pytest` (broken shebang in this venv) — always
`movio/.venv/bin/python -m pytest`.

## Scope

**In scope**

- Two new tests in `tests/test_evaluation_suite.py` (appended after
  `test_codec_reencode_casts_waveform_to_encoder_dtype`, the last test in the
  file at `d69b399` — locate by name, not by line).
- The `meta.json` fixture rewrite at `tests/test_wandb_push.py:147-153` and
  the three assertions at `:191-193` (`test_headline_metrics_are_flat_numbers`).
- Updating the count in `KAGGLE_RUN.md:427`.

**Out of scope (do not touch)**

- No production code changes: `push_to_wandb.py`'s flatten is already
  generic and correct for the real keys; `train_draft_distill.py` and
  `run_full_evaluation.py` are read-only for this plan.
- Do not add pytest fixtures/refactor `_workspace` — minimal diff only.
- Do not fix `cb_top1`'s type (plan 006 owns the `weights_only` contract; if
  the new test A fails because of payload contents, STOP — that is exactly
  the signal 006 needs).

## Git workflow

Branch: `advisor/008-training-eval-contract-tests` (sentence-case commits,
never push). Suggested commits, one per step:

1. `Pin the checkpoint meta contract with a dry-run round-trip test`
2. `Pin the 36-key evaluation row schema`
3. `Point the W&B flatten test at the real checkpoint meta keys`

If a reviewer dispatched you and maintains `plans/README.md` themselves,
skip the status-row edit; otherwise set this plan's row to DONE when the
last step's verification passes.

## Steps

### Step 1 — Test A: dry-run → evaluator round-trip (append to `tests/test_evaluation_suite.py`)

Append this test at the end of the file (after
`test_codec_reencode_casts_waveform_to_encoder_dtype`). Local imports keep
the module import graph unchanged:

```python
def test_dry_run_checkpoints_satisfy_the_evaluation_contract(tmp_path):
    """The trainer's dry-run writes real checkpoints; the evaluator's loader
    must accept them, and the meta key sets are a pinned contract (the W&B
    flatten in push_to_wandb.py and KAGGLE_RUN.md §5 read these keys).

    Runtime: this executes the real dry-run trainer on 8 fabricated
    sentences (no GPU, no network) — tens of seconds on CPU.
    """
    from argparse import Namespace

    import torch

    import train_draft_distill as T
    from codeswitch_tts.config import CodeSwitchTTSConfig

    ckpt_dir = tmp_path / "ckpt"
    dry = Namespace(
        seed=7,
        num_sentences=8,
        teacher_cache=str(tmp_path / "teacher_cache"),
        ckpt_dir=str(ckpt_dir),
        residual_sched=0.0,
    )
    T.run_dry_run(dry)

    meta = json.loads((ckpt_dir / "meta.json").read_text(encoding="utf-8"))
    assert set(meta["draft"]) == {
        "best_val_loss", "best_epoch", "epochs_run",
        "val_top1", "val_top5", "train_seconds",
    }
    assert set(meta["residual"]) == {
        "best_val_loss", "epochs_run", "cb_top1", "cb_top1_mean", "train_seconds",
    }
    assert set(meta["length_model"]) == {
        "a", "b", "min_frames", "max_frames", "n_samples", "mae_frames",
    }

    draft, residual, length_model, draft_trained, residual_trained = R._load_checkpoints(
        Namespace(ckpt=str(ckpt_dir)), CodeSwitchTTSConfig(), torch.device("cpu"),
    )
    assert draft_trained and residual_trained
    assert length_model is not None
    assert length_model.n_samples == 8
```

Why each assertion:

- The key-set assertions pin the producer contract that
  `push_to_wandb.py:240-247` flattens and `KAGGLE_RUN.md:385` tells operators
  to read (`codeswitch_ckpt/meta.json` → `draft`).
- The `_load_checkpoints` call is the only test in the repo that loads a
  trainer-written `.pt` through the evaluator (post-006 it goes through
  `weights_only=True`; the payload contains only tensors, plain dicts and
  primitives, so it must load).
- `length_model.n_samples == 8` catches a silent switch of the
  `LengthModel.fit` input counting (the eval harness uses `n_samples` in its
  log line and the meta loader at `run_full_evaluation.py:3175-3183`).

**Verify**

```bash
cd /home/tripathiji/projects/all_work
movio/.venv/bin/python -m pytest tests/test_evaluation_suite.py -q -p no:cacheprovider \
  -k test_dry_run_checkpoints_satisfy_the_evaluation_contract
```

Expected: `1 passed`. If it fails inside `_load_checkpoints` with an
`UnpicklingError`/`weights_only` error while plan 006 HAS landed — that is a
real defect in the producer payloads; STOP and report (do not add
`weights_only=False` to the test).

### Step 2 — Test B: pin the 36-key row schema (append to `tests/test_evaluation_suite.py`)

```python
def test_chunked_synthesis_row_schema_is_pinned():
    """Pin the config-B/C per-utterance row schema (exactly 36 keys).

    Config-C rows are this schema plus ``gate_threshold``, ``gate_stat`` and
    ``fallback`` (added by ``synth_C``), so any change here changes the JSONL
    contract that every table, figure and the W&B flatten read.
    """
    import types
    from argparse import Namespace

    import torch

    from codeswitch_tts.config import CodeSwitchTTSConfig
    from codeswitch_tts.frontend.normalizer import TanglishNormalizer

    synth = R.EvalSynthesizer.__new__(R.EvalSynthesizer)
    synth.args = Namespace(boundary_frames=3)
    synth.config = CodeSwitchTTSConfig()
    synth.normalizer = TanglishNormalizer()
    synth.sample_rate = synth.config.audio.sample_rate
    synth.frames_per_chunk = int(synth.config.audio.frames_per_chunk)
    synth._dac_fallback = None
    synth._speech_backend = True
    synth.draft_trained = True
    synth.residual_trained = True
    synth.length_model = None          # exercises the formula fallback in _frames_for
    synth.scorer = None                # verify=False never touches the scorer
    synth.device = torch.device("cpu")

    class _FakeDraft:
        def generate_chunk(self, text, num_frames, temperature=1.0, device=None):
            tokens = torch.zeros((1, num_frames), dtype=torch.long)
            probs = torch.full((1, num_frames, 1024), 1.0 / 1024.0)
            return types.SimpleNamespace(tokens=tokens, probabilities=probs)

    class _FakeResidual:
        # **kwargs keeps this compatible with plan 012's return_logits= flag
        def __call__(self, cb0_tokens=None, temperature=1.0, sample=False, **kwargs):
            n = int(cb0_tokens.shape[-1])
            return types.SimpleNamespace(
                full_codebooks=torch.zeros((1, 9, n), dtype=torch.long)
            )

    synth.draft = _FakeDraft()
    synth.residual = _FakeResidual()
    synth.decode_codes = lambda codes: np.zeros(int(codes.shape[-1]) * 512, dtype=np.float32)

    utt = {
        "id": "ut-schema",
        "register": "transit",
        "text": "Driver-ku OTP 5291 sollunga, total fare Rs. 245.",
    }
    row, wave = synth._chunked_synthesis(utt, utt_index=0, seed=42, verify=False)

    expected_keys = {
        # _base_row (11)
        "schema", "config", "config_label", "utt_id", "register", "utt_index",
        "text", "normalized_text", "sample_rate", "untrained_modules", "error",
        # row.update (24)
        "ttfa_compute_ms", "ttfa_e2e_ms", "total_ms", "audio_sec", "rtf",
        "draft_ms", "verify_ms_total", "residual_ms_total", "dac_ms_total",
        "accept_true", "accept_synth", "logq_draft", "logq_kept", "n_frames",
        "n_transitions", "chunks", "verify_mode", "seed", "draft_confidence",
        "switch_frames", "frame_zones_b64", "frame_accept_true_b64",
        "frame_accept_synth_b64", "frame_valid_b64",
        # _attach_zone_stats (1)
        "zone_stats",
    }
    assert set(row.keys()) == expected_keys
    assert row["config"] == "B"
    assert row["error"] is None
    assert isinstance(wave, np.ndarray) and wave.ndim == 1 and wave.size > 0
```

Why this is safe / non-flaky:

- `verify=False` → `cfg == "B"`, the synthetic-target branch is skipped, and
  `scorer=None` is never touched (`run_full_evaluation.py:1004` guards with
  `if verify:`).
- The fake residual returns all-zero codebooks; `decode_codes` is stubbed to
  zeros, and `_postprocess` survives zero audio because
  `normalize_peak` guards `peak > 1e-6`
  (`codeswitch_tts/utils/audio.py:63-68`).
- The generated row is deterministic: `seed=42` seeds the draft call and
  the zone assignment reads only the utterance text.

**Verify**

```bash
cd /home/tripathiji/projects/all_work
movio/.venv/bin/python -m pytest tests/test_evaluation_suite.py -q -p no:cacheprovider \
  -k test_chunked_synthesis_row_schema_is_pinned
```

Expected: `1 passed`. If the set comparison fails, do NOT "fix" the test by
adding the unexpected key — the row schema changed and that must be a
deliberate, reviewed change (report it).

### Step 3 — Rewrite the W&B fixture + assertions to the real producer keys

**3a.** In `tests/test_wandb_push.py`, replace the `meta.json` block inside
`_workspace` (currently `:147-153`):

```python
    (ckpt_dir / "meta.json").write_text(json.dumps({
        "corpus_size": 300, "corpus_source": "codeswitch_corpus.txt",
        "target_source": "lm", "residual_scheduled_sampling": 0.5,
        "draft": {"best_val": 5.1126, "best_epoch": 10, "top1_pct": 7.13, "top5_pct": 21.32},
        "residual": {"best_val": 6.6032, "top1_cb1_8_pct": 1.65},
        "length_model": {"a": 3.84, "b": 95.5, "mae_frames": 33.1},
    }), encoding="utf-8")
```

with (keys copied from the real producer, `train_draft_distill.py:839-846`,
`:1058-1064`, `LengthModel.to_dict` — values are arbitrary but realistic):

```python
    (ckpt_dir / "meta.json").write_text(json.dumps({
        "corpus_size": 300, "corpus_source": "codeswitch_corpus.txt",
        "target_source": "lm", "residual_scheduled_sampling": 0.5,
        "draft": {"best_val_loss": 5.1126, "best_epoch": 10, "epochs_run": 12,
                  "val_top1": 0.0713, "val_top5": 0.2132, "train_seconds": 123.4},
        "residual": {"best_val_loss": 6.6032, "epochs_run": 8, "cb_top1": 0.0165,
                     "cb_top1_mean": 0.0198, "train_seconds": 45.6},
        "length_model": {"a": 3.84, "b": 95.5, "min_frames": 30, "max_frames": 430,
                         "n_samples": 300, "mae_frames": 33.1},
    }), encoding="utf-8")
```

**3b.** In `test_headline_metrics_are_flat_numbers` (def at `:177`), replace
the three train-side assertions (currently `:191-193`):

```python
    assert metrics["train/draft/top1_pct"] == pytest.approx(7.13)
    assert metrics["train/residual/top1_cb1_8_pct"] == pytest.approx(1.65)
    assert metrics["train/corpus_size"] == pytest.approx(300.0)
```

with:

```python
    assert metrics["train/draft/best_val_loss"] == pytest.approx(5.1126)
    assert metrics["train/draft/val_top1"] == pytest.approx(0.0713)
    assert metrics["train/draft/val_top5"] == pytest.approx(0.2132)
    assert metrics["train/residual/cb_top1"] == pytest.approx(0.0165)
    assert metrics["train/residual/cb_top1_mean"] == pytest.approx(0.0198)
    assert metrics["train/length_model/mae_frames"] == pytest.approx(33.1)
    assert metrics["train/corpus_size"] == pytest.approx(300.0)
    # the keys the old fixture invented must not exist anywhere
    assert not any("_pct" in key for key in metrics if key.startswith("train/"))
```

The last line is the regression guard: `train/*_pct` keys never come from
the real `meta.json` (repo-wide grep before this change: `top1_pct` appears
only in this test file). Note `metrics["train/draft/epochs_run"]` and
`train_seconds` will also be present — do not assert their absence.

**Verify**

```bash
cd /home/tripathiji/projects/all_work
movio/.venv/bin/python -m pytest tests/test_wandb_push.py -q -p no:cacheprovider
```

Expected: `12 passed` (the same 12 tests; this step adds none).

### Step 4 — Update the hand-maintained test count

The suite now has **previous count + 2** tests (98 → 100 on a 98-baseline
tree). Run the full suite and read the real number from the summary line,
then update `KAGGLE_RUN.md:427`:

- From: `* Local test suite: \`python -m pytest tests/ -q\` (98 tests, no GPU and no network needed).`
- To: the same line with the new count.

**Verify**

```bash
cd /home/tripathiji/projects/all_work
movio/.venv/bin/python -m pytest tests/ -q -p no:cacheprovider
grep -n "Local test suite" KAGGLE_RUN.md
```

Expected: `100 passed` (on the 98-baseline tree; if earlier plans in the
execution order already added tests, expect that count + 2 — the pytest
summary is the source of truth) and the line contains the same number.

## Test plan

| Test | File | What it pins |
|---|---|---|
| `test_dry_run_checkpoints_satisfy_the_evaluation_contract` | `tests/test_evaluation_suite.py` | producer meta key sets; evaluator loader round-trip (incl. post-006 `weights_only=True`); `LengthModel.n_samples` |
| `test_chunked_synthesis_row_schema_is_pinned` | `tests/test_evaluation_suite.py` | exact 36-key row schema for config B/C |
| `test_headline_metrics_are_flat_numbers` (edited) | `tests/test_wandb_push.py` | W&B flatten against the **real** checkpoint meta keys; no fabricated `train/*_pct` keys |

Expected totals: `tests/test_evaluation_suite.py` 55 → 57;
`tests/test_wandb_push.py` stays 12; whole suite previous + 2.

## Done criteria

- [ ] Both new tests pass in isolation and in the full suite.
- [ ] `tests/test_wandb_push.py` passes with the real-key fixture; a
      repo-wide grep for `top1_pct`/`top5_pct` finds no hits.
- [ ] No production file is modified by this plan
      (`git diff --stat d69b399..HEAD -- run_full_evaluation.py train_draft_distill.py push_to_wandb.py` shows only plan-006/007 changes if those landed, nothing from 008).
- [ ] `KAGGLE_RUN.md:427` carries the updated count.
- [ ] `plans/README.md` status row updated (unless the reviewer maintains it).

## STOP conditions

- The dependency gate shows `weights_only=False` at both `torch.load`
  sites → plan 006 has not landed → STOP and run 006 first.
- Test A exceeds ~3 minutes on CPU or fails inside `run_dry_run` →
  STOP and report (the dry-run path itself is then the finding; do not
  shrink the test around it without escalating).
- Test B fails on the key-set comparison → the row schema drifted; STOP
  and report the actual key set (adding/removing keys is a deliberate
  schema change with paper-artifact consequences).
- Any quoted "Current state" snippet cannot be located by content
  (only line numbers moved is fine) → STOP.
- `git status` shows modifications outside `tests/`, `KAGGLE_RUN.md`,
  and `plans/` → STOP.

## Maintenance notes

- Test A intentionally runs the real trainer rather than hand-crafted
  payloads. If the suite ever needs to be faster, the escape hatch is to
  move it behind a `-m slow` marker — do NOT delete the round-trip, because
  it is the only end-to-end guard on the checkpoint format (including
  plan 006's `weights_only=True` flip).
- Test B's `expected_keys` set is the JSONL row contract. When plan 012
  lands it touches the residual forward signature, not this schema; if any
  later plan legitimately extends the row, update `expected_keys` in the
  same commit and note the consumer impact (tables/figures/W&B) in the
  commit message.
- The W&B fixture now mirrors `train_draft_distill.py`'s real meta. If the
  trainer's stats blocks gain keys, this fixture should follow — it is the
  only place in the tests that models the checkpoint meta.
