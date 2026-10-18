# Plan 002: Stop dropping residual training shard tails; log dropped frames

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**: `git diff --stat d69b399..HEAD -- train_draft_distill.py tests/test_evaluation_suite.py KAGGLE_RUN.md`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: S
- **Risk**: LOW
- **Depends on**: none
- **Category**: bug
- **Planned at**: commit `d69b399`, 2026-10-19

## Why this matters

The residual decoder (the component that makes CB0-only verification viable)
is trained on ~1 s windows of each teacher shard, but the window loop's range
arithmetic silently skips each shard's trailing partial window. For shard
length `t`, coverage is `86·floor(t/86)` frames — the last `t mod 86` frames
are never windowed. For `t=140`, 54/140 = 38.6 % of frames are dropped; for
`t=171` (just under two windows), 85/171 = 49.7 % are dropped. The
`if end - start < 24: continue` guard was written to *keep* short tails (and
`end = min(start + self.WINDOW, t)` shows truncated tail windows were
intended), but the range never reaches them. Nothing is logged, so the loss is
invisible. Fix: iterate `range(0, t, self.WINDOW)`, and print how many frames
are actually dropped so future regressions are visible.

## Current state

Files involved:

- `train_draft_distill.py` — teacher capture + draft/residual distillation.
  The bug is in `ResidualTrainer.build_windows` (`:907-920`); the window-count
  log lives in `ResidualTrainer.train` (`:1006-1009`).
- `tests/test_evaluation_suite.py` — where the new regression test goes.
- `KAGGLE_RUN.md` — line 427 is a hand-maintained test count.

### The loop, as it exists today

`train_draft_distill.py:858-920` (excerpt):

```python
class ResidualTrainer:
    WINDOW = 86  # ~1 s of DAC frames; matches the streaming chunk size
    ...
    def build_windows(self, shards: List[TeacherShard],
                      draft_cb0: Optional[Dict[int, np.ndarray]] = None):
        windows = []
        for s in shards:
            t = s.frames
            draft_tokens = (draft_cb0 or {}).get(int(s.index))
            for start in range(0, max(1, t - self.WINDOW + 1), self.WINDOW):   # <-- line 913
                end = min(start + self.WINDOW, t)
                if end - start < 24:
                    continue
                draft_slice = (draft_tokens[start:end]
                               if draft_tokens is not None else None)
                windows.append((s.codes[0, start:end], s.codes[1:9, start:end], draft_slice))
        return windows
```

Why it drops tails: for `t ≥ 86` the starts are `0, 86, …, 86·floor(t/86) − 86`,
so the final `t mod 86` frames are never visited. For `t < 86+24 = 110`-ish
small shards the situation is similar (`range(0, max(1, t-85), 86)` yields only
start 0, and `end = t`). The `< 24` guard is dead for its intended purpose —
it can only fire for shards shorter than 24 frames.

Concrete arithmetic (verified):

| shard `t` | starts today | windows today | frames dropped | starts after fix | windows after fix | dropped after fix |
|---|---|---|---|---|---|---|
| 100 | {0} | 1 (`0:86`) | 14 | {0, 86} | 1 (`0:86`; `86:100` is 14 < 24) | 14 (logged) |
| 120 | {0} | 1 (`0:86`) | 34 | {0, 86} | 2 (`0:86`, `86:120`) | 0 |
| 171 | {0} | 1 (`0:86`) | 85 | {0, 86} | 2 (`0:86`, `86:171`) | 0 |

Downstream consumers of `build_windows` output: only `ResidualTrainer.train`
(`:1000`, `windows = self.build_windows(shards, draft_cb0)`), followed by the
train/val split by window index (`:1004-1005`, `i % 10 == 9` → val) and the
count log (`:1006-1009`):

```python
        print(
            f"[Residual] {len(windows)} windows of {self.WINDOW} frames "
            f"({len(train_w)} train / {len(val_w)} val)"
        )
```

`TeacherShard` (`:203-214`) is a plain dataclass: `index, raw_text, norm_text,
n_chars, frames, codes (9,T), audio_sec, target_source, gen_ms`. Existing
tests construct it directly, e.g. `tests/test_corpus_builder.py:227-229`:

```python
    codes = np.zeros((9, 20), dtype=np.int32)
    stale = T.TeacherShard(index=0, raw_text="வணக்கம் சார்.", norm_text="வணக்கம் சார்.",
                           n_chars=12, frames=20, codes=codes, audio_sec=0.5,
                           target_source="lm", gen_ms=1.0)
```

Repo test conventions: plain pytest functions; `train_draft_distill` is
imported inside test bodies as `import train_draft_distill as T` (see
`tests/test_evaluation_suite.py:595` and `tests/test_corpus_builder.py:210`).

## Commands you will need

| Purpose        | Command                                                            | Expected on success |
|----------------|--------------------------------------------------------------------|---------------------|
| Tests (full)   | `movio/.venv/bin/python -m pytest tests/ -q -p no:cacheprovider`   | previous count + 1 new test, all pass |
| Tests (target) | `movio/.venv/bin/python -m pytest tests/test_evaluation_suite.py -q -p no:cacheprovider` | all pass |
| Syntax gate    | `movio/.venv/bin/python -m py_compile train_draft_distill.py`      | exit 0              |
| Drift check    | `git diff --stat d69b399..HEAD -- train_draft_distill.py tests/test_evaluation_suite.py KAGGLE_RUN.md` | empty output before starting |

Note: the project venv is uv-managed and its `pytest` console script has a
broken shebang — always invoke `movio/.venv/bin/python -m pytest` from the
repo root, never `movio/.venv/bin/pytest` and never bare `pytest`.

## Scope

**In scope** (the only files you should modify):

- `train_draft_distill.py` — `build_windows` only.
- `tests/test_evaluation_suite.py` — one new test.
- `KAGGLE_RUN.md` — only the test-count line 427.

**Out of scope** (do NOT touch, even though they look related):

- The `< 24` minimum-window guard — keep it; it is the intended minimum for
  both a window's trainability and its batch statistics.
- The train/val split by window index (`:1004-1005`) — no change; more windows
  only shifts which indices land in val.
- `train_draft_distill.py`'s draft (CB0) path, teacher capture, or shard
  loading.
- The `[Residual] N windows ...` log at `:1006-1009` — keep it as is; the new
  dropped-frames line is additional.

## Git workflow

- Branch: `git checkout -b advisor/002-residual-window-coverage`.
- One commit: `Cover residual training shard tails and log dropped frames`
  (repo house style — descriptive sentence case, see `git log`: "Fix the
  teacher-forced scorer logits layout + Config C codebook vocab").
- Do NOT push or open a PR.

## Steps

### Step 1: Fix the loop and add dropped-frame accounting

Replace `train_draft_distill.py:907-920` with:

```python
    def build_windows(self, shards: List[TeacherShard],
                      draft_cb0: Optional[Dict[int, np.ndarray]] = None):
        windows = []
        total_frames = 0
        windowed_frames = 0
        for s in shards:
            t = s.frames
            total_frames += int(t)
            draft_tokens = (draft_cb0 or {}).get(int(s.index))
            for start in range(0, int(t), self.WINDOW):
                end = min(start + self.WINDOW, int(t))
                if end - start < 24:
                    continue
                windowed_frames += end - start
                draft_slice = (draft_tokens[start:end]
                               if draft_tokens is not None else None)
                windows.append((s.codes[0, start:end], s.codes[1:9, start:end], draft_slice))
        dropped = total_frames - windowed_frames
        if dropped:
            print(f"[Residual] windows cover {windowed_frames}/{total_frames} teacher "
                  f"frames; {dropped} frames dropped (tails shorter than 24 frames)")
        return windows
```

Notes:

- `range(0, t, self.WINDOW)` is the whole fix; `end = min(start + WINDOW, t)`
  now produces the truncated tail windows the guard was written for.
- `int(t)` guards against numpy integer types in `range`.
- The log fires only when frames are actually dropped (t=120 drops nothing;
  t=100 logs `14 frames dropped`).
- Behaviour change summary: t=120 now yields 2 windows instead of 1; t=100
  still yields 1 window, but the 14-frame tail skip is now explicit and
  logged; shards < 24 frames still yield 0 windows.

**Verify**: `movio/.venv/bin/python -m py_compile train_draft_distill.py` → exit 0.

### Step 2: Regression test for coverage and the dropped-frames log

Add to `tests/test_evaluation_suite.py`, directly after
`test_module_arg_parsers_exist` (currently `:586-600`). `np` is already
imported at module top; import `T` inside the test per repo convention:

```python
def test_residual_windows_cover_shard_tails(capsys):
    import train_draft_distill as T

    trainer = T.ResidualTrainer.__new__(T.ResidualTrainer)  # build_windows needs only WINDOW

    def _shard(index, frames):
        return T.TeacherShard(
            index=index, raw_text="x", norm_text="x", n_chars=1, frames=frames,
            codes=np.zeros((9, frames), dtype=np.int32), audio_sec=1.0,
            target_source="lm", gen_ms=1.0)

    # t=120: the 34-frame tail window is kept -> full coverage, nothing dropped
    windows = trainer.build_windows([_shard(0, 120)])
    assert [w[0].shape[0] for w in windows] == [86, 34]
    assert "frames dropped" not in capsys.readouterr().out

    # t=100: the 14-frame tail is below the 24-frame minimum -> dropped + logged
    windows = trainer.build_windows([_shard(1, 100)])
    assert [w[0].shape[0] for w in windows] == [86]
    assert "14 frames dropped" in capsys.readouterr().out
```

**Verify**: `movio/.venv/bin/python -m pytest tests/test_evaluation_suite.py -q -p no:cacheprovider`
→ all pass, including the new test.

### Step 3: Full suite, runbook count, index

3a. `movio/.venv/bin/python -m pytest tests/ -q -p no:cacheprovider` → all pass;
the count is the previous total + 1 (99 → 100 if plan 001 already landed; if
not, 98 → 99). Use the pytest summary line as the source of truth.

3b. Update the count in `KAGGLE_RUN.md:427` — currently:
`* Local test suite: \`python -m pytest tests/ -q\` (98 tests, no GPU and no network needed).`
Replace only the number with the exact count from your run.

3c. Update the `002` row in `plans/README.md` (status → DONE, or BLOCKED +
one-line reason).

**Verify**: `grep -n "tests)" KAGGLE_RUN.md` → shows the new count;
`git status --porcelain` → only `train_draft_distill.py`,
`tests/test_evaluation_suite.py`, `KAGGLE_RUN.md` modified.

## Test plan

- **New** `test_residual_windows_cover_shard_tails` in
  `tests/test_evaluation_suite.py` (case table above): t=120 → windows
  `[86, 34]`, no drop log; t=100 → windows `[86]`, `14 frames dropped` logged.
- Structural pattern: `tests/test_corpus_builder.py:219-230` (TeacherShard
  construction) + `tests/test_evaluation_suite.py:586-600` (module-import
  style).
- No existing test reads `build_windows` (verified: `grep -n "build_windows"
  tests/` → no matches), so nothing else should change behaviour.
- Verification: `movio/.venv/bin/python -m pytest tests/ -q -p no:cacheprovider`
  → all pass, N+1 tests.

## Done criteria

ALL must hold:

- [ ] `grep -n "max(1, t - self.WINDOW + 1)" train_draft_distill.py` → no matches
- [ ] `grep -n "frames dropped" train_draft_distill.py` → 1 match
- [ ] `movio/.venv/bin/python -m pytest tests/ -q -p no:cacheprovider` → all pass, previous count + 1
- [ ] `grep -n "tests)" KAGGLE_RUN.md` → count matches your pytest summary
- [ ] `git status --porcelain` → only the three in-scope files modified
- [ ] `plans/README.md` 002 row updated

## STOP conditions

Stop and report back (do not improvise) if:

- The `build_windows` excerpt does not match the live code (drifted repo).
- `grep -rn "build_windows" --include=*.py . | grep -v movio` finds a caller
  other than `ResidualTrainer.train` (`:1000`) — the changed window count
  could then affect something this plan did not anticipate.
- `TeacherShard`'s constructor signature differs from the excerpt above.
- A step's verification fails twice after a reasonable fix attempt.

## Maintenance notes

- The dropped-frames log is the alarm for this class of bug: if a future shard
  length change or window-size change makes coverage incomplete, the log says
  so on the first training run. Keep it.
- Adding up to one window per shard modestly increases residual training time;
  `ResidualTrainer.train` already caps wall time via `max_minutes`, so no
  budget change is needed on Kaggle.
- The train/val split (`i % 10 == 9`) remains by window index; with tail
  windows now included, val windows are still adjacent-in-time to train
  windows of the same shard (disjoint frames, not utterance-disjoint) — this is
  a known, accepted property of the current design, not a regression.
- If `WINDOW` or the `< 24` guard is ever changed, update
  `test_residual_windows_cover_shard_tails`'s expected shapes (86/34 and
  14-frame drop) in the same commit.
