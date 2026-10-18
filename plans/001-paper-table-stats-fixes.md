# Plan 001: Fix four paper-table statistics defects (cold TTFA, Table V dispersion, switch-distance sentinel, gate-threshold pooling)

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**: `git diff --stat d69b399..HEAD -- run_full_evaluation.py tests/test_evaluation_suite.py KAGGLE_RUN.md`
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

Four independent statistics defects make the paper artifacts lie or mislead:
Table V prints `±0.0` for every gate operating point (std key-name mismatch),
the W&B cold-TTFA scalars are silently never logged on real runs (the producer
emits a raw float where the consumer requires a `_stats` dict), short
utterances' "no switch yet" frames are mixed into the *low* distance buckets of
the distance-to-switch analysis (the sentinel contradicts its own docstring),
and Table I can silently pool several gate thresholds into one row that is
labelled as the un-gated hybrid. All four are verified in the current code; the
fixes are small and mechanical. Their common cost: a reviewer who re-runs the
evaluation can contradict a number the paper prints.

## Current state

Files involved:

- `run_full_evaluation.py` — evaluation/production harness (aggregator, LaTeX
  table builders, figure code, main loop). ~3,639 lines.
- `tests/test_evaluation_suite.py` — pytest unit tests for those helpers.
- `KAGGLE_RUN.md` — runbook; line 427 is a hand-maintained test count.
- `push_to_wandb.py` — W&B consumer. **Read-only in this plan** — its
  expectations are already correct; the producer must be fixed to match it.

### Defect 1 (C05): cold-TTFA producer emits a float, consumer requires a dict

Producer — `run_full_evaluation.py:1647-1655` (inside `aggregate_results`):

```python
agg[cfg] = {
    "n": len(ok),
    "errors": len(err),
    "ttfa_compute": _stats([r.get("ttfa_compute_ms") for r in ok]),
    "ttfa_e2e": _stats([r.get("ttfa_e2e_ms") for r in ok]),
    "ttfa_compute_cold": first.get("ttfa_compute_ms"),      # <-- raw float
    "ttfa_e2e_cold": first.get("ttfa_e2e_ms"),              # <-- raw float
    "ttfa_compute_warm": _stats([r.get("ttfa_compute_ms") for r in warm], ci=True),
```

Consumer — `push_to_wandb.py:168-169`:

```python
put("latency/ttfa_compute_cold_ms", _stat(agg.get("ttfa_compute_cold"), "p50"))
put("latency/ttfa_e2e_cold_ms", _stat(agg.get("ttfa_e2e_cold"), "p50"))
```

`_stat` returns `None` for non-dicts and the `_put` helper drops `None` — so
these two W&B scalars are absent from every real push. The test suite cannot
catch it because `tests/test_wandb_push.py:48-49` hand-builds the fixture as
`"ttfa_compute_cold": _stats([base + 100])`, i.e. the *tested* schema is the
dict, while the *produced* schema is a float.

In-repo consumers of the two keys that must keep working after the change:

- `run_full_evaluation.py:2624` (Table I cold column):
  `_tex(a.get("ttfa_compute_cold"), 0),`
- `run_full_evaluation.py:2158-2166` (figure, panel "(a) time to first audio"):
  `cold = agg[c].get("ttfa_compute_cold")` then `float(cold)`.
- `_session_cold` (`run_full_evaluation.py:1585-1590`; rendered at
  `:2644-2651`; pushed at `push_to_wandb.py:230-234`) stores raw floats
  deliberately and is consumed by a *different* push path — leave it alone.
- `tests/test_evaluation_suite.py:354`:
  `assert agg["C"]["ttfa_compute_cold"] == pytest.approx(260.0)`.

### Defect 2 (C04): Table V's CB0 column always renders `±0.0`

`run_full_evaluation.py:2986-2997` (`build_table_V`) — last column of each row:

```python
_mean_pm(entry.get("accept_true") or {}, "mean"),
```

`_mean_pm` (`:2566-2572`) looks up `key + "_std"` → `"mean_std"`. But
`entry["accept_true"]` is a `_stats` dict built in `aggregate_gate_sweep`
(`:1899`: `"accept_true": _stats([...])`), whose keys are
`n/mean/std/p50/p90` (`_stats`, `:1494-1500`). So `mean_std` is always `None`
and `_mean_pm` defaults the std to `0.0` → every cell renders `X.Y±0.0`.
The correct helper for `_stats` dicts is `_pm` (`:2558-2563`, reads `std`);
Table I already uses it for the same dict shape (`:2634-2635`). The zone
tables use `_mean_pm` correctly with the *zone* shape (`accept_true` +
`accept_true_std`, built at `:1554-1562`) — **do not change `_mean_pm` itself,
only this one call site.**

### Defect 3 (C07): switch-distance sentinel lands in low buckets for n < 16

`run_full_evaluation.py:370-388`:

```python
def switch_distances(n_frames: int, switch_frames: Sequence[int]) -> np.ndarray:
    """Frames since the most recent Tamil<->Latin switch per frame.

    Frames before the first switch (or in utterances without switches) get the
    sentinel ``n_frames``, i.e. they fall in the ``16+`` bucket.
    """
    n = max(0, int(n_frames))
    dist = np.full(n, n, dtype=np.int64)
    if n == 0:
        return dist
    switches = sorted({int(s) for s in (switch_frames or []) if 0 <= int(s) < n})
    if not switches:
        return dist
    last = np.zeros(n, dtype=np.int64)
    for s in switches:
        last[s:] = s
    dist = np.arange(n, dtype=np.int64) - last
    dist[: switches[0]] = n
    return dist
```

Buckets (`:365-367`): `("0-1",0,2), ("2-3",2,4), ("4-7",4,8), ("8-15",8,16),
("16+",16,10**9)`, masked by `(dist >= lo) & (dist < hi)` (`:1929-1933`).
For `n < 16` the sentinel `n` lands in a *low* bucket — e.g. `n=10` → "8-15" —
directly contradicting the docstring ("they fall in the `16+` bucket").

### Defect 4 (C08): Table I can silently pool multiple gate thresholds

`run_full_evaluation.py:1591-1604`:

```python
for cfg in configs:
    ok = [r for r in rows if r.get("config") == cfg and not r.get("error")]
    err = [r for r in rows if r.get("config") == cfg and r.get("error")]
    if cfg == "C" and any(float(r.get("gate_threshold") or 0.0) != float(gate_threshold)
                          for r in ok):
        filtered = [r for r in ok
                    if float(r.get("gate_threshold") or 0.0) == float(gate_threshold)]
        if filtered:  # never let the gate sweep empty the main comparison
            ok = filtered
            err = [r for r in err
                   if float(r.get("gate_threshold") or 0.0) == float(gate_threshold)]
    if not ok:
        agg[cfg] = {"n": 0, "errors": len(err)}
        continue
```

Table I always requests `gate_threshold=0.0` (`_finish`, `:3426-3429`). With
`--gate-mode synth --gate-thresholds 0.6,0.9`, the explicit list is returned
verbatim without adding 0.0 (`parse_gate_thresholds`, `:1738-1747`; only
`auto` prepends 0.0, `:1729`), and the run loop synthesises config C once per
threshold (`:3334-3356`). Then `filtered` is empty and the code keeps **all**
C rows — a silent mixture of two operating points presented as "the un-gated
hybrid". The row records `"gate_threshold": float(gate_threshold)` (the
*requested* 0.0, `:1674`), so the mixture is also unlabeled.

### Test anchors (current expectations)

- `tests/test_evaluation_suite.py:354` — pins cold TTFA as a float.
- `tests/test_evaluation_suite.py:606-616` (`test_switch_distances_and_buckets`)
  — pins the current sentinel (`switch_distances(10, []) == 10` at line 614).
- `tests/test_evaluation_suite.py:794-803` (`test_gate_threshold_filtering_in_aggregate`)
  — pins the pooling fallback: `aggregate_results(rows, ["C"], gate_threshold=0.9)`
  keeps all 6 C rows (`assert fallback["C"]["n"] == 6`).
- Fixture `_fabricated_rows()` (`:280-345`) has 3 C rows at τ=0.0 (i=0,1,2) and
  3 C rows at τ=0.5 (i=0 fallback + i=1,2 speculative).
- `tests/test_wandb_push.py:48-49` (`_stats([...])` fixture) and `:183`
  (`latency/ttfa_compute_cold_ms/A == 420.0`) must both keep passing untouched.

Repo conventions: pytest module-style tests, `import run_full_evaluation as R`;
console messages are short prints prefixed `[Gate]` / `[MCD]` etc.; acceptance
percentages are stored on a 0–100 scale.

## Commands you will need

| Purpose        | Command                                                            | Expected on success |
|----------------|--------------------------------------------------------------------|---------------------|
| Tests (full)   | `movio/.venv/bin/python -m pytest tests/ -q -p no:cacheprovider`   | `98 passed` before changes, `99 passed` after |
| Tests (target) | `movio/.venv/bin/python -m pytest tests/test_evaluation_suite.py tests/test_wandb_push.py -q -p no:cacheprovider` | all pass |
| Syntax gate    | `movio/.venv/bin/python -m py_compile run_full_evaluation.py`      | exit 0              |
| Drift check    | `git diff --stat d69b399..HEAD -- run_full_evaluation.py tests/test_evaluation_suite.py KAGGLE_RUN.md` | empty output before starting |

Note: the project venv is uv-managed and its `pytest` console script has a
broken shebang — always invoke `movio/.venv/bin/python -m pytest` from the
repo root, never `movio/.venv/bin/pytest` and never bare `pytest`.

## Scope

**In scope** (the only files you should modify):

- `run_full_evaluation.py`
- `tests/test_evaluation_suite.py`
- `KAGGLE_RUN.md` — only the test-count line 427.

**Out of scope** (do NOT touch, even though they look related):

- `push_to_wandb.py` and `tests/test_wandb_push.py` — the consumer is already
  correct; if it fails after your change, your producer change is wrong.
- `_session_cold` handling (`run_full_evaluation.py:1585-1590`, `:2644-2651`)
  — deliberately raw floats, consumed by a different push path.
- Any table other than Table V's CB0 cell; any figure code other than the cold
  marker extraction at `:2160`.
- `_mean_pm` itself — it is correct for the zone shape used by Tables II/III.
- `switch_distance_table` masking logic (`:1929-1933`) — the analysis is right;
  only the sentinel input is wrong.

## Git workflow

- Branch: `git checkout -b advisor/001-paper-table-stats` (repo history has no
  branch prefixes to copy, so use this advisor convention).
- One commit per defect (4 code commits, tests in the same commit as their
  fix). Commit messages in the repo's house style — descriptive sentence case,
  e.g. from `git log`: "Fix the UTMOS loader and make the confidence-gate curve
  real (derived sweep + auto thresholds)". Suggested messages:
  1. `Emit cold-TTFA values as stats dicts so W&B cold scalars are logged`
  2. `Report real dispersion in Table V's CB0 acceptance column`
  3. `Send short-utterance switch-distance sentinels to the 16+ bucket`
  4. `Stop Table I pooling gate thresholds; record the tau actually used`
- Do NOT push or open a PR.

## Steps

### Step 1: Emit cold TTFA as `_stats` dicts (C05)

1a. Before editing, confirm the consumer set (expected exactly the matches
below; anything else → STOP):
`grep -rn "ttfa_compute_cold\|ttfa_e2e_cold" --include=*.py . | grep -v movio`
→ producer at `run_full_evaluation.py:1652-1653`, consumers at `:2160`,
`:2624`, `push_to_wandb.py:168-169`, fixtures at `tests/test_evaluation_suite.py:354`
and `tests/test_wandb_push.py:48-49`.

1b. Replace `run_full_evaluation.py:1652-1653` with:

```python
            "ttfa_compute_cold": _stats([first.get("ttfa_compute_ms")]),
            "ttfa_e2e_cold": _stats([first.get("ttfa_e2e_ms")]),
```

(One value → `n=1`, `std=0`; a `None` value → empty `_stats` dict whose `p50`
is `None`, which renders/consumes as `--`/absent, same as today.)

1c. Fix the two in-repo readers so they extract `p50`:

- `:2624`: `_tex(a.get("ttfa_compute_cold"), 0),` →
  `_tex((a.get("ttfa_compute_cold") or {}).get("p50"), 0),`
- `:2160`: `cold = agg[c].get("ttfa_compute_cold")` →
  `cold = (agg[c].get("ttfa_compute_cold") or {}).get("p50")`

1d. Update the existing assertion in `tests/test_evaluation_suite.py`
(`test_aggregation_and_zone_weighting`, formerly line 354):

```python
    cold = agg["C"]["ttfa_compute_cold"]
    assert cold["n"] == 1 and cold["p50"] == pytest.approx(260.0)
    assert agg["C"]["ttfa_e2e_cold"]["p50"] == pytest.approx(300.0)
```

(The first C row is `i=0` with `ttfa_compute_ms=260.0`, `ttfa_e2e_ms=300.0`.)

**Verify**: `movio/.venv/bin/python -m pytest tests/test_evaluation_suite.py tests/test_wandb_push.py -q -p no:cacheprovider`
→ all pass (test_wandb_push passing untouched proves the consumer contract is
now satisfied by the producer shape).

### Step 2: Real dispersion in Table V's CB0 column (C04)

2a. Replace `run_full_evaluation.py:2996`:

```python
            _pm(entry.get("accept_true") or {}),
```

(`entry["accept_true"]` is the `_stats` dict; `_pm` reads its `std`.)

2b. Add a regression test in `tests/test_evaluation_suite.py` after
`test_table_five_caption_states_the_gate_mode` (currently ends at `:791`):

```python
def test_table_five_cb0_reports_std():
    sweep = [{
        "threshold": 0.85, "n": 2, "fallback_utts": 0, "fallback_rate": 0.0,
        "accept_true": R._stats([40.0, 50.0]),
    }]
    tex = R.build_table_V(sweep, None, derived=True, thresholds=[0.85])
    # mean 45.0, population std 5.0 -> real dispersion, not "+/-0.0"
    assert "45.0$\\pm$5.0" in tex
```

**Verify**: `movio/.venv/bin/python -m pytest tests/test_evaluation_suite.py -q -p no:cacheprovider`
→ all pass, including the new test and the two existing `build_table_V`
tests.

### Step 3: Sentinel `max(n, 16)` for switch distances (C07)

3a. Replace `run_full_evaluation.py:370-388` with:

```python
def switch_distances(n_frames: int, switch_frames: Sequence[int]) -> np.ndarray:
    """Frames since the most recent Tamil<->Latin switch per frame.

    Frames before the first switch (or in utterances without switches) get a
    sentinel that is guaranteed to land in the ``16+`` reference bucket:
    ``max(n_frames, 16)``.  Using ``n_frames`` alone would send short
    utterances' sentinel frames into the low-distance buckets.
    """
    n = max(0, int(n_frames))
    sentinel = max(n, 16)
    dist = np.full(n, sentinel, dtype=np.int64)
    if n == 0:
        return dist
    switches = sorted({int(s) for s in (switch_frames or []) if 0 <= int(s) < n})
    if not switches:
        return dist
    last = np.zeros(n, dtype=np.int64)
    for s in switches:
        last[s:] = s
    dist = np.arange(n, dtype=np.int64) - last
    dist[: switches[0]] = sentinel
    return dist
```

3b. Update `test_switch_distances_and_buckets` (`tests/test_evaluation_suite.py:606-616`):

- Keep the `n=20` assertions unchanged (`sentinel = max(20, 16) = 20`).
- Replace line 614 (`assert (R.switch_distances(10, []) == 10).all()`) with:

```python
    # short utterance: sentinel is max(n, 16) so it cannot land in a low bucket
    assert (R.switch_distances(10, []) == 16).all()
    short = R.switch_distances(10, [3])
    assert short[0] == 16 and short[3] == 0
    assert ((short >= 16) & (short < 10 ** 9)).sum() == 3  # frames 0-2
```

**Verify**: `movio/.venv/bin/python -m pytest tests/test_evaluation_suite.py -q -p no:cacheprovider`
→ all pass, including `test_switch_distances_and_buckets`,
`test_frame_accept_arrays_requires_valid_bits` and
`test_switch_distance_table_aggregates_and_pairs` (none of which change
behaviour for the 172-frame fixtures).

### Step 4: Min-τ fallback + warning + `gate_threshold_used` (C08)

4a. Replace `run_full_evaluation.py:1591-1604` with:

```python
    for cfg in configs:
        ok = [r for r in rows if r.get("config") == cfg and not r.get("error")]
        err = [r for r in rows if r.get("config") == cfg and r.get("error")]
        used_threshold = float(gate_threshold)
        if cfg == "C" and ok:
            present = sorted({float(r.get("gate_threshold") or 0.0) for r in ok})
            if used_threshold not in present:
                used_threshold = present[0]  # smallest measured tau; never pool tau values
                print(f"[Gate] WARNING: Table I requested tau={float(gate_threshold):.2f} "
                      f"but no config-C rows exist at that threshold; using "
                      f"tau={used_threshold:.2f} (measured: {present}). Refresh rows with "
                      f"--force or pass a matching --gate-thresholds to get the requested tau.")
            if any(t != used_threshold for t in present):
                ok = [r for r in ok
                      if float(r.get("gate_threshold") or 0.0) == used_threshold]
                err = [r for r in err
                       if float(r.get("gate_threshold") or 0.0) == used_threshold]
        if not ok:
            agg[cfg] = {"n": 0, "errors": len(err)}
            continue
```

4b. Add the used-threshold field next to the existing requested one in the
`agg[cfg]` dict (`:1674`, currently
`"gate_threshold": float(gate_threshold) if cfg == "C" else None,`):

```python
            "gate_threshold": float(gate_threshold) if cfg == "C" else None,
            "gate_threshold_used": used_threshold if cfg == "C" else None,
```

4c. Update `test_gate_threshold_filtering_in_aggregate`
(`tests/test_evaluation_suite.py:794-803`) to pin the new contract:

```python
def test_gate_threshold_filtering_in_aggregate(capsys):
    rows = _fabricated_rows()
    default = R.aggregate_results(rows, ["C"], gate_threshold=0.0)
    gated = R.aggregate_results(rows, ["C"], gate_threshold=0.5)
    assert default["C"]["n"] == 3          # un-gated rows only
    assert default["C"]["gate_threshold_used"] == pytest.approx(0.0)
    assert gated["C"]["n"] == 3            # 2 speculative + 1 fallback row
    assert gated["C"]["fallback_utts"] == 1
    assert gated["C"]["gate_threshold_used"] == pytest.approx(0.5)
    # a threshold with no rows falls back to the smallest measured tau, warns,
    # and never pools different taus into one row
    fallback = R.aggregate_results(rows, ["C"], gate_threshold=0.9)
    assert fallback["C"]["n"] == 3
    assert fallback["C"]["gate_threshold_used"] == pytest.approx(0.0)
    assert "WARNING" in capsys.readouterr().out
```

(The fixture has C rows only at τ ∈ {0.0, 0.5}; the old behaviour returned all
6 rows pooled for τ=0.9.)

**Verify**: `movio/.venv/bin/python -m pytest tests/test_evaluation_suite.py -q -p no:cacheprovider`
→ all pass. Then confirm no other test reads the pooled fallback:
`grep -n "gate_threshold=0.9" tests/test_evaluation_suite.py` → only the
updated test.

### Step 5: Full suite, runbook count, index

5a. Run the full suite:
`movio/.venv/bin/python -m pytest tests/ -q -p no:cacheprovider` → expect
`99 passed` (98 + the new `test_table_five_cb0_reports_std`).

5b. Update the count in `KAGGLE_RUN.md:427` — currently:
`* Local test suite: \`python -m pytest tests/ -q\` (98 tests, no GPU and no network needed).`
Replace only the number `98` with the exact count from the pytest summary line
(99). Do not touch anything else in the file.

5c. Update the `001` row in `plans/README.md` (status → DONE, or BLOCKED +
one-line reason). If the file does not exist yet, skip and note it in your
report.

**Verify**: `grep -n "tests)" KAGGLE_RUN.md` → line shows the new count;
`git status --porcelain` → only `run_full_evaluation.py`,
`tests/test_evaluation_suite.py`, `KAGGLE_RUN.md` modified.

## Test plan

All in `tests/test_evaluation_suite.py` (module imports `run_full_evaluation as R`):

1. **Updated** `test_aggregation_and_zone_weighting` — cold TTFA now a
   `_stats` dict (`n == 1`, `p50 == 260.0`; e2e `p50 == 300.0`).
2. **New** `test_table_five_cb0_reports_std` — `build_table_V` renders
   `45.0$\pm$5.0` from `_stats([40.0, 50.0])` (regression for the always-±0.0
   bug).
3. **Updated** `test_switch_distances_and_buckets` — sentinel is
   `max(n, 16)`; short utterances land in "16+".
4. **Updated** `test_gate_threshold_filtering_in_aggregate` — min-τ fallback,
   `WARNING` printed, `gate_threshold_used` recorded, no pooling (n == 3, not 6).
5. Untouched and must keep passing: `tests/test_wandb_push.py` entire file,
   especially `test_headline_metrics_are_flat_numbers` (`:177-194`).

Structural pattern to follow: the existing tests in
`tests/test_evaluation_suite.py` (plain functions, `pytest.approx`,
module-level fixtures like `_fabricated_rows` / `_gate_rows_fixture`).

Verification: `movio/.venv/bin/python -m pytest tests/ -q -p no:cacheprovider`
→ `99 passed`.

## Done criteria

ALL must hold:

- [ ] `movio/.venv/bin/python -m pytest tests/ -q -p no:cacheprovider` → `99 passed`
- [ ] `grep -n 'ttfa_compute_cold": first.get' run_full_evaluation.py` → no matches
- [ ] `grep -n '_mean_pm(entry.get("accept_true"' run_full_evaluation.py` → no matches
- [ ] `grep -n "dist = np.full(n, n" run_full_evaluation.py` → no matches
- [ ] `grep -n "gate_threshold_used" run_full_evaluation.py` → at least 1 match
- [ ] `grep -n "99 tests" KAGGLE_RUN.md` → exactly 1 match
- [ ] `git status --porcelain` → only the three in-scope files modified
- [ ] `plans/README.md` 001 row updated

## STOP conditions

Stop and report back (do not improvise) if:

- Any "Current state" excerpt does not match the live code (the repo has
  drifted since this plan was written — check with the drift command first).
- Your pre-edit grep in Step 1a finds a consumer of `ttfa_compute_cold` /
  `ttfa_e2e_cold` that expects a raw float and is not listed in this plan.
- `tests/test_wandb_push.py` fails after the Step 1 producer change — that
  would mean the consumer contract assumption in this plan is wrong; do not
  "fix" `push_to_wandb.py` without reporting back.
- A step's verification fails twice after a reasonable fix attempt.
- Fixing something appears to require touching `push_to_wandb.py`,
  `_session_cold`, or the `_mean_pm` helper itself.

## Maintenance notes

- If Table I ever starts printing τ (it currently prints none), it must read
  `agg[cfg]["gate_threshold_used"]`, not `gate_threshold` — the latter remains
  the *requested* value for provenance.
- The cold-TTFA keys are now `_stats` dicts; any future consumer must use
  `_stat(...)/.get("p50")`-style access. The single-value bootstrap CI fields
  are absent (ci=False) — fine; `n=1` is correct for "first row of the run".
- The `switch_distance_table` analysis depends on the sentinel contract; if a
  new bucket below 16 is ever added, revisit `sentinel = max(n, 16)`.
- Deferred out of this plan on purpose: C06 (`sosfiltfilt` guard, ~12–15
  sample inputs) and C09 (crossfade applied to contiguous chunks) are separate
  findings; they are listed as not-selected in `plans/README.md`. Also
  deferred: making `push_to_wandb` tests build their fixture from a real
  `aggregate_results` call (plan 008 reconciles that fixture).
- Reviewers should scrutinize: that the Step 4 `err` list is filtered with the
  same τ as `ok`; that `prepare`-style helpers elsewhere do not depend on
  `aggregate_results` returning raw floats for cold keys.
