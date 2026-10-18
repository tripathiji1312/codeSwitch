# Plan 005: Secrets hygiene — stop persisting/uploading the W&B API key; replace whole-directory uploads with an explicit artifact allow-list

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**: `git diff --stat d69b399..HEAD -- run_full_evaluation.py push_to_wandb.py tests/test_wandb_push.py tests/test_evaluation_suite.py KAGGLE_RUN.md CodeSwitch_TTS_Kaggle.ipynb`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P0 (credential leak; also silently uploads model weights)
- **Effort**: S–M
- **Risk**: LOW — the key value is only used for `wandb.login(key=...)`; the artifact allow-list only removes files that were never meant to ship
- **Depends on**: none (touches different regions of `push_to_wandb.py` than plan 008, which edits the train-flatten section only)
- **Category**: security
- **Planned at**: commit `d69b399`, 2026-10-19

## Why this matters

Two related trust defects in the W&B path (findings SECURITY-01 and SECURITY-04
in `plans/research/findings-2026-10-18.md`, aspect 2):

1. **The W&B API key is written to disk and uploaded.** Passing
   `--wandb-api-key` (documented in `run_full_evaluation.py:3125-3127` and
   `KAGGLE_RUN.md`) puts the cleartext key into `run_meta.json`
   (`run_full_evaluation.py:3287-3288` → `:3306`), into the W&B run config
   (`push_to_wandb.py:468`) and into the downloadable artifact
   (`push_to_wandb.py:501`). Anyone with project access, or a shared artifact
   link, can read the credential. The env-var/Kaggle-secret paths are safe;
   only the CLI path leaks, but that is precisely the path the runbook
   recommends as a fallback.
2. **The artifact uploads entire directories.** `push_to_wandb.py:503-504`
   builds `dirs = [figures_dir, ckpt_dir, audio_dir]` and `:637-639` calls
   `artifact.add_dir(...)` on each, so `.pt` weights (tens of MB, plus the
   leaked `run_meta.json`) leave the machine — while the module docstring
   (`:13`) promises "ckpt meta" only, and the notebook's own manual zip
   deliberately excludes `codeswitch_ckpt/*.pt` (cell 24, `-x
   'codeswitch_ckpt/*.pt'`). Stray working files placed in those directories
   also ride along.

Neither defect breaks any functionality; both are fail-closed fixes.

## Current state

Files involved:

- `run_full_evaluation.py` — evaluation harness; builds and writes `run_meta.json`.
- `push_to_wandb.py` — W&B pusher (stdlib-only at import; `wandb` is lazy).
- `tests/test_wandb_push.py` — offline pusher tests with a `_FakeWandb` stub.
- `tests/test_evaluation_suite.py` — offline eval tests (imports `run_full_evaluation as R`).
- `KAGGLE_RUN.md` — runbook; artifact description (:253, :287-288), test count (:427).
- `CodeSwitch_TTS_Kaggle.ipynb` — markdown cells 21 and 23 describe the artifact bundle.

### Defect 1 (SECURITY-01): the key reaches `run_meta.json` → W&B config → artifact

Producer — `run_full_evaluation.py:3287-3288`:

```python
    run_meta = {
        "args": vars(args),
```

`vars(args)` includes `wandb_api_key` when the user passed `--wandb-api-key`
(parsed at `:3125-3127`). The dict is written to `results.parent/run_meta.json`
at `:3306-3308`.

Consumer — `push_to_wandb.py:468`:

```python
        config["eval_args"] = (run_meta.get("args") or {})
```

and `push_to_wandb.py:499-502` (artifact file list):

```python
    artifacts: List[Path] = [path for path in [
        figures_dir / "summary.json", figures_dir / "tables.tex",
        results, results.parent / "run_meta.json", ckpt_dir / "meta.json",
    ] if path.is_file()]
```

The key value itself is only ever used at `push_to_wandb.py:579-581`
(`resolve_api_key(...)` → `wandb.login(key=key, relogin=True)`), so removing it
from the serialized dicts changes nothing observable.

### Defect 2 (SECURITY-04): whole directories are uploaded

`push_to_wandb.py:503-504`:

```python
    dirs = [path for path in ([figures_dir, ckpt_dir, audio_dir] if upload_artifact else [])
            if path is not None and Path(path).is_dir()]
```

carried in the plan dict at `:515` (`"artifact_dirs": dirs`), printed in the
dry-run at `:565-566`, and uploaded at `:637-639`:

```python
        for path in plan["artifact_dirs"]:
            if Path(path).is_dir():
                artifact.add_dir(str(path))
```

Verified consumers of `artifact_dirs` (grep): only `push_to_wandb.py`
(`:515`, `:565`, `:637`) and `tests/test_wandb_push.py:293`. Nothing else reads it.

### What the fix must keep green

- `tests/test_wandb_push.py:257` — `summary.json` stays in `report["artifact_files"]`.
- `tests/test_wandb_push.py:292` — `raw_results.jsonl` stays in the artifact files.
- `tests/test_wandb_push.py:285` — 3 audio items logged with `max_audio=1` (logging path unchanged).

## Steps

### Step 0 — Manual action: rotate any W&B key that was passed via the CLI (human, not a code change)

Check whether any run used the CLI key path, then rotate:

```bash
cd /home/tripathiji/projects/all_work
ls -1 eval_results*/run_meta.json 2>/dev/null; git grep -n "wandb-api-key\|api-key" -- KAGGLE_RUN.md CodeSwitch_TTS_Kaggle.ipynb | head
```

If any `run_meta.json` exists on disk from a run that used `--wandb-api-key`
(open one and look for a non-`None` `args.wandb_api_key` — **do not paste the
value anywhere**), or if such a run was ever pushed to W&B:

1. Revoke + regenerate the key in the W&B account settings.
2. Delete the affected W&B run/artifact in the W&B UI (the key is inside the
   run config and the artifact).

Record the outcome (done / not applicable) in the PR description. This step is
outside the repo; the code fix below only stops *future* leaks.

### Step 1 — Producer fix: redact credential-looking args before `run_meta` is serialized

In `run_full_evaluation.py`, insert after `package_versions()` (which ends at
`:246`, immediately before `def digits_to_words_digitwise` at `:249`):

```python
#: substring match on the lowercase argparse dest — matching options never reach run_meta.json
_SECRET_ARG_HINTS = ("api_key", "token", "secret", "password")


def _run_meta_args(args: argparse.Namespace) -> Dict[str, Any]:
    """``vars(args)`` with credential-looking options redacted (SECURITY-01)."""
    return {key: ("<redacted>" if any(hint in key.lower() for hint in _SECRET_ARG_HINTS)
                  else value)
            for key, value in vars(args).items()}
```

Then change `run_full_evaluation.py:3288` from:

```python
        "args": vars(args),
```

to:

```python
        "args": _run_meta_args(args),
```

Note: `"wandb_api_key"` contains the substring `"api_key"`, so it is redacted;
`--api-key`'s dest would be too. `argparse`, `Dict` and `Any` are already
imported in this module.

**Verify**

```bash
cd /home/tripathiji/projects/all_work
git grep -n "vars(args)" run_full_evaluation.py        # expect exactly one hit: inside _run_meta_args
movio/.venv/bin/python -c "import ast, pathlib; ast.parse(pathlib.Path('run_full_evaluation.py').read_text()); print('syntax ok')"
```

(Two `vars(args)` hits is fine if the second is the `for ... in vars(args).items()`
line inside the new helper; there must be **no** remaining `"args": vars(args)`.)

### Step 2 — Consumer fix: scrub `eval_args` (defense in depth) and replace `add_dir` with an allow-list

All edits in `push_to_wandb.py`:

**2a.** In the `typing` import (line 33), add `Mapping`:

```python
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
```

**2b.** Insert after `sanitize_key()` (ends at `:107`, immediately before
`def _num` at `:110`):

```python
#: substring match on the lowercase key — anything matching is redacted before upload
_SECRET_ARG_HINTS = ("api_key", "token", "secret", "password")


def _scrub_secrets(mapping: Any) -> Dict[str, Any]:
    """Copy of ``mapping`` with credential-looking values replaced by ``<redacted>``.

    Defense in depth for SECURITY-01: a ``run_meta.json`` produced before the
    producer-side fix may still contain the key on disk.
    """
    return {str(key): ("<redacted>" if any(hint in str(key).lower() for hint in _SECRET_ARG_HINTS)
                       else value)
            for key, value in dict(mapping or {}).items()}
```

**2c.** Change `:468` from:

```python
        config["eval_args"] = (run_meta.get("args") or {})
```

to:

```python
        config["eval_args"] = _scrub_secrets(run_meta.get("args") or {})
```

**2d.** In `build_plan`, replace `:499-504` (the `artifacts` + `dirs` block) with:

```python
    artifacts: List[Path] = [path for path in [
        figures_dir / "summary.json", figures_dir / "tables.tex",
        results, ckpt_dir / "meta.json",
    ] if path.is_file()]
    media: List[Path] = [Path(path) for _, path, _ in figures] + \
        [Path(path) for _, path, _ in audio]
```

(`run_meta.json` is deliberately gone from the list: it is duplicative
(versions/args are already in the run config) and it is the SECURITY-01 vector.
`media` adds only the *selected* figure PNGs and *selected* audio — no
directories, no weights, no stray files.)

**2e.** In the plan dict, change `:514-515` from:

```python
        "artifact_files": artifacts,
        "artifact_dirs": dirs,
```

to:

```python
        "artifact_files": artifacts + media,
```

**2f.** In the verbose print (`:521-524`), change `f"{len(artifacts) + len(dirs)} artifact entries")` to `f"{len(plan['artifact_files'])} artifact entries")`.

**2g.** In the dry-run block, delete the two lines `:565-566`:

```python
            for path in plan["artifact_dirs"]:
                print(f"        dir     {path}")
```

(the `for path in plan["artifact_files"]` loop above it already prints every entry).

**2h.** In the artifact block, delete the four lines `:637-639`:

```python
        for path in plan["artifact_dirs"]:
            if Path(path).is_dir():
                artifact.add_dir(str(path))
```

**2i.** Update the module docstring bullet `:12-13` from:

```
* a few *labelled* audio samples (best/median/worst CER per config),
* one artifact with the complete JSONL / figures / audio / ckpt meta for download.
```

to:

```
* a few *labelled* audio samples (best/median/worst CER per config),
* one artifact with an explicit allow-list only — the JSONL, ``tables.tex``,
  ``summary.json``, ``codeswitch_ckpt/meta.json``, the figure PNGs and the
  selected audio.  Model weights, ``run_meta.json`` and any other stray files
  in those directories are never uploaded.
```

and append one sentence to the key-resolution paragraph (`:19-23`):

```
Credential options are redacted before anything is serialized: the key never
lands in ``run_meta.json``, the W&B config or the artifact.
```

**Verify**

```bash
cd /home/tripathiji/projects/all_work
git grep -n "artifact_dirs\|add_dir" push_to_wandb.py      # expect: no output
movio/.venv/bin/python -c "import push_to_wandb as W; print(W._scrub_secrets({'wandb_api_key':'x','seed':1}))"
# expect: {'wandb_api_key': '<redacted>', 'seed': 1}
```

### Step 3 — Update the two affected assertions in `tests/test_wandb_push.py`

Replace `tests/test_wandb_push.py:288-293`:

```python
    # artifact bundle carries the JSONL, tables and directories
    artifact = fake.artifacts[0]
    assert artifact["name"].startswith("codeswitch-eval-")
    assert artifact["type"] == "evaluation"
    assert any(str(f).endswith("raw_results.jsonl") for f in artifact["files"])
    assert str(ws["figures"]) in artifact["dirs"] and str(ws["audio"]) in artifact["dirs"]
    assert fake.finished == 1
```

with:

```python
    # artifact bundle is an explicit allow-list: files yes, whole directories no
    artifact = fake.artifacts[0]
    assert artifact["name"].startswith("codeswitch-eval-")
    assert artifact["type"] == "evaluation"
    assert any(str(f).endswith("raw_results.jsonl") for f in artifact["files"])
    assert artifact["dirs"] == []                       # nothing is add_dir'ed any more
    assert any(str(f).endswith(".wav") for f in artifact["files"])   # selected audio rides along
    assert all(not str(f).endswith((".pt", "run_meta.json")) for f in artifact["files"])
    assert fake.finished == 1
```

**Verify**

```bash
cd /home/tripathiji/projects/all_work
movio/.venv/bin/python -m pytest tests/test_wandb_push.py -q -p no:cacheprovider
# expect: 10 passed (9 existing + 1 new added in Step 4 — run again after Step 4)
```

### Step 4 — Add three new tests

**4a.** Append to `tests/test_wandb_push.py` (after `test_push_offline_sets_wandb_mode`, before the `_FakeWandb` class):

```python
def test_eval_args_are_scrubbed_before_logging(tmp_path, monkeypatch):
    """A pre-fix run_meta.json on disk must not leak its key into the W&B config."""
    ws = _workspace(tmp_path)
    meta_path = ws["results"].parent / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["args"]["wandb_api_key"] = "sk-live-LEAK"
    meta["args"]["hf_token"] = "hf_LEAK"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake.module)
    W.push(results=ws["results"], figures_dir=ws["figures"], audio_dir=ws["audio"],
            ckpt_dir=ws["ckpt"], max_audio=0, verbose=False)
    eval_args = fake.init_kwargs["config"]["eval_args"]
    assert eval_args["wandb_api_key"] == "<redacted>"
    assert eval_args["hf_token"] == "<redacted>"
    assert eval_args["seed"] == 42 and eval_args["ckpt"] == "codeswitch_ckpt"
    assert "LEAK" not in json.dumps(fake.init_kwargs, default=str)


def test_artifact_allowlist_excludes_weights_run_meta_and_stray_files(tmp_path):
    ws = _workspace(tmp_path)
    (ws["ckpt"] / "draft_cb0.pt").write_bytes(b"\x00" * 8)          # a weight file must NOT ship
    (ws["figures"] / "stray_notes.txt").write_text("x", encoding="utf-8")  # stray file must NOT ship
    plan = W.build_plan(ws["results"], ws["figures"], ws["audio"], ws["ckpt"], verbose=False)
    files = {str(p) for p in plan["artifact_files"]}
    assert str(ws["figures"] / "summary.json") in files
    assert str(ws["figures"] / "tables.tex") in files
    assert str(ws["ckpt"] / "meta.json") in files
    assert any(f.endswith(".png") for f in files)                   # figure PNGs ride along
    assert any(f.endswith(".wav") for f in files)                   # selected audio rides along
    assert all(not f.endswith((".pt", "run_meta.json", "stray_notes.txt")) for f in files)
    assert "artifact_dirs" not in plan
```

**4b.** Append to `tests/test_evaluation_suite.py` (next to the other argparse test at `:587-593`):

```python
def test_run_meta_args_redacts_credentials():
    args = R.build_arg_parser().parse_args(["--wandb-api-key", "sk-live-secret"])
    scrubbed = R._run_meta_args(args)
    assert scrubbed["wandb_api_key"] == "<redacted>"
    assert "sk-live-secret" not in str(scrubbed)
    assert scrubbed["ckpt"] == "codeswitch_ckpt"


def test_secret_hint_lists_agree_across_modules():
    import push_to_wandb as W

    assert R._SECRET_ARG_HINTS == W._SECRET_ARG_HINTS
```

(The second test guards against the two copies of the scrub rule drifting.)

**Verify**

```bash
cd /home/tripathiji/projects/all_work
movio/.venv/bin/python -m pytest tests/test_wandb_push.py tests/test_evaluation_suite.py -q -p no:cacheprovider
# expect: all pass; test_wandb_push.py now has 11 tests, +2 in test_evaluation_suite.py
```

### Step 5 — Doc updates: `KAGGLE_RUN.md` and the notebook cells

**5a.** `KAGGLE_RUN.md:253-254` — replace:

```
All of it (plus the raw JSONL and the checkpoint meta) is mirrored into the W&B artifact by
§3b, which is what saves a run when a Kaggle session is recycled before the zip is downloaded.
```

with:

```
The JSONL, `tables.tex`, `summary.json`, `codeswitch_ckpt/meta.json`, the figure PNGs and the
selected audio are mirrored into the W&B artifact by §3b — an explicit allow-list, so model
weights and `run_meta.json` stay local. This is what saves a run when a Kaggle session is
recycled before the zip is downloaded.
```

**5b.** `KAGGLE_RUN.md:287-288` — replace:

```
* **artifact** — `codeswitch-eval-<run>` with `raw_results.jsonl`, `tables.tex`, `summary.json`,
  `run_meta.json`, the figures, the audio and `codeswitch_ckpt/meta.json`
```

with:

```
* **artifact** — `codeswitch-eval-<run>` with `raw_results.jsonl`, `tables.tex`, `summary.json`,
  `codeswitch_ckpt/meta.json`, the figure PNGs and the selected audio (explicit allow-list;
  weights are never uploaded)
```

**5c.** `KAGGLE_RUN.md:260-261` — append to the sentence after "no key ever lands in the notebook or the repo.":

```
The key is also redacted from `run_meta.json` and the W&B config, so nothing a run uploads
contains it.
```

**5d.** Notebook — edit cells 21 and 23 with this exact script (run from the repo root;
it fails loudly if a target string is missing, and writes the JSON back byte-compatible
with the current file: `indent=1, ensure_ascii=False`, trailing newline):

```python
import json, pathlib, sys

path = pathlib.Path("CodeSwitch_TTS_Kaggle.ipynb")
nb = json.loads(path.read_text(encoding="utf-8"))

def replace_in_cell(index, needles, new_line):
    src = nb["cells"][index]["source"]
    for i, line in enumerate(src):
        if line in needles:
            src[i] = new_line
            return
    sys.exit(f"cell {index}: none of {needles!r} found")

# cell 21 (§9b markdown)
replace_in_cell(21, {"* **figures** — fig1–fig5 as images (PDFs ride in the artifact)\n"},
                "* **figures** — fig1–fig5 as images; the PNGs also ride in the artifact (PDFs stay local)\n")
replace_in_cell(21, {"* **artifact** — `raw_results.jsonl`, `tables.tex`, `summary.json`, `run_meta.json`, the figures,\n"},
                "* **artifact** — `raw_results.jsonl`, `tables.tex`, `summary.json`, the checkpoint meta, the\n")
replace_in_cell(21, {"  the audio and the checkpoint meta, so a wiped Kaggle session never loses a run\n"},
                "  figure PNGs and the selected audio (allow-listed: weights and `run_meta.json` stay local),\n"
                "  so a wiped Kaggle session never loses a run\n")
# cell 23 (§10 markdown)
replace_in_cell(23, {"The same bundle (plus the raw JSONL and the checkpoint meta) lives in the W&B artifact from\n"},
                "The same bundle (plus the raw JSONL and the checkpoint meta) lives in the W&B artifact from\n")
replace_in_cell(23, {"cell 9b, so a disconnected or wiped session never loses the numbers - `wandb` keeps them\n"},
                "cell 9b (an explicit allow-list: model weights are never uploaded), so a disconnected or\n"
                "wiped session never loses the numbers - `wandb` keeps them\n")

path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("notebook updated")
```

Note: `replace_in_cell` replaces the first matching line; the third call above
replaces **one line with two lines** (list growth is fine). After running:

```bash
cd /home/tripathiji/projects/all_work
git diff --stat -- CodeSwitch_TTS_Kaggle.ipynb        # expect: 1 file, a handful of +/- lines, NOT a whole-file rewrite
git grep -c "run_meta.json" -- CodeSwitch_TTS_Kaggle.ipynb  # expect: no hits in cells 21/23 (cell 24/other text may still mention it)
```

If `git diff --stat` shows the whole notebook rewritten, STOP and report —
the serialization recipe diverged.

### Step 6 — Full suite, test-count line, final diff check

```bash
cd /home/tripathiji/projects/all_work
movio/.venv/bin/python -m pytest tests/ -q -p no:cacheprovider
```

Expected: everything passes; on a tree where only this plan has landed:
**101 passed** (98 baseline + 3 new). Read the actual number from the output
and update `KAGGLE_RUN.md:427`:

- From: `* Local test suite: `python -m pytest tests/ -q` (98 tests, no GPU and no network needed).`
- To: same line with the exact number you just observed.

Final check:

```bash
cd /home/tripathiji/projects/all_work
git status --short
git diff --stat d69b399..HEAD
```

Expected touched files: `run_full_evaluation.py`, `push_to_wandb.py`,
`tests/test_wandb_push.py`, `tests/test_evaluation_suite.py`, `KAGGLE_RUN.md`,
`CodeSwitch_TTS_Kaggle.ipynb` (plus `plans/` which is untracked).

## STOP conditions

- **Do not paste or print the leaked key value anywhere** — tests use obvious
  fakes (`sk-live-LEAK`); if you find `sk-live-LEAK` or `LEAK` asserted as
  *absent*, that is expected. If you ever see a real key value in a diff or
  command output, stop and report (do not echo it again).
- `git diff --stat` after the notebook edit shows a whole-file rewrite → STOP.
- `test_push_logs_metrics_figures_tables_audio_and_artifact` fails on
  `artifact["dirs"] == []` → check Step 2h was applied (the `add_dir` loop is
  the only caller of `add_dir`).
- Any changed behavior in `--dry-run` output beyond the removed `dir` lines →
  STOP (the file list order should stay: summary.json, tables.tex, results,
  ckpt meta.json, then figures, then audio).
- Full suite count other than expected → report the delta; do not "fix" tests
  that measure behavior not covered by this plan.

## Maintenance notes

- The scrub rule now exists in **two** modules (`run_full_evaluation.py` +
  `push_to_wandb.py`) by design (the pusher must stay import-independent of the
  eval harness). The consistency test added in Step 4b fails if they drift.
- `--api-key` on `push_to_wandb.py` still works for login — it is simply never
  serialized. `KAGGLE_RUN.md:411`'s suggestion to pass `--api-key` remains valid.
- If a future artifact file is added, add it to the explicit list in
  `build_plan` (Step 2d) — never reintroduce `add_dir`.
- Related prompt for a future session (not part of this plan): SECURITY-02
  (pin `revision=` for the four Hub loaders) and SECURITY-03 (`weights_only`)
  are separate plans; SECURITY-03's `torch>=2.6` requirement lands in plan 006/007.
