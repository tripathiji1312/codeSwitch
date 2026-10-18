# Plan 007: Unify the dependency declarations — complete `requirements.txt`, add `requirements-dev.txt`, align the Kaggle install cells, add a startup dependency report

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. **Do not run `pip install`** — the local venv has
> no `pip` module, and installing into Kaggle is out of scope. All verification
> here is file-text + pytest in the existing venv. When done, update the status
> row for this plan in `plans/README.md` — unless a reviewer dispatched you
> and told you they maintain the index.
>
> **Drift check (run first)**: `git diff --stat d69b399..HEAD -- requirements.txt requirements-dev.txt run_full_evaluation.py KAGGLE_RUN.md CodeSwitch_TTS_Kaggle.ipynb tests/test_evaluation_suite.py`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition. **Plan 006 must have landed first** —
> this plan assumes `requirements.txt:4` already reads `torch>=2.6.0`:
> `grep -n 'torch>=' requirements.txt` must print `torch>=2.6.0`; otherwise STOP
> and run plan 006 first.

## Status

- **Priority**: P2
- **Effort**: S–M
- **Risk**: LOW–MED — the Kaggle side only *removes* two never-imported packages and gains no new installs; the local side is not executable here (no pip), so the pin choices are validated by text tests and the runbook cross-references
- **Depends on**: 006 (torch floor in `requirements.txt`)
- **Category**: dependencies / reproducibility
- **Planned at**: commit `d69b399`, 2026-10-19

## Why this matters

Findings DEPS-02, DEPS-03, DEPS-04 and DEBT-07 (`plans/research/findings-2026-10-18.md`, aspects 2 and 4): the three dependency declarations disagree — `requirements.txt` says `transformers>=4.46.1` and unbounded `protobuf>=5.29.5` while the Kaggle cells pin `transformers==4.46.1` and install `protobuf==5.29.5` LAST by ritual; the manifest omits five modules the first-party code imports (`parler_tts`, `jiwer`, `matplotlib`, `datasets`, PyYAML — the last has a fallback parser; `wandb` is deliberately optional); and `pip install -r requirements.txt` therefore yields a session where CER, WER and every figure silently disappear (each import is inside a `try/except`, so the failure mode is an empty column, not a crash). There is no dev manifest at all, so the documented test command cannot be installed from the repo.

This plan makes the manifest describe reality: exact pins where the Kaggle flow needs them, the missing direct imports added, a `requirements-dev.txt` for the suite, the Kaggle cells trimmed to match, and one startup line in the eval script that says which optional features are missing before 40 minutes of GPU time are spent.

## Current state

Files involved:

- `requirements.txt` (13 lines, full contents):
  ```
  # CodeSwitch-TTS Dependencies
  # Optimized for Python 3.10 / 3.11 with CUDA 12 on NVIDIA T4 (Kaggle GPU)

  torch>=2.2.0
  torchaudio>=2.2.0
  transformers>=4.46.1
  descript-audio-codec>=1.0.0
  soundfile>=0.12.1
  scipy>=1.11.0
  numpy>=1.24.0,<2.0.0
  accelerate>=0.28.0
  sentencepiece>=0.2.0
  protobuf>=5.29.5
  ```
  (After plan 006, line 4 reads `torch>=2.6.0`.)
- `KAGGLE_RUN.md:64-73` — the setup cell's staged install (stage 3b is
  `pip install -q jiwer matplotlib seaborn pandas`).
- `CodeSwitch_TTS_Kaggle.ipynb` cell 6 — the same staged install, same
  stage-3b line `os.system("pip install -q jiwer matplotlib seaborn pandas")`.
- `run_full_evaluation.py:6` — a docstring quickstart that repeats the same
  install line: `!pip install -q jiwer matplotlib seaborn pandas`.
- `tests/test_evaluation_suite.py` — imports `json`, `math`, `pathlib.Path`,
  `numpy`, `pytest` at the top (`:15-20`); `import run_full_evaluation as R` further down.

Verified facts used below (grep, 2026-10-19): `seaborn`/`pandas` are imported
nowhere in first-party code — the only occurrences are the install lines above
plus docstring/comments; `datasets` is imported only inside
`build_codeswitch_corpus.py:223` (Wikipedia streaming, inside `try/except`);
`jiwer` is imported inside `ASREvaluator.score` (`run_full_evaluation.py:1245`,
`try/except` → returns `(None, None)`); `matplotlib` is used by the figure
functions via `_ieee_style` (`run_full_evaluation.py:1989`) inside `try/except`
per figure; the Kaggle setup cells deliberately do **not** install torch
(platform image owns it) — keep it that way.

## Steps

### Step 1 — Rewrite `requirements.txt` (full-file replacement)

Replace the entire file with:

```
# CodeSwitch-TTS Dependencies
# Optimized for Python 3.10 / 3.11 with CUDA 12 on NVIDIA T4 (Kaggle GPU)

torch>=2.6.0
torchaudio>=2.2.0
# parler-tts requires transformers>=4.46.1,<=4.46.1: the == pin is the whole
# working range (a bare >= floor resolves to 5.x and breaks the Parler API).
transformers==4.46.1
descript-audio-codec>=1.0.0
soundfile>=0.12.1
scipy>=1.11.0
numpy>=1.24.0,<2.0.0
accelerate>=0.28.0
sentencepiece>=0.2.0
# protobuf MUST stay ==5.29.5 and be installed LAST (see KAGGLE_RUN.md §1):
# parler-tts -> descript-audiotools pins protobuf<5 and silently downgrades it
# to 4.25.9, which breaks the Parler import (no google.protobuf.runtime_version).
protobuf==5.29.5
# Evaluation + figure extras: without them CER/WER and every figure are silently
# absent (each import site is inside try/except).
jiwer>=3.0.0
matplotlib>=3.7.0
# Wikipedia corpus source (build_codeswitch_corpus.py; optional at runtime)
datasets>=2.0.0
```

Notes: `torch>=2.6.0` is plan 006's change (SECURITY-03) — copy it into your
rewrite exactly; if it is missing, STOP (wrong order). Keep line endings LF.

**Verify**

```bash
cd /home/tripathiji/projects/all_work
grep -c "==" requirements.txt                     # expect 3 (transformers, protobuf, + none else)
grep -n "torch>=\|transformers==\|protobuf==\|jiwer\|matplotlib\|datasets" requirements.txt
```

### Step 2 — Add `requirements-dev.txt`

Create a new file `requirements-dev.txt`:

```
# Local development / CI: the paper+evaluation surface of the repo.
# Install:  python -m pip install -r requirements-dev.txt
# (The repo's own venv is uv-managed and ships without pip; see README.md.)
-r requirements.txt
pytest>=7.0
```

**Verify**

```bash
cd /home/tripathiji/projects/all_work
test -f requirements-dev.txt && head -5 requirements-dev.txt
```

### Step 3 — Trim the install lines in the two Kaggle cells and the eval docstring

**3a.** `KAGGLE_RUN.md:69` — replace the line

```python
    os.system("pip install -q jiwer matplotlib seaborn pandas")
```

with

```python
    os.system("pip install -q jiwer matplotlib")   # seaborn/pandas: nothing imports them
```

**3b.** Notebook cell 6 — replace the single source line

```
"    os.system(\"pip install -q jiwer matplotlib seaborn pandas\")\n",
```

with

```
"    os.system(\"pip install -q jiwer matplotlib\")   # seaborn/pandas: nothing imports them\n",
```

Use the shared notebook-edit recipe (JSON load → replace the exact line in
`cells[6]["source"]` → `path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")`).
Fail loudly (assert the needle is present) rather than writing a wrong file.

**3c.** `run_full_evaluation.py:6` — replace the docstring line

```
    !pip install -q jiwer matplotlib seaborn pandas
```

with

```
    !pip install -q jiwer matplotlib           # versions/pins: see requirements.txt
```

**Verify**

```bash
cd /home/tripathiji/projects/all_work
git grep -n "seaborn\|pandas" -- run_full_evaluation.py KAGGLE_RUN.md CodeSwitch_TTS_Kaggle.ipynb
# expect: no hits in install lines (a prose mention inside push_to_wandb.py is OK, different file)
git diff --stat -- CodeSwitch_TTS_Kaggle.ipynb      # expect 1 file, ~1 line changed, not a rewrite
```

### Step 4 — Startup dependency report in `run_full_evaluation.py`

Insert after `package_versions()` (ends at `:246`, before `def digits_to_words_digitwise` at `:249`):

```python
#: Optional packages whose absence silently removes part of the output
_OPTIONAL_DEPS = (
    ("jiwer", "CER/WER"),
    ("matplotlib", "figures"),
    ("soundfile", "wav IO"),
    ("datasets", "Wikipedia corpus source"),
    ("wandb", "W&B push"),
)


def dependency_report() -> str:
    """One line listing optional dependencies and what their absence costs.

    Every listed import site is wrapped in ``try/except`` (silent degradation
    by design); this line makes the degradation visible before the run starts.
    """
    import importlib.util

    parts = []
    for name, effect in _OPTIONAL_DEPS:
        if importlib.util.find_spec(name) is None:
            parts.append(f"{name} MISSING -> no {effect}")
        else:
            parts.append(f"{name} ok")
    return " | ".join(parts)
```

Then in `main()` (starts at `:3185`), add as the first statement after
`args = build_arg_parser().parse_args(argv)`:

```python
    print(f"[Deps] {dependency_report()}")
```

(The gate-specific `[Gate]` prints stay where they are.)

**Verify**

```bash
cd /home/tripathiji/projects/all_work
movio/.venv/bin/python -c "import run_full_evaluation as R; print(R.dependency_report())"
# expect a single line containing 'jiwer ok' and 'matplotlib ok' (both installed in this venv)
```

### Step 5 — Tests

**5a.** Append to `tests/test_evaluation_suite.py`:

```python
def test_dependency_report_lists_optional_features():
    report = R.dependency_report()
    assert "jiwer" in report and "matplotlib" in report and "wandb" in report


def test_requirements_manifest_covers_eval_dependencies():
    root = Path(__file__).resolve().parents[1]
    lines = [l.strip() for l in (root / "requirements.txt").read_text(encoding="utf-8").splitlines()]
    specs = [l for l in lines if l and not l.startswith("#")]
    assert "transformers==4.46.1" in specs
    assert "protobuf==5.29.5" in specs
    assert "jiwer>=3.0.0" in specs and "matplotlib>=3.7.0" in specs
    assert len(specs) == len(set(specs))          # no duplicated specifier
    dev = (root / "requirements-dev.txt").read_text(encoding="utf-8")
    assert "-r requirements.txt" in dev and "pytest" in dev
```

**Verify**

```bash
cd /home/tripathiji/projects/all_work
movio/.venv/bin/python -m pytest tests/test_evaluation_suite.py -q -p no:cacheprovider -k "dependency or requirements"
# expect: 2 passed
```

### Step 6 — Full suite + test-count line + final diff

```bash
cd /home/tripathiji/projects/all_work
movio/.venv/bin/python -m pytest tests/ -q -p no:cacheprovider
```

Expected: all pass; on a tree where only this plan has landed: **100 passed**
(98 baseline + 2 new). Read the actual number and update `KAGGLE_RUN.md:427`
(`* Local test suite: ... (98 tests, ...)`) to the exact count.

```bash
git status --short
git diff --stat d69b399..HEAD
```

Expected touched files: `requirements.txt`, `requirements-dev.txt` (new),
`run_full_evaluation.py`, `KAGGLE_RUN.md`, `CodeSwitch_TTS_Kaggle.ipynb`,
`tests/test_evaluation_suite.py`.

## STOP conditions

- `grep -n 'torch>=' requirements.txt` does not print `torch>=2.6.0` → plan
  006 has not landed; STOP and run it first.
- Any other executor already rewrote `requirements.txt` (drift check) → STOP
  and reconcile by hand; do not union two rewrites.
- The notebook diff is a whole-file rewrite → STOP (wrong serialization).
- A test fails that exercises `datasets`/`wandb` presence — those are
  environment-dependent; the plan's tests only assert the *report contains the
  names*, never that a package is installed or missing.
- Do **not** install anything to "verify" the new manifest; offline text
  checks + pytest are the contract.

## Maintenance notes

- After plan 010, `README.md` documents `requirements-dev.txt` as the local
  install; keep the two files consistent if either changes.
- The `protobuf==5.29.5` pin encodes the Kaggle staged-install knowledge
  (DEPS-03). If a future Kaggle image ships protobuf >= 5.26 natively, the pin
  can relax to `>=5.29.5,<7` — but that is a separate, tested change.
- `clone_and_setup.sh` is deliberately **not** touched here (plan 010 aligns
  it with this manifest).
- `datasets` in `requirements.txt` is a floor only; Kaggle's image already
  ships it (comment at `build_codeswitch_corpus.py:224`).
