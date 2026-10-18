# Plan 010: Onboarding — README.md, AGENTS.md, .env.example, archival banners, and an honest `clone_and_setup.sh`

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**: `git diff --stat d69b399..HEAD -- README.md AGENTS.md .env.example clone_and_setup.sh REPORT_MOVIO2_PARLER.md REPORT_MOVIO_INDICF5.md requirements.txt requirements-dev.txt`
> If anything in scope changed since this plan was written, compare the
> "Current state" excerpts against the live files before proceeding (locate
> by content, not line numbers). If a quoted snippet is gone or changed,
> treat it as a STOP condition.
>
> **Dependency gates (plans 006/007)** — all must hold before starting:
> - `test -f requirements-dev.txt && head -3 requirements-dev.txt` → file
>   exists and starts with the comment block from plan 007.
> - `grep -n 'torch>=' requirements.txt` → prints `torch>=2.6.0`.
> - `grep -n "ed25eacbfa42b99156c36ebec67a733b5dbb9b79" run_full_evaluation.py`
>   → hits (plan 006's UTMOS hub pin).
> - `grep -n "revision=" codeswitch_tts/engine/speculative_engine.py codeswitch_tts/utils/audio.py`
>   → hits in both files (plan 006's parler/DAC pins).
>
> If any gate fails → STOP and run `plans/006-supply-chain-pinning.md` and/or
> `plans/007-dependency-declaration-unification.md` first. The README below
> states these pins; creating it before 006/007 would document things that
> do not exist.
>
> **Do not run `pip install` or `bash clone_and_setup.sh`** — the local venv
> has no `pip`, and executing installs is out of scope. Script verification
> is `bash -n` (syntax) + file-text checks only.

## Status

- **Priority**: P2
- **Effort**: S–M
- **Risk**: LOW — four new text files, two banner insertions, one script
  rewrite; nothing executable is run against the network
- **Depends on**: `plans/007-dependency-declaration-unification.md`
  (hard gates above)
- **Category**: docs / DX
- **Planned at**: commit `d69b399`, 2026-10-19

## Why this matters

Audit findings DOCS-08 and DOCS-09, plus the setup-script defects verified
in the 2026-10-19 entry:

- **There is no README.md, AGENTS.md, `.env.example`, CONTRIBUTING.md or
  LICENSE** (verified: none of these files exist at the repo root). A
  newcomer — or an AI agent — entering this repo finds 40 files, two 90 KB
  reports with no archival marker, a Kaggle runbook, and no entry point.
  `requirements.txt` alone cannot install what the tests need (plan 007
  fixes the manifest; this plan writes the map).
- **The two technical reports sit beside the live runbook with no
  "superseded" marker** (`REPORT_MOVIO2_PARLER.md:3` "Project folder:
  `movio-2/`"; `REPORT_MOVIO_INDICF5.md:3` "Project folder: `movio/`"; both
  trees are `.gitignore`d and not shipped). Report-stated facts get absorbed
  as current (e.g. the gating claim, corrected by plan 011).
- **`clone_and_setup.sh` ends in a failure on a fresh clone.** Its last line
  is unconditionally `python kaggle_demo.py` (`:55`), but with no checkpoint
  present the demo's own guard aborts unless `CODESWITCH_ALLOW_UNTRAINED=1`
  is set (verified: `kaggle_demo.py:75-77`). It also installs into the
  system Python (no venv) and knows nothing about `requirements-dev.txt`.
  The script is referenced by nothing (the notebook does not call it; grep
  returns zero references) — it is a local convenience entry point, so this
  plan aligns it with the README flow instead of deleting it.

## Current state

- Repo root inventory (verified): no `README.md`, `AGENTS.md`,
  `.env.example`, `CONTRIBUTING.md`, `LICENSE`; present:
  `.gitignore`, `CodeSwitch_TTS_Kaggle.ipynb`, `KAGGLE_RUN.md`,
  `PAPER_CLAIMS_AND_EVIDENCE.md`, the two `REPORT_*.md`, `build_codeswitch_corpus.py`,
  `clone_and_setup.sh`, `codeswitch_benchmark_data.py`, `codeswitch_tts/`,
  `kaggle_demo.py`, `push_to_wandb.py`, `requirements.txt`,
  `run_full_evaluation.py`, `tests/`, `train_draft_distill.py` (+ untracked
  `plans/`).
- `clone_and_setup.sh` (55 lines, complete): header `:1-5`, repo defaults
  `:7-11`, clone/pull `:20-28`, staged install `:31-46` (`pip install -q -U
  git+https://github.com/huggingface/parler-tts.git` → `transformers==4.46.1`
  → `-r requirements.txt` → `protobuf==5.29.5` last), version print `:48-49`,
  smoke `:51-55` ending `python kaggle_demo.py`.
- `requirements-dev.txt` (created by plan 007):
  ```
  # Local development / CI: the paper+evaluation surface of the repo.
  # Install:  python -m pip install -r requirements-dev.txt
  # (The repo's own venv is uv-managed and ships without pip; see README.md.)
  -r requirements.txt
  pytest>=7.0
  ```
- `kaggle_demo.py:75-77` env guard (`CODESWITCH_ALLOW_UNTRAINED=1`) and
  `:218` `All audio files are saved in: ./output_audio/`.
- Reports' first lines: `REPORT_MOVIO2_PARLER.md:1` title, `:3`
  `**Project folder:** \`movio-2/\``; `REPORT_MOVIO_INDICF5.md:1` title,
  `:3` `**Project folder:** \`movio/\``.
- Test command convention: `python -m pytest tests/ -q` (documented at
  `KAGGLE_RUN.md:427`); in the maintainer checkout the interpreter is
  `movio/.venv/bin/python` and the `pytest` console script must not be used
  (broken shebang).

## Commands you will need

| Purpose | Command | Expected |
|---|---|---|
| Script syntax | `bash -n clone_and_setup.sh` | exit 0, no output |
| File presence | `ls README.md AGENTS.md .env.example` | all three listed |
| Banners | `sed -n '1,9p' REPORT_MOVIO2_PARLER.md REPORT_MOVIO_INDICF5.md` | banner present after each title |
| Suite unchanged | `movio/.venv/bin/python -m pytest tests/ -q -p no:cacheprovider` | same count as before this plan (no tests added) |

## Scope

**In scope**: create `README.md`, `AGENTS.md`, `.env.example`; prepend
archival banners to the two report files; rewrite `clone_and_setup.sh`.

**Out of scope**: any code change; LICENSE/CONTRIBUTING (README states the
license situation factually instead); executing the script; touching
`KAGGLE_RUN.md` (plan 011 owns its truth pass); touching `requirements*.txt`.

## Git workflow

Branch: `advisor/010-dx-onboarding`. Commits:

1. `Add README with the repo map, quickstart and security notes`
2. `Add AGENTS.md and .env.example`
3. `Mark the two experiment reports as archival`
4. `align clone_and_setup.sh with the venv flow and the allow-untrained smoke`

## Steps

### Step 1 — Create `README.md`

Create the file with exactly this content (adjust nothing silently — if a
statement is wrong against the live tree, STOP and report):

````markdown
# CodeSwitch-TTS

Hybrid non-autoregressive draft + autoregressive verification TTS for
code-mixed Tamil–English (Tanglish) speech: a draft transformer proposes
Codebook-0 tokens in one pass, a residual decoder predicts Codebooks 1–8 in
parallel, and a teacher-forced scorer verifies CB0 with speculative
rejection sampling. The evaluation harness produces the paper artifacts
(Fig. 1–5, Tables I–V).

## Repository map

| Path | What it is |
|---|---|
| `codeswitch_tts/` | The package: models, engine, frontend normalizer, audio utils, packaged assets |
| `train_draft_distill.py` | Teacher capture (content-addressed shards) + draft/residual distillation; `--dry-run` smoke |
| `run_full_evaluation.py` | 60-utterance evaluation: Configs A/B/C, gate sweep, CER (+ Latin transliteration), distance-to-switch, figures/tables |
| `push_to_wandb.py` | Optional W&B push: scalars, figures, tables, audio, `codeswitch-eval-*` artifact |
| `build_codeswitch_corpus.py` | Register-balanced corpus builder (templates + Wikimedia stream + `--extra-file`/`--json-out`) |
| `kaggle_demo.py` | Quick demo (needs a trained checkpoint, or `CODESWITCH_ALLOW_UNTRAINED=1` for a wiring smoke) |
| `KAGGLE_RUN.md` | The runbook: single-GPU Kaggle workflow, budgets, troubleshooting |
| `PAPER_CLAIMS_AND_EVIDENCE.md` | Claim-to-evidence mapping and the reviewer-risk table |
| `tests/` | Pytest suite (no GPU, no network) |
| `plans/` | Implementation plans and research logs from the 2026-10 audit |

## Quickstart (Linux; CPU is enough for tests)

```bash
git clone https://github.com/tripathiji1312/codeSwitch.git
cd codeSwitch
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt          # runtime (torch, transformers, DAC, ...)
pip install -r requirements-dev.txt      # optional: pytest for the test suite

python -m pytest tests/ -q               # test suite: no GPU, no network

python train_draft_distill.py --dry-run  # trainer smoke: fabricates shards,
                                         # writes codeswitch_ckpt/*

CODESWITCH_ALLOW_UNTRAINED=1 python kaggle_demo.py   # demo wiring smoke
                                         # (random weights; audio is NOT speech)
```

For the full pipeline on a Kaggle GPU, follow `KAGGLE_RUN.md` — it carries
the staged install (which exists to survive the Parler-TTS/protobuf
conflict) and the measured budgets (~7.9 s/sentence teacher capture, ~4.5 h
for 2000 lines).

## Security notes

- **Never commit credentials.** `WANDB_API_KEY` is read from the environment
  or a Kaggle secret; `HF_TOKEN` (environment or Kaggle secret) is required
  to download the gated `ai4bharat/indic-parler-tts`. Rotate any key that
  ever appeared in a file, log, or artifact.
- **Hub downloads are revision-pinned** (Parler model + prompt tokenizer,
  DAC codec, UTMOS) and checkpoints are loaded with `weights_only=True`.
  Known gap: the `parler-tts` pip install from Git is **not** pinned (see
  `KAGGLE_RUN.md` §1 and `clone_and_setup.sh`); the UTMOS transformers
  mirror is opt-in.
- The archival reports below describe retired experiments; their numbers are
  historical and not part of the current pipeline.

## Archive

`REPORT_MOVIO2_PARLER.md` and `REPORT_MOVIO_INDICF5.md` are historical
technical reports for the retired `movio-2/` and `movio/` experiment trees
(gitignored, not shipped in this repository). They are kept for provenance
only — read `KAGGLE_RUN.md` for anything that must be true today.

## License

No `LICENSE` file is present in this repository.
````

**Verify**

```bash
cd /home/tripathiji/projects/all_work
grep -c "^## " README.md
grep -n "requirements-dev.txt\|CODESWITCH_ALLOW_UNTRAINED\|weights_only=True" README.md
```

Expected: `6`; and the three phrases appear.

### Step 2 — Create `AGENTS.md`

````markdown
# Agent notes

## Commands

- Tests (no GPU, no network): `python -m pytest tests/ -q -p no:cacheprovider`.
  In the maintainer checkout the interpreter is `movio/.venv/bin/python`; do
  **not** call the `pytest` console script there (broken shebang in that venv).
- After adding or removing tests, update the hand-maintained count in
  `KAGGLE_RUN.md:427` to the number from the pytest summary line.
- Trainer smoke: `python train_draft_distill.py --dry-run` (writes
  `codeswitch_ckpt/`).
- Demo smoke: `CODESWITCH_ALLOW_UNTRAINED=1 python kaggle_demo.py`
  (random weights; audio is not speech).

## Repo rules

- Never commit secrets. If a key appears in a file, report `file:line` and
  the key type only — never reproduce the value.
- Hub downloads are revision-pinned and checkpoints load with
  `weights_only=True`; do not remove pins or add `trust_remote_code` /
  `trust_repo` loaders without a plan.
- `movio/`, `movio-2/`, `teacher_cache/`, `paper_figures/`, `output_audio/`,
  `eval_results*/` are gitignored artifacts or retired experiment trees —
  do not edit them from this repo's context.
- `REPORT_MOVIO2_PARLER.md` and `REPORT_MOVIO_INDICF5.md` are archival;
  their numbers are historical.

## Workflow

- Implementation plans live in `plans/` with the status table in
  `plans/README.md`; verbatim research evidence is in `plans/research/`.
- One branch per plan (`advisor/NNN-slug`), sentence-case commit subjects,
  never push.
- The real pipeline is GPU-only (Kaggle). Local verification is pytest plus
  file-text checks; there is no local GPU path.
````

**Verify**

```bash
grep -n "KAGGLE_RUN.md:427\|weights_only=True\|advisor/NNN-slug" AGENTS.md
```

Expected: all three phrases present.

### Step 3 — Create `.env.example`

```dotenv
# CodeSwitch-TTS environment variables.
# Copy to .env (gitignored) or export in your shell; never commit real values.

# Weights & Biases API key for push_to_wandb.py. Resolution order:
# --api-key flag -> this environment variable -> Kaggle secret.
# Create at https://wandb.ai/authorize
WANDB_API_KEY=

# Hugging Face token. Required to download the gated model
# ai4bharat/indic-parler-tts - accept the licence first:
# https://huggingface.co/ai4bharat/indic-parler-tts
# The huggingface_hub client picks this up automatically.
HF_TOKEN=

# Demo wiring smoke only. When set to 1, kaggle_demo.py runs with randomly
# initialised weights if no checkpoint exists - the audio is NOT speech.
CODESWITCH_ALLOW_UNTRAINED=0
```

**Verify**

```bash
grep -n "WANDB_API_KEY\|HF_TOKEN\|CODESWITCH_ALLOW_UNTRAINED" .env.example
```

Expected: all three names present; no real values anywhere.

### Step 4 — Archival banners on the two reports

Prepend a banner **after the title line** (new line 2; keep one blank line
between the title, the banner, and the original content). The banners differ
only in the experiment folder name.

For `REPORT_MOVIO2_PARLER.md` (insert before the existing blank line 2, i.e.
the file becomes: title / blank / banner block / blank / original line 2 …):

```markdown
> **Archival — not maintained.** This report describes the retired
> `movio-2/` experiment (excluded from this repository). The current,
> maintained system is `codeswitch_tts/` — see `README.md` and
> `KAGGLE_RUN.md`. Numbers and claims below are historical; where they touch
> the current pipeline (for example Hugging Face gating), the runbook is
> authoritative.
```

For `REPORT_MOVIO_INDICF5.md`, the same banner with `movio/` instead of
`movio-2/`.

**Verify**

```bash
cd /home/tripathiji/projects/all_work
sed -n '1,9p' REPORT_MOVIO2_PARLER.md
sed -n '1,9p' REPORT_MOVIO_INDICF5.md
grep -c "Archival — not maintained" REPORT_MOVIO2_PARLER.md REPORT_MOVIO_INDICF5.md
```

Expected: banner visible in both; `1` for each file's count.

### Step 5 — Rewrite `clone_and_setup.sh`

Replace the whole file with (this keeps the staged install order and the
protobuf-last ritual, adds a venv + `--dev` extras + `VENV_DIR=""` override,
and replaces the unconditional demo run with the allow-untrained smoke):

```bash
#!/usr/bin/env bash
# ==============================================================================
# CodeSwitch-TTS: GitHub Clone & Environment Setup Script
# For local Linux environments. Kaggle users: follow KAGGLE_RUN.md §1.
#
# Usage:
#   bash clone_and_setup.sh          # runtime install + wiring smoke
#   bash clone_and_setup.sh --dev    # also install requirements-dev.txt (pytest)
#
# Environment overrides:
#   REPO_URL    git remote to clone (default: the upstream repository)
#   TARGET_DIR  clone destination (default: codeSwitch)
#   VENV_DIR    virtualenv location (default: .venv; set VENV_DIR="" to skip
#               and use the system python)
# ==============================================================================

set -e

REPO_URL="${REPO_URL:-https://github.com/tripathiji1312/codeSwitch.git}"
TARGET_DIR="${TARGET_DIR:-codeSwitch}"
VENV_DIR="${VENV_DIR-.venv}"
WITH_DEV=0
for arg in "$@"; do
    case "$arg" in
        --dev) WITH_DEV=1 ;;
        *) echo "Unknown argument: $arg" >&2; exit 2 ;;
    esac
done

echo "========================================================================"
echo " [CodeSwitch-TTS] Cloning Repository & Setting Up Environment"
echo "========================================================================"
echo "Repository: $REPO_URL"
echo "Target Dir: $TARGET_DIR"

# 1. Clone repository if not already present
if [ ! -d "$TARGET_DIR" ]; then
    echo ">>> Cloning repository from $REPO_URL..."
    git clone "$REPO_URL" "$TARGET_DIR"
else
    echo ">>> Target directory $TARGET_DIR already exists. Pulling latest..."
    (cd "$TARGET_DIR" && git pull) || true
fi

cd "$TARGET_DIR"

# 2. Virtualenv (skipped when VENV_DIR is empty)
if [ -n "$VENV_DIR" ]; then
    echo ">>> Creating virtualenv at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
    PY="$VENV_DIR/bin/python"
else
    PY="python3"
fi

pip_install() {
    "$PY" -m pip install "$@"
}

# 3. Staged dependency installation (prevents Kaggle / pip version conflicts)
echo ">>> Installing dependencies (staged to avoid package conflicts)..."

# Stage 1: Parler-TTS from GitHub (brings its transformers/audiotools chain).
# NOTE: this git install is not revision-pinned; see the README security notes.
pip_install -q -U git+https://github.com/huggingface/parler-tts.git

# Stage 2: Pin transformers to the exact version Parler-TTS v0.2.2 targets
pip_install -q "transformers==4.46.1"

# Stage 3: Core acoustic, audio codec, and scientific libraries
pip_install -q -r requirements.txt

# Stage 3b: optional development extras (pytest)
if [ "$WITH_DEV" = "1" ]; then
    pip_install -q -r requirements-dev.txt
fi

# Stage 4: protobuf LAST and >= 5.26 (ships google.protobuf.runtime_version).
# Parler-TTS -> descript-audiotools pins protobuf<5 and silently downgrades it
# to 4.25.9, which breaks the Parler-TTS import and forces the noise-only
# fallback.
pip_install -q "protobuf==5.29.5"

echo ">>> Active versions:"
"$PY" -c "import importlib.metadata as md; print('   protobuf:', md.version('protobuf'), '| transformers:', md.version('transformers'))"

echo "========================================================================"
echo " [CodeSwitch-TTS] Setup complete! Running the wiring smoke test..."
echo "========================================================================"

# Wiring-only smoke: a fresh clone has no trained checkpoint, so allow the
# untrained fallback (audio is random-weight noise - this checks the wiring,
# not the model).
CODESWITCH_ALLOW_UNTRAINED=1 "$PY" kaggle_demo.py

echo ">>> For real synthesis: follow KAGGLE_RUN.md to train, then run:"
echo "      $PY kaggle_demo.py"
```

**Verify** (syntax + text only; do NOT execute the script):

```bash
cd /home/tripathiji/projects/all_work
bash -n clone_and_setup.sh && echo "syntax ok"
grep -n "protobuf==5.29.5\|CODESWITCH_ALLOW_UNTRAINED=1\|--dev\|requirements-dev.txt" clone_and_setup.sh
wc -l clone_and_setup.sh
```

Expected: `syntax ok`; all four grep hits; line count ~90 lines.

### Step 6 — Confirm the suite is untouched

This plan adds no tests. Run the suite and confirm the count matches
`KAGGLE_RUN.md:427` (which the earlier plans maintain):

```bash
cd /home/tripathiji/projects/all_work
movio/.venv/bin/python -m pytest tests/ -q -p no:cacheprovider
grep -n "Local test suite" KAGGLE_RUN.md
```

Expected: same count as before this plan; no changes to `KAGGLE_RUN.md`.

## Test plan

No new pytest tests (the deliverables are text files and a shell script).
Verification is: `bash -n` for the script, grep/sed checks for every new
file and banner, and an unchanged pytest count.

## Done criteria

- [ ] `README.md`, `AGENTS.md`, `.env.example` exist with the specified
      content; README statements match the live tree (pins, commands,
      archive).
- [ ] Both reports carry the archival banner (one per file).
- [ ] `clone_and_setup.sh` passes `bash -n`; venv/`--dev`/allow-untrained
      smoke present; staged install order unchanged (parler → transformers →
      requirements → [dev] → protobuf last).
- [ ] Suite count unchanged; `KAGGLE_RUN.md` untouched.
- [ ] `plans/README.md` status row updated (unless the reviewer maintains it).

## STOP conditions

- Any dependency gate fails (006/007 not landed) → STOP.
- A README statement contradicts the live tree (e.g. a missing pin, a
  different test command, `LICENSE` present after all) → STOP and report;
  do not paper over it.
- The reports' first lines no longer match the quoted titles/project
  folders → STOP (drifted files).
- `bash -n clone_and_setup.sh` fails → fix syntax only; if the failure
  suggests the file changed underneath (unexpected content), STOP.
- Temptation to run `clone_and_setup.sh` or `pip install` → don't; that is
  out of scope for this plan.

## Maintenance notes

- The README's "Security notes" and "Known gap" sentences are tied to plan
  006's pins and the parler-tts git-install decision. If the pins change or
  the parler gap is closed, update README in the same commit.
- `clone_and_setup.sh` intentionally mirrors `KAGGLE_RUN.md` §1's staged
  install; if the runbook's install steps change (plan 007 touched the
  notebook/runbook cells), keep the two in sync.
- `AGENTS.md` is the place for agent-facing conventions that should not
  bloat the README (test-count rule, secrets rule, plans/ workflow).
