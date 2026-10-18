# Plan 011 — Docs truth pass (runbook, paper-claims, report, docstrings, notebook secret label)

> **Executor instructions** — read this first.
> You are working in `/home/tripathiji/projects/all_work` (the CodeSwitch-TTS repo).
> This plan was written against **HEAD `d69b399`**. Before touching anything run:
> ```
> git rev-parse --short HEAD        # expect: d69b399
> git status --porcelain            # expect: only untracked plans/ etc.
> ```
> If HEAD is different or tracked files are dirty, **STOP and report**.
>
> **Hard dependency gate — plan 009 must have landed first:**
> ```
> grep -c '"sentences"' build_codeswitch_corpus.py
> # expect: >= 1   (plan 009 adds the corpus-stats key this docs pass references)
> # at d69b399 this is 0 — if you still see 0, STOP: run plan 009 first.
> ```
> Also assume 006 has landed (it edits a different part of `KAGGLE_RUN.md`; 011's edits are
> disjoint). Never run concurrently with 012 (both may touch the eval docstrings).
>
> **Rules:** never `git push`; never modify files outside the Scope list; use
> `movio/.venv/bin/python` for every python command (the `movio/.venv/bin/pytest` console
> script has a broken shebang — **never** use it). Every old-string below must match
> **exactly once** — count occurrences before editing and fail loudly otherwise.

---

## Status

| Field | Value |
|---|---|
| Priority | Medium (external-facing truth; reviewers read these files) |
| Effort | ~1 h (7 files + notebook; pure text edits, no logic) |
| Risk | Low (one notebook edit — raw-text substitution keeps the file byte-identical elsewhere) |
| Depends on | 009 (gate above); assumes 006 for `KAGGLE_RUN.md` disjointness |
| Category | Docs / honesty / reproducibility |
| Planned at | 2026-10-19 against commit `d69b399` |

## Why this matters

The runbook and paper-claims files currently contradict both the code and the measured/run
reality in ways a reviewer or a Kaggle user will hit immediately: the accelerator is listed as
`T4 x1` while the notebook says `T4 x2`; the distillation step is advertised as `≈ 10 min` while
the runbook's own pass-2 command caps it at 30+20 min; the step-3 artifact list omits `fig5` that
the code produces; the budgets bullet quotes CLI defaults (12/6) that the runbook command
overrides (30/20); `PAPER_CLAIMS_AND_EVIDENCE.md`'s run chain uses a corpus recipe that was
superseded (`--wikipedia 2000`), a `--max-minutes 540` teacher cap that does not exist (the flag
only budgets draft training), and older timing/size numbers; `REPORT_MOVIO2_PARLER.md` claims the
Parler checkpoint is not gated while the Hub reports `"gated":"auto"`; `codeswitch_tts/utils/
assets.py`'s docstring points at `movio/` paths although the code resolves packed
`codeswitch_tts/assets/` paths; `kaggle_demo.py` points at a results path that no code writes;
`run_full_evaluation.py`'s module docstring advertises fig1–fig3 + Tables I–III while `_finish`
builds fig4/fig5 + Tables IV/V; and the notebook instructs users to add a **blank** HF secret
label. Evidence for every claim: `plans/research/findings-2026-10-18.md` (audit reports +
2026-10-19 anchor-verification entry).

## Current state (all anchors verified at `d69b399`; locate by content, lines will shift)

| # | File | Current (wrong) text | Verified truth |
|---|---|---|---|
| a | `KAGGLE_RUN.md` step table row 2b | `≈ 10 min` | runbook pass-2 command caps 30+20 min; measured ~15–25 min for 2000 lines |
| b | `KAGGLE_RUN.md` step table row 3 | `paper_figures/{fig1,fig2,fig3,fig4}.{png,pdf}` | code builds `fig1..fig5` (`fig4_scaling_and_registers` :2533, `fig5_gate_pareto_calibration` :2925) |
| c | `KAGGLE_RUN.md` §0 bullet | `* Accelerator: **GPU T4 x1** (16 GB)` | notebook cell 0: "Select **GPU T4 x2**" (Kaggle's only T4 option is x2) |
| d | `KAGGLE_RUN.md` (no TOC) | — | add a linkless Contents line after the step table |
| e | `KAGGLE_RUN.md` §2b Budgets bullet | `draft training ≤ 12 min, residual ≤ 6 min` | the pass-2 command in the same section uses `--max-minutes 30 --max-minutes-residual 20`; CLI defaults (12/6) only apply when flags are omitted |
| f | `KAGGLE_RUN.md` §3b bullet | `…exits 0 unless you pass `--strict`, and `_finish` still returns success.` | `--strict` exists only in `push_to_wandb.py:699`; `run_full_evaluation.py` has **no** `--strict` (grep: 0 hits) and `_finish` (:3422) always returns success/exit 0 |
| g | `PAPER_CLAIMS_AND_EVIDENCE.md` §3 block | `--wikipedia 2000`, `--max-minutes 540`, `~6-9 s/sentence → ~4-6 h for ~2500 lines`, `~40-60 min`, `--max-minutes 40`, `~35-55 min`, `--gate-thresholds 0,0.6,0.75,0.9` | runbook corpus is `--templates 2500 --wikipedia 900 --max-sentences 2000`; teacher ≈ 7.9 s/sentence → ≈ 4.5 h/2000 lines and is not capped by `--max-minutes`; draft+residual 15–25 min (caps 30/20); eval 25–40 min with `--gate-thresholds auto` (default; `auto` exists at `run_full_evaluation.py:3101`) |
| h | `REPORT_MOVIO2_PARLER.md` | ``Note: `ai4bharat/indic-parler-tts` is NOT gated`` | Hub API: `"gated":"auto"`; downloads 401 without a token (verified 2026-10-19) |
| i | `codeswitch_tts/utils/assets.py:1-7` (path is `utils/assets.py`, NOT `assets.py`) | docstring claims `movio/config/voices/`, `movio/config/gazetteer/`, `movio/models/indicf5_tanglish/vocab.txt` | code resolves `ASSETS_DIR = <pkg>/assets` (:16) → `ASSETS_DIR/voices`, `ASSETS_DIR/gazetteer/proper_nouns.txt`, `ASSETS_DIR/vocab/vocab.txt`; dir tree confirms all three |
| j | `kaggle_demo.py` banner print | `(or codeswitch_ckpt/results.jsonl)` | the evaluator writes `eval_results/raw_results.jsonl` (`run_full_evaluation.py:3068`) |
| k | `run_full_evaluation.py` module docstring | `fig1 … fig3`; `Tables I-III` | `_finish` builds `build_table_IV`/`build_table_V` (:3440-3445); `fig4`/`fig5` produced at :3552/:3558 |
| l | `CodeSwitch_TTS_Kaggle.ipynb` cell 0, `source[6]` | `add a secret with label  and your Hugging Face token.` (blank label, two spaces) | should be `add a secret with label **HF_TOKEN** and your Hugging Face token.` |

## Commands you will need

```bash
# current test count BEFORE edits (record it; expect the same number after — this plan adds no tests)
movio/.venv/bin/python -m pytest tests/ -q -p no:cacheprovider

# occurrence count helper (use before every replacement)
grep -c "<old string>" <file>

# notebook check
git show HEAD:CodeSwitch_TTS_Kaggle.ipynb > /tmp/opencode/nb_head.ipynb
```

## Scope

**In scope (exactly these files):**
1. `KAGGLE_RUN.md` — items a–f (5 edits + the new Contents line).
2. `PAPER_CLAIMS_AND_EVIDENCE.md` — the §3 run-chain code block (item g).
3. `REPORT_MOVIO2_PARLER.md` — the gating sentence (item h).
4. `codeswitch_tts/utils/assets.py` — module docstring (item i).
5. `kaggle_demo.py` — one print string (item j).
6. `run_full_evaluation.py` — module docstring only (item k; do NOT touch code).
7. `CodeSwitch_TTS_Kaggle.ipynb` — cell 0 `source[6]` only (item l).

**Out of scope (do NOT change):**
- `KAGGLE_RUN.md:427` test-count line (`(N tests, no GPU and no network needed)`) — owned by the test-adding plans (008/009).
- The runbook's `Single-T4, single-command workflow` heading and the `--no-utmos` help text's `~50 MB` (real asset 411 MB) — recorded follow-ups, not this pass.
- The two report files' archival banners — plan 010 adds those; don't duplicate.
- Any code behavior; any other docstring.

## Git workflow

```bash
git checkout -b advisor/011-docs-truth-pass
# commits:
#   "Fix the runbook's accelerator, timings, budgets and figure list"
#   "Align the paper-claims run chain with the runbook"
#   "Correct the gating claim in the Parler report"
#   "Point the asset docstring at the packaged paths"
#   "Fix the stale results path in the demo banner"
#   "Extend the evaluation docstring to fig4/fig5 and Tables IV/V"
#   "Name the HF_TOKEN secret label in the notebook"
# Never push. End with: git log --oneline main..HEAD
```

## Steps

### Step 1 — `KAGGLE_RUN.md` (five edits + Contents)

**1a. Step table, row 2b** — replace (this exact line; `grep -c '≈ 10 min' KAGGLE_RUN.md` → 1):

```
| 2b | `python train_draft_distill.py` (draft + residual) | ≈ 10 min | `codeswitch_ckpt/{draft_cb0.pt,residual_cb1_8.pt,meta.json}` |
```
→
```
| 2b | `python train_draft_distill.py` (draft + residual) | ~15–25 min | `codeswitch_ckpt/{draft_cb0.pt,residual_cb1_8.pt,meta.json}` |
```

**1b. Step table, row 3** — change the figure list only:
`paper_figures/{fig1,fig2,fig3,fig4}.{png,pdf}` → `paper_figures/{fig1,fig2,fig3,fig4,fig5}.{png,pdf}`
(leave the rest of the row untouched).

**1c.** Replace:
```
* Accelerator: **GPU T4 x1** (16 GB)
```
→
```
* Accelerator: **GPU T4 x2** (2 × 16 GB)
```

**1d.** Directly **after** the step table's last row (`| 3b | … artifact (the off-Kaggle copy of everything above) |`) and directly **before** the `---` that follows the table, insert a blank line and:

```
**Contents** (plain text — no links): 0) Notebook settings · 1) Setup cell · 2a) Corpus cell · 2b) Distillation cell · 3) Evaluation cell · 3b) Push to W&B · 4) Download everything · 5) Reading the results · 6) Troubleshooting · 7) Where the knobs live
```
(Deliberately unlinked: markdown anchors would silently rot when headings change.)

**1e. §2b Budgets bullet** — replace:
```
* **Budgets**: draft training ≤ 12 min, residual ≤ 6 min (wall-clock caps, `--max-minutes*`).
```
→
```
* **Budgets**: the pass-2 command above caps draft training at 30 min and residual at 20 min
  (`--max-minutes 30 --max-minutes-residual 20`; the CLI defaults are 12 / 6). The teacher pass
  is *not* covered by these caps — it runs to completion or until you stop it.
```

**1f. §3b bullet** — replace:
```
* A failed push never invalidates the evaluation: the script warns (`[W&B] ⚠ push failed: ...`)
  and exits 0 unless you pass `--strict`, and `_finish` still returns success.
```
→
```
* A failed push never invalidates the evaluation: `push_to_wandb.py` warns
  (`[W&B] ⚠ push failed: ...`) and exits 0 unless you pass its `--strict` flag.
  `run_full_evaluation.py` has no `--strict` at all — its `_finish` always returns success
  (exit 0), so the evaluation artifacts are never invalidated by a push.
```

**Verify:**
```bash
grep -c '≈ 10 min' KAGGLE_RUN.md                       # expect: 0
grep -c 'fig1,fig2,fig3,fig4,fig5' KAGGLE_RUN.md       # expect: 1
grep -c 'GPU T4 x1' KAGGLE_RUN.md                      # expect: 0
grep -c 'GPU T4 x2' KAGGLE_RUN.md                      # expect: 1
grep -c '\*\*Contents\*\* (plain text' KAGGLE_RUN.md   # expect: 1
grep -c 'draft training ≤ 12 min' KAGGLE_RUN.md        # expect: 0
grep -c 'no `--strict` at all' KAGGLE_RUN.md           # expect: 1
grep -c 'tests, no GPU' KAGGLE_RUN.md                  # expect: 1 (untouched count line)
```

### Step 2 — `PAPER_CLAIMS_AND_EVIDENCE.md` §3 run chain

Replace the whole bash block currently reading (verbatim):

```bash
# 1) corpus (~1 min)
python build_codeswitch_corpus.py --templates 2500 --wikipedia 2000 \
    --out codeswitch_corpus.txt --json-out codeswitch_corpus.stats.json

# 2) teacher capture  (the bottleneck: ~6-9 s/sentence → ~4-6 h for ~2500 lines)
python train_draft_distill.py --corpus-json codeswitch_corpus.txt \
    --max-minutes 540 --skip-draft --skip-residual --ckpt-dir codeswitch_ckpt
#    …re-run the same line after a session death; shards resume per sentence.

# 3) draft + residual with scheduled sampling (~40-60 min)
python train_draft_distill.py --corpus-json codeswitch_corpus.txt \
    --draft-epochs 60 --residual-sched 0.5 --max-minutes 40 --ckpt-dir codeswitch_ckpt

# 4) full evaluation with the gate sweep (~35-55 min)
python run_full_evaluation.py --ckpt codeswitch_ckpt --force \
    --gate-thresholds 0,0.6,0.75,0.9
```

with:

```bash
# 1) corpus (~1 min; 2 000 lines, 34/33/33)
python build_codeswitch_corpus.py --templates 2500 --wikipedia 900 --max-sentences 2000 \
    --out codeswitch_corpus.txt --json-out codeswitch_corpus.stats.json

# 2) teacher capture  (the bottleneck: ~7.9 s/sentence on a T4 → ~4.5 h for 2000 lines;
#    NOTE: --max-minutes does NOT cap this pass — it runs to completion or until stopped)
python train_draft_distill.py --corpus-json codeswitch_corpus.txt \
    --skip-draft --skip-residual --ckpt-dir codeswitch_ckpt
#    …re-run the same line after a session death; shards resume per sentence.

# 3) draft + residual with scheduled sampling (~15-25 min for 2000 lines;
#    caps as run in the runbook: 30 min draft / 20 min residual)
python train_draft_distill.py --corpus-json codeswitch_corpus.txt \
    --draft-epochs 60 --residual-sched 0.5 \
    --max-minutes 30 --max-minutes-residual 20 --ckpt-dir codeswitch_ckpt

# 4) full evaluation with the derived gate sweep (~25-40 min)
python run_full_evaluation.py --ckpt codeswitch_ckpt --force \
    --gate-thresholds auto
```
(Keep step 5 `--replay` and everything after the block unchanged.)

**Verify:**
```bash
grep -c 'max-minutes 540' PAPER_CLAIMS_AND_EVIDENCE.md     # expect: 0
grep -c 'wikipedia 900' PAPER_CLAIMS_AND_EVIDENCE.md       # expect: 1
grep -c '7.9 s/sentence' PAPER_CLAIMS_AND_EVIDENCE.md      # expect: 1
grep -c 'gate-thresholds auto' PAPER_CLAIMS_AND_EVIDENCE.md # expect: 1
grep -c 'gate-thresholds 0,0.6,0.75,0.9' PAPER_CLAIMS_AND_EVIDENCE.md # expect: 0
```

### Step 3 — `REPORT_MOVIO2_PARLER.md` gating correction

Replace (exact sentence; it is one line):

```
HF_TOKEN stored as a Kaggle secret (not in code). Note: `ai4bharat/indic-parler-tts` is NOT gated (unlike IndicF5 in Approach 1) — HF_TOKEN is only needed for private variants.
```

with:

```
HF_TOKEN stored as a Kaggle secret (not in code). Note: `ai4bharat/indic-parler-tts` **is gated** — the Hub API reports `"gated":"auto"` (verified 2026-10-19): accept its license and log in with a token that has access before the first run. HF_TOKEN is also needed for private variants.
```

**Verify:** `grep -c 'is NOT gated' REPORT_MOVIO2_PARLER.md` → 0; `grep -c '"gated":"auto"' REPORT_MOVIO2_PARLER.md` → 1.

### Step 4 — `codeswitch_tts/utils/assets.py` docstring

Replace the module docstring (lines 1–7 verbatim):

```python
"""Asset manager for CodeSwitch-TTS: Voice Profiles, Gazetteers, and Vocabularies.

Provides access to:
  1. Zero-shot voice profiles (ref.wav + voice.yaml) from `movio/config/voices/`
  2. Proper noun gazetteer (transit locations, ride-hailing brands) from `movio/config/gazetteer/`
  3. Character/letter dictionary from `movio/models/indicf5_tanglish/vocab.txt`
"""
```

with:

```python
"""Asset manager for CodeSwitch-TTS: Voice Profiles, Gazetteers, and Vocabularies.

Provides access to the packaged assets under `codeswitch_tts/assets/` (see `ASSETS_DIR`):
  1. Zero-shot voice profiles (ref.wav + voice.yaml) from `codeswitch_tts/assets/voices/`
  2. Proper noun gazetteer (transit locations, ride-hailing brands) from
     `codeswitch_tts/assets/gazetteer/proper_nouns.txt`
  3. Character/letter dictionary from `codeswitch_tts/assets/vocab/vocab.txt`
"""
```

**Verify:**
```bash
grep -c 'movio/config' codeswitch_tts/utils/assets.py       # expect: 0
movio/.venv/bin/python -c "from codeswitch_tts.utils.assets import get_assets_dir; p = get_assets_dir(); print(p)"
# expect: .../codeswitch_tts/assets  — and: test -d "$(movio/.venv/bin/python -c 'from codeswitch_tts.utils.assets import get_assets_dir; print(get_assets_dir())')/voices"
```

### Step 5 — `kaggle_demo.py` stale results path

Replace:

```python
    print("   and TTFA from the table above (or codeswitch_ckpt/results.jsonl) instead.")
```

with:

```python
    print("   and TTFA from the table above (or eval_results/raw_results.jsonl) instead.")
```

**Verify:** `grep -c 'codeswitch_ckpt/results.jsonl' kaggle_demo.py` → 0; `grep -c 'eval_results/raw_results.jsonl' kaggle_demo.py` → 1; `movio/.venv/bin/python -c "import ast; ast.parse(open('kaggle_demo.py').read()); print('py-ok')"`.

### Step 6 — `run_full_evaluation.py` module docstring

Replace:

```
* Publication figures (300 dpi PNG + vector PDF, IEEE style) in
  ``paper_figures/``: fig1 latency-quality tradeoff, fig2 zone acceptance,
  fig3 streaming chunk + buffer timeline.
* Copy-pasteable LaTeX for Tables I-III (stdout + ``paper_figures/tables.tex``):
  I comparative, II zone acceptance, III per-register CER/latency breakdown.
```

with:

```
* Publication figures (300 dpi PNG + vector PDF, IEEE style) in
  ``paper_figures/``: fig1 latency-quality tradeoff, fig2 zone acceptance,
  fig3 streaming chunk + buffer timeline, fig4 corpus-scaling + per-register
  intelligibility, fig5 confidence-gate Pareto + calibration.
* Copy-pasteable LaTeX for Tables I-V (stdout + ``paper_figures/tables.tex``):
  I comparative, II zone acceptance, III per-register CER/latency breakdown,
  IV distance-to-switch acceptance, V confidence-gated operating points.
```

**Verify:**
```bash
grep -c 'Tables I-III' run_full_evaluation.py     # expect: 0
grep -c 'Tables I-V' run_full_evaluation.py       # expect: 1
movio/.venv/bin/python -c "import run_full_evaluation; print('eval-import-ok')"
```

### Step 7 — Notebook cell 0 secret label (`CodeSwitch_TTS_Kaggle.ipynb`)

Do a **raw-text substitution** (preserves every other byte of the file; do NOT re-dump JSON):

```bash
movio/.venv/bin/python - <<'EOF'
from pathlib import Path
p = Path("CodeSwitch_TTS_Kaggle.ipynb")
raw = p.read_text(encoding="utf-8")
old = "add a secret with label  and your Hugging Face token."
new = "add a secret with label **HF_TOKEN** and your Hugging Face token."
assert raw.count(old) == 1, f"needle count {raw.count(old)} != 1 — STOP"
p.write_text(raw.replace(old, new), encoding="utf-8")
print("NOTEBOOK-EDIT-OK")
EOF
```

**Verify (parsed-structure diff against the committed version):**
```bash
movio/.venv/bin/python - <<'EOF'
import json, subprocess
old = json.loads(subprocess.run(
    ["git", "show", "HEAD:CodeSwitch_TTS_Kaggle.ipynb"],
    capture_output=True, text=True, check=True).stdout)
new = json.loads(open("CodeSwitch_TTS_Kaggle.ipynb", encoding="utf-8").read())
assert len(old["cells"]) == len(new["cells"]) == 25, (len(old["cells"]), len(new["cells"]))
diff_cells = [i for i, (a, b) in enumerate(zip(old["cells"], new["cells"]))
              if a["source"] != b["source"]]
assert diff_cells == [0], f"unexpected cell diffs: {diff_cells} — STOP"
o, n = old["cells"][0]["source"], new["cells"][0]["source"]
assert len(o) == len(n) == len([s for s in o]), "source length changed"
changed = [i for i, (x, y) in enumerate(zip(o, n)) if x != y]
assert changed == [6], f"changed lines {changed} — STOP"
assert n[6] == ("2. **Hugging Face Token** (Optional for faster downloads): In the top menu, "
                "go to **Add-ons -> Secrets**, add a secret with label **HF_TOKEN** and your "
                "Hugging Face token.\n"), repr(n[6])
assert old["metadata"] == new["metadata"]
print("NOTEBOOK-DIFF-OK")
EOF

# The git diff must show only that one line changing in the notebook:
git diff --stat -- CodeSwitch_TTS_Kaggle.ipynb
git diff -- CodeSwitch_TTS_Kaggle.ipynb | grep '^[+-]' | grep -v '^[+-][+-]'
# expect exactly one - line and one + line (the source[6] string)
```

### Step 8 — Full suite + final sweep

```bash
movio/.venv/bin/python -m pytest tests/ -q -p no:cacheprovider
# expect: the SAME count you recorded before starting (98 + whatever 008/009 added; e.g. 109).
# This plan adds no tests: the delta must be 0.

git diff --stat
# expect exactly: KAGGLE_RUN.md, PAPER_CLAIMS_AND_EVIDENCE.md, REPORT_MOVIO2_PARLER.md,
#   codeswitch_tts/utils/assets.py, kaggle_demo.py, run_full_evaluation.py,
#   CodeSwitch_TTS_Kaggle.ipynb — nothing else.
```

## Test plan

No new tests (docs + one string + one docstring + one notebook string). The suite count must
not change. Notebook correctness is verified structurally (Step 7's parsed comparison:
exactly one changed line, 25 cells, metadata untouched). Every replacement step lists its own
`grep -c` verify with the expected count — any mismatch is a STOP.

## Done criteria

- [ ] All Step 1 verifies pass; `KAGGLE_RUN.md` has the Contents line and no `T4 x1` / `≈ 10 min` / `≤ 12 min` bullets.
- [ ] `PAPER_CLAIMS_AND_EVIDENCE.md` run chain matches the runbook (wikipedia 900, max-sentences 2000, no `--max-minutes 540`, caps 30/20, `--gate-thresholds auto`).
- [ ] `REPORT_MOVIO2_PARLER.md` says the checkpoint **is** gated.
- [ ] `assets.py` docstring names the packed `codeswitch_tts/assets/...` paths.
- [ ] `kaggle_demo.py` points at `eval_results/raw_results.jsonl`.
- [ ] `run_full_evaluation.py` docstring says fig4/fig5 + Tables I-V.
- [ ] `NOTEBOOK-DIFF-OK` printed; notebook git diff is one line.
- [ ] Suite count unchanged; `git diff --stat` lists exactly the seven files; nothing pushed.

## STOP conditions

1. `grep -c '"sentences"' build_codeswitch_corpus.py` → 0 (plan 009 not applied) → stop.
2. Any old-string count ≠ 1 before replacement → stop (someone already edited it; re-check drift).
3. The notebook edit: needle count ≠ 1, `NOTEBOOK-DIFF-OK` not printed, or the git diff shows
   anything beyond the two expected `[+-]` lines → stop and `git checkout -- CodeSwitch_TTS_Kaggle.ipynb`.
4. pytest count changed, or `git diff --stat` lists a file outside the seven → stop.
5. HEAD ≠ `d69b399` and the listed snippets no longer match → stop, report which snippet drifted.

## Maintenance notes

- Line numbers quoted above are from `d69b399` — all snippets are content-anchored.
- Weighted-timestamp drift watch: `--max-minutes` caps only draft training (`train_draft_distill.py:1188` default 12.0; `--max-minutes-residual` :1197 default 6.0); if defaults change, Step 1e's parenthetical needs a touch-up.
- Deliberately **not** fixed here (recorded follow-ups): the `Single-T4` heading in `KAGGLE_RUN.md` (ambiguous, not wrong enough to chase), `--no-utmos` help text "~50 MB" (real asset ≈ 411 MB), `movio*/`-tree duplicate docs, and the `REPORT_MOVIO_INDICF5.md` file (covered only by plan 010's banner).
- When 012 lands it does not change any of these documents — no doc edits are required by 012.
- If someone renumbers the runbook's sections, update the Contents line in the same commit (it is deliberately unlinked, so nothing else breaks).
