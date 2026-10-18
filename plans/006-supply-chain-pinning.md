# Plan 006 — Pin supply-chain loads to immutable revisions (and load checkpoints with `weights_only=True`)

> **Executor instructions** — read this first.
> You are working in `/home/tripathiji/projects/all_work` (the CodeSwitch-TTS repo).
> This plan was written against **HEAD `d69b399`**. Before touching anything run:
> ```
> git rev-parse --short HEAD        # expect: d69b399
> git status --porcelain            # expect: only untracked plans/ (and output_audio/ if present)
> ```
> If `git rev-parse` prints anything else, or the working tree has modified tracked files,
> **STOP and report** — every anchor below (line numbers, exact snippets) assumes `d69b399`.
>
> **Hard dependency gates:** none. Execute in the locked order **before 007** (007's dependency
> declarations build on this plan's `torch>=2.6.0` floor), **before 008/010** (their gate checks
> read this plan's literals), and never concurrently with 011/012 (same files).
>
> **Rules:** never `git push`; never modify files outside the Scope list; use the repository
> interpreter `movio/.venv/bin/python` for every python command (the `movio/.venv/bin/pytest`
> console script has a broken shebang — **never** use it); run pytest as
> `movio/.venv/bin/python -m pytest tests/ -q -p no:cacheprovider` from the repo root.
> The venv has **no `pip`** — do not attempt any install.

---

## Status

| Field | Value |
|---|---|
| Priority | High (security + paper reproducibility) |
| Effort | ~1–2 h (5 files, 13 small edits, no new tests) |
| Risk | Low–medium (one upstream uncertainty: parler-tts revision pass-through — see STOP conditions) |
| Depends on | Nothing. Blocks-ish: 007, 008, 010 (ordering only) |
| Category | Security / supply-chain / reproducibility |
| Planned at | 2026-10-19 against commit `d69b399` |

## Why this matters

Every remote artifact the engine loads is fetched from a **mutable ref** (`main`, or the movable
git tag `v1.2.0`), so a compromised or force-pushed upstream can silently change the weights a
run uses — the paper's numbers stop being reproducible even though the repo commit never moved.
Two `torch.load(..., weights_only=False)` calls also allow arbitrary pickle payloads from a
replaced checkpoint file, and the `torch>=2.2.0` floor predates PyTorch's weights-only default.
All four target SHAs were verified live on 2026-10-19 as the current tips of their refs and as
resolvable, cache-addressable revisions (full evidence:
`plans/research/findings-2026-10-18.md`, entry "Supply-chain pinning, HF revision pass-through,
torch.load payload types, torch.hub SHA validation, ASR batching" + its orchestrator addendum).

## Current state (all anchors verified at `d69b399`)

| Loader | File / symbol | Current code | Mutable ref? |
|---|---|---|---|
| Parler model (target) | `codeswitch_tts/engine/speculative_engine.py`, class `SpeculativeEngine` (line 84), `_load_target_model` | `ParlerTTSForConditionalGeneration.from_pretrained(self.config.target_model_id, torch_dtype=self.dtype, attn_implementation="eager",)` | yes — repo `ai4bharat/indic-parler-tts` `main`, **gated** (`"gated":"auto"`) |
| Prompt tokenizer | same method, next call | `AutoTokenizer.from_pretrained(self.config.target_model_id)` | yes — same repo |
| Description tokenizer | same method | `AutoTokenizer.from_pretrained(self.ar_model.config.text_encoder._name_or_path)` | yes — resolves to `google/flan-t5-large` via config — **left unpinned by design** (open question, Maintenance) |
| DAC checkpoint | `codeswitch_tts/utils/audio.py`, class `DACWrapper` (line 90), `_load_dac`, attempt 1 | `model_id = "ylacombe/dac_44khZ_8kbps"` … `DacModel.from_pretrained(model_id)` | yes — alias of `parler-tts/dac_44khZ_8kbps` `main` |
| UTMOS tier 1 | `run_full_evaluation.py`, `UTMOSScorer` (class at line 1296) | `HUB_REPO = "tarepan/SpeechMOS:v1.2.0"`; `torch.hub.load(self.HUB_REPO, self.HUB_ENTRY, trust_repo=True, verbose=False)` | yes — git tag (movable) |
| UTMOS tier 2 (mirror) | same class, tier-2 block | `AutoModel.from_pretrained(model_id, trust_remote_code=True)` — runs automatically as fallback | yes — `prj-beatrice/utmos22-torch-native` `main` |
| Checkpoint loads | `run_full_evaluation.py` `_load_checkpoints` | `torch.load(draft_path, map_location="cpu", weights_only=False)` and `torch.load(residual_path, map_location="cpu", weights_only=False)` | n/a (local files) but pickle-exec risk |
| torch floor | `requirements.txt` line 4 | `torch>=2.2.0` | n/a |

**Verified pins (use exactly these strings):**

| Constant (to add) | Value | Verification (2026-10-19) |
|---|---|---|
| `SpeculativeEngine.PARLER_TTS_REVISION` | `660bc53c64ae83150613264111708a0465211237` | HF API `/api/models/ai4bharat/indic-parler-tts` → `"sha"`; `/revision/<sha>` → 200; `HfApi().model_info(..., revision=sha)` OK |
| `DACWrapper.DAC_REVISION` | `db52bea859d9411e0beb44a3ea923a8731ee4197` | HF API + `/refs` `targetCommit`; `hf_hub_download(..., revision=sha)` OK; **the local HF cache `refs/main` already equals this SHA** |
| `UTMOSScorer.HUB_REPO` (replaces tag) | `tarepan/SpeechMOS:ed25eacbfa42b99156c36ebec67a733b5dbb9b79` | GitHub API `/tags` → v1.2.0 → this SHA (== current `main` tip); installed `torch.hub._validate_not_a_forked_repo` probe: SHA ref **PASSES** |
| `UTMOSScorer.MIRROR_REVISION` | `eecd3e6e7ad9dd794308a4338c3ac7cb34691af9` | HF API `/refs` → `main` `targetCommit`; `hf_hub_download(..., revision=sha)` OK |

**Facts that justify the `weights_only=True` flip** (do not re-derive; details in the research log):
`train_draft_distill.save_checkpoint` is the only `torch.save` producer and writes
`{"state_dict": {str: tensor}, "config": dataclasses.asdict(...) (int/float only), "meta": {...}}`;
`ResidualTrainer.evaluate` returns `(top1 / max(1, total)).tolist()` so `cb_top1` is `list[float]`;
a faithful probe payload of the real classes+stats was saved and loaded successfully under
`weights_only=True`. **No producer casts are needed.** Installed torch is 2.13.0+cu130, where
`weights_only` already defaults to True — the explicit `False` at the two call sites is the only
thing overriding the safe default.

**Local-environment caveat (do not confuse it with a pin failure):** on the installed
transformers 5.16.1, a full `DacModel.from_pretrained("ylacombe/dac_44khZ_8kbps" …)` fails during
config parsing — `AttributeError: property 'frame_rate' of 'DacConfig' object has no setter`
(the repo's 4.38-era `config.json` sets `"frame_rate": 86`, read-only in 5.16.1). This is
**pre-existing and unrelated to the revision pin**; `_load_dac` already swallows it into the
silent `hf_local` (random-weights) fallback locally. Therefore the DAC pin is verified at
**hub-resolution level** here (command E below), not by a model load. On the Kaggle-pinned
transformers 4.46.1 the load is expected to work — UNVERIFIED here, marked as such.

## Commands you will need

```bash
# interpreter (always this one)
movio/.venv/bin/python --version           # confirmed working: Python 3.11.16

# test suite (baseline: 98 passed, 3 warnings, ~28 s)
movio/.venv/bin/python -m pytest tests/ -q -p no:cacheprovider

# drift / diff
git rev-parse --short HEAD
git status --porcelain
git diff --stat

# network is only needed for the optional UTMOS hub-load check (Step 4c); everything else
# in the Verify blocks runs offline.
```

## Scope

**In scope (exactly these files):**
1. `codeswitch_tts/engine/speculative_engine.py` — pin Parler model + prompt tokenizer; annotate the description tokenizer.
2. `codeswitch_tts/utils/audio.py` — pin DAC.
3. `run_full_evaluation.py` — pin the SpeechMOS hub ref; put the UTMOS mirror behind a new opt-in `--utmos-mirror` flag and pin its revision; flip both `weights_only` to `True`.
4. `requirements.txt` — `torch>=2.2.0` → `torch>=2.6.0`.
5. `KAGGLE_RUN.md` — two doc snippets that quote the hub ref / mirror behavior.

**Out of scope (do NOT do here):**
- Pinning the parler-tts **pip git-install** (`pip install -U git+https://github.com/huggingface/parler-tts.git`) — README gap owned by plan 010.
- Pinning the description tokenizer (`google/flan-t5-large`, SHA recorded below) — deliberate open question.
- Content-pinning the SpeechMOS **weight asset** (411 MB GitHub Release file, `digest: null`) — residual gap, documented, not fixable with a ref pin.
- The `dac` package's GitHub-Releases weight chain (`dac.utils.download`, no hash check) and the `movio*/`-tree loader copies — out of scope; noted in the log.
- ASR model revision pinning; `--utmos` help-text size inaccuracy (~50 MB vs 411 MB) — recorded as follow-ups, not changed.

## Git workflow

```bash
git checkout -b advisor/006-supply-chain-pinning
# ... implement, one commit per step group:
#   "Pin Parler-TTS and DAC hub revisions"
#   "Pin the SpeechMOS hub ref; gate the UTMOS mirror behind --utmos-mirror"
#   "Load evaluation checkpoints with weights_only=True"
#   "Raise the declared torch floor to 2.6"
#   "Document the SHA pins in the Kaggle runbook"
# Sentence case, no prefix tags. Never push. At the end: git log --oneline main..HEAD
```

## Steps

### Step 1 — Parler-TTS pins (`codeswitch_tts/engine/speculative_engine.py`)

**1a.** Inside `class SpeculativeEngine`, immediately **above** the line
`    def _load_target_model(self):`, insert (keep the class-level indentation):

```python
    # Pinned hub revision (commit SHA, not a tag) for the Parler-TTS checkpoints.
    # See plans/006-supply-chain-pinning.md — pinning replaces the mutable `main`
    # ref so a force-pushed/compromised upstream cannot silently change the
    # weights a run loads. Update procedure: maintenance notes of that plan.
    PARLER_TTS_REVISION = "660bc53c64ae83150613264111708a0465211237"
```

**1b.** Replace:

```python
            self.ar_model = ParlerTTSForConditionalGeneration.from_pretrained(
                self.config.target_model_id,
                torch_dtype=self.dtype,
                attn_implementation="eager",
            ).to(self.device)
```

with:

```python
            self.ar_model = ParlerTTSForConditionalGeneration.from_pretrained(
                self.config.target_model_id,
                revision=self.PARLER_TTS_REVISION,
                torch_dtype=self.dtype,
                attn_implementation="eager",
            ).to(self.device)
```

**1c.** Replace:

```python
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.target_model_id)
```

with:

```python
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.target_model_id, revision=self.PARLER_TTS_REVISION
            )
```

**1d.** Leave the description tokenizer load exactly as-is, but add the note above it:

```python
            # Deliberately NOT pin-revisioned (open question — see
            # plans/006-supply-chain-pinning.md maintenance notes): this id comes
            # from the checkpoint config (`text_encoder._name_or_path`,
            # mirrored as `google/flan-t5-large`), not from config.py.
            self.description_tokenizer = AutoTokenizer.from_pretrained(
                self.ar_model.config.text_encoder._name_or_path
            )
```

**Verify (all offline):**
```bash
grep -c "660bc53c64ae83150613264111708a0465211237" codeswitch_tts/engine/speculative_engine.py
# expect: 3   (constant + two call sites)

movio/.venv/bin/python -c "from codeswitch_tts.engine.speculative_engine import SpeculativeEngine as S; print(S.PARLER_TTS_REVISION)"
# expect: 660bc53c64ae83150613264111708a0465211237

movio/.venv/bin/python -c "import codeswitch_tts.engine.speculative_engine as s; print('engine-import-ok')"
# expect: engine-import-ok
```
Do **not** attempt a real Parler load here (gated repo; requires HF_TOKEN; ~938 M params).

### Step 2 — DAC pin (`codeswitch_tts/utils/audio.py`)

**2a.** Inside `class DACWrapper`, immediately **above** `    def _load_dac(self):`, insert:

```python
    # Pinned hub revision for the DAC checkpoint (see plans/006-supply-chain-pinning.md).
    DAC_REVISION = "db52bea859d9411e0beb44a3ea923a8731ee4197"
```

**2b.** Replace:

```python
            model_id = "ylacombe/dac_44khZ_8kbps"
            try:
                self.model = DacModel.from_pretrained(model_id).to(self.device)
                self.backend = "hf_pretrained"
```

with:

```python
            model_id = "ylacombe/dac_44khZ_8kbps"
            try:
                self.model = DacModel.from_pretrained(
                    model_id, revision=self.DAC_REVISION
                ).to(self.device)
                self.backend = "hf_pretrained"
```

**Verify (all offline):**
```bash
grep -c "db52bea859d9411e0beb44a3ea923a8731ee4197" codeswitch_tts/utils/audio.py
# expect: 2   (constant + one call site)

movio/.venv/bin/python -c "from codeswitch_tts.utils.audio import DACWrapper as W; print(W.DAC_REVISION)"
# expect: db52bea859d9411e0beb44a3ea923a8731ee4197

# Hub-level pin resolution (uses the local HF cache; snapshot dir is named after the SHA):
HF_HUB_OFFLINE=1 movio/.venv/bin/python -c "
from codeswitch_tts.utils.audio import DACWrapper
from huggingface_hub import hf_hub_download
p = hf_hub_download('ylacombe/dac_44khZ_8kbps', 'config.json', revision=DACWrapper.DAC_REVISION)
print('HF-LEVEL-PIN-OK', p)"
# expect a path ending in:
# models--ylacombe--dac_44khZ_8kbps/snapshots/db52bea859d9411e0beb44a3ea923a8731ee4197/config.json
```
**Do NOT** additionally run `DacModel.from_pretrained(...)` as a verify: on this venv it fails
with the pre-existing `frame_rate` config incompatibility (see Current state), which would be
misread as a pin failure.

### Step 3 — SpeechMOS hub ref pin (`run_full_evaluation.py`)

Replace (inside `class UTMOSScorer`, the comment block + constant):

```python
    #: ``tarepan/SpeechMOS`` is a torch.hub *repo* (the canonical UTMOS22-strong
    #: reimplementation), not a transformers checkpoint - AutoModel can never
    #: load it.  Tier 1 is the hub loader, tier 2 a transformers mirror in case
    #: the hub is unreachable.
    HUB_REPO = "tarepan/SpeechMOS:v1.2.0"
    HUB_ENTRY = "utmos22_strong"
```

with:

```python
    #: ``tarepan/SpeechMOS`` is a torch.hub *repo* (the canonical UTMOS22-strong
    #: reimplementation), not a transformers checkpoint - AutoModel can never
    #: load it. Tier 1 is the hub loader, tier 2 an opt-in (`--utmos-mirror`)
    #: transformers mirror in case the hub is unreachable.
    #: The ref is pinned to the ``v1.2.0`` commit SHA (tags are movable; see
    #: plans/006-supply-chain-pinning.md). NOTE: the 411 MB weight asset itself
    #: is fetched from a GitHub Release *inside* hubconf.py and is not
    #: content-pinned - ref pinning freezes the code, not the bytes.
    HUB_REPO = "tarepan/SpeechMOS:ed25eacbfa42b99156c36ebec67a733b5dbb9b79"
    HUB_ENTRY = "utmos22_strong"
```

**Verify (offline):**
```bash
grep -n "HUB_REPO = " run_full_evaluation.py
# expect one line with the full 40-char SHA after the colon

grep -c "SpeechMOS:v1.2.0" run_full_evaluation.py
# expect: 0

movio/.venv/bin/python -c "import run_full_evaluation as r; print(r.UTMOSScorer.HUB_REPO)"
# expect: tarepan/SpeechMOS:ed25eacbfa42b99156c36ebec67a733b5dbb9b79
```
Optional online check (only if internet is available; downloads ~411 MB — acceptable to skip):
```bash
movio/.venv/bin/python -c "
import torch
m = torch.hub.load('tarepan/SpeechMOS:ed25eacbfa42b99156c36ebec67a733b5dbb9b79','utmos22_strong',trust_repo=True,verbose=False)
print('HUB-PIN-LOAD-OK', type(m).__name__)"
# expect: HUB-PIN-LOAD-OK ... (validation passes because the SHA equals the v1.2.0 tag commit)
```

### Step 4 — UTMOS mirror: opt-in flag + revision pin (`run_full_evaluation.py`)

**4a.** Add two class constants right below `HUB_ENTRY`:

```python
    #: Opt-in fallback (--utmos-mirror): pinned mirror revision; the repo ships
    #: remote code (trust_remote_code), so the SHA pins code as well as weights.
    MIRROR_MODEL_ID = "prj-beatrice/utmos22-torch-native"
    MIRROR_REVISION = "eecd3e6e7ad9dd794308a4338c3ac7cb34691af9"
```
(Use exactly the SHA printed in Step 6 verify — copy from this file, do not retype.)

**4b.** In `__init__`, change the signature and store the flag:

```python
    def __init__(self, device, enabled: bool = True,
                 model_id: str = MIRROR_MODEL_ID,
                 mirror_enabled: bool = False,
                 verbose: bool = True):
        self.model_id = model_id
        self.mirror_enabled = mirror_enabled
```
(rest of `__init__` unchanged down to the tier-1 `except`).

**4c.** Gate + pin tier 2 — replace:

```python
        # --- tier 2: transformers mirror --------------------------------------
        try:
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
```

with:

```python
        # --- tier 2: transformers mirror (opt-in: executes remote code) -------
        if not self.mirror_enabled:
            self.reason = (f"torch.hub failed ({hub_reason}); mirror disabled "
                           "(re-run with --utmos-mirror to enable the pinned "
                           "prj-beatrice/utmos22-torch-native fallback)")
            if verbose:
                print(f"[UTMOS] ⚠ perceptual proxy unavailable ({self.reason}); "
                      "UTMOS columns will be absent.")
            return
        try:
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(
                model_id, trust_remote_code=True, revision=self.MIRROR_REVISION
            )
```
(The success/`except` bodies below stay as they are — the `except` already reports
`torch.hub failed ({hub_reason}); {type(exc).__name__}: {exc}`.)

**4d.** Construction site — replace:

```python
    utmos = UTMOSScorer(
        device,
        enabled=(not args.no_utmos and not args.skip_asr
                 and any(c in configs for c in ("A", "B", "C"))),
    )
```

with:

```python
    utmos = UTMOSScorer(
        device,
        enabled=(not args.no_utmos and not args.skip_asr
                 and any(c in configs for c in ("A", "B", "C"))),
        mirror_enabled=args.utmos_mirror,
    )
```

**4e.** Argparse — add directly below the existing `--no-utmos` argument:

```python
    p.add_argument("--utmos-mirror", action="store_true",
                   help="Enable the transformers mirror "
                        "(prj-beatrice/utmos22-torch-native, SHA-pinned) as a UTMOS "
                        "fallback when torch.hub fails. Runs remote code via "
                        "trust_remote_code; disabled by default.")
```

**Verify (offline):**
```bash
grep -c -- "--utmos-mirror" run_full_evaluation.py      # expect: 2 (arg + doc-referencing help line is inside arg help; recount after edit: expect >=2)
grep -c "mirror_enabled" run_full_evaluation.py         # expect: >=3 (signature, assignment, construction kwarg, gate)
grep -c "eecd3e6e7ad9dd794308a4338c3ac7cb34691af9" run_full_evaluation.py   # expect: 1

movio/.venv/bin/python -c "
import run_full_evaluation as r
print(r.UTMOSScorer.MIRROR_REVISION)
u = r.UTMOSScorer('cpu', enabled=False)
print('utmos-disabled-ok', u.available)"
# expect: the SHA, then: utmos-disabled-ok False
```
Then confirm the run-through still parses its CLI:
```bash
movio/.venv/bin/python run_full_evaluation.py --help | grep -A3 "utmos-mirror"
# expect the new flag with its help text
```

### Step 5 — Flip both checkpoint loads to `weights_only=True` (`run_full_evaluation.py`)

Replace **both** occurrences (draft and residual; each string occurs exactly once):

```python
        payload = torch.load(draft_path, map_location="cpu", weights_only=False)
```
→
```python
        payload = torch.load(draft_path, map_location="cpu", weights_only=True)
```

```python
        payload = torch.load(residual_path, map_location="cpu", weights_only=False)
```
→
```python
        payload = torch.load(residual_path, map_location="cpu", weights_only=True)
```

**Verify (offline):**
```bash
grep -n "weights_only" run_full_evaluation.py
# expect exactly 2 lines, both weights_only=True (no other weights_only anywhere in the file)

movio/.venv/bin/python - <<'EOF'
import tempfile, torch
from pathlib import Path
from codeswitch_tts.config import DraftModelConfig
from codeswitch_tts.models.draft_transformer import DraftTransformer
from train_draft_distill import save_checkpoint

cfg = DraftModelConfig()
p = Path(tempfile.mkdtemp()) / "draft_cb0.pt"
save_checkpoint(p, DraftTransformer(cfg), cfg,
                meta={"param_count": 1, "val_top1": 0.2, "cb_top1": [0.1] * 8})
payload = torch.load(p, map_location="cpu", weights_only=True)
print("WEIGHTS-ONLY-OK", sorted(payload.keys()))
EOF
# expect: [Save] .../draft_cb0.pt (16.81M params)
#         WEIGHTS-ONLY-OK ['config', 'meta', 'state_dict']
```
(This exact command was executed on 2026-10-19 against the real classes and passed.)

### Step 6 — torch floor (`requirements.txt`)

Replace the single line `torch>=2.2.0` with `torch>=2.6.0` (leave `torchaudio>=2.2.0` and every
other line untouched; do NOT run pip — the venv has none).

**Verify:** `grep -n "torch>=2.6.0" requirements.txt` → exactly 1 line.

### Step 7 — Kaggle runbook truth (`KAGGLE_RUN.md`)

**7a.** Replace the UTMOS/MCD bullet's first four lines (the block starting
`* **UTMOS / MCD**: UTMOS22 (`) with:

```markdown
* **UTMOS / MCD**: UTMOS22 (`torch.hub:tarepan/SpeechMOS:ed25eacbfa42b99156c36ebec67a733b5dbb9b79/utmos22_strong`,
  no listeners) scores every saved wav — it is a torch.hub repo, *not* a transformers
  checkpoint, so it needs GitHub access. The hub ref is pinned to the `v1.2.0` commit SHA;
  `--utmos-mirror` opts in to the SHA-pinned transformers fallback
  (`prj-beatrice/utmos22-torch-native`, runs remote code), and `run_meta.json` →
  `utmos_model` records which loader won. If both fail the columns are simply absent.
```
(Keep the bullet's remaining MCD sentences — unchanged.)

**7b.** In the troubleshooting table, replace the row:

```markdown
| `[UTMOS] ⚠ perceptual proxy unavailable (...)` | both loaders failed (hub needs GitHub access; the transformers mirror needs `trust_remote_code`). Everything else still runs; `run_meta.json` → `utmos_available` says so. |
```

with:

```markdown
| `[UTMOS] ⚠ perceptual proxy unavailable (...)` | both loaders failed (hub needs GitHub access; the mirror needs `--utmos-mirror` *and* `trust_remote_code`). Everything else still runs; `run_meta.json` → `utmos_available` says so. |
```

**Verify:**
```bash
grep -c "SpeechMOS:ed25eacbfa42b99156c36ebec67a733b5dbb9b79" KAGGLE_RUN.md   # expect: 1
grep -c -- "--utmos-mirror" KAGGLE_RUN.md                                   # expect: 1
grep -c "SpeechMOS:v1.2.0" KAGGLE_RUN.md                                    # expect: 0
```

### Step 8 — Full suite + final sweep

```bash
movio/.venv/bin/python -m pytest tests/ -q -p no:cacheprovider
# expect: 98 passed, 3 warnings  (baseline; this plan adds no tests)

git diff --stat
# expect only: codeswitch_tts/engine/speculative_engine.py, codeswitch_tts/utils/audio.py,
#              run_full_evaluation.py, requirements.txt, KAGGLE_RUN.md
# Any other modified file is a STOP condition.

# one-shot marker sweep (expect 1 for each):
for s in 660bc53c64ae83150613264111708a0465211237 db52bea859d9411e0beb44a3ea923a8731ee4197 ed25eacbfa42b99156c36ebec67a733b5dbb9b79 eecd3e6e7ad9dd794308a4338c3ac7cb34691af9; do
  git grep -l "$s" | tr '\n' ' '; echo
done
```

## Test plan

No new tests: this plan changes literals in remote-load calls, not logic with an offline-testable
contract (the only offline-testable piece — checkpoint loading — is exercised by the Step 5
functional check and later covered by plan 008's contract tests). The suite must stay at exactly
**98 passed**. The security property is verified by presence of the literal pins at each call
site (Steps 1–4, 7) plus hub-level resolution for the cached DAC pin (Step 2) and the
`torch.hub` acceptance probe recorded in the research log.

## Done criteria

- [ ] `SpeculativeEngine.PARLER_TTS_REVISION` exists; both `from_pretrained` calls in `_load_target_model` pass `revision=`; grep count 3.
- [ ] `DACWrapper.DAC_REVISION` exists; `DacModel.from_pretrained` passes `revision=`; offline hub-level check prints the SHA-named snapshot path.
- [ ] `UTMOSScorer.HUB_REPO` ends in the 40-char SHA; `torch.hub.load` call itself unchanged.
- [ ] `--utmos-mirror` flag exists, defaults off; tier 2 gated by it; mirror load passes `revision=`.
- [ ] Both `torch.load` sites are `weights_only=True`; functional save/load check prints `WEIGHTS-ONLY-OK [... 'state_dict']`.
- [ ] `requirements.txt` declares `torch>=2.6.0`.
- [ ] The two `KAGGLE_RUN.md` snippets updated.
- [ ] `movio/.venv/bin/python -m pytest tests/ -q -p no:cacheprovider` → 98 passed.
- [ ] `git diff --stat` touches exactly the five files above; branch `advisor/006-supply-chain-pinning`; nothing pushed.

## STOP conditions

1. `git rev-parse --short HEAD` ≠ `d69b399` or tracked files are dirty → stop, report.
2. Parler load raising `TypeError: ... unexpected keyword argument 'revision'` at runtime on a
   real environment → the installed parler-tts build differs from upstream `main` (where
   `from_pretrained` forwards `**kwargs`, verified 2026-10-19). Do **not** drop the pin silently:
   stop and report so the pip pin can be revisited.
3. `torch.hub.load` fails with `Cannot find <sha> in https://github.com/tarepan/SpeechMOS` →
   stop. Do **not** add `skip_validation=True` or revert to the tag.
4. Any `torch.serialization.UnpicklingError` when loading an existing checkpoint with the flipped
   `weights_only=True` → stop and report (do not revert to `False` without recording why).
5. pytest ≠ 98 passed, or `git diff --stat` shows files outside the five listed → stop.
6. A Verify command that requires network fails while offline → skip that single check, note it
   in the commit body, continue (not a stop).

## Maintenance notes

- **Residual gaps (do not silently close):** (a) the parler-tts *pip* install is a moving git
  HEAD — plan 010 records this in the README "not covered" list; (b) the description tokenizer
  resolves to `google/flan-t5-large` (SHA seen 2026-10-19: `0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a`)
  and stays unpinned — open question; (c) the SpeechMOS *weights* file (411 179 119 B GitHub
  Release asset, `digest: null`, `immutable: false`) is not pinned by any ref — a future plan
  could record a sha256 in the runbook; (d) the `dac` pip package's weight download has no hash
  check, and `movio-2/backend/tts_engine.py:129-138` + three `movio/` scripts keep unpinned
  copies of these calls.
- **Changing a pin later:** for HF repos read `"sha"` from `https://huggingface.co/api/models/<repo>`
  (follow redirects for the DAC alias) or `/refs` → `targetCommit`; for SpeechMOS read
  `https://api.github.com/repos/tarepan/SpeechMOS/tags`; then update the constant + doc strings and
  re-run the Step 1–4 greps. Keep constants SHAs, never tags.
- **Observable side-effects:** `run_meta["utmos_model"]` will now read
  `torch.hub:tarepan/SpeechMOS:ed25eac…/utmos22_strong` — the KAGGLE_RUN bullet above documents
  it; UTMOS mirror no longer engages automatically, so offline/failed-hub runs now honestly report
  `utmos_available:false` unless `--utmos-mirror` is passed.
- **Related follow-ups not in scope:** local venv cannot load the DAC repo at all (transformers
  5.16.1 vs 4.38-era config, `frame_rate` property) — the silent `hf_local` random-weights fallback
  it triggers is a telemetry-honesty item for a future plan; `--no-utmos` help text says "~50 MB"
  but the real asset is 411 MB.
- **Line numbers** used above are from `d69b399`; locate snippets by content, not by line, if
  earlier plans have landed.
