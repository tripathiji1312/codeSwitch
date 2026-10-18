# Plan 012 — Perf quick wins: batched ASR post-pass, pressure-gated memory guard, no unused residual logits

> **Executor instructions** — read this first.
> You are working in `/home/tripathiji/projects/all_work` (the CodeSwitch-TTS repo).
> This plan was written against **HEAD `d69b399`**. Since then one compatible commit landed:
> `f689736` ("Create the results directory before writing the provenance file" — a single
> added line at `run_full_evaluation.py:3196`). Before touching anything run:
> ```
> git log --oneline -3     # expect f689736 (or a later descendant) and d69b399 in history
> git status --porcelain   # expect: only untracked plans/ etc.
> ```
> If neither commit is in history, or tracked files are dirty, **STOP and report**.
> All snippets below are content-anchored; line numbers quoted at `d69b399` shift by **+1**
> after `run_full_evaluation.py:3196` on `f689736`.
>
> **Serialisation gate:** do **not** run concurrently with 008/009 (they edit the same
> `KAGGLE_RUN.md:427` test-count line) or with 011 (it edits the same notebook file — different
> cell). There is no hard code dependency on any plan.
>
> **Rules:** never `git push`; never modify files outside the Scope list; use
> `movio/.venv/bin/python` for every python command (the `movio/.venv/bin/pytest` console
> script has a broken shebang — **never** use it). Every old-string below must match
> **exactly once** — count occurrences before editing and fail loudly otherwise.

---

## Status

| Field | Value |
|---|---|
| Priority | Medium-High (pure waste inside the loop the paper times; ASR wall-time dominates the live run) |
| Effort | ~2–3 h |
| Risk | Medium (the evaluation synthesis loop is reworked; every change is behaviour-preserving and test-covered) |
| Depends on | nothing hard; serialise with 008/009 (count line) and 011 (notebook) |
| Category | Performance / cleanup |
| Planned at | 2026-10-19 against `d69b399` (+`f689736`) |

## Why this matters

Four audit findings (`plans/research/findings-2026-10-18.md`), each independently verified:

1. **[PERF-04] fixed-cadence `gc.collect()` + `torch.cuda.empty_cache()`**: eval `:3412–3417`
   (every 5 utterances), train `:643–646` (every 25 shards), `kaggle_demo.py:140–142` (before
   every utterance), notebook cell 10 `source[47–48]`. `empty_cache()` returns cached blocks to
   the driver, so the next allocations of persona/decoder activations must `cudaMalloc` again;
   `gc.collect()` is a full-heap collection in a process holding 2–4 models plus large numpy
   buffers. The pattern was added against fragmentation on a 16 GB T4 — the fix keeps that
   protection for the case it was added for (a GPU close to exhaustion) and makes the common
   case ~free.
2. **[PERF-05] unused residual logits**: `ResidualDecoder.forward` builds a `(8, T, 1024)`
   tensor on every inference chunk (`residual_decoder.py:129–152`; ≈14 MB at T=430) that no
   inference caller reads — eval `:1000–1004` uses `full_codebooks[0]`, streaming
   `:273–276` uses `full_codebooks`, train sanity `:1342–1346` uses `full_codebooks[0]`.
   Only training reads `.logits` (`train_draft_distill.py:954`).
3. **[A9] per-row ASR**: `ASREvaluator.transcribe` runs inline for one row at a time
   (`run_full_evaluation.py:3368–3382`). The live Kaggle run (2026-10-19, `f689736`) shows the
   cost directly: config-B/C rows print `total=120–340 ms` while the same rows report
   `elapsed 0.3–20.6 s` — the extra 0.3–20 s per row is sequential ASR (whisper-small
   hallucination loops) plus UTMOS, 120 rows in a row. The pipeline supports batching,
   verified live (findings addendum, re-verified 2026-10-19 on
   `hf-internal-testing/tiny-random-WhisperForConditionalGeneration`, `batch_size=2,3,4`:
   batched results are **count/order/text-identical** to per-item calls), and its preprocess
   **destructively pops `array`/`sampling_rate` from the caller's dict** — payloads must be
   built fresh per call. (This is the fix for the perf report's **PERF-02**.)
4. **[PERF-03] the results file is rewritten end-to-end after every utterance** (the O(n)-per-row
   rebuild + full-file `save_results` at `run_full_evaluation.py:3390–3394`). Step 4f replaces it
   with an O(1) in-place row store plus a 20-row debounced save.

## Current state (anchors verified at `d69b399`; locate by content, lines shift)

| # | File | Current | Verified truth |
|---|---|---|---|
| a | `run_full_evaluation.py:3412–3417` | `gc.collect()` + `empty_cache()` every 5 rows | unconditional cadence; the point is to run them **only under pressure** |
| b | `train_draft_distill.py:643–646` | same pattern every 25 shards | same |
| c | `kaggle_demo.py:140–142` | `if torch.cuda.is_available(): torch.cuda.empty_cache()` | same guard applies |
| d | `CodeSwitch_TTS_Kaggle.ipynb` cell 10, `source[47]/[48]` | `if torch.cuda.is_available():` / `torch.cuda.empty_cache()` | raw-JSON needle verified unique (`empty_cache` occurs once in the whole file) |
| e | `codeswitch_tts/models/residual_decoder.py:22–27, 129–152` | `logits: torch.Tensor` non-optional; `torch.cat(head_logits, …)` always | skip the `cat` for inference (keep `head_logits` list empty) |
| f | `run_full_evaluation.py:1225–1238` | one-item `transcribe()` | add a batched entry point; keep `transcribe` as the fallback |
| g | `run_full_evaluation.py:3368–3382` | ASR inline per row; `:3390–3393` O(n) row rebuild; `:3394` full-file save per row | waves → `session_waves`, one batched post-pass, O(1) row store, debounced save |
| h | `codeswitch_tts/utils/` | only `assets.py` + `audio.py` (findings :1517/:1924) | plan creates `memory.py` |

## Commands you will need

```bash
# record BEFORE edits (the delta after this plan must be exactly +3 — read the
# pytest summary, don't predict the cumulative total)
movio/.venv/bin/python -m pytest tests/ -q -p no:cacheprovider

# occurrence-count helper (use before every replacement)
grep -c "<old string>" <file>

# notebook dry-run copy (do the needle test on the copy first if you want a free rehearsal)
cp CodeSwitch_TTS_Kaggle.ipynb /tmp/opencode/nb012_dryrun.ipynb
```

## Scope

**In scope (exactly these files):**
1. `codeswitch_tts/utils/memory.py` — **new** (pressure guard).
2. `run_full_evaluation.py` — memory guard swap, ASR batching + post-pass, row store/debounce,
   `import gc` removal, one added import.
3. `train_draft_distill.py` — guard swap, `import gc` removal, one added import.
4. `kaggle_demo.py` — guard swap, one added import.
5. `CodeSwitch_TTS_Kaggle.ipynb` — cell 10 `source[47]/[48]` only.
6. `codeswitch_tts/models/residual_decoder.py` — `ResidualOutput.logits` Optional + `return_logits`.
7. `codeswitch_tts/engine/streaming_pipeline.py` — one call-site kwarg.
8. `tests/test_codeswitch_tts.py` — +2 tests.
9. `tests/test_evaluation_suite.py` — +1 test.
10. `KAGGLE_RUN.md` — the test-count line only.

**Out of scope (do NOT change):**
- Training `_epoch`/`evaluate` residual calls (`train_draft_distill.py:953–954`, `:984`) — they
  read `.logits` / `.residual_codebooks` and stay on the default `return_logits=True`.
- The draft-side `.logits` readers (`run_full_evaluation.py:549`, `train_draft_distill.py:728`,
  `:759`) — those are `DraftOutput`, untouched.
- Module docstrings (011 owns them; this plan adds no prose).
- `codeswitch_tts/utils/__init__.py` (the new module is imported by full path).
- Any CLI flag, any deps file, `.gitignore`.

## Git workflow

```bash
git checkout -b advisor/012-perf-quick-wins
# commits:
#   "Add a pressure-gated memory guard and drop the fixed-cadence cache flush"
#   "Skip the unused residual logits tensor at inference call sites"
#   "Batch the evaluation's ASR calls and recover CER on resumed rows"
#   "Coalesce result rows and debounce the results-file rewrite"
#   "Cover the memory guard, ASR batching and residual flag with tests"
#   "Update the hand-maintained test count"
# Never push. End with: git log --oneline main..HEAD
```

## Steps

### Step 1 — `codeswitch_tts/utils/memory.py` + four call sites

**1a. Create `codeswitch_tts/utils/memory.py` with exactly:**

```python
"""Memory-pressure guard shared by the training/evaluation loops.

``gc.collect()`` + ``torch.cuda.empty_cache()`` on a fixed cadence hands the
caching allocator's blocks back to the driver even when nothing is under
pressure, so the next allocations pay ``cudaMalloc`` again - and a full-heap
``gc.collect()`` runs while several models and large numpy buffers are alive.
This guard keeps the safety behaviour for the case it was added for (a GPU
close to exhaustion) and turns the common case into a couple of integer
comparisons.
"""

from __future__ import annotations

import gc

import torch

__all__ = ["maybe_free_memory"]


def maybe_free_memory(used_frac_threshold: float = 0.90, verbose: bool = False) -> bool:
    """Collect garbage and empty the CUDA cache only under real pressure.

    Returns True when a collection actually ran (CUDA available and device
    utilisation at/above *used_frac_threshold*), False otherwise.  On CPU-only
    machines this is a no-op: the previous unconditional ``gc.collect()`` bought
    nothing there because the next step is never a ``cudaMalloc``.

    ``torch.cuda.mem_get_info()`` reports (free, total) device bytes, so the
    utilisation measured here includes other processes on the same GPU - a
    deliberately conservative signal for "the next allocation may have to wait
    for the driver".
    """
    if not torch.cuda.is_available():
        return False
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except Exception:
        return False
    if not total_bytes:
        return False
    used_frac = 1.0 - (float(free_bytes) / float(total_bytes))
    if used_frac < used_frac_threshold:
        return False
    if verbose:
        print(f"[mem] pressure {used_frac:.0%} >= {used_frac_threshold:.0%}: gc + empty_cache")
    gc.collect()
    torch.cuda.empty_cache()
    return True
```

**1b. `run_full_evaluation.py` — remove `import gc`** (grep `import gc` → 1). Replace:

```
import base64
import gc
import hashlib
```
with:
```
import base64
import hashlib
```

and add the import after the transliteration import (`grep -c 'from codeswitch_tts.frontend.tamil_latin import latinize_text' → 1`):

```
from codeswitch_tts.frontend.tamil_latin import latinize_text  # noqa: E402
from codeswitch_tts.utils.memory import maybe_free_memory  # noqa: E402
```

**1c. Swap the eval gc block.** Replace:

```
                n_runs_since_gc += 1
                if n_runs_since_gc >= 5:
                    n_runs_since_gc = 0
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
```
with:
```
                n_runs_since_gc += 1
                if n_runs_since_gc >= 5:
                    n_runs_since_gc = 0
                    maybe_free_memory()
```

**1d. `train_draft_distill.py`** — remove `import gc` (grep `import gc` → 1; only use is the block below), add the import after the `codeswitch_benchmark_data` import block:

```
from codeswitch_benchmark_data import (  # noqa: E402
    assert_corpus_disjoint_from_eval,
    build_training_corpus,
    load_corpus_json,
)
from codeswitch_tts.utils.memory import maybe_free_memory  # noqa: E402
```

and replace:

```
            if len(shards) % 25 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
```
with:
```
            if len(shards) % 25 == 0:
                maybe_free_memory()
```

**1e. `kaggle_demo.py`** — add the import after the `codeswitch_tts.utils.audio` import:

```
from codeswitch_tts.utils.audio import save_wav
from codeswitch_tts.utils.memory import maybe_free_memory
```

and replace:

```
        # Warm memory / empty cache if CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```
with:
```
        # Free cached/reclaimed memory only under real pressure
        # (see codeswitch_tts/utils/memory.py)
        maybe_free_memory()
```

**1f. Notebook cell 10** — raw-text substitution (preserves every other byte; do NOT re-dump JSON):

```bash
movio/.venv/bin/python - <<'EOF'
from pathlib import Path
p = Path("CodeSwitch_TTS_Kaggle.ipynb")
raw = p.read_text(encoding="utf-8")
old = "    if torch.cuda.is_available():\\n\",\n    \"        torch.cuda.empty_cache()\\n\","
new = "    from codeswitch_tts.utils.memory import maybe_free_memory\\n\",\n    \"    maybe_free_memory()\\n\","
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
assert diff_cells == [10], f"unexpected cell diffs: {diff_cells}"
o, n = old["cells"][10]["source"], new["cells"][10]["source"]
assert len(o) == len(n) == 69
changed = [i for i, (x, y) in enumerate(zip(o, n)) if x != y]
assert changed == [47, 48], f"changed lines {changed} — STOP"
assert n[47] == "    from codeswitch_tts.utils.memory import maybe_free_memory\n", repr(n[47])
assert n[48] == "    maybe_free_memory()\n", repr(n[48])
assert old["metadata"] == new["metadata"]
print("NOTEBOOK-DIFF-OK")
EOF
```

(The needle test and the diff script above were rehearsed on a copy of the committed notebook
on 2026-10-19 — both printed `NOTEBOOK-EDIT-OK`-equivalent results and `NOTEBOOK-DIFF-OK`.)

**Verify (all of Step 1):**

```bash
grep -rc "import gc" run_full_evaluation.py train_draft_distill.py   # expect: 0 and 0
grep -c "maybe_free_memory" run_full_evaluation.py                   # expect: 2 (import + call)
grep -c "maybe_free_memory" train_draft_distill.py                   # expect: 2
grep -c "maybe_free_memory" kaggle_demo.py                           # expect: 2
grep -c "empty_cache" kaggle_demo.py                                 # expect: 0
movio/.venv/bin/python -c "
from codeswitch_tts.utils.memory import maybe_free_memory
assert maybe_free_memory() is False  # CPU box: no-op
print('GUARD-OK')"
movio/.venv/bin/python -c "import run_full_evaluation, train_draft_distill, kaggle_demo; print('IMPORTS-OK')"
```

### Step 2 — Residual logits flag (`codeswitch_tts/models/residual_decoder.py` + 3 callers)

**2a.** In `ResidualOutput` replace:

```
    logits: torch.Tensor              # (batch, 8, num_frames, 1024)
```
with:
```
    logits: Optional[torch.Tensor]    # (batch, 8, num_frames, 1024); None when return_logits=False
```
(`Optional` is already imported at `:12`.)

**2b.** Extend the `forward` signature and docstring. Replace:

```
    def forward(
        self,
        cb0_tokens: torch.Tensor,
        temperature: float = 1.0,
        sample: bool = False,
    ) -> ResidualOutput:
```
with:
```
    def forward(
        self,
        cb0_tokens: torch.Tensor,
        temperature: float = 1.0,
        sample: bool = False,
        return_logits: bool = True,
    ) -> ResidualOutput:
```

and in the Args docstring after the `sample:` line add:

```
            return_logits: Build the stacked (8, T, 1024) logits tensor.  Training
                (cross-entropy) needs it; inference callers consume only the
                codebooks, so they pass False to skip the cat and its memory peak.
```

**2c.** Skip the per-head stash and the cat:

```
        for head in self.heads:
            # head(h): (batch, codebook_size, time)
            logits_i = head(h).transpose(1, 2)  # (batch, time, codebook_size)
            head_logits.append(logits_i.unsqueeze(1))  # (batch, 1, time, codebook_size)
```
→
```
        for head in self.heads:
            # head(h): (batch, codebook_size, time)
            logits_i = head(h).transpose(1, 2)  # (batch, time, codebook_size)
            if return_logits:
                head_logits.append(logits_i.unsqueeze(1))  # (batch, 1, time, codebook_size)
```

and:

```
        # 4. Concatenate residual heads: (batch, 8, time)
        residual_codebooks = torch.cat(pred_tokens, dim=1)
        stacked_logits = torch.cat(head_logits, dim=1)  # (batch, 8, time, 1024)
```
→
```
        # 4. Concatenate residual heads: (batch, 8, time)
        residual_codebooks = torch.cat(pred_tokens, dim=1)
        stacked_logits = (torch.cat(head_logits, dim=1)  # (batch, 8, time, 1024)
                          if return_logits else None)
```

**2d.** The three inference callers pass `return_logits=False`:

- `run_full_evaluation.py:1000–1004`:
```
            with torch.inference_mode():
                res_out = self.residual(
                    cb0_tokens=torch.from_numpy(kept).unsqueeze(0).to(self.device),
                    temperature=1.0,
                    sample=True,
                    return_logits=False,
                )
```
- `codeswitch_tts/engine/streaming_pipeline.py:273–276`:
```
            with torch.inference_mode():
                residual_out: ResidualOutput = self.residual_decoder(
                    cb0_tokens=verified_cb0,
                    temperature=temperature,
                    return_logits=False,
                )
```
- `train_draft_distill.py:1342`: `res_out = residual(cb0_tokens=draft_out.tokens, sample=True)` →
  `res_out = residual(cb0_tokens=draft_out.tokens, sample=True, return_logits=False)`.

**Verify:**

```bash
movio/.venv/bin/python - <<'EOF'
import torch
from codeswitch_tts.models.residual_decoder import ResidualDecoder
m = ResidualDecoder(); m.eval()
cb0 = torch.randint(0, 1024, (1, 24))
with torch.inference_mode():
    full = m(cb0)
    light = m(cb0, return_logits=False)
assert full.logits is not None and full.logits.shape == (1, 8, 24, 1024)
assert light.logits is None
assert torch.equal(full.full_codebooks, light.full_codebooks)
print("RESIDUAL-FLAG-OK")
EOF
grep -c "return_logits=False" run_full_evaluation.py codeswitch_tts/engine/streaming_pipeline.py train_draft_distill.py
# expect: 1 and 1 and 1
```

### Step 3 — `ASREvaluator.transcribe_batch`

Insert into `ASREvaluator` directly after `transcribe()` (which stays as the per-item fallback;
its only caller is `transcribe_batch` afterwards):

```python
    def transcribe_batch(
        self, items: Sequence[Tuple[np.ndarray, int]], batch_size: int = 8
    ) -> List[Optional[str]]:
        """Transcribe ``(wave, sample_rate)`` pairs in batches, in input order.

        Builds FRESH payload dicts per call (the pipeline's preprocess pops
        ``array``/``sampling_rate`` from the dict it is given - reusing payloads
        raises ``ValueError``) and falls back to :meth:`transcribe` item-by-item
        if the batched call fails or returns a different number of results.
        """
        if not items:
            return []
        if not self.available or self.pipe is None:
            return [None] * len(items)
        payloads = [
            {"array": np.asarray(wave, dtype=np.float32), "sampling_rate": int(sample_rate)}
            for wave, sample_rate in items
        ]
        try:
            out = self.pipe(
                payloads,
                batch_size=batch_size,
                generate_kwargs={"language": "ta", "task": "transcribe"},
            )
            if isinstance(out, dict):
                out = [out]
            if not isinstance(out, list) or len(out) != len(items):
                raise ValueError(
                    f"expected {len(items)} result(s), got "
                    f"{len(out) if isinstance(out, list) else type(out).__name__}"
                )
            return [str(r.get("text", "")).strip() for r in out]
        except Exception as exc:
            print(f"[ASR] batched transcription failed ({type(exc).__name__}: {exc}); "
                  "falling back to per-item")
            return [self.transcribe(wave, sample_rate) for wave, sample_rate in items]
```

**Verify** (the dedicated test lands in Step 5; here just prove the module still imports):

```bash
movio/.venv/bin/python -c "import run_full_evaluation; print('EVAL-IMPORT-OK')"
```

### Step 4 — Evaluation loop rework: row store, debounced save, session waves, batched ASR post-pass

**4a. Coalesce on load.** Replace:

```
    rows = load_results(results_path)
```
with:
```
    rows = _coalesce_rows(load_results(results_path))
```

**4b. Add the two helpers** immediately after `save_results` (anchor: `    tmp.replace(path)`
followed by the `# ===` Figures separator — insert between):

```python
def _coalesce_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate result rows by (config, utt_id, gate_threshold); last one wins."""
    by_key: Dict[Tuple[str, str, float], Dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("config")), str(row.get("utt_id")),
               float(row.get("gate_threshold") or 0.0))
        by_key[key] = row
    return list(by_key.values())


def _store_row(rows: List[Dict[str, Any]], row: Dict[str, Any],
               row_pos: Dict[Tuple[str, str, float], int]) -> None:
    """Insert *row*, replacing the previous row for the same (config, utt, gate).

    ``row_pos`` maps the row key to its index in *rows*; replacement happens in
    place, so the list never grows duplicates and no per-row list rebuild is
    needed.  (The old rebuild-and-append moved the refreshed row to the end every
    iteration; consumers select by ``utt_index``, not by list position, so the
    only observable difference is the physical order inside the JSONL.)
    """
    key = (str(row.get("config")), str(row.get("utt_id")),
           float(row.get("gate_threshold") or 0.0))
    pos = row_pos.get(key)
    if pos is None:
        row_pos[key] = len(rows)
        rows.append(row)
    else:
        rows[pos] = row
```

**4c. Add `_backfill_asr`** immediately after `_load_wav_mono` (anchor: the
`return wav.mean(dim=0).numpy().astype(np.float32), int(sr)` / `return None` tail followed by
`def attach_mcd_from_audio` — insert between):

```python
def _backfill_asr(
    rows: List[Dict[str, Any]],
    session_waves: Dict[Tuple[str, str, float], Tuple[np.ndarray, int]],
    asr: "ASREvaluator",
    batch_size: int = 8,
    verbose: bool = True,
) -> int:
    """Fill ASR-derived columns on rows that still lack them.

    The synthesis loop no longer calls ASR per row: it keeps each wave in
    ``session_waves`` and this post-pass transcribes them in batches.  Rows from
    an earlier (interrupted) run have no in-memory wave, so their saved
    ``audio_path`` is loaded with :func:`_load_wav_mono` - that is what makes an
    interrupted run's missing ``cer`` recoverable without re-synthesising.  Rows
    that already carry ASR output are never re-transcribed; when ASR is disabled
    the loop already wrote the explicit ``None`` columns inline.
    """
    if not asr.available:
        return 0
    targets: List[Tuple[Dict[str, Any], np.ndarray, int]] = []
    for row in rows:
        if row.get("error"):
            continue
        if row.get("cer") is not None or row.get("asr_text") is not None:
            continue
        key = (str(row.get("config")), str(row.get("utt_id")),
               float(row.get("gate_threshold") or 0.0))
        loaded = session_waves.get(key)
        if loaded is None and row.get("audio_path"):
            loaded = _load_wav_mono(str(row["audio_path"]))
        if loaded is None:
            continue
        targets.append((row, np.asarray(loaded[0], dtype=np.float32), int(loaded[1])))
    if not targets:
        return 0
    if verbose:
        print(f"[ASR] post-pass: transcribing {len(targets)} row(s) "
              f"(batch_size={batch_size})")
    texts = asr.transcribe_batch([(wave, sr) for _row, wave, sr in targets],
                                 batch_size=batch_size)
    for (row, _wave, _sr), hyp in zip(targets, texts):
        row["asr_text"] = hyp
        row["cer"], row["wer"] = ASREvaluator.score(row["normalized_text"], hyp)
        row["cer_latin"], row["wer_latin"] = ASREvaluator.score_latin(
            row["normalized_text"], hyp)
    return len(targets)
```

**4d. Loop state.** Replace:

```
    # ---------------- synthesis loop ----------------
    n_runs_since_gc = 0
```
with:
```
    # ---------------- synthesis loop ----------------
    n_runs_since_gc = 0
    rows_since_save = 0
    row_pos: Dict[Tuple[str, str, float], int] = {
        (str(r.get("config")), str(r.get("utt_id")),
         float(r.get("gate_threshold") or 0.0)): i
        for i, r in enumerate(rows)
    }
    session_waves: Dict[Tuple[str, str, float], Tuple[np.ndarray, int]] = {}
```

**4e. Wave capture instead of inline ASR.** Replace:

```
                    if asr.available:
                        hyp = asr.transcribe(wave, row["sample_rate"])
                        cer, wer = ASREvaluator.score(row["normalized_text"], hyp)
                        cer_lat, wer_lat = ASREvaluator.score_latin(row["normalized_text"], hyp)
                        row["asr_text"] = hyp
                        row["cer"] = cer
                        row["wer"] = wer
                        row["cer_latin"] = cer_lat
                        row["wer_latin"] = wer_lat
                    else:
```
with:
```
                    if asr.available:
                        # ASR runs in one batched post-pass after the loop; keep the
                        # wave so that pass does not have to re-read it from disk.
                        session_waves[(cfg, utt["id"], float(threshold))] = (
                            np.asarray(wave, dtype=np.float32), int(row["sample_rate"]))
                    else:
```

**4f. O(1) store + debounced save.** Replace:

```
                # replace any previous row for (cfg, utt, gate) then persist
                rows = [r for r in rows
                        if not (r.get("config") == cfg and r.get("utt_id") == utt["id"]
                                and float(r.get("gate_threshold") or 0.0) == float(threshold))]
                rows.append(row)
                save_results(results_path, rows)
                done.add((cfg, utt["id"], float(threshold)))
```
with:
```
                # replace any previous row for (cfg, utt, gate); the full-file JSONL
                # rewrite is debounced to every 20 rows (a crash re-synthesises at most
                # 19 rows; the ASR post-pass flushes again below).
                _store_row(rows, row, row_pos)
                rows_since_save += 1
                if rows_since_save >= 20:
                    rows_since_save = 0
                    save_results(results_path, rows)
                done.add((cfg, utt["id"], float(threshold)))
```

**4g. Post-pass tail.** Replace:

```
                n_runs_since_gc += 1
                if n_runs_since_gc >= 5:
                    n_runs_since_gc = 0
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    return _finish(rows, args, configs, results_path, figures_dir, t_all)
```
with:
```
                n_runs_since_gc += 1
                if n_runs_since_gc >= 5:
                    n_runs_since_gc = 0
                    maybe_free_memory()

    # ---------------- batched ASR post-pass ----------------
    if rows_since_save:
        save_results(results_path, rows)
        rows_since_save = 0
    n_filled = _backfill_asr(rows, session_waves, asr)
    if n_filled:
        save_results(results_path, rows)
        print(f"[ASR] post-pass: CER/WER written for {n_filled} row(s)")

    return _finish(rows, args, configs, results_path, figures_dir, t_all)
```

**Verify (behaviour probe for the post-pass, run from the repo root):**

```bash
movio/.venv/bin/python - <<'EOF'
import numpy as np, soundfile as sf
import run_full_evaluation as R

wav_path = "/tmp/opencode/backfill_probe.wav"
sf.write(wav_path, np.zeros(16000, dtype=np.float32), 16000)

class _StubASR:
    available = True
    def transcribe_batch(self, items, batch_size=8):
        assert len(items) == 2, len(items)          # only u1 (session) + u2 (disk)
        return ["hyp0", "hyp1"]

rows = [
    {"config": "A", "utt_id": "u1", "gate_threshold": 0.0, "error": None,
     "normalized_text": "hello world", "cer": None, "asr_text": None},
    {"config": "A", "utt_id": "u2", "gate_threshold": 0.0, "error": None,
     "normalized_text": "hello world", "cer": None, "asr_text": None,
     "audio_path": wav_path},
    {"config": "A", "utt_id": "u3", "gate_threshold": 0.0, "error": None,
     "normalized_text": "hello world", "cer": 0.5, "asr_text": "done"},
    {"config": "A", "utt_id": "u4", "gate_threshold": 0.0, "error": "boom",
     "normalized_text": "hello world", "cer": None, "asr_text": None},
    {"config": "A", "utt_id": "u5", "gate_threshold": 0.0, "error": None,
     "normalized_text": "hello world", "cer": None, "asr_text": None,
     "audio_path": "/tmp/opencode/does-not-exist.wav"},
]
waves = {("A", "u1", 0.0): (np.zeros(16000, dtype=np.float32), 16000)}
n = R._backfill_asr(rows, waves, _StubASR(), verbose=False)
assert n == 2, n
assert rows[0]["asr_text"] == "hyp0" and rows[0]["cer"] is not None
assert rows[1]["asr_text"] == "hyp1"                    # disk fallback
assert rows[2]["cer"] == 0.5 and rows[2]["asr_text"] == "done"
assert rows[3]["cer"] is None and "asr_text" not in rows[3]
assert rows[4]["cer"] is None and "asr_text" not in rows[4]
assert R._coalesce_rows([{"config": "A", "utt_id": "u", "gate_threshold": 0.0},
                         {"config": "A", "utt_id": "u", "gate_threshold": 0.0}]) == [
       {"config": "A", "utt_id": "u", "gate_threshold": 0.0}]
print("BACKFILL-OK")
EOF
```

### Step 5 — Tests (+3)

**5a. `tests/test_codeswitch_tts.py`** — insert after `test_residual_decoder_forward` (anchor:
it ends with `assert torch.equal(res_out.full_codebooks[:, 0, :], cb0_tokens)`):

```python
def test_residual_forward_return_logits_flag():
    decoder = ResidualDecoder()
    decoder.eval()
    cb0_tokens = torch.randint(0, 1024, (1, 24))
    with torch.inference_mode():
        full = decoder(cb0_tokens, sample=False)
        light = decoder(cb0_tokens, sample=False, return_logits=False)
    assert full.logits is not None
    assert full.logits.shape == (1, 8, 24, 1024)
    assert light.logits is None
    # The codebooks the inference callers actually consume are untouched.
    assert torch.equal(full.full_codebooks, light.full_codebooks)
    assert torch.equal(full.residual_codebooks, light.residual_codebooks)
```

and after `test_audio_crossfade_and_filter` (anchor: it ends with
`assert len(filtered) == len(blended)`):

```python
def test_memory_guard_only_frees_under_pressure(monkeypatch):
    import codeswitch_tts.utils.memory as memory

    calls = []
    monkeypatch.setattr(memory.gc, "collect", lambda: calls.append("collect"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    # Plenty of free device memory: the guard must be a no-op...
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (10_000, 16_000))
    assert memory.maybe_free_memory() is False
    assert calls == []

    # ...but it must collect + empty the cache when the device is nearly full.
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (500, 16_000))
    assert memory.maybe_free_memory() is True
    assert calls == ["collect", "empty_cache"]

    # CPU-only: nothing to do.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert memory.maybe_free_memory() is False
    assert calls == ["collect", "empty_cache"]
```

**5b. `tests/test_evaluation_suite.py`** — insert after `test_asr_score_zero_for_identical`
(anchor: it ends with `assert cer2 is None and wer2 is None`):

```python
def test_asr_transcribe_batch_falls_back_with_fresh_dicts():
    class _BatchFailsThenPops:
        """Batch call fails like a transient pipeline error; per-item calls succeed.

        Each call pops ``array``/``sampling_rate`` from the payload dict — the real
        pipeline's preprocess does exactly this — so a fallback that reused the
        batch payloads would crash instead of returning texts.
        """

        def __init__(self):
            self.batch_calls = 0
            self.item_calls = 0

        def __call__(self, inputs, **kwargs):
            if isinstance(inputs, list):
                self.batch_calls += 1
                for payload in inputs:
                    payload.pop("array", None)
                    payload.pop("sampling_rate", None)
                raise RuntimeError("synthetic batch failure")
            self.item_calls += 1
            assert isinstance(inputs, dict)
            assert inputs["array"].dtype == np.float32
            assert inputs["sampling_rate"] == 16000
            inputs.pop("array")
            inputs.pop("sampling_rate")
            return {"text": f"item-{self.item_calls}"}

    asr = R.ASREvaluator("unused", device="cpu", enabled=False, verbose=False)
    asr.available = True
    asr.pipe = _BatchFailsThenPops()
    items = [(np.zeros(16, dtype=np.float32), 16000) for _ in range(3)]
    texts = asr.transcribe_batch(items, batch_size=2)
    assert texts == ["item-1", "item-2", "item-3"]
    assert asr.pipe.batch_calls == 1
    assert asr.pipe.item_calls == 3
```

**Verify:**

```bash
movio/.venv/bin/python -m pytest tests/test_codeswitch_tts.py -q -p no:cacheprovider -k "return_logits or memory_guard"
movio/.venv/bin/python -m pytest tests/test_evaluation_suite.py -q -p no:cacheprovider -k "transcribe_batch"
# expect: 2 passed and 1 passed
```

### Step 6 — Update the hand-maintained test count

The suite now has **previous count + 3** (008's +2 and 009's +9 landed earlier in the locked
order, and plans 001–007 added tests of their own — read the real number, never predict the
total). Run the full suite, read the summary number, and update `KAGGLE_RUN.md`:

- From: ``* Local test suite: `python -m pytest tests/ -q` (98 tests, no GPU and no network needed).``
- To: the same line with the real total (under the full locked order the final count is 123,
  but the pytest summary — not this arithmetic — is the source of truth).

**Verify:** `grep -n "Local test suite" KAGGLE_RUN.md` — the quoted count equals the pytest summary.

### Step 7 — Full suite + final sweep

```bash
movio/.venv/bin/python -m pytest tests/ -q -p no:cacheprovider
# expect: previous count + 3. Delta must be exactly +3 (pytest summary is the source of truth).

git diff --stat
# expect exactly: codeswitch_tts/utils/memory.py (new), codeswitch_tts/models/residual_decoder.py,
#   codeswitch_tts/engine/streaming_pipeline.py, run_full_evaluation.py, train_draft_distill.py,
#   kaggle_demo.py, CodeSwitch_TTS_Kaggle.ipynb, tests/test_codeswitch_tts.py,
#   tests/test_evaluation_suite.py, KAGGLE_RUN.md — nothing else.
```

## Test plan

| Test | File | Count |
|---|---|---|
| `test_residual_forward_return_logits_flag` | `tests/test_codeswitch_tts.py` | +1 |
| `test_memory_guard_only_frees_under_pressure` | `tests/test_codeswitch_tts.py` | +1 |
| `test_asr_transcribe_batch_falls_back_with_fresh_dicts` | `tests/test_evaluation_suite.py` | +1 |

No network, no GPU (the guard test monkeypatches `torch.cuda`; the ASR test uses a stub pipe).
The `_backfill_asr` post-pass is covered by the Step 4 heredoc probe (session wave + disk
fallback + already-filled + error + unrecoverable rows).

## Done criteria

- [ ] `codeswitch_tts/utils/memory.py` exists; the four call sites use `maybe_free_memory()`;
      `import gc` gone from eval/train (`GUARD-OK`, `IMPORTS-OK` printed).
- [ ] Notebook diff is exactly cell 10, lines 47–48 (`NOTEBOOK-DIFF-OK`); git diff shows one
      `-`/`+` pair of source entries.
- [ ] `RESIDUAL-FLAG-OK`; three inference callers pass `return_logits=False`; training untouched.
- [ ] `transcribe_batch` in place; `BACKFILL-OK`; loop uses `_store_row` + 20-row debounce +
      post-pass tail.
- [ ] 3 new tests pass; full suite delta is exactly +3; `KAGGLE_RUN.md` count updated.
- [ ] `git diff --stat` lists exactly the ten files; nothing pushed.

## STOP conditions

1. HEAD history lacks both `d69b399` and `f689736`, or tracked files are dirty → stop and report.
2. Any old-string count ≠ 1 before replacement → stop (drift; re-check which plan touched it).
3. `NOTEBOOK-DIFF-OK` not printed, or the git diff shows anything beyond the two expected
   `[+-]` source lines → `git checkout -- CodeSwitch_TTS_Kaggle.ipynb` and stop.
4. `BACKFILL-OK` not printed → stop (the post-pass logic or its helpers drifted).
5. Full-suite delta ≠ +3, or `git diff --stat` lists a file outside the ten → stop.
6. `movio/.venv/bin/python -c "import run_full_evaluation"` fails after any edit → stop.

## Maintenance notes

- Line numbers quoted are from `d69b399`; everything is content-anchored. The fresh-run fix
  `f689736` adds one line at `run_full_evaluation.py:3196`, so locations after that shift +1.
- `session_waves` holds one float32 wave per synthesised row — ≈45 MB at the full 60-utterance,
  3-config, no-gate scale (3 × 60 × ~0.25 MB) — bounded and freed when `main()` returns.
- After this plan, a `--skip-asr` run can be repaired by a later run with ASR enabled: the
  post-pass re-transcribes rows whose `cer`/`asr_text` are still missing, using the saved wavs.
  Without `--no-save-audio` that is automatic; with it, waves never hit disk, so those columns
  stay `None` (same as before for interrupted rows).
- The per-row console line prints `cer=None` during the loop now (ASR moved after it); the
  post-pass prints the summary. Per-row timings inside the JSONL are unaffected (they are
  measured inside the synth methods, before ASR in both old and new code); only the printed
  `elapsed` values drop.
- If a future `--asr-batch-size` flag is wanted, thread it into `_backfill_asr(batch_size=…)` —
  the plumbing exists, deliberately not exposed now.
- Don't reintroduce a fixed-cadence cache flush: `maybe_free_memory()` thresholds are the
  intended policy hook (0.90 default).
