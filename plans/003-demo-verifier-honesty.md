# Plan 003: Make the demo verifier honest — real teacher-forced CB0 probabilities or an explicit failure

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**: `git diff --stat d69b399..HEAD -- codeswitch_tts/engine/speculative_engine.py codeswitch_tts/engine/streaming_pipeline.py codeswitch_tts/engine/__init__.py kaggle_demo.py CodeSwitch_TTS_Kaggle.ipynb tests/test_codeswitch_tts.py KAGGLE_RUN.md`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: MED
- **Depends on**: none
- **Category**: bug
- **Planned at**: commit `d69b399`, 2026-10-19

## Why this matters

The verifier is the paper's core mechanism, but the demo path's verification
branch always throws and silently falls back to `torch.randn` logits — so
every `ChunkedStreamingPipeline` run (the notebook demo, `kaggle_demo.py`)
reports coin-flip "CB0 acceptance" numbers that cannot fail. The evaluation
suite is *not* affected (it uses its own working `TeacherForcedScorer`), but
the demo telemetry advertised to reviewers is fabricated. This plan ports the
evaluation scorer's verified teacher-forced layout into the engine, deletes
the random fallback, and makes "verifier unavailable" an explicit,
None-valued outcome instead of a silent fabrication.

Evidence: `plans/research/findings-2026-10-18.md`, correctness finding C01
(parler-tts main's `ParlerTTSForCausalLM.forward` reshapes `(1, 1+K)` as
`(-1, 9, L)` and always raises; the bare `except` converts that into random
logits; the row layout the working scorer uses is at
`run_full_evaluation.py:522-553`).

## Current state

Files involved:

- `codeswitch_tts/engine/speculative_engine.py` — the bug: `get_target_probabilities` (`:238-266`) and `verify_chunk` (`:268-349`).
- `codeswitch_tts/engine/streaming_pipeline.py` — Mode A call site (`:150-214`), result dataclasses (`:47-76`), `synthesize` averaging (`:334-336`).
- `codeswitch_tts/engine/__init__.py` — package exports.
- `run_full_evaluation.py` — **read-only reference**: the working `TeacherForcedScorer` (`:451-595`) whose layout must be mirrored.
- `kaggle_demo.py`, `CodeSwitch_TTS_Kaggle.ipynb` — print `cb0_acceptance_rate` with `:.1f` (will crash on `None`).
- `tests/test_codeswitch_tts.py` — engine tests; `tests` dir also has `test_corpus_builder.py`, `test_evaluation_suite.py`, `test_wandb_push.py` (do not touch the last one).
- `KAGGLE_RUN.md` — line 427 test count.

### Defect: the parler verification branch can never work and fails silently

`codeswitch_tts/engine/speculative_engine.py:246-266` (today):

```python
        full_seq = torch.cat([prefix_tokens, draft_tokens], dim=1)
        b, total_len = full_seq.shape
        draft_len = draft_tokens.shape[1]

        if self.backend == "indic_parler_tts":
            try:
                decoder_outputs = self.ar_model.decoder(
                    input_ids=full_seq,
                    return_dict=True,
                )
                hidden = decoder_outputs.last_hidden_state
                logits = self.ar_model.decoder.lm_heads[0](hidden)
                draft_logits = logits[:, -draft_len:, :]
            except Exception:
                draft_logits = torch.randn(b, draft_len, 1024, device=self.device)
        else:
            draft_logits = self.ar_model(full_seq)[:, -draft_len:, :]

        scaled_logits = draft_logits / max(temperature, 1e-5)
        probs = F.softmax(scaled_logits, dim=-1)
        return probs
```

Two failures (both verified live in the audit): the inner decoder reshapes
`(1, 1+K)` to `(-1, 9, L)` and raises for any non-empty input; and even if it
passed, `ParlerTTSCausalLMOutputWithCrossAttentions` has no
`last_hidden_state`. The bare `except` turns every failure into uniform random
logits, which drive `F.softmax` and the acceptance loop. The working layout
the engine must adopt (`run_full_evaluation.py:521-553`, read-only reference):

```python
    @torch.inference_mode()
    def _forward_probs(
        self, ctx: Dict[str, Any], prefix_cb0: np.ndarray, chunk_cb0: np.ndarray
    ) -> Optional[np.ndarray]:
        try:
            tokens = np.concatenate([prefix_cb0, chunk_cb0]).astype(np.int64)
            length = tokens.shape[0] + 1
            ids = torch.full((self.num_codebooks, length), self.bos_id,
                             dtype=torch.long, device=self.engine.device)
            ids[0, 1:] = torch.from_numpy(tokens).to(self.engine.device)
            kwargs: Dict[str, Any] = {
                "input_ids": ids,
                "encoder_hidden_states": ctx["encoder_hidden_states"],
                "encoder_attention_mask": ctx.get("encoder_attention_mask"),
                "use_cache": False,
                "return_dict": True,
            }
            if not self.prompt_cross_attention and ctx.get("prompt_hidden_states") is not None:
                kwargs["prompt_hidden_states"] = ctx["prompt_hidden_states"]
                kwargs["prompt_attention_mask"] = ctx.get("prompt_attention_mask")
            out = self.decoder(**kwargs)
            logits = out.logits[0].float()
            start = ctx.get("offset", 0) + prefix_cb0.shape[0]
            logits = logits[start: start + chunk_cb0.shape[0], : self.vocab]
            probs = torch.softmax(logits, dim=-1)
            return probs.detach().cpu().numpy().astype(np.float32)
        except Exception as exc:
            ...
            return None
```

Key derivations in that class that the port must copy
(`run_full_evaluation.py:459-486`): `num_codebooks = int(getattr(self.decoder,
"num_codebooks", 9))`; `vocab` = `ar.audio_encoder.config.codebook_size` (1024);
`bos_id` = `generation_config.bos_token_id or decoder_start_token_id or 1025`;
the conditioning builder `build_context` (`:488-518`) — mirrored below.

### The Mode A call site

`codeswitch_tts/engine/streaming_pipeline.py:150-214`: Mode A generates the
whole utterance (`generate_authentic_speech`, `:153`), then per chunk runs the
draft and calls `verify_chunk(draft_tokens=..., draft_probs=...)` with **no
context** (`:184-190`) and yields
`cb0_acceptance_rate=verif_res.acceptance_rate` (`:208`). Mode B (`:253-258`)
passes `prefix_tokens` and no context — it must keep working exactly as now
(standalone backend ignores context).

Consumers of the acceptance value: `synthesize` averaging
(`streaming_pipeline.py:334-336`), `kaggle_demo.py:158` and `:185`, notebook
cells 10, 12, 14 (all format `:.1f` — they crash on `None`).

Facts that shape the fix:

- `parler_tts` is **not installed** in this environment (`movio/.venv/bin/python
  -c "import parler_tts"` → `ModuleNotFoundError`) — offline tests must build
  engines with `SpeculativeEngine.__new__` and fake decoders; no real model can
  be exercised locally.
- `verify_chunk` callers (verified): `streaming_pipeline.py:186` (Mode A),
  `:253` (Mode B), `tests/test_codeswitch_tts.py:81`. No other callers.
- `get_target_probabilities` callers: only `verify_chunk` (`:294`).
- Notebook serialization is byte-exact with
  `json.dumps(nb, indent=1, ensure_ascii=False) + "\n"` (verified live) — edit
  script below preserves a minimal diff.

## Commands you will need

| Purpose        | Command                                                            | Expected on success |
|----------------|--------------------------------------------------------------------|---------------------|
| Tests (full)   | `movio/.venv/bin/python -m pytest tests/ -q -p no:cacheprovider`   | previous count + 3 new tests, all pass |
| Tests (target) | `movio/.venv/bin/python -m pytest tests/test_codeswitch_tts.py -q -p no:cacheprovider` | all pass |
| Syntax gates   | `movio/.venv/bin/python -m py_compile codeswitch_tts/engine/speculative_engine.py codeswitch_tts/engine/streaming_pipeline.py kaggle_demo.py` | exit 0 |
| Notebook parse | `movio/.venv/bin/python -c "import json; json.load(open('CodeSwitch_TTS_Kaggle.ipynb')); print('ok')"` | `ok` |
| Drift check    | `git diff --stat d69b399..HEAD -- codeswitch_tts/engine/speculative_engine.py codeswitch_tts/engine/streaming_pipeline.py codeswitch_tts/engine/__init__.py kaggle_demo.py CodeSwitch_TTS_Kaggle.ipynb tests/test_codeswitch_tts.py KAGGLE_RUN.md` | empty output before starting |

Note: the project venv is uv-managed and its `pytest` console script has a
broken shebang — always invoke `movio/.venv/bin/python -m pytest` from the
repo root.

## Scope

**In scope** (the only files you should modify):

- `codeswitch_tts/engine/speculative_engine.py`
- `codeswitch_tts/engine/streaming_pipeline.py`
- `codeswitch_tts/engine/__init__.py`
- `kaggle_demo.py` — only the two acceptance print sites.
- `CodeSwitch_TTS_Kaggle.ipynb` — only the three acceptance print lines.
- `tests/test_codeswitch_tts.py`
- `KAGGLE_RUN.md` — only the test-count line 427.

**Out of scope** (do NOT touch, even though they look related):

- `run_full_evaluation.py` — read-only reference; the eval suite is correct.
- The timing constants `residual_ms = 22.5` / `dac_ms = 18.0` and the TTFA
  formula in Mode A — plan 004 rewrites those semantics; do not touch them here.
- Refactoring `TeacherForcedScorer` to share the new engine helper — a
  deliberate follow-up, not this plan.
- Mode B's flow (`parallel speculative decoding` standalone path) — only the
  `context` parameter pass-through changes nothing for it.

## Git workflow

- Branch: `git checkout -b advisor/003-demo-verifier-honesty`.
- Suggested commits (repo house style — descriptive sentence case):
  1. `Give the demo verifier real teacher-forced CB0 probabilities`
  2. `Report demo CB0 acceptance as None when the verifier is unavailable`
  3. `Guard demo and notebook acceptance prints against missing verification`
- Do NOT push or open a PR.

## Steps

### Step 1: Add `VerifierUnavailable` and export it

In `codeswitch_tts/engine/speculative_engine.py`, above `class SpeculativeEngine`
(after `StandaloneARVerifier`, ~line 83), add:

```python
class VerifierUnavailable(RuntimeError):
    """The target evaluator cannot produce real CB0 probabilities.

    Verification must never silently fall back to random logits: a fabricated
    acceptance rate is worse than an explicitly absent one.
    """
```

In `codeswitch_tts/engine/__init__.py`, add the name to both the import list
and `__all__`:

```python
from codeswitch_tts.engine.speculative_engine import (
    SpeculativeEngine,
    SpeculativeVerificationResult,
    VerifierUnavailable,
)
```

**Verify**: `movio/.venv/bin/python -c "from codeswitch_tts.engine import VerifierUnavailable; print('ok')"` → `ok`.

### Step 2: Add the engine's prompt-context builder

Insert after `get_cached_description_encoder_outputs` (ends at `:190`), before
`generate_authentic_speech`:

```python
    @torch.inference_mode()
    def build_prompt_context(self, prompt_text: str) -> Dict[str, Any]:
        """Teacher-forced conditioning for CB0 verification (parler backend).

        Mirrors the working evaluation scorer's ``build_context``
        (run_full_evaluation.py:488-518): returns the encoder states for the
        voice description plus either the prompt hidden states
        (``prompt_cross_attention=False``: prompt prepended by the decoder,
        ``offset`` = prompt length) or the prompt concatenated onto the encoder
        states.  Raises ``VerifierUnavailable`` when the model's interfaces
        cannot provide it.
        """
        if self.backend != "indic_parler_tts":
            raise VerifierUnavailable(f"backend {self.backend!r} has no teacher-forced path")
        try:
            ar = self.ar_model
            prompt_clean = prompt_text.strip()
            if not prompt_clean.endswith((".", "!", "?", "।")):
                prompt_clean += "."
            desc = self.get_cached_description_encoder_outputs(self.config.voice_description)
            prompt_ids = self.tokenizer(prompt_clean, return_tensors="pt").to(self.device)
            prompt_cross_attention = bool(getattr(ar, "prompt_cross_attention", False))
            ctx: Dict[str, Any] = {
                "offset": 0,
                "encoder_hidden_states": desc,
                "encoder_attention_mask": None,
            }
            if prompt_cross_attention:
                embedded = ar.embed_prompts(prompt_ids.input_ids)
                positions = ar.embed_positions(embedded, 0)
                prompt_hidden = (embedded + positions.to(embedded.device)).to(desc.dtype)
                ones = torch.ones(desc.shape[:2], device=desc.device,
                                  dtype=prompt_ids.attention_mask.dtype)
                ctx["encoder_hidden_states"] = torch.cat([desc, prompt_hidden], dim=1)
                ctx["encoder_attention_mask"] = torch.cat([ones, prompt_ids.attention_mask], dim=1)
            else:
                ctx["prompt_hidden_states"] = ar.embed_prompts(prompt_ids.input_ids).to(desc.dtype)
                ctx["prompt_attention_mask"] = prompt_ids.attention_mask
                ctx["offset"] = int(prompt_ids.input_ids.shape[1])
            return ctx
        except VerifierUnavailable:
            raise
        except Exception as exc:
            raise VerifierUnavailable(
                f"prompt context build failed: {type(exc).__name__}: {exc}") from exc
```

**Verify**: `movio/.venv/bin/python -m py_compile codeswitch_tts/engine/speculative_engine.py` → exit 0.

### Step 3: Replace `get_target_probabilities` (delete the random fallback)

Replace the method body (`:238-266`) with:

```python
    @torch.inference_mode()
    def get_target_probabilities(
        self,
        prefix_tokens: torch.Tensor,
        draft_tokens: torch.Tensor,
        temperature: float = 1.0,
        context: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Parallel target probabilities P(y_t | y_<t, X) for the draft sequence.

        Parler path: teacher-forced forward through the inner decoder using the
        same (num_codebooks, L) token grid as the evaluation suite's
        TeacherForcedScorer (run_full_evaluation.py:521-553): row 0 is
        ``[bos, prefix..., draft...]``; logits at position p predict the token
        at p+1; ``context["offset"]`` accounts for prompt tokens the model
        prepends to its own logits.  Raises ``VerifierUnavailable`` when the
        target interface cannot be used - never fabricates logits.
        """
        full_seq = torch.cat([prefix_tokens, draft_tokens], dim=1)
        b, total_len = full_seq.shape
        draft_len = draft_tokens.shape[1]

        if self.backend == "indic_parler_tts":
            if context is None:
                raise VerifierUnavailable(
                    "indic_parler_tts verification needs a prompt context; call "
                    "build_prompt_context(prompt_text) once per utterance and pass it "
                    "as verify_chunk(context=...) / get_target_probabilities(context=...)")
            try:
                ar = self.ar_model
                num_codebooks = int(getattr(ar.decoder, "num_codebooks", 9))
                gc_cfg = getattr(ar, "generation_config", None)
                bos_id = int(
                    getattr(gc_cfg, "bos_token_id", None)
                    or getattr(gc_cfg, "decoder_start_token_id", None)
                    or 1025
                )
                tokens = full_seq[0].to(device=self.device, dtype=torch.long)
                length = int(tokens.shape[0]) + 1
                ids = torch.full((num_codebooks, length), bos_id,
                                 dtype=torch.long, device=self.device)
                ids[0, 1:] = tokens
                kwargs: Dict[str, Any] = {
                    "input_ids": ids,
                    "encoder_hidden_states": context["encoder_hidden_states"],
                    "encoder_attention_mask": context.get("encoder_attention_mask"),
                    "use_cache": False,
                    "return_dict": True,
                }
                if context.get("prompt_hidden_states") is not None:
                    kwargs["prompt_hidden_states"] = context["prompt_hidden_states"]
                    kwargs["prompt_attention_mask"] = context.get("prompt_attention_mask")
                out = ar.decoder(**kwargs)
                logits = out.logits[0].float()
                vocab = int(
                    getattr(getattr(getattr(ar, "audio_encoder", None), "config", None),
                            "codebook_size", 1024)
                )
                start = int(context.get("offset", 0)) + int(prefix_tokens.shape[1])
                draft_logits = logits[start:start + draft_len, :vocab]
                if draft_logits.shape[0] != draft_len:
                    raise VerifierUnavailable(
                        f"target decoder returned {logits.shape[0]} logit positions; need at "
                        f"least {start + draft_len} (offset={context.get('offset', 0)}, "
                        f"prefix={prefix_tokens.shape[1]}, draft={draft_len})")
                draft_logits = draft_logits.unsqueeze(0)   # back to (batch=1, K, vocab)
            except VerifierUnavailable:
                raise
            except Exception as exc:
                raise VerifierUnavailable(
                    f"target decoder forward failed: {type(exc).__name__}: {exc}") from exc
        else:
            draft_logits = self.ar_model(full_seq)[:, -draft_len:, :]

        scaled_logits = draft_logits / max(temperature, 1e-5)
        probs = F.softmax(scaled_logits, dim=-1)
        return probs
```

**Verify**: `movio/.venv/bin/python -m py_compile codeswitch_tts/engine/speculative_engine.py` → exit 0; then `grep -n "torch.randn(b, draft_len" codeswitch_tts/engine/speculative_engine.py` → no matches.

### Step 4: Thread `context` through `verify_chunk`

In the same file:

4a. Add the parameter: after `temperature: float = 1.0,` in `verify_chunk`
(`:274`) add `context: Optional[Dict[str, Any]] = None,`.
4b. Extend its docstring with one sentence: "``context`` is the teacher-forced
conditioning built by ``build_prompt_context`` (required on the
``indic_parler_tts`` backend; the standalone backend ignores it). Raises
``VerifierUnavailable`` instead of fabricating logits when the target interface
fails."
4c. Pass it through (`:294-298`): add `context=context,` to the
`self.get_target_probabilities(...)` call.

**Verify**: `movio/.venv/bin/python -m pytest tests/test_codeswitch_tts.py -q -p no:cacheprovider` → all pass (`test_speculative_verification` exercises the standalone backend with no context and must be green).

### Step 5: Stream pipeline — Mode A context, None-aware acceptance

In `codeswitch_tts/engine/streaming_pipeline.py`:

5a. Imports: extend the typing import (`:18`) to
`from typing import Any, Dict, Generator, List, Optional, Tuple, Union` and the
engine import (`:27`) to
`from codeswitch_tts.engine.speculative_engine import SpeculativeEngine, SpeculativeVerificationResult, VerifierUnavailable`.

5b. Dataclasses (`:55`, `:74`) — both fields become nullable:

```python
    cb0_acceptance_rate: Optional[float]   # Codebook-0 speculative acceptance rate (%); None when unverifiable
```

(add the same comment style to `PipelineBenchmarkResult.cb0_acceptance_rate`).

5c. In `__init__`, next to `self._fallback_warned = False` (`:110`), add
`self._verifier_warned = False`, and add this helper after `__init__`:

```python
    def _warn_verifier_unavailable(self, exc: Exception) -> None:
        """One warning per pipeline: never print a fabricated acceptance number."""
        if not self._verifier_warned:
            print("[ChunkedStreamingPipeline] ⚠ CB0 acceptance verification unavailable: "
                  f"{exc}. Reporting cb0_acceptance_rate=None instead of a fabricated number.")
            self._verifier_warned = True
```

5d. Mode A: immediately before the `# Divide full natural speech into
1-second chunks` comment (`:165`), insert:

```python
                # Real teacher-forced conditioning for the CB0 verifier.  Built
                # once per utterance; when it fails the acceptance telemetry is
                # reported as None instead of a fabricated number.
                verifier_ctx: Optional[Dict[str, Any]] = None
                try:
                    verifier_ctx = self.speculative_engine.build_prompt_context(
                        norm_res.normalized_text
                    )
                except VerifierUnavailable as exc:
                    self._warn_verifier_unavailable(exc)
```

5e. Replace the verifier call block (`:184-190`) with:

```python
                    t_verif_0 = time.perf_counter()
                    cb0_acceptance: Optional[float] = None
                    if verifier_ctx is not None:
                        try:
                            with torch.inference_mode():
                                verif_res = self.speculative_engine.verify_chunk(
                                    draft_tokens=draft_out.tokens,
                                    draft_probs=draft_out.probabilities,
                                    context=verifier_ctx,
                                )
                            cb0_acceptance = float(verif_res.acceptance_rate)
                        except VerifierUnavailable as exc:
                            self._warn_verifier_unavailable(exc)
                            verifier_ctx = None  # stop retrying; keep the rest honest
                    verify_ms = (time.perf_counter() - t_verif_0) * 1000.0
```

5f. In the `yield StreamedAudioChunk(...)` (`:202-213`), change
`cb0_acceptance_rate=verif_res.acceptance_rate,` to
`cb0_acceptance_rate=cb0_acceptance,`.

5g. In `synthesize` (`:334-336`), replace the averaging with:

```python
        rates = [c.cb0_acceptance_rate for c in chunks if c.cb0_acceptance_rate is not None]
        avg_acceptance: Optional[float] = (sum(rates) / len(rates)) if rates else None
```

**Verify**: `movio/.venv/bin/python -m pytest tests/test_codeswitch_tts.py -q -p no:cacheprovider` → all pass, including `test_end_to_end_streaming_pipeline` (Mode B path, standalone backend, acceptance still numeric).

### Step 6: None guards in `kaggle_demo.py`

6a. `:152-159` — replace the per-chunk print block with:

```python
        for chk in benchmark_res.chunks:
            acc_txt = "n/a" if chk.cb0_acceptance_rate is None else f"{chk.cb0_acceptance_rate:.1f}"
            print(
                f"       * Chunk {chk.chunk_index + 1}/{benchmark_res.chunks_count}: "
                f"Latency={chk.chunk_latency_ms:.1f}ms (TTFA={chk.ttfa_ms:.1f}ms) | "
                f"Draft={chk.draft_ms:.1f}ms | Verify={chk.verify_ms:.1f}ms | "
                f"Res={chk.residual_ms:.1f}ms | DAC={chk.dac_ms:.1f}ms | "
                f"CB0 Acceptance={acc_txt}%"
            )
```

6b. `:178-187` — before `row = (`, add:
`acc_txt = "n/a" if res.cb0_acceptance_rate is None else f"{res.cb0_acceptance_rate:.1f}"`,
and change the last row fragment `f"{res.cb0_acceptance_rate:<12.1f}%"` to
`f"{acc_txt:<12}%"`.

**Verify**: `movio/.venv/bin/python -m py_compile kaggle_demo.py` → exit 0; `grep -n "cb0_acceptance_rate:." kaggle_demo.py` → no matches (no remaining raw `:.1f` on the field).

### Step 7: None guards in the notebook (cells 10, 12, 14)

Run this script from the repo root (it replaces exact source lines by content,
preserving the notebook's byte-exact serialization — verified:

`json.dumps(nb, indent=1, ensure_ascii=False) + "\n"` reproduces the file):

```python
import json

path = "CodeSwitch_TTS_Kaggle.ipynb"
nb = json.load(open(path, encoding="utf-8"))

replacements = [
    (
        'print(f" CB0 Acceptance: {result.cb0_acceptance_rate:.1f}%")',
        ['acc_txt = "n/a (verifier unavailable)" if result.cb0_acceptance_rate is None '
         'else f"{result.cb0_acceptance_rate:.1f}%"',
         'print(f" CB0 Acceptance: {acc_txt}")'],
    ),
    (
        "    print(f\"{sample['category']:<22} | {res.ttfa_ms:<10.1f} | {res.total_latency_ms:<11.1f} | {res.duration_sec:<10.2f} | {res.real_time_factor:<8.3f} | {res.cb0_acceptance_rate:.1f}%\")",
        ["    acc_txt = \"n/a\" if res.cb0_acceptance_rate is None else f\"{res.cb0_acceptance_rate:.1f}%\"",
         "    print(f\"{sample['category']:<22} | {res.ttfa_ms:<10.1f} | {res.total_latency_ms:<11.1f} | {res.duration_sec:<10.2f} | {res.real_time_factor:<8.3f} | {acc_txt}\")"],
    ),
    (
        'print(f"Acceptance: {custom_res.cb0_acceptance_rate:.1f}%")',
        ['acc_txt = "n/a (verifier unavailable)" if custom_res.cb0_acceptance_rate is None '
         'else f"{custom_res.cb0_acceptance_rate:.1f}%"',
         'print(f"Acceptance: {acc_txt}")'],
    ),
]

for c in nb["cells"]:
    src = c["source"]
    new_src = []
    for line in src:
        matched = False
        for old, new_lines in replacements:
            if line.rstrip("\n") == old:
                new_src.extend(l + "\n" for l in new_lines)
                matched = True
                break
        if not matched:
            new_src.append(line)
    c["source"] = new_src

open(path, "w", encoding="utf-8").write(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print("notebook updated")
```

After running, confirm each replacement hit exactly once (the script above
replaces only on exact match; verify with the grep below — the output must
show all three guarded forms and no unguarded `:.1f}` form on the field).

**Verify**:
`movio/.venv/bin/python -c "import json; nb=json.load(open('CodeSwitch_TTS_Kaggle.ipynb')); t=''.join(''.join(c['source']) for c in nb['cells']); assert 'cb0_acceptance_rate:.1f}%' not in t; assert t.count('acc_txt') >= 3; print('ok')"` → `ok`;
`git diff --stat CodeSwitch_TTS_Kaggle.ipynb` → 1 file changed, a handful of +/- lines (not a whole-file rewrite).

### Step 8: Engine tests (fake decoder; no parler needed)

In `tests/test_codeswitch_tts.py`:

8a. Header: change `:12` to
`from codeswitch_tts.engine.speculative_engine import SpeculativeEngine, VerifierUnavailable`
and add `from types import SimpleNamespace` near the imports.

8b. Append after `test_speculative_verification` (ends `:84`):

```python
class _ScriptedDecoder:
    """Mimics the layout the real inner decoder produces.

    ``input_ids`` arrives as the (num_codebooks, L) teacher-forcing grid.  The
    real model prepends the prompt to its own logits when prompt_cross_attention
    is off, so the scripted decoder emits ``pad`` extra positions to mimic
    ``context["offset"]``.  CB0 logits peak on token 7 everywhere.
    """

    num_codebooks = 9

    def __init__(self, vocab=1030, pad=0, raise_exc=None, short_by=0):
        self.vocab = vocab
        self.pad = pad
        self.raise_exc = raise_exc
        self.short_by = short_by
        self.calls = []

    def __call__(self, input_ids, **kwargs):
        self.calls.append((input_ids.clone(), kwargs))
        if self.raise_exc is not None:
            raise self.raise_exc
        k, length = input_ids.shape
        logits = torch.zeros(k, length + self.pad - self.short_by, self.vocab)
        logits[0, :, 7] = 10.0
        return SimpleNamespace(logits=logits)


def _fake_parler_engine(decoder):
    engine = SpeculativeEngine.__new__(SpeculativeEngine)
    engine.device = torch.device("cpu")
    engine.dtype = torch.float32
    engine.backend = "indic_parler_tts"
    engine.ar_model = SimpleNamespace(
        decoder=decoder,
        generation_config=SimpleNamespace(bos_token_id=1025),
        audio_encoder=SimpleNamespace(config=SimpleNamespace(codebook_size=1024)),
    )
    return engine


def test_verifier_raises_instead_of_fabricating_logits():
    draft = torch.tensor([[7, 8, 9]])
    prefixes = torch.tensor([[11]])
    ctx = {"offset": 0, "encoder_hidden_states": torch.zeros(1, 3, 16)}

    # no context -> refuse (would previously fall through to torch.randn)
    engine = _fake_parler_engine(_ScriptedDecoder())
    with pytest.raises(VerifierUnavailable):
        engine.get_target_probabilities(prefix_tokens=prefixes, draft_tokens=draft)
    with pytest.raises(VerifierUnavailable):
        engine.verify_chunk(draft, torch.softmax(torch.randn(1, 3, 1024), dim=-1))

    # decoder interface broken -> refuse, never randomize
    broken = _fake_parler_engine(_ScriptedDecoder(raise_exc=RuntimeError("bad shape")))
    with pytest.raises(VerifierUnavailable):
        broken.get_target_probabilities(prefix_tokens=prefixes, draft_tokens=draft, context=ctx)

    # decoder returns too few positions (offset not honoured) -> refuse
    short = _fake_parler_engine(_ScriptedDecoder(pad=0))
    with pytest.raises(VerifierUnavailable):
        short.get_target_probabilities(
            prefix_tokens=prefixes, draft_tokens=draft,
            context={"offset": 2, "encoder_hidden_states": torch.zeros(1, 3, 16)},
        )


def test_target_probabilities_layout_matches_teacher_forcing():
    decoder = _ScriptedDecoder(pad=2)
    engine = _fake_parler_engine(decoder)
    ctx = {"offset": 2, "encoder_hidden_states": torch.zeros(1, 3, 16)}
    probs = engine.get_target_probabilities(
        prefix_tokens=torch.tensor([[11]]),
        draft_tokens=torch.tensor([[7, 8, 9]]),
        context=ctx,
    )
    ids, kwargs = decoder.calls[0]
    assert ids.shape == (9, 5)                    # (num_codebooks, bos + prefix + draft)
    assert ids[0, 0].item() == 1025               # bos row
    assert ids[0, 1:].tolist() == [11, 7, 8, 9]   # row 0 holds the CB0 stream
    assert kwargs["use_cache"] is False
    assert kwargs["encoder_hidden_states"].shape == (1, 3, 16)
    assert probs.shape == (1, 3, 1024)            # valid CB0 codes only
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1, 3), atol=1e-4)


def test_verify_chunk_routes_real_context():
    decoder = _ScriptedDecoder(pad=2)
    engine = _fake_parler_engine(decoder)
    ctx = {"offset": 2, "encoder_hidden_states": torch.zeros(1, 3, 16)}
    draft_tokens = torch.tensor([[7, 7, 7, 7]])
    draft_probs = torch.softmax(torch.randn(1, 4, 1024), dim=-1)
    res = engine.verify_chunk(draft_tokens, draft_probs, context=ctx)
    assert 0.0 <= res.acceptance_rate <= 100.0
    assert res.verified_tokens.dim() == 2
    assert decoder.calls, "the target decoder must actually be consulted"
```

**Verify**: `movio/.venv/bin/python -m pytest tests/test_codeswitch_tts.py -q -p no:cacheprovider` → all pass (3 new tests + the existing engine suite).

### Step 9: Full suite, runbook count, index

9a. `movio/.venv/bin/python -m pytest tests/ -q -p no:cacheprovider` → all
pass; count = previous total + 3 (e.g. 103 if plans 001–003 landed in order).

9b. Update the count in `KAGGLE_RUN.md:427` to the exact pytest number.

9c. Update the `003` row in `plans/README.md`.

**Verify**: `git status --porcelain` → only the seven in-scope files modified.

## Test plan

New tests in `tests/test_codeswitch_tts.py` (pattern: existing
`test_speculative_verification` at `:76-84`; fakes avoid loading parler, which
is not installed locally):

1. `test_verifier_raises_instead_of_fabricating_logits` — no context, broken
   decoder, and short-logit decoder all raise `VerifierUnavailable`.
2. `test_target_probabilities_layout_matches_teacher_forcing` — the `(9, L)`
   grid, bos row, `use_cache=False`, `(1, K, 1024)` probs.
3. `test_verify_chunk_routes_real_context` — `verify_chunk` consults the target
   decoder when given a context; refuses without one.

Existing tests that must stay green: `test_speculative_verification`
(standalone backend, no context), `test_end_to_end_streaming_pipeline` (Mode B
fallback path, numeric acceptance).

Verification: `movio/.venv/bin/python -m pytest tests/ -q -p no:cacheprovider`
→ all pass, previous count + 3.

## Done criteria

ALL must hold:

- [ ] `grep -n "torch.randn(b, draft_len" codeswitch_tts/engine/speculative_engine.py` → no matches
- [ ] `grep -n "VerifierUnavailable" codeswitch_tts/engine/__init__.py` → at least 2 matches (import + `__all__` entry)
- [ ] `grep -n "except Exception:" codeswitch_tts/engine/speculative_engine.py` → the old bare swallow around the decoder call is gone (remaining matches, if any, are in unrelated code — inspect them)
- [ ] `movio/.venv/bin/python -m pytest tests/ -q -p no:cacheprovider` → all pass, previous count + 3
- [ ] `movio/.venv/bin/python -c "import json; nb=json.load(open('CodeSwitch_TTS_Kaggle.ipynb')); t=''.join(''.join(c['source']) for c in nb['cells']); assert 'cb0_acceptance_rate:.1f}%' not in t and t.count('acc_txt') >= 3; print('ok')"` → `ok`
- [ ] `grep -n "tests)" KAGGLE_RUN.md` → count matches your pytest summary
- [ ] `git status --porcelain` → only the seven in-scope files modified
- [ ] `plans/README.md` 003 row updated

## STOP conditions

Stop and report back (do not improvise) if:

- Any "Current state" excerpt does not match the live code (drifted repo).
- `grep -rn "verify_chunk\|get_target_probabilities" --include=*.py . | grep -v movio`
  finds a caller beyond the three listed in "Current state" — the changed
  error behavior could affect it.
- You are tempted to make `verify_chunk` return random/fake acceptance "to keep
  the demo working": that is the bug being removed. Unavailable verification is
  reported as `None`.
- The notebook edit changes the file's JSON formatting beyond the edited lines
  (`git diff --stat` shows a whole-file rewrite) — restore with
  `git checkout -- CodeSwitch_TTS_Kaggle.ipynb` and report.
- A step's verification fails twice after a reasonable fix attempt.

## Maintenance notes

- `build_prompt_context` mirrors `TeacherForcedScorer.build_context`
  (`run_full_evaluation.py:488-518`); if parler-tts changes `embed_prompts` /
  `embed_positions`, both places must be updated together. Sharing one helper is
  a deliberate follow-up (deferred so the eval path keeps working untouched).
- The engine's `build_prompt_context` requires `self.tokenizer` and
  `self.config.voice_description` — both exist after `_load_target_model` on
  the parler backend. If a future refactor loads the model lazily, revisit.
- Mode A still verifies each chunk against the *null CB0 prefix* (as it did);
  the evaluation suite's growing-prefix protocol is the measured one. The
  acceptance numbers are now real for the given protocol, not a simulation.
- Plan 004 rewrites the Mode A TTFA/timing semantics next and must be written
  against the plan-003 state (guarded `cb0_acceptance`, `_verifier_warned`).
- Reviewers should scrutinize: that no code path converts
  `VerifierUnavailable` back into a number; that `verify_ms` still measures the
  attempted verification; that `synthesize` returns `None` (not 0.0) when no
  chunk could be verified.
