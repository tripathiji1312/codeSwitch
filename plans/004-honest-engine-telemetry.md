# Plan 004: Honest Mode A telemetry — drop the fabricated sub-latencies and fix the overclaiming docstrings

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**: this plan **depends on plan 003** (it edits the
> same code that plan 003 just changed). Run:
> `git log --oneline d69b399..HEAD -- codeswitch_tts/engine/streaming_pipeline.py codeswitch_tts/engine/speculative_engine.py kaggle_demo.py`
> - If the output is empty → plan 003 has NOT landed. STOP and run
>   `plans/003-demo-verifier-honesty.md` first.
> - If it shows plan 003's commits (`Give the demo verifier real teacher-forced
>   CB0 probabilities`, `Report demo CB0 acceptance as None …`), the "Current
>   state" excerpts below (which show the post-003 shape) apply. If they still
>   do not match the live code, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: S–M
- **Risk**: MED
- **Depends on**: plans/003-demo-verifier-honesty.md
- **Category**: bug (+ docs)
- **Planned at**: commit `d69b399` (authored pre-003), 2026-10-19

## Why this matters

Mode A (the demo path with real Parler weights) never runs the residual decoder
or DAC — the whole waveform is produced by one AR generation pass — yet it
reports `residual_ms = 22.5` and `dac_ms = 18.0` constants as measured stage
latencies, and defines "TTFA" as their sum excluding the seconds-long
generation that actually precedes any audio. The notebook's headline "<300ms
TTFA" claim comes from exactly this fabrication. This plan makes Mode A report
only what it measures: TTFA becomes wall-clock-to-chunk-readiness (including
the generation pass), the unmeasured stages become `None`, and the module
docstrings / notebook headline stop describing a pipelined streaming engine
that does not exist (synthesis is synchronous). Evidence:
`plans/research/findings-2026-10-18.md` CORRECTNESS-02, DEBT-02, DOCS-01,
DOCS-07.

## Current state

(This is the state **after plan 003**; where a line still matches the original
`d69b399` text it is called out. Line numbers below are pre-003 — locate by
content.)

- `codeswitch_tts/engine/streaming_pipeline.py` — header docstring `:1-12`
  (unchanged by 003), `StreamedAudioChunk`/`PipelineBenchmarkResult` dataclasses
  `:47-76` (003 made `cb0_acceptance_rate: Optional[float]`), Mode A loop
  `:150-214` (003 added `verifier_ctx` + guarded `cb0_acceptance`), Mode B
  `:238-300` (003 only added `context=None` pass-through), `synthesize`
  `:302-351` (003 made `avg_acceptance` None-aware).
- `codeswitch_tts/engine/speculative_engine.py` — header docstring `:1-10`
  (unchanged by 003).
- `kaggle_demo.py` — per-chunk print `:152-159` (003 added `acc_txt`), summary
  table `:178-187` (003 added `acc_txt`).
- `CodeSwitch_TTS_Kaggle.ipynb` — cell 0 headline; cell 10 TTFA print.
- `tests/test_codeswitch_tts.py` — has 003's `_ScriptedDecoder` /
  `_fake_parler_engine` helpers.
- `KAGGLE_RUN.md` — line 427 test count.

### The fabrication (Mode A, post-003 excerpt)

`streaming_pipeline.py`, Mode A loop (content around `residual_ms = 22.5`):

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
                            verifier_ctx = None
                    verify_ms = (time.perf_counter() - t_verif_0) * 1000.0

                    residual_ms = 22.5        # <-- never measured on this path
                    dac_ms = 18.0             # <-- never measured on this path

                    if chunk_idx == 0:
                        ttfa_ms = draft_ms + verify_ms + residual_ms + dac_ms
                        chunk_latency_ms = ttfa_ms
                    else:
                        chunk_latency_ms = draft_ms + verify_ms + residual_ms + dac_ms
                        ttfa_ms = chunk_latency_ms
```

`gen_time_ms` is measured at the top of Mode A
(`gen_time_ms = (time.perf_counter() - t_gen_0) * 1000.0`) and never used.
The comment above it (unchanged by 003) still says
`# Parallel speculative decoding telemetry simulation for this chunk`.

### The overclaiming docstrings (unchanged by 003)

`streaming_pipeline.py:1-12`:

```
"""Chunked Streaming Orchestrator for CodeSwitch-TTS.

Pipelined architecture streaming 1-second audio chunks:
  Chunk 1:
    - Draft CB0 (~50ms)
    - AR verify CB0 (~120ms)
    - NAR predict CB1-8 (~35ms)
    - DAC decode (~20ms)
    -> Total first chunk TTFA target: ~230-260ms (<300ms)

Subsequent chunks pipeline asynchronously while Chunk N is playing.
"""
```

(No threading/async/queue exists in the file; `synthesize` drains the whole
generator before returning.)

`speculative_engine.py:1-10` opens:

```
"""Speculative Autoregressive Verifier for Codebook 0.

Executes parallel speculative rejection sampling:
  r = P(y_t | y_<t, X) / P̂(y_t | X) >= u,  u ~ U(0,1)
...
"""
```

(The loop is sequential, a host sync per token, and `break`s at the first
rejection — first-rejection truncation.)

### The demo/notebook claims

- `kaggle_demo.py:152-159` (post-003 has `acc_txt`): prints
  `Res={chk.residual_ms:.1f}ms | DAC={chk.dac_ms:.1f}ms` — will crash on the
  new `None`s.
- Notebook cell 0: `... that achieves **<300ms Time-To-First-Audio (TTFA)** on
  NVIDIA T4 GPUs.`
- Notebook cell 10: `print(f" TTFA (ms):      {result.ttfa_ms:.1f} ms  (Target: <300 ms)")`.

Notebook serialization is byte-exact with
`json.dumps(nb, indent=1, ensure_ascii=False) + "\n"` (verified live).

## Commands you will need

| Purpose        | Command                                                            | Expected on success |
|----------------|--------------------------------------------------------------------|---------------------|
| Tests (full)   | `movio/.venv/bin/python -m pytest tests/ -q -p no:cacheprovider`   | previous count + 1 new test, all pass |
| Tests (target) | `movio/.venv/bin/python -m pytest tests/test_codeswitch_tts.py -q -p no:cacheprovider` | all pass |
| Syntax gates   | `movio/.venv/bin/python -m py_compile codeswitch_tts/engine/streaming_pipeline.py codeswitch_tts/engine/speculative_engine.py kaggle_demo.py` | exit 0 |
| Notebook parse | `movio/.venv/bin/python -c "import json; json.load(open('CodeSwitch_TTS_Kaggle.ipynb')); print('ok')"` | `ok` |
| Drift/dep check | `git log --oneline d69b399..HEAD -- codeswitch_tts/engine/streaming_pipeline.py` | shows plan 003's commits |

Note: always invoke `movio/.venv/bin/python -m pytest` from the repo root (the
venv's `pytest` console script is broken).

## Scope

**In scope** (the only files you should modify):

- `codeswitch_tts/engine/streaming_pipeline.py`
- `codeswitch_tts/engine/speculative_engine.py` — header docstring only.
- `kaggle_demo.py` — the per-chunk Res/DAC rendering only.
- `CodeSwitch_TTS_Kaggle.ipynb` — cell 0 sentence + cell 10 TTFA label only.
- `tests/test_codeswitch_tts.py` — one new test.
- `KAGGLE_RUN.md` — test-count line only.

**Out of scope** (do NOT touch, even though they look related):

- `run_full_evaluation.py`'s `ttfa_compute_ms` definition — the eval path's
  component-sum TTFA is measured and correct; do not unify the two here.
- Mode B's measured stage latencies (`residual_ms`, `dac_ms` stay real floats
  on the fallback path).
- Removing the per-chunk draft/verify loop from Mode A (a known GPU-waste,
  deferred; see Maintenance notes).
- The design-targets box in `kaggle_demo.py:204-216` — it is explicitly
  labelled "(not measured)"; leave it.
- `push_to_wandb.py` — it never consumes `StreamedAudioChunk`.

## Git workflow

- Branch: `git checkout -b advisor/004-honest-telemetry`.
- Suggested commits:
  1. `Report Mode A chunk telemetry as measured-or-None and include generation in TTFA`
  2. `Rewrite the pipeline and verifier docstrings to match the synchronous code`
  3. `Drop the <300 ms TTFA claim from the notebook headline and label`
- Do NOT push or open a PR.

## Steps

### Step 1: Mode A — measured-or-None stages, generation-inclusive TTFA

In `codeswitch_tts/engine/streaming_pipeline.py`:

1a. Just before the `# Divide full natural speech into 1-second chunks`
comment, initialize the readiness accumulator (next to the `verifier_ctx`
block added by plan 003):

```python
                # Wall-clock to each chunk's readiness: the full AR generation
                # has already been paid, then each chunk adds draft + verify.
                ready_ms = gen_time_ms
```

1b. Replace the `residual_ms = 22.5` / `dac_ms = 18.0` / `if chunk_idx == 0:
...` block (shown in "Current state") with:

```python
                    # The residual decoder and DAC are NOT executed on this path
                    # (the waveform is generated in one AR pass above), so their
                    # sub-latencies are unmeasured: report None, never constants.
                    residual_ms: Optional[float] = None
                    dac_ms: Optional[float] = None

                    # Cumulative-ready semantics: time from synthesis start to
                    # this chunk's audio being ready (chunk 0 is the true TTFA).
                    ready_ms += draft_ms + verify_ms
                    ttfa_ms = ready_ms
                    chunk_latency_ms = draft_ms + verify_ms
```

1c. Replace the stale comment above the draft call
(`# Parallel speculative decoding telemetry simulation for this chunk`) with:

```python
                    # Per-chunk draft + teacher-forced verification (real
                    # probabilities; unavailable verification reports None).
```

1d. Mode B (fallback path), same semantics for chunks > 0 — replace:

```python
            chunk_latency_ms = (time.perf_counter() - chunk_start) * 1000.0
            ttfa_ms = (time.perf_counter() - pipeline_start) * 1000.0 if chunk_idx == 0 else chunk_latency_ms
```

with:

```python
            chunk_latency_ms = (time.perf_counter() - chunk_start) * 1000.0
            ttfa_ms = (time.perf_counter() - pipeline_start) * 1000.0
```

(Chunk 0 is unchanged in value; chunks > 0 now report cumulative time-to-ready
instead of duplicating their own latency in the TTFA field.)

1e. Dataclass fields (`StreamedAudioChunk`, `:53-59`): update the comments and
types:

```python
    ttfa_ms: float                     # Wall-clock from synthesis start to this chunk's audio being ready (chunk 0 = TTFA)
    chunk_latency_ms: float            # Time taken to generate this specific chunk
    cb0_acceptance_rate: Optional[float]   # %; None when verification was unavailable
    draft_ms: float
    verify_ms: float
    residual_ms: Optional[float]       # Mode A: None (residual decoder not run)
    dac_ms: Optional[float]            # Mode A: None (DAC not run on this path)
```

**Verify**: `movio/.venv/bin/python -m py_compile codeswitch_tts/engine/streaming_pipeline.py` → exit 0;
`grep -n "22.5\|18.0" codeswitch_tts/engine/streaming_pipeline.py` → no matches.

### Step 2: Rewrite the two module docstrings

2a. `streaming_pipeline.py:1-12` → replace the whole module docstring with:

```python
"""Chunked synthesis orchestrator for CodeSwitch-TTS.

Two synchronous paths (no threading/async):

  Mode A (parler backend): the whole utterance is generated in ONE AR pass
    (``generate_authentic_speech``) and then sliced into ~1 s chunks; per chunk
    the draft runs and - when the verifier is available - is scored with real
    teacher-forced CB0 probabilities.  Residual/DAC sub-latencies are not
    measured on this path and are reported as ``None``.  ``ttfa_ms`` is
    wall-clock from synthesis start to the chunk becoming ready, so chunk 0
    includes the full generation pass.

  Mode B (offline fallback, ``allow_untrained_fallback=True`` only): per-chunk
    draft -> verify -> residual -> DAC with measured stage latencies.  Audio
    here is random-weight noise, not speech.
"""
```

2b. `speculative_engine.py:1-10` → replace the whole module docstring with:

```python
"""Sequential speculative rejection sampling for Codebook 0.

Verification walks the draft frames one at a time and truncates the chunk at
the FIRST rejection:
  r = P(y_t | y_<t, X) / Phat(y_t | X) >= u,  u ~ U(0,1)
Rejected frames resample from the residual distribution P'(y):
  P'(y) = max(0, P(y) - Phat(y)) / sum_v(max(0, P(v) - Phat(v)))

Target probabilities come from a real teacher-forced forward of the Parler
decoder; failures raise ``VerifierUnavailable`` instead of guessing.  The
evaluation suite's vectorised per-frame sampler and acceptance definition live
in ``run_full_evaluation.py`` (``rejection_sample``); the two are intentionally
different protocols for now.
"""
```

**Verify**: `grep -n "pipeline asynchronously" codeswitch_tts/engine/streaming_pipeline.py` → no matches; `grep -n "Executes parallel speculative" codeswitch_tts/engine/speculative_engine.py` → no matches.

### Step 3: `kaggle_demo.py` — "n/a (single-pass AR)" for unmeasured stages

Replace the per-chunk print loop (post-003 shape):

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

with:

```python
        for chk in benchmark_res.chunks:
            acc_txt = "n/a" if chk.cb0_acceptance_rate is None else f"{chk.cb0_acceptance_rate:.1f}"
            res_txt = "n/a (single-pass AR)" if chk.residual_ms is None else f"{chk.residual_ms:.1f}ms"
            dac_txt = "n/a (single-pass AR)" if chk.dac_ms is None else f"{chk.dac_ms:.1f}ms"
            print(
                f"       * Chunk {chk.chunk_index + 1}/{benchmark_res.chunks_count}: "
                f"Latency={chk.chunk_latency_ms:.1f}ms (TTFA={chk.ttfa_ms:.1f}ms) | "
                f"Draft={chk.draft_ms:.1f}ms | Verify={chk.verify_ms:.1f}ms | "
                f"Res={res_txt} | DAC={dac_txt} | "
                f"CB0 Acceptance={acc_txt}%"
            )
```

(The summary table at `:178-187` only prints TTFA/Total/RTF/acceptance — no
change needed there.)

**Verify**: `movio/.venv/bin/python -m py_compile kaggle_demo.py` → exit 0;
`grep -n "residual_ms:.1f\|dac_ms:.1f" kaggle_demo.py` → no matches.

### Step 4: Notebook — headline and TTFA label

Run this script from the repo root (same byte-preserving pattern as plan 003):

```python
import json

path = "CodeSwitch_TTS_Kaggle.ipynb"
nb = json.load(open(path, encoding="utf-8"))

replacements = [
    (
        "This notebook runs the unified **CodeSwitch-TTS** speculative decoding pipeline designed for Tanglish (Tamil-English code-mixed speech) that achieves **<300ms Time-To-First-Audio (TTFA)** on NVIDIA T4 GPUs.",
        ["This notebook runs the unified **CodeSwitch-TTS** speculative decoding pipeline designed for Tanglish (Tamil-English code-mixed speech). It is a **demo**: the TTFA printed below is wall-clock time-to-first-audio and includes the full autoregressive generation pass. The paper's latency numbers (component-sum TTFA-c, cold/warm separated) come from ``run_full_evaluation.py`` (Table I)."],
    ),
    (
        'print(f" TTFA (ms):      {result.ttfa_ms:.1f} ms  (Target: <300 ms)")',
        ['print(f" TTFA (ms):      {result.ttfa_ms:.1f} ms  (wall-clock to first audio, includes AR generation)")'],
    ),
]

for c in nb["cells"]:
    src = c["source"]
    for old, new_lines in replacements:
        for i, line in enumerate(src):
            text = line if line.endswith("\n") else line + "\n"
            if text.strip("\n") == old or text == old:
                src[i] = new_lines[0] + "\n"
                break

open(path, "w", encoding="utf-8").write(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print("notebook updated")
```

(Markdown cell 0 stores the sentence as one long line preceded by `\n\n` — the
comparison above uses `strip("\n")` for that line.)

**Verify**:
`movio/.venv/bin/python -c "import json; nb=json.load(open('CodeSwitch_TTS_Kaggle.ipynb')); t=''.join(''.join(c['source']) for c in nb['cells']); assert '<300ms' not in t and 'Target: <300 ms' not in t; assert 'includes the full autoregressive generation pass' in t; print('ok')"` → `ok`.

### Step 5: Stub-engine test for Mode A telemetry

Append to `tests/test_codeswitch_tts.py` (it has `_ScriptedDecoder` etc. from
plan 003; `SimpleNamespace` is imported):

```python
def test_mode_a_chunk_telemetry_is_honest():
    """Mode A: TTFA includes the generation pass; unmeasured stages are None."""
    import time as _time

    pipeline = ChunkedStreamingPipeline.__new__(ChunkedStreamingPipeline)
    pipeline.config = CodeSwitchTTSConfig()
    pipeline.device = torch.device("cpu")
    pipeline.normalizer = TanglishNormalizer()
    pipeline.speech_available = True
    pipeline._verifier_warned = False

    class _StubEngine:
        backend = "indic_parler_tts"

        def generate_authentic_speech(self, text, voice_description=None):
            _time.sleep(0.05)  # stand-in for the seconds-long AR generation
            return np.full(2 * 44100, 0.1, dtype=np.float32)

        def build_prompt_context(self, text):
            return {"offset": 0, "encoder_hidden_states": torch.zeros(1, 2, 8)}

        def verify_chunk(self, draft_tokens, draft_probs, context=None):
            assert context is not None
            return SimpleNamespace(acceptance_rate=87.5)

    pipeline.speculative_engine = _StubEngine()

    class _StubDraft:
        def generate_chunk(self, text, num_frames, device):
            from codeswitch_tts.models.draft_transformer import DraftOutput
            probs = torch.full((1, num_frames, 1024), 1.0 / 1024.0)
            return DraftOutput(logits=probs.log(), probabilities=probs,
                               tokens=torch.zeros(1, num_frames, dtype=torch.long))

    pipeline.draft_model = _StubDraft()

    chunks = list(pipeline.stream_synthesize("Warm-up only."))

    assert len(chunks) == 2
    # chunk 0 TTFA must include the generation pass (>= the 50 ms stub sleep)
    assert chunks[0].ttfa_ms >= 45.0
    assert chunks[0].cb0_acceptance_rate == pytest.approx(87.5)
    # unmeasured Mode A stages are None, not constants
    assert chunks[0].residual_ms is None and chunks[0].dac_ms is None
    # cumulative-ready semantics: chunk 1 is ready no earlier than chunk 0
    assert chunks[1].ttfa_ms >= chunks[0].ttfa_ms
```

**Verify**: `movio/.venv/bin/python -m pytest tests/test_codeswitch_tts.py -q -p no:cacheprovider` → all pass.

### Step 6: Full suite, runbook count, index

6a. `movio/.venv/bin/python -m pytest tests/ -q -p no:cacheprovider` → all
pass; count = previous total + 1.

6b. Update `KAGGLE_RUN.md:427` count to the exact pytest number.

6c. Update the `004` row in `plans/README.md`.

**Verify**: `git status --porcelain` → only the six in-scope files modified.

## Test plan

- **New** `test_mode_a_chunk_telemetry_is_honest` in
  `tests/test_codeswitch_tts.py` — stubs only `generate_authentic_speech`,
  `build_prompt_context`, `verify_chunk` and the draft model; asserts the
  generation time is inside chunk-0 TTFA, unmeasured stages are `None`, and
  TTFA is cumulative. Sleeps 50 ms once (acceptable test cost).
- Existing `test_end_to_end_streaming_pipeline` (Mode B) must stay green — it
  exercises measured `residual_ms`/`dac_ms` and the chunk-0 TTFA path.
- Verification: `movio/.venv/bin/python -m pytest tests/ -q -p no:cacheprovider`
  → all pass, previous count + 1.

## Done criteria

ALL must hold:

- [ ] `grep -n "22.5\|18.0" codeswitch_tts/engine/streaming_pipeline.py` → no matches
- [ ] `grep -n "gen_time_ms" codeswitch_tts/engine/streaming_pipeline.py` → `ready_ms = gen_time_ms` present (the measured value is used)
- [ ] `grep -n "residual_ms: Optional\[float\]" codeswitch_tts/engine/streaming_pipeline.py` → at least 1 match
- [ ] `grep -n "pipeline asynchronously" codeswitch_tts/engine/streaming_pipeline.py` → no matches
- [ ] `movio/.venv/bin/python -c "import json; nb=json.load(open('CodeSwitch_TTS_Kaggle.ipynb')); t=''.join(''.join(c['source']) for c in nb['cells']); assert '<300ms' not in t; print('ok')"` → `ok`
- [ ] `movio/.venv/bin/python -m pytest tests/ -q -p no:cacheprovider` → all pass, previous count + 1
- [ ] `git status --porcelain` → only the six in-scope files modified
- [ ] `plans/README.md` 004 row updated

## STOP conditions

Stop and report back (do not improvise) if:

- The dependency check shows plan 003 has not landed — run it first.
- The post-003 excerpts (e.g. `cb0_acceptance = float(verif_res.acceptance_rate)`, `verifier_ctx`) do not match the live code — the repo has drifted beyond both plans.
- `synthesize()`'s consumers turn out to depend on `StreamedAudioChunk.ttfa_ms` for chunk > 0 being *per-chunk* latency (grep first:
  `grep -rn "chunks\[" --include=*.py . | grep -v movio` and
  `grep -rn "ttfa_ms" --include=*.py . | grep -v movio`); report if so instead of changing semantics silently.
- A step's verification fails twice after a reasonable fix attempt.
- The notebook edit rewrites the whole file (check `git diff --stat`) — restore and report.

## Maintenance notes

- Semantics now: for **both** modes, `StreamedAudioChunk.ttfa_ms` = wall-clock
  from synthesis start to that chunk being ready; `chunk_latency_ms` = the
  chunk's own work. `PipelineBenchmarkResult.ttfa_ms` = chunk 0's value.
  Anything that compared `ttfa_ms` across chunks must be revisited.
- `residual_ms`/`dac_ms` are `None` **only** on Mode A. Renderers must keep the
  None guards (kaggle_demo does after this plan).
- Deferred on purpose (recorded in `plans/README.md`): unifying the demo chunk
  loop with `EvalSynthesizer._chunked_synthesis` (DEBT-02's bigger half) and
  the prefix-stop vs per-frame rejection-sampling divergence (DEBT-03). Those
  change acceptance-number semantics and the exported `SpeculativeVerificationResult`.
- Reviewers should scrutinize: TTFA monotonicity across chunks; that no path
  reintroduces constant sub-latencies; and that the notebook's remaining
  design-target box (cell 12) still carries its "(not measured)" disclaimer.
