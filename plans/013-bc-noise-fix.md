# Plan 013 — Fix the B/C noise path: real DAC, argmax auditing, and a retrained student

**Status:** IN PROGRESS — code/tests for Phases 1–4 landed; Kaggle retrain/eval pending · **Branch:** `fix/013-bc-noise` (one plan, one branch; never push
without the maintainer's authorization) · **Priority:** P0 (paper-blocking) ·
**Depends on:** nothing; **Conflicts with:** plan 006 (`006-supply-chain-pinning.md`)
— both edit `codeswitch_tts/utils/audio.py::DACWrapper`. Execute 013 first, then
rebase plan 006 by content, or drop plan 006's DAC-loader section.

**Evidence base:** `research/findings-2026-10-20.md` (six subagent reports,
verbatim). Read it before starting. Every claim below is traceable there, except
numbers explicitly marked **(run context)**: those come from the 2026-10-20
2 000-sentence Kaggle run described in the conversation, have no committed
`raw_results.jsonl`/`run_meta.json` in this repo, and are reference points only —
re-measure them before using them in a paper.

---

## 1. Objective & Success Criteria

**Objective.** Stop the pipeline from ever silently feeding random-weight DAC
audio into diagnostics, make the evaluator able to run Configs B/C with greedy
(argmax) decoding, retrain the draft and residual with the two measured training
defects fixed, and re-run the diagnostic + 60-utterance evaluation on Kaggle.
Latency claims must survive; B/C quality claims are re-earned, not assumed.

**What done looks like (all must be true):**

1. `DACWrapper` has no silent random path. With pretrained weights unavailable
   it either raises or uses the explicit synthetic fallback only when the caller
   opted in (`allow_fallback=True` / `--allow-untrained`). `backend` can never be
   `hf_local`.
2. `diagnose_audio_bug.py` reports a real DAC backend (`hf_pretrained` or
   `dac_official`). With that backend, every wav it writes must have
   `rms > 0`, `peak <= 0.95`, and at least one teacher wav must be speech-like by
   the diagnostic's own heuristic: `flatness < 0.3` and `zcr < 0.3`. These
   thresholds are intentionally loose heuristics (the report at
   `research/findings-2026-10-20.md` calls them placeholders); the mandatory
   check is a human listen plus the code-level gates in item 4. Do **not** gate
   on a particular shard number: `diagnose_audio_bug.py:49-54,150` samples
   `--shards 4` by default, so the sampled ids are whatever the run prints.
3. `run_full_evaluation.py --sample-mode argmax --configs B,C` runs end-to-end
   and every non-error B/C row carries `sample_mode == "argmax"` and a
   `dac_backend` from the allowed set in §7; `--sample-mode sample` still
   reproduces the old behavior. Allowed values are defined in §7: Config A →
   `parler_streamer`; B/C with the Parler backend → `parler_audio_encoder`;
   engine-less/fallback → `hf_pretrained`, `dac_official`, or (only with
   `--allow-untrained`) `fallback_neural`; anything else is a failure.
4. On the restored 2 000-sentence teacher cache, the retrained student clears the
   diagnostic's own "works on both" gate: **`rt_a >= 5 %` and `rd_a >= 5 %`**
   (the verdict at `diagnose_audio_bug.py:238-249`). `rd_a` is decisive because
   it is the residual run on the draft's CB0 — the input it sees at inference.
   Also record `cb0_a` and compare it against the restored v1
   `codeswitch_ckpt/meta.json` (`draft.val_top1` was `0.074269` in the run
   context); do not hardcode a new percentage threshold that no artifact
   supports. If `rd_a < 5 %` but `rt_a >= 5 %`, stop and fix the draft first.
5. The argmax B/C evaluation must improve on a **matched** `--sample-mode sample`
   run on the same `codeswitch_ckpt_v2`, and must approach Config A measured in
   the same run. All CER gates use the **per-utterance median** returned by
   `_cer_p50` (Latin-transliterated `cer_latin.p50` when the run recorded it;
   otherwise `cer.p50`) and are compared in **percentage points**:
   - `100 * _cer_p50(B_argmax) <= 100 * _cer_p50(A_argmax) + 10`, and
   - `B_argmax_UTMOS >= 0.75 * A_UTMOS` using `aggregates[cfg]["utmos"]["mean"]`,
     and
   - `100 * _cer_p50(C_argmax) <= 100 * _cer_p50(B_argmax) + 10`, and
   - `aggregates[cfg]["asr_fail"] == 0` for A, B, and C (`asr_fail` counts
     utterances with `cer > ASR_FAIL_CER = 2.0`; this is the loop check — the
     `asr_max_new_tokens` cap is already applied during decoding).
   If `aggregates[cfg]["utmos"]["mean"]` is `None` for A or B, the UTMOS gate
   **fails closed**; do not infer a score from the audio or skip the gate.
   If the gate fails, B/C quality claims stay BLOCKED and only a negative result
   is reported. (The old absolute baselines — B/C CER ~93–94 %, UTMOS 1.25,
   MCD ~103–105, TTFA 36/80 ms, RTF 0.05/0.08 — are from the uncommitted
   2026-10-20 2 000-sentence run described in the task context, not from a repo
   artifact. They are reference points only; the matched run is the authority.)
6. Latency does not regress against the **matched** sample-mode run on the same
   checkpoints. Use the **warm** compute-only TTFA median
   (`aggregates[cfg]["ttfa_compute_warm"]["p50"]`, which excludes the
   process's first synthesis) and the RTF median
   (`aggregates[cfg]["rtf"]["p50"]`). Both must be within ±25 % of the matched
   sample-mode value for B and for C; a `None` value fails the gate. Do not gate
   on `ttfa_e2e`, which includes client-side frame playback. The committed
   reference in `PAPER_CLAIMS_AND_EVIDENCE.md:15-16` reports NAR RTF 0.04 and
   hybrid RTF 0.07 with TTFA in the 32–139 ms range; use the fresh measured
   pair, not a hardcoded millisecond target.
7. `PAPER_CLAIMS_AND_EVIDENCE.md` is updated only with the measured numbers from
   step 5. `KAGGLE_RUN.md` is not edited; amendments go to `plans/README.md`.

**How to verify:** the checklist in §7, run top-to-bottom. No step is "done" from
reading code alone; every gate has a command or an observed number.

---

## 2. Locked Scope & Constraints

**In scope**

- `codeswitch_tts/utils/audio.py` — real-weights-only DAC loading, explicit
  `allow_fallback` opt-in, decode branch cleanup.
- `codeswitch_tts/engine/streaming_pipeline.py` — pass the opt-in through.
- `run_full_evaluation.py` — `--sample-mode {sample,argmax}`, fallback DAC uses
  the Parler audio encoder when present, row-level `sample_mode`/`dac_backend`.
- `codeswitch_tts/models/draft_transformer.py` — expose `sample` on
  `generate_chunk`.
- `diagnose_audio_bug.py` — refuse random-DAC wavs, report backend.
- `train_draft_distill.py` — frame-length bucketing for the draft, shard-disjoint
  residual validation, 100 % draft-CB0 residual conditioning, shard validation,
  provenance fields.
- `tests/test_codeswitch_tts.py`, `tests/test_evaluation_suite.py`,
  `tests/test_corpus_builder.py` — regression tests.
- `bootstrap_diagnose.py` — commit the existing untracked recovery helper (it is
  the only one-command restore+dependency path on Kaggle, and the notebook
  hard-resets to `origin/main`).
- `plans/README.md` — status/amendment notes only.

**Out of scope (explicitly rejected for this plan)**

- The "Medusa heads on Parler" pivot. Subagent 5 verified that Parler's delay
  pattern + summed per-column codebook embeddings + causal masking make the
  proposed CB0→CB1-8 single-pass generation impossible; Medusa on this model
  needs a custom decoding loop that reproduces `ParlerTTSLogitsProcessor`, and
  the closest published multi-codebook Medusa result (VADUSA, EnCodec 8-codebook)
  reports acceptance collapse. It is a larger change than fixing the student and
  is not needed to unblock B/C.
- `descript-audio-codec` supply-chain pinning (plan 006 owns it), notebook
  reshaping beyond the CLI commands below, and any change to
  `KAGGLE_RUN.md`.
- New teacher capture. The 2 000 restored shards are the training data; the
  expensive ~5 h teacher pass is not repeated in this plan.
- Config A quality. A is the reference and is unchanged.

**Hard constraints**

- Interpreter: `movio/.venv/bin/python` for everything (the venv has no `pip`;
  `movio/.venv/bin/pytest` has a broken shebang — never use it).
- Tests: `movio/.venv/bin/python -m pytest tests/ -q -p no:cacheprovider` from
  the repo root. Baseline is **120 passed** (verified 2026-10-20); `KAGGLE_RUN.md`
  still says 103 and must not be edited.
- Do not push. Commit on the branch; the maintainer merges. Kaggle notebook cell 6
  hard-resets to `origin/main`, so the fix must be merged + pushed before a
  Kaggle run can see it.
- One branch per plan; run plans sequentially. If plan 006 already landed, stop
  and re-anchor by content.
- No API/name/version invention. All identifiers below are quoted from the
  current tree or from the verified source reports.

---

## 3. Research Findings (evidence-backed)

These are the facts that change the plan. Full reports:
`research/findings-2026-10-20.md`.

1. **The random DAC is real, but it is not what made the 60-utterance eval
   noisy.** `DACWrapper._load_dac` (`codeswitch_tts/utils/audio.py:116-157`)
   tries `DacModel.from_pretrained("ylacombe/dac_44khZ_8kbps")`; that call fails
   on both transformers 4.46.1 (Kaggle pin) and 5.16.1 with
   `AttributeError: property 'frame_rate' of 'DacConfig' object has no setter`
   (the checkpoint config has `"frame_rate": 86`; the wrapper-layout weights are
   301 `model.*` tensors and cannot load into `transformers.DacModel` anyway).
   The `except` branch builds a **random** `DacModel` and sets
   `backend="hf_local"`, then returns before the official
   `descript-audio-codec` attempt. The diagnostic wavs (`diagnose_audio_bug.py:170`)
   and engine-less eval runs are invalid because of this.
   **The 60-utterance A/B/C eval did not use it**: `EvalSynthesizer.decode_codes`
   (`run_full_evaluation.py:743-753`) uses
   `engine.ar_model.audio_encoder` first when `speech_available`; the run needed
   Parler for Config A and for C's true-acceptance scorer. So the B/C noise is
   the code-level failure (draft CB0 7.45 %, residual 0.21 % on draft CB0),
   exactly as `PAPER_CLAIMS_AND_EVIDENCE.md:6,23,42` reports.
2. **`parler_model.audio_encoder` is the correct in-memory DAC.** For
   `ai4bharat/indic-parler-tts` it is a transformers `DacModel` with 223
   `audio_encoder.*` tensors matching the checkpoint; `decode(audio_codes=(B,9,T))`
   returns 44.1 kHz audio and does **not** accept `audio_scales`. The correct
   standalone repo for environments without Parler (the diagnostic) is
   `ylacombe/dac_44khz` (223-tensor transformers layout), not
   `ylacombe/dac_44khZ_8kbps` (301-tensor parler-wrapper layout).
3. **Argmax needs a boolean, not `temperature=0.0`.** `DraftTransformer.forward`
   already has `sample`, but `generate_chunk` does not expose it
   (`draft_transformer.py:217-238`), so B/C draft always samples. The evaluator
   explicitly passes `sample=True` to the residual (`run_full_evaluation.py:1003-1007`),
   overriding the model's argmax default. `temperature=0.0` still runs
   `torch.multinomial` (`max(temperature, 1e-5)`), so it is not a fix.
4. **The draft is trained with a text→frame-stretch bug.** `DraftTrainer._collate`
   pads every batch to `max_t = max(s.frames)` and forwards `num_frames=max_t`
   (`train_draft_distill.py:693-697, 721-731, 749-771`), so short sentences are
   trained against a longer frame axis than their own target. A controlled run
   with the repo's own trainer code: stock batch-max 18.35 % deployment top-1 vs
   per-sample frames 91.88 % (fabricated learnable target; mechanism check, not a
   magnitude estimate). This is the largest identified training defect.
5. **The residual is trained on clean teacher CB0 and deployed on draft CB0.**
   `ResidualTrainer` defaults to `--residual-sched 0.0`
   (`train_draft_distill.py:877-881, 949-953`); the big-run notebook uses 0.5,
   which mixes teacher and (7 %-accurate) draft CB0 inputs. Validation is
   window-level (`:1029-1030`), so adjacent frames of the same shard leak between
   train and val.
6. **The budget is not the binding constraint.** The reported run stopped at the
   epoch caps (draft 60, residual 60), not the wall-clock caps: ~5.5 min draft /
   ~8.9 min residual. 300 epochs are affordable on a T4; the teacher pass, not
   the student training, is the expensive step.
7. **The external AI's other claims:** the "5-minute TTS trap" is directionally
   right but omits findings 4–5; "99.79 % of acoustic tokens are random" is not a
   valid inference from top-1 accuracy; Route A step 3 (CB0→Parler CB1-8 in one
   parallel pass) is false (finding 8); the Medusa pivot is a much larger change
   with weak multi-codebook precedent.
8. **Parler's delay pattern blocks the proposed shortcut.** `parler-tts`
   `main@d108732c` `modeling_parler_tts.py` sums the 9 codebook embeddings per
   column (`#L1433`) and emits 9 codebook logits from one shared hidden state
   (`#L1917-1920`); codebook *k* is delayed by *k* columns in
   `build_delay_pattern_mask` (`#L214-277`). One parallel pass can *score* all 9
   rows, but generating CB1-8 of a frame still costs ~1 step/frame. The repo's
   `SpeculativeEngine` Parler verification branch is dead code that returns
   `torch.randn` (`speculative_engine.py:246-260`) — out of scope here, but do not
   use its output as evidence.
9. **The eval's true scorer is a null-context teacher-forced pass.** It fills
   rows 1-8 with BOS (`run_full_evaluation.py:531-533`) and reads `logits[0]`
   (`:552`). That protocol is unchanged in this plan; the argmax change only
   affects generated tokens.
10. **Test/seam facts.** Baseline 120 tests; `DACWrapper` has zero direct tests;
    `_chunked_synthesis` is never executed by a test (the C-gate test fakes it);
    `AudioProcessor(dac_model=object())` reliably reports `passed_model`;
    `test_synth_C_falls_back_below_threshold` monkeypatches `_chunked_synthesis`
    with a fixed 4-argument signature, so the sample mode must be read from
    `self.args`, not added as a parameter.

---

## 4. Chosen Approach & Why

**Chosen: fix the diagnostic DAC guard, add a real argmax audit path, then
retrain the existing student with the two measured training defects fixed.**

Why this beats the alternatives:

- **vs. "bind DACWrapper to `parler_model.audio_encoder`" only:** correct for the
  streaming pipeline and the eval fallback, but the standalone diagnostic has no
  Parler model, and the engine-less eval path has no Parler model either. The fix
  must also cover a real standalone DAC (`ylacombe/dac_44khz`) and must refuse
  random weights when neither source is available.
- **vs. argmax-only:** argmax removes the sampler collapse (draft CB0 argmax
  ~10.3 % vs sample ~2.5 %, and residual-on-draft-CB0 argmax ~1.65 % vs sample
  ~0.21 %; **run context**, not a committed artifact) but a ~10 %-accurate CB0
  still produces noise. It is a diagnostic and a partial mitigation, not a fix.
- **vs. retraining only:** without argmax the eval keeps sampling the residual at
  a distribution the model was never trained to sample from; and without the DAC
  guard the diagnostic cannot hear whether the fix worked.
- **vs. Medusa/Parler pivot:** larger code surface, a custom decoding loop, new
  training data, and published evidence of acoustic-token acceptance collapse
  (VADUSA: 1.85× EnCodec vs 2.94× semantic). It is the wrong first move when the
  existing student has two reproducible bugs and the whole retrain is ~1 T4-hour.

**The plan is one path with two hard gates:** the DAC/argmax fixes land first
(they are independent of training and immediately make the diagnostic honest);
the retrain lands second; the B/C quality claim is made only if the retrained
student clears §1.4 and §1.5. If it does not, the next lever is a capacity bump
for the residual or multi-sample teacher targets — explicitly deferred, not
silently attempted.

---

## 5. Implementation Plan

### Phase 0 — Drift check, branch creation, and baseline (no source changes)

```bash
cd /home/tripathiji/projects/all_work
git fetch origin
git switch -c fix/013-bc-noise
git status --short --branch
git log --oneline -5
movio/.venv/bin/python -m pytest tests/ -q -p no:cacheprovider
```

Expected: the branch is `fix/013-bc-noise`; HEAD is `6bcc188` or a descendant;
the untracked set is `bootstrap_diagnose.py`, `plans/013-bc-noise-fix.md`, and
`research/` (the plan and its evidence are new files, not drift). Pytest prints
**120 passed** (3 warnings). If the count is different, plans 003–012 have
landed — re-anchor every excerpt by content, not by line number, and record the
real pre-count in `plans/README.md`.

Commit the recovery helper before any Kaggle step; it is currently untracked and
the Kaggle notebook resets to `origin/main`, so an uncommitted helper cannot be
used:

```bash
git add bootstrap_diagnose.py plans/013-bc-noise-fix.md research/
git commit -m "docs: plan and evidence for the B/C noise fix"
```

**STOP** if `codeswitch_tts/utils/audio.py:116-157` no longer matches the shape
quoted in §3.1, or if `run_full_evaluation.py` no longer calls
`_decode_codes(self.engine.ar_model.audio_encoder, codes)` at `:746`.

### Phase 1 — Make the DAC loader real-or-refuse

**Files:** `codeswitch_tts/utils/audio.py`, `codeswitch_tts/engine/streaming_pipeline.py`,
`run_full_evaluation.py`, `diagnose_audio_bug.py`.

**1a. `DACWrapper.__init__` (`audio.py:98-114`)** — add an explicit opt-in and
store the load error:

```python
def __init__(
    self,
    config: Optional[AudioConfig] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
    model: Optional[Any] = None,
    allow_fallback: bool = False,
):
    ...
    self.allow_fallback = bool(allow_fallback)
    self.load_error: Optional[str] = None
    if model is not None:
        self.model = model
        self.backend = "passed_model"
    else:
        self.model = None
        self.backend = "none"
        self._load_dac()
```

**1b. Replace the body of `_load_dac` (`audio.py:116-157`)** with the verified
two-source flow. Do **not** build `DacModel(DacConfig(...))` anywhere.

```python
DAC_PRETRAINED_ID = "ylacombe/dac_44khz"

def _load_dac(self):
    """Load a real DAC decoder, or refuse/opt into the synthetic fallback.

    Never silently installs random weights: the previous `hf_local` branch
    produced white noise for every diagnostic wav.
    """
    errors = []
    # Attempt 1: transformers-native DAC (223-tensor layout; the same weights
    # Parler's audio_encoder carries, but loadable without Parler).
    try:
        from transformers import DacModel
        self.model = DacModel.from_pretrained(DAC_PRETRAINED_ID).to(self.device)
        self.model.eval()
        self.backend = "hf_pretrained"
        return
    except Exception as exc:
        errors.append(f"transformers DacModel({DAC_PRETRAINED_ID}): "
                      f"{type(exc).__name__}: {exc}")

    # Attempt 2: official descript-audio-codec package (DAC package layout).
    try:
        import dac
        from dac.model import DAC
        model_path = dac.utils.download(model_type="44khz")
        self.model = DAC.load(model_path).to(self.device)
        self.model.eval()
        self.backend = "dac_official"
        return
    except Exception as exc:
        errors.append(f"descript-audio-codec: {type(exc).__name__}: {exc}")

    self.model = None
    self.load_error = "; ".join(errors)
    if not self.allow_fallback:
        self.backend = "unavailable"
        raise RuntimeError(
            "No pretrained DAC weights available; refusing to install random "
            f"weights. Tried: {self.load_error}. Pass allow_fallback=True only "
            "for telemetry-only runs."
        )
    self.backend = "fallback_neural"
```

**1c. `DACWrapper.decode` (`audio.py:159-226`)** — update the branch tuple and
add the hard failure:

- First branch: `if self.backend in ("passed_model", "hf_pretrained") and self.model is not None:`
  keep the existing two `self.model.decode(...)` attempts.
- Second branch: `if self.backend == "dac_official" and self.model is not None:`
  keep the `quantizer.from_codes` + `decode` path.
- Synthetic path: guard it with `if self.backend == "fallback_neural":`.
- At the end, `raise RuntimeError("DACWrapper has no usable DAC backend: " + str(self.load_error))`.
  A random `hf_local` object must be unreachable.

**1d. `AudioProcessor.__init__` (`audio.py:229-234`)** — add
`allow_fallback: bool = False` and pass it through:

```python
def __init__(self, config: Optional[AudioConfig] = None,
             dac_model: Optional[Any] = None, allow_fallback: bool = False):
    self.config = config or DEFAULT_CONFIG.audio
    self.dac = DACWrapper(config=self.config, model=dac_model,
                          allow_fallback=allow_fallback)
```

**1e. Streaming pipeline (`streaming_pipeline.py:105-107`)** — the existing
opt-in is `config.system.allow_untrained_fallback`:

```python
self.audio_processor = AudioProcessor(
    self.config.audio,
    dac_model=dac_model,
    allow_fallback=self.config.system.allow_untrained_fallback,
)
```

This keeps `test_end_to_end_streaming_pipeline` (which sets
`allow_untrained_fallback = True`) green: it gets the synthetic fallback, not a
random `DacModel`.

**1f. Evaluator fallback (`run_full_evaluation.py:743-753`)** — when the
teacher decode fails, reuse the in-memory Parler DAC if there is one, and pass
the CLI opt-in:

```python
if self._dac_fallback is None:
    from codeswitch_tts.utils.audio import AudioProcessor
    dac_model = getattr(getattr(self.engine, "ar_model", None), "audio_encoder", None)
    self._dac_fallback = AudioProcessor(
        self.config.audio,
        dac_model=dac_model,
        allow_fallback=bool(getattr(self.args, "allow_untrained", False)),
    )
```

**1g. Row provenance (`run_full_evaluation.py`)** — `_base_row` is built at
`:904` **before** `decode_codes` (`:1014`), so it cannot know the DAC backend.
Record the backend at decode time and write it in the post-synthesis row updates.

Add to `EvalSynthesizer.__init__` (after `self._dac_fallback = None` at `:716`):

```python
self._last_dac_backend: Optional[str] = None
```

In `decode_codes` (`:743-753`), set it on both exits:

```python
if self._speech_backend:
    audio = _decode_codes(self.engine.ar_model.audio_encoder, codes)
    if audio is not None:
        self._last_dac_backend = "parler_audio_encoder"
        return audio
...
self._last_dac_backend = getattr(
    getattr(self._dac_fallback, "dac", None), "backend", "unknown"
)
return np.asarray(...)
```

In `_base_row` (`:755-770`), add only the mode (known before synthesis) and a
placeholder:

```python
"sample_mode": (
    str(getattr(self.args, "sample_mode", "sample")) if cfg in ("B", "C") else "n/a"
),
"dac_backend": None,
```

In the Config A row update (`:878-896`), add:

```python
"dac_backend": "parler_streamer",
```

In `synth_C`'s low-confidence fallback update (`:1157-1175`), add both fields
explicitly, because `synth_A` produced an A row (`sample_mode="n/a"`) and the
C row must still carry the requested B/C mode:

```python
"sample_mode": str(getattr(self.args, "sample_mode", "sample")),
"dac_backend": row.get("dac_backend") or "parler_streamer",
```

In the B/C row update (`:1058-1088`), add:

```python
"dac_backend": self._last_dac_backend or "unknown",
```

`passed_model` is the backend reported when `AudioProcessor` receives an
injected model (`audio.py:108-110`); in the eval fallback that injected model is
the Parler audio encoder, so it is an accepted label alongside
`parler_audio_encoder`. The allowed set is defined in §7 and includes
`parler_streamer`, `parler_audio_encoder`, `passed_model`, `hf_pretrained`, and
`dac_official`; `fallback_neural` is accepted only with `--allow-untrained`;
`unavailable`/`unknown` fail the row contract.

Reset `self._last_dac_backend = None` at the top of `_chunked_synthesis` so a
worker exception cannot leak the previous utterance's backend. Error rows keep
`dac_backend=None`; the evaluator's completion gate excludes `error` rows.

**1h. Diagnostic (`diagnose_audio_bug.py:168-173`)** — never silently decode
with random weights. Replace the `processor` setup with a guarded block:

```python
processor = None
if not args.no_audio:
    try:
        processor = AudioProcessor(CodeSwitchTTSConfig().audio)
        print(f"[audio] DAC backend: {processor.dac.backend}")
        if processor.dac.backend not in ("hf_pretrained", "dac_official"):
            print("[audio] ⚠ DAC backend is not a pretrained decoder; "
                  "skipping wavs (code metrics below are still valid)")
            processor = None
    except Exception as exc:
        print(f"[audio] DAC unavailable ({type(exc).__name__}: {exc}); "
              "skipping wavs (code metrics below are still valid)")
        processor = None
```

Add `"dac_backend": getattr(getattr(processor, "dac", None), "backend", None)`
to the `report` dict at `:255`.

**Tests for Phase 1 (add to `tests/test_codeswitch_tts.py`; import `DACWrapper`
at `:14`):**

```python
def test_dacwrapper_refuses_random_hf_local_weights(monkeypatch):
    from transformers import DacModel
    monkeypatch.setattr(DacModel, "from_pretrained",
                        classmethod(lambda cls, *a, **k: (_ for _ in ()).throw(RuntimeError("offline"))))
    try:  # make the test hermetic when descript-audio-codec is installed
        import dac
        monkeypatch.setattr(dac.utils, "download",
                            lambda **k: (_ for _ in ()).throw(RuntimeError("offline")))
    except ImportError:
        pass
    with pytest.raises(RuntimeError):
        DACWrapper()
    # and the opt-in path still yields a detectable synthetic backend
    wrapper = DACWrapper(allow_fallback=True)
    assert wrapper.backend == "fallback_neural"

def test_audioprocessor_passed_model_skips_loading():
    ap = AudioProcessor(dac_model=object())
    assert ap.dac.backend == "passed_model"
```

Also change the existing `test_end_to_end_streaming_pipeline`
(`tests/test_codeswitch_tts.py:98`) to take `monkeypatch` and stub the loader, so
the suite does not download the 306 MB checkpoint when the hub is online:

```python
def test_end_to_end_streaming_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(
        DACWrapper, "_load_dac",
        lambda self: (setattr(self, "backend", "fallback_neural"),
                      setattr(self, "model", None)),
    )
    ...
```

(The existing `allow_untrained_fallback = True` still exercises the synthetic
path; the assertion set is unchanged.)

(If the official `dac` package happens to be installed in the test environment,
monkeypatch `dac.utils.download` as well so the test remains hermetic.)

**Phase 1 verification**

```bash
movio/.venv/bin/python -m pytest tests/test_codeswitch_tts.py -q -p no:cacheprovider
movio/.venv/bin/python -c "from transformers import DacModel; m=DacModel.from_pretrained('ylacombe/dac_44khz'); print(type(m).__name__, sum(p.numel() for p in m.parameters()))"
```

Expected: tests pass; the loader prints `DacModel` and `76620777`-class parameter
count. If the second command cannot reach the hub locally, record it as an
environment limitation and rely on the Kaggle smoke test — do not replace it with
a random model.

### Phase 2 — Argmax plumbing

**Files:** `codeswitch_tts/models/draft_transformer.py`, `run_full_evaluation.py`.

**2a. `DraftTransformer.generate_chunk` (`draft_transformer.py:217-238`)** —
add `sample: bool = True` after `temperature` and forward it:

```python
def generate_chunk(
    self,
    text: Union[str, List[str]],
    num_frames: Optional[int] = None,
    temperature: float = 1.0,
    sample: bool = True,
    device: Optional[Union[str, torch.device]] = None,
) -> DraftOutput:
    ...
    return self.forward(
        input_ids=input_ids,
        attention_mask=mask,
        num_frames=num_frames,
        temperature=temperature,
        sample=sample,
    )
```

**2b. `run_full_evaluation.py` parser (`:3125`)** — add:

```python
p.add_argument("--sample-mode", choices=["sample", "argmax"], default="sample",
               help="Token selection for B/C draft CB0 and residual CB1-8: 'sample' "
                    "(seeded multinomial, current behaviour) or 'argmax' (greedy).")
```

**2c. `_chunked_synthesis` (`:925-929` and `:1002-1007`)** — read the mode from
`self.args` (never add a parameter; the C-gate test monkeypatches this method
with four parameters) and pass it to both models:

```python
draw_sample = getattr(self.args, "sample_mode", "sample") != "argmax"
...
draft_out = self.draft.generate_chunk(
    text=normalized, num_frames=n_frames, temperature=1.0,
    sample=draw_sample, device=self.device,
)
...
res_out = self.residual(
    cb0_tokens=torch.from_numpy(kept).unsqueeze(0).to(self.device),
    temperature=1.0,
    sample=draw_sample,
)
```

**2d. `_draft_confidence` (`:1119-1121`)** — pass the same mode for consistency
(the probabilities it reads are mode-invariant, but the call should be explicit):

```python
sample=getattr(self.args, "sample_mode", "sample") != "argmax",
```

**2e. Resume safety (`:3340-3341`, `:3365`, `:3420-3425`) and aggregation
(`:1614-1696`)** — normalize Config A's mode to the literal `"n/a"` and include
the mode at **all** key sites. `_base_row` now writes `"n/a"` for A, so
`str(r.get("sample_mode", "sample"))` would produce `"None"` and never match:

```python
# before the loop
sample_mode = str(getattr(args, "sample_mode", "sample"))
def _mode_key(row, cfg):
    if cfg == "A":
        return "n/a"
    # Rows written before this flag existed were sampled; do not relabel them
    # as the current run's mode on resume.
    return str(row.get("sample_mode") or "sample")

done = {(r.get("config"), r.get("utt_id"), float(r.get("gate_threshold") or 0.0),
         _mode_key(r, r.get("config")))
        for r in rows if not r.get("error")}
...
# membership
expected_mode = "n/a" if cfg == "A" else sample_mode
if (cfg, utt["id"], threshold, expected_mode) in done and not args.force:
    continue
...
# replace filter
rows = [r for r in rows
        if not (r.get("config") == cfg and r.get("utt_id") == utt["id"]
                and float(r.get("gate_threshold") or 0.0) == float(threshold)
                and _mode_key(r, cfg) == expected_mode)]
...
# append
done.add((cfg, utt["id"], float(threshold), expected_mode))
```

C-fallback rows keep `"sample_mode"` equal to the requested B/C mode (set in
`synth_C`), so they match the loop key. A row is `"n/a"` and matches the loop key
for Config A.

Also fail closed on mixed modes: at the start of `aggregate_results`, collect
`{str(r.get("sample_mode") or "n/a") for r in rows if r.get("config") in ("B", "C")}`;
if it has more than one element, raise `SystemExit("results mix sample and
argmax rows; use a fresh --results file")`. `run_meta.json` already records
`args.sample_mode`, but it does not prevent an old JSONL from being reused.

**Tests for Phase 2:**

```python
def test_generate_chunk_argmax_is_deterministic():
    model = DraftTransformer(); model.eval()
    out_a = model.generate_chunk("Hello", num_frames=8, sample=False)
    out_b = model.generate_chunk("Hello", num_frames=8, sample=False)
    assert torch.equal(out_a.tokens, out_b.tokens)
    assert torch.equal(out_a.tokens, out_a.probabilities.argmax(dim=-1))

def test_arg_parser_exposes_sample_mode():
    assert R.build_arg_parser().parse_args([]).sample_mode == "sample"
```

Plus the existing `test_draft_transformer_forward` must still pass unchanged
(default `sample=True`).

**Phase 2 verification**

```bash
movio/.venv/bin/python -m pytest tests/test_codeswitch_tts.py tests/test_evaluation_suite.py -q -p no:cacheprovider
movio/.venv/bin/python run_full_evaluation.py --help | grep -A1 -- --sample-mode
```

### Phase 3 — Fix draft training's frame-stretch bug

**File:** `train_draft_distill.py`.

**3a. Add a pure helper above `DraftTrainer`** (so it is unit-testable without a
model):

```python
def length_bucketed_order(lengths: Sequence[int], batch_size: int,
                          shuffle: bool = False, rng: Optional[random.Random] = None
                          ) -> List[int]:
    """Group indices with similar frame counts so `_collate`'s max_t is close
    to every sample's own length (the previous shuffle-only batches stretched
    short sentences over the longest sample's frame axis)."""
    order = sorted(range(len(lengths)), key=lambda i: int(lengths[i]))
    blocks = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if shuffle:
        (rng or random).shuffle(blocks)
    return [i for block in blocks for i in block]
```

**3b. `DraftTrainer._run_epoch` (`:705-713`)** — use it instead of
`random.shuffle(order)`:

```python
order = length_bucketed_order([s.frames for s in shards], self.batch_size,
                              shuffle=train, rng=random)
```

(`random.shuffle` on a list works; the helper returns a new list.)

**3c. `DraftTrainer.evaluate_topk` (`:749-753`)** — use the same helper with
`shuffle=False` so validation is bucketed too:

```python
order = length_bucketed_order([s.frames for s in shards], self.batch_size,
                              shuffle=False)
for start in range(0, len(order), self.batch_size):
    batch = [shards[j] for j in order[start:start + self.batch_size]]
```

**3d. Do not train the draft alone here.** A `--skip-residual` draft-only run
writes `meta.json` with an empty `residual` dict and would be immediately
overwritten; a later `--skip-draft` residual run silently disables scheduled
sampling (`train_draft_distill.py:1331`). The single training command is Phase
4f, which trains both stages in order. If you need a draft curve for debugging,
use `--dry-run` locally or inspect the combined run's `meta.json["draft"]`.
**Always pass `--ckpt-dir /tmp/codeswitch_dryrun` with `--dry-run`:** the dry run
writes real checkpoint files to the default `codeswitch_ckpt` and will overwrite
a restored v1. This happened once during implementation; the v1 artifact remains
in W&B, but a local copy must be re-restored with a valid `WANDB_API_KEY`.

**3e. Test** — add to `tests/test_corpus_builder.py` (local import, matching the
file's existing style at `:381,407`):

```python
def test_length_bucketed_order_keeps_lengths_together():
    import train_draft_distill as T
    lengths = [100, 5, 90, 7]
    order = T.length_bucketed_order(lengths, batch_size=2, shuffle=False)
    assert order == [1, 3, 2, 0]
    assert [lengths[i] for i in order] == [5, 7, 90, 100]
```

The helper sorts **ascending** by length, so `[5, 7, 90, 100]` is the correct
expectation; `[100, 90]` first would be descending and would not match the
implementation quoted in 3a.

**Phase 3 verification:** pytest only. The draft-quality gate is evaluated after
the combined Phase 4f run: compare `meta.json["draft"]["val_top1"]` against the
restored v1 `codeswitch_ckpt/meta.json` (`0.074269` in the run context). If it
did not materially improve, stop before spending evaluation time; the residual
cannot beat the CB0 it is fed.

### Phase 4 — Fix residual training: draft-CB0 input, shard-disjoint validation

**File:** `train_draft_distill.py`.

**4a. `precompute_draft_cb0` (`:896-899`)** — use the same greedy mode as the
planned eval, so training and inference see the same distribution:

```python
draft_out = self.draft.generate_chunk(
    text=shard.norm_text, num_frames=int(shard.frames),
    temperature=1.0, sample=False, device=self.device,
)
```

**4b. Scheduled-sampling clamp (`:881` and `:1023`)** — allow a true 100 %
draft-CB0 input; the current `min(0.9, ...)` silently leaves 10 % teacher input:

```python
self.sched_target = float(max(0.0, min(1.0, sched_target)))
...
target = float(max(0.0, min(1.0, sched_target if sched_target > 0 else self.sched_target)))
```

**4c. Shard-disjoint validation (`:1025-1034`)** — replace the whole block that
builds `windows` and the window-level split. The current `windows =
self.build_windows(shards, draft_cb0)` call must not remain (a literal patch
would build all windows twice and leave the guard/print referencing the wrong
variable):

```python
train_shards = [s for i, s in enumerate(shards) if i % 10 != 9]
val_shards = [s for i, s in enumerate(shards) if i % 10 == 9]
train_w = self.build_windows(train_shards, draft_cb0)
val_w = self.build_windows(val_shards, draft_cb0)
if not val_w and train_w:
    # Degenerate corpus: keep at least one validation window rather than
    # dividing by zero. This is the only allowed train/val overlap.
    val_w = train_w[-1:]
if not train_w:
    raise RuntimeError("No residual training windows could be built.")
print(
    f"[Residual] {len(train_w) + len(val_w)} windows of {self.WINDOW} frames "
    f"({len(train_w)} train / {len(val_w)} val, shard-disjoint)"
)
```

`build_windows` drops shards shorter than `WINDOW=86` (`:917-919`), so an empty
`val_w` is possible even when `val_shards` is non-empty; the fallback above is on
the **window** list, not the shard list. `ResidualTrainer.evaluate` tolerates an
empty `val_w` (`:1012` divides by `max(1, total)`), but keeping one window is
better for the early-stopping signal.

**4d. `best_epoch` and schedule provenance (`:1060-1089`)** — track the epoch
that produced `best_state`, and record how far the scheduled-sampling ramp
actually got (the ramp is epoch-based, so a wall-clock stop can leave it below
1.0 even when `residual_scheduled_sampling == 1.0`):

```python
if improved:
    best_epoch = epoch
...
final_sched_p = float(target * (epoch - 1) / max(1, epochs - 1))
stats.update(
    best_epoch=int(best_epoch),
    final_sched_p=final_sched_p,
)
```

`epoch` is the loop variable after the final iteration; if the loop is empty,
initialize `best_epoch = 0` and `final_sched_p = 0.0` before it.

**4e. Shard validation (`main`, `:1282-1288`)** — replace the unfiltered
`--skip-teacher` glob with a validating load:

```python
seen = {}
for f in shard_files:
    s = TeacherShard.load(f)
    if s.codes.ndim != 2 or s.codes.shape[0] != 9:
        raise SystemExit(f"[Teacher] {f.name}: codes shape {s.codes.shape}; expected (9, T)")
    if int(s.frames) != int(s.codes.shape[1]):
        raise SystemExit(f"[Teacher] {f.name}: frames={s.frames} but codes has "
                         f"{s.codes.shape[1]} timesteps; target would be truncated")
    if int(s.codes.min()) < 0 or int(s.codes.max()) >= 1024:
        raise SystemExit(f"[Teacher] {f.name}: codes outside [0, 1024): "
                         f"min={int(s.codes.min())} max={int(s.codes.max())}")
    if int(s.index) in seen:
        raise SystemExit(f"[Teacher] duplicate shard index {s.index}: {f.name} and {seen[int(s.index)]}")
    seen[int(s.index)] = f.name
    shards.append(s)
if len({s.target_source for s in shards}) > 1:
    print("[Teacher] ⚠ mixed target_source in cache; targets were captured with different methods")
```

**4f. Training command** — this is the **single** training run; it trains the
draft and then the residual, writes `meta.json` once after both stages, and is
the command the Phase 3/4 gates inspect. Do not run a draft-only command first:
`--skip-residual` would write an empty `residual` dict, and a later
`--skip-draft` residual run would silently disable scheduled sampling.

```bash
movio/.venv/bin/python train_draft_distill.py \
  --num-sentences 2000 --skip-teacher \
  --draft-epochs 300 --residual-epochs 300 --residual-sched 1.0 \
  --min-epochs 300 --max-minutes 60 --max-minutes-residual 90 \
  --teacher-cache teacher_cache --ckpt-dir codeswitch_ckpt_v2
```

On Kaggle this is `!python ...`; locally it can only be smoke-tested with
`--dry-run` unless the restored shards are present. The run is expected to stop
at the wall-clock caps if 300 epochs do not fit; inspect
`meta.json["draft"]["epochs_run"]` and `meta.json["residual"]["epochs_run"]`
rather than assuming 300.

**Phase 4 verification:** pytest; on Kaggle, `diagnose_audio_bug.py --ckpt
codeswitch_ckpt_v2` must report `rd_a >= 5 %` and `rt_a >= 5 %`. If `rt_a` is
healthy but `rd_a < 5 %`, the draft CB0 is still the bottleneck — do not proceed
to the B/C quality claim; go back to Phase 3.

**4g. Residual-only amendment (2026-10-20, after the v2 listening check).**
The matched v2 hybrids showed the residual is the primary bottleneck, so the
combined 4f command is **not** the next run. Use the v2 draft explicitly and
write a complete v3 directory:

```bash
movio/.venv/bin/python train_draft_distill.py \
  --skip-teacher --skip-draft --draft-ckpt codeswitch_ckpt_v2 \
  --residual-epochs 150 --residual-ramp-epochs 100 --residual-sched 1.0 \
  --residual-d-model 512 --residual-layers 6 \
  --min-epochs 150 --max-minutes-residual 90 \
  --teacher-cache teacher_cache --ckpt-dir codeswitch_ckpt_v3
```

The residual is validated on draft CB0, but the deployed state is the best one
from the **full-draft hold** (the final 50 epochs), not an early pre-ramp state.
`best_sched_p` records the selected epoch's schedule, and
`best_epoch_any`/`best_sched_p_any` preserve the all-epoch best for diagnosis.
`--skip-draft` fails closed unless `--draft-ckpt` is supplied; the draft is
copied into the output directory. Do not lower the 5 % gate; if v3 still misses
it, inspect `best_sched_p`, `cb_top1_teacher_mean`, and `cb_top1_draft_mean`
before choosing a stronger residual or draft soft targets.

### Phase 5 — Kaggle diagnostic + evaluation (user-driven session)

**Session/path rule.** Training writes `codeswitch_ckpt_v2` in the **current**
Kaggle session. The W&B `student` artifact always stores its payload under the
fixed entry name `codeswitch_ckpt` (`push_artifacts_to_wandb.py:60-64`), and
`bootstrap_diagnose.py` restores it to `codeswitch_ckpt`
(`bootstrap_diagnose.py:169-185`). Therefore:

| Situation | `--ckpt` to use |
|---|---|
| Same session after 5c | `codeswitch_ckpt_v2` |
| After a session restart / `bootstrap_diagnose.py --groups student` | `codeswitch_ckpt` |
| After a restart where you need the v2 name | `mv codeswitch_ckpt codeswitch_ckpt_v2` once, then use `codeswitch_ckpt_v2` |

Do not assume `codeswitch_ckpt_v2` survives a restart; it is a local directory,
not an artifact entry.

**5a. Restore the dependency stack and artifacts** (fresh session; requires the
`bootstrap_diagnose.py` commit from Phase 0, because the notebook resets to
`origin/main`):

```bash
!python bootstrap_diagnose.py --groups student,teacher --skip-diagnose
```

If it exits 1 because the student artifact is missing, check the
`WANDB_API_KEY`/Kaggle secret and retry with `--groups student --skip-install`;
if you only need the code metrics, `--skip-download` uses the local files.

**5b. Valid diagnostic audio (pre-retrain, v1):**

```bash
!python diagnose_audio_bug.py --ckpt codeswitch_ckpt \
  --teacher-cache teacher_cache --out /kaggle/working/audio_diag_pre
```

Gate: printed `DAC backend` is `hf_pretrained` or `dac_official`; the
`diagnostic_report.json["wavs"]` entries for the sampled shards have `rms > 0`,
`peak <= 0.95`, and at least one teacher wav has `flatness < 0.3` and
`zcr < 0.3`. The script samples `--shards 4` by default, so inspect the actual
ids it prints; do not hardcode `shard_0090`. Listen to the teacher wav and the
`student_full_argmax.wav` it writes.

**5c. Retrain both stages** with the Phase 3/4 command (new
`codeswitch_ckpt_v2`). Then run the Phase 4 verification against
`codeswitch_ckpt_v2/meta.json` before evaluating.

**5d. Push the new student**:

```bash
!python push_artifacts_to_wandb.py --group student --ckpt-dir codeswitch_ckpt_v2
```

The payload is stored under the artifact entry `codeswitch_ckpt`; this
overwrites the `codeswitch-student` `:latest`, while v1 remains in W&B version
history. On a later restore, use `--ckpt codeswitch_ckpt` (see the table above).

**5e. Diagnostic on the retrained student (same session):**

```bash
!python diagnose_audio_bug.py --ckpt codeswitch_ckpt_v2 \
  --teacher-cache teacher_cache --out /kaggle/working/audio_diag_post
```

Use a **different** `--out` directory from 5b; otherwise the pre-retrain DAC
evidence is overwritten by identically named wavs and `diagnostic_report.json`.

**5f. Argmax evaluation (fresh results dir):**

```bash
!python run_full_evaluation.py --ckpt codeswitch_ckpt_v2 \
  --configs A,B,C --sample-mode argmax \
  --results eval_results_2k_argmax/raw_results.jsonl \
  --audio-dir eval_audio_2k_argmax \
  --asr-model openai/whisper-large-v3 --force
```

**5g. Sampled baseline on the same checkpoints (A/B comparison only; B/C):**

```bash
!python run_full_evaluation.py --ckpt codeswitch_ckpt_v2 \
  --configs B,C --sample-mode sample \
  --results eval_results_2k_sample/raw_results.jsonl \
  --audio-dir eval_audio_2k_sample \
  --asr-model openai/whisper-large-v3 --force
```

If either eval exits before synthesis, first re-check the `--ckpt` path against
the table above (a restored session has `codeswitch_ckpt`, not
`codeswitch_ckpt_v2`). If the engine/ASR/UTMOS stack fails, run a 2-utterance
smoke without the metric models before the full run:

```bash
!python run_full_evaluation.py --ckpt codeswitch_ckpt_v2 --configs B,C \
  --limit 2 --skip-asr --no-utmos --no-mcd --sample-mode argmax \
  --results eval_results_smoke/raw_results.jsonl --audio-dir eval_audio_smoke --force
```

Do not treat the smoke numbers as quality evidence; it only proves the decode
path and backend fields.

**5h. Decision gate.** Run this after 5f and 5g; it reads the two
`summary.json` files and exits non-zero if any §1.5/§1.6 condition fails. The
CER helper is the same rule as `_cer_p50` (`run_full_evaluation.py:2093`):
prefer `cer_latin.p50` when the run recorded Latin values, otherwise `cer.p50`.

```bash
movio/.venv/bin/python - <<'PY'
import json, sys
from pathlib import Path

def load(path):
    return json.loads(Path(path).read_text())["aggregates"]

arg = load("eval_results_2k_argmax/summary.json")
samp = load("eval_results_2k_sample/summary.json")
fails = []

def check(ok, msg):
    print(("PASS  " if ok else "FAIL  ") + msg)
    if not ok:
        fails.append(msg)

def cer_p50(a):
    vals = (a.get("values") or {}).get("cer_latin") or []
    key = "cer_latin" if any(v is not None for v in vals) else "cer"
    v = (a.get(key) or {}).get("p50")
    return None if v is None else float(v)

for cfg in ("A", "B", "C"):
    check((arg.get(cfg) or {}).get("n", 0) > 0, f"argmax {cfg} has rows")

a_cer, b_cer, c_cer = (cer_p50(arg.get(k) or {}) for k in "ABC")
check(a_cer is not None and b_cer is not None,
      "CER p50 present for A and B")
if a_cer is not None and b_cer is not None:
    check(100 * b_cer <= 100 * a_cer + 10,
          f"B CER p50 {100*b_cer:.1f}% <= A {100*a_cer:.1f}% + 10 pp")
if b_cer is not None and c_cer is not None:
    check(100 * c_cer <= 100 * b_cer + 10,
          f"C CER p50 {100*c_cer:.1f}% <= B {100*b_cer:.1f}% + 10 pp")

a_u = ((arg.get("A") or {}).get("utmos") or {}).get("mean")
b_u = ((arg.get("B") or {}).get("utmos") or {}).get("mean")
check(a_u is not None and b_u is not None, "UTMOS mean present for A and B")
if a_u is not None and b_u is not None:
    check(b_u >= 0.75 * a_u, f"B UTMOS {b_u:.3f} >= 0.75 * A {a_u:.3f}")

for cfg in ("A", "B", "C"):
    check(int((arg.get(cfg) or {}).get("asr_fail", -1)) == 0,
          f"{cfg} asr_fail == 0 (CER <= {2.0:.1f})")

for cfg in ("B", "C"):
    for key, label in (("ttfa_compute_warm", "TTFA-compute-warm p50"),
                       ("rtf", "RTF p50")):
        x = ((arg.get(cfg) or {}).get(key) or {}).get("p50")
        y = ((samp.get(cfg) or {}).get(key) or {}).get("p50")
        ok = x is not None and y not in (None, 0) and abs(float(x) / float(y) - 1) <= 0.25
        check(ok, f"{cfg} {label} {x} vs sample {y} within +/-25%")

print(f"\n{len(fails)} gate failure(s)")
sys.exit(1 if fails else 0)
PY
```

If the gate exits 0, proceed to Phase 6. If it exits non-zero, stop: B/C quality
claims remain BLOCKED; capture the per-codebook `cb_top1` list and the
`rt_a`/`rd_a` values in the W&B run and report the negative result. Do not
loosen a threshold after seeing the numbers.

### Phase 6 — Claims and docs (only if the gate passed)

- Update `PAPER_CLAIMS_AND_EVIDENCE.md` with the measured argmax/sample table,
  the DAC-backend provenance, and the exact command lines.
- Record the new test count and the plan status in `plans/README.md`.
- Ensure `bootstrap_diagnose.py`, `plans/013-bc-noise-fix.md`, and
  `research/findings-2026-10-20.md` are committed on `fix/013-bc-noise`; the
  Kaggle notebook cannot see untracked files after its `origin/main` reset.
- Do **not** edit `KAGGLE_RUN.md`; put any runbook amendments in
  `plans/README.md`.
- Optional: update the train/eval cells of both Kaggle notebooks to the new
  commands (`--sample-mode argmax`, `--draft-epochs 300`, `--residual-sched 1.0`,
  `codeswitch_ckpt_v2`). This is a separate commit and must not touch the
  dependency cell.
- Do not push. The maintainer merges and pushes; the Kaggle session can only see
  the branch after that.

---

## 6. Risks & Mitigations

| Risk | Why it is real | Mitigation |
|---|---|---|
| `DacModel.from_pretrained("ylacombe/dac_44khz")` cannot reach the hub in the test environment | The local venv has no cached 223-tensor DAC; tests must be hermetic | Tests monkeypatch `from_pretrained`; the live load is only a smoke check. Kaggle has network. Official `descript-audio-codec` is the second attempt. |
| The new refusal breaks `test_end_to_end_streaming_pipeline` | It constructs a no-Parler pipeline and previously got the random DAC silently | `allow_fallback` is wired to the existing `config.system.allow_untrained_fallback` (the test sets it True); the synthetic path is explicit, not random. To keep the suite hermetic and fast, monkeypatch `DACWrapper._load_dac` in that test so it does not download the 306 MB `ylacombe/dac_44khz` checkpoint when the hub is reachable. |
| A resumed session cannot find `codeswitch_ckpt_v2` | W&B always restores the `student` payload under the entry name `codeswitch_ckpt` | Use the Phase 5 table: same-session → `codeswitch_ckpt_v2`; after restart → `codeswitch_ckpt` (or `mv` once). Do not push and then assume the v2 directory name survives. |
| Argmax changes Config C semantics | C's `rejection_sample` assumes proposal tokens came from `chunk_probs`; argmax is a top-1 proposal | Keep `sample` the default; `argmax` is opt-in and documented. C remains stochastic because residual resampling still runs. Report `sample_mode` per row. |
| Retraining still fails `rd_a >= 5 %` | CB1-8 given only CB0 may have a low intrinsic ceiling; no measurement exists | The gate is a STOP, not a claim. Next levers (deferred): residual capacity bump (`d_model`, layers) or multi-sample teacher targets (expensive teacher re-pass). Do not silently proceed. |
| `--skip-teacher` trains on wrong/duplicate shards | Old `--skip-teacher` globs unfiltered; `shard.index` collisions cross-wire `precompute_draft_cb0` | Phase 4e validates `(9,T)` shape, duplicate indices, and mixed `target_source` before training. |
| Old `codeswitch_ckpt` is overwritten | W&B artifact name is fixed | New `codeswitch_ckpt_v2`; the old artifact version remains in W&B history. |
| Plan 006 conflicts on `audio.py` | Both edit `DACWrapper._load_dac` | Execute 013 first; plan 006's DAC-loader section is superseded. Rebase 006 by content or drop that section. |
| Random-DAC regression returns via engine-less eval | `--configs B` with no Parler still uses the fallback | With this plan, the fallback uses `AudioProcessor(..., allow_fallback=False)` unless `--allow-untrained`, so it raises instead of silently emitting noise; the `dac_backend` row field makes it visible. |
| DAC official fallback downloads from GitHub Releases unverified | Prior audit: `dac.utils.download` uses no hash check | It is only the second attempt; the transformers-native path should win on Kaggle. Supply-chain pinning remains plan 006's job. |

---

## 7. Verification Checklist

Phase 0
- [ ] `git log --oneline -5` includes `6bcc188` (or descendant); `git status` shows only the new plan/research files plus `bootstrap_diagnose.py`.
- [ ] `movio/.venv/bin/python -m pytest tests/ -q -p no:cacheprovider` → 120 passed (or the recorded post-plan baseline).

Phase 1
- [ ] `grep -n "hf_local" codeswitch_tts/utils/audio.py` returns no reference outside comments.
- [ ] `movio/.venv/bin/python -m pytest tests/test_codeswitch_tts.py::test_dacwrapper_refuses_random_hf_local_weights tests/test_codeswitch_tts.py::test_audioprocessor_passed_model_skips_loading -q -p no:cacheprovider` → 2 passed.
- [ ] `movio/.venv/bin/python -c "from transformers import DacModel; m=DacModel.from_pretrained('ylacombe/dac_44khz'); print(type(m).__name__, sum(p.numel() for p in m.parameters()))"` → `DacModel 76620777` (verified locally 2026-10-20, first load 96.6 s; re-check on Kaggle 4.46.1).
- [ ] `AudioProcessor(dac_model=object()).dac.backend == "passed_model"`.

Phase 2
- [ ] `movio/.venv/bin/python run_full_evaluation.py --help` lists `--sample-mode {sample,argmax}`.
- [ ] `generate_chunk(sample=False)` is deterministic and equals `probabilities.argmax(-1)`.
- [ ] Existing `test_synth_C_falls_back_below_threshold` still passes (no new `_chunked_synthesis` parameter).
- [ ] `_base_row` includes `sample_mode` (`"n/a"` for A, requested mode for
      B/C); the B/C row update includes `dac_backend` set by `decode_codes`
      (not a pre-decode guess); C-gate fallback rows set `sample_mode` to the
      requested mode and keep a `dac_backend` from the allowed set.

Phase 3–4
- [ ] `test_length_bucketed_order_keeps_lengths_together` passes.
- [ ] `--dry-run --ckpt-dir /tmp/codeswitch_dryrun` still passes on CPU (never
      run it with the default checkpoint directory).
- [ ] On Kaggle, `codeswitch_ckpt_v2/meta.json` `draft.best_epoch` and `draft.val_top1` are recorded and compared against the restored v1 `meta.json` (`0.074269` in the run context); the plan does not hardcode a new top-1 threshold.
- [ ] `codeswitch_ckpt_v2/meta.json` shows `residual.epochs_run >= 60` and
      `residual.final_sched_p >= 0.95` (the actual epoch-based ramp, not just
      the configured `residual_scheduled_sampling == 1.0`). If the run stopped
      at the wall-clock cap with `final_sched_p < 0.95`, the residual has not
      seen enough draft-CB0 input — retrain before evaluating.
- [ ] Residual-only amendment: `codeswitch_ckpt_v3/meta.json` records
      `residual_only=true`, `selected_full_draft=true`, `best_sched_p=1.0`,
      `best_val_sched_p`, `best_epoch_any`, `cb_top1_teacher_mean`, and
      `cb_top1_draft_mean`; the copied `draft_cb0.pt` matches the v2 source;
      `diagnose_audio_bug.py --ckpt codeswitch_ckpt_v3` still passes the
      unchanged 5 % `rt_a`/`rd_a` gate.
- [ ] `--skip-teacher` validation rejects a shard whose `frames !=
      codes.shape[1]`, whose codes fall outside `[0, 1024)`, or whose `index`
      duplicates another shard.

Phase 5
- [ ] Diagnostic prints `DAC backend: hf_pretrained` (or `dac_official`); no `hf_local`.
- [ ] `diagnostic_report.json["wavs"]` for the sampled shards: every entry has `rms > 0`, `peak <= 0.95`; at least one teacher wav has `flatness < 0.3`, `zcr < 0.3`; listen to it.
- [ ] Diagnostic summary: `rt_a >= 5 %`, `rd_a >= 5 %`; record `cb0_a` and the sampled shard ids (no hardcoded shard number).
- [ ] Argmax eval: 60 utterances × 3 configs, 0 errors; no `[Decode] teacher DAC decode failed` in the log; B/C rows carry `sample_mode="argmax"` and `dac_backend` in the allowed set (A → `parler_streamer`; B/C with engine → `parler_audio_encoder`; engine-less → `hf_pretrained`/`dac_official`; `fallback_neural` only with `--allow-untrained`).
- [ ] Argmax B/C vs the matched sample run: `B_argmax_CER <= A_CER + 10`, `B_argmax_UTMOS >= 0.75 * A_UTMOS`, `C_argmax_CER <= B_argmax_CER + 10`, no ASR loops.
- [ ] Latency: B/C TTFA and RTF within ±25 % of the matched sample-mode run on the same checkpoints.

Phase 6
- [ ] `PAPER_CLAIMS_AND_EVIDENCE.md` contains only measured numbers from the run, with the exact commands and the checkpoint fingerprint.
- [ ] `plans/README.md` records the new test count and the 013 status; `KAGGLE_RUN.md` untouched.
- [ ] `bootstrap_diagnose.py`, the plan, and the research log are committed on the branch.
- [ ] W&B run contains the argmax/sample comparison and the diagnostic report.

---

## 8. Open Questions (decisions no source can settle)

1. **Retrain gate:** if the retrained student clears `rt_a` but not `rd_a >= 5 %`,
   do you want (a) a residual capacity bump (`d_model` 384 → 512, 4 → 6 blocks),
   (b) a second teacher pass with K samples per sentence for distributional
   targets, or (c) report the negative result and keep B/C out of the quality
   table? Recommendation: (c) first, then (a) — (a) is cheap and does not need a
   teacher re-pass.
2. **Artifact naming:** `push_artifacts_to_wandb.py --group student` overwrites
   the `codeswitch-student` `:latest`. Keep that (v1 still in history) or add a
   versioned artifact name for the v2 student? Recommendation: keep the fixed
   name; the W&B version history already preserves v1.
3. **Plan 006:** run 013 first and then rebase 006's DAC-loader section, or run
   006 first and adapt 013 to its pinned loader? Recommendation: 013 first — 006's
   current text expects the old broken behavior and would need rewriting either
   way.
