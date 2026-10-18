# Kaggle Runbook — CodeSwitch-TTS Distillation + Evaluation

Single-T4, single-command workflow. Everything is resumable; nothing needs a restart
between cells except the one-time dependency install.

| Step | Command | Wall-clock (T4) | Artifacts |
|------|---------|-----------------|-----------|
| 1 | setup cell (below) | 4–8 min + 1 restart | — |
| 2 | `python train_draft_distill.py` | 35–55 min | `teacher_cache/shard_*.npz`, `codeswitch_ckpt/{draft_cb0.pt,residual_cb1_8.pt,meta.json}` |
| 3 | `python run_full_evaluation.py --ckpt codeswitch_ckpt` | 25–40 min | `eval_results/{raw_results.jsonl,run_meta.json}`, `paper_figures/{fig1,fig2,fig3,fig4}.{png,pdf}`, `paper_figures/{tables.tex,summary.json}`, `eval_audio/{A,B,C}/*.wav` |

---

## 0) Notebook settings

* Accelerator: **GPU T4 x1** (16 GB)
* Internet: **ON**
* The target model (`ai4bharat/indic-parler-tts`) is **gated**: accept its license on
  Hugging Face, then log in once per session:

```python
%pip install -q huggingface_hub
from huggingface_hub import login
login(token="hf_xxxxxxxxxxxxxxxxx")   # token with access to ai4bharat/indic-parler-tts
```

---

## 1) Setup cell (once per session — then *Run All* again after the restart)

```python
import os, sys, importlib.metadata as md

REPO_URL = "https://github.com/tripathiji1312/codeSwitch.git"
CLONE_DIR = "/kaggle/working/codeSwitch"

# --- 1. Sync the repository (fast-forward, or clone fresh) -------------------
if os.path.exists(os.path.join(CLONE_DIR, ".git")):
    if os.system(f"git -C {CLONE_DIR} fetch origin main") != 0:
        os.system(f"rm -rf {CLONE_DIR} && git clone {REPO_URL} {CLONE_DIR}")
    else:
        os.system(f"git -C {CLONE_DIR} reset --hard FETCH_HEAD")
else:
    os.system(f"rm -rf {CLONE_DIR} && git clone {REPO_URL} {CLONE_DIR}")

# --- 2. Decide from the INSTALLED state whether to (re)install ---------------
def _deps_ok() -> bool:
    try:
        if tuple(map(int, md.version("protobuf").split(".")[:2])) < (5, 26):
            return False                      # no google.protobuf.runtime_version
        if md.version("transformers") != "4.46.1":
            return False
        md.version("parler-tts")              # raises if parler-tts is missing
        md.version("sentencepiece")
        md.version("jiwer")                   # eval-suite dependency
        md.version("matplotlib")
        return True
    except Exception:
        return False

if not _deps_ok():
    print("Installing dependencies (staged to avoid package conflicts)...")
    os.system("pip install -q -U git+https://github.com/huggingface/parler-tts.git")
    os.system("pip install -q 'transformers==4.46.1'")
    os.system("pip install -q descript-audio-codec soundfile scipy accelerate sentencepiece")
    os.system("pip install -q jiwer matplotlib seaborn pandas")
    # protobuf MUST be installed LAST and stay >= 5.26 (ships google.protobuf.runtime_version).
    # parler-tts -> descript-audiotools pins protobuf<5 and silently downgrades it to 4.25.9,
    # which breaks the Parler-TTS import and forces the random-weight fallback backend.
    os.system("pip install -q 'protobuf==5.29.5'")
    print("✓ Dependencies installed! Restarting the kernel is required.")
    print("👉 Session → Restart, then 'Run All' again (this cell becomes a no-op).")
    import IPython
    IPython.Application.instance().kernel.do_shutdown(True)
else:
    print("✓ Dependencies already installed.")

# --- 3. Sanity-check the Parler import (the real cause of static-noise runs) --
try:
    from parler_tts import ParlerTTSForConditionalGeneration  # noqa: F401
    print("✓ parler_tts import OK")
except Exception as e:
    print(f"⚠ parler_tts import FAILED: {type(e).__name__}: {e}")
    print("  Fix: !pip install -q 'protobuf==5.29.5'  then Session → Restart → Run All.")

# --- 4. Make the packaged code importable ------------------------------------
for path_candidate in [CLONE_DIR, ".", ".."]:
    if os.path.exists(os.path.join(path_candidate, "codeswitch_tts")):
        if path_candidate not in sys.path:
            sys.path.insert(0, path_candidate)
print("✓ Setup complete. Work from:", CLONE_DIR)
```

---

## 2) Distillation cell — gives the draft/residual models real weights

```python
%cd /kaggle/working/codeSwitch
!python train_draft_distill.py --num-sentences 300 --ckpt-dir codeswitch_ckpt
```

* **Resume**: safe to re-run at any time. Teacher shards are cached per sentence in
  `teacher_cache/shard_XXXX.npz`, so a second run only collects what is missing.
  `--skip-teacher` reuses the cache entirely; `--skip-draft` / `--skip-residual` skip a stage.
  The teacher loop prints an ETA every 10 sentences — Ctrl-C stops cleanly and a re-run
  resumes from the last cached sentence (use `--num-sentences 400` for a ~25% larger corpus).
* **Teacher targets**: `--target-source auto` (default) captures the LM tokens once and
  validates them by decoding the captured codes and comparing the waveform against the
  teacher's returned audio; the validation line now prints the evidence, e.g.
  `frames=241 frame_samples=123392 decoded=123392 teacher=123392 len_ratio=1.000 corr=0.999`.
  Both the length ratio (≥ 0.90) and the zero-lag correlation (≥ 0.90) must pass; otherwise
  it falls back to DAC re-encoding of the teacher waveform (`--target-source lm|codec` to
  force either). A `MISMATCH` line with `decode failed (...)` included usually means the
  audio encoder's `decode` signature differs — the error text tells you which layout failed.
* **Budgets**: draft training ≤ 12 min, residual ≤ 6 min (wall-clock caps, `--max-minutes*`).
* **Smoke test** (no GPU, ~30 s): `!python train_draft_distill.py --dry-run`
* **Custom corpus**: `--corpus-json my_sentences.json` (list of strings or one per line);
  the corpus is asserted disjoint from the 60 eval utterances.

Outputs: `codeswitch_ckpt/draft_cb0.pt`, `codeswitch_ckpt/residual_cb1_8.pt`,
`codeswitch_ckpt/meta.json` (includes the fitted text-length → frame-count model that the
evaluation uses for frame sizing). A short sanity WAV is rendered at the end.

---

## 3) Evaluation cell — 60 utterances × 3 configs + figures + LaTeX

```python
%cd /kaggle/working/codeSwitch
!python run_full_evaluation.py --ckpt codeswitch_ckpt
```

* **Resume**: results are written atomically to `eval_results/raw_results.jsonl` after every
  utterance. Re-running the same command skips completed (config, utterance) pairs; broken
  rows are retried automatically. `--force` re-runs everything; `--replay` rebuilds
  aggregates/figures/tables from the JSONL without synthesising anything (use it after a code
  update to refresh tables/figures at zero GPU cost).
* **Smoke test**: `!python run_full_evaluation.py --configs B --limit 3 --skip-asr --allow-untrained`
* **Options**: `--skip-asr` (no CER/WER, everything else still runs), `--no-save-audio`,
  `--no-true-probs` (unrelated-target acceptance only), `--play-steps 12`,
  `--boundary-frames 3`, `--dtype fp16`.

The run writes:

```
eval_results/raw_results.jsonl     # one row per (config, utterance): TTFA, RTF, CER, acceptance, chunk telemetry
eval_results/run_meta.json         # args, package versions, checkpoint provenance
paper_figures/fig1_latency_quality_tradeoff.{png,pdf}   # TTFA / RTF / CER distributions per config
paper_figures/fig2_boundary_acceptance_breakdown.{png,pdf} # per-zone CB0 acceptance (true vs ref.)
paper_figures/fig3_streaming_chunk_timeline.{png,pdf}   # streaming chunk pipeline + playback buffer
paper_figures/fig4_scaling_and_registers.{png,pdf}      # latency vs audio length + CER per register
paper_figures/tables.tex           # Tables I (comparative) + II (zones) + III (per-register), IEEE style
paper_figures/summary.json         # aggregates + figure paths
eval_audio/{A,B,C}/{utt_id}.wav    # listening samples
```

---

## 4) Download everything

```python
!cd /kaggle/working/codeSwitch && zip -qr /kaggle/working/codeswitch_artifacts.zip \
    paper_figures eval_results eval_audio codeswitch_ckpt -x 'codeswitch_ckpt/*.pt'; \
    ls -lh /kaggle/working/codeswitch_artifacts.zip
```

(Only `codeswitch_ckpt/meta.json` is included from the checkpoints — the `.pt` files stay on
Kaggle unless you want to archive them too.)

---

## 5) Reading the results

* **Table I** shows TTFA both ways: *TTFA-c* (component compute, the honest algorithmic
  number) and *TTFA-e2e* (wall-clock to the first audio buffer — for Config A it is measured
  through the real `ParlerTTSStreamer` in a background thread, which is why it is larger than
  the NAR configs).
* **Cold vs. warm TTFA**: the *first* utterance of every configuration pays one-off CUDA
  kernel/allocator/description-cache costs (on a T4 that is ~4.3 s vs ~0.7 s compute for the
  NAR configs — a ~6x outlier). Table I therefore reports the cold value as its own column and
  uses the median/p90 of the remaining 59 utterances (warm) everywhere else; Fig. 1 draws warm
  p50 bars with a black x marker for the cold value. Report warm numbers in the paper text and
  cite the cold column when discussing streaming start-up latency. **Caveat:** configurations
  are interleaved per utterance, so only the very first synthesis of the process (A/ta-01) is a
  true cold start — Table I footnotes that session value and flags the others as partially
  warmed. For a fully isolated per-configuration cold start, run one config per process,
  e.g. `--configs B --results eval_results/cold_B.jsonl`.
* **Acceptance** is reported twice: *true* (teacher-forced Parler probabilities) and *ref*
  (identical acceptance math against an **unrelated random target** — the legacy demo
  protocol). The ref column is a *mismatched-target baseline, not an upper bound*: a draft
  distilled from the real teacher agrees with the teacher more often than with a random
  target, so `acc_true > acc_ref` is the expected (and observed) ordering. A perfectly
  aligned draft would score 100 % against its own target.
* **Target correction (Config C)**: the console prints the teacher-forced mean log-likelihood
  (nats/frame) of the raw draft proposal vs. the tokens actually emitted after rejection
  sampling. If the emitted value is higher, the correction stage is doing its job (moving CB0
  toward the teacher distribution) even when the perceptual quality of the weak draft's audio
  is still poor. Only Config C rows synthesised *after* the 2026-10 update carry these fields
  (`logq_draft`/`logq_kept`); older rows show `--` until you re-run C with `--force`.
* **CER**: reported as medians in Table I (means are poisoned by whisper-small hallucination
  loops and by transliteration choices on the code-mixed registers). The console prints an
  ASR-failure census (CER > 200 %) with the offending utterance ids, and **Table III** gives
  the per-register (native / romanised / transit) CER median + mean and latency for every
  config. Listen to `eval_audio/A/rt-01.wav` etc. before blaming the metric on a specific row.
* **Fig. 2** is the boundary breakdown. Colour/arrow annotations are grey and marked `(n.s.)`
  when the boundary-vs-Tamil difference is smaller than the per-utterance standard deviation —
  at the current draft quality the zone effect is **not** resolved, so do not claim a
  boundary penalty in the paper unless the annotation is red.
* **Table III** (new) is the per-register table; it is the right place to discuss
  native-vs-code-mixed quality differences at this training scale.
* If `acc_true` is ≈ 0: the draft is not yet aligned with the teacher. Collect more teacher
  sentences (`--num-sentences 800`) or train longer (`--draft-epochs 150 --max-minutes 20`).
  Check `codeswitch_ckpt/meta.json` → `draft` for the reported top-1/top-5.
* If Fig. 2 is missing entirely: true acceptance was disabled (`--no-true-probs`) or the
  scorer could not initialise — see the `[Scorer]` lines in the log and
  `eval_results/run_meta.json` → `scorer_available`.

## 6) Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `⚠ Parler unavailable — reduced mode` or static noise | protobuf was downgraded by `descript-audiotools`. `pip install -q 'protobuf==5.29.5'`, Session → Restart, re-run. |
| `401 Client Error` / gated repo | the notebook is not logged in to Hugging Face, or the license of `ai4bharat/indic-parler-tts` was not accepted. |
| Config A/TTFA-c missing | `ParlerTTSStreamer` import failed; the run falls back to full generation and marks the row with a `note`. |
| Generation error per row | the row records `error` and is retried on the next run; the rest of the benchmark continues. |
| CUDA OOM during ASR | re-run with `--skip-asr` (everything except CER/WER still runs). |
| `TTFA` much larger for the first category/register | cold start (first synthesis in the session). `kaggle_demo.py` now runs a discarded warm-up pass first; `run_full_evaluation.py` always reports cold and warm separately. |
| `[Teacher] DAC re-encode failed: Input type (float) and bias type (c10::Half)...` | fixed — the waveform is now cast to the audio encoder's dtype before `encode()`. Pull the latest repo before re-running. |
| `[Teacher] LM token capture validation: MISMATCH` | the printed `len_ratio`/`corr` say why; the codec fallback takes over automatically. Capture and decode both try the 3-D and 4-D code layouts now, so a mismatch with `corr ≈ 1` means the trim differs, and `decode failed` means no layout matched. |
| `[Scorer] forward failed: ... 'last_hidden_state'` or Config C aborts with `operands could not be broadcast ... (1088,) (1024,)` | fixed — the scorer now reads the LM wrapper's `logits` (shape `(9, L, 1088)`, codebook 0 in row 0, valid codes in `[:1024]`). Pull the latest repo; the probe line must print `[Scorer] teacher-forced CB0 scoring: OK` for `acc_true` and Fig. 2 to appear. |
| Session dies mid-run | re-run the same command(s): teacher shards and evaluated rows are resumed. |

## 7) Where the knobs live

* Test set + training corpus: `codeswitch_benchmark_data.py` (20 native Tamil / 20 romanised
  Tanglish / 20 transit code-mixed utterances; `build_training_corpus(n, seed)`).
* Synthesis/timing/acceptance/zone logic + figures + tables: `run_full_evaluation.py`.
* Teacher collection + distillation + checkpoints: `train_draft_distill.py`.
* Local test suite: `python -m pytest tests/ -q` (39 tests, no GPU needed).
