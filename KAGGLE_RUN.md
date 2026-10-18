# Kaggle Runbook — CodeSwitch-TTS Distillation + Evaluation

Single-T4, single-command workflow. Everything is resumable; nothing needs a restart
between cells except the one-time dependency install.

| Step | Command | Wall-clock (T4) | Artifacts |
|------|---------|-----------------|-----------|
| 0 | corpus cell (`build_codeswitch_corpus.py --max-sentences 2000`) | 1–3 min | `codeswitch_corpus.txt` (2 000 lines, 34/33/33) |
| 1 | setup cell (below) | 4–8 min + 1 restart | — |
| 2 | teacher pass `--skip-draft --skip-residual` | 2–5 h (2000 lines ≈ 4.5 h) | `teacher_cache/shard_<hash>.npz` |
| 2b | `python train_draft_distill.py` (draft + residual) | ≈ 10 min | `codeswitch_ckpt/{draft_cb0.pt,residual_cb1_8.pt,meta.json}` |
| 3 | `python run_full_evaluation.py --ckpt codeswitch_ckpt` | 25–40 min | `eval_results/{raw_results.jsonl,run_meta.json}`, `paper_figures/{fig1,fig2,fig3,fig4}.{png,pdf}`, `paper_figures/{tables.tex,summary.json}`, `eval_audio/{A,B,C}/*.wav` |
| 3b | `python push_to_wandb.py` (optional) | 1–3 min | one W&B run: scalars + figures + tables + audio + `codeswitch-eval-*` artifact (the off-Kaggle copy of everything above) |

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

## 2a) Corpus cell — build a register-balanced distillation corpus

**The fresh-run corpus (recommended): 2 000 lines, ~4.5 h of teacher time**

```python
%cd /kaggle/working/codeSwitch
# template generator + real Tamil text (Wikimedia streamed + cached) + your own files
!python build_codeswitch_corpus.py --templates 2500 --wikipedia 900 --max-sentences 2000 \
    --out codeswitch_corpus.txt --json-out codeswitch_corpus.stats.json
```

```
Corpus written: codeswitch_corpus.txt | 2000 unique sentences | disjoint from eval set OK
  words/sentence: mean 8.6 (min 3, max 27)
  register mix (classified from the final text): native_tamil=680 (34%), romanized_tanglish=660 (33%), transit_codemixed=660 (33%)
```

* The benchmark is 20 native Tamil / 20 romanised / 20 transit, so the training corpus is
  **register-balanced** by construction: every base sentence is augmented `--passes` times
  (default 3), bucketed by its *classified* register, then subsampled to the
  `--balance 0.34,0.33,0.33` quota. Without this the generator's unique output is ~81 %
  romanised (pure-Tamil templates have almost no slots), i.e. the draft never sees the
  register the benchmark measures.
* Augmentation = English entity insertion (`order`, `OTP`, `UPI`, `Rs. 420`, plates …) and
  rule-based romanisation (`codeswitch_tts/frontend/tamil_latin.py`). Everything is deduped
  and asserted disjoint from the 60 eval utterances.
* `--wikipedia N` adds real Tamil text from a Wikimedia dump (streamed once and cached to
  `wikidata_tamil_sentences.txt`, so later builds are instant); `--wikipedia-file path.txt`
  uses a local file offline; `--extra-file my_sentences.txt my_sentences.json` mixes in your
  own text.
* **`--max-sentences N` is the budget knob**: the corpus size, not the epochs, is the quality
  bottleneck, but every sentence costs ≈ 7.9 s of teacher time on a T4. 2 000 lines ≈ 4.5 h,
  3 900 lines (no cap, `--templates 2500 --wikipedia 900`) ≈ 8.5 h. The cap keeps the
  34/33/33 mix and selection is deterministic and *prefix-stable*: raising the cap later only
  adds sentences, and the content-addressed teacher cache means only those new sentences are
  collected (see 2b).
* Sizing rule of thumb: `--max-sentences ≈ session_budget_hours × 450`.

## 2b) Distillation cell — gives the draft/residual models real weights

```python
%cd /kaggle/working/codeSwitch
!python train_draft_distill.py --corpus-json codeswitch_corpus.txt \
    --draft-epochs 60 --residual-sched 0.5 --max-minutes 40 --ckpt-dir codeswitch_ckpt
```

* **Budget reality**: the teacher pass is the bottleneck (≈ 7.9 s/sentence on a T4 measured
  over 300 sentences), so a 2 000-line corpus needs ≈ 4.5 h. Run the teacher pass first and
  on its own, then train — that keeps a session that dies at 90 % from losing the training
  budget too:

```python
# pass 1: teacher only  (~4.5 h for 2000 lines; Ctrl-C / session death is safe)
!python train_draft_distill.py --corpus-json codeswitch_corpus.txt \
    --skip-draft --skip-residual --ckpt-dir codeswitch_ckpt
# pass 2: draft + residual on the finished cache  (~15-25 min for 2000 lines)
!python train_draft_distill.py --corpus-json codeswitch_corpus.txt \
    --draft-epochs 60 --residual-sched 0.5 \
    --max-minutes 30 --max-minutes-residual 20 --ckpt-dir codeswitch_ckpt
```

`--max-minutes` (draft) and `--max-minutes-residual` (residual) are independent wall-clock
caps; both stages early-stop on their own validation split anyway. With 2 000 sentences an
epoch is ~7x longer than with 300, so give the residual the 20 minutes — its budget was the
binding constraint in the first full run.
* **`--max-minutes` only budgets draft training** — the teacher pass runs to completion or
  until you stop it. Ctrl-C is safe at any point; the collection line prints the cache state
  up front (`cache: 812/2000 sentences already collected (1188 to go, ~2h36m)`).
* **Save Version** (Kaggle: *Save & Run All*) after the teacher pass so `teacher_cache/`
  survives the session; the checkpoint/cache directories are a few MB (codes are int16).
* **Scheduled sampling** (`--residual-sched 0.5`): the residual decoder is trained on
  *teacher* CB0 but runs on *draft* CB0, so the exposure bias is real: with probability that
  ramps 0 → 0.5 across epochs, a training window's CB0 input is replaced by the draft's own
  CB0 for that text (the draft model is captured to `teacher_cache/draft_cb0_XXXX.npz`
  first). `--residual-sched 0.9` is the aggressive setting; 0 reproduces the old behaviour.
* **Resume**: safe to re-run at any time. Teacher shards are cached **by sentence content**
  (`teacher_cache/shard_<sha1(text)>.npz`), so growing, reordering or re-sourcing the corpus
  only collects the sentences that are actually new; position-named shards from older caches
  are re-indexed by their text and adopted (`adopted N position-named shards`).
  `--skip-teacher` reuses the cache entirely; `--skip-draft` / `--skip-residual` skip a stage.
  The teacher loop prints an ETA every 10 sentences — Ctrl-C stops cleanly and a re-run
  resumes from the last cached sentence.
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
!python run_full_evaluation.py --ckpt codeswitch_ckpt          # gate curve comes for free
```

Default `--gate-mode derived --gate-thresholds auto`: the run synthesises C only at τ=0 and then
builds the whole gate curve from the measured C and Pure-AR rows (the gate is decided *before*
synthesis, so the mixture is exact for quality/acceptance; a fallback's latency is the sum of the
two measured passes). `auto` takes the thresholds from the confidence the draft actually shows,
so the curve always has interior operating points — fixed thresholds like `0.6,0.75,0.9` fall
back on every utterance when the draft's mean top-1 is ≈0.1, which produces an empty curve. Use
`--gate-mode synth --gate-thresholds 0,0.3,0.34,0.36` if you want the gated mixture synthesised
end to end (the run prints a derived-vs-synthesised validation line whenever both exist).

* **Resume + provenance**: results are written atomically to `eval_results/raw_results.jsonl`
  after every utterance. Re-running the same command skips completed (config, utterance)
  pairs; broken rows are retried automatically. `eval_results/provenance.json` stores a hash of
  the checkpoints those rows came from, and a run with *different* weights refuses to resume
  (exit 3) instead of mixing two models' audio in one file — `--force` accepts the change and
  re-synthesises. `--force` re-runs everything; `--replay` rebuilds
  aggregates/figures/tables from the JSONL without synthesising anything (use it after a code
  update to refresh tables/figures at zero GPU cost — MCD is recomputed from the saved wavs).
* **Smoke test**: `!python run_full_evaluation.py --configs B --limit 3 --skip-asr --allow-untrained`
* **Options**: `--skip-asr` (no CER/WER, everything else still runs), `--no-save-audio`,
  `--no-true-probs` (unrelated-target acceptance only), `--no-utmos`, `--no-mcd`,
  `--play-steps 12`, `--boundary-frames 3`, `--dtype fp16`, `--wandb` (push at the end).
* **Runtime**: ≈ 25–40 min from scratch (whisper-small ASR included). With the default derived
  gate sweep no extra synthesis happens; only `--gate-mode synth` pays for the fallbacks
  (≈ 9 s each).
* **Gate**: utterance-level confidence gate on the draft's per-chunk mean top-1 probability
  (`--gate-stat utt_min|utt_mean`). Table I always reports the un-gated hybrid (τ=0); Fig. 5 +
  Table V are the Pareto curve. Real `--gate-mode synth` fallback rows are marked
  `verify_mode="ar_fallback"`, carry no acceptance numbers, and still count for latency and
  CER. Always look at the printed `[Gate]` lines: they list the thresholds, the derived-vs-
  synthesised validation and a warning if the curve has no interior point.

The run writes:

```
eval_results/raw_results.jsonl     # one row per (config, utterance, gate): TTFA, RTF, CER, acceptance, chunk + frame telemetry
eval_results/run_meta.json         # args, package versions, checkpoint provenance, scorer/UTMOS availability
paper_figures/fig1_latency_quality_tradeoff.{png,pdf}   # TTFA / RTF / CER distributions per config
paper_figures/fig2_boundary_acceptance_breakdown.{png,pdf} # per-zone CB0 acceptance (true vs ref.)
paper_figures/fig3_streaming_chunk_timeline.{png,pdf}   # streaming chunk pipeline + playback buffer
paper_figures/fig4_scaling_and_registers.{png,pdf}      # latency vs audio length + CER per register
paper_figures/fig5_gate_pareto.{png,pdf}                # confidence-gate quality/latency Pareto + fallback rate
paper_figures/tables.tex           # Tables I (comparative) + II (zones) + III (per-register) + IV (distance to switch) + V (gate sweep)
paper_figures/summary.json         # aggregates + gate_sweep + switch_distance + figure paths
eval_audio/{A,B,C}/{utt_id}.wav    # listening samples
```

All of it (plus the raw JSONL and the checkpoint meta) is mirrored into the W&B artifact by
§3b, which is what saves a run when a Kaggle session is recycled before the zip is downloaded.

---

## 3b) Push to Weights & Biases (optional, recommended)

Add `WANDB_API_KEY` once as a Kaggle secret (**Add-ons → Secrets**); the pusher resolves it as
`--api-key` → `$WANDB_API_KEY` → Kaggle secret, so no key ever lands in the notebook or the repo.

```python
%cd /kaggle/working/codeSwitch
!pip install -q wandb
!python push_to_wandb.py --project codeswitch-tts --tags kaggle,t4 \
    --figures-dir paper_figures --audio-dir eval_audio --ckpt-dir codeswitch_ckpt \
    --results eval_results/raw_results.jsonl
```

Or skip the extra cell and let the evaluation push itself:

```python
!python run_full_evaluation.py --ckpt codeswitch_ckpt --wandb --wandb-group c2500
```

What lands in the run:

* **scalars** — TTFA-c (warm/cold), TTFA-e2e, total latency, RTF, CER, `CER_lat`, WER, UTMOS, MCD,
  CB0 acceptance (teacher-forced + mismatched reference), the target-distribution
  log-likelihoods, the entire gate sweep (`gate/*/tau_*`), the distance-to-switch buckets and
  the training metrics from `codeswitch_ckpt/meta.json` (draft/residual val, CB0 top-1/top-5,
  length model, corpus size)
* **figures** — fig1–fig5 as W&B images (captions match the LaTeX ones)
* **tables** — configs / gate sweep / distance-to-switch / per-register, read directly in the UI
* **audio** — best/median/worst-CER sample per config, captioned with utterance, register and CER
* **artifact** — `codeswitch-eval-<run>` with `raw_results.jsonl`, `tables.tex`, `summary.json`,
  `run_meta.json`, the figures, the audio and `codeswitch_ckpt/meta.json`

Practical notes:

* **Zero-GPU refresh**: after a code change run `!python run_full_evaluation.py --replay --wandb`
  (or just re-run the pusher) — tables/figures are rebuilt from the saved JSONL and re-uploaded.
* **Overwrite/regroup**: `--run-id <wandb-run-id> --group c2500` appends to an existing run
  (`resume=allow`); `--group` is the natural place for the corpus size, so the 300-sentence and
  2500-sentence sweeps can be compared side by side.
* **Preview**: `--dry-run` prints every scalar group, figure, table, audio sample and artifact
  entry without touching the network (this is also what the test suite exercises).
* **Offline**: `--offline` sets `WANDB_MODE=offline` and writes the run to disk (sync later with
  `wandb sync`), useful when the session has no internet.
* A failed push never invalidates the evaluation: the script warns (`[W&B] ⚠ push failed: ...`)
  and exits 0 unless you pass `--strict`, and `_finish` still returns success.

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
* **CER_lat (script-robust CER)**: whisper-small freely chooses between Tamil script and Latin
  for code-mixed speech, which inflates the plain CER for reasons that have nothing to do with
  intelligibility. Every CER is therefore also computed after transliterating *both* the
  reference and the hypothesis to Latin (`codeswitch_tts/frontend/tamil_latin.py`); Table I / III
  and the figures use this variant when it is present. Report both numbers and say why they
  differ — a reviewer will ask.
* **Uncertainty**: every median column of Table I carries a 95 % percentile-bootstrap interval
  (2 000 resamples, fixed seed, no scipy) because 60 utterances cannot settle a 10-pt gap by
  itself. Quote the interval in the text, not just the point estimate.
* **UTMOS / MCD**: UTMOS22 (`torch.hub:tarepan/SpeechMOS:v1.2.0/utmos22_strong`, no listeners)
  scores every saved wav — it is a torch.hub repo, *not* a transformers checkpoint, so it needs
  GitHub access (a transformers mirror is tried as a fallback, and `run_meta.json` →
  `utmos_model` records which loader won). If both fail the columns are simply absent. MCD is
  DTW-aligned mel-cepstral distortion against the Pure-AR rendering of the *same text* (a
  different sampling seed, so it is a distribution-level spectral distance — say so in the
  caption). Both are optional (`--no-utmos`, `--no-mcd`) and are recomputed from the wavs by
  `--replay`, i.e. changing the metric does not cost a GPU run.
* **Fig. 2** is the boundary breakdown. Colour/arrow annotations are grey and marked `(n.s.)`
  when the boundary-vs-Tamil difference is smaller than the per-utterance standard deviation —
  at the current draft quality the zone effect is **not** resolved, so do not claim a
  boundary penalty in the paper unless the annotation is red.
* **Fig. 5 + Table V (gate sweep)**: the Pareto view of the confidence gate — quality (median
  CER_lat) against total latency, with the fallback rate on the second panel. The gate is the
  practical answer to "your NAR is fast but bad": τ=0 is the pure hybrid, higher τ buys CER back
  with AR latency. **Table IV** is the distance-to-switch curve: per-frame CB0 acceptance as a
  function of frames since the last language switch, with paired per-utterance bootstrap
  deltas vs the ≥ 16-frame reference bucket — the code-switching-specific science claim
  (do not read the zone buckets in Fig. 2 as a switch effect; they are not distance-resolved).
* **Table III** is the per-register table; it is the right place to discuss
  native-vs-code-mixed quality differences at this training scale.
* If `acc_true` is ≈ 0: the draft is not yet aligned with the teacher. Build a bigger corpus
  (`build_codeswitch_corpus.py --templates 2500 --wikipedia 2000`) and train it out
  (`--draft-epochs 60 --max-minutes 40`), or train longer on the same corpus. Check
  `codeswitch_ckpt/meta.json` → `draft` for the reported top-1/top-5.
* If Fig. 2 is missing entirely: true acceptance was disabled (`--no-true-probs`) or the
  scorer could not initialise — see the `[Scorer]` lines in the log and
  `eval_results/run_meta.json` → `scorer_available`.
* If Fig. 5 / Table V are missing: no gate sweep was recorded. Re-run with
  `--gate-thresholds 0,0.6,0.75,0.9` (old JSONLs contain only τ=0 rows).

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
| `[UTMOS] ⚠ perceptual proxy unavailable (...)` | both loaders failed (hub needs GitHub access; the transformers mirror needs `trust_remote_code`). Everything else still runs; `run_meta.json` → `utmos_available` says so. |
| `[Gate] ⚠ every requested threshold makes the same decision` | thresholds are outside the draft's confidence range (e.g. `0.6` when the draft's mean top-1 is `0.1`). Re-run with `--gate-thresholds auto` (default), which uses quantiles of the measured confidence. |
| MCD column shows `--` | no saved wavs (`--no-save-audio`) or `soundfile`/`torchaudio` missing. Re-run (or `--replay`) with the wavs in place. |
| `KeyError: 'residual_sched'`/old CLI flags | pull the latest repo — the corpus builder, scheduled sampling, the derived gate sweep and the new metrics are all post-2026-10. |
| `[Ckpt] ❌ these results were produced with different checkpoints` (exit 3) | you retrained and pointed the evaluation at the same `--results` file. Pass `--force` to re-synthesise, or use a different `--results` path to keep both runs. |
| Session dies mid-run | re-run the same command(s): teacher shards (content-addressed) and evaluated rows are resumed. |
| `[W&B] ⚠ push failed: ...` (e.g. `CommError`, `UsageError: api_key not configured`) | the notebook's *Internet* switch is off, or the key is missing. Add the Kaggle secret `WANDB_API_KEY` (or pass `--api-key`), or `pip install -q wandb` first if the import failed. The evaluation artifacts are unaffected — push afterwards with `!python push_to_wandb.py`. |
| W&B run shows `undefined` scalars or an empty panel | the metric is legitimately absent (e.g. MCD without saved wavs, UTMOS when both loaders failed, acceptance for A/B). `--dry-run` lists exactly which scalars exist. |
| A corpus change made the teacher pass recompute everything | expected: cached shards are validated against the sentence text (a positional cache must not train on another corpus's codes), and only changed positions are recollected — the summary line reports `N recollected after a corpus change`. |

## 7) Where the knobs live

* Test set + template corpus: `codeswitch_benchmark_data.py` (20 native Tamil / 20 romanised
  Tanglish / 20 transit code-mixed utterances; `build_training_corpus(n, seed)`).
* Corpus assembly (real text, mixing, romanisation, register balancing, size cap
  `--max-sentences`): `build_codeswitch_corpus.py`; the shared Tamil→Latin transliterator is
  `codeswitch_tts/frontend/tamil_latin.py`.
* Synthesis/timing/acceptance/zone/gate/switch-distance logic + metrics + figures + tables:
  `run_full_evaluation.py`.
* Teacher collection + distillation + scheduled sampling + checkpoints: `train_draft_distill.py`.
* W&B upload (scalars, panels, tables, media, artifact): `push_to_wandb.py`; `--wandb` on
  `run_full_evaluation.py` calls it at the end, and the notebook has a dedicated cell (§3b).
* Local test suite: `python -m pytest tests/ -q` (101 tests, no GPU and no network needed).
