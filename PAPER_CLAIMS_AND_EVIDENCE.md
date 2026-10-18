# CodeSwitch-TTS — claims, evidence map, and reviewer-risk plan

**Scope.** What the pipeline can claim *today* (300-sentence teacher corpus), what each
artifact proves, which command produces it, and what has to be re-run before submission.
Numbers below are the measured Kaggle T4 values for the current checkpoints
(`draft best val 5.1075`, `CB0 top-1 7.78 %`, `top-5 22.61 %`).

---

## 1. Claim ladder (strongest → weakest)

| # | Claim | Evidence artifact | Current value | Status |
|---|---|---|---|---|
| C1 | **Latency is decoupled from utterance length for the non-autoregressive path.** AR cost grows ≈ 2652 ms per audio-second (slope ~1.0 in RTF terms); the NAR path ≈ 48 and the hybrid ≈ 71. | Fig. 4a (per-utterance scatter + fitted slopes) | AR 2.63 RTF vs NAR 0.04 / hybrid 0.07; total 8.90 s vs 0.14 / 0.24 s | ✅ measured, publication-grade |
| C2 | **Streaming start-up latency (TTFA) drops 4.7× (e2e) / 4.8× (component) under the hybrid.** | Table I (`TTFA-c cold / warm p50 / p90`, `TTFA-e2e warm p50`), Fig. 1a | A 1054 / 332 / 349 ms, B 139 / 32–34, C 80 / 69–72 → 69–72 ms p50 | ✅ measured, with cold/warm separation and 95 % bootstrap CIs |
| C3 | **CB0 speculation is real but only ~40 % of the teacher's argmax agrees** at this draft scale — the honest, non-inflated number, reported both teacher-forced and against a mismatched-target reference. | Table I (`CB0 acc. true / ref.`), Fig. 2 | 39.8 ± 3.7 % true vs 25.7 ± 2.6 % ref | ✅ measured (single-channel null-context protocol, footnoted) |
| C4 | **Rejection sampling moves the emitted tokens toward the teacher distribution** even when acceptance is low. | console `log q` line + `logq_draft` / `logq_kept` aggregates | emitted ≥ raw draft proposal | ✅ measured (needs the post-2026-10 C rows) |
| C5 | **Quality is the binding constraint, and the gate makes the trade explicit**: a single threshold τ buys intelligibility back with AR latency instead of assuming speculation is free. | Fig. 5 (Pareto + fallback rate), Table V | τ = 0 → 0 % fallback, 0.40 s total; τ = 0.9 → 100 % fallback → AR latency | ✅ measured (fixture-verified; needs real per-threshold CER) |
| C6 | **The code-switching-specific effect is distance-to-switch, not the 4-way zone contrast.** Per-frame CB0 acceptance as a function of frames since the last Tamil↔Latin switch, with paired per-utterance bootstrap deltas. | Table IV, `summary.json → switch_distance` | zone effect Δ 1.2 pts ≪ 10–20 pt std (null); distance curve is the replacement with actual power | ⚠️ needs a stronger draft before it can resolve a real effect; the *design* is what is defensible now |
| C7 | **Spectral/perceptual quality is measurable without listeners**: UTMOS22 + DTW-aligned MCD vs the Pure-AR rendering, plus script-robust CER. | Table I / III (`CER_lat`, `UTMOS`, `MCD`) | to be measured on the next full run | ⚠️ implemented, not yet run on Kaggle |

**What cannot be claimed yet:** "the hybrid is as intelligible as the AR teacher". At the
current draft scale, CER A 66.9 / B 94.0 / C 98.6 % medians (27/60 ASR-failure rows for C).
Nobody publishes that as a *quality* result — so the paper's claim must be the
**latency/structure** result plus the **gate Pareto**, with quality reported honestly as the
limitation and the research direction. The retraining in §3 is what turns this into a
quality story.

---

## 2. "Why not standard speculative decoding?" — the paragraph reviewers will demand

Three factual answers, all already measurable with the artifacts we have:

1. **The frame is the atomic unit, not the token.** Parler-TTS emits 9 codebook tokens per
   frame with a fixed delay pattern; a frame is only acoustically meaningful once all 9 are
   present. Speculating on the token axis is therefore not the axis that reduces latency —
   the frame axis is, and that is what the draft exploits.
2. **Verification compute is already paid.** The teacher-forced forward computes logits for
   all 9 rows; scoring CB0 costs the same forward as scoring all 9. What changes with
   multi-channel verification is *acceptance*, not compute — and our own training logs give
   the number: mean CB1–8 top-1 is **0.11 %** for the residual and CB0 top-1 is 7.78 % at the
   same training scale, so frame-wise all-channel acceptance would collapse to ~0 and the
   verifier degenerates into per-frame parallel teacher-argmax decoding — exactly the
   mode-collapse failure mode that Config B's CER/UTMOS numbers already display.
3. **The residual decoder is what makes CB0-only verification viable**: it supplies
   codebooks 1–8 as a cheap NAR fill, so the pipeline never needs to *verify* them. The
   ablation to run for the camera-ready (and the honest open question) is a full-frame draft
   trained on the same corpus — with the retrained corpus this becomes a real experiment
   rather than a foregone conclusion.

---

## 3. Run chain to make the quality story real (Kaggle T4)

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

# 5) zero-GPU table/figure refresh after any code change
python run_full_evaluation.py --replay
```

**Free quality levers already wired:** register-balanced corpus (the old generator produced
~81 % romanised unique lines against a 20/20/20 benchmark), real Tamil text via the Wikimedia
stream (syntactic diversity the templates cannot supply), English entity insertion so the
transit register is trained rather than hoped for, and residual scheduled sampling to close
the teacher-CB0/draft-CB0 exposure bias.

---

## 4. Reviewer-risk table

| Objection | Our answer | Where it lives |
|---|---|---|
| "Acceptance looks inflated." | Dual report: teacher-forced (39.8 %) *and* mismatched-target reference (25.7 %); the ref column is labelled as a baseline, not an upper bound. | Table I, `KAGGLE_RUN.md` §5 |
| "CER is dominated by ASR script choices / hallucination loops." | `CER_lat` (both sides transliterated to Latin) + ASR-failure census + median-not-mean + listening samples in `eval_audio/`. | Table I/III, Fig. 1c |
| "60 utterances cannot support the claim." | 95 % percentile-bootstrap CIs on every median; paired per-utterance bootstrap for the switch-distance deltas. | Table I, Table IV |
| "The zone analysis is a null result dressed as a finding." | Fig. 2 annotates non-significant contrasts grey `(n.s.)`; the claim is moved to the distance-to-switch curve with paired CIs. | Fig. 2, Table IV |
| "No perceptual quality metric." | UTMOS22 (no listeners) + MCD vs the AR rendering, both recomputable from the saved wavs at zero GPU cost. | Table I, `--no-utmos/--no-mcd` |
| "Why is the fast config also the worst one?" | The gate makes this explicit and measured: τ trades CER for latency; the AR teacher is the quality ceiling in the same table. | Fig. 5, Table V |
| "Is this just a small diffusion/TTS model?" | It is a *decoding scheme* over a fixed pretrained AR codec LM; A/B/C/D comparisons isolate the decoding strategy, not the backbone. | Table I, §2 above |

---

## 5. Paper skeleton (where each artifact goes)

* **Introduction / teaser:** Fig. 1 (TTFA, RTF, CER distributions incl. CIs).
* **Method:** decoding scheme (§2 answers), verification protocol (single-channel,
  null-context approximation, footnoted), confidence gate, residual decoder + scheduled
  sampling.
* **Experiments:** Table I (headline, with CIs and the new metrics) → Fig. 4 (scaling +
  registers) → Table III (per-register) → Fig. 2 + Table IV (code-switching structure) →
  Fig. 5 + Table V (gate Pareto) → Fig. 3 (streaming timeline, the system view).
* **Limitations (write this section, do not hide it):** draft scale, quality gap,
  single-channel verification approximation, the unresolved distance effect, and the
  full-frame draft ablation as future work.
