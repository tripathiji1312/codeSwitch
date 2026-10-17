# TECHNICAL REPORT: Movio — IndicF5 CFM-DiT Streaming TTS for Tamil, Tanglish, and English

**Project folder:** `movio/`
**Approach number:** Approach 1
**Core model:** `ai4bharat/IndicF5` — Conditional Flow Matching + Diffusion Transformer (CFM-DiT)
**Primary author(s):** Movio TTS team
**Report version:** 1.0 — September 2026
**Purpose of this report:** Complete technical reference. Anyone reading this document — human or AI — should be able to fully understand what was built, why every decision was made, what every number means, and how to reproduce or continue the work.

---

## TABLE OF CONTENTS

1. [Problem Statement](#1-problem-statement)
2. [Why This Approach Was Chosen](#2-why-this-approach-was-chosen)
3. [High-Level Architecture](#3-high-level-architecture)
4. [Component Deep Dive: Stage A — Text Normalization](#4-component-deep-dive-stage-a--text-normalization)
5. [Component Deep Dive: Stage B — Tanglish Router](#5-component-deep-dive-stage-b--tanglish-router)
6. [Component Deep Dive: Stage C — Acoustic Synthesis (IndicF5Engine)](#6-component-deep-dive-stage-c--acoustic-synthesis-indicf5engine)
7. [Component Deep Dive: Stage D — Audio Transport](#7-component-deep-dive-stage-d--audio-transport)
8. [The IndicF5 Model — Everything About It](#8-the-indicf5-model--everything-about-it)
9. [Language Support: Tamil, Tanglish, and English](#9-language-support-tamil-tanglish-and-english)
10. [The en2ta Transliteration Engine](#10-the-en2ta-transliteration-engine)
11. [The Chunking Algorithm](#11-the-chunking-algorithm)
12. [Caching Architecture](#12-caching-architecture)
13. [Server and Concurrency Model](#13-server-and-concurrency-model)
14. [All Configuration Values and Numbers](#14-all-configuration-values-and-numbers)
15. [Performance Targets and Latency Budget](#15-performance-targets-and-latency-budget)
16. [Training: How the Tanglish LoRA Was Created](#16-training-how-the-tanglish-lora-was-created)
17. [Evaluation Framework](#17-evaluation-framework)
18. [Deployment Architecture](#18-deployment-architecture)
19. [The Complete File Map](#19-the-complete-file-map)
20. [All Dependencies](#20-all-dependencies)
21. [Known Issues, Tracked Bugs, and Future Work](#21-known-issues-tracked-bugs-and-future-work)
22. [Why Every Key Decision Was Made](#22-why-every-key-decision-was-made)
23. [Critical Warnings and Gotchas](#23-critical-warnings-and-gotchas)

---

## 1. PROBLEM STATEMENT

### The Domain
Movio is a cab/ride-hailing service that needs a real-time AI voice agent — an IVR-like system that can speak to passengers and drivers in the languages they use in everyday life in Tamil Nadu, India.

### The Language Reality
Tamil-speaking users in Tamil Nadu do not speak pure, formal Tamil or pure English. They speak a mix that the linguistic community calls **Tanglish** (Tamil + English). There are three distinct registers that the TTS system must handle seamlessly:

1. **Pure Tamil** — written in the Tamil Unicode script (e.g., "உங்கள் வாகனம் வந்துவிட்டது"). This is used in formal notifications, but most users don't type this way.

2. **Tanglish — Romanized Tamil** — Tamil words written in Latin letters (e.g., "Unga cab vandhuchu", "driver kitta pesungal"). This is how most Tamil speakers type on mobile keyboards because Tamil IME is not universal.

3. **Code-mixed Tamil+English** — native Tamil Unicode + English words mixed in the same sentence (e.g., "உங்கள் pickup location எங்கே?", "Driver enna solraru?"). This is how WhatsApp messages and app notifications look.

### The Requirements
1. **Naturalness**: The output must sound like a real human Tamil speaker, not a robotic synthesis. This is non-negotiable for user trust.
2. **Latency**: First audio must reach the user within **500ms** of the text being submitted — that is the hard p99 target for a live voice agent. Anything slower feels like a broken call.
3. **Domain accuracy**: Transport-specific entities must be pronounced correctly: vehicle registration numbers (TN45AB1234), OTP codes (4821), phone numbers, booking IDs, times (7:30 PM), distances (4.5 km), fares (₹250), etc.
4. **Speaker consistency**: All output should sound like the same voice — the product has a brand voice called "Jaya" (female, neutral Tamil accent).
5. **Scalability**: Must handle 15–20 concurrent streams from a single GPU server.

### Why This Is Hard
- **Existing open-source Tamil TTS models** (MMS-TTS, VITS, FastSpeech2) produce robotic, low-naturalness output for Tanglish. They were trained only on formal Tamil and do not handle code-mixing.
- **Commercial TTS APIs** (ElevenLabs, Murf) do not handle Tamil well and cost $0.015–$0.040 per audio-minute — which is financially unsustainable at scale.
- **Zero-shot multilingual TTS models** exist but were not designed for the specific prosody patterns of Tamil-English code-mixing.
- **Latency**: Autoregressive TTS models (like Parler-TTS) generate audio token-by-token and cannot produce the first audio chunk in under 500ms without streaming tricks that are complex to implement.

---

## 2. WHY THIS APPROACH WAS CHOSEN

### The Four Alternatives Considered

The project evaluated four TTS approaches in a "Blueprint" document (`TTS-Solutions-Blueprint.md`):

| # | Approach | Model | NFE | Sample Rate | Latency Estimate | Quality |
|---|---|---|---|---|---|---|
| 1 | **IndicF5 CFM-DiT** (this approach) | ai4bharat/IndicF5 | 12 steps | 24kHz | ~280ms/chunk | High |
| 2 | Hybrid (phrase cache + MMS fallback) | facebook/mms-tts-tam | N/A (VITS) | 16kHz | <50ms cached | Low/Med |
| 3 | VITS + FastSpeech2 cascade | samprabin/tamil_vits | N/A | 22kHz | ~50–100ms | Medium |
| 4 | Autoregressive LM (Parler-TTS) | ai4bharat/indic-parler-tts | N/A (AR) | 44.1kHz | ~3–5s (streaming) | High |

### Why IndicF5 Was Selected (Approach 1)

1. **Non-Autoregressive (NAR)**: CFM-DiT generates the full mel spectrogram in a fixed number of ODE steps, not token-by-token. This means the latency per chunk is **predictable and bounded** — 12 Euler steps on a T4 GPU takes ~250–400ms regardless of text length (within the chunk budget). Autoregressive models (Approach 4) get slower as text gets longer.

2. **Zero-shot voice cloning**: IndicF5 is a **zero-shot speaker adaptation model** — it clones the target voice from a short reference audio clip (9–12 seconds) without any fine-tuning. This means adding new voices is instant and costless. VITS/MMS require full fine-tuning per voice.

3. **Indian multilingual pre-training**: IndicF5 was pre-trained on 1,417 hours of 11 Indian languages including Tamil, which means it has strong priors for Tamil phonotactics and prosody — even before the Tanglish LoRA fine-tune.

4. **Code-mix awareness**: The training data includes IndicVoices-R, which contains English-Indic code-mixed utterances. The base model can already handle Tamil+English mixing to some degree; the Tanglish LoRA improves this.

5. **MIT license**: Commercially safe for deployment.

6. **LoRA fine-tuning path**: The F5-TTS training framework (which IndicF5 is based on) has an established LoRA fine-tuning pipeline that can be run on Kaggle free T4 GPUs in a single session.

### Why the Other Approaches Were Rejected

- **Approach 2 (MMS fallback)**: MMS-TTS Tamil has low naturalness (UTMOS ~1.8, compared to IndicF5's ~2.28–3.12 range). The phrase cache works for templated utterances but fails on dynamic content (exact fares, times, addresses).

- **Approach 3 (VITS/FS2)**: VITS is limited to a fixed 56-token Tamil vocabulary. It cannot handle Tanglish or code-mixed text without a complete frontend normalization pipeline that converts everything to pure Tamil — which degrades naturalness for English loanwords. Also 22kHz vs 24kHz sample rate gives slightly worse high-frequency audio quality.

- **Approach 4 (Parler-TTS)**: Autoregressive generation means 3–5 seconds to first audio on a T4 GPU, which is 6–10× over the 500ms target. The streaming approach was explored (movio-2 folder) but the fundamental architecture means more complex streaming logic is needed to compensate.

---

## 3. HIGH-LEVEL ARCHITECTURE

```
                         INPUT TEXT
                             │
                     ┌───────▼───────┐
                     │   Stage A     │   movio/textnorm/normalizer.py
                     │  Text Norm    │   DomainRuleEngine + WFST + NeMo
                     └───────┬───────┘
                             │ Normalized text
                     ┌───────▼───────┐
                     │   Stage B     │   movio/router/tanglish_router.py
                     │  Tanglish     │   LID → Xlit → <cs> insertion
                     │   Router      │
                     └───────┬───────┘
                             │ Script-unified text + language tags
                     ┌───────▼───────┐
                     │ en2ta.py      │   movio/router/en2ta.py
                     │ English →     │   3-tier: override dict → CMU+ARPAbet → IndicXlit
                     │ Tamil script  │
                     └───────┬───────┘
                             │ Pure Tamil Unicode text
              ┌──────────────▼──────────────┐
              │         Cache Check          │   movio/cache/audio_cache.py
              │  sha256(voice|steps|text)    │   Redis or in-memory LRU
              └──────────────┬──────────────┘
                             │ MISS ────────────────────────────────► CACHE HIT → PCM16
                     ┌───────▼───────┐
                     │   Stage C     │   movio/acoustic/indicf5_engine.py
                     │  IndicF5      │   CFM-DiT, NFE=12, Euler, sway=-1.0
                     │  Engine       │
                     └───────┬───────┘
                             │ float32 audio @ 24kHz
                     ┌───────▼───────┐
                     │   Stage D     │   movio/pipeline.py + movio/server/app.py
                     │  Transport    │   PCM16 re-slice to 2400B/50ms frames
                     └───────┬───────┘
                             │ Binary WebSocket frames
                          CLIENT
                    (Web Audio API player)
```

The pipeline is run inside `TTSPipeline` in `movio/pipeline.py`. The server in `movio/server/app.py` is a FastAPI application that exposes:

- `GET /healthz` — readiness probe
- `POST /tts` — full utterance, returns base64 WAV + timing breakdown JSON
- `POST /tts/wav` — full utterance, returns raw WAV binary
- `WS /tts/stream` — streaming endpoint, real-time PCM16 push

---

## 4. COMPONENT DEEP DIVE: STAGE A — TEXT NORMALIZATION

**File:** `movio/textnorm/normalizer.py`
**File:** `movio/textnorm/domain_rules.py`

### Purpose
Convert raw input text — which may contain numbers, codes, abbreviations, Tamil native digits, mixed scripts — into a form that IndicF5 can pronounce correctly and naturally. If the number "4821" is passed directly, IndicF5 may say "four thousand eight hundred twenty-one" instead of "four eight two one" (OTP cadence). Stage A ensures domain-specific verbalization happens first.

### Three-Layer Architecture

**Layer 1: WFST (Weighted Finite State Transducer)**
- Library: `pynini` + `indic-text-normalization`
- Language: Tamil (`"ta"`)
- Latency: <1ms (once loaded, all operations are FST composition)
- Status: Optional. If `pynini` is not installed (it requires a non-trivial C++ build), this layer is skipped. The system logs a warning and falls through to Layer 2.
- Purpose: General Tamil text normalization according to the IndicNLP specification — handles native Tamil numerals, date formats, standard abbreviations.

**Layer 2: NeMo ITN (Inverse Text Normalization)**
- Library: `nemo_text_processing`
- Language: Tamil fallback
- Latency: ~50ms on first call (model load), ~5ms subsequent calls
- Status: Optional fallback. Only triggered if Layer 1 is unavailable AND the text contains digits.
- Purpose: Same as Layer 1 but uses NeMo's grammar-based normalizer. Heavier dependency but more robust.
- Trigger condition: `if not wfst_available and contains_digits(text): try nemo_normalize(text)`

**Layer 3: DomainRuleEngine (ALWAYS RUNS)**
- File: `movio/textnorm/domain_rules.py`
- Latency: <1ms (pure regex, deterministic)
- Status: Always runs, regardless of Layer 1/2 availability. Even if Layer 1 runs successfully, Layer 3 runs after it.
- Purpose: Transport-domain-specific verbalization that general normalizers don't know about.

### DomainRuleEngine Rules (all 12 rule categories)

All rules are applied in the order listed. Each rule returns the modified text and a flag indicating whether it fired.

| # | Rule | Input Example | Output Example | Why |
|---|---|---|---|---|
| 1 | Tamil native digits | ௪௮௨௧ | 4821 | IndicF5 was trained on ASCII digits, not Tamil digit codepoints U+0BE6–U+0BEF |
| 2 | Ordinals | "1st pickup", "3rd lane" | "first pickup", "third lane" | IndicF5 handles spelled ordinals better |
| 3 | Booking IDs | TN45AB1234 | "T N four five A B one two three four" | Letter-by-letter + digit-by-digit for booking references |
| 4 | Vehicle registration plates | TN82CS1312 | "T N eight two C S one three one two" | Indian vehicle plate format: state code + district + series + number |
| 5 | Phone numbers | 9876543210 (10 digits) | "nine eight seven six five four three two one zero" | Digit-by-digit for phone numbers |
| 6 | Time | "7:30 PM" | "seven thirty PM" | Natural spoken time |
| 7 | Dates | "25/08/2026" | "August twenty-five, twenty twenty-six" | Tamil-region date format DD/MM/YYYY |
| 8 | Distances | "4.5 km" | "four point five kilometers" | Unit expansion |
| 9 | Currency | "₹250", "Rs.250" | "rupees two hundred fifty" / "ரூபாய் இருநூற்று ஐம்பது" | Tamil or English depending on script context |
| 10 | OTP/PIN codes | "4821" (standalone 4-digit) | "four eight two one" | Digit-by-digit for OTP cadence |
| 11 | Abbreviations | "St.", "Rd.", "Nr.", "opp." | "Street", "Road", "Near", "opposite" | From gazetteer `config/gazetteer/proper_nouns.txt` |
| 12 | Gazetteer proper nouns | "T Nagar", "Chennai Central" | kept as-is | Prevent re-verbalization of known place names |

### Tamil Number Verbalization

The domain rule engine includes a `tamil_number()` function that converts integers to Tamil words:

- 0–99 fully hardcoded as a lookup table (e.g., 21 → "இருபத்தொன்று")
- Hundreds: "X நூறு" (e.g., 300 → "முந்நூறு")
- Thousands: "X ஆயிரம்" (e.g., 5000 → "ஐந்தாயிரம்")
- Up to 99,999

An `english_number()` function mirrors this for English output (one, two, ..., one thousand, etc.).

### Output
`NormalizationResult` dataclass:
- `text: str` — normalized text
- `backend_used: str` — "wfst" | "nemo" | "domain_rules" | "domain_rules_only"
- `latency_ms: float` — wall-clock time of normalization

### Stage Timeout
50ms hard timeout enforced by `asyncio.wait_for()` in the pipeline. If NeMo cold-loads during this window (takes ~50ms), the timeout may fire. The pipeline catches `asyncio.TimeoutError` and proceeds with whatever normalization completed. In practice, warmup runs NeMo before the first request.

---

## 5. COMPONENT DEEP DIVE: STAGE B — TANGLISH ROUTER

**Files:** `movio/router/tanglish_router.py`, `movio/router/lid.py`, `movio/router/xlit.py`

### Purpose
Identify the language of each token in the input, convert romanized Tamil (Latin-script Tamil) to Tamil Unicode, insert code-switch boundary markers, and output a single script-unified string for Stage C.

### Language Identification (LID)

**Primary: HeuristicLID** (`movio/router/lid.py`)

A rule-based classifier that labels each token:

| Class | Detection Method |
|---|---|
| `ta_native` | Contains Unicode in range U+0B80–U+0BFF (Tamil block) |
| `ta_roman` | Matches lexicon of ~40 common Tanglish words (unga, iruku, vanakkam, enna, sollu, etc.) OR has Tamil vowel suffixes (-inga, -adu, -la, -tu) |
| `en` | All ASCII letters, not in Tanglish lexicon |
| `proper_noun` | Found in the gazetteer (`config/gazetteer/proper_nouns.txt`) |
| `hi_native` | Contains Devanagari (U+0900–U+097F) |
| `other` | Everything else (numbers, punctuation) |

**Optional: IndicLIDBackend** (`movio/router/lid.py`)

If `ai4bharat/IndicLID` and `fasttext` are available, this 47-class classifier is used instead. It classifies text into:
- 24 native-script Indian languages
- 21 romanized Indian language variants
- English
- Others

The IndicLID backend provides better accuracy for ambiguous Tanglish tokens (e.g., "man" could be English or Tamil romanization of "மன்") but has a ~15ms startup overhead. Falls back to HeuristicLID if unavailable.

**Configuration:** `stage_b.router.lid_backend: "auto"` — uses IndicLID if available, heuristic otherwise.

### Transliteration of Romanized Tamil

**IndicXlit** (`movio/router/xlit.py`):
- Backend: CTranslate2-quantized `Singla0009/all-indic-transliteration`
- Latency: ~1ms per word
- Beam size: 4
- Language tag: `__ta__` (indicates target script = Tamil)
- Status: Set to `xlit_backend: "none"` in current config — bypassed in favor of `en2ta.py`

**en2ta.py** (see Section 10) is the active transliteration path. It was built to handle English words more accurately than IndicXlit for transport domain vocabulary.

### `<cs>` Code-Switch Token Insertion

At every point where language transitions occur (Tamil-native → English, or English → Tamil-native), the router inserts a `<cs>` token:

```
Input:  "உங்கள் pickup location எங்கே?"
Tokens: [ta_native:"உங்கள்"] [en:"pickup"] [en:"location"] [ta_native:"எங்கே"]
After:  "உங்கள் <cs> pickup location <cs> எங்கே?"
```

**Why**: Research by Thomas (2018) on code-switching in Indic TTS showed that explicit boundary marking produces more natural prosodic transitions at code-switch points compared to naive concatenation. The `<cs>` token acts as a prosodic hint to the acoustic model (after it is stripped, the boundary position is preserved in spacing and context).

**Stripping**: The `<cs>` token is stripped before Stage C. The pipeline does:
```python
synth_text = route.normalized_text.replace("<cs>", " ").strip()
# Collapse multiple spaces
synth_text = re.sub(r'\s+', ' ', synth_text).strip()
```

### Proper Noun Gazetteer

File: `config/gazetteer/proper_nouns.txt`
Contents: 37 entries — Chennai district names, local landmarks, ride-hailing app names.
Examples: "Chennai Central", "T Nagar", "Anna Nagar", "Uber", "Ola", "Airport", "CMBT"

The router checks multi-token windows (up to 3 tokens) against the gazetteer. Matches are labeled `proper_noun` and bypass transliteration — they are passed as Latin text to Stage C, which routes them through the English phoneme path in IndicF5.

**Why proper nouns are kept in Latin**: Place names like "T Nagar" transliterated to Tamil ("டி நகர்") sound different from how real speakers say them. Keeping them in Latin script allows IndicF5 to apply its English pronunciation rules and produce the correct accent.

### Stage Timeout
50ms hard timeout. Same logic as Stage A.

### Output
`RouterResult` dataclass:
- `normalized_text: str` — text with `<cs>` tokens at boundaries
- `token_labels: List[Tuple[str, str]]` — (token, class) pairs
- `cs_boundaries: List[int]` — character positions of transitions

---

## 6. COMPONENT DEEP DIVE: STAGE C — ACOUSTIC SYNTHESIS (IndicF5Engine)

**File:** `movio/acoustic/indicf5_engine.py`

### Purpose
Convert the script-unified Tamil Unicode text into speech audio at 24kHz. This is the core synthesis stage and accounts for ~250–400ms of the total TTFA budget.

### Model Loading Sequence

1. `_ensure_indicf5_path()` — inserts `third_party/IndicF5/` at the front of `sys.path` and purges any existing `f5_tts` modules from `sys.modules`. This is essential because the pip-installed `f5-tts==1.1.22` package has different architecture parameters and would load the wrong model class if it shadows the vendored code.

2. `hf_hub_download("ai4bharat/IndicF5", "model.safetensors")` — downloads the 337M-parameter CFM-DiT checkpoint. Requires `HF_TOKEN` environment variable (model is gated on HuggingFace Hub, requires accepting license terms).

3. `hf_hub_download("ai4bharat/IndicF5", "checkpoints/vocab.txt")` — downloads the character vocabulary (~2,545 characters covering Tamil Unicode + multilingual script characters).

4. `load_model(DiT, _MODEL_CFG, ckpt_path, mel_spec_type="vocos", ode_method="euler")` — loads the CFM-DiT model with the specific architecture configuration:
   ```python
   _MODEL_CFG = dict(
       dim=1024,       # Model dimension
       depth=22,       # Transformer depth (22 layers)
       heads=16,       # Attention heads
       ff_mult=2,      # Feed-forward multiplier
       text_dim=512,   # Text encoder dimension
       conv_layers=4   # ConvNeXt-v2 text encoder layers
   )
   ```

5. `.eval()` + `torch.compile(mode="reduce-overhead")` — JIT compilation for CUDA inference. `reduce-overhead` mode (vs `max-autotune`) is chosen because it has faster compile time and avoids the 2–3 minute `max-autotune` warmup penalty.

6. `load_vocoder(vocoder_name="vocos", is_local=False, device)` — loads the Vocos neural vocoder from `charactr/vocos-mel-24khz`. This converts the mel spectrogram produced by the CFM-DiT into a waveform.

### The CFM-DiT Inference Process

**Conditional Flow Matching (CFM)** is a generative model technique where:
- The model learns to transform a standard Gaussian noise distribution into the target mel spectrogram distribution
- This transformation is done as an ODE (ordinary differential equation) integration
- The "flow" is conditioned on: (1) the text sequence, and (2) a reference audio clip (for speaker identity)

**How inference works:**
1. Start with random Gaussian noise `z_0 ~ N(0, I)` of the target length
2. Estimate target length from text length and reference audio duration ratio
3. Integrate the ODE from `t=0` to `t=1` using NFE (Number of Function Evaluations) = 12 Euler steps
4. Each Euler step: `z_{t+dt} = z_t + dt * velocity_prediction(z_t, text, ref_audio)`
5. The velocity field is predicted by the DiT (Diffusion Transformer) backbone
6. After integration, `z_1` is the mel spectrogram
7. Vocos converts mel → waveform

**NFE = 12 (Euler) vs Alternatives:**

| Method | NFE | Quality | Latency @ T4 |
|---|---|---|---|
| 8 Euler steps | 8 | Slightly lower (WER ~4.16%) | ~150–250ms |
| **12 Euler steps** | **12** | **Good (WER ~2.67%)** | **~250–400ms** |
| 16 Euler steps | 16 | Near-best (WER ~2.53%) | ~350–550ms |
| 24 midpoint steps | 24 | Best (WER ~2.42%) | ~700–1000ms |
| 32 Euler steps | 32 | Reference (WER ~2.42%) | ~800–1200ms |

12 Euler steps is the chosen default because:
- WER 2.67% is only 0.25% absolute worse than the best quality at 32 steps
- Latency is ~2.67× faster than 32 steps
- Latency is ~4× faster than the original 24 midpoint steps (midpoint requires 2 function evals per step)
- 8 steps begins to show audible quality degradation (WER jumps to 4.16%)
- 16 steps is a safe fallback if 12 steps fail quality smoke tests

**Sway Sampling (coefficient = -1.0):**

Sway sampling is a technique from the EPSS paper (Efficient Progressive Speech Synthesis) that redistributes the ODE time steps non-uniformly. With sway coefficient = -1.0 (left-sway), more steps are allocated to the early (high-noise) part of the denoising trajectory. This is because most of the semantic content (pitch contour, phoneme identity) is established in the early steps. Effect: at low NFE (12 steps), sway sampling significantly improves WER and speaker similarity compared to uniform step spacing.

**CFG Strength = 1.0:**

Classifier-Free Guidance (CFG) is a technique where the model runs twice — once conditioned on the prompt, once unconditioned — and blends the two: `output = uncond + cfg_strength * (cond - uncond)`. CFG strength 1.0 means the guidance scale is effectively disabled (produces: `output = uncond + 1.0 * (cond - uncond) = cond`), which is equivalent to using just the conditioned output. This was chosen to avoid the additional 2× compute cost of CFG while maintaining quality — at 12 NFE, quality is limited by step count more than guidance strength.

**Speed = 0.95:**

The speed parameter scales the estimated synthesis duration. Speed 0.95 means 5% slower than the model's natural speech rate. This was tuned empirically — the base IndicF5 model tends to speak slightly fast for transport domain announcements (OTPs, addresses), and 0.95 produces more natural pacing.

**ODE method = Euler:**

Euler (1 function evaluation per step) was chosen over midpoint (2 evals/step) because at NFE=12, Euler and midpoint produce nearly identical quality but midpoint costs 2× more compute. The F5-TTS paper validates this choice: at 16 NFE, Euler achieves WER 2.53% (paper's benchmark number), which is the reference for quality comparison.

### Zero-Shot Voice Cloning

The engine does not have speaker embeddings or a speaker ID lookup table. Instead, it conditions on raw reference audio:

```python
ref_audio = load_audio("config/voices/ta_female_neutral/ref.wav")  # 9.3s
ref_text = "..."  # transcript of the reference audio
audio_output = infer_batch_process(
    ref_audio, ref_text,          # Speaker conditioning
    [synth_text],                  # Text to synthesize
    model, vocoder,
    nfe_step=12, cfg_strength=1.0,
    sway_sampling_coef=-1.0, speed=0.95,
    cross_fade_duration=0, device=device
)
```

Two voice profiles are provided:
- `ta_female_neutral` — 9.3-second reference audio, female Tamil speaker
- `ta_male_neutral` — 12.4-second reference audio, male Tamil speaker

The reference audio tensor and spectrogram are cached in memory after first use (`_ref_tensor_cache`, `_ref_audio_cache` dictionaries keyed by voice name) to avoid re-loading from disk on every chunk.

### Streaming Synthesis

The engine implements `synthesize_stream()` as a Python generator. This is the core of the real-time streaming:

```python
def synthesize_stream(self, text, voice_name, min_syl=10, max_syl=28, speed=0.95):
    chunks = chunk_text(text, min_syl=min_syl, max_syl=max_syl)
    for chunk in chunks:
        audio = self.synthesize_chunk(chunk, voice_name, flow_steps=12, speed=speed)
        audio = trim_silence(audio, threshold_db=-38.0, min_silence_ms=15.0, trail_silence_ms=60.0)
        audio = apply_fade(audio, fade_ms=5.0, sr=24000)
        yield audio
        # Yield silence between chunks
        if is_sentence_end(chunk):
            yield silence(350, sr=24000)  # 350ms after sentences
        else:
            yield silence(200, sr=24000)  # 200ms after clauses
```

The streaming path (WebSocket `/tts/stream`) uses NFE=12 with `min_syl=18, max_syl=40` (larger chunks than the default to reduce the per-chunk overhead and produce more natural prosody across longer phrases).

### Audio Post-Processing per Chunk

1. **Silence trim**: `librosa.effects.split(top_db=38)` — removes leading and trailing silence below -38dBFS. Minimum leading silence to remove: 15ms. Trailing silence to remove: 60ms. This prevents chunks starting/ending with dead silence that makes speech sound clipped.

2. **Fade**: 5ms linear fade-in and fade-out applied to each chunk. This prevents click artifacts at chunk boundaries when consecutive chunks are played back-to-back.

### Inter-Chunk Silence

Between chunks, the pipeline inserts silence based on the chunking boundary type:
- After a clause (comma, semicolon): **200ms** of silence
- After a sentence (period, question mark, exclamation, Devanagari danda): **350ms** of silence

These values were chosen by listening tests. 200ms is enough for a breath at a clause boundary; 350ms gives a natural sentence pause that sounds like a real speaker.

---

## 7. COMPONENT DEEP DIVE: STAGE D — AUDIO TRANSPORT

**Files:** `movio/pipeline.py`, `movio/server/app.py`, `movio/utils/audio.py`

### PCM16 Encoding

All audio is transmitted as **16-bit signed integer PCM** (not WAV, not MP3, not Opus — raw PCM16):

```python
def float_to_pcm16(audio: np.ndarray) -> bytes:
    audio = np.clip(audio, -1.0, 1.0)   # Prevent overflow
    return (audio * 32767).astype(np.int16).tobytes()  # Little-endian
```

Why PCM16 instead of WAV or compressed formats:
- **WAV** would require a header on each frame, adding overhead
- **Opus/MP3** compression has a ~20ms algorithmic latency and requires a codec on the client
- **PCM16** is zero-latency, trivial to decode in the browser via `Int16Array`

Why `× 32767` instead of `× 32768`: The signed int16 range is [-32768, +32767]. Using 32767 ensures the positive half doesn't clip, which matters for audio quality on positive peaks.

### Frame Slicing

The 50ms frame size is enforced at transport:

```python
ws_chunk_ms = 50        # Configuration value
samples_per_frame = 24000 * 50 // 1000 = 1200 samples
bytes_per_frame = 1200 * 2 = 2400 bytes   # 2 bytes per int16 sample
```

The pipeline re-slices synthesized audio (which may be any length) into 2400-byte frames:
```python
for i in range(0, len(pcm16_bytes), 2400):
    frame = pcm16_bytes[i : i + 2400]
    await ws.send_bytes(frame)
```

Why 50ms frames:
- Smaller frames (e.g., 20ms) require more WebSocket messages, increasing TCP overhead
- Larger frames (e.g., 100ms) increase buffering latency on the client
- 50ms is the standard streaming audio chunk size used by most real-time audio systems (matches WebRTC's preferred frame size for 24kHz audio)

### TTFA (Time to First Audio) Measurement

The pipeline logs TTFA (time from request receipt to first audio byte sent):

```python
request_start = time.perf_counter()
# ... Stages A, B, en2ta ...
# ... First synthesize_chunk() call ...
first_audio_yielded = time.perf_counter()
ttfa_ms = (first_audio_yielded - request_start) * 1000
logger.info("stream TTFA %.1f ms", ttfa_ms)
```

### WebSocket Protocol

The server sends both JSON control frames and binary data frames over the same WebSocket connection.

**Outgoing JSON control frames:**
```json
{"type": "start"}
{"type": "progress", "phase": "AB", "message": "Stage A/B: normalizing + routing..."}
{"type": "progress", "phase": "C", "message": "Stage C: synthesizing chunk 1/3..."}
{"type": "end", "chunks": 3}
```

**Outgoing binary frames:** Raw PCM16 bytes, 2400 bytes per frame (50ms).

**Incoming JSON from client:**
```json
{"text": "Unga OTP enna?", "voice": "ta_female_neutral", "speed": 0.95}
```

---

## 8. THE INDICF5 MODEL — EVERYTHING ABOUT IT

**Source:** `ai4bharat/IndicF5` on HuggingFace Hub (gated — requires accepting terms)
**Local path:** `third_party/IndicF5/` (vendored fork)

### Architecture: Conditional Flow Matching + Diffusion Transformer

**Full model name:** CFM-DiT (Conditional Flow Matching with Diffusion Transformer backbone)

**Architecture parameters:**
```python
dim = 1024          # Internal model dimension (embedding + hidden size throughout)
depth = 22          # Number of transformer layers (Attention + FeedForward blocks)
heads = 16          # Multi-head attention heads per layer (head_dim = 1024/16 = 64)
ff_mult = 2         # Feed-forward expansion factor (FF hidden = 1024 × 2 = 2048)
text_dim = 512      # Text encoder output dimension (ConvNeXt-v2 output)
conv_layers = 4     # Number of ConvNeXt-v2 layers in the text encoder
```

**Parameter count:** ~337 million

**Sample rate:** 24,000 Hz (24kHz)

**Mel spectrogram configuration:**
```
n_fft = 1024           # FFT size
hop_length = 256       # Hop between frames (256/24000 = 10.67ms resolution)
win_length = 1024      # Window size
n_mel_channels = 100   # Number of mel filterbank channels
mel_spec_type = "vocos"  # CRITICAL: must match vocoder
```

**Why `mel_spec_type = "vocos"` is critical:** IndicF5 was trained with the `vocos` mel spectrogram variant, which uses 100 mel channels. The older F5-TTS default uses 80 mel channels. If you load the IndicF5 weights but use a vocoder that expects 80 channels, the vocoder will produce completely garbled, robotic audio because the channel count mismatch means every frequency band is mapped to the wrong filter. This was a common source of bugs in early versions of the integration.

**Text encoding:**
- Tokenization: Character-level (each character in the vocabulary is a token)
- Vocabulary size: ~2,545 characters
- Covers: All Tamil Unicode characters (U+0B80–U+0BFF), Devanagari, Bengali, Telugu, Kannada, Malayalam, Odia, Gujarati, Punjabi, Assamese, English (A-Z, a-z), digits (0-9), punctuation

**Training data summary:**

| Dataset | Size | Languages | Notes |
|---|---|---|---|
| Rasa | Large | Tamil, multiple | Conversational Tamil audio |
| IndicTTS | Medium | 11 Indian languages | IIT Madras TTS dataset |
| LIMMITS | Medium | 13 Indian languages | Microsoft Research India |
| IndicVoices-R | Large | 22 Indian languages | Crowdsourced, UTMOS ~2.1 |
| **Total** | **1,417 hours** | **11 languages** | |

**Languages in IndicF5 vocabulary:** Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu

**License:** MIT (model weights AND project code)

### The DiT Backbone (Diffusion Transformer)

The DiT (Diffusion Transformer) is based on the architecture from the "Scalable Diffusion Models with Transformers" paper, adapted for audio. Each layer contains:

1. **Self-attention over mel frames** — allows the model to attend over all positions in the output spectrogram simultaneously (unlike autoregressive models that can only attend to past positions)
2. **Cross-attention over text tokens** — conditions each mel frame on the relevant text characters (alignment is learned, not forced)
3. **Feed-forward network** — expansion ratio 2× (2048 hidden dims)
4. **AdaLN conditioning** — the current flow time `t ∈ [0,1]` is used to adaptively shift and scale the LayerNorm parameters, allowing the model's behavior to vary smoothly across the denoising trajectory

### The ConvNeXt-v2 Text Encoder

The text is processed by a ConvNeXt-v2 text encoder (4 layers, output dim 512) before being passed to the DiT's cross-attention. This encoder:
- Takes character embeddings as input
- Produces context-aware character representations that capture local phonological context
- Uses depthwise convolutions instead of self-attention, which is faster and better for local context in TTS

### The Vocos Vocoder

**Model:** `charactr/vocos-mel-24khz`
**Architecture:** Single-pass Fourier spectral coefficient prediction

Unlike traditional vocoders (HiFi-GAN, WaveGlow) that iteratively up-sample the mel spectrogram, Vocos directly predicts the STFT magnitude and phase from the mel spectrogram in a single forward pass:

1. Backbone: ConvNeXt blocks operating on mel spectrogram
2. Output: ISTFT magnitude + phase coefficients
3. Apply ISTFT to reconstruct waveform
4. No iterative refinement — single forward pass → waveform

Why Vocos over HiFi-GAN: Vocos is faster (single pass vs iterative), produces comparable quality for 24kHz audio, and was specifically designed to work with the `vocos` mel spec variant that IndicF5 uses.

---

## 9. LANGUAGE SUPPORT: TAMIL, TANGLISH, AND ENGLISH

### Tamil (Pure)

**Input format:** Tamil Unicode script (U+0B80–U+0BFF)

**Example:** "உங்கள் வாகனம் வந்துவிட்டது. நன்றி."

**Normalization applied:**
- Tamil native digits (U+0BE6–U+0BEF: ௦௧௨...) converted to ASCII (0123...) by the `TA_DIGITS` translation table
- Currency symbol ₹ → "ரூபாய்" (in Tamil context) or "rupees" (in English context)
- Tamil months: ஜனவரி (January) through டிசம்பர் (December)
- Tamil ordinals: முதல் (first) through முப்பத்தொன்றாவது (thirty-first)
- Tamil time periods: காலை (morning), மதியம் (afternoon), மாலை (evening), இரவு (night)

**LID classification:** `ta_native`

**Routing:** Passed directly to IndicF5 without transliteration. IndicF5 has strong Tamil synthesis priors from 1,417 hours of training data.

### Tanglish — Romanized Tamil

**Input format:** Latin letters representing Tamil sounds

**Examples:**
- "Unga cab vandhuchu" = "உங்கள் cab வந்துச்சு"
- "driver kitta pesungal" = "driver கிட்ட பேசுங்கள்"
- "enna nadakirathu?" = "என்ன நடக்கிறது?"

**LID classification:** `ta_roman`

**Routing:**
1. HeuristicLID classifies as `ta_roman`
2. IndicXlit or en2ta converts to Tamil Unicode
3. Result passed to IndicF5 as native Tamil text

**Common Tanglish vocabulary in heuristic lexicon (40 words):**
unga, inga, enna, avan, aval, iruku, vandhu, sollu, soli, paru, po, va, adi, mela, kila, kitta, kaaga, maatom, thaan, ellam, nallannu, ippo, appuram, enga, epdi, eppo, anupa, varuvaro, solluvaro, vanakkam, nandri, sari, achcha, ayyo, yenna, illa, aamaa, illai, konjam

### Code-Mixed Tamil+English

**Input format:** Tamil Unicode + English words in same sentence

**Examples:**
- "உங்கள் pickup location எங்கே?"
- "Driver enna solraru?"
- "OTP 4821 sollunga"

**LID classification:** Mixed — `ta_native` + `en` tokens

**Routing:**
1. Token-level LID identifies each word's language
2. `<cs>` tokens inserted at boundaries
3. English words passed to `en2ta.py` for transliteration to Tamil script
4. Merged text passed to IndicF5 as unified Tamil Unicode

### English (Transport Domain)

**Input format:** Pure English text

**Examples:**
- "Your cab is arriving in 5 minutes"
- "Please confirm your pickup location"
- "Your OTP is 4821"

**LID classification:** `en`

**Routing:**
1. Text normalization handles numbers and entities
2. en2ta.py converts all English words to Tamil phonetic script
3. Proper nouns (from gazetteer: "Chennai Central", "Airport") kept in Latin
4. IndicF5 synthesizes the Tamil-script phonetic rendering

**Why English words are transliterated rather than passed directly:**
IndicF5 was trained on Indian languages. While it can handle some English words (especially common loanwords), it was not trained on extensive English text. For reliable pronunciation of English words in a Tamil accent, converting to Tamil script first produces more consistent output. For example, "pickup" → "பிக்கப்" → IndicF5 pronounces with Tamil phonology, which is what a Tamil speaker would naturally say.

---

## 10. THE en2ta TRANSLITERATION ENGINE

**File:** `movio/router/en2ta.py` (656 lines)

This is a completely local, zero-network-call English → Tamil phonetic script converter built specifically for the transport domain.

### Why a Custom Transliteration Engine?

IndicXlit (the standard transliteration tool) works well for romanized Tamil (Tamil words written in Latin). But for actual English words, it often produces incorrect Tamil representations because it is trained to reverse romanized Indian languages, not to phonetically adapt English loanwords.

The `en2ta.py` engine was purpose-built for English → Tamil phonetic adaptation, handling:
- Transport domain vocabulary (pickup, drop, fare, OTP, booking, etc.)
- Number words (four → ஃபோர், not "நான்கு" which is the Tamil word for four)
- Common English loanwords in Tamil speech

### The 3-Tier Architecture

**Tier 1: Override Dictionary (~50 words)**

A hardcoded dictionary of the most common English words in the transport domain with their exact "correct" Tamil phonetic spellings. Examples:

```python
OVERRIDE_DICT = {
    "pickup": "பிக்கப்",
    "drop": "டிராப்",
    "otp": "ஓ டி பி",
    "ola": "ஓலா",
    "uber": "ஊபர்",
    "fare": "ஃபேர்",
    "zero": "ஸீரோ",
    "one": "வன்",
    "two": "டூ",
    "three": "த்ரீ",
    "four": "ஃபோர்",
    "five": "ஃபைவ்",
    "six": "சிக்ஸ்",
    "seven": "செவன்",
    "eight": "எயிட்",
    "nine": "நைன்",
    "booking": "புக்கிங்",
    "cancel": "கேன்சல்",
    ...
}
```

**Why separate number words (four → ஃபோர்) from Tamil number words (நான்கு)?**
In Tanglish, when reading digit sequences like OTPs, people say the English digit words ("four eight two one") not the Tamil words ("நான்கு எட்டு இரண்டு ஒன்று"). The override dictionary captures this convention.

**Tier 2: CMU Pronunciation Dictionary + ARPAbet → Tamil**

For words not in the override dictionary, the engine uses the CMU Pronouncing Dictionary (123,000 English words with ARPAbet phoneme sequences) and converts the ARPAbet phoneme codes to Tamil script phoneme by phoneme.

**ARPAbet consonant mapping:**
```
B   → ப       CH  → ச       D   → ட       DH  → த
F   → ஃப      G   → க       HH  → ஹ       JH  → ஜ
K   → க       L   → ல       M   → ம       N   → ன/ந
NG  → ங       P   → ப       R   → ர       S   → ஸ
SH  → ஷ       T   → ட       TH  → த       V   → வ
W   → வ       Y   → ய       Z   → ஸ       ZH  → ஷ
```

**Special N rule:** N → ந (nasal) at word-initial or before vowels; N → ன (final/retroflex nasal) at word-end. This distinction matters for naturalness — Tamil has two /n/ sounds with different places of articulation.

**ARPAbet vowel mapping (initial form / dependent form):**
```
AA (father) → ஆ / ா      AE (cat)   → ஆ / ா
AH (about)  → அ / ""     AO (law)   → ஆ / ா
AW (how)    → அவ் / வ்   AY (buy)   → ஐ / ை
EH (bed)    → எ / ெ      ER (bird)  → அர் / ர்
EY (day)    → ஏ / ே      IH (bit)   → இ / ி
IY (beet)   → ஈ / ீ      OW (boat)  → ஓ / ோ
OY (boy)    → ஓய் / ோய்  UH (book)  → உ / ு
UW (boot)   → ஊ / ூ
```

Initial form is used when the vowel is word-initial or follows a vowel-less consonant. Dependent form is used after a consonant (as a diacritic).

**CMU Dict is loaded via NLTK:**
```python
from nltk.corpus import cmudict
_cmu = cmudict.dict()  # Loads on first call, cached after
```

**Tier 3: IndicXlit CTranslate2 (fallback for OOV)**

For words not found in the override dictionary or CMU Dict (new words, proper names, slang), the IndicXlit CTranslate2 model is used as a final fallback. This is the same model used in the router's xlit.py but invoked only for individual words, not sentences:
- Model: `Singla0009/all-indic-transliteration` (CTranslate2 quantized)
- Language tag: `__ta__`
- Beam size: 4
- Local paths tried first: `models/indicxlit_ct2/`, `downloads/indicxlit_ct2_fp32/`

### The `_format_natural_digits()` Function

A unique feature for OTP/PIN/code digit reading. Instead of reading all digits as a continuous stream, it groups them into pairs:

```
Input:  "four eight two one"
Output: "ஃபோர் எயிட், த்ரீ டூ"  (paired, with comma between pairs)
```

**Why pairs with commas?**
- Real humans reading OTPs pause naturally after every 2–3 digits
- The comma gives IndicF5 a light breath pause without causing the chunker to split mid-sequence (periods would cause a hard split)
- Digit pairs "four eight, two one" are easier to remember and repeat than "four eight two one" as a stream

### The `_break_clusters()` Function

Tamil is a phonological system with strict consonant cluster rules. Foreign words often contain consonant clusters (like "str", "tr", "br") that do not exist in Tamil. The engine has a `_break_clusters()` function that inserts helping vowels to make the clusters pronounceable in Tamil phonology:

```
"program" ARPAbet: P R OW G R AE M
After cluster break: ப்ர → பிர (insert ி between the stop ப் and liquid ர்)
→ "புரோகிரஆம்"
```

**`_VALID_CLUSTERS` whitelist:**
Tamil allows certain consonant clusters natively:
- Geminates: க்க, ட்ட, ப்ப, ன்ன, ல்ல, ர்ர, ம்ம, ண்ண, ள்ள, த்த, ச்ச
- Nasal + stop: ம்ப, ந்த, ன்ற, ண்ட, ங்க
- Liquid + consonant: ர்க, ல்வ, etc.

Any cluster not in this whitelist is broken by inserting ி (dependent form of "i" vowel) between the consonants.

### The Single-ஃ Rule

The ஃ (Aytham) character in Tamil is used to represent the aspirated /f/ sound borrowed from English (ஃப = "pha" ≈ "fa"). However, repeating ஃ twice in the same word produces distorted audio in IndicF5 — the model was not exposed to double-ஃ in training.

The rule: **Only the first occurrence of ஃ per word is retained.** Subsequent occurrences are replaced with the unaspirated version:
```
"ஃபோன் ஃபேர்" → keep both (separate words, each has one ஃ — OK)
"ஃபிஃபி" → "ஃபிபி" (internal second ஃ removed — prevents garbling)
```

### Text Preprocessing Steps (before tier lookup)

1. Expand ordinals: "1st" → "first", "22nd" → "twenty-second"
2. Expand contractions: "don't" → "do not", "I'm" → "I am", etc.
3. Expand symbols: "₹" → "ரூபாய்", "@" → "at", "%" → "percent"
4. Format vehicle plates: "TN45AB1234" → spell_alnum_code format
5. Format times: "7:30 am" → "seven thirty AM"
6. Digit sequences → Tamil loanword digit words (uses override dict number words)

---

## 11. THE CHUNKING ALGORITHM

**File:** `movio/acoustic/chunking.py`

### Why Chunking Is Necessary

IndicF5 generates the full mel spectrogram of the entire input text in a single inference run. For long texts (e.g., a 15-word sentence), this means:
1. The ODE must integrate over a longer duration mel spectrogram (more samples)
2. The self-attention in the DiT scales quadratically with mel spectrogram length
3. Memory usage grows with text length
4. Latency to first audio equals latency to synthesize the entire sentence

Chunking solves this by splitting the input into shorter segments that can be synthesized independently and streamed out one at a time. The user hears the first chunk in ~300ms while the subsequent chunks are still being synthesized.

### Why Syllable-Budget (not Character-Budget)

A fixed character budget (e.g., 135 characters as in upstream F5-TTS) would create wildly different-length chunks depending on the language:
- Tamil: syllables are typically 2-3 characters each (அ = 1 char, ஆன் = 3 chars)
- English: words in Tamil transliteration can be much longer (pickup = பிக்கப் = 6 chars)

A syllable budget ensures chunks take roughly the same acoustic duration to synthesize, giving predictable latency per chunk.

### Syllable Counting

**Tamil:**
- Count independent vowels: matches `[அஆஇஈஉஊஎஏஐஒஓஔ]` (14 vowel characters)
- Count vowel signs on non-vowel syllables: matches vowel diacritic codepoints U+0BBE–U+0BD7 on a consonant base
- Each vowel or consonant+vowel-sign = 1 syllable
- Consonant clusters (pulli ் followed by consonant) = 0 additional syllables

**English (in transliterated form):**
- Count vowel letter groups: `[aeiouy]+` per group = 1 syllable
- "pickup" → 2 vowel groups (i, u) = 2 syllables
- "OTP" → 1 vowel group (O) + T + P treated as 1 syllable each if solo consonants

### Three-Phase Chunking

**Phase 1: Clause split**
```python
re.split(r"(?<=[,;.!?।])\s+", text)
```
Split at punctuation boundaries. This preserves natural prosodic units. A single comma-separated clause becomes one chunk if it fits in the syllable budget.

**Phase 2: Conjunction split (if still too long)**
If any clause exceeds `max_syl`, it is further split at Tamil/English conjunctions:
- English: "and", "but", "or", "because", "while", "after", "before"
- Tamil: "மற்றும்" (and), "ஆனால்" (but), "எனவே" (therefore)

**Phase 3: Merge pass**
After splitting, small fragments are merged back:
- Fragments shorter than `max(3, min_syl // 2)` syllables are merged into the previous non-digit chunk
- Digit segments are paired greedily: consecutive digit-word segments are merged until `max_syl` is reached

**Digit segment detection:**
A chunk is classified as a "digit segment" if ≥50% of its words are digit words (zero, one, two, ..., nine, ten, ...). This allows OTP codes to be read as a single group with natural pairing.

### Configuration Values

| Parameter | Value | Used In |
|---|---|---|
| `min_syl` | 10 | Engine default (`synthesize_stream()`) |
| `max_syl` | 28 | Engine default |
| `min_syl` | **18** | **WebSocket streaming path (pipeline.py:287)** |
| `max_syl` | **40** | **WebSocket streaming path** |
| Inter-clause silence | 200ms | Between comma/semicolon splits |
| Inter-sentence silence | 350ms | After period/question/exclamation |
| Fade per chunk | 5ms | Both ends of each chunk |

The streaming path uses larger chunks (min_syl=18, max_syl=40) than the engine default (10, 28) because:
- Larger chunks have better prosody (more natural phrase arc)
- Fewer chunks means fewer synthesis calls, reducing per-call overhead
- The 500ms TTFA target is still met with max_syl=40 at NFE=12 on GPU

---

## 12. CACHING ARCHITECTURE

**File:** `movio/cache/audio_cache.py`

### Two-Layer Cache Design

**Layer 1: Offline Phrase Cache (disk)**
- Location: `models/phrase_cache/`
- Format: 6 PCM16 files named `{sha256(voice|text)[:16]}.pcm`
- Pre-synthesized offline for common transport templates
- Hit latency: ~5ms (disk read)
- Managed by: `movio/acoustic/phrase_cache.py`
- Templates: "Your cab has arrived", "Please confirm OTP", etc.

**Layer 2: Runtime AudioCache (Redis or in-memory LRU)**
- Backend: Redis (production) or Python `OrderedDict` LRU (development)
- Hit latency: <50ms (Redis network + deserialize), <5ms (in-memory)

### Runtime Cache Key

```python
key = f"movio:tts:{hashlib.sha256(f'{voice}|{steps}|{text}'.encode()).hexdigest()}"
```

**Why `steps` is in the key:** If the NFE step count changes (e.g., from 12 to 8 during a config update), cached audio from the old configuration would have different quality. Including `steps` in the key ensures the cache automatically invalidates when the quality setting changes, without needing a manual flush.

**Why `voice` is in the key:** The same text spoken by the female voice and male voice must produce different audio. Voice is the first segment in the hash input.

### Cache Parameters

| Parameter | Value | Meaning |
|---|---|---|
| TTL | 604,800 seconds (7 days) | Cache entries expire after 7 days |
| Max per-entry size | 8 MB | Entries larger than 8MB are not cached (prevents cache poisoning from very long utterances) |
| Max entries (in-memory) | 2,048 | LRU eviction after this limit |
| Redis maxmemory | 8 GB (production) | `allkeys-lru` eviction policy |
| Write mode | Fire-and-forget | Cache writes are `asyncio.create_task()` — don't block the streaming response |

### Cache Write-Back

After a cache miss is resolved (audio synthesized), the cache write is asynchronous:
```python
asyncio.create_task(audio_cache.set(key, pcm16_bytes))
```

This means the response is not delayed waiting for Redis to acknowledge the write. If the write fails (Redis unavailable, entry too large), it is logged and silently dropped — the cache is a "best effort" optimization, not required for correctness.

---

## 13. SERVER AND CONCURRENCY MODEL

**File:** `movio/server/app.py`

### Server Details
- Framework: FastAPI 0.141.1
- ASGI server: Uvicorn 0.52.4 with uvloop (async I/O)
- Python version: ≥3.11 (required for `asyncio.TaskGroup`)
- Server version string in code: "v0.5.0"

### Two Separate Concurrency Paths

**Path 1: POST /tts and POST /tts/wav (serial queue)**
```
Incoming HTTP request
→ asyncio.Queue (size=unlimited)
→ _synthesis_worker coroutine (single consumer)
→ TTSPipeline.synthesize()
→ Returns full audio
```
All POST requests go through a single `asyncio.Queue` and a single `_synthesis_worker` coroutine. This ensures only one full synthesis runs at a time on the POST path, preventing GPU thrash (two PyTorch models competing for GPU memory).

**Path 2: WS /tts/stream (dedicated thread executor)**
```
Incoming WebSocket connection
→ Direct call to TTSPipeline.stream_pcm_chunks()
→ _stream_executor (ThreadPoolExecutor, max_workers=1)
→ Streaming generator
→ Binary frames sent to client
```
The WebSocket streaming path uses a separate `ThreadPoolExecutor(max_workers=1, thread_name_prefix="movio-stream")`. The generator is iterated via `loop.run_in_executor(_stream_executor, lambda: next(gen, None))`. This allows the WebSocket to stream audio while the event loop continues to handle other requests.

**Why two separate paths?**
The POST path needs to wait for the complete audio before returning a response. The WS path needs to send audio in real-time. If they shared a queue, a streaming request would block all subsequent POST requests for its full duration (potentially 3–5 seconds for a long utterance). The separation allows:
1. WS streams: First chunk TTFA ~280ms, subsequent chunks every ~350ms
2. POST requests: Full audio in 1–3 seconds, queued behind each other

### Stages A and B Concurrency

Stages A and B run in the asyncio thread pool (not the stream executor):
```python
norm = await asyncio.wait_for(
    loop.run_in_executor(None, normalizer.normalize, text),
    timeout=0.05  # 50ms
)
route = await asyncio.wait_for(
    loop.run_in_executor(None, router.route, norm.text),
    timeout=0.05  # 50ms
)
```

The 50ms timeout on each stage prevents a slow WFST or NeMo call from blocking the event loop indefinitely. If either stage times out, an `asyncio.TimeoutError` is caught and the pipeline proceeds with whatever partial result is available (or the un-normalized text for safety).

### Background Warmup

The server starts accepting requests immediately. Model loading happens in a background task:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(_background_warmup(pipeline))
    yield
```

During warmup:
- `GET /healthz` returns `{"status": "ok", "ready": false}`
- POST/WS requests will still be accepted but Stage C will fail immediately (model not loaded)

After warmup completes (~20–30 seconds for model download + `torch.compile`):
- `GET /healthz` returns `{"status": "ok", "ready": true}`
- All endpoints function normally

---

## 14. ALL CONFIGURATION VALUES AND NUMBERS

**File:** `config/settings.yaml`

Every configurable value in the system, with explanations:

### Server
| Key | Value | Meaning |
|---|---|---|
| `server.host` | `"0.0.0.0"` | Listen on all interfaces |
| `server.port` | `8000` | HTTP/WS port |
| `server.ws_chunk_ms` | `50` | WebSocket frame size in milliseconds = 1200 samples = 2400 bytes |
| `server.sample_rate` | `24000` | Output audio sample rate (must match IndicF5 native rate) |
| `server.opus_bitrate_kbps` | `24` | Placeholder; Opus compression not implemented |

### Stage A — Text Normalization
| Key | Value | Meaning |
|---|---|---|
| `stage_a.text_normalization.use_wfst` | `true` | Attempt WFST normalization if `pynini` available |
| `stage_a.text_normalization.wfst_language` | `"ta"` | Tamil WFST grammar |
| `stage_a.text_normalization.nemo_fallback` | `true` | Use NeMo if WFST unavailable and digits present |
| `stage_a.text_normalization.domain_rules` | `true` | Always run DomainRuleEngine (even after WFST) |
| `stage_a.text_normalization.gazetteer_path` | `"config/gazetteer/proper_nouns.txt"` | Proper noun list path |

### Stage B — Tanglish Router
| Key | Value | Meaning |
|---|---|---|
| `stage_b.router.lid_backend` | `"auto"` | Use IndicLID if installed, else heuristic |
| `stage_b.router.xlit_backend` | `"none"` | Disable IndicXlit; use en2ta.py instead |
| `stage_b.router.lid_target_langs` | `["ta", "en", "ta_roman"]` | Only classify into these 3 classes |
| `stage_b.router.insert_cs_tokens` | `true` | Insert `<cs>` at language transitions |
| `stage_b.router.cs_token` | `"<cs>"` | Code-switch boundary token string |
| `stage_b.router.keep_proper_nouns_latin` | `true` | Keep gazetteer matches in Latin script |
| `stage_b.router.gazetteer_path` | `"config/gazetteer/proper_nouns.txt"` | Same gazetteer |
| `stage_b.router.max_xlit_word_len` | `24` | Skip tokens longer than 24 chars (likely not a single word) |

### Stage C — IndicF5 Engine
| Key | Value | Meaning |
|---|---|---|
| `stage_c.indicf5.model_path` | `"base"` | `"base"` = download from HF; else path to local bundle directory |
| `stage_c.indicf5.device` | `"cuda"` | Use GPU (auto-falls back to CPU if CUDA unavailable) |
| `stage_c.indicf5.num_flow_steps` | `12` | NFE = 12 Euler steps (the key quality/speed tradeoff) |
| `stage_c.indicf5.sway_sampling_coef` | `-1.0` | Left-sway for better low-NFE quality |
| `stage_c.indicf5.cfg_strength` | `1.0` | CFG scale (1.0 = disabled) |
| `stage_c.indicf5.ode_method` | `"euler"` | ODE integrator (euler = 1 eval/step) |
| `stage_c.indicf5.speed` | `0.95` | Speech rate (0.95 = 5% slower than natural) |
| `stage_c.indicf5.voices_dir` | `"config/voices"` | Directory containing voice profiles |
| `stage_c.indicf5.default_voice` | `"ta_female_neutral"` | Default voice when not specified |

### Cache
| Key | Value | Meaning |
|---|---|---|
| `cache.backend` | `"memory"` | Use in-memory LRU (development); set `"redis"` for production |
| `cache.redis_url` | `"redis://localhost:6379/0"` | Redis connection URL |
| `cache.ttl_seconds` | `604800` | 7 days TTL |
| `cache.max_audio_bytes_mb` | `8` | Skip caching entries larger than 8MB |

### Pipeline Timeouts
| Key | Value | Meaning |
|---|---|---|
| `pipeline.stage_timeout_ms.stage_a` | `50` | Text normalization: 50ms hard cutoff |
| `pipeline.stage_timeout_ms.stage_b` | `50` | Routing: 50ms hard cutoff |
| `pipeline.stage_timeout_ms.stage_c` | `500` | Synthesis: 500ms hard cutoff (equals TTFA target) |

### Hybrid/MMS (Not Used in Production)
| Key | Value | Meaning |
|---|---|---|
| `stage_c.hybrid.cache_dir` | `"models/phrase_cache"` | Offline phrase cache location |
| `stage_c.hybrid.default_voice` | `"ta_female_neutral"` | Voice for phrase cache |
| `stage_c.mms.model_id` | `"facebook/mms-tts-tam"` | MMS-TTS Tamil model (not wired) |

---

## 15. PERFORMANCE TARGETS AND LATENCY BUDGET

### The 500ms Target

**p99 TTFA ≤ 500ms** at **15–20 concurrent streams** on GPU is the primary performance target. This comes from UX research: voice agent latency above 500ms breaks the "live conversation" illusion and makes the interaction feel like a broken phone call.

### The Latency Budget Breakdown

| Stage | Budget | Achieved (estimate, warm GPU) | Notes |
|---|---|---|---|
| Stage A — Text Normalization | 70ms | ~5ms (domain rules) / ~50ms (NeMo cold) | Domain rules are <1ms, NeMo first call is slow |
| Stage B — LID + Routing | 35ms | ~5ms (heuristic) / ~15ms (IndicLID) | Heuristic is much faster |
| Pre-synthesis (en2ta, cache check) | ~20ms | ~5ms (in-memory) | CTranslate2 IndicXlit ~1ms/word |
| Stage C — First chunk synthesis | 280ms | ~250–400ms @ T4, NFE=12 | T4 is the benchmark GPU (Kaggle free tier) |
| Stage D — PCM16 encoding + WS send | 50ms | ~5ms | Negligible overhead |
| **TOTAL p99 TTFA** | **≤500ms** | **~435ms estimate** | **65ms headroom** |

### NFE Step Ladder

The NFE step count is the primary quality/latency knob:

| NFE | ODE Method | Est. Latency/chunk (T4) | WER | UTMOS | Use Case |
|---|---|---|---|---|---|
| 8 | Euler | 150–250ms | ~4.16% | ~3.36 | Preview / draft only |
| **12** | **Euler** | **250–400ms** | **~2.67%** | **~3.45** | **Production default** |
| 16 | Euler | 350–550ms | ~2.53% | ~3.5 | Quality fallback |
| 24 | Midpoint | 700–1000ms | ~2.42% | ~3.55 | Max quality (slow) |
| 32 | Euler | 800–1200ms | ~2.42% | ~3.6 | Reference (too slow) |

Note: WER values from F5-TTS paper (F5-TTS is the upstream model for IndicF5). IndicF5 Tamil numbers are slightly different due to domain-specific training.

### Hardware Targets

| Hardware | Max Concurrent Streams | TTFA p99 |
|---|---|---|
| T4 GPU (16GB) | 5–6 | ~400–500ms |
| A100 40GB (3× replicas) | 15–20 | ≤500ms |
| L4 24GB (2× replicas) | 10–12 | ~450ms |
| CPU (no GPU) | 1 | ~3–5 seconds (IMPOSSIBLE for 500ms target) |

### Observed Benchmark Results

From `bench/results.json` (single concurrency, **cache-hit only** — no real synthesis):
```
concurrency_1:
  ttfa_p50_ms:   13.0
  ttfa_p95_ms:   13.0
  ttfa_p99_ms:   13.0
  ttfa_mean_ms:  13.0
  e2e_p50_ms:    21.8
  e2e_p99_ms:    21.8
  req_per_sec:   45.4
  audio_minutes: 0.04
  n_errors:      0
```

⚠️ **Important caveat:** These 13ms numbers reflect the cache LRU lookup time only — no model inference occurred. These are NOT end-to-end synthesis numbers. Real synthesis on cold cache with GPU will be ~250–400ms per chunk.

### Cost Analysis

From `bench/cost_analysis.py`:

| Metric | Value |
|---|---|
| Hardware cost (A100 40GB + L4 24GB + CPU) | ~$3.00/hour |
| Throughput at 15 concurrent streams | ~650 audio-minutes/hour |
| Cost per audio-minute | ~$0.0046 |
| Commercial API cost (ElevenLabs, Murf) | $0.015–$0.040/minute |
| Savings vs commercial | **4–9× cheaper** |

---

## 16. TRAINING: HOW THE TANGLISH LORA WAS CREATED

**Files:** `training/scripts/`, `training/configs/lora_tanglish.yaml`, `training/kaggle/movio_kaggle_training.ipynb`

### Why Fine-Tuning Was Needed

The base IndicF5 model was trained on 1,417 hours of formal Indian language speech. While it handles Tamil well, it was not specifically trained on:
1. Tanglish code-mixing (Tamil+English in the same utterance)
2. Transport domain vocabulary (OTP, booking ID, vehicle plate, fare)
3. The specific brand voice (female Tamil speaker with the characteristics defined in `ta_female_neutral/ref.wav`)

### Training Data Sources

| Dataset | Size | Source | Type |
|---|---|---|---|
| IndicVoices-R | Large (filtered to top 30%) | ai4bharat/indic_voices_r | Crowdsourced multi-speaker Tamil |
| Rasa Tamil | Medium | Common Voice variant | Conversational Tamil |
| Domain augmentation | Small (synthesized) | From base IndicF5 | Tanglish transport phrases |

### Training Pipeline (6 Steps)

**Step 1: Download data (`01_download_data.py`)**
Downloads `indicvoices_r` and Rasa Tamil datasets from HuggingFace.

**Step 2: Corpus preparation (`02_prepare_corpus.py`)**
Audio quality filtering using a heuristic quality score combining three signals:

```python
def quality_score(audio, sr):
    rms_90 = np.percentile(rms_frames, 90)
    rms_10 = np.percentile(rms_frames, 10)
    snr_proxy = min((rms_90 / (rms_10 + 1e-8)), 40.0) / 40.0  # [0, 1]

    clip_ratio = np.sum(np.abs(audio) > 0.98) / len(audio)
    clip_score = 1.0 - min(clip_ratio / 0.1, 1.0)  # [0, 1]

    spec_flatness = spectral_flatness(audio)  # geometric mean / arithmetic mean
    flatness_score = 1.0 - spec_flatness  # Low flatness = speech-like = good

    return (snr_proxy + clip_score + flatness_score) / 3.0
```

Filter criteria:
- Duration: 1.0–12.0 seconds
- SNR proxy: ≥ 12.0 dB
- Quality score: Top 30% kept (70% discarded)

The top-30% filter is aggressive but necessary: IndicVoices-R crowdsourced data has UTMOS ~2.1 (fair quality). Including the lower-quality 70% would degrade the fine-tuned model's naturalness below the base model's level (~2.27–3.12).

Splits: 94% train / 3% val / 3% test, seed=42

**Step 3: Build dataset (`03_build_dataset.py`)**
Converts filtered audio to Arrow format (HuggingFace `datasets` library) at 24kHz mono, creates metadata.csv with `audio_path | transcript | duration` columns.

**Step 4: Train LoRA (`04_train_lora.py`)**

**Method A (recommended): F5-TTS native finetune via `finetune_cli.py`**

LoRA hyperparameters (`training/configs/lora_tanglish.yaml`):
```yaml
lora_rank: 32       # LoRA rank (balance between capacity and size)
lora_alpha: 64      # Scale = alpha/rank = 64/32 = 2.0 (common setting)
lora_dropout: 0.05  # Small dropout for regularization
lora_target_modules:  # Which weight matrices to apply LoRA to
  - "attention.qkv"     # Query/Key/Value projections in attention
  - "attention.proj"    # Attention output projection
  - "attention.out"     # Optional additional output
  - "feed_forward.w1"   # FFN gate projection (SwiGLU gate)
  - "feed_forward.w3"   # FFN value projection (SwiGLU value)
  - "feed_forward.w2"   # FFN output projection
lora_bias: "none"   # Don't apply LoRA to biases
```

F5-TTS trainer hyperparameters:
```yaml
learning_rate: 5e-6         # Very small LR for fine-tuning pre-trained model
lr_scheduler: cosine        # Cosine annealing
num_warmup_updates: 200     # Warmup for first 200 gradient updates
num_epochs: 6               # 6 full epochs over filtered dataset
batch_frames: variable      # F5-TTS uses frame-budget batching, not fixed batch size
save_per_updates: 50000     # Checkpoint every 50k gradient steps
last_per_updates: 2000      # Keep last 2k checkpoints for best-of selection
fp16: true                  # Mixed precision on T4/P100
```

**Method B (experimental): PEFT LoRA**
```yaml
learning_rate: 1e-4           # Higher LR for PEFT approach
gradient_accumulation_steps: 8
batch_size_per_device: 2
effective_batch_size: 16       # 2 × 8 = 16
weight_decay: 0.01
warmup_ratio: 0.05             # 5% warmup
max_grad_norm: 1.0             # Gradient clipping
```

**Weight Conversion (critical step in `04_train_lora.py`):**

IndicF5 safetensors saves weights with `torch.compile()` applied, which renames all keys to have `ema_model._orig_mod.` prefix. The F5-TTS trainer expects clean keys. The conversion:
1. Strips `ema_model._orig_mod.` prefix from all key names
2. Removes vocoder keys (F5-TTS trainer doesn't use them)
3. Adds `initted=True`, `step=0` EMA buffers (required by `ema_pytorch` wrapper)
4. Hard-links converted file to avoid wasting 2×1.4GB disk (Kaggle has 20GB limit)
5. Cleans HF download cache after conversion (~1.4GB freed)

**Step 5: Export servable bundle (`05_merge_export.py`)**

Exports the fine-tuned model as a directory bundle:
```
models/indicf5_tanglish_v2/
├── model.pt           # Fine-tuned weights (merged LoRA → base if PEFT method)
├── vocab.txt          # Character vocabulary (NEVER modified during training)
└── config.json        # Architecture config (same as base model)
```

**NEVER modify the vocabulary:** The vocab.txt is frozen at the base model's ~2,545 characters. Adding or removing characters would invalidate all the pretrained embedding weights and cause catastrophic quality degradation.

**Step 6: Evaluate (`06_evaluate.py`)**

UTMOS + WER gate before the fine-tuned model is promoted:
```
UTMOS gate: ≥ (base model UTMOS) − 0.05   [fine-tune should not degrade naturalness]
WER gate:   ≤ (base model WER) + 2% absolute   [fine-tune should not degrade intelligibility]
```

If either gate fails, the checkpoint is rejected. Training continues until a checkpoint passes both gates.

### Kaggle Training Environment

The training is designed to run entirely within a Kaggle notebook (free T4 GPU, 30GB RAM, 20GB disk, 12h session limit). The notebook at `training/kaggle/movio_kaggle_training.ipynb` runs all 6 steps in sequence within a single 12-hour session.

Two fine-tuned model bundles are stored in `models/`:
- `models/indicf5_tanglish/` — v1 bundle (first fine-tune)
- `models/indicf5_tanglish_v2/` — v2 bundle (improved fine-tune with quality filtering)

---

## 17. EVALUATION FRAMEWORK

**Files:** `eval/`, `eval/testsets/tanglish_transport_200.tsv`

### Test Set

`eval/testsets/tanglish_transport_200.tsv` — 200 utterances covering:
- Tamil native text
- Romanized Tanglish
- Code-mixed Tamil+English
- Transport domain entities: OTPs, fares, distances, vehicle plates, times
- All 3 voice styles

### Automated Metrics

| Metric | Tool | Gate |
|---|---|---|
| WER (Word Error Rate) | `vasista22/whisper-tamil-base` — Tamil ASR | ≤ base model WER + 2% absolute |
| UTMOS (Naturalness MOS) | `utmos22_strong` predictor | ≥ base model UTMOS − 0.05 |
| Speaker Similarity | `resemblyzer` (VGG-based speaker encoder) | cosine ≥ 0.70 per voice, no drop > 0.10 vs baseline |
| Per-chunk DiT latency | CUDA Event timers | max < 5,000ms |

**UTMOS calibration note:** The absolute UTMOS scale for Tamil Tanglish synthesis is 2.28–3.12 (not the 4.0+ that English TTS models achieve on English benchmarks). Tamil is a tonal language with phonetic characteristics that the UTMOS model (trained mostly on English) underrates. The gate is therefore relative to the base model, not an absolute threshold.

### Human Evaluation Protocol

File: `eval/HUMAN_EVAL_PROTOCOL.md`

5-dimension rating protocol (all rated 1–5 by native Tamil speakers):
1. **Overall naturalness** — Does it sound like a real human Tamil speaker?
2. **Pronunciation accuracy** — Are all words pronounced correctly?
3. **Code-switch smoothness** — Do Tamil↔English transitions sound natural?
4. **Prosody** — Is pitch/rhythm/pace natural for the domain?
5. **Entity accuracy** — Are vehicle plates, OTPs, times, amounts read correctly?

Gate for production promotion: MOS ≥ 4.0 on dimensions 1–4; 100% accuracy on dimension 5.

---

## 18. DEPLOYMENT ARCHITECTURE

**Files:** `deployment/Dockerfile`, `deployment/docker-compose.yml`, `deployment/triton/`

### Production Docker Stack

```yaml
# deployment/docker-compose.yml
services:
  redis:
    image: redis:7-alpine
    command: redis-server --maxmemory 8gb --maxmemory-policy allkeys-lru
    # 8GB LRU cache for shared audio cache across all TTS replicas

  tts:
    image: movio-tts:latest    # Built from deployment/Dockerfile
    build:
      context: ..
      dockerfile: deployment/Dockerfile  # nvidia/cuda:12.4.1-cudnn-runtime
    environment:
      - REDIS_URL=redis://redis:6379/0
      - HF_TOKEN=${HF_TOKEN}       # Required for gated IndicF5 weights
      - DEVICE=cuda
    deploy:
      replicas: 3                  # 3 TTS instances per A100 (share GPU via MIG or time-slicing)
    ports:
      - "8000-8002:8000"           # Expose ports 8000, 8001, 8002
```

### Triton Inference Server (Advanced Deployment)

For production-scale deployment, the `deployment/triton/` directory provides a Triton Python backend:

**`deployment/triton/model_repo/indicf5_streaming/config.pbtxt`:**
```
max_batch_size: 6               # Maximum 6 requests per batch
backend: "python"               # Python backend
model_transaction_policy { decoupled: True }  # Streaming mode
dynamic_batching {
  preferred_batch_size: [4, 6]  # Try to form batches of 4 or 6
  max_queue_delay_microseconds: 5000  # Wait up to 5ms for batching
}
instance_group [{ kind: KIND_GPU; gpus: [0]; count: 3 }]  # 3 GPU instances
parameters { key: "NUM_FLOW_STEPS"; string_value: "12" }
```

The Triton backend micro-batches multiple synthesize requests together into a single `infer_batch_process()` call, which is more GPU-efficient than running each request separately.

### Development Docker Stack (CPU)

`docker-compose.yaml` in the project root uses the CPU image (`python:3.11-slim`). This is for development and testing without a GPU:
- No Redis (uses in-memory LRU)
- Single TTS replica (no scaling)
- Model runs on CPU (TTFA ~3–5 seconds, not suitable for production)

---

## 19. THE COMPLETE FILE MAP

```
movio/
├── movio/                          # Main Python package
│   ├── __init__.py                 # version="0.1.0"; suppresses noisy library warnings
│   ├── __main__.py                 # Entry point: imports main() from server.app
│   ├── pipeline.py                 # TTSPipeline: orchestrates Stages A→B→C→D
│   ├── acoustic/
│   │   ├── __init__.py
│   │   ├── engine_base.py          # Abstract base class AcousticEngine
│   │   ├── indicf5_engine.py       # [ACTIVE] IndicF5 CFM-DiT engine
│   │   ├── chunking.py             # Syllable-budget text chunker
│   │   ├── vocoder.py              # Vocos neural vocoder wrapper
│   │   ├── hybrid_engine.py        # [NOT WIRED] Phrase cache + MMS fallback
│   │   ├── mms_engine.py           # [NOT WIRED] Facebook MMS-TTS Tamil
│   │   ├── cascaded_engine.py      # [NOT WIRED] VITS→FastSpeech2 cascade
│   │   ├── vits_engine.py          # [NOT WIRED] Legacy VITS engine
│   │   ├── fastspeech2_engine.py   # [NOT WIRED] Legacy FastSpeech2 engine
│   │   ├── vits_text_processor.py  # 56-token Tamil VITS vocabulary
│   │   ├── transliterate.py        # Legacy transliterator (dead code)
│   │   └── phrase_cache.py         # Offline phrase pre-synthesis tool
│   ├── cache/
│   │   └── audio_cache.py          # Redis/LRU cache with SHA256 keying
│   ├── router/
│   │   ├── tanglish_router.py      # Stage B: token-level LID + transliteration
│   │   ├── lid.py                  # HeuristicLID + IndicLIDBackend
│   │   ├── xlit.py                 # IndicXlit wrapper (currently unused)
│   │   └── en2ta.py                # [ACTIVE] English→Tamil phonetic script
│   ├── server/
│   │   ├── app.py                  # FastAPI server with REST + WebSocket
│   │   └── static/index.html       # Live streaming web demo UI (~1500 lines)
│   ├── textnorm/
│   │   ├── normalizer.py           # Stage A: WFST→NeMo→DomainRules orchestration
│   │   └── domain_rules.py         # 12-rule transport domain verbalizer
│   └── utils/
│       └── audio.py                # PCM16/WAV/resample/crossfade/trim/stopwatch
│
├── config/
│   ├── settings.yaml               # All runtime configuration
│   ├── gazetteer/proper_nouns.txt  # 37 proper nouns (Chennai places + brands)
│   └── voices/
│       ├── ta_female_neutral/
│       │   ├── voice.yaml          # Voice profile: name, sample_rate, description
│       │   └── ref.wav             # 9.3s reference audio for zero-shot cloning
│       └── ta_male_neutral/
│           ├── voice.yaml
│           └── ref.wav             # 12.4s reference audio
│
├── third_party/
│   └── IndicF5/                    # Vendored ai4bharat/IndicF5 fork
│       ├── model.py                # INF5Model HuggingFace wrapper
│       ├── config.json             # Model architecture config
│       ├── checkpoints/vocab.txt   # Character vocabulary (~2545 chars)
│       └── f5_tts/
│           ├── api.py              # F5TTS high-level inference API
│           ├── model/
│           │   ├── cfm.py          # Conditional Flow Matching core
│           │   ├── modules.py      # Model modules
│           │   └── backbones/
│           │       ├── dit.py      # DiT backbone (Diffusion Transformer)
│           │       ├── mmdit.py    # MM-DiT (Multi-Modal DiT variant)
│           │       └── unett.py    # U-Net T backbone
│           └── infer/
│               └── utils_infer.py  # load_model, infer_batch_process, etc.
│
├── models/
│   ├── indicf5_tanglish/           # Fine-tuned v1 bundle
│   ├── indicf5_tanglish_v2/        # Fine-tuned v2 bundle
│   ├── indicxlit_ct2/              # Local CTranslate2 IndicXlit
│   ├── phrase_cache/               # 6 pre-synthesized PCM files
│   └── vits/tamil_vits/            # Legacy VITS Tamil checkpoint (not used)
│
├── training/
│   ├── scripts/01_download_data.py
│   ├── scripts/02_prepare_corpus.py    # Audio quality filtering pipeline
│   ├── scripts/03_build_dataset.py     # HF Arrow dataset builder
│   ├── scripts/04_train_lora.py        # LoRA training with weight conversion
│   ├── scripts/05_merge_export.py      # Export servable bundle
│   ├── scripts/06_evaluate.py          # UTMOS + WER gate
│   ├── configs/lora_tanglish.yaml      # LoRA hyperparameters
│   └── kaggle/movio_kaggle_training.ipynb  # All-in-one Kaggle notebook
│
├── bench/
│   ├── benchmark.py                # TTFA/E2E latency benchmark (p99 sweep)
│   └── cost_analysis.py            # $/audio-minute cost calculator
│
├── eval/
│   ├── make_testset.py             # 200-utterance TSV generator
│   ├── run_quality_eval.py         # Automated WER/UTMOS/similarity eval
│   ├── HUMAN_EVAL_PROTOCOL.md      # 5-dimension human eval protocol
│   └── testsets/tanglish_transport_200.tsv  # Frozen 200-utterance test set
│
├── deployment/
│   ├── Dockerfile                  # Production GPU image (nvidia/cuda:12.4.1)
│   ├── docker-compose.yml          # 3× TTS + Redis 8GB stack
│   └── triton/model_repo/          # NVIDIA Triton Python backend
│
├── tests/
│   ├── test_pipeline.py            # 23 offline unit tests (no model weights needed)
│   └── test_chunking_smart.py      # Smart chunking unit tests
│
├── Dockerfile                      # Development CPU image (python:3.11-slim)
├── docker-compose.yaml             # Development stack
├── pyproject.toml                  # Build system + deps + ruff config
├── requirements.txt                # Pip-compatible dep shim
├── pytest.ini                      # asyncio_mode=auto
└── README.md                       # Quickstart and solution overview
```

---

## 20. ALL DEPENDENCIES

### Core Runtime Dependencies (pinned in uv.lock)

| Library | Version | Purpose |
|---|---|---|
| `numpy` | 1.26.4 | Array operations, PCM16 encoding |
| `pyyaml` | ≥6.0 | Configuration file loading |
| `soundfile` | 0.14.0 | WAV read/write |
| `librosa` | 0.11.0 | Resampling, silence detection |
| `fastapi` | 0.141.1 | REST + WebSocket server framework |
| `uvicorn[standard]` | 0.52.4 | ASGI server with uvloop |
| `websockets` | 17.1 | WebSocket protocol implementation |
| `redis` | 8.1.0 | Redis cache client |
| `httpx` | 0.28.1 | HTTP client (health checks) |
| `psutil` | 7.2.2 | Resource monitoring for benchmarks |
| `torch` | 2.13.0+cu130 | PyTorch GPU inference |
| `transformers` | 5.16.1 | HuggingFace model loading |
| `huggingface_hub` | 1.29.0 | Model download from HF Hub |
| `f5-tts` | 1.1.22 | Training framework only (NOT used for serving) |
| `safetensors` | 0.8.0 | Efficient model weight loading |
| `vocos` | 0.1.0 | Vocos neural vocoder |
| `ctranslate2` | (installed) | IndicXlit fast CPU inference |
| `nltk` + cmudict | (installed) | CMU Pronunciation Dictionary |
| `pydub` | (imported) | Fallback silent audio segment generation |
| `torchaudio` | (needed) | Reference audio loading and resampling |

### Optional Dependencies (graceful degradation)

| Library | Purpose | Fallback Behavior |
|---|---|---|
| `pynini` + `indic-text-normalization` | WFST Tamil normalization | Skip to domain rules |
| `nemo_text_processing` | NeMo ITN normalizer | Skip to domain rules |
| `ai4bharat-transliteration` (IndicXlit) | Transliteration | Use en2ta.py CTranslate2 only |
| `ai4bharat/IndicLID` + `fasttext` | Language identification | Use HeuristicLID |
| `resemblyzer` | Speaker similarity eval | Skip similarity metric |
| `jiwer` | WER/CER computation | Skip WER metric |
| `SpeechMOS` (utmos22_strong) | UTMOS naturalness MOS | Skip UTMOS metric |
| `vasista22/whisper-tamil-base` | Tamil ASR for WER | Skip WER |
| `datasets` (HuggingFace) | Training dataset building | N/A |
| `peft` | Experimental PEFT LoRA | Use F5-TTS trainer path |
| `onnxruntime` | ONNX inference for VITS/FS2 | Not used in active pipeline |

### External Services

| Service | Purpose | Authentication |
|---|---|---|
| HuggingFace Hub | Download IndicF5 weights | `HF_TOKEN` env variable (gated model) |
| HuggingFace Hub | Download IndicXlit CTranslate2 | No auth (public model) |
| Redis 7 | Shared audio cache | Redis URL in config |

### Python Version
- Required: ≥3.11, <3.14
- Development: CPython 3.11 (managed by `uv`)
- Package manager: `uv` with locked dependencies in `uv.lock`

---

## 21. KNOWN ISSUES, TRACKED BUGS, AND FUTURE WORK

### Active Issues

1. **TTFA on cold cache still ~435ms on T4**: The 500ms target is met, but there's only ~65ms headroom. On a loaded T4 with 5–6 concurrent streams, p99 may breach 500ms. Mitigation: The 3-replica A100 production setup distributes load to keep within target.

2. **NeMo cold-start exceeds 50ms timeout**: On first request after server restart, NeMo text processing takes ~50ms to initialize (model load). This can cause Stage A to time out and fall through to domain rules only. Mitigation: Background warmup task runs Stage A once before marking server ready.

3. **pyproject.toml description mismatch**: The pyproject.toml says "Solution 4 (VITS)" but the actual implementation is Solution 1 (IndicF5). This is a stale comment from an earlier development phase.

4. **Version string inconsistency**: `__init__.py` says version=0.1.0, `server/app.py` says version=v0.5.0, `pyproject.toml` says version=0.2.0. These should be unified.

5. **Missing torchaudio in requirements**: `torchaudio` is used in `indicf5_engine.py` for reference audio loading but is not in `requirements.txt`. If someone installs from requirements only (not uv.lock), torchaudio will be missing.

### Not Yet Implemented

1. **Opus audio compression**: `settings.yaml` has `opus_bitrate_kbps: 24` but the pipeline always sends raw PCM16. Adding Opus would reduce bandwidth by ~5×.

2. **GPU FP8 quantization**: The latency budget mentions FP8 for Stage C (could reduce 12-step latency to ~150–200ms). Not yet implemented.

3. **Triton micro-batching**: The Triton backend stub exists in `deployment/triton/` but `1/model.py` is incomplete. The current production setup uses FastAPI with 3 replicas instead.

4. **Vocos fine-tuning**: The current Vocos is used off-the-shelf. Fine-tuning Vocos on the Tanglish domain could improve audio quality for code-mixed speech.

5. **Indic-NLP text normalization**: The WFST layer uses `pynini` but the more powerful `indic-nlp-library` (which handles ZWJ/ZWNJ, Unicode normalization, Tamil script compound character handling) is not integrated.

### Future Work (from PLAN and research documents)

1. **Streaming DiT inference**: The ultimate latency improvement would be modifying the CFM-DiT to support autoregressive mel spectrogram generation (each frame conditioned on previous frames), enabling sub-100ms TTFA without chunking. This is an active research direction.

2. **Dedicated Tanglish LoRA v3**: The v2 LoRA was trained on general Tamil. A v3 trained specifically on audio of transport announcements (booking confirmations, OTP readings, navigation instructions) would likely improve domain-specific prosody.

3. **Speaker adapter network**: Currently, new voices require a new reference audio file. A small speaker adapter network trained to convert any voice description text to a speaker embedding would allow dynamic voice creation.

4. **Real-time adaptation**: The model could adapt to background noise/microphone characteristics of the call to produce audio that sounds right in the user's specific acoustic environment.

---

## 22. WHY EVERY KEY DECISION WAS MADE

| Decision | Alternative Considered | Why This Choice |
|---|---|---|
| IndicF5 (CFM-DiT) over Parler-TTS | Parler-TTS (Approach 4/movio-2) | NAR synthesis gives predictable chunk latency; AR is unbounded |
| NFE=12 over NFE=8 or NFE=20 | See step ladder table | 12 steps: best quality/speed tradeoff for 500ms budget |
| Euler ODE over midpoint | Midpoint: 2 evals/step | Same quality at 12 NFE, but 2× cheaper |
| Sway=-1.0 | No sway (uniform steps) | Measurably better WER at low NFE per EPSS paper |
| CFG=1.0 (disabled) | CFG=2.0 (stronger guidance) | 2× cheaper; quality bottleneck is NFE not CFG at 12 steps |
| Speed=0.95 | Speed=1.0 (natural) | Domain: transport announcements should be slightly slow for clarity |
| Character-level tokenization | BPE or word-level | Character level works best for code-mixed text; no OOV problem |
| Syllable budget chunking | Character count chunking | Equal acoustic duration per chunk → predictable streaming latency |
| 3-tier en2ta over IndicXlit only | IndicXlit for all English | Override dict covers most transport vocab; CMU covers OOV English |
| Single-ஃ rule | Allow multiple ஃ | Double ஃ causes IndicF5 to garble audio (training data never had it) |
| Digit pairing with commas | Continuous digit stream | Real human cadence for OTP is pairs; commas = breath not split |
| `<cs>` tokens internal only | No boundary marking | Prosody improvement at code-switch boundaries per bilingual TTS research |
| Fire-and-forget cache writes | Blocking cache writes | Cache write never delays streaming response |
| SHA256(voice|steps|text) key | SHA256(text) only | Step count change auto-invalidates cache; voice change produces correct audio |
| Vendor third_party/IndicF5 | Install via pip f5-tts | pip f5-tts has different architecture; must use vendored version |
| Top-30% quality filter | All training data | IndicVoices-R bottom 70% degrades UTMOS below base model |
| Kaggle T4 training | Local GPU or AWS | Kaggle free tier: T4, 30GB RAM, free — zero training cost |
| Redis for production cache | Only in-memory | Redis is shared across 3 TTS replicas; in-memory would miss on other instances |
| min_syl=18, max_syl=40 for streaming | Default 10, 28 | Larger chunks: better prosody, fewer synthesis calls per utterance |

---

## 23. CRITICAL WARNINGS AND GOTCHAS

### ⚠️ NEVER Use pip f5-tts with IndicF5 Weights

The `f5-tts==1.1.22` pip package has a different architecture config from `ai4bharat/IndicF5`. If you import from the pip package, the model will load silently but produce completely garbled audio because the dimensions don't match. The solution is already in the codebase:

```python
# In indicf5_engine.py:
for k in list(sys.modules):
    if k.startswith("f5_tts"):
        del sys.modules[k]   # Purge pip f5_tts before importing vendored version
sys.path.insert(0, "third_party/IndicF5")
```

If you ever `import f5_tts` before calling `_ensure_indicf5_path()`, you will get the wrong version. Always call `_ensure_indicf5_path()` first.

### ⚠️ mel_spec_type MUST Be "vocos"

IndicF5 was trained with 100-channel vocos mel spectrograms. If you use any vocoder that expects 80 mel channels (HiFi-GAN, the old F5-TTS default), the output will be robotic/garbled. Always verify:
```python
mel_spec_type = "vocos"    # REQUIRED — not "bigvgan" or "other"
n_mel_channels = 100       # REQUIRED — not 80
```

### ⚠️ HF_TOKEN Required

The `ai4bharat/IndicF5` model is gated on HuggingFace Hub. You MUST set the `HF_TOKEN` environment variable to your HuggingFace access token before running `engine.load()`. Without it, the download will fail with a 401 Unauthorized error.

### ⚠️ Never Modify vocab.txt

The vocabulary file `checkpoints/vocab.txt` must never be modified. It maps character indices to the embedding layer of the DiT. Adding or removing characters would misalign all text inputs with the model's embeddings, causing catastrophic quality degradation or crashes.

### ⚠️ CPU Inference Cannot Meet 500ms Target

On CPU, a single 12-step CFM-DiT inference takes ~3–5 seconds per chunk. There is no way to meet the 500ms TTFA target on CPU. A GPU is strictly required for production deployment.

### ⚠️ torch.compile() Takes ~2 Minutes on First GPU Use

When `torch.compile(mode="reduce-overhead")` runs on the first inference call, it takes 1–3 minutes to JIT-compile the model. This is why the background warmup task exists — it triggers a dummy synthesis call during server startup so the compile cost is paid before any user traffic arrives.

---

*End of Report — Movio IndicF5 Approach*

*This report was generated from complete code analysis of all files in the `movio/` directory as of September 2026.*
