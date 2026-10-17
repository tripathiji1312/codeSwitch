# TECHNICAL REPORT: Movio-2 — Indic-Parler-TTS Autoregressive Streaming TTS for Tamil, Tanglish, and English

**Project folder:** `movio-2/`
**Approach number:** Approach 2
**Core model:** `ai4bharat/indic-parler-tts` — Autoregressive Audio Language Model (ParlerTTS)
**Repository remote:** `https://github.com/tripathiji1312/movio_new.git`
**Primary author(s):** Movio TTS team
**Report version:** 1.0 — September 2026
**Purpose of this report:** Complete technical reference. Anyone reading this document — human or AI — should be able to fully understand what was built, why every decision was made, what every number means, how the system works, what problems were solved, and what remains to be done.

---

## TABLE OF CONTENTS

1. [Problem Statement and Goals](#1-problem-statement-and-goals)
2. [Why This Approach Was Chosen](#2-why-this-approach-was-chosen)
3. [High-Level Architecture](#3-high-level-architecture)
4. [The Primary Model: ai4bharat/indic-parler-tts](#4-the-primary-model-ai4bharatindic-parler-tts)
5. [Component: Text Normalization (backend/normalizer.py)](#5-component-text-normalization-backendnormalizerpy)
6. [Component: Sentence Chunking (backend/chunker.py)](#6-component-sentence-chunking-backendchunkerpy)
7. [Component: TTS Engine (backend/tts_engine.py)](#7-component-tts-engine-backendtts_enginepy)
8. [Component: FastAPI WebSocket Server (backend/server.py)](#8-component-fastapi-websocket-server-backendserverpy)
9. [Audio Post-Processing Pipeline](#9-audio-post-processing-pipeline)
10. [The LRU Audio Cache](#10-the-lru-audio-cache)
11. [Frontend: Main TTS Demo (frontend/index.html)](#11-frontend-main-tts-demo-frontendindexhtml)
12. [Frontend: Conversation Interrupt Demo (frontend/conversation.html)](#12-frontend-conversation-interrupt-demo-frontendconversationhtml)
13. [Language Support: Tamil, Tanglish, and English](#13-language-support-tamil-tanglish-and-english)
14. [All Configuration Values and Numbers](#14-all-configuration-values-and-numbers)
15. [Performance Targets and Metrics](#15-performance-targets-and-metrics)
16. [Deployment: Kaggle GPU Notebook](#16-deployment-kaggle-gpu-notebook)
17. [The Complete File Map](#17-the-complete-file-map)
18. [All Dependencies](#18-all-dependencies)
19. [Known Issues, Tracked Bugs, and Future Plans](#19-known-issues-tracked-bugs-and-future-plans)
20. [Why Every Key Decision Was Made](#20-why-every-key-decision-was-made)
21. [Critical Warnings and Gotchas](#21-critical-warnings-and-gotchas)
22. [Git History: What Was Built and When](#22-git-history-what-was-built-and-when)

---

## 1. PROBLEM STATEMENT AND GOALS

### The Domain Context
Movio-2 is building the same product as Movio (Approach 1): a real-time AI voice agent for a cab service that speaks to Tamil Nadu users in their natural language — which is not pure Tamil, not pure English, but a blend called **Tanglish**.

The specific requirements are identical to Approach 1:
1. **Natural sounding**: Human-quality Tamil/Tanglish speech (not robotic synthesis)
2. **Low latency**: First audio must reach the user within **500ms** of the text being submitted
3. **Domain accuracy**: OTPs, vehicle plates, booking IDs, fares, times, distances must be pronounced correctly
4. **Speaker consistency**: All output sounds like the same brand voice ("Jaya")
5. **Real-time streaming**: Audio must start playing before synthesis is complete

### What Makes This Approach Different from Approach 1

Where Approach 1 (movio/) chose a **Non-Autoregressive CFM-DiT model** (IndicF5) because of its predictable fixed-latency synthesis, Approach 2 (movio-2/) chose an **Autoregressive Audio Language Model** (indic-parler-tts) because of:

1. **Higher voice naturalness ceiling**: Autoregressive LMs model full audio token sequences and produce more natural prosody and intonation
2. **Description-conditioned synthesis**: ParlerTTS uses a natural language voice description ("Jaya speaks with a slightly low-pitched voice at a moderate pace") instead of a reference audio clip — no speaker audio recording needed
3. **Production simplicity**: No reference audio management, no zero-shot voice cloning complexity
4. **State-of-the-art Tamil quality**: `ai4bharat/indic-parler-tts` achieves 75.48% Tamil Native Speaker Score — the highest among open-source Indian TTS models at the time of development

### The Core Engineering Challenge of This Approach

Autoregressive models generate audio token by token. For a typical 10-word Tamil sentence, the model needs to generate ~300–500 audio tokens. On a T4 GPU at roughly 86 tokens/second, this takes **3–5 seconds** — 6–10× over the 500ms target.

The engineering problem this project solves: **How do you stream real-time audio from an autoregressive model that is inherently sequential?**

The answer is the `ParlerTTSStreamer` class, which decodes audio incrementally as tokens are generated, producing playable audio chunks every 0.3 seconds instead of waiting for the entire sequence to finish.

---

## 2. WHY THIS APPROACH WAS CHOSEN

### Comparison with Approach 1 (IndicF5)

| Property | Approach 1 (IndicF5) | Approach 2 (Parler-TTS) |
|---|---|---|
| Model type | Non-Autoregressive (NAR), CFM-DiT | Autoregressive (AR), audio LM |
| Model size | 337M parameters | 900M parameters |
| Sample rate | 24kHz | 44.1kHz |
| Voice conditioning | Reference audio (9–12s clip) | Natural language description |
| First audio latency | ~280ms per first chunk | ~300–500ms first audio (streamed) |
| Naturalness ceiling | High | Very high (AR models model full sequence) |
| Tanglish handling | Via transliteration pipeline | Direct code-mix in training data |
| Streaming | Chunk-by-chunk (non-AR is naturally chunkable) | Token-by-token via ParlerTTSStreamer |
| Fine-tuning needed? | Yes (LoRA for Tanglish) | Not yet (base model handles Tamil well) |
| License | MIT | Apache 2.0 |

### Why ParlerTTS Was Selected

1. **Higher naturalness**: ParlerTTS uses a 0.9B autoregressive language model over DAC codec tokens — a fundamentally richer generative model than 12-step ODE integration. The model can learn subtle prosodic patterns (emphasis, hesitation, rhythm) that are harder to capture with fixed-step flow matching.

2. **44.1kHz output**: The DAC (Descript Audio Codec) operates at 44,100Hz. This gives better high-frequency audio quality (up to 22.05kHz Nyquist) compared to IndicF5's 24kHz (12kHz Nyquist). The difference is perceptible especially in consonants (s, sh, ch sounds) and fricatives.

3. **No reference audio management**: Approach 1 requires maintaining high-quality reference audio recordings for each voice. Approach 2 uses a text description. This is simpler and means voices can be created/modified by changing a text string.

4. **ai4bharat/indic-parler-tts quality**: This specific model was published at Interspeech 2025 with 75.48% ± 2.18% Tamil Native Speaker Score (NSS) — a metric where human raters determine whether a synthesized voice sounds like a native speaker. This is significantly higher than other open-source Tamil TTS options.

5. **Direct Tanglish handling**: The model was trained on 21 Indian languages including mixed-script data. It handles some degree of code-mixing natively without needing a separate transliteration pipeline.

### The Trade-Off

The trade-off accepted by choosing this approach:
- **More complex streaming implementation**: Extracting real-time audio from an AR model requires the `ParlerTTSStreamer` architecture, which is significantly more engineering-intensive than the NAR chunking approach
- **Higher memory footprint**: 0.9B at bfloat16 = ~1.8GB vs 0.6GB for IndicF5 (337M at float16)
- **Concurrency limitation**: Only one `model.generate()` can run at a time due to PyTorch GPU state (enforced by asyncio.Lock). Approach 1 can potentially run multiple smaller inferences.

---

## 3. HIGH-LEVEL ARCHITECTURE

```
                         INPUT TEXT
                             │
                    ┌────────▼────────┐
                    │  BROWSER SIDE   │   frontend/index.html (JS)
                    │  autoTagText()  │   JavaScript regex pipeline:
                    │                 │   {{OTP}}, {{AMOUNT}}, {{CODE}}...
                    └────────┬────────┘
                             │ Tagged text (e.g., "Your OTP is {{OTP:4821}}")
                             │ WebSocket send {type:"synthesize", text:...}
                    ┌────────▼────────┐
                    │  FastAPI Server │   backend/server.py
                    │  /ws endpoint   │
                    └────────┬────────┘
                             │
             ┌───────────────▼───────────────┐
             │       split_into_sentences()   │   backend/chunker.py
             │       + normalize_text()       │   backend/normalizer.py
             └───────────────┬───────────────┘
                             │ List of normalized chunks
             ┌───────────────▼───────────────┐
             │        asyncio.Lock()          │   Serialize GPU access
             │     (one generation at a time) │
             └───────────────┬───────────────┘
                             │
              ┌──────────────▼──────────────┐
              │          Cache Check         │   SHA256(text + config fingerprint)
              │      (LRU OrderedDict)       │   200 entries, ~84MB max
              └──────────────┬──────────────┘
                             │ MISS                        HIT → send PCM16 bytes
              ┌──────────────▼──────────────┐
              │      async_stream_generate() │   backend/tts_engine.py
              │                             │
              │  [Background Thread]        │
              │  ┌─────────────────────┐    │
              │  │ flan-t5-large        │    │
              │  │ (description encoder)│    │
              │  │ → cached hidden     │    │
              │  │   state (once)      │    │
              │  └─────────┬───────────┘    │
              │            │                │
              │  ┌─────────▼───────────┐    │
              │  │ parler-tts 0.9B AR  │    │
              │  │ model.generate()    │    │
              │  │ - do_sample=True    │    │
              │  │ - temp=1.0          │    │
              │  │ - seed=42           │    │
              │  │ - ParlerTTSStreamer  │    │
              │  └─────────┬───────────┘    │
              │            │ Every 26 tokens │
              │  ┌─────────▼───────────┐    │
              │  │  DAC Decoder        │    │
              │  │  (9 codebooks →     │    │
              │  │   float32 audio)    │    │
              │  └─────────┬───────────┘    │
              │            │                │
              │  loop.call_soon_threadsafe() │
              └──────────────┬──────────────┘
                             │ Every ~0.3s: numpy float32 audio array
              ┌──────────────▼──────────────┐
              │    Audio Post-Processing     │   backend/server.py
              │    (live stream path):       │
              │    np.clip → PCM16 → send    │
              └──────────────┬──────────────┘
                             │ Binary PCM16 frames
              ┌──────────────▼──────────────┐
              │   After chunk complete:      │
              │   postprocess_full_audio()   │
              │   HPF 60Hz → silence trim →  │
              │   LUFS normalize → cache     │
              └──────────────┬──────────────┘
                             │
                          CLIENT
                  (Web Audio API scheduling)
```

### Two WebSocket Endpoints

**`/ws` — Main synthesis endpoint** (`ws_synthesize` handler):
- Accepts: `{type: "synthesize", text: "..."}`
- Streams: binary PCM16 audio frames
- Response protocol: `{type:"chunks",...}` → `{type:"chunk_start",...}` → binary → `{type:"chunk_end",...}` → `{type:"done"}`

**`/ws/conversation` — Conversation endpoint** (`ws_conversation` handler):
- Accepts: `{type: "speak"/"interrupt", text: "..."}`
- Streams: binary PCM16 frames with 4-byte generation ID header
- Supports: Mid-sentence interrupt/preempt

---

## 4. THE PRIMARY MODEL: ai4bharat/indic-parler-tts

### Model Overview

| Property | Value |
|---|---|
| HuggingFace ID | `ai4bharat/indic-parler-tts` |
| Architecture | ParlerTTS (autoregressive audio language model) |
| Total parameters | 0.9 billion |
| Audio codec | DAC (Descript Audio Codec) at 44.1kHz |
| Text encoder (voice) | `google/flan-t5-large` |
| Text encoder (speech) | Same flan-t5-large tokenizer |
| Sample rate | 44,100 Hz |
| License | Apache 2.0 |
| Published | Interspeech 2025 |
| Languages | 21 Indian languages |

### Architecture Deep Dive: How ParlerTTS Works

ParlerTTS is a two-stage system:

**Stage 1: Voice description encoding (flan-t5-large)**

The voice characteristics are specified in natural language:
```
"Jaya speaks with a slightly low-pitched voice at a moderate pace.
 The recording is very high quality, clear and close-sounding,
 with no background noise."
```

This description is tokenized and processed by `google/flan-t5-large` (a 780M-parameter T5 encoder-decoder). The encoder's `last_hidden_state` becomes the conditioning signal that shapes the synthesized voice. This is how you get "Jaya's voice" — not by providing a reference recording, but by describing the voice's characteristics in English text.

**Why flan-t5-large specifically?** The ParlerTTS research found that instruction-tuned T5 models (flan-t5) produce better voice conditioning representations than standard T5 because they were trained to understand and follow natural language instructions — which is exactly what a voice description is.

**Stage 2: Autoregressive audio token generation**

The speech text (e.g., "Unga OTP enna?") is also tokenized by the same flan-t5-large tokenizer and processed. The model then auto-regressively generates DAC codec tokens conditioned on both the voice description (from Stage 1) and the speech text.

**The DAC Codec:**

DAC (Descript Audio Codec) is a neural audio codec that represents audio as discrete tokens:
- Architecture: Encoder → Residual Vector Quantizer (RVQ) → Decoder
- Operating rate: 44,100 Hz
- Number of codebooks: 9 parallel codebooks
- Codebook size: 1,024 entries per codebook
- Upsampling ratios: 8 × 8 × 4 × 2 = 512
- Frame rate: `44100 / 512 ≈ 86.13 tokens per second per codebook`

For a 3-second Tamil sentence, the model generates approximately:
- `3 seconds × 86.13 tokens/sec × 9 codebooks = ~2325 total tokens`

These 9 parallel codebooks represent different "levels" of audio detail:
- Codebook 1 (coarsest): Rough spectral envelope and fundamental pitch
- Codebooks 2–9 (finer): Progressive detail — timbre, fine pitch, noise characteristics

The ParlerTTS model generates all 9 codebooks simultaneously (using a "delay pattern" where codebook n starts generating n steps after codebook 1).

### Training Data

| Dataset | Hours | Languages | Notes |
|---|---|---|---|
| IndicTTS | Large | 13 Indian languages | IIT Madras TTS corpus |
| LIMMITS | Medium | 13 languages | Microsoft Research India |
| IndicVoices-R | Large | 22 languages | Crowdsourced read speech |
| Rasa | Medium | Tamil, multiple | Conversational Tamil |
| (Others) | - | Multiple | See paper for full list |
| **Total** | **1,806 hours** | **21 languages** | Published at Interspeech 2025 |

### Tamil Specifics

- **Tamil training hours**: 52.25 hours (29,204 utterances)
- **Tamil Native Speaker Score (NSS)**: 75.48% ± 2.18%
- **Tamil voices in model**: 69 voice descriptions available; recommended for transport domain: "Jaya" and "Kavitha"

The Tamil NSS metric means 75.48% of the time, native Tamil speakers rated the synthesized voice as sounding like a native speaker. This is the highest reported NSS for open-source Tamil TTS at time of publishing.

### Model Loading (backend/tts_engine.py)

```python
from transformers import AutoTokenizer, ParlerTTSForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = ParlerTTSForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=TORCH_DTYPE,   # bfloat16 on Ampere+ GPU, float16 otherwise
    attn_implementation="eager"  # NOT "sdpa" or "flash_attention_2"
)
model = model.to(DEVICE)
model.eval()
```

**Why `attn_implementation="eager"`?**

The standard optimized attention implementations (SDPA = Scaled Dot Product Attention, Flash Attention 2) are broken for this specific ParlerTTS model variant. Community reports and the `ai4bharat/indic-parler-tts` model card both note that using SDPA or FA2 with this model produces incorrect attention patterns — the model generates gibberish or crashes. `eager` (PyTorch standard attention) is the only working implementation. This is a known limitation that costs approximately 1.5–2× performance vs SDPA.

**TORCH_DTYPE logic:**
```python
TORCH_DTYPE = (
    torch.bfloat16 if DEVICE.startswith("cuda") and torch.cuda.is_bf16_supported()
    else torch.float16 if DEVICE.startswith("cuda")
    else torch.float32
)
```

- **bfloat16**: Used on Ampere+ GPUs (RTX 3xxx, A100, H100, etc.). bfloat16 has the same dynamic range as float32 (8 exponent bits) but lower precision (7 mantissa bits vs 23 for float32). This is preferred over float16 for LLMs because LLMs are sensitive to large activation values — float16 (5 exponent bits) overflows more easily.
- **float16**: Used on older CUDA GPUs (T4, V100) that don't support bfloat16. Slightly lower stability.
- **float32**: Used on CPU. Full precision but 2× memory and 2× slower than float16.

### Model Memory Footprint

| Precision | Memory for 0.9B model | Notes |
|---|---|---|
| bfloat16 | ~1.8 GB | Preferred on A100/H100 |
| float16 | ~1.8 GB | On T4/V100 |
| float32 | ~3.6 GB | CPU only |

Plus the DAC decoder (~100MB) and flan-t5-large encoder (~300MB at fp16).

Total GPU memory required: approximately **4–6 GB** on float16.

---

## 5. COMPONENT: TEXT NORMALIZATION (backend/normalizer.py)

### Purpose

Convert raw user text containing numbers, codes, abbreviations, and mixed scripts into a form that the ParlerTTS model can pronounce correctly. The normalization is designed for the transport domain.

### Architecture: Three Stages in Sequence

```
Raw Text
    │
    ▼
Stage 1: Pronunciation Override Substitution
    │ Regex word-boundary substitution of 32 known abbreviations
    ▼
Stage 2: {{TAG:VALUE}} Expansion
    │ Structured entity expansion with explicit tag markers
    ▼
Stage 3: Heuristic Fallback (auto-detects untagged entities)
    │ For text without explicit tags
    ▼
Normalized Text (ready for TTS)
```

### Stage 1: Pronunciation Overrides

A dictionary of 32 abbreviations with their spoken forms:

```python
PRONUNCIATION_OVERRIDES = {
    "mins": "minutes",      "min": "minutes",       "sec": "seconds",
    "secs": "seconds",      "hrs": "hours",         "hr": "hour",
    "km": "kilometers",     "kms": "kilometers",    "kg": "kilograms",
    "kgs": "kilograms",     "gm": "grams",          "gms": "grams",
    "ltr": "liters",        "ltrs": "liters",       "amt": "amount",
    "approx": "approximately", "govt": "government", "dept": "department",
    "est": "estimated",     "avg": "average",       "info": "information",
    "nos": "numbers",       "appt": "appointment",  "addr": "address",
    "msg": "message",       "txn": "transaction",   "acct": "account",
    "bal": "balance",       "qty": "quantity",      "ref": "reference",
    "ID": "I D",            "Journey": "ஜர்னி",    # One Tamil override
}
```

Applied via `re.sub()` with word boundary (`\b`) anchors and `re.IGNORECASE` flag. Case-insensitive matching ensures "KM", "km", "Km" all expand to "kilometers".

**The Tamil override "Journey" → "ஜர்னி":** The English word "Journey" appears in Movio's app UI (journey started, journey ended). In Tamil speech context, it should be pronounced as the loanword "ஜர்னி" (zhar-ni) rather than spelled out in English. This is the only Tamil word in the override table; all others produce English expansions.

### Stage 2: {{TAG:VALUE}} Structured Entity Expansion

The `{{TAG:VALUE}}` format provides explicit markup for structured entities. When the client uses this format, there is no ambiguity about what type of entity is being expressed.

**Tag types:**

| Tag | Example Input | Example Output | Method |
|---|---|---|---|
| `{{OTP:4821}}` | `{{OTP:4821}}` | ", four eight two one" | `spell_digits()` |
| `{{PHONE:9876543210}}` | `{{PHONE:9876543210}}` | ", nine eight seven six five four three two one zero" | `spell_digits()` |
| `{{AMOUNT:1250}}` | `{{AMOUNT:1250}}` | "one thousand two hundred fifty" | `indic_num2words(1250, lang="en")` |
| `{{AMOUNT:12.5}}` | `{{AMOUNT:12.5}}` | "twelve point five" | Decimal handling |
| `{{TIME:14:30}}` | `{{TIME:14:30}}` | "two thirty P M" | `spell_time()` |
| `{{TIME:09:00}}` | `{{TIME:09:00}}` | "nine o'clock A M" | Special case: :00 → "o'clock" |
| `{{CODE:TN45AB}}` | `{{CODE:TN45AB}}` | "T N four five A B" | `spell_alnum_code()` |
| `{{NUM:3}}` | `{{NUM:3}}` | "three" | `spell_amount()` |

**`spell_time()` logic:**
```python
def spell_time(hh, mm):
    h = int(hh) % 24
    m = int(mm)
    period = "A M" if h < 12 else "P M"   # Spoken with spaces for clarity
    h12 = h % 12 or 12                     # 0 → 12 for 12AM/12PM
    hour_word = indic_num2words(h12, lang="en")
    if m == 0:
        return f"{hour_word} o'clock {period}"
    elif m < 10:
        return f"{hour_word} oh {indic_num2words(m, lang='en')} {period}"  # "seven oh three"
    else:
        return f"{hour_word} {indic_num2words(m, lang='en')} {period}"
```

**Why "A M" with a space instead of "AM"?** The model's tokenizer treats "AM" as a single token that may be pronounced as a word. "A M" forces it to pronounce each letter separately, producing "ay em" rather than "am" (as in "I am"). This is the correct spoken form for time expressions.

**`spell_alnum_code()` logic:**
```python
def spell_alnum_code(code):
    parts = []
    prev = None
    for ch in code:
        if ch.isdigit():
            parts.append(EN_DIGIT_WORDS[ch])   # '4' → 'four'
            prev = ch
        elif ch.isalpha():
            letter = EN_LETTER_NAMES.get(ch.upper(), ch.upper())  # 'A' → 'A'
            if prev and prev.upper() == ch.upper():
                parts[-1] = f"double {letter}"  # 'AA' → 'double A'
            else:
                parts.append(letter)
            prev = ch
    return " ".join(parts)
```

The "double X" handling: If two consecutive identical letters appear (e.g., "AA" in a booking ID), the second is replaced with "double A" rather than "A A". This produces more natural speech — real humans say "double A" for repeated letters in codes.

**Comma before OTP/PHONE/CODE tags:**
```python
if kind in ("OTP", "PHONE", "CODE"):
    preceding = text[:match.start()].rstrip()
    if preceding and preceding[-1] not in ",:.!?":
        return ", " + expansion   # Prepend comma for natural pause
```
This adds a comma before digit sequences so the TTS produces a brief pause: "Your OTP is, four eight two one" rather than "Your OTP is four eight two one" (no pause). The pause makes digit sequences easier to understand.

### Stage 3: Heuristic Fallback

For text that was NOT tagged by the client (plain text input), the heuristic normalizer auto-detects and expands common entities:

**Indian vehicle registration detection:**
```python
STATE_CODES = "AN|AP|AR|AS|BR|CG|CH|DD|DL|DN|GA|GJ|HP|HR|JH|JK|KA|KL|LA|LD|MH|ML|MN|MP|MZ|NL|OD|OR|PB|PY|RJ|SK|TN|TR|TS|UA|UK|UP|WB"
INDIAN_CAR_RE = re.compile(rf'\b({STATE_CODES})[ -]?([0-9]{{1,2}})[ -]?([A-Za-z]{{1,3}})?[ -]?([0-9]{{1,4}})\b', re.IGNORECASE)
```
Covers all 35 Indian state/UT codes. Format: `TN 45 AB 1234` → `T N four five A B one two three four`.

**BH-series registration detection:**
```python
BH_CAR_RE = re.compile(r'\b([0-9]{2})[ -]?(BH)[ -]?([0-9]{4})[ -]?([A-Za-z]{1,2})\b', re.IGNORECASE)
```
Covers the newer Bharat (BH) series: `24 BH 1234 AB`.

**Alphanumeric code detection:**
```python
ALNUM_CODE_RE = re.compile(r'\b(?=[A-Za-z0-9]{5,12}\b)(?=[A-Za-z0-9]*[A-Za-z])(?=[A-Za-z0-9]*\d)[A-Za-z0-9]{5,12}\b')
```
Detects 5–12 character codes that contain both letters and digits (e.g., booking IDs, PNR codes). Uses three lookaheads: (1) total length 5-12, (2) contains at least one letter, (3) contains at least one digit.

**Time detection:**
```python
TIME_RE = re.compile(r'\b([01]?\d|2[0-3]):([0-5]\d)\b')
```
Matches HH:MM in 24-hour format. Calls `spell_time()`.

**Long digit detection:**
```python
LONG_DIGIT_RE = re.compile(r'\b\d{4,}\b')
```
Any run of 4+ digits → digit-by-digit spelling (treats as OTP/phone/code).

**Short number detection:**
```python
SHORT_NUM_RE = re.compile(r'\b\d{1,3}\b')
```
Numbers 0–999 → spelled out as cardinal words via `indic_num2words`.

**`indic_numtowords` library (version 1.1.0):**
- Supports 22 Indian languages including Tamil (`lang='ta'`) and English (`lang='en'`)
- Installed: `indic_numtowords 1.1.0` (confirmed in `.venv/lib/python3.14/site-packages/`)
- Integers: `indic_num2words(1250, lang="en")` → "one thousand two hundred fifty"
- Decimal: `indic_num2words(12, lang="en") + " point " + digit_by_digit(5)` → "twelve point five"
- **Known Tamil decimal bug**: `indic_num2words(2.5, lang='ta')` raises `KeyError: '.5'`. Workaround: split decimal and spell each digit separately with "புள்ளி" (Tamil for "point").

### Known Bug: Tamil Numbers Always Spoken in English

The normalizer has a `contains_tamil()` function:
```python
TAMIL_RE = re.compile(r'[஀-௿]')
def contains_tamil(text: str) -> bool:
    return bool(TAMIL_RE.search(text))
```

But `spell_amount()` always calls `indic_num2words(n, lang="en")` regardless of whether the input contains Tamil:
```python
def spell_amount(num_str, lang="en"):
    # ...
    result = indic_num2words(int(whole), lang="en")   # BUG: hardcoded "en"
```

For a message like "உங்கள் fare ₹250" ("Your fare is ₹250"), the ₹250 would be expanded to "two hundred fifty" (English) instead of "இருநூற்று ஐம்பது" (Tamil). This is tracked as a known bug in PLAN.md.

---

## 6. COMPONENT: SENTENCE CHUNKING (backend/chunker.py)

### Purpose

Split long input text into chunks of at most 100 characters that can be individually synthesized. Shorter chunks produce faster first-audio delivery (the first chunk is sent before subsequent chunks are synthesized). Merge very short chunks together to avoid awkwardly short utterances.

### Three-Level Cascade

**Level 1: Sentence boundary split**

```python
SENTENCE_SPLIT_RE = re.compile(
    r'(?<!\b[A-Za-z]\.)(?<!\bDr\.)(?<!\bMr\.)(?<!\bMs\.)(?<!\bMrs\.)(?<!\bSt\.)(?<!\bNo\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<=[.!?।])\s+|\n+'
)
```

The negative lookbehind clauses prevent splitting at:
- Single-letter abbreviations: "T. Nagar" (the period after "T" is not a sentence end)
- Title abbreviations: Dr., Mr., Ms., Mrs., St., No.
- Latin abbreviations: e.g., i.e.

The Devanagari danda (।) is included as a sentence terminator for Hindi/Sanskrit fragments.

**Level 2: Long sentence splitting (cascaded)**

If a sentence exceeds `MAX_CHUNK_CHARS = 100`, it is further split using this cascade:

*2a. Punctuation split:*
```python
PUNCT_SPLIT_RE = re.compile(r'(?<=[,;:\-—])\s+')
```
Splits at comma, semicolon, colon, em-dash — but only if the split point is NOT inside a `{{TAG:VALUE}}` span (protected by the `_inside_tag()` function).

*2b. Conjunction split (if 2a didn't help):*
```python
CONJ_SPLIT_RE = re.compile(
    r'\s+(?=(?:and|but|or|because|while|after|before|மற்றும்|ஆனால்|எனவே)\b)',
    re.IGNORECASE
)
```
The zero-width lookahead `(?=...)` ensures the conjunction word is kept at the START of the second part, not left at the end of the first part. This gives each part a more natural beginning.

*2c. Word boundary split (fallback):*
If neither 2a nor 2b produced useful splits, split greedily at word boundaries ensuring each piece is ≤ 100 characters.

**Level 3: Small chunk merge**

After splitting, pieces shorter than `MIN_MERGE_CHARS = 25` characters are merged back into the preceding chunk if the combined length ≤ 100 characters:

```python
MIN_MERGE_CHARS = 25
for piece in pieces:
    candidate = (current + " " + piece) if current else piece
    if not current:
        current = piece
    elif len(piece) < MIN_MERGE_CHARS and len(candidate) <= MAX_CHUNK_CHARS:
        current = candidate   # Merge short piece into previous
    else:
        merged.append(current)
        current = piece
```

**Why MIN_MERGE_CHARS = 25?** Chunks shorter than ~25 characters produce audio too short for the ParlerTTS model to establish proper prosody — the model doesn't have enough context to set pitch, rhythm, and intonation. 25 characters is approximately 2–3 Tamil words, which is the minimum for natural prosody.

### Configuration Values

| Parameter | Value | Meaning |
|---|---|---|
| `MAX_CHUNK_CHARS` | `100` | Maximum characters per TTS chunk |
| `MIN_MERGE_CHARS` | `25` | Merge chunks shorter than this with previous |

These values are imported from `backend/config.py`.

### Protected Tag Spans

```python
protected_spans = [m.span() for m in TAG_RE.finditer(sentence)]
def _inside_tag(pos):
    return any(start <= pos < end for start, end in protected_spans)
```

The chunker identifies all `{{TAG:VALUE}}` spans before splitting. Any potential split point that falls inside a tag span is skipped. This prevents `{{AMOUNT:1250}}` from being split as `{{AMOUNT:12` | `50}}` across two separate chunks.

---

## 7. COMPONENT: TTS ENGINE (backend/tts_engine.py)

### The IndicParlerTTSEngine Class

The TTS engine wraps the ParlerTTS model and implements:
1. Model loading with warmup
2. Description encoder caching (key optimization)
3. Token count estimation
4. Streaming audio generation via `ParlerTTSStreamer`
5. Async bridge from background thread to event loop

### Description Encoder Caching

**The key optimization**: The flan-t5-large text encoder normally runs for every `model.generate()` call. Since `VOICE_DESCRIPTION` never changes during a session, its encoder output can be computed once and reused:

```python
_description_cache = {}  # Module-level cache

def _get_cached_description_hidden(description_text, model, tokenizer, device):
    if description_text not in _description_cache:
        # First call: compute and cache
        inputs = tokenizer(description_text, return_tensors="pt").to(device)
        with torch.no_grad():
            hidden = model.get_text_encoder()(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                return_dict=True
            ).last_hidden_state

            # Apply projection if encoder and decoder have different hidden sizes
            if model.text_encoder.config.hidden_size != model.decoder.config.hidden_size:
                hidden = model.enc_to_dec_proj(hidden)

            # Apply attention mask (zero out padding positions)
            hidden = hidden * inputs.attention_mask[..., None]

        _description_cache[description_text] = hidden.detach()

    # Return fresh wrapper each call (prevent mutation from cross-attention ops)
    return BaseModelOutput(last_hidden_state=_description_cache[description_text].clone())
```

**Why `clone()`?** The ParlerTTS model's cross-attention layers may modify the encoder_outputs tensor in-place during generation (some attention implementations write back attention weights to the key-value cache, which can corrupt the original tensor). `clone()` ensures each generate call gets its own copy.

**Savings**: The flan-t5-large encoder takes approximately 20–50ms to run. By caching it, every generation call after the first saves this cost. Over 1,000 requests with 3 chunks each, this saves approximately 60–150 seconds of compute time.

### Token Count Estimation

```python
CHARS_PER_SEC = 13.0   # Characters of Tamil/Tanglish text per second of speech

estimated_seconds = len(prompt_clean) / CHARS_PER_SEC
min_new_tokens = max(10, int(frame_rate * estimated_seconds * 0.85))
estimated_max_tokens = int(frame_rate * estimated_seconds * 1.6)
```

**Why 13.0 chars/second?** This was empirically derived from Tamil/Tanglish speech. Tamil script is more compact than Latin — a typical Tamil word like "வந்துவிட்டது" (12 chars) takes about 0.8–1.0 seconds to say. English words like "pickup" (6 chars) take about 0.5 seconds. The average across mixed Tamil/English text works out to ~13 characters per second of speech.

**Why 0.85 for min_new_tokens?** The model may end early if it generates an end-of-sequence token. A minimum token count prevents premature truncation — the model is forced to generate at least 85% of the estimated duration. This prevents short, clipped output.

**Why 1.6 for max_new_tokens?** The model may "hallucinate" extra tokens beyond the actual speech (this happens especially with repetitive content). A maximum token count caps this. 1.6× gives 60% headroom for natural variation in speaking pace without allowing indefinite generation.

**frame_rate:**
```python
frame_rate = model.audio_encoder.config.frame_rate
# = 44100 / 512 ≈ 86.13 tokens/second
```

512 is the product of the DAC upsampling ratios: `8 × 8 × 4 × 2 = 512`.

### The ParlerTTSStreamer

`ParlerTTSStreamer` is a custom subclass of HuggingFace's `BaseStreamer`:

```python
class ParlerTTSStreamer(BaseStreamer):
    def __init__(self, model, device, play_steps_in_s=0.3):
        self.device = device
        self.model = model
        self.frame_rate = model.audio_encoder.config.frame_rate  # ~86.13
        self.play_steps = int(self.frame_rate * play_steps_in_s)  # int(86.13 * 0.3) ≈ 26
        self.stride = self.model.decoder.config.num_codebooks * (self.play_steps - 1)
        self.token_cache = None
        self.to_yield = 0
        self.audio_queue = queue.Queue()
```

**How it works:**
1. `model.generate()` calls `streamer.put(token)` after generating each step (1 Euler step = all 9 codebooks for one time frame)
2. The streamer accumulates tokens in `self.token_cache`
3. Every `self.play_steps = 26` tokens, it calls `apply_delay_pattern_mask()` which runs the DAC decoder on the accumulated token cache
4. The decoded audio (float32 numpy array at 44.1kHz) is placed in `self.audio_queue`
5. `streamer.end()` is called when generation completes, placing a sentinel value in the queue

**Why 26 steps (0.3 seconds)?**

`play_steps_in_s = 0.3` was chosen by experimentation. At 86.13 tokens/sec, 26 tokens represent 302ms of audio. This is the sweet spot for the streaming chunk size:
- Too small (e.g., 10 tokens = 116ms): DAC decoder runs too frequently, adding overhead; very short chunks have poor quality
- Too large (e.g., 100 tokens = 1.16s): Users wait too long between audio chunks; perceived latency is high
- 26 tokens (302ms): Smooth streaming, reasonable DAC decoder frequency

**The stride calculation:**
```python
self.stride = self.model.decoder.config.num_codebooks * (self.play_steps - 1)
# = 9 * (26 - 1) = 9 * 25 = 225
```

The stride controls overlap handling. The DAC codec uses a "delay pattern" where each of the 9 codebooks starts one step after the previous. To avoid double-decoding overlapping regions, each new audio chunk starts `stride` tokens into the current token cache. The first 225 tokens of each new chunk overlap with the previous chunk's tail and are discarded from the audio output.

### The Async Bridge

The model runs in a background thread (to avoid blocking the event loop), but the WebSocket handler is async. The bridge:

```python
# In the background thread:
async def _enqueue(audio):
    await asyncio.sleep(0)  # yield to event loop
    async_queue.put_nowait(audio)

loop.call_soon_threadsafe(
    loop.create_task,
    asyncio.coroutine(_enqueue)(audio_piece)
)
```

Or more simply:
```python
loop.call_soon_threadsafe(async_queue.put_nowait, audio_piece)
```

The WebSocket handler `async for` over the queue:
```python
async for audio_piece in async_queue:
    if audio_piece is None:  # Sentinel: generation complete
        break
    pcm16 = (np.clip(audio_piece, -1.0, 1.0) * 32767).astype(np.int16)
    await websocket.send_bytes(pcm16.tobytes())
```

### GPU Warmup

After loading, the engine performs two warmup inferences:

```python
warmup_texts = ["வணக்கம்.", "Hello, welcome."]
for text in warmup_texts:
    # Run short generation (max_new_tokens=50) to:
    # 1. Trigger JIT compilation and CUDA kernel caching
    # 2. Warm up flan-t5-large encoder (first call is slow)
    # 3. Cache the description encoder output
    await synthesize_short(text, max_new_tokens=50)
```

Why two warmup texts? One Tamil, one English — this ensures the tokenizer handles both scripts during warmup, and both model branches are warm before real traffic arrives.

### Generation Parameters

```python
model.generate(
    encoder_outputs=cached_description_output,    # Pre-computed voice description
    prompt_input_ids=tokenized_speech_text,        # Text to synthesize
    do_sample=True,           # Sampling mode (not greedy)
    temperature=1.0,          # Full temperature (no reduction)
    min_new_tokens=...,       # Prevent early truncation
    max_new_tokens=...,       # Prevent hallucination runoff
    streamer=streamer,        # ParlerTTSStreamer for real-time decode
)
```

**Why `temperature=1.0`?**

Temperature controls randomness in token sampling. Temperature < 1.0 makes the distribution "sharper" (less random, more deterministic). An earlier version used `temperature=0.3`, which produced very consistent audio but slightly stilted, over-smooth prosody. `temperature=1.0` (the default/full-temperature setting) was restored because:
1. Natural speech has inherent variability — even the same sentence sounds slightly different each time a human says it
2. `temperature=0.3` was causing micro-stutters on certain Tamil conjunctions
3. Combined with `seed=42` (fixed), `temperature=1.0` still produces reproducible audio (seed determines the random stream)

**Why `seed=42`?**
```python
torch.manual_seed(GENERATION_SEED)   # GENERATION_SEED = 42
torch.cuda.manual_seed(GENERATION_SEED)
```
A fixed seed ensures that for the same text + voice description, the model always produces the same audio output. This is required for the cache to work correctly — if the same text produced different audio each time, cached audio would be inconsistent with fresh audio.

**Why NO `top_k` or `repetition_penalty`?**

Earlier versions used `top_k` and `repetition_penalty=1.15`. These were removed because:
- `top_k`: Limits the candidate token pool, which can cut off valid prosodic patterns. For DAC codec tokens, a token that represents a repeated sound (e.g., a sustained vowel) may have high probability and should not be excluded by top_k.
- `repetition_penalty=1.15`: The DAC codec legitimately repeats tokens for sustained sounds, vowels, and silence. Penalizing repetition caused pitch shifts and micro-stutters at syllable boundaries — the penalty inadvertently discouraged the model from sustaining natural phonemes.

---

## 8. COMPONENT: FASTAPI WEBSOCKET SERVER (backend/server.py)

### Server Details

- **Framework**: FastAPI (latest)
- **ASGI server**: Uvicorn with uvloop
- **Port**: 8000 (or `PORT` environment variable)
- **Static files**: Served by FastAPI StaticFiles (Note: PLAN.md says use nginx/Caddy in production for this)

### Concurrency Architecture

**Single asyncio.Lock for all generation:**
```python
_generation_lock = asyncio.Lock()
```

The lock wraps the entire per-chunk generation loop. Only one `model.generate()` can run at a time. This prevents:
1. GPU memory corruption (PyTorch global state)
2. Two threads writing to the same CUDA context simultaneously
3. KV-cache corruption from interleaved generation

**This limits concurrent users to 1 active synthesis + N waiting in the event loop queue.**

Why not run multiple streams in parallel? The 0.9B ParlerTTS model uses ~4–6GB GPU memory. On a typical deployment GPU (T4 = 16GB), running 2 concurrent streams would use 8–12GB and would cause CUDA OOM (out of memory) errors due to KV cache allocation during generation.

### Main Synthesis Protocol (`/ws` endpoint)

**Complete message sequence for a 3-chunk utterance:**

```
CLIENT → SERVER:
{type:"synthesize", text:"Unga OTP enna? Driver enga irukaru? Booking ID nu sollunge."}

SERVER → CLIENT:
{type:"chunks", raw_chunks:["Unga OTP enna?","Driver enga irukaru?","Booking ID nu sollunge."],
                chunks:["Unga OTP enna?","Driver enga irukaru?","Booking I D nu sollunge."],
                num_chunks:3}

--- ACQUIRE LOCK ---

{type:"chunk_start", index:0, cached:false}
[binary PCM16 bytes: frame 1 of chunk 0]
[binary PCM16 bytes: frame 2 of chunk 0]
[binary PCM16 bytes: frame 3 of chunk 0]
{type:"chunk_end", index:0, end_punct:"?"}

{type:"chunk_start", index:1, cached:false}
[binary PCM16 bytes: frame 1 of chunk 1]
...
{type:"chunk_end", index:1, end_punct:"?"}

{type:"chunk_start", index:2, cached:false}
...
{type:"chunk_end", index:2, end_punct:"."}

--- RELEASE LOCK ---

{type:"done"}
```

**`end_punct` field**: Reports the last character of the chunk (`.`, `?`, `!`, `,`, or empty). The frontend uses this to determine the pause duration before the next sentence begins.

### Audio Post-Processing for Cache

While streaming, audio pieces are sent raw (just `np.clip` + PCM16). After all pieces for a chunk are collected, post-processing is applied to produce the cached version:

```python
def postprocess_full_audio(samples, sample_rate):
    # Step 1: High-pass filter at 60Hz (remove DC offset and rumble)
    sos = scipy.signal.butter(4, 60, 'hp', fs=sample_rate, output='sos')
    samples = scipy.signal.sosfilt(sos, samples)

    # Step 2: Silence trim from tail
    window = 256
    rms_frames = [np.sqrt(np.mean(samples[i:i+window]**2))
                  for i in range(0, len(samples)-window, window)]
    last_active = len(rms_frames) - 1
    for i in range(len(rms_frames)-1, -1, -1):
        if rms_frames[i] > SILENCE_THRESHOLD:  # 0.005 RMS
            last_active = i
            break
    tail_guard = int(0.04 * sample_rate)  # 40ms guard after last active frame
    keep_samples = min((last_active + 1) * window + tail_guard, len(samples))
    samples = samples[:keep_samples]

    # Step 3: Fade-out (480 samples ≈ 10.9ms)
    if len(samples) >= FADE_SAMPLES:  # FADE_SAMPLES = 480
        fade = np.linspace(1.0, 0.0, FADE_SAMPLES)
        samples[-FADE_SAMPLES:] *= fade

    # Step 4: LUFS normalization to -13 LUFS
    meter = pyloudnorm.Meter(sample_rate)
    loudness = meter.integrated_loudness(samples)
    if loudness > -70.0:   # Only normalize non-silent audio
        target_gain = 10 ** ((TARGET_LUFS - loudness) / 20)  # dB to linear gain
        samples = samples * target_gain

    # Step 5: Final clip to prevent overflow
    return np.clip(samples, -1.0, 1.0)
```

**Why the High-Pass Filter at 60Hz?**
The DAC codec occasionally introduces low-frequency artifacts (sub-bass rumble, DC offset) that are not audible on headphones but cause distortion on phone speakers and can trigger audio processing pipelines to clip. A 4th-order Butterworth high-pass at 60Hz removes everything below 60Hz (inaudible for speech) while preserving all speech content (fundamental frequencies of Tamil speech start at ~85–300Hz for the pitch range). Order 4 is chosen because:
- Order 2: gentle roll-off, some sub-bass leaks through
- Order 4: steeper roll-off, good rejection without phase distortion issues
- Order 8+: too much phase distortion, causes pre-ringing

**Why LUFS Normalization to -13 LUFS?**
LUFS (Loudness Units relative to Full Scale) is a perceptual loudness measure standardized by ITU-R BS.1770-4. The -13 LUFS target:
- Is 3 LUFS louder than the -16 LUFS target in PLAN.md (adjusted after testing — the transport voice agent needs to be louder to cut through ambient noise in a moving cab)
- Ensures consistent volume across different utterances (some Tamil words produce louder synthesis than others)
- Is within the loudness range of major streaming platforms (Spotify: -14 LUFS, YouTube: -14 LUFS)

**Why 40ms tail guard?**
After finding the last active (above-threshold) audio frame, we keep 40ms beyond it. This prevents cutting off consonant releases and word-final nasal sounds that gradually fade below the RMS threshold before they're perceptually complete.

**Why 480-sample fade-out?**
480 samples at 44.1kHz = ~10.9ms. This is long enough to avoid a hard click at the end of the audio (which would happen with an abrupt cutoff) but short enough not to audibly affect the last word. 10ms is the standard fade length for anti-click purposes in professional audio production.

### Conversation WebSocket Protocol (`/ws/conversation` endpoint)

The conversation endpoint is designed for phone-call-like interactions where the user can interrupt the AI while it's speaking.

**Generation ID Protocol:**

Every binary frame carries a 4-byte big-endian generation ID as a header:
```python
gen_id_bytes = gen_id.to_bytes(4, byteorder='big')
frame_with_header = gen_id_bytes + pcm16_bytes
await websocket.send_bytes(frame_with_header)
```

The browser discards frames whose generation ID doesn't match the current expected ID:
```javascript
const frameGenId = new DataView(event.data).getUint32(0, false);  // big-endian
if (frameGenId !== currentGenId) return;  // Stale frame, discard
const pcmData = event.data.slice(4);      // Remove 4-byte header, use audio
```

**Why is the generation ID needed?** When the user interrupts, the server has usually already sent several audio frames for the current generation into the WebSocket buffer. These frames are "in flight" and will arrive at the browser even after the interrupt is processed. Without the generation ID, the browser would play these stale frames from the interrupted generation before playing the new generation — producing a jumbled mix.

**AbortCriteria for mid-generation cancellation:**
```python
class AbortCriteria:
    def __init__(self):
        self._event = threading.Event()

    def __call__(self, input_ids, scores, **kwargs):
        return self._event.is_set()   # Returns True to stop generation

    def abort(self):
        self._event.set()
```

PyTorch's `model.generate()` calls the stopping criteria after each decoding step. If `abort()` is called from the async handler (via `asyncio.create_task`), the event is set, and the next iteration of `model.generate()` calls the stopping criteria which returns `True`, causing the generation loop to exit cleanly.

**Why not just kill the thread?** Killing a Python thread mid-operation is not safe — it could leave the PyTorch CUDA context in an inconsistent state, corrupt GPU memory, or deadlock the CUDA stream. The `AbortCriteria` approach allows the generation to stop at a clean token boundary.

---

## 9. AUDIO POST-PROCESSING PIPELINE

### In-stream vs. Post-stream Processing

Two separate processing paths exist:

**In-stream (sent to client immediately):**
1. `np.clip(audio_piece, -1.0, 1.0)` — prevent overflow
2. `(audio_piece * 32767).astype(np.int16).tobytes()` — convert to PCM16
3. → `websocket.send_bytes()`

**Post-stream (applied to full chunk before caching):**
1. High-pass filter 60Hz (Butterworth order 4)
2. Silence trim from tail (256-sample RMS window, threshold 0.005, 40ms guard)
3. Fade-out (480 samples linear ramp)
4. LUFS normalization to -13 LUFS (pyloudnorm, ITU-R BS.1770-4)
5. `np.clip(-1.0, 1.0)`
6. → Cache as PCM16 bytes

**Why different processing for in-stream vs cached?**
- **In-stream**: Must be minimal latency — any additional processing adds delay
- **Cached**: No latency constraint — the full post-processing runs before caching, so future cache hits include the full quality treatment

### Frontend Silence Trimming (JavaScript)

The browser also applies silence trimming to remove leading silence before the first phoneme:

```javascript
function trimAudioSilence(float32Array, sampleRate) {
    const SILENCE_THRESHOLD = 0.0008;   // RMS threshold
    const PAD_MS = 80;                   // Keep 80ms before first detected sound
    const padSamples = Math.round(PAD_MS * sampleRate / 1000);  // 80ms × 44100 = 3528 samples

    const windowSize = 256;
    let firstActiveWindow = 0;
    for (let i = 0; i < float32Array.length - windowSize; i += windowSize) {
        let sumSq = 0;
        for (let j = 0; j < windowSize; j++) sumSq += float32Array[i+j] ** 2;
        const rms = Math.sqrt(sumSq / windowSize);
        if (rms > SILENCE_THRESHOLD) {
            firstActiveWindow = i;
            break;
        }
    }
    const startSample = Math.max(0, firstActiveWindow - padSamples);
    const trimmed = float32Array.slice(startSample);

    // Apply 5ms fade-in to prevent click
    const fadeIn = Math.min(Math.round(0.005 * sampleRate), trimmed.length / 8);
    for (let i = 0; i < fadeIn; i++) trimmed[i] *= i / fadeIn;

    return trimmed;
}
```

**Why 0.0008 as silence threshold?** This is empirically determined for ParlerTTS output. ParlerTTS sometimes generates a few milliseconds of very low-amplitude noise before the speech begins (a DAC codec artifact). 0.0008 RMS (~-62dBFS) reliably distinguishes this pre-speech noise from actual speech onset.

**Why 80ms pad?** The first 80ms before the detected speech onset is preserved. This catches the very start of consonant plosives (like 'p', 'b', 't') whose onset is below the threshold but whose voice burst needs to be included. Without the pad, plosive onsets would be clipped.

---

## 10. THE LRU AUDIO CACHE

### Implementation

```python
from collections import OrderedDict
import hashlib, json

class LRUAudioCache:
    def __init__(self, maxsize=200):
        self._cache = OrderedDict()
        self._maxsize = maxsize  # 200 entries

    def get(self, key):
        if key in self._cache:
            self._cache.move_to_end(key)  # Mark as recently used
            return self._cache[key]
        return None

    def put(self, key, value):
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)  # Evict LRU (oldest)
        self._cache[key] = value
```

### Cache Key Generation

```python
_CONFIG_FINGERPRINT = hashlib.sha256(
    json.dumps({
        "model": MODEL_ID,
        "voice": VOICE_DESCRIPTION,
        "temperature": TEMPERATURE
    }, sort_keys=True).encode()
).hexdigest()[:8]   # First 8 hex chars

def cache_key(text: str) -> str:
    return hashlib.sha256(f"{text}|{_CONFIG_FINGERPRINT}".encode()).hexdigest()
```

**Why include the config fingerprint in the key?**
If any of these change — the model version, the voice description, the temperature — the synthesized audio would sound different. The 8-character fingerprint ensures cached audio from one configuration is never served under a different configuration. The 8-character truncation provides 32 bits of collision resistance, which is more than sufficient for a 200-entry cache.

### Cache Statistics

Maximum memory usage: `200 entries × ~431KB/entry = ~84MB`

Where does 431KB come from?
- A typical 3-second Tamil sentence: 3.0s × 44100 Hz × 2 bytes/sample (int16) = **264,600 bytes ≈ 258KB**
- Longer sentences (up to 5 seconds): 5.0s × 44100 × 2 = **441,000 bytes ≈ 431KB**
- This is a rough upper estimate per entry

84MB total is within the memory budget of any deployment GPU.

### Hit Latency

- **Cache hit**: ~0.5ms (OrderedDict lookup + move_to_end)
- **Cache miss + synthesis**: ~300–500ms first audio

TTFA for a cache hit is therefore dominated by the text normalization + chunking time (5–10ms total), making the total TTFA ~10–15ms for cached responses.

---

## 11. FRONTEND: MAIN TTS DEMO (frontend/index.html)

**File size**: ~1,867 lines
**Technology**: Vanilla JavaScript + Web Audio API + Canvas API

### Key JavaScript Components

**1. Auto-Tagger (`autoTagText()` function, ~150 lines)**

The same tag-expansion logic as the Python normalizer, implemented in JavaScript. Applied to user input before sending to the server.

Priority order (highest to lowest):
1. Preserve existing `{{TAG:VALUE}}` tags (don't re-tag)
2. Duration expressions (5 minutes, 3 mins, 2 hrs)
3. Time of day (HH:MM, HH:MM am/pm)
4. Title abbreviations (Dr., Mr., T. Nagar)
5. Indian vehicle registrations (all 35 state codes)
6. BH-series registrations
7. Booking/Order/PNR IDs (keyword-triggered: "booking id", "order id", "PNR")
8. Standalone alphanumeric codes (5-12 chars, mixed letters+digits)
9. OTP/PIN patterns (keyword-triggered: "OTP", "PIN", "code", "password")
10. Indian phone numbers (10-digit starting with 6-9, or +91 prefix)
11. Currency (₹, Rs., INR, $, USD, ரூபாய்)
12. Amount keywords (fare, cost, total, balance, amount, விலை, கட்டணம்)
13. Any remaining bare numbers → `{{AMOUNT:N}}`

The frontend auto-tagger runs before sending, and the backend normalizer runs again on the server side. This dual application ensures even if the frontend is bypassed (e.g., direct API calls), the server still normalizes correctly.

**2. WebSocket Client**

```javascript
function connect() {
    ws = new WebSocket("ws://host/ws");
    ws.binaryType = "arraybuffer";

    ws.onmessage = (event) => {
        if (event.data instanceof ArrayBuffer) {
            handleAudioFrame(event.data);   // Binary: audio data
        } else {
            handleControlFrame(JSON.parse(event.data));  // JSON: control
        }
    };
}
```

Auto-reconnect with 3000ms delay if connection drops.

**3. Audio Scheduling**

Each sentence's audio arrives as multiple binary PCM16 frames. The browser collects these into `currentChunkPieces[]`, then on `chunk_end`, assembles them and schedules playback:

```javascript
function scheduleSentence(endPunct) {
    // Merge all frames into single Float32Array
    let totalLength = currentChunkPieces.reduce((sum, p) => sum + p.length, 0);
    let merged = new Float32Array(totalLength);
    let offset = 0;
    for (const piece of currentChunkPieces) {
        merged.set(piece, offset);
        offset += piece.length;
    }

    // Trim leading silence, apply 5ms fade-in
    const trimmed = trimAudioSilence(merged, sampleRate);

    // Create Web Audio buffer and schedule
    const buffer = audioCtx.createBuffer(1, trimmed.length, sampleRate);
    buffer.copyToChannel(trimmed, 0);
    const source = audioCtx.createBufferSource();
    source.buffer = buffer;

    // 3ms gain ramp to prevent click at source start
    const gainNode = audioCtx.createGain();
    gainNode.gain.setValueAtTime(0, nextBoundary);
    gainNode.gain.linearRampToValueAtTime(1.0, nextBoundary + 0.003);
    source.connect(gainNode).connect(audioCtx.destination);

    source.start(nextBoundary);
    const duration = trimmed.length / sampleRate;
    nextBoundary += duration;

    // Add inter-sentence pause based on punctuation
    const PAUSES = { ',': 0.08, '?': 0.20, '!': 0.18, '.': 0.15 };
    nextBoundary += PAUSES[endPunct] ?? 0.15;

    source.onended = () => { /* mark chunk done */ };
}
```

**4. Real-time Frequency Visualizer**

Canvas-based bar graph using Web Audio API AnalyserNode:
- FFT size: 128 bins
- Smoothing time constant: 0.8
- Bin height scaled logarithmically to frequency
- Updated at 60fps via `requestAnimationFrame`

**5. Full Audio Player**

After all chunks are received, the frontend assembles a complete audio waveform with 100ms inter-sentence gaps and draws:
- Waveform canvas (peak amplitude per pixel)
- Playback cursor (scrubbing via click)
- WAV download button (PCM16 WAV blob)

**6. TTFA Measurement**

```javascript
let requestStart = performance.now();  // Set when "synthesize" sent
let ttfa = null;

function handleAudioFrame(buffer) {
    if (ttfa === null) {
        ttfa = performance.now() - requestStart;
        metrics.ttfa = ttfa;    // Display in metrics HUD
    }
    // ... process audio
}
```

TTFA is measured client-side as the time from sending the `synthesize` message to receiving the first binary (audio) frame. This includes network round-trip time, server processing, and first audio chunk synthesis.

**7. Preset Phrases**

The UI provides preset buttons for common transport domain phrases:
- "Your OTP is {{OTP:4821}}"
- "Driver is {{AMOUNT:2}} minutes away"
- "Your fare is {{AMOUNT:250}} rupees"
- "Vehicle number {{CODE:TN45AB1234}}"
- "Booking confirmed at {{TIME:14:30}}"

---

## 12. FRONTEND: CONVERSATION INTERRUPT DEMO (frontend/conversation.html)

**File size**: ~907 lines

This frontend demonstrates a more advanced use case: an ongoing phone-call-like conversation where the AI can be interrupted mid-sentence.

### Key Differences from Main UI

**1. Sample Rate Pinning**
```javascript
audioCtx = new AudioContext({ sampleRate: 44100 });
```
Forces the AudioContext to 44.1kHz — the DAC codec's native rate. Without this, the browser's OS default (typically 48kHz) would force a resampler, creating subtle pitch variations and seams at chunk boundaries during multi-chunk streaming.

**2. Master Gain Node**
```javascript
masterGain = audioCtx.createGain();
masterGain.connect(audioCtx.destination);
// All sources connect to masterGain, not directly to destination
```
A single master gain node allows instant volume control and fade-out for all playing audio simultaneously (critical for interrupt).

**3. Audio End Time Tracking**
```javascript
let audioEndTime = 0;
// Updated whenever a new audio piece is scheduled:
audioEndTime = nextPlayTime + (samples.length / sampleRate);
```

Used to detect if audio is currently playing:
```javascript
const isAudioPlaying = audioCtx && audioCtx.currentTime < audioEndTime;
```

**4. Interrupt Detection and Handling**
```javascript
function sendMessage(text) {
    const isSpeaking = currentGenId !== null;
    const isAudioPlaying = audioCtx && audioCtx.currentTime < audioEndTime;

    if (isSpeaking || isAudioPlaying) {
        // INTERRUPT: fade out current audio immediately
        masterGain.gain.linearRampToValueAtTime(0, audioCtx.currentTime + 0.05);  // 50ms fade
        activeSources.forEach(src => src.stop(audioCtx.currentTime + 0.06));
        activeSources = [];
        nextPlayTime = audioCtx.currentTime + 0.06;

        currentGenId++;    // Increment gen ID — server will tag new generation
        ws.send(JSON.stringify({ type: "interrupt", text: autoTagText(text) }));
    } else {
        ws.send(JSON.stringify({ type: "speak", text: autoTagText(text) }));
    }
}
```

**5. Generation ID Header Parsing**
```javascript
ws.onmessage = async (event) => {
    if (event.data instanceof ArrayBuffer) {
        const view = new DataView(event.data);
        const frameGenId = view.getUint32(0, false);   // big-endian

        if (frameGenId !== currentGenId) return;       // Discard stale frames

        // Extract audio (skip 4-byte header)
        const pcm16 = new Int16Array(event.data, 4);
        const float32 = new Float32Array(pcm16.length);
        for (let i = 0; i < pcm16.length; i++) float32[i] = pcm16[i] / 32768;

        scheduleAudio(float32);
    }
};
```

**6. Audio Scheduling Details**

Scheduling with 150ms initial lookahead and 20ms minimum ahead:
```javascript
function scheduleAudio(float32Data) {
    const now = audioCtx.currentTime;
    const LOOKAHEAD = 0.15;    // 150ms initial lookahead
    const MIN_AHEAD = 0.02;    // 20ms minimum ahead

    if (nextPlayTime < now + MIN_AHEAD) {
        nextPlayTime = now + LOOKAHEAD;  // Reset lookahead if too close
    }

    const buffer = audioCtx.createBuffer(1, float32Data.length, audioCtx.sampleRate);
    buffer.copyToChannel(float32Data, 0);
    const source = audioCtx.createBufferSource();
    source.buffer = buffer;
    source.connect(masterGain);    // Connect to master gain (not directly to destination)
    source.start(nextPlayTime);

    activeSources.push(source);
    nextPlayTime += float32Data.length / audioCtx.sampleRate;
    audioEndTime = nextPlayTime;  // Update end time tracker
}
```

**Why 150ms initial lookahead?** The Web Audio API scheduler needs some buffer time to queue audio to the OS audio driver. Too small a lookahead causes audio dropouts when the JavaScript thread is briefly blocked by garbage collection or other tasks. 150ms is generous enough to prevent all dropout issues while still feeling responsive.

**Why 20ms minimum ahead?** Even for subsequent chunks that arrive late (e.g., network jitter), we never schedule audio less than 20ms in the future. Scheduling too close to `now` can cause the chunk to miss its scheduled start, resulting in a gap.

---

## 13. LANGUAGE SUPPORT: TAMIL, TANGLISH, AND ENGLISH

### Tamil (Pure Script)

**Input format:** Tamil Unicode (U+0B80–U+0BFF)

**Detection:** `TAMIL_RE = re.compile(r'[஀-௿]')` — covers entire Unicode Tamil block

**Model capability:** ai4bharat/indic-parler-tts was trained on 52.25 hours of Tamil speech. The model handles native Tamil script natively without any transliteration.

**Tamil-specific features in normalizer:**
- Tamil Unicode block detection (`contains_tamil()`)
- Tamil conjunction words in chunker: `மற்றும்` (and), `ஆனால்` (but), `எனவே` (therefore)
- One Tamil pronunciation override: "Journey" → "ஜர்னி"

**Tamil-specific features in JavaScript tagger:**
- Tamil time units: `நிமிடங்கள்`, `நிமிடம்`, `வினாடிகள்`, `மணி`
- Tamil currency: `ரூபாய்`, `ரூ`
- Tamil OTP/phone keywords: `ரகசிய எண்`, `கடவுச்சொல்`, `தொடர்புக்கு`, `தொலைபேசி`, `அழைக்க`
- Tamil amount keywords: `விலை`, `கட்டணம்`, `தொகை`, `வாடகை`, `செலவு`, `பணம்`

### Tanglish (Romanized + Code-Mixed)

**Input format:** Either pure Romanized Tamil ("Unga cab vandhuchu") or mixed with native Tamil ("உங்கள் pickup location எங்கே?")

**Model capability:** ai4bharat/indic-parler-tts was trained on code-mixed Indian language data. The model handles Tamil+English mixing better than most single-language TTS models. However, it is not perfectly tuned for Tanglish — the PLAN.md includes Phase 5 for IndicF5 voice cloning as a quality upgrade.

**No dedicated transliteration pipeline:** Unlike Approach 1 (movio/), this approach does NOT have a Tanglish router or en2ta transliterator. Code-mixed text is passed directly to the model. The model's ability to handle this varies:
- Common Tanglish: Works well ("Unga OTP enna?" → natural synthesis)
- Less common romanizations: May produce incorrect pronunciation
- This is a known quality gap tracked for Phase 4 (indic-nlp-library) in PLAN.md

### English (Transport Domain)

**Input format:** Pure English text

**Model capability:** The model has extensive multilingual English-Indic training. English text is synthesized with a South Indian accent (Tamil accent), which is appropriate for the target user base.

**Normalization support:**
- All number types (cardinal, ordinal, decimal)
- Times (12-hour and 24-hour formats)
- Vehicle registrations (all 35 Indian state codes)
- Alphanumeric codes and booking IDs
- Currency (₹, $, Rs.)

### Known Language Gaps

| Gap | Affected Language | Workaround | Priority |
|---|---|---|---|
| Tamil numbers spoken in English | Tamil | Use {{AMOUNT:N}} tag | High |
| Tamil numeral chars (௧-௯) not converted | Tamil | Avoid using Tamil numeral chars | Medium |
| indic_numtowords decimal crash in Tamil | Tamil | Manual digit-by-digit | Medium |
| Tamil abbreviations not expanded (செ.மீ, கி.மீ) | Tamil | Manual substitution | Low |
| Romanized Tamil pronunciation inconsistency | Tanglish | Phase 4: indic-nlp-library | Medium |
| No Unicode normalization (ZWJ/ZWNJ) | Tamil | Phase 4: indic-nlp-library | Low |

---

## 14. ALL CONFIGURATION VALUES AND NUMBERS

**File:** `backend/config.py`

### Complete Configuration Reference

| Parameter | Value | Type | Meaning |
|---|---|---|---|
| `DEVICE` | `"cuda:0"` or `"cpu"` | str | GPU device (auto-detected) |
| `TORCH_DTYPE` | `bfloat16` / `float16` / `float32` | dtype | Model precision (tiered by GPU capability) |
| `MODEL_ID` | `"ai4bharat/indic-parler-tts"` | str | HuggingFace model identifier |
| `VOICE_DESCRIPTION` | `"Jaya speaks with a slightly low-pitched voice at a moderate pace. The recording is very high quality, clear and close-sounding, with no background noise."` | str | 28-word natural language voice description |
| `CHARS_PER_SEC` | `13.0` | float | Characters per second estimate for token budgeting |
| `MAX_CHUNK_CHARS` | `100` | int | Maximum characters per TTS chunk (chunker) |
| `PLAY_STEPS_IN_S` | `0.3` | float | Streaming chunk interval: every 0.3s ≈ 26 tokens |
| `GENERATION_SEED` | `42` | int | Fixed random seed for reproducible/cacheable audio |
| `TEMPERATURE` | `1.0` | float | Token sampling temperature (1.0 = natural distribution) |
| `CACHE_MAX_ENTRIES` | `200` | int | LRU cache maximum entries (~84MB max) |
| `PORT` | `8000` | int | HTTP server port (env `PORT` overrides) |

### Audio Processing Constants (backend/server.py)

| Constant | Value | Meaning |
|---|---|---|
| `SILENCE_THRESHOLD` | `0.005` | RMS threshold for tail silence detection (~-46dBFS) |
| `FADE_SAMPLES` | `480` | Fade-out length at 44.1kHz = 10.9ms |
| `TARGET_LUFS` | `-13.0` | LUFS normalization target (ITU-R BS.1770-4) |
| `_XFADE_SAMPLES` | `256` | Crossfade at DAC chunk boundaries = 5.8ms |
| HPF cutoff | `60 Hz` | Butterworth high-pass filter cutoff |
| HPF order | `4` | Butterworth filter order |
| HPF gate | `-70.0 LUFS` | Skip normalization if signal below this (silence) |

### Audio Scheduling Constants (frontend)

| Constant | Value | Meaning |
|---|---|---|
| Silence threshold (JS) | `0.0008` | RMS threshold for leading silence trim (~-62dBFS) |
| Silence pad (JS) | `80ms` | Keep 80ms before first detected sound |
| Fade-in (JS) | `5ms` or `length/8` | Anti-click fade at audio start |
| Comma pause | `80ms` | Inter-sentence gap after comma |
| Question mark pause | `200ms` | Inter-sentence gap after ? |
| Exclamation pause | `180ms` | Inter-sentence gap after ! |
| Default period pause | `150ms` | Inter-sentence gap after . |
| Full audio sentence gap | `100ms` | Gap in assembled full audio player |
| AudioContext lookahead | `150ms` | Initial Web Audio API scheduling lookahead |
| Min ahead scheduling | `20ms` | Minimum time ahead for scheduling |
| Interrupt fade-out | `50ms` | Master gain fade on interrupt |
| Interrupt fade-in | `20ms` | New generation onset after interrupt |
| Post-interrupt silence | `60ms` | Gap before new generation audio starts |
| Volume default | `3.0×` | Default gain on main TTS UI |

### Model Architecture Constants (from DAC codec config)

| Constant | Value | Source |
|---|---|---|
| DAC sample rate | `44,100 Hz` | DAC codec (`ylacombe/dac_44khZ_8kbps`) |
| DAC upsampling ratios | `[8, 8, 4, 2]` | Product = 512 = hop_length |
| Number of codebooks | `9` | RVQ depth |
| Codebook size | `1,024` | Entries per codebook |
| Frame rate | `44100 / 512 ≈ 86.13` tokens/sec | Per codebook |
| play_steps (derived) | `int(86.13 × 0.3) = 26` | Tokens per streamer chunk |
| stride (derived) | `9 × (26 - 1) = 225` | Token overlap to skip |

### Token Budget Formulas

For text with `n` characters:
```
estimated_seconds = n / 13.0
frame_rate = 86.13
min_new_tokens = max(10, int(86.13 × estimated_seconds × 0.85))
max_new_tokens = int(86.13 × estimated_seconds × 1.6)
```

Example for "Unga OTP enna vandhuchu" (25 characters):
```
estimated_seconds = 25 / 13.0 = 1.92s
min_new_tokens = max(10, int(86.13 × 1.92 × 0.85)) = max(10, 140) = 140
max_new_tokens = int(86.13 × 1.92 × 1.6) = 264
```
So for this phrase, the model generates 140–264 tokens (representing 1.63–3.07 seconds of audio).

---

## 15. PERFORMANCE TARGETS AND METRICS

### Primary Target
**TTFA < 400ms** (revised from 500ms in PLAN.md §2, after streaming implementation)

Note: The PLAN.md initially targets 500ms (§1), then updates to 400ms as achievable (§2) after the streaming bug fix.

### What the Streaming Bug Was

In an earlier commit, the server was collecting all audio pieces in a list and only sending them AFTER generation completed:

```python
# OLD (broken) code:
pieces = []
async for audio_piece in gen:
    pieces.append(audio_piece)  # Collect all first
for piece in pieces:
    await ws.send_bytes(pcm16(piece))  # Then send — too late!
```

This made TTFA equal to the full synthesis time (3–5 seconds). The fix was to send each piece immediately as it arrives:

```python
# FIXED code:
pieces = []
async for audio_piece in gen:
    pieces.append(audio_piece)
    await ws.send_bytes(pcm16(audio_piece))  # Send immediately!
```

This change (git commit `96643d0`) reduced TTFA from ~3–5s to the first ParlerTTSStreamer chunk latency (~300–500ms).

### Latency Breakdown (estimated, after fix)

| Stage | Time |
|---|---|
| Text normalization + chunking | ~5ms |
| First chunk: flan-t5-large encoding (cached) | ~20ms |
| First chunk: model.generate first 26 tokens | ~200–350ms |
| DAC decode: 26 tokens → audio | ~20ms |
| WebSocket frame transmission | ~5ms |
| **Total TTFA (estimated)** | **~250–400ms** |

### Competitor TTFA Comparison (from research/findings-2026-09-08.md)

| Service | TTFA | Notes |
|---|---|---|
| ElevenLabs Flash v2.5 | < 75ms | Streaming + edge CDN |
| Sarvam Bulbul V3 | < 250ms TTFB | Indian language specialist |
| Fish Audio S2 Pro | ~100ms on H200 | High-end GPU |
| **Movio-2 (target)** | **< 400ms** | T4 GPU on Kaggle |

### Hardware Performance

| GPU | TTFA Estimate | Concurrent Users |
|---|---|---|
| Kaggle T4 (16GB) | ~300–500ms | 1 (serial) |
| AWS G5 (A10G 24GB) | ~150–250ms | 2–3 |
| A100 40GB | ~100–150ms | 4–6 |
| H200 80GB | ~50–100ms | 8–12 |

Note: Multiple concurrent users require careful GPU memory management — at bfloat16, the 0.9B model uses ~1.8GB, leaving ~14.2GB for KV cache and activations on a T4. Two concurrent users would use ~4–6GB total and likely cause OOM during peak activation memory.

### Tamil NSS (Native Speaker Score): 75.48% ± 2.18%
- Source: Interspeech 2025 paper
- Method: Human evaluation by Tamil native speakers
- Comparison: No public benchmark for IndicF5 Tamil NSS available
- Interpretation: 75.48% of the time, native speakers identify the voice as a native speaker

### Real-Time Factor (RTF) Estimates
From research notes:
- T4 GPU: ~5–6 seconds of generation for 10 output tokens → RTF ~3.0 (3× slower than real-time without streaming)
- A100 40GB: ~0.5–1s for 10 tokens → RTF ~0.3 (0.3× real-time = 3× faster than real-time)

RTF = synthesis_time / audio_duration. RTF < 1.0 means faster than real-time; > 1.0 means slower.

Note: The streaming approach makes RTF irrelevant for perceived latency — users hear the first audio chunk long before the full synthesis completes.

---

## 16. DEPLOYMENT: KAGGLE GPU NOTEBOOK

### Kaggle Notebook (`kaggle_notebook.ipynb`)

The project runs on Kaggle free T4 GPU instances (no AWS, no local GPU required). The Kaggle notebook provides a complete deployment in 13 cells:

**Cell 1–3: Installation**
```python
!pip install torch transformers parler-tts fastapi uvicorn[standard] \
    numpy soundfile accelerate sentencepiece indic-numtowords \
    pyloudnorm nest_asyncio -q
```

**Cell 4: HuggingFace Authentication**
```python
from kaggle_secrets import UserSecretsClient
HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
from huggingface_hub import login
login(HF_TOKEN)
```
HF_TOKEN stored as a Kaggle secret (not in code). Note: `ai4bharat/indic-parler-tts` is NOT gated (unlike IndicF5 in Approach 1) — HF_TOKEN is only needed for private variants.

**Cell 5: Clone Repository**
```python
!git clone https://github.com/tripathiji1312/movio_new.git movio2
```

**Cell 6: Start Server**
```python
import nest_asyncio
nest_asyncio.apply()

import uvicorn
from movio2.backend.main import app
# Start in background thread so notebook cell doesn't block
import threading
thread = threading.Thread(target=uvicorn.run, args=(app,), kwargs={"host":"0.0.0.0","port":8000})
thread.daemon = True
thread.start()
```

**Cell 7: Cloudflare Tunnel**
```python
# Download cloudflared binary
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64

# Start tunnel in background
import subprocess
proc = subprocess.Popen(['./cloudflared-linux-amd64', 'tunnel', '--url', 'http://localhost:8000'],
                         stderr=subprocess.PIPE)

# Extract public URL from cloudflared output
import re, time
for _ in range(30):
    line = proc.stderr.readline().decode()
    url_match = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com', line)
    if url_match:
        public_url = url_match.group(0)
        print(f"Public URL: {public_url}")
        break
    time.sleep(1)
```

**Why Cloudflare Tunnel instead of ngrok?**
- ngrok free tier requires an account and has limited bandwidth
- Cloudflare Tunnel (`cloudflared`) is completely free, no account required, no bandwidth limits
- Works for WebSocket connections (ngrok free tier has WebSocket limitations)
- URL is randomly generated per session (HTTPS guaranteed, Cloudflare CDN)

### Development Notebook (`indic-parler-livestream (1).ipynb`)

A 32-cell development/prototyping notebook used during development. Contains:
- Earlier versions of the streaming pipeline
- Peak normalization (replaced by LUFS normalization)
- Volume control experiments
- Jitter buffer implementation (later replaced by lookahead scheduling)
- ngrok tunnel setup (replaced by cloudflared)
- Debug audio quality tests

This notebook documents the evolution of the approach — the progression from "working but poor quality" to "working and high quality" across 22 git commits.

---

## 17. THE COMPLETE FILE MAP

```
movio-2/
├── backend/
│   ├── __init__.py             # Empty
│   ├── config.py               # All configuration constants (23 lines)
│   ├── normalizer.py           # Text normalization pipeline (189 lines)
│   ├── chunker.py              # Sentence/chunk splitting (99 lines)
│   ├── tts_engine.py           # TTS model wrapper + streaming (231 lines)
│   ├── server.py               # FastAPI WebSocket server (303 lines)
│   └── main.py                 # Entry point: load model + start uvicorn (13 lines)
│
├── frontend/
│   ├── index.html              # Main TTS demo UI (1867 lines)
│   └── conversation.html       # Phone call interrupt demo (907 lines)
│
├── research/
│   ├── INDEX.md                # Research log index
│   └── findings-2026-09-08.md  # Deep research: streaming, latency, models (428 lines)
│
├── PLAN.md                     # Full implementation plan with 7 phases (769 lines)
├── requirements.txt            # 11 Python dependencies
├── run.py                      # Top-level entrypoint (4 lines)
│
├── indic-parler-livestream (1).ipynb  # Development notebook (32 cells)
└── kaggle_notebook.ipynb              # Production deployment notebook (13 cells)
│
├── .claude/settings.local.json    # Claude AI tool permission allowlist
└── .git/                          # Git repo (22 commits, remote: github.com/tripathiji1312/movio_new.git)
```

Total Python code: ~835 lines across 6 backend files
Total frontend code: ~2,774 lines across 2 HTML/JS files

---

## 18. ALL DEPENDENCIES

### Python Dependencies (requirements.txt)

| Package | Version | Purpose |
|---|---|---|
| `torch` | (latest PyTorch) | GPU tensor operations, model inference |
| `transformers` | `>=4.46.1` | ParlerTTS model class, tokenizer, AutoModel |
| `parler-tts` | (from git/pip) | ParlerTTS model + ParlerTTSStreamer |
| `accelerate` | (latest) | HuggingFace device management (auto-detect GPU) |
| `sentencepiece` | (latest) | Tokenizer backend for flan-t5-large |
| `soundfile` | (latest) | Audio I/O (WAV read/write in notebooks) |
| `indic-numtowords` | 1.1.0 (confirmed installed) | Indian language number-to-words (22 languages) |
| `fastapi` | (latest) | Async web framework + WebSocket |
| `uvicorn[standard]` | (latest) | ASGI server with uvloop |
| `numpy` | (latest) | Audio array operations, PCM16 conversion |
| `pyloudnorm` | `>=0.1.0` | LUFS measurement + normalization (ITU-R BS.1770-4) |

### Transitive Dependencies (from pip install, not in requirements.txt)

| Package | Source | Purpose |
|---|---|---|
| `scipy` | via pyloudnorm | `scipy.signal.butter` + `sosfilt` for HPF |
| `websockets` | via fastapi | WebSocket protocol |
| `starlette` | via fastapi | ASGI middleware |
| `pydantic` | via fastapi | Request/response validation |
| `httpx` | via fastapi | HTTP client |
| `huggingface_hub` | via transformers | Model downloading |
| `safetensors` | via transformers | Efficient model loading |
| `tokenizers` | via transformers | Fast tokenizer C++ implementation |
| `descript-audio-codec` | via parler-tts | DAC encoder/decoder |

### Development/Kaggle-Only Dependencies

| Package | Purpose |
|---|---|
| `nest_asyncio` | Allow asyncio event loop nesting in Jupyter |
| `kaggle_secrets` | Kaggle's secret manager for HF_TOKEN |
| `huggingface_hub` (login) | HuggingFace CLI login |
| `cloudflared` (binary) | Cloudflare tunnel for public URL |

### JavaScript Dependencies (CDN)

| Resource | URL | Purpose |
|---|---|---|
| Google Fonts: Plus Jakarta Sans | fonts.googleapis.com | Body typography |
| Google Fonts: JetBrains Mono | fonts.googleapis.com | Monospace (code, metrics) |
| Web Audio API | Built-in browser | Audio playback, scheduling, analysis |
| WebSocket API | Built-in browser | Real-time server connection |
| Canvas API | Built-in browser | Visualizer, waveform |

No external JavaScript libraries (no jQuery, no React, no Three.js). All functionality is pure vanilla JavaScript + browser APIs.

### Python Version

- Runtime: Python 3.14 (installed in `.venv/`)
- `.venv/lib/python3.14/site-packages/` contains only `indic_numtowords` and `pip`
- The rest of the dependencies are expected to be installed system-wide or via `pip install -r requirements.txt`

---

## 19. KNOWN ISSUES, TRACKED BUGS, AND FUTURE PLANS

### Active Bugs

**Bug 1: Tamil numbers always spoken in English**
- **Location**: `backend/normalizer.py:spell_amount()`
- **Cause**: `lang="en"` hardcoded; ignores `contains_tamil()` function that exists but is never called
- **Impact**: "Your fare is ₹250" → "your fare is two hundred fifty" (English) instead of "your fare is இருநூற்று ஐம்பது" (Tamil)
- **Fix**: Check `contains_tamil(original_text)` before deciding `lang`; call `indic_num2words(n, lang='ta')` for Tamil context

**Bug 2: `indic_numtowords` decimal crash in Tamil mode**
- **Location**: Would occur in `spell_amount()` if `lang='ta'` were used with decimal input
- **Cause**: `indic_num2words(2.5, lang='ta')` raises `KeyError: '.5'` — library bug (integer-only in Tamil mode)
- **Fix**: Manual split at decimal point: integer part via `num2words(int_part, lang='ta')` + "புள்ளி" + digit-by-digit decimal

**Bug 3: Tamil numeral characters not converted**
- **Location**: `backend/normalizer.py`
- **Cause**: Tamil Unicode numerals ௧ (U+0BE7) through ௯ (U+0BEF) are not recognized as numbers
- **Impact**: Input "₹௨௫0" would not be normalized correctly
- **Fix**: Add `TA_DIGITS = str.maketrans("௦௧௨௩௪௫௬௭௮௯", "0123456789")` and apply before normalization

**Bug 4: No Tamil abbreviation expansion**
- **Location**: `backend/normalizer.py:PRONUNCIATION_OVERRIDES`
- **Cause**: Table only covers English abbreviations + one Tamil word
- **Missing**: செ.மீ (centimeters), கி.மீ (kilometers), ரூ (rupees), திரு (Mr. in Tamil), திருமதி (Mrs. in Tamil), etc.
- **Fix**: Add Tamil abbreviations to PRONUNCIATION_OVERRIDES

**Bug 5: No health check endpoint**
- **Location**: `backend/server.py`
- **Cause**: Not implemented yet
- **Impact**: Deployment orchestration (Docker health checks, Kubernetes probes) cannot verify server readiness
- **Fix**: Add `GET /health → {"status": "ok", "model_loaded": bool}`

### PLAN.md — 7 Implementation Phases

The full implementation plan in `PLAN.md` defines 7 phases:

**Phase 1 (COMPLETE): Critical Streaming Fix**
- Fix the audio collection bug (send each frame immediately, not after all collected)
- Implement ParlerTTSStreamer
- Git commit: `96643d0` — "Optimize WebSocket audio streaming by implementing caching and concatenation of audio pieces"

**Phase 2 (COMPLETE): Audio Quality Improvements**
- LUFS normalization to -13 LUFS
- High-pass filter at 60Hz
- Silence trimming with tail guard
- Fade-in/fade-out
- Description encoder caching
- Git commits: `6780c05` through `0441b3a`

**Phase 3 (PENDING): Tamil Text Normalization**
- Integrate `indic-nlp-library` for proper Tamil Unicode normalization
- Fix Tamil numeral character conversion
- Fix Tamil numbers in English bug
- Add Tamil abbreviation expansion
- Estimated: 2–3 days

**Phase 4 (PENDING): Advanced Tamil Processing**
- `indic-nlp-library` for ZWJ/ZWNJ normalization
- Compound character handling
- Tamil morphological analysis for better chunking
- Estimated: 3–5 days

**Phase 5 (PENDING): IndicF5 Voice Cloning Integration**
- Add `tts_engine_f5.py` parallel to `tts_engine.py`
- New endpoint `/ws/voice-clone` accepting reference audio
- Use IndicF5 for voice cloning use cases
- Keep parler-tts for description-conditioned synthesis
- Estimated: 2–3 days

**Phase 6 (PENDING): Production Hardening**
- Add `/health` endpoint
- Add rate limiting
- Add Prometheus metrics
- Move static files to nginx
- Redis cache instead of in-memory LRU
- Estimated: 2–3 days

**Phase 7 (PENDING): Tanglish-Specific Fine-tuning**
- Collect 50–100 hours of Tanglish transport domain audio
- Fine-tune indic-parler-tts on Tanglish pronunciation patterns
- Evaluate with Tamil NSS before/after
- Estimated: 2–3 weeks (data collection dominates)

---

## 20. WHY EVERY KEY DECISION WAS MADE

| Decision | Alternative | Reason |
|---|---|---|
| ai4bharat/indic-parler-tts | IndicF5 (Approach 1) | Higher Tamil NSS (75.48%); no reference audio needed |
| ParlerTTSStreamer for streaming | Wait for complete generation | Without streaming: TTFA = full synthesis time (3–5s) |
| `play_steps_in_s = 0.3` | 0.1 or 0.5 | 0.3s: best balance of chunk frequency vs audio quality |
| `temperature = 1.0` | 0.3 (previous default) | Natural prosody; 0.3 was causing micro-stutters |
| `seed = 42` | Random seed | Required for cache consistency (same text = same audio) |
| `attn_implementation="eager"` | "sdpa" or "flash_attention_2" | Both broken for this model (community reports + model card) |
| bfloat16 priority over float16 | Always float16 on CUDA | bfloat16 safer for LLMs: 5-bit vs 8-bit exponent range |
| LUFS normalization to -13 | Peak normalization | Perceptual loudness consistency; peak norm misses quiet parts |
| HPF at 60Hz, order 4 | No filter / different frequency | 60Hz: above all sub-bass but below speech fundamental |
| Butterworth HPF (not Chebyshev) | Chebyshev / elliptic | Butterworth: maximally flat passband, no ripple |
| asyncio.Lock (single GPU) | Multi-process parallelism | 0.9B at bf16 = 1.8GB; two concurrent = likely OOM on T4 |
| Config fingerprint in cache key | Text-only cache key | Config change auto-invalidates without manual flush |
| "double X" in spell_alnum_code | "X X" for repeated letters | Natural: humans say "double A" not "A A" for repeated letters |
| Cloudflare Tunnel | ngrok | Free, no account, no bandwidth limit, WebSocket compatible |
| Python 3.14 | Python 3.11 | Kaggle environment provides 3.14 |
| Frontend auto-tagger (JavaScript) | Server-only normalization | Preview shows exact tagged text; catch normalization errors client-side |
| Production cache of post-processed audio | Cache raw audio | Cached audio always includes LUFS normalization; consistent volume |
| 80ms silence pad in JS trimmer | 0ms or 30ms pad | 80ms: preserve plosive consonant attacks (p, b, t onset below threshold) |
| Generation ID protocol in conversation | Server-side flow control | Race condition: stale frames already in WebSocket buffer arrive after interrupt |
| Sample rate pinning at 44.1kHz | Use browser default (48kHz) | Browser default would resample at chunk boundaries, causing seams |
| Master gain node for interrupt | Per-source gain | Fade out ALL playing audio with one ramp; simpler, glitch-free |

---

## 21. CRITICAL WARNINGS AND GOTCHAS

### ⚠️ "sdpa" and "flash_attention_2" Are Broken for This Model

```python
# WRONG - will produce garbage audio or crash:
model = ParlerTTSForConditionalGeneration.from_pretrained(
    MODEL_ID, attn_implementation="sdpa"  # DO NOT USE
)

# CORRECT:
model = ParlerTTSForConditionalGeneration.from_pretrained(
    MODEL_ID, attn_implementation="eager"  # Required
)
```

This is not a general limitation of ParlerTTS — it's specific to `ai4bharat/indic-parler-tts`. The model card and community discussions confirm this. Future model versions may fix it.

### ⚠️ Do NOT Use `repetition_penalty` with This Model

The DAC codec legitimately repeats tokens for sustained phonemes. Any `repetition_penalty > 1.0` will cause:
- Pitch shifts on sustained vowels
- Unnatural consonant truncation
- "Choppy" or "robotic" output despite the model being high quality

### ⚠️ Only One Concurrent Generation

The `asyncio.Lock()` serializes all generation. If two users submit synthesis requests simultaneously, the second user will wait for the first to complete before their synthesis begins. This can cause noticeable delays (3–5 seconds) for the second user.

For production at scale, run multiple instances (load balancer + 3–4 Kaggle notebooks) rather than trying to run concurrent generation on a single instance.

### ⚠️ The Fixed Seed Is Required for Cache Correctness

`GENERATION_SEED = 42` must remain fixed. If you make synthesis non-deterministic (remove the seed), the same text will produce different audio on cache miss vs cache hit, and the cached audio will sound different from what the user heard on first synthesis. This can be confusing in A/B tests.

To explore voice variety, change `VOICE_DESCRIPTION` (which changes the config fingerprint and auto-invalidates the cache).

### ⚠️ The `contains_tamil()` Function Is Not Called

The function exists in `backend/normalizer.py` but is never used in the normalization pipeline. Tamil detection is needed for language-conditional normalization. Until Phase 3 is implemented, all numbers will be spoken in English regardless of input language.

### ⚠️ Python 3.14 Compatibility

The project uses Python 3.14 (from Kaggle's environment). Not all libraries (especially `parler-tts` and `descript-audio-codec`) have been extensively tested on 3.14. If you encounter compatibility issues, switching to Python 3.11 in a virtual environment is the safe fallback.

---

## 22. GIT HISTORY: WHAT WAS BUILT AND WHEN

The git history of 22 commits tells the story of development (newest first):

| # | Commit | What Changed | Why |
|---|---|---|---|
| 1 | `a9762ba` | Track audio end time + auto-tagging | Enable interrupt detection without server state |
| 2 | `96643d0` | Fix streaming: send bytes inside loop (TTFA fix) | **THE KEY FIX**: TTFA went from 3–5s to ~300–500ms |
| 3 | `7ca4646` | Crossfade logic + master gain | Smoother chunk transitions |
| 4 | `b130b28` | WebSocket conversation endpoint + abort | Enable interrupt/preempt functionality |
| 5 | `8e026f6` | Remove preload mechanism, seamless playback | Preload was adding unnecessary delay |
| 6 | `fc829ca` | Jitter buffer (later removed) | Attempted to fix gaps — superseded by lookahead scheduling |
| 7 | `868c175` | Volume control slider | UX improvement |
| 8 | `b688515` | Loudness boost for live stream | Audio too quiet on mobile speakers |
| 9 | `0441b3a` | LUFS normalization + HPF | Consistent loudness; remove sub-bass artifacts |
| 10 | `6780c05` | Refactor audio processing, fade in/out | Fix timing issues, prevent clicks |
| 11 | `cecbaee` | TF/Flax env vars | Suppress TensorFlow import warnings |
| 12 | `c6b2fd1` | TTS config refactor + audio improvements | Code cleanup |
| 13 | `a4bf669` | "double X" in spell_alnum_code | Natural pronunciation for repeated letters |
| 14 | `89a4632` | EN_LETTER_NAMES consistency + digit spelling | Fix inconsistencies in code spelling |
| 15 | `3eeaf51` | Sentence splitting + ID pronunciation | Better chunking for complex sentences |
| 16 | `88f7154` | Time spelling with AM/PM period | "seven thirty AM" → "seven thirty A M" |
| 17 | `05ad75d` | Pronunciation overrides + sentence split | More natural abbreviation expansion |
| 18 | `668fed7` | Audio processing refactor + HTML cleanup | Technical debt cleanup |
| 19 | `a0d4852` | LRU cache + full audio player UI | Cache reduces repeat request latency; UI improvement |
| 20 | `42195a9` | Kaggle notebook installation | First deployable Kaggle notebook |
| 21 | `a47d55a` | Kaggle notebook installation refactor | Fix pip install order |
| 22 | `024ff9e` | Clone correct repository | Fix notebook to use movio_new repo |

**The most important commits:**
- `96643d0`: Fixed TTFA from 3–5 seconds to ~300–500ms. This was the fundamental fix that made the streaming approach viable.
- `a0d4852`: Added LRU cache. Repeat requests now take ~10ms instead of ~300ms.
- `0441b3a`: LUFS normalization. Fixed inconsistent loudness that made some utterances inaudible and others too loud.
- `b130b28`: Conversation endpoint with interrupt. This is the unique feature that enables phone-call-like interaction.

---

*End of Report — Movio-2 Indic-Parler-TTS Approach*

*This report was generated from complete code analysis of all files in the `movio-2/` directory as of September 2026.*
