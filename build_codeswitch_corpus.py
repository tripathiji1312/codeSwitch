"""Build a code-switched distillation corpus from real text + controlled mixing.

Why this exists
---------------
The draft/residual models are distilled from the Parler teacher, so the *targets*
are always teacher audio; what the corpus controls is the **text distribution**
the student sees.  A template-only corpus teaches the draft a handful of
sentence shapes, which shows up as low CB0 acceptance and unintelligible hybrid
audio at evaluation time.  This script assembles a wider corpus:

  * ``--max-sentences N``- cap the size (teacher time is the budget)
                           (``codeswitch_benchmark_data.build_training_corpus``),
                           which covers the assistant/transit registers the
                           benchmark measures;
  * ``--wikipedia N``    - *real* Tamil sentences from a Wikipedia dump
                           (``wikimedia/wikipedia`` 20231101.ta, streamed) or
                           from ``--wikipedia-file``, filtered to TTS-able
                           lengths: real syntax and vocabulary diversity that
                           templates cannot provide;
  * ``--extra-file PATH``- your own text (WhatsApp/transcript exports, published
                           code-switching corpora).  Repeatable; the best
                           generalisation signal if you have it;
  * ``--code-mix-rate``  - English entity insertion (order/OTP/UPI/fare/plate...)
                           into Tamil sentences, matching the benchmark mix;
  * ``--romanize-rate``  - rule-based Tamil -> Latin transliteration of a word
                           subset, so the draft also trains on Tanglish input.

Output: one sentence per line (``--out``), deduplicated, guaranteed disjoint from
the evaluation set, with a provenance/register summary printed on stdout.

Usage on Kaggle
---------------
    !python build_codeswitch_corpus.py --templates 2500 --wikipedia 2500 \
        --code-mix-rate 0.45 --romanize-rate 0.25 --out codeswitch_corpus.txt
    !python train_draft_distill.py --corpus-json codeswitch_corpus.txt \
        --draft-epochs 60 --residual-sched 0.5 --max-minutes 40
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from codeswitch_benchmark_data import (
    EVAL_UTTERANCES,
    assert_corpus_disjoint_from_eval,
    build_training_corpus,
)
from codeswitch_tts.frontend.tamil_latin import TAMIL_RANGE as _TAMIL_RANGE
from codeswitch_tts.frontend.tamil_latin import transliterate_word

# ---------------------------------------------------------------------------
# Code-mix material: English/Tanglish spans the benchmark actually contains
# ---------------------------------------------------------------------------
ENTITY_PHRASES: List[str] = [
    "order", "delivery", "delivery partner", "parcel", "courier", "invoice",
    "bill", "GST bill", "discount coupon", "offer code", "booking", "reservation",
    "cancellation", "refund", "cash on delivery", "UPI payment", "Google Pay",
    "PhonePe", "net banking", "card payment", "balance", "statement", "passbook",
    "cheque", "withdrawal", "deposit", "account number", "IFSC code",
    "OTP", "one time password", "verification code", "tracking number",
    "reference number", "token number", "mobile number", "whatsapp", "email",
    "address", "location", "live location", "pin code", "landmark", "route",
    "shortcut", "traffic", "signal", "toll gate", "service road", "bypass",
    "platform number", "ticket", "seat number", "coach", "berth", "conductor",
    "driver", "auto", "cab", "bike", "petrol bunk", "charging point",
    "service center", "customer care", "helpline", "warranty", "service request",
    "charger", "battery", "screen guard", "network", "recharge", "data pack",
    "postpaid connection", "sim card", "Aadhaar card", "PAN card", "driving licence",
    "hospital", "appointment", "tablet", "medicine", "doctor", "scan report",
    "insurance", "policy number", "claim", "premium", "salary", "payslip",
    "electricity bill", "water bill", "gas connection", "rent", "advance amount",
    "grocery", "vegetables", "milk packet", "food delivery", "hotel", "lunch",
    "dinner", "breakfast", "tea", "coffee", "snacks",
]

#: Fill-in patterns for the code-mixed register (Tamil sentence + English span).
MIX_PREFIX: List[str] = [
    "{entity} ready ஆகிடுச்சு.",
    "{entity} confirm ஆகிடுச்சு.",
    "{entity} update வந்துச்சு.",
    "{entity} details அனுப்பிட்டேன்.",
    "{entity} problem இல்ல.",
    "{entity} நாளைக்கு வரும்.",
    "{entity} இப்போ வேணும்.",
    "{entity} பத்தி ஒரு call வரும்.",
]

MIX_MID: List[str] = [
    "நான் {entity} பாக்கறேன்.",
    "அவங்க {entity} கொடுத்தாங்க.",
    "உங்க {entity} இங்க இருக்கு.",
    "இந்த {entity} கொஞ்சம் நேரம் எடுக்கும்.",
    "அந்த {entity} ரொம்ப முக்கியம்.",
    "எங்க {entity} தயாரா இருக்கு.",
    "நீங்க {entity} check பண்ணுங்க.",
    "சார், {entity} வந்துடுச்சு.",
]

#: Numeric/entity templates that carry digits (OTP, fare, plate) - the transit
#: register the benchmark measures hardest.
NUMERIC_TEMPLATES: List[str] = [
    "உங்க OTP {otp4}, இதை யாருக்கும் சொல்லாதீங்க.",
    "OTP {otp4} - five minutes மட்டும் valid.",
    "Fare {fare} rupees, change இருக்கா?",
    "Auto fare {fare}, meter-ல போகலாம்.",
    "Vehicle number TN {otp2} AB {otp4}.",
    "Order {order} இன்னும் ten minutes-ல வரும்.",
    "Token number {otp2}, counter three-ல காத்திருங்க.",
    "Room number {otp2}, first floor-ல இருக்கு.",
    "Platform {otp1} - train ten minutes late.",
    "Account last four digits {otp4}, confirm பண்ணுங்க.",
    "Delivery partner {otp2} minutes-ல வருவாரு.",
    "Bill amount {fare} rupees, UPI-ல அனுப்புங்க.",
]

_ENTITY_RE = re.compile(r"\{(\w+)\}")


def fill_entity(template: str, rng: random.Random, entity: Optional[str] = None) -> str:
    """Fills ``{entity}`` / numeric slots in a code-mix template."""

    def _repl(match: re.Match) -> str:
        slot = match.group(1)
        if slot == "entity":
            return entity or rng.choice(ENTITY_PHRASES)
        if slot == "otp4":
            return str(rng.randint(1000, 9999))
        if slot == "otp2":
            return str(rng.randint(10, 99))
        if slot == "otp1":
            return str(rng.randint(1, 9))
        if slot == "fare":
            return str(rng.choice([45, 60, 75, 85, 95, 120, 150, 180, 245, 280,
                                   350, 420, 599, 750, 1200, 1250]))
        if slot == "order":
            return str(rng.randint(1000, 9999))
        return match.group(0)

    return _ENTITY_RE.sub(_repl, template)


def romanize(text: str, rng: random.Random, word_rate: float = 1.0) -> str:
    """Romanizes Tamil words (``word_rate`` = fraction of eligible words).

    Defaults to full-sentence romanisation: the benchmark's romanized register
    is written entirely in Latin, and mixed Tamil/Latin script inside one
    sentence would teach the draft a third, unnatural convention.
    """
    tokens = re.split(r"(\s+)", text)
    out: List[str] = []
    for token in tokens:
        stripped = token.strip()
        if stripped and _TAMIL_RANGE.search(stripped) and rng.random() < word_rate:
            prefix = token[: len(token) - len(token.lstrip())]
            suffix = token[len(token.rstrip()):]
            out.append(prefix + transliterate_word(stripped) + suffix)
        else:
            out.append(token)
    return "".join(out)


def classify_register(text: str) -> str:
    """Honest post-hoc register classification of a corpus line."""
    if not _TAMIL_RANGE.search(text):
        return "romanized_tanglish"
    if re.search(r"[A-Za-z]", text):
        return "transit_codemixed"
    return "native_tamil"


# ---------------------------------------------------------------------------
# Real-text sources
# ---------------------------------------------------------------------------
def clean_sentence(raw: str) -> Optional[str]:
    """Normalises a candidate sentence and rejects non-TTS-able material."""
    text = re.sub(r"\s+", " ", raw).strip()
    text = text.replace("\u200b", "").strip()
    if not text or not _TAMIL_RANGE.search(text):
        return None
    if any(ch in text for ch in "()[]{}<>|/\\*#@\"“”‘’"):
        return None
    if re.search(r"https?://|www\.", text):
        return None
    if re.search(r"[\u0b80-\u0bff]*\d{3,}", text):
        return None
    if not text.endswith((".", "!", "?")):
        text += "."
    words = text.split()
    if not (4 <= len(words) <= 22):
        return None
    if len(re.findall(r"[A-Za-z]", text)) > 0.35 * max(1, len(text)):
        return None  # keep this source Tamil-dominant; mixing is added later
    return text


def sentences_from_text(blob: str) -> List[str]:
    out: List[str] = []
    for chunk in re.split(r"[.!?।\n]+", blob):
        cleaned = clean_sentence(chunk)
        if cleaned:
            out.append(cleaned)
    return out


def load_wikipedia_sentences(
    n: int, cache: Path, dataset: str = "wikimedia/wikipedia",
    config_name: str = "20231101.ta", seed: int = 7,
) -> List[str]:
    """Streams Tamil Wikipedia sentences (cached to ``cache`` after the first run)."""
    if cache.exists():
        cached = [line.strip() for line in cache.read_text(encoding="utf-8").splitlines()
                  if line.strip()]
        if len(cached) >= n:
            random.Random(seed).shuffle(cached)
            return cached[:n]
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover - Kaggle has datasets installed
        print(f"[Corpus] Wikipedia source unavailable ({type(exc).__name__}: {exc}); "
              "skipping real-text sentences.")
        return []
    print(f"[Corpus] streaming {dataset}/{config_name} ...")
    collected: List[str] = []
    seen: set[str] = set()
    try:
        stream = load_dataset(dataset, config_name, split="train", streaming=True)
        for row in stream:
            for sentence in sentences_from_text(str(row.get("text", ""))):
                key = sentence.casefold()
                if key in seen:
                    continue
                seen.add(key)
                collected.append(sentence)
                if len(collected) >= max(n * 3, n):
                    break
            if len(collected) >= max(n * 3, n):
                break
    except Exception as exc:
        print(f"[Corpus] Wikipedia streaming failed ({type(exc).__name__}: {exc}); "
              "continuing with the other sources.")
    if collected:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text("\n".join(collected), encoding="utf-8")
        print(f"[Corpus] cached {len(collected)} real Tamil sentences -> {cache}")
    random.Random(seed).shuffle(collected)
    return collected[:n]


def load_extra_files(paths: Sequence[str]) -> List[str]:
    out: List[str] = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            print(f"[Corpus] ⚠ extra file not found: {p}")
            continue
        raw = p.read_text(encoding="utf-8", errors="ignore")
        stripped = raw.lstrip()
        if stripped.startswith("["):
            try:
                data = json.loads(raw)
                lines = [str(x["text"]) if isinstance(x, dict) else str(x) for x in data]
            except Exception:
                lines = raw.splitlines()
        else:
            lines = raw.splitlines()
        kept = 0
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cleaned = clean_sentence(line) or (line if _TAMIL_RANGE.search(line) else None)
            if cleaned:
                out.append(cleaned)
                kept += 1
        print(f"[Corpus] extra file {p.name}: kept {kept} sentences")
    return out


# ---------------------------------------------------------------------------
# Mixing + assembly
# ---------------------------------------------------------------------------
def inject_entity(sentence: str, entity: str, rng: random.Random) -> str:
    """Inserts an English span *into* a Tamil sentence (three natural styles).

    The base sentence is preserved: mid-sentence noun insertion, a prefixed
    request fragment, or a trailing clause.  TTS-able in all three cases.
    """
    style = rng.random()
    words = sentence.split()
    if style < 0.45 and len(words) >= 4:
        # mid-sentence English noun: "உங்க order நாளைக்கு வரும்."
        pos = rng.randint(1, max(1, len(words) - 3))
        words.insert(pos, entity)
        return " ".join(words)
    if style < 0.75:
        # prefixed English request fragment
        template = rng.choice(MIX_PREFIX)
        return fill_entity(template, rng, entity) + " " + sentence
    # trailing English clause
    template = rng.choice(MIX_MID)
    return sentence.rstrip(".!?") + ". " + fill_entity(template, rng, entity)


def augment(
    sentence: str, rng: random.Random, code_mix_rate: float, romanize_rate: float,
    numeric_rate: float,
) -> Tuple[str, str]:
    """Applies code-mix / romanisation augmentation; returns (text, register)."""
    text = sentence
    roll = rng.random()
    if roll < numeric_rate:
        # transit register: OTP / fare / plate templates dominate the benchmark
        text = fill_entity(rng.choice(NUMERIC_TEMPLATES), rng)
    elif roll < numeric_rate + code_mix_rate:
        text = inject_entity(sentence, rng.choice(ENTITY_PHRASES), rng)
    if rng.random() < romanize_rate:
        text = romanize(text, rng)
    return text, classify_register(text)


REGISTERS: Tuple[str, ...] = ("native_tamil", "romanized_tanglish", "transit_codemixed")


def build_corpus(
    n_templates: int = 2000, n_wikipedia: int = 0, extra_files: Sequence[str] = (),
    code_mix_rate: float = 0.35, romanize_rate: float = 0.30, numeric_rate: float = 0.12,
    seed: int = 7, wiki_cache: str = "wikidata_tamil_sentences.txt", passes: int = 3,
    balance: Sequence[float] = (0.34, 0.33, 0.33), max_sentences: int = 0,
) -> Tuple[List[str], Dict[str, int]]:
    """Assembles a register-balanced corpus and returns ``(sentences, counts)``.

    Each base sentence is augmented ``passes`` times with fresh random rolls;
    the resulting lines are bucketed by their *classified* register and then
    subsampled to the requested quota, so the student sees roughly the same
    register mix the benchmark measures (the benchmark itself is 20/20/20).

    ``max_sentences`` caps the output while keeping that mix - the corpus size is
    what the teacher pass pays for (~7.9 s/sentence on a T4), so this is the knob
    that makes a corpus fit one Kaggle session.  Selection is deterministic for a
    given seed, and a smaller cap is a prefix of a larger one, so raising the cap
    later only adds sentences (the teacher cache is content-addressed, so only the
    new lines are collected).
    """
    rng = random.Random(seed)
    sources: List[Tuple[str, List[str]]] = []
    if n_templates:
        sources.append(("templates", build_training_corpus(n=n_templates, seed=seed)))
    if n_wikipedia:
        sources.append(("wikipedia", load_wikipedia_sentences(
            n_wikipedia, Path(wiki_cache), seed=seed)))
    if extra_files:
        sources.append(("extra", load_extra_files(extra_files)))

    by_register: Dict[str, List[str]] = {r: [] for r in REGISTERS}
    seen: set[str] = set()
    for source, sentences in sources:
        kept = 0
        for sentence in sentences:
            for _ in range(max(1, int(passes))):
                text, register = augment(sentence, rng, code_mix_rate, romanize_rate,
                                         numeric_rate)
                text = re.sub(r"\s+", " ", text).strip()
                if not text or (not _TAMIL_RANGE.search(text) and not re.search(r"[A-Za-z]", text)):
                    continue
                key = text.casefold()
                if key in seen:
                    continue
                seen.add(key)
                by_register.setdefault(register, []).append(text)
                kept += 1
        print(f"[Corpus] source {source}: kept {kept} augmented lines")

    weights = [max(0.0, float(w)) for w in balance]
    if len(weights) != len(REGISTERS) or sum(weights) <= 0:
        weights = [1.0 / len(REGISTERS)] * len(REGISTERS)
    weights = [w / sum(weights) for w in weights]
    available = {r: len(by_register.get(r, [])) for r in REGISTERS}
    target_total = min(
        (int(available[r] / w) for r, w in zip(REGISTERS, weights) if w > 0 and available[r]),
        default=0,
    )
    if max_sentences and max_sentences > 0:
        capped = min(target_total, int(max_sentences))
        if capped < target_total:
            print(f"[Corpus] capping {target_total} -> {capped} sentences (--max-sentences)")
        target_total = capped
    # per-register quota: floor first, then hand the flooring remainder out to the
    # registers with head-room so the corpus hits the requested size exactly
    quotas = {r: (min(int(target_total * w), available[r]) if target_total else available[r])
              for r, w in zip(REGISTERS, weights)}
    if target_total:
        remainder = target_total - sum(quotas.values())
        for r in REGISTERS:
            if remainder <= 0:
                break
            room = available[r] - quotas[r]
            add = min(max(room, 0), remainder)
            quotas[r] += add
            remainder -= add

    corpus: List[str] = []
    counts: Dict[str, int] = {}
    for r, w in zip(REGISTERS, weights):
        pool = by_register.get(r, [])
        quota = quotas[r]
        if w > 0 and available[r] < 10:
            print(f"[Corpus] ⚠ register '{r}' has only {available[r]} lines; "
                  "raise --templates/--wikipedia or lower --romanize-rate/--code-mix-rate.")
        rng.shuffle(pool)
        chosen = pool[:quota]
        corpus.extend(chosen)
        counts[r] = len(chosen)
    rng.shuffle(corpus)
    return corpus, counts


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build a code-switched distillation corpus from real text + controlled mixing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out", type=str, default="codeswitch_corpus.txt")
    p.add_argument("--templates", type=int, default=2000,
                   help="Sentences from the deterministic conversational generator (0 disables).")
    p.add_argument("--wikipedia", type=int, default=0,
                   help="Real Tamil sentences from wikimedia/wikipedia (needs internet; 0 disables).")
    p.add_argument("--wikipedia-file", type=str, default="wikidata_tamil_sentences.txt",
                   help="Cache/alternative dump of Tamil sentences (one per line).")
    p.add_argument("--extra-file", action="append", default=[],
                   help="Additional corpus file (repeatable). JSON list or one sentence per line.")
    p.add_argument("--code-mix-rate", type=float, default=0.35)
    p.add_argument("--romanize-rate", type=float, default=0.30)
    p.add_argument("--numeric-rate", type=float, default=0.12,
                   help="Fraction of sentences rebuilt as OTP/fare/plate templates.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--passes", type=int, default=3,
                   help="Augmentation passes per base sentence (more passes = more candidates "
                        "per register before quota balancing).")
    p.add_argument("--balance", type=str, default="0.34,0.33,0.33",
                   help="Target register quota as native,romanized,transit weights.")
    p.add_argument("--max-sentences", type=int, default=0,
                   help="Cap the corpus size while keeping the register mix (0 = no cap). "
                        "The teacher pass costs ~7.9 s/sentence on a T4, so this is how the "
                        "corpus is made to fit one Kaggle session (e.g. 2000 -> ~4.5 h).")
    p.add_argument("--json-out", type=str, default=None,
                   help="Optional JSON copy ({text, register} objects).")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    balance = [float(x) for x in str(args.balance).split(",") if str(x).strip()]
    corpus, counts = build_corpus(
        n_templates=args.templates,
        n_wikipedia=args.wikipedia,
        extra_files=args.extra_file,
        code_mix_rate=args.code_mix_rate,
        romanize_rate=args.romanize_rate,
        numeric_rate=args.numeric_rate,
        seed=args.seed,
        wiki_cache=args.wikipedia_file,
        passes=args.passes,
        balance=balance,
        max_sentences=args.max_sentences,
    )
    if not corpus:
        print("[Corpus] ❌ no sentences assembled")
        return 2
    assert_corpus_disjoint_from_eval(corpus)
    out = Path(args.out)
    out.write_text("\n".join(corpus) + "\n", encoding="utf-8")
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps([{"text": t} for t in corpus], ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
    words = [len(s.split()) for s in corpus]
    print("\n" + "=" * 74)
    print(f"Corpus written: {out} | {len(corpus)} unique sentences | disjoint from eval set OK")
    print(f"  words/sentence: mean {sum(words) / len(words):.1f} "
          f"(min {min(words)}, max {max(words)})")
    total = max(1, sum(counts.values()))
    print("  register mix (classified from the final text): "
          + ", ".join(f"{k}={v} ({100.0 * v / total:.0f}%)" for k, v in sorted(counts.items())))
    print(f"  chars: {sum(len(s) for s in corpus)} | "
          f"estimated audio at "
          f"{sum(len(s) for s in corpus) / 14.0:.0f}s-equivalent of text")
    print("=" * 74)
    print("Next: python train_draft_distill.py --corpus-json " + str(out)
          + " --draft-epochs 60 --residual-sched 0.5 --max-minutes 40")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
