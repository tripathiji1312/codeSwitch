"""Shared benchmark data for the CodeSwitch-TTS evaluation suite.

This module is imported by both `train_draft_distill.py` (distillation
companion) and `run_full_evaluation.py` (main evaluation / figure script).

Contents
--------
- ``EVAL_UTTERANCES``: the frozen 60-utterance evaluation set
  (20 native Tamil, 20 romanized Tanglish, 20 transit & commerce code-mixed).
  This set is the paper's test set: it must NEVER be used for distillation
  training (the training corpus generator is disjoint by construction and
  verified with ``assert_corpus_disjoint_from_eval``).
- ``build_training_corpus``: seeded, template-based generator that produces a
  bootstrap Tanglish/Tamil corpus for teacher self-distillation.  Swap in a
  real corpus with ``--corpus-json`` if available.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Dict, List, Sequence

# ---------------------------------------------------------------------------
# Frozen evaluation set (60 utterances)
# ---------------------------------------------------------------------------
# Registers: "native_tamil", "romanized_tanglish", "transit_codemixed"
EVAL_UTTERANCES: List[Dict[str, str]] = [
    # ---- Native Tamil (20) -------------------------------------------------
    {"id": "ta-01", "register": "native_tamil", "text": "வணக்கம், உங்கள் ஆர்டர் தயாராகிவிட்டது."},
    {"id": "ta-02", "register": "native_tamil", "text": "நான் ஐந்து நிமிடத்தில் வந்துவிடுகிறேன்."},
    {"id": "ta-03", "register": "native_tamil", "text": "இடது பக்கம் திரும்பி நேராகச் செல்லுங்கள்."},
    {"id": "ta-04", "register": "native_tamil", "text": "எங்கள் கடை இன்று மாலை ஆறு மணிக்கு மூடுகிறது."},
    {"id": "ta-05", "register": "native_tamil", "text": "அடுத்த சிக்னலில் வலது பக்கம் திரும்புங்கள்."},
    {"id": "ta-06", "register": "native_tamil", "text": "உங்கள் பில் தொகை இருநூற்று ஐம்பது ரூபாய்."},
    {"id": "ta-07", "register": "native_tamil", "text": "டாக்சி ஐந்து நிமிடத்தில் வரும், காத்திருங்கள்."},
    {"id": "ta-08", "register": "native_tamil", "text": "இந்த நெடுஞ்சாலையில் போக்குவரத்து அதிகம்."},
    {"id": "ta-09", "register": "native_tamil", "text": "மழை பெய்வதால் விமானம் தாமதமாகும்."},
    {"id": "ta-10", "register": "native_tamil", "text": "உங்கள் பயணம் இனிதே நிறைவடையட்டும்."},
    {"id": "ta-11", "register": "native_tamil", "text": "தயவுசெய்து கொஞ்சம் பொறுங்கள், நான் உதவி செய்கிறேன்."},
    {"id": "ta-12", "register": "native_tamil", "text": "இன்று இரவு உணவை வீட்டிலேயே சாப்பிடலாம்."},
    {"id": "ta-13", "register": "native_tamil", "text": "அந்தப் பழைய பாலம் இப்போது பழுதுபார்க்கப்படுகிறது."},
    {"id": "ta-14", "register": "native_tamil", "text": "நாளை காலை எட்டு மணிக்கு அலுவலகம் செல்கிறேன்."},
    {"id": "ta-15", "register": "native_tamil", "text": "எனக்கு இந்த முகவரி சரியாகப் புரியவில்லை."},
    {"id": "ta-16", "register": "native_tamil", "text": "அம்மா சமையல் செய்து கொண்டிருக்கிறார்."},
    {"id": "ta-17", "register": "native_tamil", "text": "உங்களுக்கு உதவி தேவைப்பட்டால் என்னை அழையுங்கள்."},
    {"id": "ta-18", "register": "native_tamil", "text": "நகரப் பேருந்து அடுத்த நிலையத்திற்குச் செல்கிறது."},
    {"id": "ta-19", "register": "native_tamil", "text": "குழந்தைகள் பூங்காவில் விளையாடிக் கொண்டிருக்கிறார்கள்."},
    {"id": "ta-20", "register": "native_tamil", "text": "கடைசி நேர மாற்றத்திற்கு மன்னிக்கவும்."},
    # ---- Romanized Tanglish (20) ------------------------------------------
    {"id": "rt-01", "register": "romanized_tanglish", "text": "Unga order vandhuduchu, gate-ku vaanga."},
    {"id": "rt-02", "register": "romanized_tanglish", "text": "Enakku route therila, map paarunga."},
    {"id": "rt-03", "register": "romanized_tanglish", "text": "Naan five minutes-la varen, wait pannunga."},
    {"id": "rt-04", "register": "romanized_tanglish", "text": "Auto ippo junction-la nikkudhu."},
    {"id": "rt-05", "register": "romanized_tanglish", "text": "Saaptiya? Illa na saaptutu varen."},
    {"id": "rt-06", "register": "romanized_tanglish", "text": "Cab driver phone eduthu pannala."},
    {"id": "rt-07", "register": "romanized_tanglish", "text": "Intha address correct-a theriyala."},
    {"id": "rt-08", "register": "romanized_tanglish", "text": "Konjam slow-a pesunga, purila."},
    {"id": "rt-09", "register": "romanized_tanglish", "text": "Nethu night romba mazhai vandhuchu."},
    {"id": "rt-10", "register": "romanized_tanglish", "text": "Office-ku late aagum, traffic romba jasthi."},
    {"id": "rt-11", "register": "romanized_tanglish", "text": "Amma veetla irukanga, naan ippo varen."},
    {"id": "rt-12", "register": "romanized_tanglish", "text": "Inga left turn pannunga, apram right."},
    {"id": "rt-13", "register": "romanized_tanglish", "text": "Ennoda booking cancel pannunga please."},
    {"id": "rt-14", "register": "romanized_tanglish", "text": "Bus stop-la ten minutes wait panna sollunga."},
    {"id": "rt-15", "register": "romanized_tanglish", "text": "Evening six manikku meet panlama?"},
    {"id": "rt-16", "register": "romanized_tanglish", "text": "Avan veetuku poyittan, naalaiku varen."},
    {"id": "rt-17", "register": "romanized_tanglish", "text": "Ange poi parcel vaangikonga."},
    {"id": "rt-18", "register": "romanized_tanglish", "text": "Ticket rate konjam kammiya irukku."},
    {"id": "rt-19", "register": "romanized_tanglish", "text": "Intha shop-la discount kidaikuma?"},
    {"id": "rt-20", "register": "romanized_tanglish", "text": "Weekend-la park-ku polam, seriya?"},
    # ---- Transit & commerce code-mixed (20) --------------------------------
    {"id": "tc-01", "register": "transit_codemixed", "text": "Driver-ku OTP 8421 sollunga, fare Rs. 350."},
    {"id": "tc-02", "register": "transit_codemixed", "text": "Car number TN 09 AB 1234 pickup point-la nikkudhu."},
    {"id": "tc-03", "register": "transit_codemixed", "text": "Auto meter-la Rs. 245 kaattuthu, UPI-la anuppunga."},
    {"id": "tc-04", "register": "transit_codemixed", "text": "Order ID 7834 confirm aagiduchu, ETA seven minutes."},
    {"id": "tc-05", "register": "transit_codemixed", "text": "Bus number 21G Koyambedu stop-la vandhuchu."},
    {"id": "tc-06", "register": "transit_codemixed", "text": "Hotel room 402-ku check-in time two o'clock."},
    {"id": "tc-07", "register": "transit_codemixed", "text": "Package tracking number 5591 courier office-la irukku."},
    {"id": "tc-08", "register": "transit_codemixed", "text": "Train 12675 platform 7-la nikkuthu, late illa."},
    {"id": "tc-09", "register": "transit_codemixed", "text": "Flight six thirty-ku depart aagum, gate number 12."},
    {"id": "tc-10", "register": "transit_codemixed", "text": "Meter reading Rs. 180.50, cash illa na GPay pannunga."},
    {"id": "tc-11", "register": "transit_codemixed", "text": "Booking reference 4127-la seat 23A confirm."},
    {"id": "tc-12", "register": "transit_codemixed", "text": "Toll charge Rs. 85, fastag-la cut aagum."},
    {"id": "tc-13", "register": "transit_codemixed", "text": "Cab TN 22 CD 5678 five minutes-la vandhudum."},
    {"id": "tc-14", "register": "transit_codemixed", "text": "PIN 600028 area-la delivery two days aagum."},
    {"id": "tc-15", "register": "transit_codemixed", "text": "Aadhaar last four digits 9932 confirm pannunga."},
    {"id": "tc-16", "register": "transit_codemixed", "text": "Wallet balance Rs. 1250, recharge pannunga please."},
    {"id": "tc-17", "register": "transit_codemixed", "text": "Parcel weight 2.5 kg, charge Rs. 120."},
    {"id": "tc-18", "register": "transit_codemixed", "text": "Auto fare Rs. 95, meter-la add pannunga."},
    {"id": "tc-19", "register": "transit_codemixed", "text": "OTP 3312 one minute-la expire aagum, fast-a sollunga."},
    {"id": "tc-20", "register": "transit_codemixed", "text": "Parking slot B 47-la car park pannunga."},
]

REGISTER_LABELS: Dict[str, str] = {
    "native_tamil": "Native Tamil",
    "romanized_tanglish": "Romanized Tanglish",
    "transit_codemixed": "Transit Code-Mixed",
}

# ---------------------------------------------------------------------------
# Training corpus generator (distillation bootstrap; disjoint from eval set)
# ---------------------------------------------------------------------------

_TAMIL_TEMPLATES: List[str] = [
    "வணக்கம் {name}, உங்கள் {noun} தயாராகிவிட்டது.",
    "நான் {mins} நிமிடத்தில் வந்துவிடுகிறேன், {name}.",
    "இடது பக்கம் திரும்பி {place} வரை நேராகச் செல்லுங்கள்.",
    "{place} கடை இன்று {time} மணிக்கு மூடுகிறது.",
    "அடுத்த சிக்னலில் வலது பக்கம் திரும்புங்கள்.",
    "உங்கள் மொத்த தொகை {ta_num} ரூபாய்.",
    "டாக்சி இன்னும் {mins} நிமிடத்தில் வரும், காத்திருங்கள்.",
    "{place} சாலையில் போக்குவரத்து நெரிசல் அதிகம்.",
    "மழை பெய்வதால் வண்டி தாமதமாகும்.",
    "உங்கள் பயணம் இனிதே நிறைவடையட்டும்.",
    "தயவுசெய்து சிறிது பொறுங்கள், நான் உதவுகிறேன்.",
    "இன்று இரவு உணவை வீட்டிலேயே சாப்பிடுவோம்.",
    "நாளை காலை {time} மணிக்கு அலுவலகம் செல்கிறேன்.",
    "எனக்கு இந்த முகவரி சரியாகப் புரியவில்லை.",
    "உதவி தேவைப்பட்டால் என்னை அழையுங்கள்.",
    "பேருந்து அடுத்த நிலையத்திற்குச் செல்கிறது.",
    "குழந்தைகள் பூங்காவில் விளையாடுகிறார்கள்.",
    "சிறிது நேரத்தில் மீண்டும் அழைக்கிறேன்.",
    "இந்தப் பகுதியில் புதிய கடை திறக்கப்பட்டுள்ளது.",
    "நாளை மறுநாள் வெளியூர் பயணம் உள்ளது.",
    "{place} வழியாகச் செல்வது வேகமாக இருக்கும்.",
]

#: Tamil-script-only templates with slots, so the *unique* yield of the native
#: register is combinatorial rather than a handful of fixed sentences.
_TAMIL_SLOT_TEMPLATES: List[str] = [
    "{place_ta} போய் சேர்ந்ததும் எனக்கு ஒரு அழைப்பு போடுங்கள்.",
    "இந்த {shop_ta} {time_ta} மணிக்கு திறக்கும்.",
    "உங்கள் {noun_ta} {day_ta} காலையில் வந்துவிடும்.",
    "{place_ta} வரைக்கும் ஆட்டோவில் போனால் {amount_ta} ரூபாய் ஆகும்.",
    "நாளைக்கு {shop_ta} மூடியிருக்கும், முன்பே வாங்கிக் கொள்ளுங்கள்.",
    "{name_ta}, நீங்கள் சொன்ன {noun_ta} இன்னும் தயாராகவில்லை.",
    "சுமார் {min_ta} நிமிடத்தில் வந்து விடுவேன்.",
    "இந்த {noun_ta} யாருடையது என்று சொல்லுங்கள்.",
    "{place_ta} பக்கம் திரும்பினால் விரைவாக போகலாம்.",
    "{day_ta} {time_ta} மணிக்கு வருகிறேன்.",
    "தயவுசெய்து {min_ta} நிமிடம் காத்திருங்கள்.",
    "உங்கள் {noun_ta} க்கு {amount_ta} ரூபாய் ஆகும்.",
    "எனக்கு {place_ta} வழி தெரியாது, வழி சொல்லுங்கள்.",
    "இந்த {shop_ta} நல்ல தரமானது.",
    "{name_ta}, {noun_ta} கொஞ்சம் தாமதம் ஆகும்.",
    "நான் {day_ta} {place_ta} போகிறேன்.",
    "அந்த {noun_ta} எங்கே இருக்கிறது என்று பாருங்கள்.",
    "எங்கள் {shop_ta} {time_ta} மணி வரை திறந்திருக்கும்.",
    "இந்த பகுதியில் {place_ta} எங்கே இருக்கிறது?",
    "{min_ta} நிமிடத்தில் {place_ta} சேர்ந்து விடுவோம்.",
    "{place_ta} சாலையில் இன்று நெரிசல் அதிகம்.",
    "உங்கள் {noun_ta} நேற்றே அனுப்பப்பட்டது, நாளை வந்து சேரும்.",
    "நான் {time_ta} மணிக்கு {shop_ta} அருகில் இருப்பேன்.",
    "எங்கள் {shop_ta} க்கு {day_ta} வருகை அதிகம்.",
    "{name_ta}, {place_ta} வழியா வருகிறீர்களா?",
    "இந்த {noun_ta} விலை {amount_ta} ரூபாய் என்று நினைக்கிறேன்.",
    "தினமும் {time_ta} மணிக்கு பேருந்து வரும்.",
    "என் {noun_ta} இன்னும் இரண்டு நாள் ஆகும்.",
    "{place_ta} ல் புதிய {shop_ta} ஒன்று திறந்திருக்கிறது.",
    "{min_ta} நிமிடத்திற்கு ஒரு முறை பேருந்து வரும்.",
]

_TANGLISH_TEMPLATES: List[str] = [
    "Unga {noun} vandhuduchu, {name}.",
    "Enakku route therila, map paarunga.",
    "Naan {min_en} minutes-la varen, wait pannunga.",
    "Auto ippo {place} junction-la nikkudhu.",
    "Saaptiya? Illa na saaptutu varen.",
    "Cab driver phone eduthu pannala.",
    "Intha address correct-a theriyala.",
    "Konjam slow-a pesunga, purila.",
    "Nethu night romba mazhai vandhuchu.",
    "Office-ku late aagum, traffic romba jasthi.",
    "Amma veetla irukanga, naan ippo varen.",
    "Inga left turn pannunga, apram right.",
    "Ennoda booking cancel pannunga please.",
    "Bus stop-la {min_en} minutes wait panna sollunga.",
    "Evening {time} manikku meet panlama?",
    "Ange poi parcel vaangikonga.",
    "Ticket rate konjam kammiya irukku.",
    "Intha shop-la discount kidaikuma?",
    "Weekend-la park-ku polam, seriya?",
    "Innoru {mins} nimisham-le reach aayiduven.",
    "Neenga {place} side-a vandhinga na correct.",
    "Sari, naan udane vanthu paakren.",
    "Phone-la location share pannunga.",
    "{noun} ready aagiduchu, vaangikonga.",
]

_TRANSIT_TEMPLATES: List[str] = [
    "Driver-ku OTP {otp} sollunga, fare Rs. {fare}.",
    "Car number {plate} pickup point-la nikkudhu.",
    "Auto meter-la Rs. {fare} kaattuthu, UPI-la anuppunga.",
    "Order ID {order} confirm aagiduchu, ETA {min_en} minutes.",
    "Bus number {bus} {place} stop-la vandhuchu.",
    "Hotel room {room}-ku check-in time {time} o'clock.",
    "Package tracking number {order} courier office-la irukku.",
    "Train {train} platform {plat}-la nikkuthu, late illa.",
    "Booking reference {order}-la seat {seat} confirm.",
    "Toll charge Rs. {fare}, fastag-la cut aagum.",
    "Cab {plate} {min_en} minutes-la vandhudum.",
    "PIN {pin} area-la delivery two days aagum.",
    "Wallet balance Rs. {fare}, recharge pannunga please.",
    "Parcel weight {kg} kg, charge Rs. {fare}.",
    "Auto fare Rs. {fare}, meter-la add pannunga.",
    "OTP {otp} one minute-la expire aagum, fast-a sollunga.",
    "Parking slot {slot}-la car park pannunga.",
    "Flight {time}-ku depart aagum, gate number {plat}.",
    "Aadhaar last four digits {otp4} confirm pannunga.",
    "Meter reading Rs. {fare}, cash illa na GPay pannunga.",
    "Courier {order} address-ku {min_en} minutes-la varum.",
    "Ride fare estimate Rs. {fare}, confirm pannunga.",
]

_SLOTS: Dict[str, List[str]] = {
    "name": ["sir", "madam", "boss", "unga friend", "driver", "customer"],
    "name_ta": ["சார்", "மேடம்", "அண்ணா", "அக்கா", "தம்பி", "தங்கச்சி"],
    "noun": ["order", "parcel", "booking", "food", "ticket", "payment"],
    "noun_ta": ["ஆர்டர்", "பார்சல்", "புக்கிங்", "பில்", "டிக்கெட்", "பணம்",
                "மளிகை பொருள்", "மருந்து"],
    "shop_ta": ["கடை", "ஹோட்டல்", "மருத்துவமனை", "வங்கி", "சந்தை", "பள்ளி",
                "பேக்கரி", "பூங்கா"],
    "place": ["T Nagar", "Anna Nagar", "Velachery", "Guindy", "Chromepet",
              "Tambaram", "Adyar", "Koyambedu", "Egmore", "Mylapore",
              "Porur", "Ambattur", "Perungudi", "Pallikaranai"],
    "place_ta": ["தி நகர்", "அண்ணா நகர்", "வேளச்சேரி", "கிண்டி", "சுரேந்திர நகர்",
                 "தாம்பரம்", "அடையாறு", "கோயம்பேடு", "எழும்பூர்", "மயிலாப்பூர்",
                 "போரூர்", "அம்பத்தூர்", "பெருங்குடி", "பள்ளிக்கரணை"],
    "min_en": ["two", "three", "five", "seven", "ten", "fifteen"],
    "mins": ["இரண்டு", "மூன்று", "ஐந்து", "ஏழு", "பத்து"],
    "min_ta": ["இரண்டு", "மூன்று", "ஐந்து", "ஏழு", "பத்து", "பதினைந்து"],
    "time": ["six", "seven", "eight", "nine", "ten"],
    "time_ta": ["ஆறு", "ஏழு", "எட்டு", "ஒன்பது", "பத்து", "பதினொன்று"],
    "day_ta": ["இன்று", "நாளை", "நேற்று", "இன்று மாலை", "நாளை காலை",
               "வாரம் இறுதியில்"],
    "ta_num": ["இருநூறு", "முந்நூறு", "நானூறு", "ஐந்நூறு", "எண்ணூறு"],
    "amount_ta": ["நூறு", "இருநூறு", "முந்நூறு", "நானூறு", "ஐந்நூறு", "எண்ணூறு",
                  "ஆயிரம்", "ஆயிரத்து ஐந்நூறு"],
    "kg": ["1.5", "2.5", "3", "5"],
    "slot": ["A 12", "B 47", "C 09", "D 23"],
    "seat": ["12A", "23B", "7C", "15D"],
    "plat": ["3", "5", "7", "9"],
    "bus": ["21G", "45B", "58K", "12M"],
    "train": ["12675", "22631", "12695", "16057"],
    "room": ["201", "305", "402", "518"],
}

_ENTITY_RE = re.compile(r"\{(\w+)\}")
_NUM_TEMPLATES = re.compile(r"\{(\w+)\}")


def _normalize_for_compare(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().casefold())


def _fill_template(template: str, rng: random.Random) -> str:
    def _repl(match: re.Match) -> str:
        slot = match.group(1)
        if slot == "otp":
            return str(rng.randint(1000, 9999))
        if slot == "otp4":
            return str(rng.randint(1000, 9999))
        if slot == "fare":
            return str(rng.choice([45, 60, 75, 85, 95, 120, 150, 180, 245, 280,
                                   350, 420, 599, 750, 1200, 1250]))
        if slot == "order":
            return str(rng.randint(1000, 9999))
        if slot == "pin":
            return str(rng.choice([600028, 600042, 600096, 600119, 560001, 400001]))
        if slot == "plate":
            state = rng.choice(["TN 09", "TN 22", "KA 05", "KA 51", "AP 31"])
            return f"{state} {rng.choice(['AB', 'CD', 'EF', 'GH'])} {rng.randint(1000, 9999)}"
        pool = _SLOTS.get(slot)
        if pool is None:
            return match.group(0)
        return rng.choice(pool)

    return _ENTITY_RE.sub(_repl, template)


def build_training_corpus(n: int = 400, seed: int = 7) -> List[str]:
    """Deterministic, template-based bootstrap corpus for teacher distillation.

    The generated corpus is guaranteed disjoint from ``EVAL_UTTERANCES``.
    """
    rng = random.Random(seed)
    eval_norm = {_normalize_for_compare(u["text"]) for u in EVAL_UTTERANCES}
    pool = (_TAMIL_TEMPLATES + _TAMIL_SLOT_TEMPLATES + _TANGLISH_TEMPLATES
            + _TRANSIT_TEMPLATES)
    corpus: List[str] = []
    seen: set[str] = set()
    attempts = 0
    while len(corpus) < n and attempts < n * 60:
        attempts += 1
        sentence = _fill_template(rng.choice(pool), rng)
        key = _normalize_for_compare(sentence)
        if key in seen or key in eval_norm:
            continue
        seen.add(key)
        corpus.append(sentence)
    if len(corpus) < n:
        raise RuntimeError(
            f"Corpus generator produced only {len(corpus)}/{n} unique sentences; "
            "increase template diversity."
        )
    rng.shuffle(corpus)
    return corpus


def assert_corpus_disjoint_from_eval(corpus: Sequence[str]) -> None:
    eval_norm = {_normalize_for_compare(u["text"]) for u in EVAL_UTTERANCES}
    overlap = [s for s in corpus if _normalize_for_compare(s) in eval_norm]
    if overlap:
        raise ValueError(
            "Training/eval leakage detected: the training corpus contains "
            f"evaluation sentences: {overlap[:3]}"
        )


def load_corpus_json(path: str | Path) -> List[str]:
    """Loads an external training corpus (one sentence per line or JSON list)."""
    import json

    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    stripped = raw.lstrip()
    if stripped.startswith("["):
        data = json.loads(raw)
        corpus = [str(x["text"]) if isinstance(x, dict) else str(x) for x in data]
    else:
        corpus = [line.strip() for line in raw.splitlines()]
    corpus = [line for line in corpus if line and not line.startswith("#")]
    if not corpus:
        raise ValueError(f"No sentences found in corpus file: {p}")
    return corpus


def eval_register_counts() -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for utt in EVAL_UTTERANCES:
        counts[utt["register"]] = counts.get(utt["register"], 0) + 1
    return counts


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    print(f"Eval utterances: {len(EVAL_UTTERANCES)} | registers: {eval_register_counts()}")
    corpus = build_training_corpus()
    assert_corpus_disjoint_from_eval(corpus)
    print(f"Bootstrap training corpus: {len(corpus)} unique sentences (disjoint OK)")
    for s in corpus[:5]:
        print("  -", s)
    for s in corpus[-5:]:
        print("  -", s)
