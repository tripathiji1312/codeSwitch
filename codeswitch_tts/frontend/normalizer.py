"""3-Tier Text Normalizer for Code-Mixed Tamil-English (Tanglish) Transit Domain.

Architecture:
  - Tier 1: Language Identification (LID) & Token Classification
  - Tier 2: Phonotactic Disambiguation & ISO 15919 Transliteration (preserving English transit loanwords)
  - Tier 3: Context-Aware Entity Verbalizer for Transit (OTPs digit-by-digit, vehicle plates, INR fares)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set

# Regex patterns
TAMIL_SCRIPT_RE = re.compile(r"[\u0B80-\u0BFF]")
LATIN_TOKEN_RE = re.compile(r"^[A-Za-z0-9'-]+$")

# Transit English loanwords that must NEVER be transliterated into garbled Tamil script
ENGLISH_TRANSIT_LOANWORDS: Set[str] = {
    "cab", "taxi", "driver", "pickup", "drop", "location", "fare", "total", "trip",
    "ride", "booking", "car", "auto", "vehicle", "traffic", "signal", "junction",
    "airport", "station", "platform", "flyover", "highway", "express", "delay",
    "delayed", "cancel", "cancelled", "confirm", "confirmed", "arrive", "arrived",
    "waiting", "service", "route", "map", "ac", "non-ac", "sedan", "hatchback",
    "suv", "toll", "charge", "charges", "cash", "online", "payment", "upi",
    "gpay", "phonepe", "paytm", "app", "phone", "call", "sir", "madam", "ok",
    "okay", "hi", "hello", "please", "thanks", "thank", "you", "ready", "start",
    "end", "speed", "km", "kms", "kilometer", "kilometers", "meter", "meters",
    "min", "mins", "minute", "minutes", "hr", "hrs", "hour", "hours", "otp"
}

# High-frequency Tanglish (Romanized Tamil) lexical dictionary
TANGLISH_LEXICON: Dict[str, str] = {
    "unga": "உங்க",
    "ungal": "உங்கள்",
    "ungala": "உங்களை",
    "ungalku": "உங்களுக்கு",
    "ungaal": "உங்களால்",
    "enga": "எங்க",
    "engal": "எங்கள்",
    "enge": "எங்கே",
    "engaey": "எங்கே",
    "ippo": "இப்போ",
    "ippozhudhu": "இப்பொழுது",
    "ippam": "இப்பம்",
    "appo": "அப்போ",
    "eppo": "எப்போ",
    "vandhuchu": "வந்துச்சு",
    "vandhuvittadhu": "வந்துவிட்டது",
    "vandhuruchu": "வந்துருச்சு",
    "varum": "வரும்",
    "varudhu": "வருது",
    "varen": "வரேன்",
    "vaanga": "வாங்க",
    "vanga": "வாங்க",
    "ponga": "போங்க",
    "poonga": "போங்க",
    "poga": "போக",
    "sollunga": "சொல்லுங்க",
    "solunga": "சொல்லுங்க",
    "solli": "சொல்லி",
    "pesunga": "பேசுங்க",
    "pesungal": "பேசுங்கள்",
    "kitta": "கிட்ட",
    "kitte": "கிட்டே",
    "irukku": "இருக்கு",
    "iruku": "இருக்கு",
    "irukanga": "இருக்காங்க",
    "illa": "இல்ல",
    "illai": "இல்லை",
    "seri": "சரி",
    "sari": "சரி",
    "romba": "ரொம்ப",
    "konjam": "கொஞ்சம்",
    "podu": "போடு",
    "pannu": "பண்ணு",
    "pannunga": "பண்ணுங்க",
    "panra": "பண்ற",
    "panren": "பண்றேன்",
    "enna": "என்ன",
    "edhu": "எது",
    "yaar": "யார்",
    "epdi": "எப்படி",
    "eppadi": "எப்படி",
    "indha": "இந்த",
    "andha": "அந்த",
    "idhu": "இது",
    "adhu": "அது",
    "naan": "நான்",
    "nee": "நீ",
    "neenga": "நீங்க",
    "namma": "நம்ம",
    "avanga": "அவங்க",
    "ivanga": "இவங்க",
    "aana": "ஆனா",
    "aanal": "ஆனால்",
    "apram": "அப்புறம்",
    "appuram": "அப்புறம்",
    "neram": "நேரம்",
    "dooram": "தூரம்",
    "mudindhadhu": "முடிந்தது",
    "mudinjadhu": "முடிஞ்சது",
    "vanakkam": "வணக்கம்",
    "nanri": "நன்றி",
    "dhan": "தான்",
    "thaan": "தான்",
}

# Common Tamil postpositions / case suffixes attached to English nouns (e.g. driver-ku)
TAMIL_SUFFIX_MAP: Dict[str, str] = {
    "-ku": "-க்கு",
    "ku": "-க்கு",
    "-la": "-ல",
    "la": "-ல",
    "-oda": "-ஓட",
    "oda": "-ஓட",
    "-kitta": "-கிட்ட",
    "kitta": "-கிட்ட",
    "-a": "-ஐ",
    "-lerundhu": "-லிருந்து",
    "lerundhu": "-லிருந்து",
}

# English digit words
EN_DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

EN_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen"
]

EN_TENS = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"
]

# Indian State Codes for Vehicle Plates
INDIAN_STATE_CODES = (
    "AN|AP|AR|AS|BR|CG|CH|DD|DL|DN|GA|GJ|HP|HR|JH|JK|KA|KL|LA|LD|"
    "MH|ML|MN|MP|MZ|NL|OD|OR|PB|PY|RJ|SK|TN|TR|TS|UA|UK|UP|WB|BH"
)

VEHICLE_PLATE_RE = re.compile(
    rf"\b({INDIAN_STATE_CODES})[- ]?(\d{{1,2}})[- ]?([A-Za-z]{{1,3}})?[- ]?(\d{{3,4}})\b",
    re.IGNORECASE,
)

OTP_EXPLICIT_RE = re.compile(r"\b(?:otp|code|pin)[-:\s]+(\d{3,6})\b", re.IGNORECASE)
FARE_RE = re.compile(r"\b(?:rs\.?|inr|₹)\s*(\d+(?:\.\d{1,2})?)\b", re.IGNORECASE)
FARE_SUFFIX_RE = re.compile(r"\b(\d+)\s*(?:rs\.?|rupees|rupee)\b", re.IGNORECASE)
TIME_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)(?:\s*(am|pm|a\.m\.|p\.m\.))?\b", re.IGNORECASE)
DISTANCE_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(km|kms|kilometers?|m|meters?)\b", re.IGNORECASE)


def english_int_to_words(n: int) -> str:
    """Convert an integer (0 to 999,999) to English words."""
    if n < 0:
        return "minus " + english_int_to_words(-n)
    if n < 20:
        return EN_ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return EN_TENS[tens] + ("-" + EN_ONES[ones] if ones > 0 else "")
    if n < 1000:
        hundreds, rem = divmod(n, 100)
        res = EN_ONES[hundreds] + " hundred"
        if rem > 0:
            res += " " + english_int_to_words(rem)
        return res
    if n < 1_000_000:
        thousands, rem = divmod(n, 1000)
        res = english_int_to_words(thousands) + " thousand"
        if rem > 0:
            res += " " + english_int_to_words(rem)
        return res
    return " ".join(EN_DIGITS[int(d)] for d in str(n))


def spell_digits(num_str: str) -> str:
    """Read a number sequence digit-by-digit."""
    return " ".join(EN_DIGITS[int(d)] for d in num_str if d.isdigit())


def spell_alphanumeric(code: str) -> str:
    """Spell an alphanumeric string clearly (e.g. 'TN01AB1234' -> 'T N zero one A B one two three four')."""
    out: List[str] = []
    for char in code.upper():
        if char.isdigit():
            out.append(EN_DIGITS[int(char)])
        elif char.isalpha():
            out.append(char)
    return " ".join(out)


@dataclass
class NormalizationResult:
    """Result of the 3-tier normalization pipeline."""
    original_text: str
    normalized_text: str
    detected_language: str
    entity_replacements: List[Tuple[str, str]] = field(default_factory=list)
    tokens: List[Tuple[str, str]] = field(default_factory=list)  # (token, label)


class TanglishNormalizer:
    """3-Tier Normalizer and Transliterator for Code-Mixed Tamil-English Transit Text."""

    def __init__(self):
        self.lexicon = TANGLISH_LEXICON
        self.loanwords = ENGLISH_TRANSIT_LOANWORDS

    # ─────────────────────────────────────────────────────────────────────────
    # Tier 1: Language & Script Identification
    # ─────────────────────────────────────────────────────────────────────────
    def identify_language(self, text: str) -> str:
        """Detect overall register: 'ta' (Tamil), 'en' (English), or 'code-mixed'."""
        tamil_chars = len(TAMIL_SCRIPT_RE.findall(text))
        latin_chars = sum(1 for ch in text if ch.isascii() and ch.isalpha())

        if tamil_chars > 0 and latin_chars > 0:
            return "code-mixed"
        if tamil_chars > 0:
            return "ta"
        
        # Check if Latin text contains Tanglish lexicon
        words = [w.strip(".,!?:;").lower() for w in text.split()]
        tanglish_count = sum(1 for w in words if w in self.lexicon)
        if tanglish_count > 0:
            return "code-mixed"
        return "en"

    # ─────────────────────────────────────────────────────────────────────────
    # Tier 3: Context-Aware Transit Entity Verbalizer
    # ─────────────────────────────────────────────────────────────────────────
    def verbalize_entities(self, text: str) -> Tuple[str, List[Tuple[str, str]]]:
        """Verbalizes OTPs, vehicle numbers, fares, time, and distances."""
        replacements: List[Tuple[str, str]] = []
        result = text

        # 1. OTPs & Verification Codes (digit-by-digit)
        def _replace_otp(m: re.Match) -> str:
            digits = m.group(1)
            spoken = f"OTP {spell_digits(digits)}"
            replacements.append((m.group(0), spoken))
            return spoken

        result = OTP_EXPLICIT_RE.sub(_replace_otp, result)

        # 2. Vehicle Registration Plates
        def _replace_plate(m: re.Match) -> str:
            state = m.group(1).upper()
            district = m.group(2)
            series = m.group(3).upper() if m.group(3) else ""
            num = m.group(4)
            plate_raw = f"{state} {district} {series} {num}".strip()
            spoken = spell_alphanumeric(plate_raw)
            replacements.append((m.group(0), spoken))
            return spoken

        result = VEHICLE_PLATE_RE.sub(_replace_plate, result)

        # 3. Currency / Fares in INR
        def _replace_fare(m: re.Match) -> str:
            val_str = m.group(1)
            if "." in val_str:
                rupees, paise = val_str.split(".", 1)
                r_num = int(rupees)
                p_num = int(paise[:2])
                r_word = english_int_to_words(r_num)
                spoken = f"{r_word} rupees"
                if p_num > 0:
                    spoken += f" and {english_int_to_words(p_num)} paise"
            else:
                r_num = int(val_str)
                spoken = f"{english_int_to_words(r_num)} rupees"
            replacements.append((m.group(0), spoken))
            return spoken

        result = FARE_RE.sub(_replace_fare, result)

        def _replace_fare_suffix(m: re.Match) -> str:
            val_str = m.group(1)
            r_num = int(val_str)
            spoken = f"{english_int_to_words(r_num)} rupees"
            replacements.append((m.group(0), spoken))
            return spoken

        result = FARE_SUFFIX_RE.sub(_replace_fare_suffix, result)

        # 4. Time
        def _replace_time(m: re.Match) -> str:
            h = int(m.group(1))
            minute = int(m.group(2))
            meridiem = m.group(3)
            h_word = english_int_to_words(h if h <= 12 else h - 12)
            m_word = "o'clock" if minute == 0 else english_int_to_words(minute)
            spoken = f"{h_word} {m_word}"
            if meridiem:
                spoken += " " + ("A M" if "a" in meridiem.lower() else "P M")
            replacements.append((m.group(0), spoken))
            return spoken

        result = TIME_RE.sub(_replace_time, result)

        # 5. Distance
        def _replace_dist(m: re.Match) -> str:
            num = m.group(1)
            unit = m.group(2).lower()
            if "." in num:
                int_part, dec_part = num.split(".")
                num_spoken = f"{english_int_to_words(int(int_part))} point {spell_digits(dec_part)}"
            else:
                num_spoken = english_int_to_words(int(num))
            unit_spoken = "kilometers" if "km" in unit else "meters"
            spoken = f"{num_spoken} {unit_spoken}"
            replacements.append((m.group(0), spoken))
            return spoken

        result = DISTANCE_RE.sub(_replace_dist, result)

        # 6. Any remaining isolated 4-digit numbers (often unlabelled OTPs or pin codes)
        def _replace_isolated_otp(m: re.Match) -> str:
            digits = m.group(1)
            spoken = spell_digits(digits)
            replacements.append((m.group(0), spoken))
            return spoken

        result = re.sub(r"(?<!\w)(\d{4})(?!\w)", _replace_isolated_otp, result)

        return result, replacements

    # ─────────────────────────────────────────────────────────────────────────
    # Tier 2: Phonotactic Disambiguation & Transliteration
    # ─────────────────────────────────────────────────────────────────────────
    def disambiguate_and_transliterate(self, text: str) -> str:
        """Disambiguate Romanized Tamil into Tamil script while strictly preserving
        English transit loanwords.
        """
        tokens = text.split()
        processed_tokens: List[str] = []

        for token in tokens:
            # Preserve punctuation at ends
            leading = re.match(r"^[^\w]+", token)
            trailing = re.search(r"[^\w]+$", token)
            prefix = leading.group(0) if leading else ""
            suffix = trailing.group(0) if trailing else ""
            core = token[len(prefix):len(token) - len(suffix)] if (prefix or suffix) else token

            if not core:
                processed_tokens.append(token)
                continue

            # Check if token is already Tamil Unicode
            if TAMIL_SCRIPT_RE.search(core):
                processed_tokens.append(token)
                continue

            lower_core = core.lower()

            # Check for suffixed English transit loanword: e.g. "driver-ku", "location-ku"
            suffixed_loanword = False
            for suf_en, suf_ta in TAMIL_SUFFIX_MAP.items():
                if lower_core.endswith(suf_en):
                    stem = lower_core[:-len(suf_en)].rstrip("-")
                    if stem in self.loanwords:
                        # Preserve original casing of stem, attach Tamil suffix
                        orig_stem = core[:len(stem)]
                        processed_tokens.append(f"{prefix}{orig_stem}{suf_ta}{suffix}")
                        suffixed_loanword = True
                        break
            if suffixed_loanword:
                continue

            # Check if pure English transit loanword (e.g. "cab", "pickup", "fare", "driver")
            if lower_core in self.loanwords:
                processed_tokens.append(token)
                continue

            # Check Tanglish dictionary
            if lower_core in self.lexicon:
                processed_tokens.append(f"{prefix}{self.lexicon[lower_core]}{suffix}")
                continue

            # Check if word has a suffix attached to a Tanglish word: e.g. "ungaluku"
            matched_tanglish_suffix = False
            for suf_en, suf_ta in TAMIL_SUFFIX_MAP.items():
                if lower_core.endswith(suf_en):
                    stem = lower_core[:-len(suf_en)].rstrip("-")
                    if stem in self.lexicon:
                        ta_stem = self.lexicon[stem]
                        processed_tokens.append(f"{prefix}{ta_stem}{suf_ta}{suffix}")
                        matched_tanglish_suffix = True
                        break
            if matched_tanglish_suffix:
                continue

            # Default: preserve token if English-like or unknown
            processed_tokens.append(token)

        return " ".join(processed_tokens)

    # ─────────────────────────────────────────────────────────────────────────
    # Full 3-Tier Execution Pipeline
    # ─────────────────────────────────────────────────────────────────────────
    def normalize(self, text: str) -> NormalizationResult:
        """Run the complete 3-tier normalization pipeline."""
        cleaned = text.strip()
        cleaned = re.sub(r"\s+", " ", cleaned)

        # Tier 1: LID
        lang = self.identify_language(cleaned)

        # Tier 3: Verbalize Transit Entities first so digits don't get broken by transliteration
        verbalized_text, replacements = self.verbalize_entities(cleaned)

        # Tier 2: Phonotactic Disambiguation & Transliteration
        normalized_text = self.disambiguate_and_transliterate(verbalized_text)

        # Clean multiple spaces and ensure proper ending punctuation
        normalized_text = re.sub(r"\s+", " ", normalized_text).strip()
        if not normalized_text.endswith((".", "!", "?", "।")):
            normalized_text += "."

        return NormalizationResult(
            original_text=text,
            normalized_text=normalized_text,
            detected_language=lang,
            entity_replacements=replacements,
        )
