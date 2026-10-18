# Plan 009: Pin the normalizer's exact output strings, make `--json-out` write the stats its help promises, and smoke-test `kaggle_demo.py`

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**: `git diff --stat d69b399..HEAD -- tests/test_codeswitch_tts.py tests/test_corpus_builder.py build_codeswitch_corpus.py codeswitch_benchmark_data.py kaggle_demo.py KAGGLE_RUN.md`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; locate
> snippets by content (line numbers may have shifted). If a quoted snippet is
> gone or changed, treat it as a STOP condition.

## Status

- **Priority**: P2
- **Effort**: S–M
- **Risk**: LOW — one writer's output shape changes (fixes DOCS-06), one
  loader branch is extended, everything else is tests. The loaders'
  existing contracts are kept: the JSON-list branch of `load_extra_files`
  and the txt/JSON-list contract of `load_corpus_json` are untouched.
- **Depends on**: none
- **Category**: tests + docs-behavior (DOCS-06)
- **Planned at**: commit `d69b399`, 2026-10-19

## Why this matters

1. **The normalizer is the front door of every dataset row and every CER
   metric, and its exact output strings are effectively untested.** The
   existing suite covers transit entities only (`tests/test_codeswitch_tts.py:18-46`).
   Seven high-traffic transformations (digit strings, times, decimals,
   currency, bare numbers, code-mixed units, the flagship transit sentence)
   were verified live at `d69b399` and are pinned here as exact strings, so
   a regex tweak that silently shifts `Jam 5:30` → `five:thirty` or adds a
   period is caught by tests instead of by a CER column moving in the paper.
2. **`--json-out` contradicts its own help text and the docs' name for it.**
   Help says "Optional JSON copy ({text, register} objects)"
   (`build_codeswitch_corpus.py:451-452`) but the writer emits a bare
   `[{"text": t}]` list with no register and no stats (`:478-482`), while all
   three docs name the file `codeswitch_corpus.stats.json`
   (`KAGGLE_RUN.md:107`, `PAPER_CLAIMS_AND_EVIDENCE.md:59`, notebook cell 16)
   — so anyone opening the saved "stats" file finds no stats at all
   (audit finding DOCS-06). Fix: write `{n, counts, sentences:[{text,
   register}]}` (the register mix was already computed and is only printed
   to stdout today) and make `load_extra_files` accept the dict shape, so
   the stats file round-trips into `--extra-file`.
3. **`kaggle_demo.py` — the first script every newcomer runs — has no test.**
   With `CODESWITCH_ALLOW_UNTRAINED=1` it is a pure-CPU, no-network wiring
   smoke that exercises the whole package import graph, the pipeline
   fallback, and WAV output. A subprocess smoke test locks that behavior.

Evidence: `plans/research/findings-2026-10-18.md`, 2026-10-19 entry (claims
23, 26, 32, 33; normalizer outputs) and the docs audit (DOCS-06, DOCS-09).

## Current state

- `tests/test_codeswitch_tts.py` — imports at `:1-14` include `pytest`,
  `torch`, `numpy`, `Path`, `TanglishNormalizer`. `test_normalizer_transit_entities`
  ends at `:46` (`assert "Driver-க்கு" in res.normalized_text`), and
  `def test_draft_transformer_forward():` starts at `:49` — the new block
  goes between them. The telemetry-only pipeline test is
  `test_end_to_end_streaming_pipeline` at `:98-111`.
- Normalizer API: `TanglishNormalizer().normalize(text)` returns a
  `NormalizationResult` with `.normalized_text` / `.detected_language` /
  `.entity_replacements` (`codeswitch_tts/frontend/normalizer.py:196-233, 426`).
- **Verified outputs at `d69b399`** (run live with `movio/.venv/bin/python`,
  recorded in the 2026-10-19 log entry):

  | input | `.normalized_text` |
  |---|---|
  | `TN 09 AB 1234` | `T N zero nine A B one two three four.` |
  | `5:30 pm` | `five thirty P M.` |
  | `2.5 km` | `two point five kilometers.` |
  | `Rs. 420` | `four hundred twenty rupees.` |
  | `8421` | `eight four two one.` |
  | `2 km தூரம்` | `two kilometers தூரம்.` |
  | `Driver-ku OTP 5291 sollunga, total fare Rs. 245.` | `Driver-க்கு OTP five two nine one சொல்லுங்க, total fare two hundred forty-five rupees.` |

- `build_codeswitch_corpus.py` — `load_extra_files` at `:255-283` (JSON list
  branch detects `stripped.startswith("[")`, elements are dicts with a
  `"text"` key or bare strings; txt fallback is `raw.splitlines()`);
  `--json-out` argparse at `:451-452`; writer at `:478-482`:
  ```python
  if args.json_out:
      Path(args.json_out).write_text(
          json.dumps([{"text": t} for t in corpus], ensure_ascii=False, indent=1),
          encoding="utf-8",
      )
  ```
  `main(argv=None)` is at `:454`; `classify_register` is public (used by
  `tests/test_corpus_builder.py:103`); `REGISTERS` is public (used at `:113`).
- `tests/test_corpus_builder.py` — the extra-file test is
  `test_corpus_extra_file_accepts_txt_and_json` at `:139-148`
  (asserts 3 loaded, missing file → `[]`).
- `kaggle_demo.py` — env guard at `:75-77`:
  ```python
  if os.environ.get("CODESWITCH_ALLOW_UNTRAINED", "0") == "1":
      # Telemetry-only smoke test: random-weight modules, audio is NOT speech.
      config.system.allow_untrained_fallback = True
  ```
  it saves WAVs under `./output_audio/` (printed at `:218`) and ends with
  `if __name__ == "__main__": run_benchmark()` (`:220-222`).
- `KAGGLE_RUN.md:427` — hand-maintained test count (98 at `d69b399`).
- Keep the round-trip assertion at `tests/test_corpus_builder.py:129-130`
  (`data.load_corpus_json(out) == corpus`) green — that loader's txt
  contract is not changed by this plan.

## Commands you will need

| Purpose | Command | Expected |
|---|---|---|
| Normalizer tests | `movio/.venv/bin/python -m pytest tests/test_codeswitch_tts.py -q -p no:cacheprovider -k normalizer` | all pass |
| Corpus tests | `movio/.venv/bin/python -m pytest tests/test_corpus_builder.py -q -p no:cacheprovider` | previous count + 1 |
| Demo smoke only | `movio/.venv/bin/python -m pytest tests/test_codeswitch_tts.py -q -p no:cacheprovider -k kaggle_demo` | `1 passed` (CPU, no network; allow up to ~2 min) |
| Full suite | `movio/.venv/bin/python -m pytest tests/ -q -p no:cacheprovider` | previous count + 9 |

Never use `movio/.venv/bin/pytest` (broken shebang) — always
`movio/.venv/bin/python -m pytest`.

## Scope

**In scope**

- New parametrized normalizer test after `tests/test_codeswitch_tts.py:46`.
- `--json-out` help text + writer change in `build_codeswitch_corpus.py`.
- `load_extra_files` dict-root branch in the same file.
- One new corpus test + one demo smoke test.
- `KAGGLE_RUN.md:427` count update.

**Out of scope (do not touch)**

- `codeswitch_benchmark_data.load_corpus_json` (`:321-337`) — the training
  loader keeps its txt / JSON-list contract; the new stats file is a
  `--extra-file`-style artifact, not a `--corpus-json` input (see
  Maintenance notes).
- `kaggle_demo.py` itself — no source change, only a test.
- Any doc text: after this change the docs' name `codeswitch_corpus.stats.json`
  finally matches the file, so no doc edit is needed (plan 011 re-checks the
  wording).

## Git workflow

Branch: `advisor/009-normalizer-and-cli-tests`. Commits:

1. `Pin the normalizer's exact output strings`
2. `Make --json-out write n, counts and per-sentence registers`
3. `Accept the stats dict in load_extra_files and cover the corpus CLI`
4. `Smoke-test kaggle_demo with allow-untrained`

## Steps

### Step 1 — Parametrized exact-string normalizer test

Insert after the last line of `test_normalizer_transit_entities`
(`tests/test_codeswitch_tts.py:46`) and before
`def test_draft_transformer_forward():`:

```python
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("TN 09 AB 1234", "T N zero nine A B one two three four."),
        ("5:30 pm", "five thirty P M."),
        ("2.5 km", "two point five kilometers."),
        ("Rs. 420", "four hundred twenty rupees."),
        ("8421", "eight four two one."),
        ("2 km தூரம்", "two kilometers தூரம்."),
        (
            "Driver-ku OTP 5291 sollunga, total fare Rs. 245.",
            "Driver-க்கு OTP five two nine one சொல்லுங்க, "
            "total fare two hundred forty-five rupees.",
        ),
    ],
)
def test_normalizer_exact_strings(text, expected):
    """Exact-string pin: these outputs feed every corpus row and every CER
    computation, so a regex tweak must fail here, not in a paper column."""
    assert TanglishNormalizer().normalize(text).normalized_text == expected
```

**Verify**

```bash
cd /home/tripathiji/projects/all_work
movio/.venv/bin/python -m pytest tests/test_codeswitch_tts.py -q -p no:cacheprovider -k normalizer
```

Expected: all normalizer tests pass (the 7 new cases plus the existing
ones). If any expected string differs, STOP — the normalizer changed since
this plan was verified; report the actual output instead of editing the
expectation.

### Step 2 — Make `--json-out` write what its help promises

**2a.** Help text (currently `:451-452`):

```python
    p.add_argument("--json-out", type=str, default=None,
                   help="Optional JSON copy ({text, register} objects).")
```

→

```python
    p.add_argument("--json-out", type=str, default=None,
                   help="Optional JSON stats copy: "
                        "{n, counts, sentences: [{text, register}]}.")
```

**2b.** Writer (currently `:478-482`):

```python
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps([{"text": t} for t in corpus], ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
```

→

```python
    if args.json_out:
        payload = {
            "n": len(corpus),
            "counts": {str(k): int(v) for k, v in counts.items()},
            "sentences": [{"text": t, "register": classify_register(t)} for t in corpus],
        }
        Path(args.json_out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
```

(`counts` is the register-count dict already computed by `build_corpus` and
printed at `:486-493`; `classify_register` is the same classifier the
builder uses at `:103`. `str(k)` guards non-str keys for JSON.)

**Verify**

```bash
cd /home/tripathiji/projects/all_work
movio/.venv/bin/python build_codeswitch_corpus.py --templates 60 --wikipedia 0 \
    --out /tmp/opencode/corpus_check.txt --json-out /tmp/opencode/corpus_check.json --seed 5
movio/.venv/bin/python - <<'EOF'
import json
p = json.load(open("/tmp/opencode/corpus_check.json", encoding="utf-8"))
assert set(p) == {"n", "counts", "sentences"}, p.keys()
assert p["n"] == len(p["sentences"]) == sum(p["counts"].values())
assert all(set(s) == {"text", "register"} for s in p["sentences"])
print("ok", p["n"], p["counts"])
EOF
```

Expected: `ok <N> {...}` with the register counts. (This is a CLI-level
check of the same contract the new test pins. Keep outputs under
`/tmp/opencode`; delete them afterwards.)

### Step 3 — Extend `load_extra_files` to accept the stats dict

In `build_codeswitch_corpus.py`, `load_extra_files` currently branches on
`stripped.startswith("[")` and otherwise treats the file as txt lines. Add
a dict branch **before** the list branch so both JSON shapes work:

```python
        stripped = raw.lstrip()
        if stripped.startswith("{"):
            try:
                payload = json.loads(raw)
                sentences = payload.get("sentences") if isinstance(payload, dict) else None
                if isinstance(sentences, list):
                    lines = [str(x["text"]) if isinstance(x, dict) else str(x) for x in sentences]
                else:
                    lines = raw.splitlines()
            except Exception:
                lines = raw.splitlines()
        elif stripped.startswith("["):
            try:
                data = json.loads(raw)
                lines = [str(x["text"]) if isinstance(x, dict) else str(x) for x in data]
            except Exception:
                lines = raw.splitlines()
        else:
            lines = raw.splitlines()
```

(Only the new `if` branch is added; the existing `elif`/`else` bodies are
unchanged — the JSON-list test at `tests/test_corpus_builder.py:139-148`
must stay green.)

**Verify**

```bash
cd /home/tripathiji/projects/all_work
movio/.venv/bin/python - <<'EOF'
import json, tempfile, pathlib
import build_codeswitch_corpus as B
tmp = pathlib.Path(tempfile.mkdtemp())
js = tmp / "stats.json"
js.write_text(json.dumps({"n": 2, "counts": {"native_tamil": 2}, "sentences": [
    {"text": "நாளை காலையில் வருகிறேன்.", "register": "native_tamil"},
    {"text": "இன்று மழை பெய்யும்.", "register": "native_tamil"}]}, ensure_ascii=False), encoding="utf-8")
loaded = B.load_extra_files([str(js)])
assert loaded == ["நாளை காலையில் வருகிறேன்.", "இன்று மழை பெய்யும்."], loaded
print("ok")
EOF
```

Expected: `ok`. Also run the existing corpus tests (see Step 3b below).

### Step 3b — Corpus CLI test (append to `tests/test_corpus_builder.py`)

Append after `test_corpus_extra_file_accepts_txt_and_json`:

```python
def test_json_out_writes_stats_and_round_trips_through_extra_file(tmp_path: Path):
    out = tmp_path / "corpus.txt"
    stats = tmp_path / "corpus.stats.json"
    rc = B.main(["--templates", "120", "--wikipedia", "0",
                 "--out", str(out), "--json-out", str(stats), "--seed", "5"])
    assert rc == 0
    payload = json.loads(stats.read_text(encoding="utf-8"))
    assert set(payload) == {"n", "counts", "sentences"}
    assert payload["n"] == len(payload["sentences"]) == sum(payload["counts"].values())
    assert {s["register"] for s in payload["sentences"]} <= set(B.REGISTERS)
    assert all(set(s) == {"text", "register"} for s in payload["sentences"])
    # the stats file round-trips through --extra-file
    loaded = B.load_extra_files([str(stats)])
    assert len(loaded) == payload["n"]

    # exact round-trip on a hand-made file (no clean_sentence ambiguity)
    js = tmp_path / "stats_min.json"
    js.write_text(json.dumps({
        "n": 2, "counts": {"native_tamil": 2},
        "sentences": [
            {"text": "நாளை காலையில் வருகிறேன்.", "register": "native_tamil"},
            {"text": "இன்று மழை பெய்யும்.", "register": "native_tamil"},
        ],
    }, ensure_ascii=False), encoding="utf-8")
    assert B.load_extra_files([str(js)]) == [
        "நாளை காலையில் வருகிறேன்.", "இன்று மழை பெய்யும்.",
    ]
```

(`json` and `Path` are already imported at module level in this test file —
used by `test_corpus_extra_file_accepts_txt_and_json`.)

**Verify**

```bash
cd /home/tripathiji/projects/all_work
movio/.venv/bin/python -m pytest tests/test_corpus_builder.py -q -p no:cacheprovider
```

Expected: previous count + 1 passed (21 → 22 on the baseline tree).

### Step 4 — `kaggle_demo.py` allow-untrained smoke test

Append to the end of `tests/test_codeswitch_tts.py`:

```python
def test_kaggle_demo_allow_untrained_smoke(tmp_path):
    """Wiring smoke for the newcomer entry point: runs kaggle_demo.py in a
    temp cwd with the untrained fallback explicitly allowed (no GPU, no
    network, random-weight audio). Success = exit 0 and WAVs in ./output_audio/."""
    import os
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parent.parent
    env = dict(os.environ, CODESWITCH_ALLOW_UNTRAINED="1",
               PYTHONPATH=str(repo_root))
    proc = subprocess.run(
        [sys.executable, str(repo_root / "kaggle_demo.py")],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, (
        f"kaggle_demo failed (rc={proc.returncode})\n"
        f"--- stdout tail ---\n{proc.stdout[-2000:]}\n"
        f"--- stderr tail ---\n{proc.stderr[-2000:]}"
    )
    wavs = sorted((tmp_path / "output_audio").glob("*.wav"))
    assert wavs, "kaggle_demo wrote no WAV files under ./output_audio/"
```

Notes for the executor:

- `kaggle_demo.py` writes `./output_audio/` **relative to the current
  working directory** (print at `:218`), hence the `cwd=tmp_path`.
- `sys.executable` is the venv python under pytest; the script directory is
  the repo root, so `import codeswitch_tts` resolves without installation
  (`PYTHONPATH` is set anyway for robustness).
- Expected runtime: tens of seconds on CPU (it synthesizes a handful of
  utterances through the standalone/fake pipeline).

**Verify**

```bash
cd /home/tripathiji/projects/all_work
movio/.venv/bin/python -m pytest tests/test_codeswitch_tts.py -q -p no:cacheprovider -k kaggle_demo
```

Expected: `1 passed`. If it times out (300 s) or fails on a network access,
STOP and report — the smoke must not require the network.

### Step 5 — Update the hand-maintained test count

New tests: 7 (parametrized) + 1 (corpus CLI) + 1 (demo smoke) = **previous
count + 9**. (On a bare 98-baseline tree that is 107; by the time 009 runs in
the locked order, earlier plans — including 008's +2 — have already raised the
count. Never predict the total: record the count before this plan, expect +9,
and write what the pytest summary prints.) Run the full suite, read the
summary number, and update `KAGGLE_RUN.md:427`.

**Verify**

```bash
cd /home/tripathiji/projects/all_work
movio/.venv/bin/python -m pytest tests/ -q -p no:cacheprovider
grep -n "Local test suite" KAGGLE_RUN.md
```

Expected: your recorded pre-009 count + 9 (that is `107 passed` only on a bare
98-baseline tree where no other plan has landed — the pytest summary is the
source of truth) and `KAGGLE_RUN.md:427` matching it.

## Test plan

| Test | File | Count |
|---|---|---|
| `test_normalizer_exact_strings[...]` | `tests/test_codeswitch_tts.py` | +7 |
| `test_json_out_writes_stats_and_round_trips_through_extra_file` | `tests/test_corpus_builder.py` | +1 |
| `test_kaggle_demo_allow_untrained_smoke` | `tests/test_codeswitch_tts.py` | +1 |
| `test_corpus_extra_file_accepts_txt_and_json` (existing) | `tests/test_corpus_builder.py` | must stay green |
| `test_corpus_is_register_balanced_and_disjoint` (existing) | `tests/test_corpus_builder.py` | must stay green |

Whole suite: previous + 9.

## Done criteria

- [ ] All 7 normalizer cases pass; outputs match the table byte-for-byte.
- [ ] `build_codeswitch_corpus.py --json-out` writes `{n, counts, sentences}`;
      the CLI manual check prints `ok`.
- [ ] `load_extra_files` accepts the stats dict AND the old txt / JSON-list
      shapes (existing tests green).
- [ ] `kaggle_demo` smoke passes offline.
- [ ] `KAGGLE_RUN.md:427` updated to the real suite count.
- [ ] `plans/README.md` status row updated (unless the reviewer maintains it).

## STOP conditions

- Any normalizer expected string mismatches → the normalizer changed since
  `d69b399`; STOP and report the actual outputs (do not silently adopt them).
- The corpus CLI emits a different register per sentence than
  `classify_register` (equality assertion in Step 3 fails) → STOP; the
  builder and the classifier disagree and that needs a decision.
- The demo smoke needs network or exceeds 300 s → STOP.
- `test_corpus_is_register_balanced_and_disjoint` (the
  `load_corpus_json` round-trip at `:129-130`) breaks → STOP; this plan is
  not allowed to change that contract.
- Quoted "Current state" snippets cannot be located by content → STOP.

## Maintenance notes

- The two JSON contracts are deliberately different now:
  - `--corpus-json` (`codeswitch_benchmark_data.load_corpus_json:321-337`)
    stays txt-or-JSON-list; it is the training input.
  - `--json-out` writes the stats dict, consumed by humans and by
    `--extra-file` (`load_extra_files`).
  Do not "unify" them without updating `KAGGLE_RUN.md:107`,
  `PAPER_CLAIMS_AND_EVIDENCE.md:59` and notebook cell 16 together.
- The normalizer pins are exact-string by design. When a normalizer change
  is intentional, the expectation table is updated in the same commit and
  the commit message should say which paper metric (CER/Table I) is
  affected.
- If `kaggle_demo.py`'s output directory ever changes, update the smoke
  assertion and the print at `kaggle_demo.py:218` in the same change.
