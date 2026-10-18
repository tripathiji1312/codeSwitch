# Research index — improve-skill runs

Durable log of deep research performed with the `improve` skill on this repo. Each findings file contains the FULL verbatim reports of all audit subagents for that run (not condensed).

| File | Date | Summary |
|---|---|---|
| [findings-2026-10-18.md](findings-2026-10-18.md) | 2026-10-18 | Full first audit of CodeSwitch-TTS at HEAD `d69b399` across all nine playbook categories: correctness (9 findings), security + dependencies (9), tests + DX (12), tech-debt + performance (12), docs (10), direction (6). Evidence base for the plans in `plans/`. |

Notes for future runs / agents:
- Line numbers in reports drift as the repo evolves — re-grep by symbol before relying on any `file:line`.
- Read each entry's "Not verified" section before treating a finding as fact.
- Cross-report contradictions and their resolutions are recorded at the end of each entry.
- Wave-1 reports were read against a moving tree (`8bee28c` → `d69b399`); wave-2 reports verified against `d69b399` with md5s.
