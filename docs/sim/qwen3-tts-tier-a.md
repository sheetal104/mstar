# Qwen3-TTS Tier A — M\*-Sim validation (summary)

Full tables and pipeline steps: https://github.com/sheetal104/mstar-sim-tier-a-qwen3tts/blob/feat/qwen3-tts-tier-a-integration-lab/docs/qwen3-tts-tier-a-results.md

**Branch:** `keisuke/mstar-sim` (simulator). Small CLI fix: `fix/simulate-cli-json-asdict` (`simulate --json` uses `asdict(spec)`; example disagg YAML).

## Gates (colocated v2, B200)

| Gate | Result |
|------|--------|
| V2 step costs | PASS |
| V1 semantics | FAIL (codec aggregation — known) |
| V3 E2E | FAIL (~23–26% sim fast vs measured) |
| V4 disagg vs colocated | Concordant (measured + sim rank disagg ahead) |

## Measured req/s (profile recovery, same workload shape)

| Layout | req/s |
|--------|------:|
| Colocated | 4.82 |
| Disagg | 5.13 |

Sim disagg Codec GPU util is **not** trustworthy until missing stepdb rows are fixed; use V2 + V4 for placement exploration, not absolute ms.
