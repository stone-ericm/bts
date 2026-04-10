# BTS Cloud Migration -- Cutover Runbook

**Spec:** `docs/superpowers/specs/2026-04-09-bts-cloud-migration-design.md`

## Phases

| Phase | Script | Duration |
|---|---|---|
| 2 | `phase2-shadow-diff.py` | 7+ days of matching |
| 3 | `phase3-cutover.sh` | ~1 evening |
| 3-rollback | `phase3-rollback.sh` | ~5 minutes |
| 4 | `phase4-decommission.sh` | ~1 day |

## Phase 2: Shadow Validation

Fly runs in shadow mode (shadow_mode=true in orchestrator.toml).
Daily comparison:

    python3 scripts/cutover/phase2-shadow-diff.py --date $(date -d yesterday +%Y-%m-%d)

Exit gate: 7 consecutive strict-match days + lineup timing params finalized.

## Phase 3: Cutover

    ./scripts/cutover/phase3-cutover.sh

Interactive prompts at each step. If anything wobbles in 48 hours:

    ./scripts/cutover/phase3-rollback.sh

## Phase 4: Decommission

After 48+ hours of stable Fly operation:

    ./scripts/cutover/phase4-decommission.sh
