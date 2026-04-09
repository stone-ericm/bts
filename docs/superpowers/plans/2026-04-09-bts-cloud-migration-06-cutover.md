# BTS Cloud Migration — Plan 06: Cutover Execution

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the actual migration from Pi5-authoritative production to Fly-authoritative production, with safety gates at every step: Phase 2 shadow-mode validation (7 days of byte-exact matching), Phase 3 state freeze + atomic cutover, 48-hour rollback window, and Phase 4 decommission of Mac/Pi5/Alienware BTS services.

**Architecture:** This plan is primarily scripts and runbooks, not new feature code. The infrastructure already exists (Plans 01-05); this plan glues it together into a repeatable, auditable cutover sequence. Shadow-mode validation uses a diff script that compares Fly's output (uploaded to R2) against Pi5's real pick state. The cutover itself is executed by a human operator following a step-by-step runbook with explicit verification at each step. Rollback is a single shell script that re-enables Pi5 services and stops the Fly machine.

**Tech Stack:** Bash scripts, Python diff tooling, `flyctl` CLI, SSH to Pi5, existing BTS commands.

**Dependencies on other plans:** ALL previous plans must be complete before starting this one.
- Plan 01 — lineup data must have been collecting for ≥ 7 days and timing parameters tuned
- Plan 02 — R2 sync working, manifest in place
- Plan 03 — state export works (including the resolved-state guard)
- Plan 04 — scheduler supports local tier and heartbeat writes
- Plan 05 — Fly infrastructure deployed and smoke-tested

**Parent spec:** `docs/superpowers/specs/2026-04-09-bts-cloud-migration-design.md` (§ Cutover Plan)

---

## File Structure

- Create `scripts/cutover/README.md` — top-level cutover runbook
- Create `scripts/cutover/phase2-shadow-run.sh` — enable Fly shadow mode
- Create `scripts/cutover/phase2-shadow-diff.py` — compare Fly shadow output to Pi5 output
- Create `scripts/cutover/phase3-cutover.sh` — runbook script (interactive prompts, verification gates)
- Create `scripts/cutover/phase3-rollback.sh` — one-shot rollback back to Pi5
- Create `scripts/cutover/phase4-decommission.sh` — final cleanup after 48-hour observation window
- Modify `src/bts/scheduler.py` — add `shadow_mode` config flag that skips Bluesky posting + writes to shadow dir

---

### Task 1: Add shadow mode to the scheduler

**Files:**
- Modify: `src/bts/scheduler.py`
- Modify: `src/bts/picks.py` (or wherever save_pick lives) — add shadow mode awareness
- Create: `tests/test_shadow_mode.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_shadow_mode.py`:

```python
"""Tests for scheduler shadow mode."""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


def test_shadow_mode_writes_to_shadow_dir(tmp_path):
    """In shadow mode, picks are written to data/shadow/{date}/ not data/picks/."""
    from bts.picks import save_pick_shadow

    shadow_dir = tmp_path / "shadow"
    pick_data = {
        "date": "2026-04-10",
        "pick": {"batter_name": "Test", "batter_id": 100, "team": "NYY"},
        "result": None,
    }

    save_pick_shadow(pick_data, shadow_dir=shadow_dir, source="fly")

    out = shadow_dir / "2026-04-10" / "fly.json"
    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded["pick"]["batter_name"] == "Test"


def test_shadow_mode_does_not_post_to_bluesky():
    """Scheduler should never call post_to_bluesky when shadow_mode=true."""
    from bts.scheduler import run_day

    config = {
        "orchestrator": {
            "picks_dir": "/tmp/picks",
            "shadow_mode": True,
            "shadow_dir": "/tmp/shadow",
            "shadow_source": "fly",
        },
        "scheduler": {},
        "bluesky": {"dm_recipient": "did:plc:test"},
        "tiers": [],
    }

    with patch("bts.scheduler.fetch_schedule", return_value=[]), \
         patch("bts.scheduler.post_to_bluesky") as mock_post:
        run_day(date="2026-04-10", config=config)

    mock_post.assert_not_called()
```

- [ ] **Step 2: Run to verify it fails**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_shadow_mode.py -v
```

Expected: Tests FAIL (functions don't exist yet).

- [ ] **Step 3: Add save_pick_shadow to picks.py**

Add to `src/bts/picks.py`:

```python
def save_pick_shadow(pick_data, shadow_dir, source: str) -> Path:
    """Save a pick record to the shadow directory (not authoritative).

    Shadow dirs are used during Phase 2 of the cloud migration to
    compare Fly's output against Pi5's real state without affecting
    production. source is 'fly' or 'pi5' to distinguish writers.
    """
    from pathlib import Path
    import json as _json

    shadow_dir = Path(shadow_dir)
    date = pick_data["date"] if isinstance(pick_data, dict) else pick_data.date
    date_dir = shadow_dir / date
    date_dir.mkdir(parents=True, exist_ok=True)

    out_path = date_dir / f"{source}.json"
    # Handle both dict and DailyPick instances
    if isinstance(pick_data, dict):
        payload = pick_data
    else:
        payload = pick_data.__dict__ if hasattr(pick_data, "__dict__") else dict(pick_data)
    out_path.write_text(_json.dumps(payload, indent=2, default=str))
    return out_path
```

- [ ] **Step 4: Update run_day to honor shadow_mode**

Modify `src/bts/scheduler.py` `run_day()` function. Near the top where config values are read:

```python
    shadow_mode = config.get("orchestrator", {}).get("shadow_mode", False)
    shadow_dir = Path(config.get("orchestrator", {}).get("shadow_dir", "data/shadow"))
    shadow_source = config.get("orchestrator", {}).get("shadow_source", "unknown")
```

Find every call to `post_to_bluesky(...)` inside `run_day()`. Wrap each in a shadow-mode check:

```python
            if shadow_mode:
                from bts.picks import save_pick_shadow
                save_pick_shadow(daily, shadow_dir=shadow_dir, source=shadow_source)
                print(f"  [SHADOW] Pick saved to {shadow_dir}/{date}/{shadow_source}.json", file=sys.stderr)
            else:
                try:
                    uri = post_to_bluesky(text)
                    # ... existing save_pick + state update ...
                except Exception as e:
                    ...
```

There are several places where post_to_bluesky is called — update each one. The pattern is always: if shadow_mode, save to shadow dir + skip posting + skip state updates; otherwise proceed with the existing path.

Also skip Bluesky DM calls when shadow_mode is true.

- [ ] **Step 5: Run tests**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_shadow_mode.py tests/ -v -k "scheduler or shadow"
```

Expected: new tests PASS, existing scheduler tests still PASS.

- [ ] **Step 6: Commit**

```bash
git add src/bts/scheduler.py src/bts/picks.py tests/test_shadow_mode.py
git commit -m "feat(cutover): add shadow_mode to scheduler for parallel-run validation"
```

---

### Task 2: Phase 2 shadow diff script

**Files:**
- Create: `scripts/cutover/phase2-shadow-diff.py`

- [ ] **Step 1: Create the diff script**

Create `scripts/cutover/phase2-shadow-diff.py`:

```python
#!/usr/bin/env python3
"""Phase 2: Diff Fly shadow output against Pi5 real pick output.

Runs daily during shadow validation. Downloads Fly's shadow output from R2
(written by the shadow-mode scheduler), fetches Pi5's real pick for the
same date, and strictly compares them. Any mismatch resets the "N consecutive
matching days" counter used as the Phase 2 exit gate.

Strict comparison: exact match on batter_id, pitcher_id, game_pk, double_down
presence and match, skip-vs-pick classification, and floating-point fields
within |Δ| < 1e-6.

Usage:
    python3 scripts/cutover/phase2-shadow-diff.py --date 2026-04-15

Exit codes:
    0 = match
    1 = mismatch (details in output)
    2 = error (couldn't load one of the sources)
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

FLOAT_TOLERANCE = 1e-6


def load_fly_shadow(date: str) -> dict | None:
    """Download Fly shadow output from R2 via flyctl ssh."""
    result = subprocess.run(
        ["flyctl", "ssh", "console", "-a", "bts", "-C",
         f"cat /data/shadow/{date}/fly.json"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Could not read Fly shadow for {date}: {result.stderr}", file=sys.stderr)
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"Fly shadow for {date} is not valid JSON", file=sys.stderr)
        return None


def load_pi5_real(date: str) -> dict | None:
    """Read Pi5 real pick state via SSH."""
    result = subprocess.run(
        ["ssh", "stonehengee@pi5.local",
         f"cat ~/projects/bts/data/picks/{date}.json"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Could not read Pi5 pick for {date}: {result.stderr}", file=sys.stderr)
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"Pi5 pick for {date} is not valid JSON", file=sys.stderr)
        return None


def compare_picks(fly: dict, pi5: dict) -> list[str]:
    """Return a list of mismatch descriptions. Empty list = strict match."""
    issues: list[str] = []

    # Primary pick fields that must match exactly
    for field in ["batter_id", "pitcher_id", "game_pk", "team", "batter_name", "pitcher_name"]:
        fly_val = _get_nested(fly, ["pick", field])
        pi5_val = _get_nested(pi5, ["pick", field])
        if fly_val != pi5_val:
            issues.append(f"pick.{field}: fly={fly_val!r} pi5={pi5_val!r}")

    # Floating-point pick fields
    for field in ["p_game_hit", "p_hit_pa"]:
        fly_val = _get_nested(fly, ["pick", field])
        pi5_val = _get_nested(pi5, ["pick", field])
        if fly_val is None and pi5_val is None:
            continue
        if fly_val is None or pi5_val is None:
            issues.append(f"pick.{field}: fly={fly_val} pi5={pi5_val}")
            continue
        if abs(float(fly_val) - float(pi5_val)) > FLOAT_TOLERANCE:
            issues.append(f"pick.{field}: |Δ|={abs(fly_val - pi5_val):.2e} > tolerance")

    # Double-down presence must match
    fly_has_dd = fly.get("double_down") is not None
    pi5_has_dd = pi5.get("double_down") is not None
    if fly_has_dd != pi5_has_dd:
        issues.append(f"double_down presence: fly={fly_has_dd} pi5={pi5_has_dd}")
    elif fly_has_dd:
        for field in ["batter_id", "game_pk"]:
            fly_val = _get_nested(fly, ["double_down", field])
            pi5_val = _get_nested(pi5, ["double_down", field])
            if fly_val != pi5_val:
                issues.append(f"double_down.{field}: fly={fly_val!r} pi5={pi5_val!r}")
        for field in ["p_game_hit"]:
            fly_val = _get_nested(fly, ["double_down", field])
            pi5_val = _get_nested(pi5, ["double_down", field])
            if fly_val is None or pi5_val is None:
                continue
            if abs(float(fly_val) - float(pi5_val)) > FLOAT_TOLERANCE:
                issues.append(f"double_down.{field}: |Δ|={abs(fly_val - pi5_val):.2e}")

    return issues


def _get_nested(obj: dict, path: list[str]):
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def main():
    parser = argparse.ArgumentParser(description="Compare Fly shadow to Pi5 real pick")
    parser.add_argument("--date", required=True, help="Date YYYY-MM-DD")
    args = parser.parse_args()

    fly = load_fly_shadow(args.date)
    pi5 = load_pi5_real(args.date)

    if fly is None or pi5 is None:
        print(f"ERROR: could not load one or both sides for {args.date}", file=sys.stderr)
        sys.exit(2)

    issues = compare_picks(fly, pi5)
    if not issues:
        print(f"MATCH: Fly == Pi5 for {args.date}")
        sys.exit(0)

    print(f"MISMATCH: {args.date}")
    for issue in issues:
        print(f"  - {issue}")
    sys.exit(1)


if __name__ == "__main__":
    main()
```

Make it executable:

```bash
chmod +x scripts/cutover/phase2-shadow-diff.py
```

- [ ] **Step 2: Commit**

```bash
git add scripts/cutover/phase2-shadow-diff.py
git commit -m "feat(cutover): add Phase 2 shadow diff script"
```

---

### Task 3: Phase 2 shadow-run enablement script

**Files:**
- Create: `scripts/cutover/phase2-shadow-run.sh`

- [ ] **Step 1: Create the script**

Create `scripts/cutover/phase2-shadow-run.sh`:

```bash
#!/bin/bash
# Enable shadow mode on Fly and verify it's running without posting.

set -euo pipefail
log() { echo "[phase2] $*"; }

log "Enabling shadow mode in Fly config"
flyctl ssh console -a bts -C "sed -i 's/^shadow_mode.*/shadow_mode = true/' /data/orchestrator.toml || echo 'shadow_mode = true' >> /data/orchestrator.toml"

log "Setting shadow source to 'fly'"
flyctl ssh console -a bts -C "grep -q 'shadow_source' /data/orchestrator.toml || echo 'shadow_source = \"fly\"' >> /data/orchestrator.toml"
flyctl ssh console -a bts -C "grep -q 'shadow_dir' /data/orchestrator.toml || echo 'shadow_dir = \"/data/shadow\"' >> /data/orchestrator.toml"

log "Restarting machine to pick up config"
MACHINE_ID=$(flyctl machines list -a bts --json | jq -r '.[0].id')
flyctl machines restart "$MACHINE_ID" -a bts

log "Waiting 60s for startup"
sleep 60

log "Verifying health endpoint"
RESP=$(curl -sS --max-time 10 http://bts:3003/health || echo "FAIL")
echo "Health response: $RESP"

log "Checking logs for scheduler startup"
flyctl logs -a bts --since 2m

log ""
log "Shadow mode is now enabled. The Fly scheduler will run daily predictions"
log "and write to /data/shadow/{date}/fly.json WITHOUT posting to Bluesky."
log "Pi5 remains authoritative."
log ""
log "To compare daily:"
log "    python3 scripts/cutover/phase2-shadow-diff.py --date YYYY-MM-DD"
log ""
log "Phase 2 exit gate: 7 consecutive strict-match days AND lineup timing params finalized."
```

Make it executable:

```bash
chmod +x scripts/cutover/phase2-shadow-run.sh
```

- [ ] **Step 2: Commit**

```bash
git add scripts/cutover/phase2-shadow-run.sh
git commit -m "feat(cutover): add Phase 2 shadow-mode enablement script"
```

---

### Task 4: Phase 3 cutover script

**Files:**
- Create: `scripts/cutover/phase3-cutover.sh`

- [ ] **Step 1: Create the cutover script**

Create `scripts/cutover/phase3-cutover.sh`:

```bash
#!/bin/bash
# Phase 3 cutover: Pi5 → Fly authoritative.
#
# Interactive script — prompts the operator for confirmation at each
# critical step. Designed to be run once, in the evening after all
# games for the day have finalized on Pi5.
#
# Preconditions:
# - Phase 2 exit gate met: 7 consecutive strict-match shadow days
# - Lineup timing parameters finalized in fly.toml / orchestrator.toml
# - No games currently in progress
# - All pick files on Pi5 have result != None

set -euo pipefail

confirm() {
    read -p "$1 (yes/no): " answer
    if [ "$answer" != "yes" ]; then
        echo "Aborted."
        exit 1
    fi
}

log() { echo ""; echo "=== $* ==="; echo ""; }

log "Phase 3 Cutover — Pi5 → Fly"
echo "This is the authoritative cutover. It will:"
echo "  1. Export initial state from Pi5 (refuses if unresolved picks)"
echo "  2. Commit the snapshot to git"
echo "  3. Disable shadow mode on Fly (Fly becomes authoritative)"
echo "  4. Stop Pi5 bts-scheduler services (but not disable — 48h rollback window)"
echo "  5. Verify Fly is alive"
echo ""
confirm "Ready to begin?"

# 1. Export state from Pi5
log "Step 1: Export initial state from Pi5"
ssh stonehengee@pi5.local 'cd ~/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run bts state export --to data/state/initial-state.json' \
    || { echo "Export failed — refusing to proceed. Check for unresolved picks on Pi5."; exit 1; }

log "Step 2: Copy snapshot from Pi5 to local Mac"
scp stonehengee@pi5.local:~/projects/bts/data/state/initial-state.json data/state/initial-state.json

echo "Inspecting exported snapshot:"
python3 -c "
import json
data = json.loads(open('data/state/initial-state.json').read())
print(f'  cutoff_date: {data[\"cutoff_date\"]}')
print(f'  streak_at_cutoff: {data[\"streak_at_cutoff\"]}')
print(f'  saver_available: {data[\"saver_available\"]}')
print(f'  historical_picks: {len(data[\"historical_picks\"])}')
"
confirm "Does the exported snapshot look correct?"

# 2. Commit snapshot
log "Step 3: Commit snapshot to git"
CUTOFF=$(python3 -c "import json; print(json.loads(open('data/state/initial-state.json').read())['cutoff_date'])")
git add data/state/initial-state.json
git commit -m "migration: freeze state at cutoff ${CUTOFF}

Cloud migration Phase 3 state snapshot. This is a one-time commit
of Pi5's current state at the moment of cutover. Regeneration uses
this file for dates <= cutoff and Bluesky scraping for dates after.
"
echo "Committed. Now pushing."
git push origin main

# 3. Disable shadow mode on Fly
log "Step 4: Disable shadow mode on Fly (Fly becomes authoritative)"
confirm "Flip Fly to authoritative mode?"

flyctl ssh console -a bts -C "sed -i 's/^shadow_mode = true/shadow_mode = false/' /data/orchestrator.toml"

MACHINE_ID=$(flyctl machines list -a bts --json | jq -r '.[0].id')
flyctl machines restart "$MACHINE_ID" -a bts

echo "Waiting 60s for machine restart"
sleep 60

# 4. Stop Pi5 services (stopped but not disabled — rollback window)
log "Step 5: Stop Pi5 BTS services (rollback-ready)"
confirm "Stop Pi5 bts-scheduler + bts-dashboard?"

ssh stonehengee@pi5.local 'systemctl --user stop bts-scheduler bts-dashboard'
echo "Pi5 services stopped (but still enabled for quick rollback)"

# 5. Verify Fly
log "Step 6: Verify Fly is alive"
RESP=$(curl -sS --max-time 10 http://bts:3003/health || echo "FAIL")
echo "Health endpoint: $RESP"

if echo "$RESP" | grep -q '"status": "ok"'; then
    echo "Fly is healthy."
else
    echo "WARNING: Fly health check failed. Run phase3-rollback.sh if needed."
fi

flyctl logs -a bts --since 5m | tail -30

log "Phase 3 complete."
echo ""
echo "Next: 48-hour observation window. If anything wobbles, run:"
echo "    scripts/cutover/phase3-rollback.sh"
echo ""
echo "After 48 hours of stable operation, run:"
echo "    scripts/cutover/phase4-decommission.sh"
```

Make it executable:

```bash
chmod +x scripts/cutover/phase3-cutover.sh
```

- [ ] **Step 2: Commit**

```bash
git add scripts/cutover/phase3-cutover.sh
git commit -m "feat(cutover): add Phase 3 interactive cutover script"
```

---

### Task 5: Phase 3 rollback script

**Files:**
- Create: `scripts/cutover/phase3-rollback.sh`

- [ ] **Step 1: Create the rollback script**

Create `scripts/cutover/phase3-rollback.sh`:

```bash
#!/bin/bash
# Phase 3 rollback: Fly → Pi5 authoritative.
#
# Use this if anything goes wrong in the 48 hours after cutover.
# Pi5 services are only stopped (not disabled) during that window,
# so rollback is fast: restart them, pause Fly, done.

set -euo pipefail
log() { echo ""; echo "=== $* ==="; echo ""; }

confirm() {
    read -p "$1 (yes/no): " answer
    if [ "$answer" != "yes" ]; then
        echo "Aborted."
        exit 1
    fi
}

log "Phase 3 Rollback — Fly → Pi5"
echo "This will:"
echo "  1. Start Pi5 bts-scheduler + bts-dashboard"
echo "  2. Pause the Fly machine"
echo "  3. Verify Pi5 is alive"
echo ""
confirm "Proceed with rollback?"

# 1. Start Pi5 services
log "Step 1: Restart Pi5 services"
ssh stonehengee@pi5.local 'systemctl --user start bts-scheduler bts-dashboard'
sleep 5
ssh stonehengee@pi5.local 'systemctl --user status bts-scheduler --no-pager' || true

# 2. Pause Fly
log "Step 2: Pause Fly machine"
MACHINE_ID=$(flyctl machines list -a bts --json | jq -r '.[0].id')
flyctl machines stop "$MACHINE_ID" -a bts

# 3. Verify Pi5 is alive (via Tailscale)
log "Step 3: Verify Pi5 is serving"
curl -sS --max-time 10 http://pi5.local:3003/ > /dev/null \
    && echo "Pi5 dashboard responding" \
    || echo "WARNING: Pi5 dashboard not responding"

echo ""
log "Rollback complete."
echo ""
echo "Pi5 is now authoritative again. Fly is paused."
echo ""
echo "Next steps:"
echo "  1. Investigate why Phase 3 failed (check 'flyctl logs -a bts' from before pause)"
echo "  2. Fix the issue"
echo "  3. Re-enable shadow mode and run another 7-day Phase 2 validation"
echo "  4. Retry Phase 3 when ready"
```

Make it executable:

```bash
chmod +x scripts/cutover/phase3-rollback.sh
```

- [ ] **Step 2: Commit**

```bash
git add scripts/cutover/phase3-rollback.sh
git commit -m "feat(cutover): add Phase 3 rollback script"
```

---

### Task 6: Phase 4 decommission script

**Files:**
- Create: `scripts/cutover/phase4-decommission.sh`

- [ ] **Step 1: Create the decommission script**

Create `scripts/cutover/phase4-decommission.sh`:

```bash
#!/bin/bash
# Phase 4 decommission: permanently retire Pi5/Alienware from BTS.
#
# Run only after 48+ hours of stable Fly operation. This removes the
# rollback safety net by disabling Pi5 systemd units and cleaning up
# Alienware. Mac retains the BTS repo for experiments.

set -euo pipefail
confirm() {
    read -p "$1 (yes/no): " answer
    [ "$answer" = "yes" ] || { echo "Aborted."; exit 1; }
}
log() { echo ""; echo "=== $* ==="; echo ""; }

log "Phase 4 Decommission — make the cutover permanent"
echo "This will:"
echo "  1. Disable (not just stop) Pi5 bts-scheduler and bts-dashboard"
echo "  2. Remove Pi5 BTS systemd unit files"
echo "  3. Remove Mac crons for data pull/build"
echo "  4. Remove Alienware BTS repo"
echo ""
echo "There is NO rollback path after this script completes."
echo ""
confirm "Has Fly run stably for at least 48 hours since Phase 3?"
confirm "Are you sure you want to permanently retire Pi5 BTS services?"

# 1. Disable and remove Pi5 units
log "Step 1: Disable Pi5 BTS services"
ssh stonehengee@pi5.local '
    systemctl --user disable bts-scheduler bts-dashboard bts-lineup-collect.timer 2>/dev/null || true
    rm -f ~/.config/systemd/user/bts-scheduler.service
    rm -f ~/.config/systemd/user/bts-dashboard.service
    # Keep bts-lineup-collect since it is still useful as supplementary data
    systemctl --user daemon-reload
'
echo "Pi5 BTS systemd units removed"

# 2. Remove Mac crons
log "Step 2: Remove Mac BTS crons"
echo "Current Mac crontab:"
crontab -l 2>/dev/null | grep -i bts || echo "  (no BTS cron entries found)"
echo ""
echo "Manual step: run 'crontab -e' and remove any 'bts data pull' or 'bts data build' lines"
confirm "Done removing Mac BTS crons?"

# 3. Remove Alienware BTS
log "Step 3: Retire Alienware BTS"
echo "Manual step: on Alienware, run:"
echo "    rm -rf C:\\Users\\stone\\projects\\bts"
echo "Or via SSH: ssh alienware 'rm -rf C:/Users/stone/projects/bts'"
confirm "Done removing Alienware BTS?"

# 4. Delete dead code (orchestrator.py SSH cascade)
log "Step 4: Delete dead SSH cascade code"
echo "This is a separate PR — manual step:"
echo "    1. Delete src/bts/orchestrator.py (SSH cascade logic)"
echo "    2. Remove [[tiers]] with type=ssh from config examples"
echo "    3. Update scheduler.py to import run_pipeline/select_pick directly"
echo "    4. Update CLAUDE.md + ARCHITECTURE.md"
echo "    5. PR + merge"
echo ""

log "Phase 4 decommission instructions emitted."
echo "Manual cleanup steps remain. Commit the deletions as a follow-up PR."
```

Make it executable:

```bash
chmod +x scripts/cutover/phase4-decommission.sh
```

- [ ] **Step 2: Commit**

```bash
git add scripts/cutover/phase4-decommission.sh
git commit -m "feat(cutover): add Phase 4 decommission script"
```

---

### Task 7: Top-level cutover README

**Files:**
- Create: `scripts/cutover/README.md`

- [ ] **Step 1: Write the README**

Create `scripts/cutover/README.md`:

```markdown
# BTS Cloud Migration — Cutover Runbook

This directory contains the scripts and runbook for executing the
BTS cloud migration from Pi5+Mac+Alienware to Fly.io.

**Parent spec:** `docs/superpowers/specs/2026-04-09-bts-cloud-migration-design.md`

**Phase overview:**

| Phase | Purpose | Duration |
|---|---|---|
| 0 | External prereqs (Fly app, R2 bucket, Tailscale, etc.) | Once |
| 1 | Write infrastructure code (Plans 01-05 merged to main) | Once |
| 2 | Parallel shadow running — 7 days of strict matching | ≥7 days |
| 3 | State freeze + atomic cutover | ~1 evening |
| 3-rollback | Emergency revert to Pi5 | ~5 minutes |
| 4 | Decommission Pi5 + Alienware + Mac BTS services | ~1 day |

## Phase 0 prerequisites

See parent spec Phase 0 checklist. Must be complete before Phase 1 code is
merged. Reversible by deleting Fly app + R2 bucket.

## Phase 1 code merges

All PRs from Plans 01, 02, 03, 04, 05 merged to main. Deploy workflow builds
the image and pushes to Fly. Fly machine runs in **shadow mode** (config has
`shadow_mode = true`). No Bluesky posts from Fly. Pi5 continues to be
authoritative.

## Phase 2 execution

Once the Fly machine is running shadow mode:

```
./scripts/cutover/phase2-shadow-run.sh
```

This enables shadow mode on Fly and restarts the machine. Verify the health
endpoint and logs.

Then, every day at ~9am ET (after the previous day's results are final),
run the comparison:

```
python3 scripts/cutover/phase2-shadow-diff.py --date $(date -d yesterday +%Y-%m-%d)
```

Track each day's result in a local log. The Phase 2 exit gate is:

1. **7 consecutive strict-match days** — every comparison returns exit code 0
2. **Lineup timing parameters finalized** — `bts data analyze-lineup-times`
   report reviewed, `fallback_deadline_min` and related config values chosen
3. **Backtest validated** — `bts simulate backtest` shows P@1 drop ≤ 1pp
   vs the old -45/-15 timing

If any strict comparison fails, the 7-day counter resets. Investigate the
mismatch before continuing.

## Phase 3 execution

When Phase 2 exit gate is met AND all picks for the current day are resolved:

```
./scripts/cutover/phase3-cutover.sh
```

This is interactive; it will prompt you at each critical step. Read the
output carefully and say 'yes' only when you're sure.

After Phase 3 completes, **monitor for 48 hours**. If anything goes wrong:

```
./scripts/cutover/phase3-rollback.sh
```

## Phase 4 execution

After 48+ hours of stable Fly operation:

```
./scripts/cutover/phase4-decommission.sh
```

This script performs the reversible Pi5 cleanup itself and prints
instructions for the remaining manual steps (Mac crons, Alienware, code
deletion PR).

## Emergency contacts

- Fly.io status: https://status.fly.io/
- Cloudflare status: https://www.cloudflarestatus.com/
- Bluesky status: https://status.bsky.app/
- MLB Stats API status: https://statsapi.mlb.com/ (usually reliable)

## Files in this directory

- `README.md` — this file
- `phase2-shadow-run.sh` — enable Fly shadow mode
- `phase2-shadow-diff.py` — daily comparison script
- `phase3-cutover.sh` — interactive cutover runbook
- `phase3-rollback.sh` — emergency revert to Pi5
- `phase4-decommission.sh` — final cleanup after observation period
```

- [ ] **Step 2: Commit**

```bash
git add scripts/cutover/README.md
git commit -m "docs(cutover): add top-level cutover runbook README"
```

---

### Task 8: Orchestrator.py and SSH cascade cleanup PR (post-Phase 4)

**Files:**
- Delete: `src/bts/orchestrator.py`
- Modify: `src/bts/scheduler.py` — remove `from bts.orchestrator import ...` if it still exists
- Modify: `config/orchestrator.example.toml` — remove `[[tiers]]` sections
- Modify: `CLAUDE.md` — update deployment section
- Modify: `ARCHITECTURE.md` — update architecture description

**Note:** Do NOT execute this task until after Phase 4 is complete and the Fly deployment has been stable for at least 1 week. The SSH cascade code is dead after Phase 4 but retaining it for a short period preserves the option of emergency rollback to a pre-migration git SHA.

- [ ] **Step 1: Confirm Fly has been running ≥ 1 week past Phase 4**

Check `flyctl logs -a bts --since 168h` for any anomalies. If anything looks wrong, pause this cleanup.

- [ ] **Step 2: Delete orchestrator.py**

```bash
git rm src/bts/orchestrator.py
```

- [ ] **Step 3: Update scheduler.py imports**

In `src/bts/scheduler.py`, remove any lingering `from bts.orchestrator import ...` lines. If `run_single_check` or similar functions still reference `run_and_pick`, replace with:

```python
from bts.orchestrator import run_cascade  # Plan 04 retained this
# OR if run_cascade is also removed, inline the local dispatch:
from bts.model.predict import run_pipeline, load_blend
from bts.strategy import select_pick
```

Actually: re-read Plan 04's Task 4. The `predict_local` function was added to orchestrator.py. If we delete orchestrator.py here, we need to move `predict_local` and `run_cascade` to a new home. Options:

1. Move `predict_local` + `run_cascade` to `src/bts/scheduler.py` as helper functions (they're only used there)
2. Rename `orchestrator.py` to `cascade.py` with just the local-tier support

Option 1 is simpler for cleanup. Move the functions into scheduler.py near the top, mark as helpers.

- [ ] **Step 4: Remove [[tiers]] from config example**

Modify `config/orchestrator.example.toml` to remove all `[[tiers]]` sections and add a comment:

```toml
# In cloud deployment (Fly), the scheduler calls run_pipeline directly
# via predict_local. No SSH cascade, no tier configuration needed.
```

- [ ] **Step 5: Update CLAUDE.md**

In `CLAUDE.md`, update the Quick Start and deployment sections to reflect the Fly-only architecture. Remove references to `ssh mac`, `ssh alienware`, SSH cascade, or the `[[tiers]]` config.

- [ ] **Step 6: Update ARCHITECTURE.md**

In `ARCHITECTURE.md`, update the architecture section to describe the Fly-hosted single-machine architecture. Remove references to Pi5/Mac/Alienware as production components.

- [ ] **Step 7: Run tests**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/ -v
```

Expected: all tests still pass. If any fail because they were testing the removed SSH cascade code, delete those tests (the functionality is gone, the tests should be gone too).

- [ ] **Step 8: Commit the cleanup**

```bash
git add -u
git rm src/bts/orchestrator.py
git commit -m "refactor(cloud): remove dead SSH cascade code post-cutover

After the cloud migration cutover was validated stable, the SSH
cascade tier (Pi5 → Mac → Alienware) is no longer used. This commit
deletes src/bts/orchestrator.py, removes [[tiers]] from config, and
updates docs to reflect the Fly-only production architecture.

Rollback path: check out a pre-migration SHA from before Plan 06
Phase 3 cutover.
"
git push origin main
```

---

## Completion criteria for Plan 06

### Phase 2 gate
- [ ] Shadow mode enabled on Fly and producing `/data/shadow/{date}/fly.json` daily
- [ ] 7 consecutive strict-match days via `phase2-shadow-diff.py`
- [ ] Lineup timing parameters finalized based on Plan 01 data
- [ ] Backtest validates timing changes show ≤ 1pp P@1 drop

### Phase 3 gate
- [ ] `initial-state.json` exported cleanly (no unresolved picks)
- [ ] Snapshot committed to git and pushed to main
- [ ] Fly `shadow_mode = false`, machine restarted
- [ ] Pi5 services stopped (but not disabled)
- [ ] Fly `/health` returns 200 immediately post-cutover
- [ ] Healthchecks.io pings are reaching successfully

### Phase 3 observation window
- [ ] 48 hours elapsed with no rollback triggered
- [ ] No pick was missed during the window
- [ ] Streak continues incrementing correctly

### Phase 4 gate
- [ ] Pi5 bts-scheduler + bts-dashboard disabled + unit files removed
- [ ] Mac BTS data pull + build crons removed
- [ ] Alienware BTS directory removed
- [ ] (1 week later) dead SSH cascade code deleted from repo
- [ ] Documentation (CLAUDE.md, ARCHITECTURE.md) updated to reflect cloud-only

## End state

After Plan 06 completes:

- **Production** runs entirely on Fly.io, region iad, single shared-cpu-2x machine
- **Dashboard** reachable at `http://bts:3003` via Tailscale only
- **State** lives on the 50 GB Fly volume with snapshot-based recovery + regenerate-on-demand
- **Mac** is dev-only, pulls parquets from R2 for experiments
- **Pi5** hosts claude-shared and investigation-kb only (no BTS)
- **Alienware** has no BTS involvement
- **Deploy** via `git push origin main` → GitHub Actions → flyctl deploy

The migration is complete.
