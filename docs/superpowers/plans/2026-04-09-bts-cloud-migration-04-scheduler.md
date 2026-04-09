# BTS Cloud Migration — Plan 04: Scheduler Refactors

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the scheduler daemon and related code so it can run on the Fly cloud VM: add heartbeat writes for health monitoring, extract hardcoded timing constants into config, add a local execution mode (no SSH cascade) for Fly, consolidate the Bluesky password helper, and add a health HTTP endpoint to the dashboard.

**Architecture:** The scheduler stays structurally the same but gains three new capabilities: (1) a heartbeat file written every 30 seconds containing current state, (2) a config-driven dispatch for "local" vs "cascade" execution mode (Pi5 uses cascade during migration transition, Fly uses local), (3) a `/health` endpoint in the dashboard that reads the heartbeat and reports 200/503. The Bluesky password reading is consolidated into a single helper used by both `posting.py` and `dm.py`, reading a new `BTS_BLUESKY_APP_PASSWORD` env var as the primary source with the old variable names as fallback.

**Tech Stack:** Python 3.12, Flask (existing dashboard), stdlib for heartbeat, existing `bts.posting`/`bts.dm` modules.

**Dependencies on other plans:** None. Pi5 can continue to run the existing cascade mode while Fly uses the new local mode. Both branches exist simultaneously.

**Parent spec:** `docs/superpowers/specs/2026-04-09-bts-cloud-migration-design.md` (§ Monitoring and alerting, § Scheduler timing tuning, § Bluesky password consolidation)

---

## File Structure

- Modify `src/bts/scheduler.py` — add heartbeat writes, config-driven mode dispatch
- Modify `src/bts/orchestrator.py` — add a "local" tier type alongside existing "ssh"
- Modify `src/bts/posting.py` — extend `get_bluesky_password()` for new env var
- Modify `src/bts/dm.py` — use shared helper instead of rolling its own
- Modify `src/bts/web.py` — add `/health` endpoint
- Modify `config/orchestrator.example.toml` — document new timing config keys
- Create `src/bts/heartbeat.py` — small module for heartbeat read/write, ~60 lines
- Create `tests/test_heartbeat.py` — ~80 lines
- Create `tests/test_local_tier.py` — ~100 lines
- Create `tests/test_bluesky_password.py` — ~80 lines
- Create `tests/test_health_endpoint.py` — ~80 lines

---

### Task 1: Heartbeat module

**Files:**
- Create: `src/bts/heartbeat.py`
- Create: `tests/test_heartbeat.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_heartbeat.py`:

```python
"""Tests for heartbeat module."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from bts.heartbeat import write_heartbeat, read_heartbeat, is_heartbeat_fresh, HeartbeatState


def test_write_and_read_heartbeat(tmp_path: Path):
    hb_path = tmp_path / ".heartbeat"
    now = datetime(2026, 4, 9, 15, 30, tzinfo=timezone.utc)

    write_heartbeat(hb_path, state="running", now_utc=now)
    hb = read_heartbeat(hb_path)

    assert hb is not None
    assert hb["state"] == "running"
    assert hb["timestamp"] == now.isoformat()


def test_read_missing_heartbeat_returns_none(tmp_path: Path):
    assert read_heartbeat(tmp_path / "nonexistent") is None


def test_is_fresh_true_when_recent(tmp_path: Path):
    hb_path = tmp_path / ".heartbeat"
    now = datetime(2026, 4, 9, 15, 30, tzinfo=timezone.utc)
    write_heartbeat(hb_path, state="running", now_utc=now)

    # Check 2 minutes later — within the 3-minute freshness window
    check_time = now + timedelta(minutes=2)
    assert is_heartbeat_fresh(hb_path, max_age_sec=180, now_utc=check_time) is True


def test_is_stale_when_old(tmp_path: Path):
    hb_path = tmp_path / ".heartbeat"
    now = datetime(2026, 4, 9, 15, 30, tzinfo=timezone.utc)
    write_heartbeat(hb_path, state="running", now_utc=now)

    check_time = now + timedelta(minutes=10)
    assert is_heartbeat_fresh(hb_path, max_age_sec=180, now_utc=check_time) is False


def test_sleeping_state_is_fresh_even_if_old(tmp_path: Path):
    """A heartbeat in state 'sleeping_until_X' is fresh as long as X is in the future."""
    hb_path = tmp_path / ".heartbeat"
    now = datetime(2026, 4, 9, 10, 0, tzinfo=timezone.utc)
    wake = now + timedelta(hours=5)

    write_heartbeat(hb_path, state="sleeping", now_utc=now, sleeping_until=wake)

    # Check 2 hours later — normally stale, but sleeping_until is still in future
    check_time = now + timedelta(hours=2)
    assert is_heartbeat_fresh(hb_path, max_age_sec=180, now_utc=check_time) is True


def test_sleeping_past_wake_time_is_stale(tmp_path: Path):
    """Once sleeping_until passes, the heartbeat is stale."""
    hb_path = tmp_path / ".heartbeat"
    now = datetime(2026, 4, 9, 10, 0, tzinfo=timezone.utc)
    wake = now + timedelta(hours=1)

    write_heartbeat(hb_path, state="sleeping", now_utc=now, sleeping_until=wake)

    # Check after wake time but no fresh heartbeat
    check_time = now + timedelta(hours=2)
    assert is_heartbeat_fresh(hb_path, max_age_sec=180, now_utc=check_time) is False


def test_atomic_write_uses_tmp_rename(tmp_path: Path):
    """Verify there's no torn state during write."""
    hb_path = tmp_path / ".heartbeat"
    now = datetime(2026, 4, 9, 15, 30, tzinfo=timezone.utc)
    write_heartbeat(hb_path, state="running", now_utc=now)

    # No leftover .tmp file
    tmp_file = hb_path.with_suffix(".tmp")
    assert not tmp_file.exists()
    # Actual file exists and has valid JSON
    assert hb_path.exists()
    data = json.loads(hb_path.read_text())
    assert data["state"] == "running"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_heartbeat.py -v
```

Expected: All tests FAIL with `ModuleNotFoundError: No module named 'bts.heartbeat'`.

- [ ] **Step 3: Write the implementation**

Create `src/bts/heartbeat.py`:

```python
"""Heartbeat file read/write for scheduler health monitoring.

The scheduler writes a heartbeat every 30 seconds with its current state.
The dashboard's /health endpoint reads the heartbeat and decides whether
the scheduler is alive, making Fly's HTTP health check work without
needing IPC between processes.

During long sleeps between lineup checks, the heartbeat's 'state' field
indicates sleeping_until_X so the staleness check knows the scheduler
is intentionally quiet, not hung.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class HeartbeatState:
    """Constants for well-known heartbeat state values."""
    RUNNING = "running"
    SLEEPING = "sleeping"
    WAITING_FOR_GAMES = "waiting_for_games"
    IDLE_END_OF_DAY = "idle_end_of_day"


def write_heartbeat(
    path: Path,
    state: str,
    now_utc: Optional[datetime] = None,
    sleeping_until: Optional[datetime] = None,
    extra: Optional[dict] = None,
) -> None:
    """Write a heartbeat JSON file atomically (via .tmp + rename).

    Args:
        path: heartbeat file path (e.g. /data/.heartbeat)
        state: one of HeartbeatState.* values
        now_utc: timestamp for the heartbeat (defaults to now)
        sleeping_until: if state is 'sleeping', when to wake up
        extra: optional additional fields to include
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    payload = {
        "state": state,
        "timestamp": now_utc.isoformat(),
    }
    if sleeping_until is not None:
        payload["sleeping_until"] = sleeping_until.isoformat()
    if extra:
        payload.update(extra)

    tmp = path.with_suffix(".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(payload))
    tmp.rename(path)


def read_heartbeat(path: Path) -> Optional[dict]:
    """Read the current heartbeat. Returns None if missing or unreadable."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def is_heartbeat_fresh(
    path: Path,
    max_age_sec: int = 180,
    now_utc: Optional[datetime] = None,
) -> bool:
    """Return True if the heartbeat is fresh enough to indicate a live scheduler.

    A heartbeat in state='sleeping' with sleeping_until in the future is
    considered fresh regardless of age. This prevents the health check from
    firing during long overnight sleeps.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    hb = read_heartbeat(path)
    if hb is None:
        return False

    # If sleeping, trust sleeping_until
    if hb.get("state") == HeartbeatState.SLEEPING:
        sleeping_until = hb.get("sleeping_until")
        if sleeping_until:
            wake = datetime.fromisoformat(sleeping_until)
            if wake > now_utc:
                return True

    # Otherwise, check age
    ts = datetime.fromisoformat(hb["timestamp"])
    age_sec = (now_utc - ts).total_seconds()
    return age_sec <= max_age_sec
```

- [ ] **Step 4: Run tests**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_heartbeat.py -v
```

Expected: All seven tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bts/heartbeat.py tests/test_heartbeat.py
git commit -m "feat(scheduler): add heartbeat module for health monitoring"
```

---

### Task 2: Wire heartbeat writes into scheduler.py

**Files:**
- Modify: `src/bts/scheduler.py`

- [ ] **Step 1: Add heartbeat writes at key scheduler loop points**

In `src/bts/scheduler.py`, at the top:

```python
from bts.heartbeat import write_heartbeat, HeartbeatState
```

In `run_day()` near the top (after `picks_dir = Path(...)` line):

```python
    heartbeat_path = Path(config.get("orchestrator", {}).get("heartbeat_path", "data/.heartbeat"))
    write_heartbeat(heartbeat_path, state=HeartbeatState.RUNNING)
```

Inside the main loop (around line 552, where `for run_info in runs:` starts), add a heartbeat write before and during each iteration. Specifically, right before `time.sleep(wait_secs)`:

```python
        if now < target:
            write_heartbeat(
                heartbeat_path,
                state=HeartbeatState.SLEEPING,
                sleeping_until=target.astimezone(UTC),
            )
            wait_secs = (target - now).total_seconds()
            print(f"  Sleeping until {target.strftime('%H:%M ET')} "
                  f"({wait_secs / 60:.0f} min)...", file=sys.stderr)
            time.sleep(wait_secs)
            write_heartbeat(heartbeat_path, state=HeartbeatState.RUNNING)
```

At the end of `run_day()`, before returning:

```python
    write_heartbeat(heartbeat_path, state=HeartbeatState.IDLE_END_OF_DAY)
```

Also add heartbeat during the result polling loop in `run_result_polling` — at the start of each poll iteration:

```python
        write_heartbeat(heartbeat_path, state=HeartbeatState.RUNNING,
                       extra={"phase": "result_polling"})
```

But `run_result_polling` doesn't have `heartbeat_path` — we need to pass it. Change the signature:

```python
def run_result_polling(
    game_pk: int,
    date: str,
    picks_dir: Path,
    heartbeat_path: Path | None = None,
    poll_interval_min: int = 15,
    cap_hour_et: int = 5,
) -> str:
```

And inside the `while True:` loop, if `heartbeat_path` is set, call `write_heartbeat(heartbeat_path, state=HeartbeatState.RUNNING, extra={"phase": "result_polling"})`.

Update the caller in `run_day()` (around line 747) to pass `heartbeat_path`:

```python
            status = run_result_polling(
                game_pk, date, picks_dir,
                heartbeat_path=heartbeat_path,
                poll_interval_min=poll_interval_min,
                cap_hour_et=cap_hour_et,
            )
```

- [ ] **Step 2: Run existing scheduler tests to ensure no regression**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/ -v -k "scheduler"
```

Expected: All existing scheduler tests still PASS.

- [ ] **Step 3: Commit**

```bash
git add src/bts/scheduler.py
git commit -m "feat(scheduler): write heartbeat file at loop checkpoints"
```

---

### Task 3: Extract hardcoded timing constants into config

**Files:**
- Modify: `src/bts/scheduler.py`
- Modify: `config/orchestrator.example.toml`

- [ ] **Step 1: Read the current hardcoded constants**

Look at `scheduler.py:626-640` and `:680` for the hardcoded `15` (fallback deadline). These are the values to extract. The `45` (lineup check offset) is already config-driven as `lineup_check_offset_min` per `scheduler.py:500`.

- [ ] **Step 2: Update `run_day` to read from config**

At the top of `run_day` where other config values are read (around line 500):

```python
    fallback_deadline_min = sched_config.get("fallback_deadline_min", 15)
    missed_pick_alert_min = sched_config.get("missed_pick_alert_min", 10)
```

Then in the block around `scheduler.py:626-640`, replace:

```python
            fallback_deadline = pick_game_et - timedelta(minutes=15)
```

with:

```python
            fallback_deadline = pick_game_et - timedelta(minutes=fallback_deadline_min)
```

And in the block around `scheduler.py:677-681`, replace:

```python
            if mins_to_game <= 15:
```

with:

```python
            if mins_to_game <= fallback_deadline_min:
```

- [ ] **Step 3: Update config example**

Modify `config/orchestrator.example.toml` to add the new keys under `[scheduler]`:

```toml
[scheduler]
lineup_check_offset_min = 45   # When to start polling for confirmed lineups before first pitch
fallback_deadline_min = 15     # Force-post deadline if nothing confirmed by then
missed_pick_alert_min = 10     # Alert threshold (minutes before first pitch) if no pick posted

# Other existing keys below...
early_lock_gap = 0.03
cluster_min = 10
doubleheader_recheck_min = 15
results_poll_interval_min = 15
results_cap_hour_et = 5
default_init_hour_et = 10
early_game_buffer_min = 60
```

**Note:** the defaults of `45`/`15`/`10` preserve current behavior for Pi5. The Fly production config will override these to larger values (e.g., 60/35/30) after Plan 01's lineup data analysis confirms the appropriate numbers.

- [ ] **Step 4: Run scheduler tests**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/ -v -k "scheduler"
```

Expected: All tests PASS. Default values mean zero behavior change from today.

- [ ] **Step 5: Commit**

```bash
git add src/bts/scheduler.py config/orchestrator.example.toml
git commit -m "feat(scheduler): extract fallback_deadline and alert_min into config"
```

---

### Task 4: Local execution tier in orchestrator

**Files:**
- Modify: `src/bts/orchestrator.py`
- Create: `tests/test_local_tier.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_local_tier.py`:

```python
"""Tests for the local execution tier (no SSH)."""
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from bts.orchestrator import predict_local, run_cascade


def test_predict_local_calls_run_pipeline_directly():
    fake_predictions = pd.DataFrame([
        {"batter_name": "A", "batter_id": 1, "p_game_hit": 0.85},
    ])
    with patch("bts.orchestrator.run_pipeline", return_value=fake_predictions) as mock_run:
        result = predict_local(date="2026-04-10")

    assert result is not None
    assert len(result) == 1
    mock_run.assert_called_once()


def test_predict_local_returns_none_on_exception():
    with patch("bts.orchestrator.run_pipeline", side_effect=RuntimeError("disk full")):
        result = predict_local(date="2026-04-10")
    assert result is None


def test_cascade_supports_local_tier_type():
    tiers = [
        {"name": "local", "type": "local"},
    ]
    fake_df = pd.DataFrame([{"batter_name": "X", "p_game_hit": 0.80}])
    with patch("bts.orchestrator.predict_local", return_value=fake_df):
        df, tier = run_cascade(tiers=tiers, date="2026-04-10")
    assert df is not None
    assert tier == "local"


def test_cascade_default_ssh_type_when_unspecified():
    """Existing [[tiers]] entries without type field still work (backward compat)."""
    tiers = [
        {"name": "mac", "ssh_host": "mac", "bts_dir": "/path", "timeout_min": 5},
    ]
    with patch("bts.orchestrator.ssh_predict", return_value=pd.DataFrame([{"p_game_hit": 0.9}])) as mock_ssh:
        df, tier = run_cascade(tiers=tiers, date="2026-04-10")
    assert df is not None
    assert tier == "mac"
    mock_ssh.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_local_tier.py -v
```

Expected: Tests FAIL with `ImportError: cannot import name 'predict_local'`.

- [ ] **Step 3: Add predict_local to orchestrator.py**

Add to `src/bts/orchestrator.py`:

```python
def predict_local(
    date: str,
    data_dir: str = "data/processed",
    models_dir: str = "data/models",
) -> pd.DataFrame | None:
    """Run predictions locally in-process (no SSH cascade).

    Equivalent to ssh_predict but runs run_pipeline directly. Used when
    the scheduler runs on the same machine as the data and models (i.e.,
    on the Fly cloud VM). Returns None on any failure, matching ssh_predict's
    contract.
    """
    from bts.model.predict import run_pipeline, save_blend, load_blend
    from pathlib import Path

    models_path = Path(models_dir)
    cache_path = models_path / f"blend_{date}.pkl"
    cached_blend = None
    if cache_path.exists():
        print(f"  [local] Loading cached model from {cache_path}", file=sys.stderr)
        cached_blend = load_blend(cache_path)

    try:
        predictions = run_pipeline(
            date, data_dir,
            cached_blend=cached_blend,
            save_blend_path=cache_path if not cached_blend else None,
        )
        return predictions
    except Exception as e:
        print(f"  [local] Prediction failed: {e}", file=sys.stderr)
        return None
```

And update `run_cascade` to dispatch on tier type:

```python
def run_cascade(
    tiers: list[dict],
    date: str,
) -> tuple[pd.DataFrame | None, str | None]:
    """Try each tier in order until one succeeds."""
    for tier in tiers:
        name = tier["name"]
        tier_type = tier.get("type", "ssh")  # Default ssh for backward compat
        print(f"Trying {name} ({tier_type})...", file=sys.stderr)

        if tier_type == "local":
            df = predict_local(date=date)
        elif tier_type == "ssh":
            df = ssh_predict(
                tier["ssh_host"],
                tier["bts_dir"],
                date,
                timeout_sec=tier["timeout_min"] * 60,
                platform=tier.get("platform", "unix"),
            )
        else:
            print(f"  [{name}] Unknown tier type: {tier_type}", file=sys.stderr)
            continue

        if df is not None:
            print(f"  [{name}] Success — {len(df)} predictions", file=sys.stderr)
            return df, name

    return None, None
```

- [ ] **Step 4: Run tests**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_local_tier.py tests/ -v -k "orchestrat or cascade"
```

Expected: New tests PASS. Existing orchestrator/cascade tests still PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bts/orchestrator.py tests/test_local_tier.py
git commit -m "feat(scheduler): add local execution tier type (no SSH)"
```

---

### Task 5: Bluesky password consolidation

**Files:**
- Modify: `src/bts/posting.py`
- Modify: `src/bts/dm.py`
- Create: `tests/test_bluesky_password.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_bluesky_password.py`:

```python
"""Tests for unified Bluesky password helper."""
from unittest.mock import patch

import pytest


def test_new_env_var_is_primary(monkeypatch):
    monkeypatch.delenv("BTS_BLUESKY_APP_PASSWORD", raising=False)
    monkeypatch.delenv("BTS_BLUESKY_PASSWORD", raising=False)
    monkeypatch.delenv("BTS_BLUESKY_DM_PASSWORD", raising=False)

    monkeypatch.setenv("BTS_BLUESKY_APP_PASSWORD", "new-unified-password")

    # Mock keychain lookup to return empty
    with patch("bts.posting.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 1  # keychain miss
        from bts.posting import get_bluesky_password
        assert get_bluesky_password() == "new-unified-password"


def test_falls_back_to_legacy_bts_bluesky_password(monkeypatch):
    monkeypatch.delenv("BTS_BLUESKY_APP_PASSWORD", raising=False)
    monkeypatch.setenv("BTS_BLUESKY_PASSWORD", "legacy-posting-password")
    monkeypatch.delenv("BTS_BLUESKY_DM_PASSWORD", raising=False)

    with patch("bts.posting.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 1
        from bts.posting import get_bluesky_password
        assert get_bluesky_password() == "legacy-posting-password"


def test_dm_module_uses_shared_helper(monkeypatch):
    monkeypatch.setenv("BTS_BLUESKY_APP_PASSWORD", "shared-password")

    with patch("bts.posting.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 1
        from bts.dm import get_bluesky_dm_password
        assert get_bluesky_dm_password() == "shared-password"


def test_dm_module_falls_back_to_legacy_dm_var(monkeypatch):
    monkeypatch.delenv("BTS_BLUESKY_APP_PASSWORD", raising=False)
    monkeypatch.delenv("BTS_BLUESKY_PASSWORD", raising=False)
    monkeypatch.setenv("BTS_BLUESKY_DM_PASSWORD", "legacy-dm-password")

    with patch("bts.posting.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 1
        from bts.dm import get_bluesky_dm_password
        assert get_bluesky_dm_password() == "legacy-dm-password"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_bluesky_password.py -v
```

Expected: Tests FAIL — current implementation only checks `BTS_BLUESKY_PASSWORD`, not the new unified var.

- [ ] **Step 3: Update posting.py**

Modify `src/bts/posting.py`'s `get_bluesky_password()` function. Find the current implementation (around line 60-90) and replace with:

```python
def get_bluesky_password() -> str:
    """Get the Bluesky app password.

    Priority order:
    1. BTS_BLUESKY_APP_PASSWORD env var (new canonical name)
    2. BTS_BLUESKY_PASSWORD env var (legacy posting name)
    3. BTS_BLUESKY_DM_PASSWORD env var (legacy DM name)
    4. macOS keychain 'bluesky-bts-app-password-dm'
    5. macOS keychain 'bluesky-bts-app-password' (legacy)
    """
    import os
    import subprocess

    # 1. New canonical env var
    pwd = os.environ.get("BTS_BLUESKY_APP_PASSWORD")
    if pwd:
        return pwd

    # 2. Legacy posting env var
    pwd = os.environ.get("BTS_BLUESKY_PASSWORD")
    if pwd:
        return pwd

    # 3. Legacy DM env var
    pwd = os.environ.get("BTS_BLUESKY_DM_PASSWORD")
    if pwd:
        return pwd

    # 4. macOS keychain (new canonical)
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-a", "claude-cli",
             "-s", "bluesky-bts-app-password-dm", "-w"],
            capture_output=True, text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except FileNotFoundError:
        pass

    # 5. macOS keychain (legacy)
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-a", "claude-cli",
             "-s", "bluesky-bts-app-password", "-w"],
            capture_output=True, text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except FileNotFoundError:
        pass

    raise RuntimeError(
        "Bluesky app password not found. Set BTS_BLUESKY_APP_PASSWORD "
        "or add to macOS keychain."
    )
```

- [ ] **Step 4: Update dm.py to use the shared helper**

Modify `src/bts/dm.py`. Replace the existing password helper (around lines 20-40) with:

```python
def get_bluesky_dm_password() -> str:
    """Alias for posting.get_bluesky_password — same password, different context.

    Kept as a separate function name for call-site clarity. DM and posting
    use the same app password since consolidation.
    """
    from bts.posting import get_bluesky_password
    return get_bluesky_password()
```

Update the DM-sending code that calls the old password helper to call `get_bluesky_dm_password()` instead.

- [ ] **Step 5: Run tests**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_bluesky_password.py -v
```

Expected: All four tests PASS.

- [ ] **Step 6: Run full test suite to catch regressions**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/ -v -k "posting or dm"
```

Expected: All existing tests still PASS.

- [ ] **Step 7: Commit**

```bash
git add src/bts/posting.py src/bts/dm.py tests/test_bluesky_password.py
git commit -m "feat(bluesky): consolidate password helper with new unified env var"
```

---

### Task 6: `/health` HTTP endpoint in dashboard

**Files:**
- Modify: `src/bts/web.py`
- Create: `tests/test_health_endpoint.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_health_endpoint.py`:

```python
"""Tests for the /health endpoint."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from bts.heartbeat import write_heartbeat, HeartbeatState


@pytest.fixture
def web_client(tmp_path, monkeypatch):
    """Build a Flask test client pointing at a temp heartbeat path."""
    heartbeat_path = tmp_path / ".heartbeat"
    monkeypatch.setenv("BTS_HEARTBEAT_PATH", str(heartbeat_path))

    # Reload web module so it picks up the env var
    import importlib
    import bts.web as web
    importlib.reload(web)

    app = web.app
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client, heartbeat_path


def test_health_200_when_heartbeat_fresh(web_client):
    client, heartbeat_path = web_client
    write_heartbeat(heartbeat_path, state=HeartbeatState.RUNNING)

    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"
    assert data["scheduler_state"] == HeartbeatState.RUNNING


def test_health_503_when_heartbeat_missing(web_client):
    client, _ = web_client
    response = client.get("/health")
    assert response.status_code == 503
    data = response.get_json()
    assert data["status"] == "stale"


def test_health_200_when_sleeping(web_client):
    client, heartbeat_path = web_client
    wake = datetime.now(timezone.utc) + timedelta(hours=3)
    write_heartbeat(
        heartbeat_path,
        state=HeartbeatState.SLEEPING,
        sleeping_until=wake,
    )

    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["scheduler_state"] == HeartbeatState.SLEEPING
```

- [ ] **Step 2: Run test to verify it fails**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_health_endpoint.py -v
```

Expected: Tests FAIL because `/health` endpoint doesn't exist yet.

- [ ] **Step 3: Add /health route to web.py**

In `src/bts/web.py`, near the top with other imports:

```python
import os
from bts.heartbeat import read_heartbeat, is_heartbeat_fresh
```

Find where `app = Flask(...)` is defined and add a constant near it:

```python
HEARTBEAT_PATH = Path(os.environ.get("BTS_HEARTBEAT_PATH", "data/.heartbeat"))
```

Add a route handler:

```python
@app.route("/health")
def health():
    """Health check endpoint for Fly HTTP checks and external monitors.

    Returns 200 if scheduler heartbeat is fresh (including intentional sleeps).
    Returns 503 if heartbeat is missing, stale, or unreadable.
    """
    hb = read_heartbeat(HEARTBEAT_PATH)
    if hb is None:
        return jsonify({"status": "stale", "reason": "no heartbeat file"}), 503

    fresh = is_heartbeat_fresh(HEARTBEAT_PATH, max_age_sec=180)
    if not fresh:
        ts = hb.get("timestamp", "unknown")
        return jsonify({
            "status": "stale",
            "last_heartbeat": ts,
            "scheduler_state": hb.get("state"),
        }), 503

    return jsonify({
        "status": "ok",
        "last_heartbeat": hb.get("timestamp"),
        "scheduler_state": hb.get("state"),
        "sleeping_until": hb.get("sleeping_until"),
    }), 200
```

If `jsonify` is not already imported at the top of `web.py`, add it to the Flask imports.

- [ ] **Step 4: Run tests**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_health_endpoint.py -v
```

Expected: All three tests PASS.

- [ ] **Step 5: Run the full web test suite to catch regressions**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/ -v -k "web"
```

Expected: All existing web tests still PASS.

- [ ] **Step 6: Commit**

```bash
git add src/bts/web.py tests/test_health_endpoint.py
git commit -m "feat(web): add /health endpoint reading scheduler heartbeat"
```

---

### Task 7: Integration smoke test

**Files:** (no new files, just a smoke-test runbook)

- [ ] **Step 1: Run the full test suite**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/ -v
```

Expected: all tests pass, no regressions. Count should include the new tests from Tasks 1-6.

- [ ] **Step 2: Manual smoke test — local scheduler with heartbeat**

In one terminal:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run bts schedule --config ~/.bts-orchestrator.toml --dry-run
```

In another terminal:

```bash
cat data/.heartbeat
```

Expected: heartbeat file contains JSON like `{"state": "running", "timestamp": "..."}` and updates over time.

- [ ] **Step 3: Manual smoke test — /health endpoint**

Start the dashboard:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -c "from bts.web import app; app.run(port=3003, debug=False)" &
DASH_PID=$!

# Heartbeat is missing → expect 503
curl -s http://localhost:3003/health | python -m json.tool

# Write a heartbeat and retry
UV_CACHE_DIR=/tmp/uv-cache uv run python -c "
from bts.heartbeat import write_heartbeat
from pathlib import Path
write_heartbeat(Path('data/.heartbeat'), state='running')
"

curl -s http://localhost:3003/health | python -m json.tool

# Clean up
kill $DASH_PID
```

Expected: first curl returns `{"status": "stale", ...}`; second curl returns `{"status": "ok", "scheduler_state": "running", ...}`.

- [ ] **Step 4: Commit the smoke test notes in plan (no code)**

Nothing to commit — this task verifies the changes work together.

---

## Completion criteria for Plan 04

- [ ] All tests pass: `uv run pytest tests/test_heartbeat.py tests/test_local_tier.py tests/test_bluesky_password.py tests/test_health_endpoint.py -v`
- [ ] Existing scheduler tests still pass (no regressions): `uv run pytest tests/ -v -k "scheduler"`
- [ ] Heartbeat file is written when `bts schedule --dry-run` runs locally
- [ ] `/health` endpoint returns 200 with fresh heartbeat, 503 when stale
- [ ] `predict_local` in orchestrator.py produces the same DataFrame shape as `ssh_predict`
- [ ] Pi5 continues to work with the cascade config (no behavior change for existing deployment)
- [ ] `BTS_BLUESKY_APP_PASSWORD` env var is honored by both posting and DM code paths

**Next plan:** `05-fly.md` — Fly infrastructure (Dockerfile, fly.toml, CI/CD workflow). Depends on Plan 02 (R2 sync) and this plan (heartbeat + /health).
