# BTS Model Orchestration — Design Spec

## Goal

Make daily BTS picks reliable without human intervention, even if the Mac is off for days. Pi5 orchestrates model runs across a cascade of compute machines: Mac → Alienware → cloud VPS.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Pi5 (Orchestrator)                             │
│  - Cron: 11am / 4pm / 7:30pm / 1am ET          │
│  - Cascade: Mac → Alienware → Cloud VPS         │
│  - Owns: pick logic, Bluesky, streak, dashboard │
└──────────┬──────────┬──────────┬────────────────┘
           │ SSH      │ SSH      │ SSH
     ┌─────▼──┐  ┌────▼─────┐  ┌▼──────────┐
     │  Mac   │  │Alienware │  │ Cloud VPS │
     │ Worker │  │ Worker   │  │ Worker    │
     └────────┘  └──────────┘  └───────────┘
     Each: date in → predictions JSON out
     Static data (parquets) lives on each worker
```

### Separation of Concerns

**Workers** are stateless compute. They receive a date, fetch today's lineups from the MLB API, run the 12-model blend, and return ranked predictions as JSON to stdout. No pick logic, no Bluesky credentials, no state.

**Pi5** is the brain. It receives predictions, applies the densest bucket + override strategy, saves picks, posts to Bluesky, tracks the streak, serves the dashboard, and sends DM notifications on failure.

### Cascade Logic

Pi5 tries each tier sequentially. On success (valid JSON on stdout, exit code 0), it stops and processes the predictions. On failure (SSH error, timeout, non-zero exit, invalid JSON), it moves to the next tier.

| Tier | SSH Host | Timeout | Notes |
|------|----------|---------|-------|
| Mac | macbook-pro.local (LAN) | 5 min | Primary — known ~2 min runtime |
| Alienware | alienware (Tailscale) | 10 min | Untested — generous default |
| Cloud VPS | TBD | 15 min | Provider TBD, researched at implementation |

If all tiers fail, Pi5 sends a Bluesky DM to @stonehengee.

## Worker Interface: `bts predict-json`

New CLI command. The worker's entire contract:

```bash
# Pi5 runs this on a worker via SSH:
ssh mac "cd ~/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run bts predict-json --date 2026-04-01"
```

**Input:** `--date YYYY-MM-DD`

**Output:** JSON array to stdout — ranked predictions:

```json
[
  {
    "batter_name": "Jacob Wilson",
    "batter_id": 123456,
    "team": "ATH",
    "lineup": 1,
    "pitcher_name": "José Suarez",
    "pitcher_id": 654321,
    "game_pk": 789012,
    "game_time": "2026-04-01T23:10:00Z",
    "p_hit_pa": 0.312,
    "p_game_hit": 0.763,
    "flags": "PROJECTED"
  }
]
```

**Implementation:** Thin wrapper around `run_pipeline()`. All log/status messages go to stderr so stdout is clean JSON. Returns the full ranked list — Pi5 applies the pick strategy.

## Pick Logic Extraction: `strategy.py`

Currently the densest bucket + override logic is inline in `cli.py`'s `run` command (~200 lines mixing prediction and pick logic). This gets extracted into a shared module.

**New module: `src/bts/strategy.py`**

```python
def select_pick(
    predictions: pd.DataFrame,
    date: str,
    picks_dir: Path,
) -> DailyPick | None:
    """Apply densest bucket + override strategy to predictions.

    - Filters to games not yet started
    - Applies densest bucket (most games) as default window
    - Non-densest picks override only if P(game hit) > 78%
    - Checks for double-down (P(both hit) >= 65%)
    - Checks if current pick is locked (game started or already posted)
    - Returns DailyPick ready to save/post, or None
    """
```

**Callers:**
- `bts run` (local convenience): calls `run_pipeline()` → `select_pick()` → `save_pick()` → `post_to_bluesky()`
- Pi5 orchestrator: parses JSON into DataFrame → `select_pick()` → `save_pick()` → `post_to_bluesky()`

Same function, two callers. No divergence.

## Refactored `bts run`

Stays as a local convenience command. Becomes a thin wrapper:

1. Call `run_pipeline()` (prediction)
2. Call `select_pick()` (strategy — shared module)
3. Call `save_pick()` + `post_to_bluesky()` (output)

Useful for manual overrides and development. Not the production path.

## Optional Dependencies

LightGBM moves from required to optional in `pyproject.toml`:

```toml
[project]
dependencies = ["click", "pandas", "requests", "pyarrow", ...]

[project.optional-dependencies]
model = ["lightgbm", "scikit-learn"]
```

- **Mac / Alienware / Cloud:** `uv sync --extra model`
- **Pi5:** `uv sync` (pick logic + posting only, no LightGBM)

The `predict-json` command uses lazy imports for LightGBM so it only fails if you actually try to run predictions on a machine without model deps.

## Pi5 Orchestrator: `orchestrator.py`

Python script on Pi5. Core flow:

```
for tier in config.tiers:
    predictions = ssh_predict(tier.host, date, tier.timeout)
    if predictions is not None:
        break
else:
    dm_stonehengee("All tiers failed for {date}")
    return

pick = select_pick(predictions, date, picks_dir)
if pick:
    save_pick(pick, picks_dir)
    post_to_bluesky(pick)
```

**Config file** on Pi5 (`~/.bts-orchestrator.yaml`):

```yaml
tiers:
  - name: mac
    ssh_host: macbook-pro.local
    bts_dir: /Users/stone/projects/bts
    timeout_min: 5
  - name: alienware
    ssh_host: alienware
    bts_dir: /c/Users/stone/projects/bts
    timeout_min: 10
  - name: cloud
    ssh_host: bts-cloud  # SSH config alias, provider TBD
    bts_dir: /home/bts/projects/bts
    timeout_min: 15

bluesky:
  handle: beatthestreakbot.bsky.social
  post_password_keychain: bluesky-bts-app-password
  dm_password_keychain: bluesky-bts-app-password-dm
  dm_recipient: stonehengee.bsky.social

picks_dir: /home/stonehengee/projects/bts/data/picks
```

**Cron on Pi5** (same schedule as current Mac cron):

```
0 11 * * * /path/to/orchestrator.py --date $(date +%Y-%m-%d)
0 16 * * * /path/to/orchestrator.py --date $(date +%Y-%m-%d)
30 19 * * * /path/to/orchestrator.py --date $(date +%Y-%m-%d)
0 1  * * * cd /path/to/bts && uv run bts check-results --date $(date -d yesterday +%Y-%m-%d)
```

## Deployment Setup

### Pi5 (Orchestrator)
- Clone BTS repo (or use existing if dashboard is already deployed)
- `uv sync` (no `--extra model`)
- Install cron entries
- Store Bluesky app passwords (posting + DM)
- SSH config entries for mac, alienware, cloud
- Config file at `~/.bts-orchestrator.yaml`

### Mac (Primary Worker)
- Already has repo + data + deps
- Verify SSH accessible from Pi5 via LAN (`ssh macbook-pro.local`)
- No other changes needed

### Alienware (Secondary Worker)
- Clone BTS repo
- `uv sync --extra model`
- Rsync parquet data from Mac: `rsync -az mac:~/projects/bts/data/processed/ data/processed/`
- Verify `bts predict-json --date 2025-09-15` works
- Verify SSH accessible from Pi5 via Tailscale

### Cloud VPS (Tertiary Worker)
- Provider TBD — researched at implementation time
- Requirements: SSH access, 4GB+ RAM, persistent ~2GB storage, Python 3.12
- Repo cloned, `uv sync --extra model`, parquet data pre-staged
- SSH accessible from Pi5 (public IP or Tailscale)

### Data Refresh
When parquets are rebuilt (start of season, new data pull), rsync to all workers:
```bash
rsync -az data/processed/ alienware:~/projects/bts/data/processed/
rsync -az data/processed/ bts-cloud:~/projects/bts/data/processed/
```

This is infrequent — 2-3 times per season.

## Error Handling

- **SSH failure:** connection refused, timeout, auth error → log, next tier
- **Command failure:** non-zero exit code → log stderr output, next tier
- **Invalid JSON:** stdout doesn't parse as JSON → log raw output, next tier
- **No games:** valid empty predictions → skip the day, no pick, no post
- **All tiers fail:** DM @stonehengee via Bluesky with error summary
- **Logging:** all attempts logged to `data/picks/orchestrator.log` on Pi5

## Bluesky Credentials

- **Posting:** `bluesky-bts-app-password` in macOS keychain (Mac) / equivalent on Pi5
- **DM notifications:** `bluesky-bts-app-password-dm` in keychain — new password with DM scope
- Old posting password stays for backwards compatibility; new DM password used for failure notifications

## What This Does NOT Change

- `bts check-results` — unchanged, runs directly on Pi5 at 1am ET (no cascade needed — only reads MLB API + local pick files, no LightGBM)
- `bts data pull` / `bts data build` — unchanged, run manually on Mac
- Web dashboard — unchanged, serves from Pi5
- Bluesky posting logic — unchanged, just called from Pi5 instead of Mac
- 12-model blend, features, evaluation — no model changes
