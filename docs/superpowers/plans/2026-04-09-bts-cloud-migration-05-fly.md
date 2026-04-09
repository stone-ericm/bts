# BTS Cloud Migration — Plan 05: Fly Infrastructure

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Fly.io deployment infrastructure: Dockerfile for the BTS container image, fly.toml for app configuration, GitHub Actions workflow for git-driven auto-deploys, entrypoint script that supervises scheduler + dashboard + periodic cron jobs inside the container, and Tailscale sidecar for private dashboard access.

**Architecture:** Single Docker container runs a shell-based supervisor (no full-blown PID 1 framework) that starts: (1) Tailscale daemon joining the tailnet as `tag:bts-prod`, (2) `bts schedule` scheduler daemon, (3) Flask dashboard serving `/health` + the main UI, (4) a lightweight cron loop for nightly data pull/build, reconcile, check-results, and Healthchecks.io pings. Persistent state lives on a Fly volume mounted at `/data`. Deploys happen via git push to main → GitHub Actions → `flyctl deploy --remote-only`.

**Tech Stack:** Docker (python:3.12-slim-bookworm base), Fly.io machines + volumes, `flyctl` CLI, GitHub Actions with `superfly/flyctl-actions`, Tailscale client (installed in image), shell supervisor script.

**Dependencies on other plans:**
- Plan 02 (R2 sync) — the container's bootstrap step calls `bts data sync-from-r2` to populate the volume with parquets on cold start
- Plan 04 (scheduler refactors) — the container uses the "local" tier type and the `/health` endpoint is served by the dashboard

**Parent spec:** `docs/superpowers/specs/2026-04-09-bts-cloud-migration-design.md` (§ Fly.io deployment, § Container, § Dashboard access, § Deployment flow)

---

## File Structure

- Create `Dockerfile` — container image definition, ~60 lines
- Create `fly.toml` — Fly app + machine config, ~80 lines
- Create `scripts/fly-entrypoint.sh` — supervisor script that starts all services, ~100 lines
- Create `scripts/fly-cron-loop.sh` — lightweight cron for periodic jobs, ~60 lines
- Create `scripts/fly-bootstrap.sh` — one-shot cold-start: sync from R2, pull current season, etc., ~40 lines
- Create `.github/workflows/deploy.yml` — CI/CD workflow, ~40 lines
- Create `.dockerignore` — exclude local data, git dir, tests from image, ~20 lines
- Create `docs/fly-operations.md` — runbook for fly-related ops (deploy, rollback, inspect), ~100 lines

---

### Task 1: Dockerfile and .dockerignore

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`

- [ ] **Step 1: Create the .dockerignore**

Create `.dockerignore`:

```
# Exclude local data and git from the image — volume mounts handle runtime data
.git/
.gitignore
data/raw/
data/processed/
data/models/blend_*
data/models/probable_pitcher_lookup.json
data/picks/
data/backup_results/
data/lineup_posting_times/
data/shadow/
data/simulation/
data/validation/
data/external/

# Python artifacts
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/

# Build artifacts
dist/
build/
*.egg-info/

# IDE
.vscode/
.idea/

# Tests — not needed in production image
tests/

# Docs — not needed in runtime
docs/
CLAUDE.md
ARCHITECTURE.md
README.md

# Experiments and scripts
experiments/
notebooks/
```

- [ ] **Step 2: Create the Dockerfile**

Create `Dockerfile`:

```dockerfile
# Multi-stage build: install deps in one layer, final runtime in another
FROM python:3.12-slim-bookworm AS deps

# System deps for LightGBM + SSL + Tailscale + utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Install Tailscale
RUN curl -fsSL https://pkgs.tailscale.com/stable/debian/bookworm.noarmor.gpg \
    | tee /usr/share/keyrings/tailscale-archive-keyring.gpg > /dev/null \
    && curl -fsSL https://pkgs.tailscale.com/stable/debian/bookworm.tailscale-keyring.list \
    | tee /etc/apt/sources.list.d/tailscale.list \
    && apt-get update \
    && apt-get install -y tailscale \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Copy dependency files first so this layer caches
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/tmp/uv-cache \
    UV_CACHE_DIR=/tmp/uv-cache uv sync --extra model --frozen --no-dev

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/fly-entrypoint.sh scripts/fly-cron-loop.sh scripts/fly-bootstrap.sh ./scripts/

# Ensure scripts are executable
RUN chmod +x scripts/fly-entrypoint.sh scripts/fly-cron-loop.sh scripts/fly-bootstrap.sh

# Create data directory (will be overridden by volume mount at runtime)
RUN mkdir -p /data

# Use tini as PID 1 for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["./scripts/fly-entrypoint.sh"]
```

- [ ] **Step 3: Build the image locally to verify it works**

```bash
docker build -t bts:test .
```

Expected: build succeeds (may take several minutes for the first build due to apt-get + uv sync). No errors.

- [ ] **Step 4: Verify the image runs without crashing**

```bash
docker run --rm bts:test uv run bts --help
```

Expected: prints BTS CLI help text. (The container is not joining Tailscale or running the scheduler — this just verifies the image boots.)

- [ ] **Step 5: Commit**

```bash
git add Dockerfile .dockerignore
git commit -m "feat(fly): add Dockerfile + .dockerignore for BTS container"
```

---

### Task 2: Entrypoint script

**Files:**
- Create: `scripts/fly-entrypoint.sh`

- [ ] **Step 1: Write the entrypoint script**

Create `scripts/fly-entrypoint.sh`:

```bash
#!/bin/bash
# BTS Fly container entrypoint.
#
# Starts (in order):
# 1. Tailscale daemon (joins tailnet as tag:bts-prod)
# 2. Cold bootstrap if data volume is empty
# 3. Dashboard web server
# 4. Scheduler daemon
# 5. Cron loop (background)
#
# Uses simple shell supervision: if any critical service exits, the
# container exits (letting Fly restart it). Logs to stdout/stderr for
# Fly log collection.

set -euo pipefail

log() { echo "[entrypoint] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

log "Starting BTS Fly entrypoint"

# --- 1. Tailscale ---
if [ -n "${TS_AUTHKEY:-}" ]; then
    log "Starting Tailscale daemon"
    mkdir -p /var/lib/tailscale
    # tailscaled needs to run in the background
    tailscaled --state=/var/lib/tailscale/tailscaled.state \
               --socket=/var/run/tailscale/tailscaled.sock &
    TAILSCALED_PID=$!

    # Wait for tailscaled to be ready
    for i in $(seq 1 30); do
        if tailscale status >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done

    log "Joining tailnet with ephemeral key"
    tailscale up \
        --authkey="${TS_AUTHKEY}" \
        --hostname="bts-fly" \
        --advertise-tags=tag:bts-prod \
        --accept-routes=false \
        --ssh=false
    log "Tailscale joined: $(tailscale ip -4)"
else
    log "WARNING: TS_AUTHKEY not set — Tailscale will not join the tailnet"
fi

# --- 2. Cold bootstrap if needed ---
cd /app
if [ ! -f /data/processed/pa_2026.parquet ]; then
    log "No parquets on volume — running cold bootstrap"
    ./scripts/fly-bootstrap.sh
else
    log "Parquets present on volume — skipping bootstrap"
fi

# --- 3. Dashboard (Flask) in background ---
log "Starting dashboard on 0.0.0.0:3003"
export BTS_HEARTBEAT_PATH=/data/.heartbeat
UV_CACHE_DIR=/tmp/uv-cache uv run python -c "
from bts.web import app
app.run(host='0.0.0.0', port=3003, debug=False, use_reloader=False)
" &
DASHBOARD_PID=$!
log "Dashboard PID: $DASHBOARD_PID"

# --- 4. Cron loop in background ---
log "Starting cron loop"
./scripts/fly-cron-loop.sh &
CRON_PID=$!
log "Cron PID: $CRON_PID"

# --- 5. Scheduler daemon (foreground) ---
log "Starting scheduler daemon"
export BTS_HEARTBEAT_PATH=/data/.heartbeat

# Trap signals to cleanly shut down children
cleanup() {
    log "Shutting down"
    kill "$DASHBOARD_PID" "$CRON_PID" 2>/dev/null || true
    if [ -n "${TAILSCALED_PID:-}" ]; then
        kill "$TAILSCALED_PID" 2>/dev/null || true
    fi
    exit 0
}
trap cleanup SIGTERM SIGINT

# Scheduler runs in foreground; systemd-style restart loop
while true; do
    UV_CACHE_DIR=/tmp/uv-cache uv run bts schedule --config /data/orchestrator.toml || true
    log "Scheduler exited, restarting in 60s"
    sleep 60
done
```

Make it executable:

```bash
chmod +x scripts/fly-entrypoint.sh
```

- [ ] **Step 2: Commit**

```bash
git add scripts/fly-entrypoint.sh
git commit -m "feat(fly): add fly-entrypoint supervisor script"
```

---

### Task 3: Cron loop and bootstrap scripts

**Files:**
- Create: `scripts/fly-cron-loop.sh`
- Create: `scripts/fly-bootstrap.sh`

- [ ] **Step 1: Write the bootstrap script**

Create `scripts/fly-bootstrap.sh`:

```bash
#!/bin/bash
# Cold bootstrap: runs when the Fly volume is empty (first boot of a
# fresh machine, or after a volume swap). Populates /data with enough
# state to start running the scheduler.

set -euo pipefail
log() { echo "[bootstrap] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

cd /app
mkdir -p /data/processed /data/models /data/picks /data/raw /data/lineup_posting_times

log "Downloading parquets from R2"
UV_CACHE_DIR=/tmp/uv-cache uv run bts data sync-from-r2 \
    || { log "sync-from-r2 failed — cannot bootstrap"; exit 2; }

# Point bts data/models at the volume via symlinks for tools that read
# the default paths (most scheduler code reads data/processed relative to cwd)
rm -rf data/processed data/models data/picks data/raw data/lineup_posting_times
ln -s /data/processed data/processed
ln -s /data/models data/models
ln -s /data/picks data/picks
ln -s /data/raw data/raw
ln -s /data/lineup_posting_times data/lineup_posting_times

# If initial-state.json is in the repo, bootstrap picks from it
if [ -f /app/data/state/initial-state.json ]; then
    log "Found initial-state.json — regenerating pick files from it"
    UV_CACHE_DIR=/tmp/uv-cache uv run bts state regenerate \
        --snapshot /app/data/state/initial-state.json \
        --out-picks-dir /data/picks \
        || log "WARNING: state regenerate failed (may be OK if no Bluesky data yet)"
fi

# Pull current-season raw feeds to catch up
log "Pulling current-season raw feeds"
UV_CACHE_DIR=/tmp/uv-cache uv run bts data pull \
    || log "WARNING: data pull failed (will retry on next cron)"

# Build current-season parquet
log "Building current-season parquet"
UV_CACHE_DIR=/tmp/uv-cache uv run bts data build \
    || log "WARNING: data build failed (will retry on next cron)"

# Write config file to /data so it persists across deploys
if [ ! -f /data/orchestrator.toml ]; then
    log "Writing default orchestrator config"
    cat > /data/orchestrator.toml <<'EOF'
[orchestrator]
picks_dir = "/data/picks"
heartbeat_path = "/data/.heartbeat"

[bluesky]
dm_recipient = "did:plc:replace-me"

[scheduler]
lineup_check_offset_min = 60
fallback_deadline_min = 35
missed_pick_alert_min = 30
early_lock_gap = 0.03
cluster_min = 10
doubleheader_recheck_min = 15
results_poll_interval_min = 15
results_cap_hour_et = 5
default_init_hour_et = 10
early_game_buffer_min = 60

[[tiers]]
name = "local"
type = "local"
EOF
fi

log "Bootstrap complete"
```

Make it executable:

```bash
chmod +x scripts/fly-bootstrap.sh
```

- [ ] **Step 2: Write the cron loop script**

Create `scripts/fly-cron-loop.sh`:

```bash
#!/bin/bash
# Lightweight cron loop for BTS periodic jobs.
#
# Runs forever, checking the clock every minute to decide which jobs
# should fire now. Logs to stdout (captured by Fly logs). Jobs run
# sequentially to avoid resource contention with the scheduler.
#
# Schedule (ET):
# - 01:00  check-results (safety net; scheduler usually resolves during game)
# - 02:00  reconcile (8-day scoring rescan)
# - 03:00  data pull + build + sync-to-r2 (refresh parquets for next day)
# - */5    lineup time collection (every 5 min during active window)
# - */5    healthchecks.io ping

set -euo pipefail
log() { echo "[cron] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

cd /app
export UV_CACHE_DIR=/tmp/uv-cache

HC_PING_URL="${HEALTHCHECKS_PING_URL:-}"

while true; do
    HOUR=$(TZ=America/New_York date +%H)
    MIN=$(TZ=America/New_York date +%M)
    YESTERDAY=$(TZ=America/New_York date -d "yesterday" +%Y-%m-%d)

    # 01:00 ET check-results
    if [ "$HOUR" = "01" ] && [ "$MIN" = "00" ]; then
        log "Running check-results for $YESTERDAY"
        uv run bts check-results --date "$YESTERDAY" || log "check-results failed"
    fi

    # 02:00 ET reconcile
    if [ "$HOUR" = "02" ] && [ "$MIN" = "00" ]; then
        log "Running reconcile"
        uv run bts reconcile || log "reconcile failed"
    fi

    # 03:00 ET data pull + build + sync-to-r2
    if [ "$HOUR" = "03" ] && [ "$MIN" = "00" ]; then
        log "Running nightly data refresh"
        uv run bts data pull && uv run bts data build && uv run bts data sync-to-r2 \
            || log "data refresh failed"
    fi

    # Every 5 min: lineup time collection (runs its own active-window check internally)
    if [ $((10#$MIN % 5)) -eq 0 ]; then
        uv run bts data collect-lineup-times \
            --out-dir /data/lineup_posting_times \
            2>&1 | head -5 || log "collect-lineup-times failed"
    fi

    # Every 5 min: healthchecks.io ping
    if [ $((10#$MIN % 5)) -eq 0 ] && [ -n "$HC_PING_URL" ]; then
        curl -fsS --max-time 5 "$HC_PING_URL" >/dev/null \
            || log "healthchecks ping failed"
    fi

    sleep 60
done
```

Make it executable:

```bash
chmod +x scripts/fly-cron-loop.sh
```

- [ ] **Step 3: Commit**

```bash
git add scripts/fly-bootstrap.sh scripts/fly-cron-loop.sh
git commit -m "feat(fly): add bootstrap + cron loop scripts"
```

---

### Task 4: fly.toml

**Files:**
- Create: `fly.toml`

- [ ] **Step 1: Create the Fly app config**

Create `fly.toml`:

```toml
# Fly.io app configuration for BTS production
# Deploy with: flyctl deploy --remote-only
# Or via GitHub Actions on push to main (see .github/workflows/deploy.yml)

app = "bts"
primary_region = "iad"

[build]

[env]
  PYTHONUNBUFFERED = "1"
  UV_CACHE_DIR = "/tmp/uv-cache"
  BTS_HEARTBEAT_PATH = "/data/.heartbeat"
  TZ = "America/New_York"

[mounts]
  source = "data"
  destination = "/data"
  initial_size = "50gb"

[[services]]
  protocol = "tcp"
  internal_port = 3003
  auto_stop_machines = false
  auto_start_machines = true
  min_machines_running = 1

  [[services.ports]]
    port = 3003
    # No handlers = raw TCP (only reachable via Tailscale, not public internet)

  [services.concurrency]
    type = "connections"
    hard_limit = 50
    soft_limit = 25

[[services.http_checks]]
  interval = "60s"
  timeout = "5s"
  grace_period = "2m"
  method = "get"
  path = "/health"
  protocol = "http"
  tls_skip_verify = false

[[vm]]
  cpu_kind = "shared"
  cpus = 2
  memory_mb = 4096

[deploy]
  strategy = "immediate"
```

**Important notes about this config:**
- `min_machines_running = 1` with `auto_stop_machines = false` keeps the machine always-on
- The `[[services]]` section uses raw TCP on port 3003 with no public handlers — the dashboard is only reachable via Tailscale. This avoids the need for a public IPv4 address.
- Health check polls `/health` every 60s with a 2-min grace period on startup
- `shared-cpu-2x` with 4 GB RAM is the sizing from the spec

- [ ] **Step 2: Validate fly.toml syntax**

```bash
flyctl config validate
```

Expected: "Configuration is valid"

Note: You need to have run `fly apps create bts` first (Phase 0 Task in cutover plan). If not, this step will fail with "app not found" — that's fine for now, validation of the toml syntax still happens before the app lookup.

- [ ] **Step 3: Commit**

```bash
git add fly.toml
git commit -m "feat(fly): add fly.toml with shared-cpu-2x + 50GB volume"
```

---

### Task 5: GitHub Actions deploy workflow

**Files:**
- Create: `.github/workflows/deploy.yml`

- [ ] **Step 1: Create the workflow**

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy BTS to Fly

on:
  push:
    branches: [main]
    paths:
      - "src/**"
      - "scripts/fly-**"
      - "Dockerfile"
      - ".dockerignore"
      - "fly.toml"
      - "pyproject.toml"
      - "uv.lock"
      - ".github/workflows/deploy.yml"

concurrency:
  group: deploy-bts
  cancel-in-progress: true

jobs:
  deploy:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup flyctl
        uses: superfly/flyctl-actions/setup-flyctl@master

      - name: Deploy to Fly
        run: flyctl deploy --remote-only --strategy immediate
        env:
          FLY_API_TOKEN: ${{ secrets.FLY_API_TOKEN }}
```

**Key properties:**
- `paths:` filter ensures any data-only commits don't trigger deploys
- `concurrency.cancel-in-progress: true` means if you push twice in quick succession, the older deploy is cancelled
- `permissions: contents: read` minimizes the scope available to the deploy workflow

- [ ] **Step 2: Verify the workflow syntax**

The easiest way to verify: commit, push, and see if GitHub accepts the workflow file. GitHub's actions YAML linter is very strict and will reject invalid syntax immediately.

Or use `actionlint` if installed:

```bash
actionlint .github/workflows/deploy.yml
```

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/deploy.yml
git commit -m "feat(fly): add GitHub Actions deploy workflow"
```

---

### Task 6: Operations runbook

**Files:**
- Create: `docs/fly-operations.md`

- [ ] **Step 1: Write the runbook**

Create `docs/fly-operations.md`:

```markdown
# Fly Operations Runbook for BTS

This document is the go-to reference for day-to-day Fly operations on the
BTS production deployment. If something's wrong and you're tired, this is
what to grep for.

## Quick status

    # Is the machine running?
    flyctl status -a bts

    # Recent logs (streaming)
    flyctl logs -a bts

    # SSH into the running container
    flyctl ssh console -a bts

    # Machine list with IDs
    flyctl machines list -a bts

## Deployment

Normal deploys happen automatically via GitHub Actions on push to main.

### Manual deploy

    flyctl deploy --remote-only --strategy immediate

The `--strategy immediate` means the new machine replaces the old one as
soon as it's built. Expect ~30-60 seconds of missed heartbeats during
the swap; Fly HTTP checks tolerate 2 consecutive failures before panicking.

### Rollback to a previous image

    flyctl releases -a bts                # list recent releases
    flyctl releases rollback <version>    # rollback to a specific one

## Volume management

The `data` volume at `/data` holds parquets, daily model caches, pick history,
and scheduler state. It's regional (stuck in iad).

### Inspect the volume

    flyctl volumes list -a bts
    flyctl ssh console -a bts -C "ls -la /data"
    flyctl ssh console -a bts -C "du -sh /data/*"

### Restore from a snapshot

    flyctl volumes snapshots list bts_data_volume
    flyctl volumes snapshots create bts_data_volume    # force a snapshot now
    # To restore: create a new volume from snapshot, then attach to a new machine
    flyctl volumes create data_restore --snapshot-id <snapshot> --size 50 --region iad -a bts

## Common issues

### "No pick was made today"

1. Check if the scheduler is running:
        flyctl logs -a bts --since 2h | grep -i scheduler
2. Check heartbeat file:
        flyctl ssh console -a bts -C "cat /data/.heartbeat"
3. Check /health endpoint via Tailscale:
        curl http://bts:3003/health
4. If scheduler is wedged: restart the machine
        flyctl machines restart <machine-id> -a bts

### "Deploy failed"

Check the GitHub Actions run log first (Actions tab of the repo). Common causes:
- `FLY_API_TOKEN` expired or revoked → regenerate via `fly tokens create deploy -a bts`, update GitHub Secret
- Docker build failure → check the Dockerfile step that failed; most commonly a dependency issue in `uv.lock`
- Image size too large → check `.dockerignore` is excluding `data/raw/`

### "Dashboard unreachable via Tailscale"

1. Check Tailscale sidecar joined:
        flyctl ssh console -a bts -C "tailscale status"
2. Verify tag and ACL:
        # On your local machine
        tailscale status | grep bts-fly
3. Restart tailscaled inside the container:
        flyctl ssh console -a bts -C "pkill tailscaled"
        flyctl machines restart <machine-id> -a bts

### "I need to run a one-off command"

    flyctl ssh console -a bts -C "cd /app && uv run bts data verify-manifest"

Use `ssh console -C` for ephemeral commands; use plain `ssh console` for an interactive shell.

## Secrets rotation

### Rotate Bluesky app password

1. Create a new app password in Bluesky admin UI
2. `flyctl secrets set -a bts BTS_BLUESKY_APP_PASSWORD=<new-value>`
3. Revoke the old password in Bluesky
4. Verify: `flyctl logs -a bts --since 10m | grep -i bluesky`

### Rotate R2 API token

1. Create new R2 API token in Cloudflare dashboard
2. `flyctl secrets set -a bts R2_ACCESS_KEY_ID=<new> R2_SECRET_ACCESS_KEY=<new>`
3. Revoke old token in Cloudflare
4. Verify: `flyctl ssh console -a bts -C "cd /app && uv run bts data verify-manifest"`

### Rotate Tailscale auth key

1. Create new pre-auth key in Tailscale admin with tag `tag:bts-prod`
2. `flyctl secrets set -a bts TS_AUTHKEY=<new-key>`
3. Restart the machine to force Tailscale to re-auth with the new key
4. Verify: `tailscale status` on your local shows bts-fly is still online
```

- [ ] **Step 2: Commit**

```bash
git add docs/fly-operations.md
git commit -m "docs(fly): add operations runbook for Fly deployments"
```

---

### Task 7: End-to-end smoke test

**Files:** (no new files, just a runbook)

This task validates that the infrastructure all fits together. Must be executed after Plan 02 (R2 sync) and Plan 04 (heartbeat + local tier) are complete and their commits are on main.

- [ ] **Step 1: Prerequisites**

Before running this task, confirm:
- `fly apps create bts` has been run (Phase 0 Step 1)
- R2 bucket `bts-backup-data` exists with parquets uploaded (Phase 0 Steps 3 + 10)
- Fly secrets are set: `BTS_BLUESKY_APP_PASSWORD`, `R2_*`, `TS_AUTHKEY`
- GitHub Secret `FLY_API_TOKEN` is set
- Tailscale ACL has `tag:bts-prod` configured to allow your account to reach it

- [ ] **Step 2: First manual deploy**

```bash
flyctl deploy --remote-only
```

Expected: image builds remotely (several minutes first time), machine is created, starts, Fly reports "Machine is ready".

- [ ] **Step 3: Check logs for bootstrap completion**

```bash
flyctl logs -a bts --since 15m
```

Expected: see lines from the entrypoint script:
- `[entrypoint] Starting BTS Fly entrypoint`
- `[entrypoint] Tailscale joined: 100.x.x.x`
- `[bootstrap] Downloading parquets from R2`
- `[bootstrap] Bootstrap complete`
- `[entrypoint] Dashboard PID: ...`
- `[entrypoint] Starting scheduler daemon`

- [ ] **Step 4: Verify Tailscale connectivity**

```bash
tailscale status | grep bts-fly
curl -sS http://bts:3003/health | python3 -m json.tool
```

Expected: tailscale status shows `bts-fly` as online; curl returns `{"status": "ok", "scheduler_state": "...", ...}`.

- [ ] **Step 5: Verify health check is passing**

```bash
flyctl status -a bts
```

Expected: status shows "passing" for the HTTP health check.

- [ ] **Step 6: Inspect volume contents**

```bash
flyctl ssh console -a bts -C "ls -la /data"
flyctl ssh console -a bts -C "cat /data/.heartbeat"
```

Expected: `/data` has `processed/`, `models/`, `picks/`, `raw/`, `.heartbeat` present. Heartbeat file contains JSON with `state: running` or `state: sleeping`.

- [ ] **Step 7: Commit a tiny code change to verify the auto-deploy works**

Make any trivial code change in `src/`, commit, push:

```bash
git commit -am "test: verify deploy workflow fires on push"
git push origin main
```

Watch the GitHub Actions tab to see the deploy workflow run. Expected: workflow succeeds in several minutes.

- [ ] **Step 8: Commit the smoke test log**

Nothing to commit — this task verifies Plan 05's changes work end-to-end. If any step fails, debug and fix before marking Plan 05 complete.

---

## Completion criteria for Plan 05

- [ ] Docker image builds locally without errors: `docker build -t bts:test .`
- [ ] `fly.toml` is valid: `flyctl config validate`
- [ ] GitHub Actions deploy workflow is syntactically valid and runs on push-to-main
- [ ] Fly app has been created with volume attached
- [ ] Fly machine boots and completes bootstrap within 3-5 minutes
- [ ] Dashboard is reachable via Tailscale at `http://bts:3003`
- [ ] `/health` endpoint returns 200 with scheduler_state "running" or "sleeping"
- [ ] Fly HTTP health check is passing (visible in `flyctl status`)
- [ ] Deploying a trivial change via git push triggers a zero-downtime swap
- [ ] `docs/fly-operations.md` has been reviewed for accuracy

**Next plan:** `06-cutover.md` — Cutover execution runbook. Depends on all previous plans being complete and validated.
