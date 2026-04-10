# Fly Operations Runbook for BTS

Go-to reference for day-to-day Fly operations on the BTS production deployment.

## Quick status

    flyctl status -a bts          # machine status
    flyctl logs -a bts            # recent logs (streaming)
    flyctl ssh console -a bts     # SSH into container
    flyctl machines list -a bts   # machine IDs

## Deployment

Normal deploys happen automatically via GitHub Actions on push to main.

### Manual deploy

    flyctl deploy --remote-only --strategy immediate

Expect ~30-60s of missed heartbeats during swap; Fly health checks
tolerate 2 consecutive failures before restart.

### Rollback

    flyctl releases -a bts                # list recent releases
    flyctl releases rollback <version>    # rollback to a specific one

## Volume management

The data volume at /data holds parquets, model caches, pick history,
and scheduler state. Regional (iad).

    flyctl volumes list -a bts
    flyctl ssh console -a bts -C "ls -la /data"
    flyctl ssh console -a bts -C "du -sh /data/*"

### Restore from snapshot

    flyctl volumes snapshots list <vol-id>
    flyctl volumes snapshots create <vol-id>   # force snapshot now

## Common issues

### No pick was made today

1. Check scheduler: `flyctl logs -a bts --since 2h | grep -i scheduler`
2. Check heartbeat: `flyctl ssh console -a bts -C "cat /data/.heartbeat"`
3. Check /health: `curl http://bts:3003/health` (via Tailscale)
4. Restart: `flyctl machines restart <id> -a bts`

### Deploy failed

Check GitHub Actions log. Common causes:
- FLY_API_TOKEN expired -> `fly tokens create deploy -a bts`, update GitHub Secret
- Docker build failure -> check Dockerfile step
- Image too large -> check .dockerignore

### Dashboard unreachable via Tailscale

1. `flyctl ssh console -a bts -C "tailscale status"`
2. `tailscale status | grep bts-fly`
3. Restart: `flyctl machines restart <id> -a bts`

### One-off command

    flyctl ssh console -a bts -C "cd /app && uv run bts data verify-manifest"

## Secrets rotation

### Bluesky app password

1. Create new password in Bluesky admin
2. `flyctl secrets set -a bts BTS_BLUESKY_APP_PASSWORD=<new>`
3. Revoke old password
4. Verify: `flyctl logs -a bts --since 10m | grep bluesky`

### R2 API token

1. Create new token in Cloudflare
2. `flyctl secrets set -a bts R2_ACCESS_KEY_ID=<new> R2_SECRET_ACCESS_KEY=<new>`
3. Revoke old token
4. Verify: `flyctl ssh console -a bts -C "cd /app && uv run bts data verify-manifest"`

### Tailscale auth key

1. Create new pre-auth key with tag:bts-prod
2. `flyctl secrets set -a bts TS_AUTHKEY=<new>`
3. Restart machine
4. Verify: `tailscale status | grep bts-fly`
