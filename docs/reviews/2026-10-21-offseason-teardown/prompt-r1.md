# Sign-off review: deleting the Hetzner Cloud server "bts-mlb" for the MLB offseason

You are reviewing an operations plan before an irreversible step (deleting a cloud server, keeping a snapshot). Assume the plan has at least one defect or gap. Find what would be lost, what would fail when the box is recreated in spring (March 2027), and what must be checked or done BEFORE the delete. Do not restate the plan. No praise, no padding.

## Situation (facts observed 2026-10-21, Hetzner Cloud API + ssh)
- Server `bts-mlb`, id 126463369, type CPX42 (8 vCPU / 16 GB / 320 GB disk, ~30 GB used), datacenter hel1-dc2 (Helsinki), Debian 12, user `bts`, uptime 27 weeks. Public IPv4 204.168.243.169 = Primary IP 126036156; IPv6 2a01:4f9:c014:884d::/64 = Primary IP 126036157. A second server in the same Hetzner project (`miniflux`, id 134199377, Ashburn) must NOT be touched.
- Purpose: "Beat the Streak" (MLB) pick pipeline. Season is over; next games late March 2027. It ran: a `bts-scheduler` user service (daily wakeup, logs "No games scheduled"), a `bts-dashboard` user service (HTTP on the Tailscale IP :3003), two systemd user timers (`bts-live-forward-capture.timer` 05:00 ET, `bts-live-forward-resolve.timer` 07:00 ET), and a 17-line crontab (01:00 grading, 02:00/07:40 reconcile, Bluesky posting 01:10/02:10/10:30/13:30, 23:30 skip-policy-shadow-update, 07:45 park-drag-refresh, a 03:00 job, every-5-min lineup-time collection + two healthchecks.io heartbeat pings, every-30-min capture-static, backups `ops` every 3 h, `archive` daily 04:50, restic prune Sundays).
- Backups: a project CLI `bts backup` wraps restic → repository on a Hetzner Storage Box (separate billing, not on this server). Sets: `ops` and `archive`. `archive` paths: data/leaderboard, data/hetzner_results, data/external. (Set definitions from the repo are appended below.) `bts backup status` after the final runs: ops ok=True finished 2026-10-21T05:38:31Z snapshot 978d5e19…; archive ok=True finished 05:38:42Z snapshot db083dd5…. `bts backup restore-drill --target <tmp>` → {"ok": true, "problems": [], "snapshot": "978d5e19…"}, 936 files restored.
- The repo on the box: /home/bts/projects/bts at commit 619abda (also on GitHub). `data/` is 12 GB total (subdirs: external health_state hetzner_results leaderboard lineup_posting_times models models_det picks picks_det_DEPRECATED processed raw raw_c1). Sibling experiment checkouts: bts-c1 (1.3G), bts-live-forward (865M), bts-w12, bts-w23, bts-gen8522412, bts-shadow-backfill. Home dir also has: ~/.ssh (64K), ~/.config/systemd/user (64K), ~/logs (6.9M), ~/bin, ~/scripts, crontab backups, `.env` with these variable NAMES: BTS_BLUESKY_APP_PASSWORD HEALTHCHECKS_PING_URL UV_CACHE_DIR BTS_SCHEDULER_HEARTBEAT_PING_URL BTS_SECRET_VULTR_API_TOKEN BTS_SECRET_HETZNER_CLOUD_TOKEN BTS_LGBM_DETERMINISTIC RESTIC_PASSWORD. Python env is a uv venv (.venv) rebuilt by `uv sync --locked` on deploy.
- Tailscale: node `bts-hetzner` 100.100.43.24, key expiry disabled. `loginctl` Linger=yes for `bts`. Enabled user units: bts-dashboard.service, bts-scheduler.service, the two live-forward timers (+ gpg/dirmngr sockets).
- Deploys: GitHub Actions workflow on push to branch `deploy`: runs the test suite, then `appleboy/ssh-action` SSHes as **root** using a GitHub secret key `HETZNER_SSH_KEY` to the public IPv4, `git pull`s, `sudo -Hu bts uv sync --locked --extra model`, restarts the two services, canary check with auto-rollback. The `bts` user's own authorized key is the operator's Mac key.
- healthchecks.io: the two checks this box pinged (`bts-scheduler-heartbeat`, `bts-hetzner-heartbeat`) were PAUSED by the owner in the UI before shutdown.
- DNS (Cloudflare, all zones): no A/AAAA record points at either IP. Nothing else on the owner's other machines depends on this box (checked memory/notes); a Mac ssh alias `bts-hetzner` resolves via Tailscale, not the public IP.
- Hetzner billing facts (docs): servers are billed until deleted regardless of power state; CPX42 hel1 = €69.49/mo excl. IPv4; snapshots billed per compressed GB (~€0.014/GB/mo); unassigned Primary IPs are billed until deleted.
- Hetzner API token: the one from the box's `.env` (used by an audit-fleet script to create/delete CPX51 servers) was copied into the operator's Mac Keychain so the delete and the spring recreate can be driven from the Mac.

## Steps already executed (in this order, 01:37–01:41 ET)
1. Deploy of 619abda verified on the box.
2. `systemctl --user stop` of bts-scheduler, bts-dashboard, and both live-forward timers (NOT disabled — intent: a restored box starts them via linger).
3. Final `bts backup run --set ops` and `--set archive`; status ok for both; restore drill ok (above).
4. Primary IP 126036156 (IPv4) set `auto_delete=false`. IPv6 Primary IP 126036157 left `auto_delete=true`.
5. `hcloud server shutdown 126463369` (ACPI); status `off` after 15 s.
6. `hcloud server create-image --type snapshot --description "bts-mlb offseason 2026-10-21 CPX42 hel1 deploy 619abda" 126463369` → image 446156433, status available, disk_size 320 GB, image_size 8.91 GB. An older snapshot 407062225 (2026-07-10, 4.91 GB) also exists.
7. Server is currently OFF, not deleted.

## Remaining steps as planned
8. `hcloud server delete 126463369`.
9. Memory notes + a "DOWN for offseason" banner in the repo's CLAUDE.md.

## Spring restore recipe as currently written
Create a server from snapshot 446156433, type CPX42, location hel1, assigning Primary IP 126036156 at creation, ssh key = whatever is in the snapshot's authorized_keys; boot; `ssh bts-hetzner` over Tailscale (node identity expected to persist because /var/lib/tailscale is in the snapshot and key expiry is disabled); check `systemctl --user status bts-scheduler bts-dashboard` (expected to auto-start via linger); un-pause the two healthchecks (they self-resume on the next ping); `git push origin main:deploy` to land winter changes. If the IPv4 was not kept, update the workflow's SSH host.

## Questions to answer
1. What state is covered by neither the restic sets nor the Hetzner snapshot, or only by the snapshot in a way that is awkward to use?
2. Hetzner-specific pitfalls of recreating from a 320 GB-disk snapshot: server-type constraints; Primary IP assignment at create; cloud-init re-running on first boot of a snapshot-based server (hostname, ssh host keys regenerated, root password reset mail, user-data); Tailscale node reuse; linger; the root authorized_keys used by the GitHub workflow; swap/fstab; clock; months of pending unattended-upgrades.
3. Does deleting the IPv6 Primary IP matter for anything typical here?
4. Delete the old snapshot 407062225 or keep it — risk each way?
5. After the delete, what is still billed?
6. What could still alarm or break elsewhere because the box is gone (beyond the paused checks and the deploy workflow)?
7. Is the executed order defective anywhere (e.g., stop-not-disable, shutdown-before-snapshot)?

## Deliverable
Write to codex-out.md: (a) a ranked list of concrete findings, each with severity (BLOCKER / SHOULD-FIX / NOTE), the exact check or command that would close it, and what evidence closes it; (b) one final line: `VERDICT: SIGN OFF — delete` or `VERDICT: DO NOT DELETE YET — <reason>`. Be specific about disagreements; if the plan is fine as written, say so briefly and stop.

## Appendix A — backup set definitions from the repo (grep)

## Appendix B — deploy workflow SSH step (sanitized)
    77:        uses: appleboy/ssh-action@0ff4204d59e8e51228ff73bce53f80d53301dee2  # v1.2.5
    78-        env:
    79-          # Deploy the TRIGGERING commit, not the mutable ref (round-3 F2):
    80-          # resetting to origin/deploy at execution time let an in-flight run
    81-          # ship a LATER queued commit whose own test/environment gate had
    82-          # not run yet. github.sha is exactly what this run's gate tested.
    83-          TARGET_SHA: ${{ github.sha }}
    84-        with:
    86-          username: root
    88-          envs: TARGET_SHA
    89-          script: |
    90-            set -e
    91-            cd /home/bts/projects/bts

## Appendix C — crontab (URLs redacted) as of 2026-10-20
    (see Situation; full lines contain ping URLs and were not copied)
