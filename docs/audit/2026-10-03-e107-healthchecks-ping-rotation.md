# E107: Healthchecks ping URL exposed in the public repo — what to do (2026-10-03)

**For Eric, with his Healthchecks login.** Nothing here blocks other work. No ping URL is copied into this file, the register or any log.

## What was found (checked 2026-10-03, by hash comparison only)
- **One real ping URL was ever committed.** `d5c466c` put it into `scripts/cron-setup-hetzner.sh`, and `681eb8c` (2026-06-06) took it out. It is still readable in the public git history.
- **The box no longer uses that URL.** The every-5-minute liveness ping in the box's crontab and `HEALTHCHECKS_PING_URL` in `~/projects/bts/.env` point at a **different** check. Their check IDs hash differently from the committed one, and so does the 9/14 crontab backup.
- **The second URL was never committed.** That is the scheduler-heartbeat URL (`BTS_SCHEDULER_HEARTBEAT_PING_URL` in `.env`, used by `scripts/check_heartbeat.py` for the stale-heartbeat `/fail` ping). Two other matches in history (`d4dae2c`, `8542035`) are placeholders such as `XXXX…` and `<uuid>`.
- **The risk is the old check, if it still exists.** Anyone who reads the history can ping it: they could keep it falsely "up" or trigger fail alerts. The live monitoring does not depend on it.
- **Healthchecks cannot change a check's ping UUID.** The UUID is immutable (https://healthchecks.io/docs/http_api/). So "rotating" a URL means creating a new check and retiring the old one.

## Step 1 (recommended): retire the leaked check, with no box change
1. **See the leaked check ID in your own terminal only:**
   ```bash
   git -C ~/projects/bts show d5c466c -- scripts/cron-setup-hetzner.sh | grep -o 'hc-ping.com/[A-Za-z0-9-]*' | sort -u
   ```
2. **Find it in Healthchecks.** Log in at healthchecks.io and open the project. For each check, compare its ping URL with that ID.
3. **If a check with that ID exists,** delete it: open the check, then Settings, then Delete. The box does not use it, so nothing else changes.
   - If that check unexpectedly shows recent pings, stop and tell Claude before deleting. Something besides the box would be pinging it.
4. **If no check has that ID,** it was already deleted, and E107 is closed.

## Step 2 (optional): replace the current liveness check too
Do this only if you want belt and braces, for example if the current URL may have been shared somewhere else. This step changes the box.

1. **Create the new check.** In Healthchecks, add a new check with the same schedule as the current box liveness check (period 5 minutes; copy its grace time) and the same notification integrations. Copy its ping URL.
2. **Back up the crontab and `.env` on the box** (as user `bts`):
   ```bash
   ssh bts-hetzner
   cd ~/projects/bts
   crontab -l > ~/crontab.bak-$(date +%Y%m%d)-e107
   cp .env ~/.env.bak-$(date +%Y%m%d)-e107 && chmod 600 ~/.env.bak-*-e107
   ```
3. **Update `.env`.** Edit `~/projects/bts/.env` and set `HEALTHCHECKS_PING_URL=` to the new URL. Keep the file mode 600.
4. **Update the one crontab line BY HAND.** Run `crontab -e`, find the `*/5 * * * * curl -fsS --max-time 5 https://hc-ping.com/…` line, which ends in `# BTS-HETZNER`, and replace only the URL.
   - **Do NOT run `bash scripts/cron-setup-hetzner.sh install` during the offseason.** It deletes every `# BTS-HETZNER` line and re-adds the full set, which re-enables the `check-pick-entered` line commented out on 9/14 (the season-over silence). The full reinstall belongs to the 2027 season-start checklist, when that line is wanted back.
5. **Verify.** Within 5 minutes the new check shows its first ping ("up") in Healthchecks. On the box, `crontab -l | grep -c BTS-HETZNER` should show the same count as in the backup.
6. **Retire the old check:** pause or delete the previous liveness check, but only after the new one has pinged.
7. **The heartbeat URL** (`BTS_SCHEDULER_HEARTBEAT_PING_URL`) needs no rotation; it was never committed. If you rotate it anyway, it is read from `.env` at each cron run (the crontab line sources `.env`). So it takes only a new check plus the `.env` edit, with no crontab change.

## Afterwards
Tell Claude which steps you did. The register's E107 record and the 2027 checklist will record the outcome.
