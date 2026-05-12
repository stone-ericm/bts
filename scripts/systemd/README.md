# Systemd units for Pi5 deployment

## Live-forward candidate capture

Install as user units on Hetzner after the production checkout contains
`scripts/live_forward_capture_once.py` and the frozen live-forward checkout is
on the parity-guard branch:

    mkdir -p ~/.config/systemd/user
    cp scripts/systemd/bts-live-forward-capture.service ~/.config/systemd/user/
    cp scripts/systemd/bts-live-forward-capture.timer ~/.config/systemd/user/
    systemctl --user daemon-reload
    systemctl --user enable --now bts-live-forward-capture.timer

The timer runs every 15 minutes from 05:00 through 22:45 local server time.
This is safe because the runner is idempotent:

- exits successfully while `data/picks/YYYY-MM-DD.json` is absent;
- refuses to export if the pick file already has a result;
- refuses partial artifact directories;
- verifies an existing manifest instead of exporting again;
- writes `capture_status.json` next to successful or failed artifact attempts.

Verify it is active:

    systemctl --user list-timers bts-live-forward-capture
    journalctl --user -u bts-live-forward-capture -n 100 --no-pager

Run one manual poll:

    systemctl --user start bts-live-forward-capture.service

## Live-forward candidate resolution

Install as user units on Hetzner after the production checkout contains
`scripts/live_forward_resolve_once.py`:

    mkdir -p ~/.config/systemd/user
    cp scripts/systemd/bts-live-forward-resolve.service ~/.config/systemd/user/
    cp scripts/systemd/bts-live-forward-resolve.timer ~/.config/systemd/user/
    systemctl --user daemon-reload
    systemctl --user enable --now bts-live-forward-resolve.timer

Treat timer installation as a separate approved ops step; merging the runner
does not enable the resolver on production.

The timer runs at 07:00, 12:00, 18:00, and 22:00 server-local time. This is
safe because the runner is idempotent:

- scans preoutcome artifacts under
  `data/validation/decision_weighted_lgbm_v0_live_forward/`;
- exits successfully when no preoutcome artifacts exist yet;
- treats missing processed PA outcomes as `pending_outcomes`, not as misses;
- treats known postponed/cancelled source-date rows as terminal voids with
  null `actual_hit`/`n_pas` and `outcome_status=void_*`;
- verifies an existing resolved manifest instead of resolving again;
- refuses partial resolved directories without a manifest;
- writes per-date status JSON under
  `data/validation/decision_weighted_lgbm_v0_live_forward_resolved_status/`.

Verify it is active:

    systemctl --user list-timers bts-live-forward-resolve
    journalctl --user -u bts-live-forward-resolve -n 100 --no-pager

Run one manual resolver pass:

    systemctl --user start bts-live-forward-resolve.service

## Lineup time collection

Install as user units on Pi5:

    mkdir -p ~/.config/systemd/user
    cp scripts/systemd/bts-lineup-collect.service ~/.config/systemd/user/
    cp scripts/systemd/bts-lineup-collect.timer ~/.config/systemd/user/
    systemctl --user daemon-reload
    systemctl --user enable --now bts-lineup-collect.timer

Verify it's running:

    systemctl --user list-timers bts-lineup-collect
    journalctl --user -u bts-lineup-collect -f

Collected data accumulates under `~/projects/bts/data/lineup_posting_times/`.
To pull the data back to Mac for analysis:

    rsync -az pi5:~/projects/bts/data/lineup_posting_times/ \
        ~/projects/bts/data/lineup_posting_times/
