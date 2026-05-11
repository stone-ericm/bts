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
