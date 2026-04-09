# Systemd units for Pi5 deployment

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
