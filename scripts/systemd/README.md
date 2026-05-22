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
- refuses to export if processed PA outcomes already contain rows for the date;
- refuses partial artifact directories;
- verifies an existing manifest instead of exporting again when its locked
  decision payload still matches the current production pick. Full-file SHA
  drift from post-game `result` / `slot_results` updates is recorded but is not
  treated as stale decision drift;
- auto-recaptures snapshot drift only through the timer's
  `--auto-recapture-on-snapshot-drift` flag, and only before the pick resolves
  and before processed PA outcomes exist for the date;
- writes `capture_status.json` next to successful or failed artifact attempts.

The scheduler also queues this one-shot unit with `systemctl --user start
--no-block bts-live-forward-capture.service` immediately after a production pick
locks, so the at-lock artifact is refreshed without waiting for the next
15-minute timer tick.

Verify it is active:

    systemctl --user list-timers bts-live-forward-capture
    journalctl --user -u bts-live-forward-capture -n 100 --no-pager

Run one manual poll:

    systemctl --user start bts-live-forward-capture.service

## Shadow prediction

The scheduler can keep the historical inline shadow behavior, or it can queue a
one-shot shadow unit after the production pick locks. The out-of-process mode is
opt-in so code deploy and unit installation can be staged safely.

Install as a user unit on Hetzner after the production checkout contains
`scripts/shadow_predict_once.py`:

    mkdir -p ~/.config/systemd/user
    cp scripts/systemd/bts-shadow-prediction.service ~/.config/systemd/user/
    systemctl --user daemon-reload

Before enabling the unit, reconcile it against the live scheduler environment:

    systemctl --user cat bts-scheduler
    systemctl --user show bts-scheduler -p Environment
    systemctl --user cat bts-shadow-prediction.service

Use the live scheduler service as the source of truth for project root, config
path, environment file, and model-affecting BTS settings such as
`BTS_LGBM_DETERMINISTIC`, `BTS_LGBM_RANDOM_STATE`, and thread-count variables;
older checked-in scheduler unit files may be stale. The checked-in shadow unit
sources `/home/bts/projects/bts/.env` but does not hardcode those values. Update
the unit first if the live scheduler uses a different environment source.

Then enable the scheduler to use it by setting:

    shadow_model = true
    shadow_model_unit = "bts-shadow-prediction.service"

in `/home/bts/.bts-orchestrator.toml`, and restart only the scheduler service.

This mode must preserve the same orchestrator config path and BTS environment as
the scheduler path, but moves the expensive shadow prediction into its own
systemd unit. Missing shadow artifacts still alert through health checks; when
the unit exposes OOM evidence in `systemctl --user show`, the WARN is promoted
by the alert policy.

Before leaving `shadow_model_unit` enabled, compare one sample locked date
against the inline path and confirm the shadow pick and provenance match.

Verify:

    systemctl --user status --no-pager -l bts-shadow-prediction.service
    journalctl --user -u bts-shadow-prediction -n 100 --no-pager

Run one manual shadow attempt after a pick is locked:

    systemctl --user start bts-shadow-prediction.service

This unit does not serialize memory-heavy analytics jobs by itself. Keep the
live-forward capture/shadow collision fix and swap/cgroup sizing as separate
approved ops work.

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
