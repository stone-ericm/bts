"""`bts watchdog run JOB`: run one registered watchdog job (registration R3; the cron line comes with the
intent-aware install).

Jobs are registered in `JOBS` by the W pieces (W1 onwards); W0 registers none. Notifications go through
`bts.dm.send_dm` (it returns the message id), unless `--no-send` (results are still written and alerts stay
pending).
"""
from __future__ import annotations

from pathlib import Path

import click

from bts.watchdog.clock import SystemClock
from bts.watchdog.notify import Notifier
from bts.watchdog.root import JobBusy, OwnedRoot
from bts.watchdog.runner import run_job

JOBS: dict[str, list] = {}


@click.group()
def watchdog():
    """The C1 rank-2 watchdog: read-only checks that alert and never repair."""


@watchdog.command("run")
@click.argument("job")
@click.option("--data-dir", default="data", type=click.Path(path_type=Path), help="The data directory (root: <data-dir>/watchdog)")
@click.option("--dm-recipient", default=None, help="Bluesky handle for alerts (required unless --no-send)")
@click.option("--no-send", is_flag=True, help="Write results and queue alerts without sending (they stay pending)")
def run(job, data_dir, dm_recipient, no_send):
    """Run one registered job. Exit 1 on any infrastructure failure (results or notification); never report that
    as completion."""
    import sys
    if job not in JOBS:
        raise click.ClickException(f"unknown watchdog job {job!r} (registered: {sorted(JOBS) or 'none'})")
    if not no_send and not dm_recipient:
        raise click.ClickException("--dm-recipient is required unless --no-send")
    root = OwnedRoot.under(data_dir)
    clock = SystemClock()
    send = None
    if not no_send:
        from bts.dm import send_dm
        send = send_dm
    notifier = Notifier(root, recipient=None if no_send else dm_recipient, send=send, clock=clock)
    try:
        out = run_job(job, JOBS[job], root=root, clock=clock, notifier=notifier)
    except JobBusy as exc:
        click.echo(f"watchdog: {exc}; skipping", err=True)
        return
    counts = {}
    for r in out.results:
        counts[r.status.value] = counts.get(r.status.value, 0) + 1
    click.echo(f"watchdog {job}: {counts} -> {out.path}")
    if out.persist_error:
        click.echo(f"watchdog {job}: RESULTS NOT PERSISTED: {out.persist_error}", err=True)
    if out.notify_error:
        click.echo(f"watchdog {job}: NOTIFICATION FAILURE: {out.notify_error}", err=True)
    if not out.ok:
        sys.exit(1)
