"""`bts backup` command group (audit F5) — restic operational-state backup.

Thin click wrapper over bts.data.backup; all restic interaction and status
bookkeeping lives there. Exit codes are cron-friendly: nonzero on any
failed backup/drill so `&&` chains and log-grep both work.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import click


@click.group()
def backup():
    """Encrypted, versioned restic backups of operational state to R2."""
    pass


@backup.command("run")
@click.option("--set", "set_name", type=click.Choice(["ops", "archive"]),
              required=True, help="Backup set (ops = picks+health, 3h cadence; "
                                  "archive = leaderboard/results/external, daily)")
@click.option("--repo-root", default=".", type=click.Path(exists=True),
              help="Repo root containing data/ (default: cwd)")
def backup_run(set_name: str, repo_root: str):
    """Run one backup set + retention forget; writes backup_status.json."""
    from bts.data import backup as backup_mod

    entry = backup_mod.run_backup(set_name, Path(repo_root), env=dict(os.environ))
    click.echo(json.dumps(entry, indent=2))
    if not entry.get("ok"):
        raise SystemExit(1)


@backup.command("status")
@click.option("--repo-root", default=".", type=click.Path(exists=True))
def backup_status(repo_root: str):
    """Print the per-set backup status file."""
    from bts.data import backup as backup_mod

    status = backup_mod.read_status(Path(repo_root))
    if not status:
        click.echo("no backup status recorded (backup_status.json absent/empty)")
        return
    click.echo(json.dumps(status, indent=2))


@backup.command("prune")
@click.option("--repo-root", default=".", type=click.Path(exists=True))
def backup_prune(repo_root: str):
    """Reclaim space from forgotten snapshots (weekly)."""
    from bts.data import backup as backup_mod

    result = backup_mod.run_prune(env=dict(os.environ))
    click.echo(json.dumps(result, indent=2))
    if not result.get("ok"):
        raise SystemExit(1)


@backup.command("restore-drill")
@click.option("--repo-root", default=".", type=click.Path(exists=True))
@click.option("--target", required=True, type=click.Path(),
              help="Empty directory to restore the latest ops snapshot into")
def backup_restore_drill(repo_root: str, target: str):
    """Restore latest ops snapshot to --target and verify saver/ledger/decisions."""
    from bts.data import backup as backup_mod

    result = backup_mod.restore_drill(
        repo_root=Path(repo_root), target=Path(target), env=dict(os.environ),
    )
    click.echo(json.dumps(result, indent=2))
    if not result.get("ok"):
        raise SystemExit(1)
