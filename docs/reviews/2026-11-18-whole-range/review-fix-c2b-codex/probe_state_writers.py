"""Copy actual neighboring state-writer output into a fresh C2 slate."""

from contextlib import redirect_stderr
from datetime import date
from io import StringIO
import json
from pathlib import Path
import subprocess

from probe_receipt_matrix import DAY, NOW, SHAPE, check, stage_pick
from probe_support import deadline, fixtures, world_at

from bts import cli, dm, heartbeat, progress, reconcile_receipt, sd_notify
from bts.daily_decision import write_decision
from bts.data import backup
from bts.entry_receipt import receipts_root
from bts.health import alert, attention, memory_growth, slate_auc
from scripts import check_heartbeat


REVIEW = Path(__file__).resolve().parent


def build_state(root, patch, opening):
    import sys

    sys.path.insert(0, str(REVIEW / "replay" / "codex"))
    from probe_operator import watchdog

    day = date.fromisoformat(DAY)
    with world_at(root, patch, opening, NOW) as (_, directory, _, _, scoped):
        health = directory.parent / "health_state"
        marker = health / "pick_entry_check.json"
        fetch = health / "contest_streak_fetch_status.json"
        churn = health / "scheduler_churn.json"
        warn = health / "warn_attention_state.json"
        memory = health / "memory_growth_history.jsonl"
        durations = health / "cascade_stage_durations.jsonl"

        daily = stage_pick(DAY, directory)
        decision = write_decision(DAY, directory, action="single", source="mdp", scoreable=True,
                                  delivery_status="locked_unconfirmed", primary={"batter_id": daily.pick.batter_id})
        assert decision is not None
        _, notices = watchdog(opening, directory, NOW, daily, {DAY: (daily.pick.batter_id,)}, 0)
        assert not notices
        assert json.loads(marker.read_bytes())["status"] == "confirmed"
        entry_path, = (receipts_root(directory) / DAY).glob("*.json")
        dm.send_dm.return_value = "fetch-warning-id"
        assert cli._contest_fetch_alert(fetch, "offline.test", "synthetic identity failure")
        dm.send_dm.assert_called_once()
        check_heartbeat.save_churn_samples(churn, "fixture.service", [{"ts": NOW.isoformat(), "n": 1}])
        attention.update_warn_streaks([alert.Alert("WARN", "result_resolution", "fixture")], state_path=warn, today=day)
        assert memory_growth._append_history(memory, day, 2048)
        memory_one_row = memory.read_bytes()
        assert memory_growth._append_history(memory, day, 2049)

        scoped.setattr(sd_notify, "notify_watchdog", lambda: None)
        with heartbeat.heartbeat_watchdog(root / "watchdog-heartbeat", date=DAY, durations_path=durations):
            progress.mark("review")

        (directory / "slates").mkdir()
        slate_auc.check(directory, root / "data", today=day)
        auc = slate_auc._status_path(directory)
        reconcile = reconcile_receipt.ReconcileReceipt(clock=lambda: NOW, lookback_days=3)
        reconcile_path = reconcile.publish(directory)
        assert reconcile_path is not None

        backup_root = root / "backup-root"
        (backup_root / "data" / "picks").mkdir(parents=True)
        (backup_root / "data" / "health_state").mkdir()

        def restic(args, **kwargs):
            output = '{"message_type":"summary","snapshot_id":"fixture"}\n' if args[1] == "backup" else ""
            return subprocess.CompletedProcess(args, returncode=0, stdout=output, stderr="")

        backup_status = backup.run_backup("ops", backup_root, env={"RESTIC_PASSWORD": "offline-fixture",
                                           "BTS_RESTIC_REPOSITORY": str(root / "restic-repo")},
                                          runner=restic, now_fn=lambda: NOW)
        assert backup_status["ok"]
        backup_path = backup_root / "data" / "health_state" / backup.STATUS_FILENAME

        legacy = json.loads((REVIEW / "replay" / "legacy_dispatch-base.json").read_text())
        legacy_payloads = {
            f"legacy {case} {n} {name}": document.encode()
            for case, captured in legacy.items()
            for n, (name, _, document) in enumerate(captured["dispatch_checkpoints"])
            if document is not None
        }
        local_payloads = {
            path.name if path not in {entry_path, reconcile_path} else str(path.parent.parent.name): path.read_bytes()
            for path in (marker, fetch, churn, warn, memory, durations, auc, entry_path, reconcile_path,
                         backup_path, backup_path.with_suffix(".json.lock"))
        }
        return local_payloads | legacy_payloads | {"one memory-growth JSONL row": memory_one_row}


def shape(payload):
    try:
        value = json.loads(payload)
    except ValueError:
        return "empty or JSONL rather than an object"
    assert isinstance(value, dict)
    assert set(value) != set(SHAPE)
    return ", ".join(sorted(value))


def main():
    with deadline(45), fixtures("state-writers") as (root, patch, opening):
        with redirect_stderr(StringIO()):
            payloads = build_state(root / "writers", patch, opening)
        for index, (label, original) in enumerate(payloads.items()):
            print(f"{label} top level: {shape(original)}", flush=True)
            for route in ("direct", "classification"):
                check(root / f"copy-{index}-{route}", patch, opening, route, label, original, True)
    print(f"PASS: {len(payloads)} actual outputs copied into both routes, none suppressed")


if __name__ == "__main__":
    main()
