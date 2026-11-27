import shutil
import sys
from pathlib import Path


workspace, scratch, mode = Path.cwd(), Path(sys.argv[1]).resolve(), sys.argv[2]
assert scratch.is_relative_to(Path("/tmp").resolve()) and not scratch.is_relative_to(workspace)
for name in ("scheduler.py", "strategy.py"):
    shutil.copyfile(workspace / "src" / "bts" / name, scratch / "src" / "bts" / name)


def replace(source, original, changed):
    assert source.count(original) == 1, original
    return source.replace(original, changed, 1)


if mode == "dm":
    path = scratch / "src" / "bts" / "scheduler.py"
    source = path.read_text()
    end_of_warning = '"including pending makeup games, and to update the stored boundary for a later run."\n    )'
    send = '''
    from bts.health.alert import Alert, dispatch_dm_for_health_alerts

    recipient = _config.get("bluesky", {}).get("dm_recipient")
    if recipient:
        dispatch_dm_for_health_alerts(
            [], recipient, warn_attention=[Alert(
                "WARN", "season_boundary", "Season boundary behind schedule",
                incident_key="season_boundary:behind_schedule",
            )],
            status_path=_picks_dir.parent / "health_state" / "season_boundary" / f"{slate_date}.json",
            now_et_date=slate_day, require_message_id=True,
        )
'''
    path.write_text(replace(source, end_of_warning, end_of_warning + "\n" + send))
elif mode == "sync":
    path = scratch / "src" / "bts" / "scheduler.py"
    source = path.read_text()
    path.write_text(replace(source, '    _queue_season_boundary_line(\n        f"SEASON',
                            '    _write_season_boundary_line(\n        f"SEASON'))
elif mode == "floor":
    path = scratch / "src" / "bts" / "strategy.py"
    source = path.read_text()
    source = replace(source, "_regime(None, streak, date, min_days=1)", "_regime(None, streak, date)")
    path.write_text(replace(source, "        min_days=1,\n", ""))
elif mode != "restore":
    raise ValueError(mode)
print(mode, scratch / "src" / "bts")
