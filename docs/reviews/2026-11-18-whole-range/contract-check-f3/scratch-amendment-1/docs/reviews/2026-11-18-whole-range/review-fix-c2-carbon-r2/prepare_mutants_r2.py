"""Prepare scratch mutant copies under .codex-review/wr-c2-r2/mutants/ without modifying any tracked file."""

from __future__ import annotations

from pathlib import Path
import shutil

ROOT = Path("/Users/eric/projects/bts-work/wr-c2-r2-carbon")
HEAD_SRC = ROOT / "src"
MUTANTS_DIR = ROOT / ".codex-review" / "wr-c2-r2" / "mutants"


def apply_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    assert count == 1, f"{path}: expected 1 match for {old!r}, got {count}"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


MUTANTS = {
    "m01_drop_schema_check": [
        (
            "bts/health/alert.py",
            '        if (prior and prior.get("schema_version") == HEALTH_DM_STATUS_SCHEMA_VERSION\n'
            '                and prior.get("status") == "sent" and prior.get("sent_day") == day',
            '        if (prior and prior.get("status") == "sent" and prior.get("sent_day") == day',
        ),
    ],
    "m02_early_classification_before_decision": [
        (
            "bts/scheduler.py",
            '            _write_classification_decision(\n'
            '                picks_dir, date,\n'
            '                action=("double" if ld.double_down else "single"),\n'
            '                delivered=delivered,\n'
            '                attempted=attempted,\n'
            '                primary=_row_from_daily(ld.pick),\n'
            '                double_down=_row_from_daily(ld.double_down),\n'
            '                state=state, daily=ld,\n'
            '            )\n'
            '            if attempted and not delivered:\n'
            '                _alert_delivery_outcome_unknown(config, picks_dir, date)',
            '            if attempted and not delivered:\n'
            '                _alert_delivery_outcome_unknown(config, picks_dir, date)\n'
            '            _write_classification_decision(\n'
            '                picks_dir, date,\n'
            '                action=("double" if ld.double_down else "single"),\n'
            '                delivered=delivered,\n'
            '                attempted=attempted,\n'
            '                primary=_row_from_daily(ld.pick),\n'
            '                double_down=_row_from_daily(ld.double_down),\n'
            '                state=state, daily=ld,\n'
            '            )',
        ),
    ],
    "m03_follow_receipt_dir_symlink": [
        (
            "bts/health/alert.py",
            "directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)",
            "directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)",
        ),
    ],
    "m04_drop_post_send_entry_recheck": [
        (
            "bts/health/alert.py",
            "        _check_receipt_entry(directory_fd, name)\n"
            "        os.replace(temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)",
            "        os.replace(temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)",
        ),
    ],
    "m05_drop_resurface_guard": [
        (
            "bts/health/alert.py",
            "    prior = None\n"
            "    if not require_message_id:\n"
            "        prior = _read_health_dm_delivery_status(status_path)\n"
            "        resurface = _resurface_note(prior)  # H6: resurface a prior failed DM\n"
            "        if resurface:\n"
            '            body = resurface + ("\\n\\n" + body if body else "")',
            "    prior = _read_health_dm_delivery_status(status_path)\n"
            "    resurface = _resurface_note(prior)\n"
            "    if resurface:\n"
            '        body = resurface + ("\\n\\n" + body if body else "")',
        ),
    ],
    "m06_accept_none_id": [
        (
            "bts/health/alert.py",
            "def _usable_message_id(value: object) -> bool:\n    return isinstance(value, str) and bool(value.strip())",
            "def _usable_message_id(value: object) -> bool:\n    return value is None or (isinstance(value, str) and bool(value.strip()))",
        ),
    ],
    "m07_wall_date_key": [
        (
            "bts/scheduler.py",
            "            now_et_date=slate,",
            "            now_et_date=_now_et().date(),",
        ),
    ],
    "m08_wall_date_filename": [
        (
            "bts/scheduler.py",
            '            status_path=picks_dir.parent / "health_state" / "delivery_outcome_unknown" / f"{slate}.json",',
            '            status_path=picks_dir.parent / "health_state" / "delivery_outcome_unknown" / f"{_now_et().date()}.json",',
        ),
    ],
    "m09_shared_status_file": [
        (
            "bts/scheduler.py",
            '            status_path=picks_dir.parent / "health_state" / "delivery_outcome_unknown" / f"{slate}.json",',
            '            status_path=picks_dir.parent / "health_state" / "health_dm_delivery_status.json",',
        ),
    ],
    "m10_drop_classification_call": [
        (
            "bts/scheduler.py",
            "            if attempted and not delivered:\n                _alert_delivery_outcome_unknown(config, picks_dir, date)\n",
            "",
        ),
    ],
    "m11_follow_file_symlinks": [
        (
            "bts/health/alert.py",
            "        info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)",
            "        info = os.stat(name, dir_fd=directory_fd, follow_symlinks=True)",
        ),
        (
            "bts/health/alert.py",
            "        fd = os.open(name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=directory_fd)",
            "        fd = os.open(name, os.O_RDONLY | os.O_NONBLOCK, dir_fd=directory_fd)",
        ),
    ],
    "m12_allow_hardlinks": [
        (
            "bts/health/alert.py",
            "    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:",
            "    if not stat.S_ISREG(info.st_mode):",
        ),
    ],
}


def main() -> None:
    if MUTANTS_DIR.exists():
        shutil.rmtree(MUTANTS_DIR)
    MUTANTS_DIR.mkdir(parents=True)
    for name, edits in MUTANTS.items():
        dst = MUTANTS_DIR / name / "src"
        shutil.copytree(HEAD_SRC, dst)
        for rel, old, new in edits:
            apply_once(dst / rel, old, new)
    print(f"PREPARED {len(MUTANTS)} scratch mutants under {MUTANTS_DIR}")


if __name__ == "__main__":
    main()
