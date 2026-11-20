"""Create scratch mutant copies under .codex-review/wr-c2-review/mutants/ without touching tracked files."""

from __future__ import annotations

from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HEAD_SRC = ROOT / "src"
MUTANTS_ROOT = HERE / "mutants"


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert text.count(old) == 1, (path, old, text.count(old))
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> None:
    if MUTANTS_ROOT.exists():
        shutil.rmtree(MUTANTS_ROOT)
    MUTANTS_ROOT.mkdir(parents=True)

    specs = {
        "m1a_wall_date_both": (
            "bts/scheduler.py",
            "            status_path=picks_dir.parent / \"health_state\" / \"delivery_outcome_unknown\" / f\"{slate}.json\",\n"
            "            now_et_date=slate,",
            "            status_path=picks_dir.parent / \"health_state\" / \"delivery_outcome_unknown\" / f\"{_now_et().date()}.json\",\n"
            "            now_et_date=_now_et().date(),",
        ),
        "m1b_wall_date_now_et_only": (
            "bts/scheduler.py",
            "            now_et_date=slate,",
            "            now_et_date=None,",
        ),
        "m1c_wall_date_filename_only": (
            "bts/scheduler.py",
            "            status_path=picks_dir.parent / \"health_state\" / \"delivery_outcome_unknown\" / f\"{slate}.json\",",
            "            status_path=picks_dir.parent / \"health_state\" / \"delivery_outcome_unknown\" / f\"{_now_et().date()}.json\",",
        ),
        "m2_shared_file": (
            "bts/scheduler.py",
            "            status_path=picks_dir.parent / \"health_state\" / \"delivery_outcome_unknown\" / f\"{slate}.json\",",
            "            status_path=picks_dir.parent / \"health_state\" / \"health_dm_delivery_status.json\",",
        ),
        "m3_accept_none_id": (
            "bts/health/alert.py",
            "def _usable_message_id(value: object) -> bool:\n    return isinstance(value, str) and bool(value.strip())",
            "def _usable_message_id(value: object) -> bool:\n    return value is None or (isinstance(value, str) and bool(value.strip()))",
        ),
        "m4_drop_classification_call": (
            "bts/scheduler.py",
            "            if attempted and not delivered:\n                _alert_delivery_outcome_unknown(config, picks_dir, date)\n",
            "",
        ),
        "m5_before_write_classification_decision": (
            "bts/scheduler.py",
            "            _write_classification_decision(\n"
            "                picks_dir, date,\n"
            "                action=(\"double\" if ld.double_down else \"single\"),\n"
            "                delivered=delivered,\n"
            "                attempted=attempted,\n"
            "                primary=_row_from_daily(ld.pick),\n"
            "                double_down=_row_from_daily(ld.double_down),\n"
            "                state=state, daily=ld,\n"
            "            )\n"
            "            if attempted and not delivered:\n"
            "                _alert_delivery_outcome_unknown(config, picks_dir, date)",
            "            if attempted and not delivered:\n"
            "                _alert_delivery_outcome_unknown(config, picks_dir, date)\n"
            "            _write_classification_decision(\n"
            "                picks_dir, date,\n"
            "                action=(\"double\" if ld.double_down else \"single\"),\n"
            "                delivered=delivered,\n"
            "                attempted=attempted,\n"
            "                primary=_row_from_daily(ld.pick),\n"
            "                double_down=_row_from_daily(ld.double_down),\n"
            "                state=state, daily=ld,\n"
            "            )",
        ),
        "m6a_banner_on_own_failed": (
            "bts/health/alert.py",
            "        if (prior and prior.get(\"status\") == \"sent\" and prior.get(\"sent_day\") == day\n"
            "                and prior.get(\"sent_sources\") == alert_key and _usable_message_id(prior.get(\"message_id\"))):\n"
            "            return False",
            "        if (prior and prior.get(\"status\") == \"sent\" and prior.get(\"sent_day\") == day\n"
            "                and prior.get(\"sent_sources\") == alert_key and _usable_message_id(prior.get(\"message_id\"))):\n"
            "            return False\n"
            "        if prior and prior.get(\"status\") == \"failed\":\n"
            "            body = \"⚠️ Previous health alert DM FAILED to send (1 CRITICAL) — it may have been missed.\\n\\n\" + body",
        ),
        "m6b_banner_drop_require_id_guard": (
            "bts/health/alert.py",
            "    prior = None\n"
            "    if not require_message_id:\n"
            "        prior = _read_health_dm_delivery_status(status_path)\n"
            "        resurface = _resurface_note(prior)  # H6: resurface a prior failed DM\n"
            "        if resurface:\n"
            "            body = resurface + (\"\\n\\n\" + body if body else \"\")",
            "    prior = _read_health_dm_delivery_status(status_path)\n"
            "    resurface = _resurface_note(prior)\n"
            "    if resurface:\n"
            "        body = resurface + (\"\\n\\n\" + body if body else \"\")",
        ),
        "m7a_follow_file_symlinks": (
            "bts/health/alert.py",
            "def _check_receipt_entry(directory_fd: int, name: str) -> None:\n"
            "    try:\n"
            "        info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)\n"
            "    except FileNotFoundError:\n"
            "        return\n"
            "    _check_receipt_file(info)\n\n\n"
            "def _read_receipt(directory_fd: int, name: str) -> dict | None:\n"
            "    _check_receipt_entry(directory_fd, name)\n"
            "    try:\n"
            "        fd = os.open(name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=directory_fd)",
            "def _check_receipt_entry(directory_fd: int, name: str) -> None:\n"
            "    try:\n"
            "        info = os.stat(name, dir_fd=directory_fd, follow_symlinks=True)\n"
            "    except FileNotFoundError:\n"
            "        return\n"
            "    _check_receipt_file(info)\n\n\n"
            "def _read_receipt(directory_fd: int, name: str) -> dict | None:\n"
            "    _check_receipt_entry(directory_fd, name)\n"
            "    try:\n"
            "        fd = os.open(name, os.O_RDONLY | os.O_NONBLOCK, dir_fd=directory_fd)",
        ),
        "m7b_follow_dir_symlinks": (
            "bts/health/alert.py",
            "directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)",
            "directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)",
        ),
        "m8_drop_replace_entry_recheck": (
            "bts/health/alert.py",
            "        _check_receipt_entry(directory_fd, name)\n"
            "        os.replace(temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)",
            "        os.replace(temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)",
        ),
        "m9_allow_hardlinks": (
            "bts/health/alert.py",
            "if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:",
            "if not stat.S_ISREG(info.st_mode):",
        ),
    }

    for name, (rel_file, old, new) in specs.items():
        dst = MUTANTS_ROOT / name / "src"
        shutil.copytree(HEAD_SRC, dst)
        replace_once(dst / rel_file, old, new)

    print(f"Prepared {len(specs)} scratch mutants under {MUTANTS_ROOT}")


if __name__ == "__main__":
    main()
