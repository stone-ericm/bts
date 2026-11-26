"""Replay the committed tests against one production edit at a time in scratch."""

from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory

from pytest_family import NEW, REVIEW, ROOT, pytest_run


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new, 1)


def run():
    review_source = ROOT / "src" / "bts" / "scheduler.py"
    text = review_source.read_text()
    positions = [match.start() for match in re.finditer(r"except BaseException", text)]
    exception_lines = [text.count("\n", 0, position) + 1 for position in positions]
    row1_positions = [position for position, line in zip(positions, exception_lines) if line in (118, 125, 2944)]
    assert len(row1_positions) == 3, exception_lines
    mutants = [
        ("remove-call", "else:\n                _morning_state_check(config, picks_dir, date, now, first_check_et)",
         "else:\n                pass"),
        ("remove-delivery-mode", 'if (delivery_mode in {"dm", "public"} and _contest_state_required(config)',
         "if (_contest_state_required(config)"),
        ("remove-required", 'if (delivery_mode in {"dm", "public"} and _contest_state_required(config)',
         'if (delivery_mode in {"dm", "public"}'),
        ("lose-dm-mode", 'if (delivery_mode in {"dm", "public"} and _contest_state_required(config)',
         'if (delivery_mode == "public" and _contest_state_required(config)'),
        ("lose-public-mode", 'if (delivery_mode in {"dm", "public"} and _contest_state_required(config)',
         'if (delivery_mode == "dm" and _contest_state_required(config)'),
        ("remove-prior-lock", 'and (previous_state is None or not previous_state.pick_locked)):', '):'),
        ("remove-dry-gate", 'if dry_run:\n        print("  (--dry-run: not executing checks)"',
         'if dry_run:\n        _morning_state_check(config, picks_dir, date, _now_et(), runs[0]["time_et"])\n'
         '        print("  (--dry-run: not executing checks)"'),
        ("remove-no-games-gate", 'if not games:\n        print(f"No games scheduled for {date}.", file=sys.stderr)',
         'if not games:\n        _morning_state_check(config, picks_dir, date, _now_et(), _now_et() + timedelta(hours=1))\n'
         '        print(f"No games scheduled for {date}.", file=sys.stderr)'),
        ("remove-empty-gate", 'if first_check_et is None:\n                _print_morning_check',
         'if False:\n                _print_morning_check'),
        ("remove-margin", 'elif first_check_et - now < timedelta(minutes=MORNING_CHECK_MIN_LEAD_MIN):', 'elif False:'),
        ("nine-minutes", "MORNING_CHECK_MIN_LEAD_MIN = 10", "MORNING_CHECK_MIN_LEAD_MIN = 9"),
        ("inclusive-boundary", "first_check_et - now < timedelta(minutes=MORNING_CHECK_MIN_LEAD_MIN)",
         "first_check_et - now <= timedelta(minutes=MORNING_CHECK_MIN_LEAD_MIN)"),
        ("remove-date-return", "if date != now.date().isoformat():\n        return\n    try:", "try:"),
        ("remove-scoreable-return", "if is_scoreable_commit(date, picks_dir, None):\n            return",
         "if is_scoreable_commit(date, picks_dir, None):\n            pass"),
        ("remove-delivered-return", "(pick_was_delivered(daily) or daily.delivery_attempted):\n        return",
         "daily.delivery_attempted:\n        return"),
        ("remove-attempt-return", "(pick_was_delivered(daily) or daily.delivery_attempted):\n        return",
         "pick_was_delivered(daily):\n        return"),
        ("refusal-in-dm", 'f"a lineup check remains. First planned lineup check at {first_check_et:%H:%M} ET."',
         'f"a lineup check remains. First planned lineup check at {first_check_et:%H:%M} ET. {error}"'),
        ("shared-status", 'status_path=health_dir / "contest_morning_check_dm_delivery_status.json",',
         'status_path=health_dir / "health_dm_delivery_status.json",'),
        ("different-warn-key", 'warn_attention=[Alert("WARN", "contest_state", message, incident_key="contest_state:stale_reading")],',
         'warn_attention=[Alert("WARN", "contest_state", message, incident_key="contest_state:morning_stale")],'),
        ("remove-recipient-condition", '        if recipient:\n            message = (\n                f"Morning check for {date}: no usable',
         '        if True:\n            message = (\n                f"Morning check for {date}: no usable'),
        ("write-scheduler-state", '    health_dir = picks_dir.parent / "health_state"\n    recipient',
         '    save_state(load_state(date, picks_dir), picks_dir)\n    health_dir = picks_dir.parent / "health_state"\n    recipient'),
        ("double-read", "        state = load_decision_streak_state(picks_dir, require_contest_state=True, now=now)",
         "        state = load_decision_streak_state(picks_dir, require_contest_state=True, now=now)\n"
         "        state = load_decision_streak_state(picks_dir, require_contest_state=True, now=now)"),
        ("double-read-refusal", "    except ContestStateError as error:\n        _print_morning_check(",
         "    except ContestStateError as error:\n        try:\n"
         "            load_decision_streak_state(picks_dir, require_contest_state=True, now=now)\n"
         "        except ContestStateError:\n            pass\n        _print_morning_check("),
    ]
    results = {}
    for name, before, after in mutants:
        results[name] = replace_once(text, before, after)
    for name, position in zip(("narrow-failure-print", "narrow-advisory-print", "narrow-boundary"), row1_positions):
        results[name] = text[:position] + text[position:].replace("except BaseException", "except Exception", 1)
    for name, prefix in (("omit-date-critical", "                "), ("omit-date-warn", "            ")):
        line = f"\n{prefix}now_et_date=now.date(),"
        results[name] = replace_once(text, line, "")

    with TemporaryDirectory(prefix="cs6b1-mutations-", dir=REVIEW) as temp:
        source = Path(temp) / "src"
        shutil.copytree(REVIEW / "base" / "src", source)
        scheduler = source / "bts" / "scheduler.py"
        dispatcher = source / "bts" / "health" / "alert.py"
        stock_dispatcher = dispatcher.read_text()
        gate_module = source / "bts" / "entry_intent.py"
        stock_gate = gate_module.read_text()
        for name, variant in [*results.items(), ("remove-dedup-return", text),
                              ("require-scheduler-flag-only", text), ("require-health-flag-only", text)]:
            scheduler.write_text(variant)
            if name == "remove-dedup-return":
                dispatcher.write_text(replace_once(stock_dispatcher, "                return False\n\n    if not dm_recipient:",
                                                   "                pass\n\n    if not dm_recipient:"))
            elif name == "require-scheduler-flag-only":
                scheduler.write_text(replace_once(text, "and _contest_state_required(config)\n",
                                                   'and config.get("scheduler", {}).get("contest_state_required")\n'))
            elif name == "require-health-flag-only":
                scheduler.write_text(replace_once(text, "and _contest_state_required(config)\n",
                                                   'and config.get("health_checks", {}).get("contest_state_expected")\n'))
            code = pytest_run(f"mut-{name}", [NEW], source=source, options=("--tb=short",))
            assert code in {0, 1}, (name, code)
            dispatcher.write_text(stock_dispatcher)
            gate_module.write_text(stock_gate)


if __name__ == "__main__":
    run()
