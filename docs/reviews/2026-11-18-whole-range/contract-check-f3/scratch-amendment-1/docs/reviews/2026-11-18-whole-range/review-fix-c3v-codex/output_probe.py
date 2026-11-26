"""Check both archived summary branches and test a scratch newline-rendering fix."""

from datetime import datetime
import inspect
from unittest.mock import patch

from probe_common import examine, receipt, run_against_archives, validate


PAST = "2027-04-16"


def probes(scratch, revision):
    from bts import cli, picks
    from tests.test_wr_c3v_corpus import dump, pick_data, seed_streak

    results = {}
    for name, corrections in (("old-stub-empty", []), ("old-stub-corrections", [
        {"date": "2027-12-30", "batter": "B", "old_result": "hit", "new_result": "miss"},
    ])):
        root = scratch / name / "picks"
        seed_streak(root, season=2026)
        clock = lambda: datetime.fromisoformat("2028-01-01T01:00:00+00:00")
        with patch.object(picks, "reconcile_results", return_value=corrections), \
             patch.object(picks, "load_streak", wraps=picks.load_streak) as load:
            results[name] = examine(root, clock=clock, direct=False)
            load.assert_called_once_with(root, season=2027)

    root = scratch / "corrected-saved" / "picks"
    seed_streak(root, streak=9)
    dump(root / "2027-04-19.json", pick_data("2027-04-19", "hit"))
    with patch.object(picks, "resolve_daily_slot_results", return_value={"pick": "miss"}):
        results["corrected-saved"] = examine(root, lookback=8)

    for label, date in (("canonical", PAST), ("newline", PAST + "\n")):
        root = scratch / label / "picks"
        seed_streak(root, streak=9)
        (root / f"{date}.json").write_text("{}")
        results[label] = examine(root, lookback=8)
        if revision == "current":
            source = inspect.getsource(cli.reconcile.callback)
            source = source[source.index("def reconcile("):]
            original = '    if blocked:\n        click.echo(\n            f"Season replay unavailable for {replay[\'blocked_date\']} '
            updated = '    if blocked:\n        blocked_date = replay["blocked_date"].replace("\\n", r"\\n")\n' \
                      '        click.echo(\n            f"Season replay unavailable for {blocked_date} '
            assert source.count(original) == 1
            namespace = dict(vars(cli))
            exec(compile(source.replace(original, updated), "<scratch-cli>", "exec"), namespace)
            with patch.object(cli.reconcile, "callback", namespace["reconcile"]):
                results[label + "-escaped"] = examine(root, lookback=8)
    return results


def check(base, current):
    canonical_fixed = current.pop("canonical-escaped")
    newline_fixed = current.pop("newline-escaped")
    assert canonical_fixed == current["canonical"]
    old_newline = base.pop("newline")
    new_newline = current.pop("newline")
    assert old_newline["strict"] == new_newline["strict"] == newline_fixed["strict"]
    assert new_newline["receipt"] == newline_fixed["receipt"]
    assert new_newline["stdout"] == newline_fixed["stdout"]
    assert len(new_newline["stderr"].splitlines()) == 2
    assert newline_fixed["stderr"] == (
        r"Season replay unavailable for 2027-04-16\n (no final result on the pick); streak retained, not recalculated."
        "\n"
    )
    counts = validate(base, current)
    for name in ("old-stub-empty", "old-stub-corrections"):
        assert receipt(current[name])["replay"] == {"state": "not_reached"}
    assert current["old-stub-empty"]["stdout"].endswith("No scoring changes detected. Streak: 0\n")
    assert current["old-stub-corrections"]["stdout"].endswith("Streak recalculated: 0\n")
    assert current["corrected-saved"]["stdout"].endswith("hit -> miss\nStreak recalculated: 0\n")
    return {"receipt_states": counts, "escaped_one_line": newline_fixed["stderr"].rstrip()}


if __name__ == "__main__":
    run_against_archives(__file__, probes, check)
