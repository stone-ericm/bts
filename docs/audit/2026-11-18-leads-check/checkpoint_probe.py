import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from click.testing import CliRunner

from bts import daily_decision, shadow_eval, skip_policy_shadow as shadow
from bts.cli import cli


class Clock(datetime):
    pinned = datetime(2027, 7, 10, 19, tzinfo=timezone.utc)

    @classmethod
    def now(cls, tz=None):
        return cls.pinned.astimezone(tz) if tz else cls.pinned.replace(tzinfo=None)


START = Clock.pinned
GAP = (START - timedelta(days=5)).date().isoformat()
LAST = (START - timedelta(days=4)).date().isoformat()
OUTCOMES = {}


def decision(root, day, result):
    pk = date.fromisoformat(day).toordinal()
    OUTCOMES[pk] = result
    assert daily_decision.write_decision(
        day, root, action="skip", source="mdp", scoreable=False, delivery_status="not_applicable",
        primary={"game_pk": pk, "batter_id": 1, "batter_name": "Example Batter", "p_game_hit": .75},
    )


def seed(root):
    days = [(START - timedelta(days=offset)).date().isoformat()
            for offset in [*range(34, 5, -1), 4]]
    for index, day in enumerate(days):
        result = "miss" if index == 0 else "hit"
        decision(root, day, result)
        row = shadow.record_skip_from_decision(day, root, now=START)
        assert shadow.reconcile_decision(row, hit_checker=lambda _: result, now=START)
        shadow.decision_path(day, root).write_text(json.dumps(row))
    return days


def update(root, when, *, no_reconcile=False):
    Clock.pinned = when
    path = root / "status.json"
    args = ["skip-policy-shadow-update", "--picks-dir", str(root), "--status-output", str(path)]
    if no_reconcile:
        args.append("--no-reconcile")
    called = []

    def check(rank):
        called.append(rank["game_pk"])
        return OUTCOMES[rank["game_pk"]]

    with (
        patch("datetime.datetime", Clock), patch.object(shadow, "datetime", Clock),
        patch.object(shadow, "make_hit_checker", return_value=check),
        patch.object(shadow_eval, "_current_git_commit", return_value="synthetic"),
    ):
        result = CliRunner().invoke(cli, args, catch_exceptions=False)
        assert result.exit_code == 0, result.output
    return json.loads(path.read_text()), called


def observed(status):
    band = status["shadow_band_hit_rate"]
    basis = band["verdict_basis"]
    cutoff = basis["eligibility_cutoff_date"]
    eligible = [r["date"] for r in status["rows"] if r["date"] <= cutoff
                and r["shadow_pick_result"] in ("hit", "miss")]
    assert eligible == sorted(eligible) and len(eligible) == basis["eligible_n"]
    return band["verdict"], basis["eligible_n"], basis["n_used"], basis["hits_used"], eligible[:30]


def revision(root, *, pending_first):
    days = seed(root)
    if pending_first:
        decision(root, GAP, "miss")
    first, first_calls = update(root, START, no_reconcile=pending_first)
    before = observed(first)
    assert before == ("above_breakeven", 30, 30, 29, days) and not first_calls
    if pending_first:
        assert first["counts"]["pending"] == 1
        when = START + timedelta(days=1)
    else:
        assert GAP not in [row["date"] for row in first["rows"]]
        # A historical decision arrives during the outage. Its age on return
        # is exactly 10 days, and it precedes the last member of the first look.
        decision(root, GAP, "miss")
        when = START + timedelta(days=5)
    second, second_calls = update(root, when)
    after = observed(second)
    assert after == ("straddles_breakeven", 31, 30, 28, days[:-1] + [GAP])
    assert second_calls == [date.fromisoformat(GAP).toordinal()]
    assert LAST not in after[4]
    third, third_calls = update(root, when + timedelta(days=1))
    assert observed(third) == after and not third_calls
    print(f"{'aged pending without outage' if pending_first else 'five-day outage with historical gap'}: "
          f"{before[:4]} -> {after[:4]}; {GAP} replaces {LAST}")


def correction_and_control(root):
    days = seed(root)
    first, _ = update(root, START)
    before = observed(first)
    new_day = START.date().isoformat()
    decision(root, new_day, "miss")
    later, called = update(root, START + timedelta(days=5))
    assert called == [date.fromisoformat(new_day).toordinal()]
    after = observed(later)
    assert before == ("above_breakeven", 30, 30, 29, days)
    assert after == ("above_breakeven", 31, 30, 29, days)
    print("five-day outage with append-only dates: eligible 30 -> 31, original look stable")

    # Two aged authoritative corrections remove existing members without
    # removing their on-disk shadow records.
    for day in days[1:3]:
        assert daily_decision.write_decision(
            day, root, action="single", source="mdp", scoreable=True, delivery_status="delivered")
    corrected, called = update(root, START + timedelta(days=6))
    assert not called
    assert corrected["counts"]["aged_superseded_records"] == days[1:3]
    assert all(shadow.decision_path(day, root).exists() for day in days[1:3])
    assert observed(corrected)[:4] == ("insufficient_n", 29, None, None)
    print("aged authoritative corrections: 30-member above-breakeven look disappears; shadow files retained")


if __name__ == "__main__":
    with TemporaryDirectory() as tmp, patch("socket.socket.connect", side_effect=AssertionError("network")):
        root = Path(tmp)
        revision(root / "gap", pending_first=False)
        revision(root / "pending", pending_first=True)
        correction_and_control(root / "correction")
