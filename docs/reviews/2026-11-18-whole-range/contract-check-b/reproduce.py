"""Run the relevant offline slice probes on the worktree under review."""

import json
import shlex
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
SLICES = HERE.parents[1] / "docs/reviews/2026-11-18-whole-range"
for index in (1, 2, 3):
    sys.path.insert(0, str(SLICES / f"slice-{index}-codex"))

import cron_probe
import day_probe
import opening_edges
from opening_rehearsal import Season
import refusals
from clocks_and_scoring import fetch_fixture
from offline import instant, offline


def opening(probe, *args):
    with TemporaryDirectory(prefix="opening-", dir=HERE) as folder:
        season = Season(Path(folder))
        from bts.dm import send_dm

        season.real_transport = send_dm
        season.old_seed()
        with season.run_offline(), redirect_stderr(season.journal):
            if probe == "checklist":
                from opening_rehearsal import run_case

                run_case(season, "automatic_match", False)
                saver = season.command("saver-state", contains="saver_state: not_earned")
                refusal = season.command("saver-state", "--init", "not_earned", "--season", "2027",
                                         expected=1, contains="already initialized")
                return saver.strip(), refusal.strip()
            season.restore()
            return probe(season, *args)


def outcomes():
    with TemporaryDirectory(prefix="outcome-", dir=HERE) as folder, redirect_stdout(StringIO()):
        root = Path(folder)
        cases = {case[0]: case for case in day_probe.CASES}
        single = day_probe.run_case(root / "single", cases["mid_public"], False)
        double = day_probe.run_case(root / "double", cases["mid_double"], False)
        with_receipt = cron_probe.run_entry(root / "receipt")
    for result, decision in ((single, None), (with_receipt, with_receipt["decision"]), (double, None)):
        outcome = json.loads(result["day_outcome"]["stdout"])
        if decision is None:
            decision = result["after_scheduler"][f"data/picks/{outcome['date']}/decision.json"]
        print("B3", decision["action"], outcome["state"], outcome["pick"])
    return with_receipt


def malformed_status_failure_lane():
    from bts import cli, contest_fetch

    with offline(instant("2027-03-24", 10)) as (world, patch):
        world.automatic(7)
        world.health.mkdir(exist_ok=True)
        status = world.health / "contest_streak_fetch_status.json"
        fetch_fixture(world, patch, rounds={9: world.now.date()}, rows={})
        for contents in (b"null", b"\xff"):
            status.write_bytes(contents)
            result = world.call("fetch-contest-streak", "--picks-dir", str(world.picks),
                                "--dm-recipient", "offline.invalid", code=1, catch=True)
            assert not world.sends("dm") and status.read_bytes() == contents
            print("B5 failure lane", repr(contents), type(result.exception).__name__)

        fetch_fixture(world, patch, rounds={9: world.now.date()})
        status.write_bytes(b"null")
        result = world.call("fetch-contest-streak", "--picks-dir", str(world.picks), code=1, catch=True)
        assert isinstance(result.exception, AttributeError)
        print("B5 success lane", repr(status.read_bytes()), type(result.exception).__name__)

        status.write_text(json.dumps({"last_alert_at_by_category": {"actionable": 123}}))
        fetch_fixture(world, patch, rounds={9: world.now.date()}, rows={})
        result = world.call("fetch-contest-streak", "--picks-dir", str(world.picks),
                            "--dm-recipient", "offline.invalid", code=1, catch=True)
        print("B5 malformed object member", type(result.exception).__name__)


def status_legacy_and_time():
    from bts.contest_state import load_decision_streak_state

    with offline(instant("2027-12-31", 23, 59)) as (world, _):
        world.automatic(7)
        auto = world.account / "contest_streak.json"
        observation = json.loads(auto.read_text())
        del observation["recorded_at"]
        auto.write_text(json.dumps(observation))
        live = load_decision_streak_state(world.picks, now=world.now, require_contest_state=True)
        print("B4 missing recorded time accepted", live.observation_kind, live.contest_recorded_at)
        world.call("set-contest-streak", "--picks-dir", str(world.picks), "--streak", "1")
        result = world.call("season-contract", "status", "--picks-dir", str(world.picks))
        assert "legacy manual override is no longer accepted" in result.output
        world.dated(1)
        world.now = instant("2028-01-01", 0)
        result = world.call("season-contract", "status", "--picks-dir", str(world.picks))
        assert "switch is other_season" in result.output


def main():
    saver = opening(opening_edges.ambiguous_saver_instructions, False)
    print("B1", saver["warning"])
    for key in ("generic_command_status", "dated_command_status"):
        print(key, next(line for line in saver[key].splitlines() if line.startswith("live decision now:")))
    unconfirmed = opening(opening_edges.restart_at_send, "corrected_after_marker")
    print("B2", unconfirmed["snapshot"]["decision"]["delivery_status"],
          [item["body"] for item in unconfirmed["messages"] if "Fix it now!" in item["body"]])
    receipt = outcomes()
    print("B2 delivered control", [event for attempt in receipt["entry"] for event in attempt["dm"]])
    cutover = opening(opening_edges.cutover_with_masked_file)
    print("B4", *(next(line for line in cutover[key].splitlines() if line.startswith("live decision now:"))
                  for key in ("before", "after")))
    status_legacy_and_time()
    refusals.fetch_failures()
    malformed_status_failure_lane()
    print("B6", *opening("checklist"))
    # Verify the catch-all concerns through the controls above and confirm its wording.
    kickoff = (SLICES / "kickoff-fix-b.md").read_text()
    assert "| B7 | Everything else | | Unchanged |" in kickoff
    print("B7 catch-all has no independent defect; controls ran above")


if __name__ == "__main__":
    main()
