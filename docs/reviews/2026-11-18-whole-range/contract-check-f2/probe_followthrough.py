import datetime as dt
import shlex
from pathlib import Path
from tempfile import TemporaryDirectory

from click.testing import CliRunner

from bts import cli, contest_state
from bts.saver_state import load_saver_state, maybe_auto_earn_saver
from bts.simulate.mdp import load_policy
from bts.strategy import DecisionContext, resolve_policy_decision

from probe_support import DAY, ET, HERE, REAL_DATETIME, auto, clock, command, flag, manual


with TemporaryDirectory(dir=HERE, prefix="followthrough-") as directory:
    root = Path(directory)
    when = REAL_DATETIME(2027, 8, 20, 11, tzinfo=ET)
    picks = root / "wrong use" / "picks"
    flag(picks, "active")
    manual(picks, when - dt.timedelta(days=1), state="active", streak=9)
    auto(picks, when, streak=9)
    table, boundaries, length = load_policy("data/models/mdp_policy.npz")
    context = DecisionContext(.79, .80, True, when.date().isoformat(), True, {
        "policy_table": table, "boundaries": boundaries, "season_length": length,
    })
    decision = contest_state.load_decision_streak_state(picks, now=when, require_contest_state=True)
    before = resolve_policy_decision(context, decision.streak, decision.saver_available).action
    command(picks, when, "saver-state", "--use", "--season", str(when.year))
    decision = contest_state.load_decision_streak_state(picks, now=when, require_contest_state=True)
    after = resolve_policy_decision(context, decision.streak, decision.saver_available).action
    print("wrong marking at streak 9", before, "->", after)
    assert (before, after) == ("single", "skip")

    picks = root / "initialize" / "picks"
    auto(picks, when, streak=9, best=9)
    decision = contest_state.load_decision_streak_state(picks, now=when, require_contest_state=True)
    assert not decision.saver_available
    maybe_auto_earn_saver(picks, best_streak=9, season=when.year)
    decision = contest_state.load_decision_streak_state(picks, now=when, require_contest_state=True)
    print("missing flag after low-best fetch", decision.saver_flag_state, decision.saver_available)
    assert decision.saver_flag_state == "not_earned" and decision.saver_available

    new_year = REAL_DATETIME(2028, 1, 1, 0, 10, tzinfo=ET)
    with CliRunner().isolated_filesystem(temp_dir=root):
        picks = Path("data/picks")
        flag(picks, "active", season=new_year.year)
        with clock(new_year):
            failed = CliRunner().invoke(cli.cli, ["saver-state", "--use", "--season", "2027"])
            followup = shlex.split(failed.output.split("`")[1])
            print("suggested after old-season failure", followup)
            assert failed.exit_code == 1 and followup[0] == "bts"
            assert CliRunner().invoke(cli.cli, followup[1:]).exit_code == 0
            assert load_saver_state(picks, season=2027).state == "used"
            assert load_saver_state(picks, season=2028).state == "uninitialized"
            print("old-season init replaced new-year active flag")
