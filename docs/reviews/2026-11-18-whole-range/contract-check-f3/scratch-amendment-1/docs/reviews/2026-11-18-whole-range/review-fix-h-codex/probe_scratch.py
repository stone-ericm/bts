"""Change one behavior per scratch source copy and run the listed test groups."""
from pathlib import Path
from shutil import copytree
import subprocess
import sys
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TESTS = (
    "tests/test_wr_h_longest.py", "tests/simulate/test_tail_policy.py",
    "tests/test_wr_g_guard.py", "tests/test_strategy.py", "tests/test_decide_action.py",
    "tests/test_scheduler_tail_objective.py", "tests/test_daily_decision_v3.py",
    "tests/test_scheduler_decision_record.py", "tests/test_scheduler_decision_record_integration.py",
    "tests/scripts/season_walk/",
)
TAIL = "simulate/tail_policy.py"
STRATEGY = "strategy.py"
MUTATIONS = {
    "double-first-tie": (TAIL, '    return ("single", "double", "skip")[values.index(gain)]',
                         '    return "double" if values[1] == gain else ("single", "double", "skip")[values.index(gain)]'),
    "early-action": (TAIL, "p_hit * (1.0 + gain), p_both * (2.0 + gain)",
                     "p_hit * (2.0 + gain), p_both * (1.0 + gain)"),
    "allow-two-bins": (STRATEGY, "        if tail.n_bins != 1:\n"
                       '            raise ValueError("tail policy must have one quality bin")\n', ""),
    "allow-nonpositive": (STRATEGY, '        if int(policy["season_length"]) <= 0:\n'
                          '            raise ValueError("base policy season length must be positive")\n', ""),
    "extend-at-zero": (TAIL, " and days_remaining > 0", ""),
    "use-base-horizon": (STRATEGY, "        return OBJECTIVE_LONGEST, min(days, LONGEST_STREAK_HORIZON)",
                         '        base_horizon = int((mdp or {}).get("season_length") or 180)\n'
                         "        horizon = min(LONGEST_STREAK_HORIZON, base_horizon) if base_horizon > 0 else LONGEST_STREAK_HORIZON\n"
                         "        return OBJECTIVE_LONGEST, min(days, horizon)"),
    "skip-when-degraded": (STRATEGY, '        return "single", reason, None', '        return "skip", reason, None'),
    "drop-objective": ("daily_decision.py", 'OBJECTIVES = ("reach57", "emax_season_best", "longest_streak")',
                       'OBJECTIVES = ("reach57", "emax_season_best")'),
    "read-before-regime": (STRATEGY,
        '    if _regime(None, streak, date)[0] == OBJECTIVE_LONGEST:\n'
        "        try:\n"
        "            mdp = _load_mdp()\n"
        "        except Exception as exc:\n"
        '            mdp = {"base_error": f"{type(exc).__name__}: {exc}"}\n'
        "    else:\n"
        "        mdp = _load_mdp()\n"
        "    ctx = DecisionContext(",
        "    try:\n"
        "        mdp = _load_mdp()\n"
        "    except Exception as exc:\n"
        "        if _regime(None, streak, date)[0] != OBJECTIVE_LONGEST:\n"
        "            raise\n"
        '        mdp = {"base_error": f"{type(exc).__name__}: {exc}"}\n'
        "    ctx = DecisionContext("),
    "no-longest-bar": (STRATEGY, "        if _regime(None, streak, date)[0] == OBJECTIVE_LONGEST:\n"
                       "            return 0.0", "        if _regime(None, streak, date)[0] == OBJECTIVE_LONGEST:\n"
                       "            return None"),
}


def boundary_check(value):
    from datetime import date
    from unittest.mock import patch

    from bts import orchestrator, strategy
    from bts.simulate import tail_policy as tp
    from probe_delivery import world
    from scripts.audit.season_walk import walk

    assert value == tp.LONGEST_STREAK_FROM
    mdp = strategy._load_mdp()
    for streak in (55, 56, 57, 58, 59, 100):
        for text, raw in (("2027-09-27", 0), ("2027-09-26", 1), ("2027-07-15", 74)):
            active = raw > 0 and streak >= value
            ctx = strategy.DecisionContext(.96, .95, True, text, True, mdp)
            assert (tp.mdp_objective(streak, raw) == "longest_streak") == active
            assert (strategy._regime(mdp, streak, text)[0] == "longest_streak") == active
            dec = strategy.resolve_policy_decision(ctx, streak, False)
            assert (dec.objective == "longest_streak") == active
            if active:
                assert dec.action == ("double" if raw == 1 else "single")
                with patch.object(strategy, "_load_mdp", side_effect=AssertionError("unexpected disk read")) as load:
                    assert strategy.effective_pick_bar(streak, text, False) == 0.0
                    load.assert_not_called()
    day = date(2027, 9, 26)
    assert walk.DEFAULT_STATES[-1].streak == value
    row = walk.walk_one(day, walk.DEFAULT_STATES[-1])
    if value == 56:
        assert row["action"] == "double" and walk.check([row]) == []
    else:
        assert row["setup"] and "57" in row["setup"][0]
    with TemporaryDirectory() as temp:
        with world(Path(temp), day, "dated", streak=min(value, 57)) as (cfg, frame, statuses):
            _, selection, _ = orchestrator.run_and_pick(cfg, str(day))
            assert selection.action == ("double" if value == 56 else "skip")
            direct = strategy.select_pick(frame, str(day), Path(temp), streak=value,
                                          saver_available=False, game_statuses_detailed=statuses)
            assert direct.action == "double" and direct.decision.objective == "longest_streak"
    print(f"constant {value}: dispatch, resolver, fresh selection and walk input follow the constant")
    print("walk run", row["action"], "setup", row["setup"])


def worker(src, kind):
    sys.path[:0] = [src, str(REPO)]
    import bts

    print(f"import {bts.__file__}", flush=True)
    assert Path(bts.__file__).resolve() == Path(src) / "bts" / "__init__.py"
    if kind.startswith("boundary-"):
        boundary_check(int(kind.split("-")[1]))
        return
    import pytest

    # The separate archived parity test asserts the real checkout source path and
    # hardcodes it for its child, so it is a check of the real source, not this copy.
    raise SystemExit(pytest.main(["-p", "no:cacheprovider", *TESTS, "-q", "--tb=short",
                                  "--maxfail=1", "-k", "not legacy_parity_against_archived_base"]))


def run(kind):
    with TemporaryDirectory(prefix=f"{kind}-", dir=HERE) as temp:
        src = Path(temp) / "src"
        copytree(REPO / "src", src)
        if kind.startswith("boundary-"):
            value = int(kind.split("-")[1])
            name, old, new = TAIL, "LONGEST_STREAK_FROM = 57", f"LONGEST_STREAK_FROM = {value}"
        else:
            name, old, new = MUTATIONS[kind]
        path = src / "bts" / name
        source = path.read_text()
        assert source.count(old) == 1, (kind, old)
        path.write_text(source.replace(old, new, 1))
        try:
            result = subprocess.run(
                [sys.executable, __file__, "--worker", str(src), kind], cwd=REPO,
                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=25)
        except subprocess.TimeoutExpired as exc:
            (HERE / f"scratch-{kind}.log").write_bytes(exc.stdout or b"")
            print(kind, "timed out", flush=True)
            return
        (HERE / f"scratch-{kind}.log").write_text(result.stdout)
        last_line = result.stdout.strip().splitlines()[-1]
        print(kind, "exit", result.returncode, last_line, flush=True)
        if kind.startswith("boundary-"):
            assert result.returncode == 0, result.stdout
        else:
            assert result.returncode in (0, 1), result.stdout


if __name__ == "__main__":
    if sys.argv[1] == "--worker":
        worker(*sys.argv[2:])
    else:
        for case in sys.argv[1:]:
            run(case)
