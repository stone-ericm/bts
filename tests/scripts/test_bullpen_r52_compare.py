"""Known-answer tests for the R52 bullpen comparison (synthetic backtest files only)."""

import json
import math
import os
from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from scripts.audit import bullpen_r52_compare as r52

SEEDS = (1, 2, 3, 4)
SEASONS = (2021, 2022)
DAYS = 200


def _write(root, arm, seed, season, hits, *, done=True, year=None, days=DAYS):
    run = root / "out" / arm / f"seed_{seed}"
    run.mkdir(parents=True, exist_ok=True)
    start = date(year or season, 1, 1)
    rows = []
    for i in range(days):
        d = (start + timedelta(days=i)).isoformat()
        rows.append({"date": d, "rank": 1, "batter_id": 1, "p_game_hit": 0.8, "actual_hit": int(i < hits), "n_pas": 4})
        rows.append({"date": d, "rank": 2, "batter_id": 2, "p_game_hit": 0.7, "actual_hit": 1, "n_pas": 4})
    pd.DataFrame(rows).to_parquet(run / f"backtest_{season}.parquet")
    if done:
        (run / "DONE").touch()


BASE = datetime.now().replace(microsecond=0) - timedelta(days=2)


def _log(root, seeds=SEEDS, *, ok=True):
    """A serial queue's log, one two-hour slot per run, and every file stamped inside its run."""
    lines = []
    k = 0
    for s in seeds:
        for arm in r52.ARMS:
            start = BASE + timedelta(hours=2 * k)
            end = start + timedelta(hours=1)
            lines += [f"{start:%Y-%m-%d %H:%M:%S} start arm={arm} seed={s}",
                      f"{end:%Y-%m-%d %H:%M:%S} end   arm={arm} seed={s} rc={0 if ok else 1}"]
            run = root / "out" / arm / f"seed_{s}"
            for f in run.glob("backtest_*.parquet"):
                t = (start + timedelta(minutes=1)).timestamp()
                os.utime(f, (t, t))
            if (run / "DONE").exists():
                t = (end + timedelta(milliseconds=500)).timestamp()
                os.utime(run / "DONE", (t, t))
            k += 1
    (root / "progress.log").write_text("\n".join(lines) + "\n")


def _grid(root, hits_old, hits_new, *, days=lambda y: DAYS):
    for s in SEEDS:
        for y in SEASONS:
            _write(root, "old", s, y, hits_old(s, y), days=days(y))
            _write(root, "new", s, y, hits_new(s, y), days=days(y))
    _log(root)


def _cmp(root):
    return r52.compare(root / "out", SEEDS, SEASONS)[0]


def _main(root, *extra):
    return r52.main(["--root", str(root / "out"), *extra], _grid=(SEEDS, SEASONS))


def test_identical_arms_pass(tmp_path):
    _grid(tmp_path, lambda s, y: 150, lambda s, y: 150)
    res = _cmp(tmp_path)
    assert res["mean_d"] == 0 and res["verdict"] == "PASS"


def test_one_point_drop_everywhere_fails_condition_1_and_passes_condition_2_at_equality(tmp_path):
    _grid(tmp_path, lambda s, y: 150, lambda s, y: 148)  # exactly -1.0 pt in every season
    res = _cmp(tmp_path)
    assert math.isclose(res["mean_d"], -0.01)
    assert res["condition_1_lower_bound_above_-0.005"] is False
    assert res["condition_2_no_season_below_-0.010"] is True  # equal to the floor passes
    assert res["verdict"] == "FAIL"


def test_lower_bound_exactly_at_floor_fails(tmp_path):
    _grid(tmp_path, lambda s, y: 138, lambda s, y: 137)  # exactly -0.5 pt everywhere, sd 0
    res = _cmp(tmp_path)
    assert res["condition_1_lower_bound_above_-0.005"] is False
    assert res["condition_2_no_season_below_-0.010"] is True
    assert res["verdict"] == "FAIL"


def test_season_exactly_at_floor_passes(tmp_path):
    # 2021 exactly -1.0 pt, 2022 +1.0 pt: d = 0 for every seed, so condition 1 holds
    _grid(tmp_path, lambda s, y: 150, lambda s, y: 148 if y == 2021 else 152)
    res = _cmp(tmp_path)
    assert res["condition_1_lower_bound_above_-0.005"] is True
    assert res["condition_2_no_season_below_-0.010"] is True
    assert res["verdict"] == "PASS"


def test_lower_bound_matches_hand_computation(tmp_path):
    delta = {1: 2, 2: 0, 3: -2, 4: 4}
    _grid(tmp_path, lambda s, y: 150, lambda s, y: 150 + delta[s])
    res = _cmp(tmp_path)
    d = [delta[s] / DAYS for s in SEEDS]
    mean = sum(d) / 4
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / 3)
    assert math.isclose(res["lower_bound_95"], mean - r52.t_quantile_95(3) * sd / 2, rel_tol=1e-12)
    assert math.isclose(r52.t_quantile_95(7), 1.8945786050900058)


def test_seasons_weigh_equally_whatever_their_length(tmp_path):
    # 2021 has 100 days at -2 pt, 2022 has 300 days at +2 pt: equal weights give d = 0;
    # day weighting would give +1 pt. (Condition 2 fails on 2021; that is not what this test checks.)
    _grid(tmp_path, lambda s, y: 70 if y == 2021 else 210,
          lambda s, y: 68 if y == 2021 else 216, days=lambda y: 100 if y == 2021 else 300)
    res = _cmp(tmp_path)
    assert res["mean_d"] == 0


def test_one_bad_season_fails_condition_2_even_when_pooled_mean_is_fine(tmp_path):
    _grid(tmp_path, lambda s, y: 150, lambda s, y: 147 if y == 2021 else 153)
    res = _cmp(tmp_path)
    assert res["condition_1_lower_bound_above_-0.005"] is True
    assert res["condition_2_no_season_below_-0.010"] is False
    assert res["verdict"] == "FAIL"


def test_cli_has_no_grid_override(tmp_path):
    _grid(tmp_path, lambda s, y: 150, lambda s, y: 150)
    with pytest.raises(SystemExit):
        r52.main(["--root", str(tmp_path / "out"), "--seeds", "1,2"])
    # the real grid is fixed: a four-seed, two-season tree is rule unmet for it
    assert r52.main(["--root", str(tmp_path / "out")]) == 2


def test_duplicate_grid_is_refused(tmp_path):
    _grid(tmp_path, lambda s, y: 150, lambda s, y: 150)
    with pytest.raises(r52.Incomplete):
        r52.compare(tmp_path / "out", (1, 1), SEASONS)
    with pytest.raises(r52.Incomplete):
        r52.compare(tmp_path / "out", SEEDS, (2021, 2021))


@pytest.mark.parametrize("damage", ["no_done", "no_file", "truncated", "no_date", "other_year",
                                    "two_rank1", "rank_gap", "null_hit", "nonbinary_hit",
                                    "shifted_days", "fewer_days"])
def test_bad_cell_is_rule_unmet(tmp_path, damage):
    _grid(tmp_path, lambda s, y: 150, lambda s, y: 150)
    run = tmp_path / "out" / "new" / "seed_3"
    path = run / "backtest_2022.parquet"
    df = pd.read_parquet(path)
    if damage == "no_done":
        (run / "DONE").unlink()
    elif damage == "no_file":
        path.unlink()
    elif damage == "truncated":
        path.write_bytes(path.read_bytes()[:100])
    elif damage == "no_date":
        df.drop(columns="date").to_parquet(path)
    elif damage == "other_year":
        df.loc[0, "date"] = "2023-01-01"
        df.to_parquet(path)
    elif damage == "two_rank1":
        df.loc[1, "rank"] = 1
        df.to_parquet(path)
    elif damage == "rank_gap":
        df.loc[1, "rank"] = 3
        df.to_parquet(path)
    elif damage == "null_hit":
        df["actual_hit"] = df["actual_hit"].astype(float)
        df.loc[0, "actual_hit"] = float("nan")
        df.to_parquet(path)
    elif damage == "nonbinary_hit":
        df.loc[0, "actual_hit"] = 2
        df.to_parquet(path)
    elif damage == "shifted_days":
        df["date"] = (pd.to_datetime(df["date"]) + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
        df.to_parquet(path)
    elif damage == "fewer_days":
        df.iloc[:-2].to_parquet(path)
    _log(tmp_path)  # file times back inside the run: the damage is what must be caught
    with pytest.raises(r52.Incomplete):
        _cmp(tmp_path)
    assert _main(tmp_path) == 2


def test_file_written_outside_its_logged_run_is_rule_unmet(tmp_path):
    _grid(tmp_path, lambda s, y: 150, lambda s, y: 150)
    stale = tmp_path / "out" / "old" / "seed_2" / "backtest_2021.parquet"
    old = (datetime.now() - timedelta(days=2)).timestamp()
    os.utime(stale, (old, old))
    with pytest.raises(r52.Incomplete, match="outside its run"):
        _cmp(tmp_path)


def test_run_without_logged_success_is_rule_unmet(tmp_path):
    _grid(tmp_path, lambda s, y: 150, lambda s, y: 150)
    _log(tmp_path, ok=False)
    with pytest.raises(r52.Incomplete, match="no successful attempt"):
        _cmp(tmp_path)


def test_later_attempt_supersedes_an_earlier_success(tmp_path):
    _grid(tmp_path, lambda s, y: 150, lambda s, y: 150)
    log = tmp_path / "progress.log"
    later = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log.write_text(log.read_text() + f"{later} start arm=new seed=1\n")  # started again, no end yet
    with pytest.raises(r52.Incomplete, match="no successful attempt"):
        _cmp(tmp_path)


def test_json_is_written_before_the_verdict_and_a_bad_path_gives_no_verdict(tmp_path, tmp_path_factory, capsys):
    _grid(tmp_path, lambda s, y: 150, lambda s, y: 150)
    elsewhere = tmp_path_factory.mktemp("results")  # outside the evidence tree
    out = elsewhere / "r.json"
    assert _main(tmp_path, "--json", str(out)) == 0
    assert json.loads(out.read_text())["verdict"] == "PASS"
    capsys.readouterr()
    assert _main(tmp_path, "--json", str(elsewhere)) == 2  # a directory: cannot be written
    assert "R52 verdict" not in capsys.readouterr().out


def test_mc_trials_must_be_positive(tmp_path):
    with pytest.raises(SystemExit):
        r52.main(["--root", str(tmp_path), "--mc-trials", "0"])


def test_exit_codes(tmp_path):
    _grid(tmp_path, lambda s, y: 150, lambda s, y: 150)
    assert _main(tmp_path) == 0
    other = tmp_path / "b"
    other.mkdir()
    _grid(other, lambda s, y: 150, lambda s, y: 140)
    assert _main(other) == 1


def test_both_arms_with_the_same_wrong_year_are_rule_unmet(tmp_path):
    # identical day sets in both arms, but the 2022 files hold 2023 dates: only the year check catches it
    _grid(tmp_path, lambda s, y: 150, lambda s, y: 150)
    for arm in r52.ARMS:
        _write(tmp_path, arm, 2, 2022, 150, year=2023)
    _log(tmp_path)
    with pytest.raises(r52.Incomplete, match="outside 2022"):
        _cmp(tmp_path)


# --- round 2 and the follow-up (must still catch) -------------------------------------------------

def _set_log(root, lines):
    (root / "progress.log").write_text("\n".join(lines) + "\n")


def _stamps():
    now = datetime.now()
    return ((now - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S"),
            (now + timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S"))


def test_file_replaced_after_done_is_rule_unmet(tmp_path):
    _grid(tmp_path, lambda s, y: 150, lambda s, y: 150)
    path = tmp_path / "out" / "new" / "seed_1" / "backtest_2021.parquet"
    later = (datetime.now() + timedelta(seconds=30)).timestamp()
    os.utime(path, (later, later))  # still inside the logged window, but after DONE
    with pytest.raises(r52.Incomplete, match="DONE"):
        _cmp(tmp_path)


def test_same_days_cut_from_both_arms_of_one_seed_is_rule_unmet(tmp_path):
    _grid(tmp_path, lambda s, y: 150, lambda s, y: 150)
    for arm in r52.ARMS:
        _write(tmp_path, arm, 3, 2021, 150, days=150)
    _log(tmp_path)
    with pytest.raises(r52.Incomplete, match="covers 150 days"):
        _cmp(tmp_path)


def test_success_line_after_a_failed_end_is_rule_unmet(tmp_path):
    _grid(tmp_path, lambda s, y: 150, lambda s, y: 150)
    lines = (tmp_path / "progress.log").read_text().splitlines()
    lines[1] = lines[1].replace("rc=0", "rc=1")
    lines.insert(2, lines[1].replace("rc=1", "rc=0"))
    _set_log(tmp_path, lines)
    with pytest.raises(r52.Incomplete, match="no open start"):
        _cmp(tmp_path)


def test_two_different_runs_open_at_once_is_rule_unmet(tmp_path):
    _grid(tmp_path, lambda s, y: 150, lambda s, y: 150)
    t0, t1 = _stamps()
    _set_log(tmp_path, [f"{t0} start arm=old seed=1", f"{t0} start arm=new seed=1",
                        f"{t1} end   arm=new seed=1 rc=0", f"{t1} end   arm=old seed=1 rc=0"])
    with pytest.raises(r52.Incomplete, match="is open"):
        _cmp(tmp_path)


def test_restart_of_the_same_unfinished_run_is_accepted(tmp_path):
    # the queue's real history: a run killed without an end, then started again and finished
    _grid(tmp_path, lambda s, y: 150, lambda s, y: 150)
    earlier = (BASE - timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S")
    lines = (tmp_path / "progress.log").read_text().splitlines()
    _set_log(tmp_path, [f"{earlier} start arm=old seed=1", "a note line that is not a record"] + lines)
    assert _cmp(tmp_path)["verdict"] == "PASS"


def test_time_going_backwards_is_rule_unmet(tmp_path):
    _grid(tmp_path, lambda s, y: 150, lambda s, y: 150)
    lines = (tmp_path / "progress.log").read_text().splitlines()
    _set_log(tmp_path, lines + ["2020-01-01 00:00:00 start arm=old seed=9"])
    with pytest.raises(r52.Incomplete, match="backwards"):
        _cmp(tmp_path)


@pytest.mark.parametrize("bad", [b"\xff\xfe", b"2026-11-32 22:00:00 start arm=old seed=1\n"])
def test_unreadable_log_is_rule_unmet(tmp_path, bad):
    _grid(tmp_path, lambda s, y: 150, lambda s, y: 150)
    (tmp_path / "progress.log").write_bytes(bad)
    assert _main(tmp_path) == 2


@pytest.mark.parametrize("rank", ["string", "inf", "bool"])
def test_non_numeric_or_infinite_ranks_are_rule_unmet(tmp_path, rank):
    _grid(tmp_path, lambda s, y: 150, lambda s, y: 150)
    path = tmp_path / "out" / "old" / "seed_2" / "backtest_2022.parquet"
    df = pd.read_parquet(path)
    if rank == "string":
        df["rank"] = df["rank"].astype(str)
    elif rank == "inf":
        df["rank"] = df["rank"].astype(float)
        df.loc[0, "rank"] = float("inf")
    else:
        df["rank"] = df["rank"] == 1
    df.to_parquet(path)
    _log(tmp_path)
    assert _main(tmp_path) == 2


def test_json_inside_the_evidence_tree_is_refused_and_inputs_untouched(tmp_path):
    _grid(tmp_path, lambda s, y: 150, lambda s, y: 150)
    target = tmp_path / "out" / "old" / "seed_1" / "backtest_2021.parquet"
    before = target.read_bytes()
    assert _main(tmp_path, "--json", str(target)) == 2
    assert _main(tmp_path, "--json", str(tmp_path / "progress.log")) == 2
    assert target.read_bytes() == before
    assert _main(tmp_path) == 0


def test_result_carries_the_digest_of_the_bytes_scored(tmp_path):
    import hashlib
    _grid(tmp_path, lambda s, y: 150, lambda s, y: 150)
    res = _cmp(tmp_path)
    path = tmp_path / "out" / "new" / "seed_4" / "backtest_2022.parquet"
    cell = next(c for c in res["cells"] if c["seed"] == 4 and c["season"] == 2022)
    assert cell["new_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_missing_streak_metric_is_labelled_unavailable():
    assert r52._metric(None).startswith("unavailable")
    assert r52._metric(float("nan")).startswith("unavailable")
    assert r52._metric(0.0123) == 0.0123


def test_done_marker_touched_after_its_run_is_rule_unmet(tmp_path):
    _grid(tmp_path, lambda s, y: 150, lambda s, y: 150)
    (tmp_path / "out" / "old" / "seed_4" / "DONE").touch()  # now: long after the logged end
    with pytest.raises(r52.Incomplete, match="DONE written"):
        _cmp(tmp_path)
