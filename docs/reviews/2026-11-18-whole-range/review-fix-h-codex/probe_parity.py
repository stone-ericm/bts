"""Compare the archived source and reviewed source with the same cases."""
import csv
from collections import Counter
from dataclasses import asdict, replace
from datetime import date, timedelta
from functools import cache
from hashlib import sha256
from itertools import product
import json
import logging
from math import nextafter
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def worker(expected_src, rollback_dir, archived):
    import bts
    from bts import daily_decision, strategy
    from bts.simulate import tail_policy as tp

    print(bts.__file__, flush=True)
    assert Path(bts.__file__).resolve() == Path(expected_src) / "bts" / "__init__.py"
    record = daily_decision.load_decision("2027-08-02", rollback_dir)
    assert record["objective"] == "longest_streak"
    assert daily_decision.decision_objective(record) == ("unknown" if archived else "longest_streak")
    assert daily_decision.is_scoreable_commit("2027-08-02", rollback_dir, None)
    if archived:
        assert daily_decision.write_decision(
            "2027-08-03", rollback_dir, action="single", source="mdp", objective="longest_streak",
            delivery_status="delivered", scoreable=True) is None
    print("rollback" if archived else "reviewed", daily_decision.decision_objective(record), "scoreable", flush=True)
    logging.disable(logging.CRITICAL)
    raw_days = cache(strategy._days_remaining)
    strategy._days_remaining = raw_days
    policy = strategy._load_mdp()
    tail = policy["tail"]
    freq, hit, both = [.5, .5], [.75, .75], [.5, .5]
    two = replace(tail, boundaries=[.87], bin_freq=freq, bin_p_hit=hit, bin_p_both=both,
                  policy_table=tp.solve_emax_season_best(freq, hit, both).policy,
                  manifest=tp.tail_manifest(n_bins=2, hits=6, both=4, late_seed_days=8, bin_counts=[4, 4]))
    tp.validate_tail_policy(two, expected_base_sha=policy["base_sha256"])
    modes = (policy, None, policy | {"tail": None}, policy | {"policy_table": None},
             policy | {"tail": replace(tail, base_policy_sha256="b" * 64)},
             policy | {"policy_table": None, "base_sha256": None, "tail": None, "base_error": "bad"},
             policy | {"tail": two}, policy | {"season_length": 3}, policy | {"season_length": 240},
             policy | {"season_length": 0}, policy | {"season_length": -1})
    probabilities = sorted({.01, .99, *(p for edge in (*policy["boundaries"], .80, .87)
                                        for p in (nextafter(edge, 0), edge, nextafter(edge, 1)))})
    bests = [(None, None), *product(range(58), ("trusted", "untrusted"))]
    digests = {}
    counts = Counter()
    objectives = Counter()
    raw_seen, effective_seen = set(), set()

    def check(group, d, s, saver, p, p2, best, status, allow, mode):
        ctx = strategy.DecisionContext(p, p2, p2 is not None, d, allow, mode, best, status)
        dec = strategy.resolve_policy_decision(ctx, s, saver)
        bar = strategy.effective_pick_bar(s, d, saver, mode, best, status)
        raw = raw_days(d)
        raw_seen.add(raw)
        effective_seen.add(dec.days_effective)
        digests.setdefault(group, sha256()).update(
            json.dumps([d, raw, s, saver, p, p2, best, status, allow, asdict(dec), bar], sort_keys=True).encode())
        counts[group] += 1
        objectives[dec.objective] += 1
        return dec

    for raw, s, saver in product(range(272), range(57), (False, True)):
        d = str(date(2028, 9, 28) - timedelta(days=raw))
        for i, p in enumerate(probabilities):
            best, status = bests[(raw * 7 + s * 13 + i + saver) % len(bests)]
            check("shipped", d, s, saver, p, .88 if i % 3 else None, best, status, bool(i % 2), policy)
    for mode, raw, (best, status), s, saver in product(modes, (0, 1, 5, 28, 180, 181, 271),
                                                      bests, (0, 17, 56), (False, True)):
        d = str(date(2028, 9, 28) - timedelta(days=raw))
        check("artifact", d, s, saver, .87, .81, best, status, True, mode)
    for mode, day, s, saver, p, allow in product(
            modes, ("2026-09-28", "2027-09-27", "2027-12-31", "2028-09-28", "2029-12-31"),
            (0, 56, 57, 58, 100, 10**60), (False, True), probabilities[::4], (False, True)):
        check("zero-days", day, s, saver, p, .98, 57, "trusted", allow, mode)
    for mode, day, s, saver, p in product(
            modes, ("2026-03-30", "2026-09-27", "2027-03-31", "2027-09-26",
                    "2029-01-01", "2029-09-27"), (0, 26, 55, 56), (False, True), (.1, .99)):
        check("calendar", day, s, saver, p, .78, 56, "trusted", True, mode)
    for s, cutoff in ((0, .55), (9, .55), (10, .60), (15, .60), (16, .65), (45, .65), (46, .65)):
        for p, p2, allow in product((nextafter(.80, 0), .80, nextafter(.80, 1), 1.0),
                                   (None, nextafter(cutoff, 0), cutoff, nextafter(cutoff, 1)), (False, True)):
            check("heuristic", "2027-08-02", s, True, p, p2, None, None, allow, None)
    with (REPO / "docs/reviews/2026-11-18-whole-range/contract-g/affected-states.csv").open() as stream:
        for row in csv.DictReader(stream):
            s, raw, saver, bin_index = (int(row[f]) for f in ("streak", "days_left", "saver", "bin"))
            d = str(date(2027, 9, 27) - timedelta(days=raw))
            dec = check("G guards", d, s, bool(saver), policy["boundaries"][bin_index - 1],
                        .91, s, "trusted", True, policy)
            assert dec.guard and dec.action == row["guarded_action"], row
    with patch.object(strategy, "_load_mdp", side_effect=ValueError("loader failed")):
        if archived:
            try:
                strategy.effective_pick_bar(56, "2027-08-02", True)
            except ValueError:
                print("legacy display exception propagated on base", flush=True)
            else:
                raise AssertionError("expected base to raise")
        else:
            assert strategy.effective_pick_bar(56, "2027-08-02", True) is None
            print("legacy display exception returns None on reviewed source", flush=True)
    for s, d in product((-1, False, True),
                        ("2027-09-27", "2027-09-26", "2027-09-24", "2027-09-23",
                         "2028-04-01", "2028-03-31", "2028-01-01")):
        try:
            check("invalid-streak", d, s, True, .96, .95, None, None, True, policy)
        except (IndexError, TypeError) as exc:
            bar = strategy.effective_pick_bar(s, d, True, policy)
            digests.setdefault("invalid-streak", sha256()).update(repr((s, d, type(exc).__name__, str(exc),
                                                                         bar)).encode())
            counts["invalid-streak"] += 1
            print("invalid streak", repr(s), d, type(exc).__name__, str(exc), "bar", bar, flush=True)
    print(json.dumps(dict(groups={k: [counts[k], digests[k].hexdigest()] for k in digests},
                          objectives=objectives, raw_days=[min(raw_seen), max(raw_seen)],
                          effective_days=[min(effective_seen), max(effective_seen)]), sort_keys=True))


def main():
    from bts import daily_decision

    with TemporaryDirectory(prefix="archive-", dir=HERE) as temp:
        root = Path(temp)
        archive = subprocess.run(["git", "archive", "0a2bec10", "src"], cwd=REPO,
                                 capture_output=True, check=True, timeout=5).stdout
        subprocess.run(["tar", "-xf", "-", "-C", root], input=archive, check=True, timeout=5)
        assert daily_decision.write_decision(
            "2027-08-02", root, action="single", source="mdp", objective="longest_streak",
            delivery_status="delivered", scoreable=True)
        results = []
        for path in (root / "src", REPO / "src"):
            result = subprocess.run(
                [sys.executable, __file__, "--worker", str(path), str(root), str(int(path.is_relative_to(root)))],
                env=os.environ | {"PYTHONPATH": str(path)}, cwd=REPO, capture_output=True,
                text=True, timeout=42)
            print(result.stdout, end="", flush=True)
            if result.returncode:
                raise RuntimeError(result.stderr)
            results.append(json.loads(result.stdout.splitlines()[-1]))
        assert results[0] == results[1]
        print("all decision fields, bars and group digests match")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        worker(sys.argv[2], Path(sys.argv[3]), bool(int(sys.argv[4])))
    else:
        main()
