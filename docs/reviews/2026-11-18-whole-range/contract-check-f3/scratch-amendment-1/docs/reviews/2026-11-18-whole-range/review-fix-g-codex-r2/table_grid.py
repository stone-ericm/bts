"""Compare the separate base and head imports, using the contract CSV as oracle."""
import csv
from dataclasses import asdict
from datetime import date, timedelta
from functools import cache
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONTRACT = ROOT / "docs/reviews/2026-11-18-whole-range/contract-g"
BASE = HERE / "base/src"
HEAD = ROOT / "src"


def sample(side):
    import bts
    from bts import strategy
    from bts.simulate.mdp import lookup_action

    source = HEAD if side == "head" else BASE
    assert Path(bts.__file__).resolve().is_relative_to(source)
    print(side, bts.__file__, flush=True)
    artifact = strategy._load_mdp()
    assert artifact["base_sha256"].startswith("66d15471") and artifact["tail"] is not None
    table = artifact["policy_table"]
    edges = [0.0, *artifact["boundaries"], 1.0]
    probabilities = [
        (quality, p)
        for quality, (lower, upper) in enumerate(zip(edges, edges[1:]))
        for p in [lower, (lower + upper) / 2, math.nextafter(upper, -math.inf)]
    ]
    probabilities.extend([(0, -0.1), (0, math.nan), (4, 1.0), (4, 1.1)])
    expected = {}
    with (CONTRACT / "affected-states.csv").open() as csv_file:
        for row in csv.DictReader(csv_file):
            key = tuple(int(row[name]) for name in ("streak", "days_left", "saver", "bin"))
            assert key not in expected and row["table_action"] == "skip"
            expected[key] = (row["guarded_action"], {
                "rule": "monotone_skip_v1", "bin": key[-1],
                "from_bin": int(row["from_bin"]), "table_action": "skip",
            })
    assert len(expected) == 1511
    if side == "head":
        from bts.simulate.mdp import lookup_action_guarded

    raw_actions = bytearray()
    changed = set()
    for s, d, v, (q, p) in product(range(58), range(181), (False, True), probabilities):
        args = table, artifact["boundaries"], s, d, v, p
        raw = lookup_action(*args)
        raw_actions.append({"skip": 0, "single": 1, "double": 2}[raw])
        if side == "head":
            key = (s, d, int(v), q)
            guarded = lookup_action_guarded(*args)
            assert guarded == expected.get(key, (raw, None)), (key, p, guarded, raw)
            if guarded[1] is not None:
                assert raw == "skip" and guarded[0] != raw
                check_types(guarded[1])
                changed.add(key)
    if side == "head":
        assert changed == expected.keys()

    strategy._days_remaining = cache(strategy._days_remaining)
    start, end = date(2027, 3, 24), date(2027, 9, 27)
    dates = [(start + timedelta(days=i)).isoformat() for i in range((end - start).days + 1)]
    shared_digest = hashlib.sha256()
    bar_values = []
    dated_count = dated_changed = 0
    covered = set()
    for day, s, v in product(dates, range(58), (False, True)):
        bar = strategy.effective_pick_bar(s, day, v, artifact, s, "trusted")
        bar_values.append(bar)
        decisions = []
        for q, p in probabilities:
            context = strategy.DecisionContext(p, 0.95, True, day, True, artifact, s, "trusted")
            decision = strategy.resolve_policy_decision(context, s, v)
            fields = vars(decision).copy()
            guard = fields.pop("guard", None)
            key = (s, decision.days_effective, int(v), q)
            affected = expected.get(key) if decision.objective == "reach57" else None
            if side == "head" and affected:
                assert (decision.action, guard) == affected
                assert decision.policy_action == decision.action and decision.pick_bar is None
                check_types(guard)
                assert json.loads(json.dumps(asdict(decision)))["guard"] == affected[1]
                fields.update(action="skip", policy_action="skip", pick_bar=bar)
                dated_changed += 1
                covered.add(key)
            else:
                assert guard is None
                if affected:
                    assert decision.action == decision.policy_action == "skip"
                    assert decision.pick_bar == bar
            if side == "head" and decision.objective == "reach57":
                if bar is None:
                    assert decision.action == "skip"
                elif p >= bar:
                    assert decision.action != "skip"
                    if p == bar:
                        assert guard is None
            decisions.append(fields)
            dated_count += 1
        shared_digest.update(json.dumps(decisions, sort_keys=True).encode())
    if side == "head":
        assert covered == expected.keys() and dated_changed == 7125
    assert (len(raw_actions), dated_count, len(bar_values)) == (398924, 414352, 21808)
    print(f"{len(raw_actions)} raw, {dated_count} dated, {len(bar_values)} bars, "
          f"{len(changed)} changed table cells, {dated_changed} changed dated calls")
    (HERE / f"{side}-grid.json").write_text(json.dumps({
        "raw": hashlib.sha256(raw_actions).hexdigest(),
        "shared": shared_digest.hexdigest(),
        "bars": bar_values,
    }))


def check_types(guard):
    assert type(guard) is dict
    assert all(type(key) is str for key in guard)
    assert all(type(guard[key]) is int for key in ("bin", "from_bin"))
    assert all(type(guard[key]) is str for key in ("rule", "table_action"))


if __name__ == "__main__":
    if len(sys.argv) == 2:
        sample(sys.argv[1])
    else:
        for label, src in [("base", BASE), ("head", HEAD)]:
            subprocess.run([sys.executable, __file__, label], cwd=ROOT, check=True,
                           timeout=45, env=os.environ | {"PYTHONPATH": str(src)})
        assert (HERE / "base-grid.json").read_bytes() == (HERE / "head-grid.json").read_bytes()
        output = HERE / "affected-states.csv"
        subprocess.run([sys.executable, str(CONTRACT / "enumerate_states.py"), str(ROOT),
                        str(output)], cwd=ROOT, check=True, timeout=10)
        assert output.read_bytes() == (CONTRACT / output.name).read_bytes()
        print("Raw lookups, unaffected decision fields, and bars agree. CSV matches byte for byte.")
