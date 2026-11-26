"""Check a healthy short table and a SHA-matched broken table through delivery."""
from dataclasses import replace
from datetime import date, timedelta
from hashlib import sha256
from itertools import product
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from bts import picks, scheduler, strategy
from bts.health import tail_policy as health
from bts.simulate import mdp, tail_policy as tp
from probe_delivery import checked_record, deliver, world


def run(expect_scalar_fallback=False):
    shipped = strategy._load_mdp()
    tail = shipped["tail"]
    with TemporaryDirectory() as temp:
        parent = Path(temp)
        for typ, day in (("short", date(2027, 9, 23)), ("scalar", date(2027, 9, 26))):
            base_path = parent / f"{typ}-base.npz"
            tail_path = parent / f"{typ}-tail.npz"
            with np.load(mdp.DEFAULT_POLICY_PATH) as fixture:
                bundle = dict(fixture)
            if typ == "short":
                table = bundle["policy_table"][:, :2]
                horizon = 1
            else:
                table = np.array(0, dtype=np.int8)
                horizon = 180
            np.savez_compressed(base_path, **(bundle | {"policy_table": table, "season_length": np.array(horizon)}))
            base_sha = sha256(base_path.read_bytes()).hexdigest()
            tp.save_tail_policy(replace(tail, base_policy_sha256=base_sha), tail_path)
            paired = tp.load_tail_policy(tail_path, expected_base_sha=base_sha)
            assert health.check(base_path=base_path, tail_path=tail_path, today=day) == []
            folder = parent / typ
            with (world(folder, day, "dated") as (cfg, frame, statuses),
                  patch.object(mdp, "DEFAULT_POLICY_PATH", base_path),
                  patch.object(tp, "DEFAULT_TAIL_POLICY_PATH", tail_path),
                  patch.object(strategy, "_mdp_cache", None)):
                text = str(day)
                result = scheduler.run_single_check(text, list(statuses), set(), cfg, .01)
                saved = picks.load_pick(text, folder)
                with patch("bts.dm.send_dm", return_value="captured-pair") as send:
                    assert deliver(saved, cfg, folder, result["selection"])
                    message = send.call_args.args[1]
                rec = checked_record(folder, text, saved, "delivered")
                if typ == "short":
                    assert rec["tail_policy_sha256"] == paired.sha256 and rec["degraded_reason"] is None
                    assert saved.policy_decision["days_effective"] == 4
                    assert rec["action"] == "single" and saved.double_down is None
                else:
                    assert saved.policy_decision["days_effective"] == 1
                    if expect_scalar_fallback:
                        assert rec["action"] == "single" and saved.double_down is None
                        assert rec["tail_policy_sha256"] is None and "forced single" in rec["degraded_reason"]
                        assert "Batter 0" in message and "Batter 1" not in message
                    else:
                        assert rec["tail_policy_sha256"] == paired.sha256 and rec["degraded_reason"] is None
                        assert rec["action"] == "double" and saved.double_down.batter_id == 101
                        assert "Batter 0" in message and "Batter 1" in message
                    try:
                        mdp.lookup_action_guarded(table, bundle["boundaries"].tolist(), 56, 1, False, .84, horizon)
                    except IndexError:
                        print("scalar array cannot supply an actual reach-57 policy lookup")
                    else:
                        raise AssertionError("expected scalar base table to be unusable")
                print(typ, text, "base table shape", table.shape, "state 57/57/used",
                      "days", saved.policy_decision["days_effective"], "delivered", rec["action"],
                      "slots", rec["primary"]["batter_name"],
                      rec["double_down"]["batter_name"] if saved.double_down else None,
                      "decision tail sha", rec["tail_policy_sha256"])
    count = 0
    for raw, streak, saver, prob in product(range(1, 272), (57, 58, 60, 100), (False, True), (.01, .51, .99)):
        text = str(date(2028, 9, 28) - timedelta(days=raw))
        for allow, partner in ((True, True), (False, True), (True, False)):
            ctx = strategy.DecisionContext(prob, .91 if partner else None, partner, text, allow,
                                           shipped, 57, "trusted")
            decision = strategy.resolve_policy_decision(ctx, streak, saver)
            policy = "double" if raw <= 3 else "single"
            want = policy if allow and partner else "single"
            assert (decision.policy_action, decision.action, decision.tail_sha256,
                    decision.days_effective) == (policy, want, tail.sha256, min(raw, 180))
            count += 1
    print("H1–H3", count, "resolutions over every positive raw day 1–271")


if __name__ == "__main__":
    run("--fixed" in sys.argv)
