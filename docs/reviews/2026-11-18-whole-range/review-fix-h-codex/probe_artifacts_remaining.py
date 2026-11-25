"""Complete the loader checks with a validated two-bin fixture."""
from dataclasses import replace
from datetime import date, datetime, time
from hashlib import sha256
import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from bts import strategy
from bts.health import tail_policy as health
from bts.simulate import mdp, tail_policy as tp
from probe_artifacts import rewrite, select
from scripts.audit.season_walk import walk


def main():
    shipped = strategy._load_mdp()
    tail = shipped["tail"]
    day = date(2027, 9, 26)
    now = datetime.combine(day, time(12), walk.ET)
    expected = {
        "tail-two-bin": "one quality bin",
        "tail-malformed-manifest": "TypeError",
        "base-missing": "FileNotFoundError",
        "base-corrupt": "ValueError",
        "base-directory": "IsADirectoryError",
        "base-missing-table": "KeyError",
        "base-oversize": "too large",
        "base-zero": "season length must be positive",
        "base-negative": "season length must be positive",
        "both-absent": "sha unavailable",
        "loader-error": "RuntimeError",
        "rule-error": "simulated rule error",
        "matched-shapeless-table": None,
    }
    for name, reason in expected.items():
        with TemporaryDirectory() as temp:
            root = Path(temp)
            base, tail_path = root / "base.npz", root / "tail.npz"
            copyfile(mdp.DEFAULT_POLICY_PATH, base)
            copyfile(tp.DEFAULT_TAIL_POLICY_PATH, tail_path)
            if name == "tail-two-bin":
                freq, hit, both = [.5, .5], [.75, .75], [.50, .50]
                tp.save_tail_policy(replace(
                    tail, boundaries=[.82], bin_freq=freq, bin_p_hit=hit, bin_p_both=both,
                    policy_table=tp.solve_emax_season_best(freq, hit, both).policy,
                    manifest=tp.tail_manifest(n_bins=2, hits=6, both=4, late_seed_days=8, bin_counts=[4, 4]),
                ), tail_path)
            elif name == "tail-malformed-manifest":
                rewrite(tail_path, manifest=np.array(json.dumps(tail.manifest | {"bin_counts": [None]})))
            elif name in {"base-missing", "both-absent"}:
                base.unlink()
                if name == "both-absent":
                    tail_path.unlink()
            elif name == "base-corrupt":
                base.write_bytes(b"not an NPZ")
            elif name == "base-directory":
                base.unlink()
                base.mkdir()
            elif name == "base-missing-table":
                np.savez(base, season_length=np.array(180), boundaries=np.array([]))
            elif name == "base-oversize":
                with base.open("r+b") as stream:
                    stream.truncate(tp.MAX_ARTIFACT_BYTES + 1)
            elif name in {"base-zero", "base-negative", "matched-shapeless-table"}:
                if name == "matched-shapeless-table":
                    rewrite(base, policy_table=np.array(0, dtype=np.int8))
                else:
                    rewrite(base, season_length=np.array(0 if name == "base-zero" else -1))
                tp.save_tail_policy(replace(tail, base_policy_sha256=sha256(base.read_bytes()).hexdigest()),
                                    tail_path)
            with (walk.offline(now), patch.object(mdp, "DEFAULT_POLICY_PATH", base),
                  patch.object(tp, "DEFAULT_TAIL_POLICY_PATH", tail_path),
                  patch.object(strategy, "_mdp_cache", None)):
                assert strategy.effective_pick_bar(tp.LONGEST_STREAK_FROM, str(day), True) == 0.0
                with (patch.object(strategy, "_load_mdp", wraps=strategy._load_mdp) as loader,
                      patch.object(strategy, "longest_streak_action",
                                   wraps=strategy.longest_streak_action) as rule):
                    if name == "loader-error":
                        loader.side_effect = RuntimeError("artificial reader failure")
                    if name == "rule-error":
                        rule.side_effect = ArithmeticError("simulated rule error")
                    result = select(root, day, tp.LONGEST_STREAK_FROM)
                d = result.decision
                assert d.objective == "longest_streak" and d.source == "mdp" and d.days_effective == 1
                if reason:
                    assert result.action == d.policy_action == "single" and d.tail_sha256 is None
                    assert d.degraded_reason.startswith("longest-streak rule unavailable (")
                    assert d.degraded_reason.endswith("); forced single")
                    assert reason.lower() in d.degraded_reason.lower(), (name, reason, d.degraded_reason)
                    print(name, d.degraded_reason)
                else:
                    loaded = strategy._load_mdp()
                    assert loaded["policy_table"].shape == () and loaded["base_error"] is None
                    assert loaded["base_sha256"] == loaded["tail"].base_policy_sha256
                    assert health.check(base_path=base, tail_path=tail_path, today=day) == []
                    assert result.action == d.policy_action == "double" and d.tail_sha256 and not d.degraded_reason
                    print(name, "shape", loaded["policy_table"].shape, "health alerts 0",
                          "committed choice", result.action, "sha", d.tail_sha256)
                if name == "tail-malformed-manifest":
                    try:
                        health.check(base_path=base, tail_path=tail_path, today=day)
                    except TypeError:
                        print("malformed manifest raises from artifact health, as A9 says")
                    else:
                        raise AssertionError("expected artifact health TypeError")
    for error in (KeyboardInterrupt, SystemExit):
        for seam in ("_load_mdp", "longest_streak_action"):
            with TemporaryDirectory() as temp, walk.offline(now), patch.object(
                    strategy, seam, side_effect=error("interrupted")):
                try:
                    select(Path(temp), day, tp.LONGEST_STREAK_FROM)
                except error as exc:
                    assert str(exc) == "interrupted"
                    print(seam, type(exc).__name__, "propagated")
                else:
                    raise AssertionError((seam, error))


if __name__ == "__main__":
    main()
