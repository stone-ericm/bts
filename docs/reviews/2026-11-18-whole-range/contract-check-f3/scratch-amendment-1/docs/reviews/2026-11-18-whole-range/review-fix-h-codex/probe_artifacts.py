"""Drive deliberately damaged on-disk policies through fresh selection."""
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
from bts.simulate import mdp, tail_policy as tp
from scripts.audit.season_walk import walk


def rewrite(path, **fields):
    with np.load(path) as archive:
        data = dict(archive)
    np.savez_compressed(path, **(data | fields))


def select(root, day, streak):
    frame = walk._predictions(day, .79).head(2)
    statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in frame.game_pk}
    return strategy.select_pick(frame, str(day), root, streak=streak,
                                saver_available=True, best_streak=57, best_status="trusted",
                                game_statuses_detailed=statuses)


def main():
    shipped = strategy._load_mdp()
    tail = shipped["tail"]
    day = date(2027, 9, 26)
    now = datetime.combine(day, time(12), walk.ET)
    both = {
        "tail-missing": "tail policy",
        "tail-corrupt": "tail policy",
        "tail-truncated": "tail policy",
        "tail-directory": "tail policy",
        "tail-wrong-schema": "schema",
        "tail-wrong-rates": "finite",
        "tail-oversize": "too large",
        "tail-unrelated": "sha256 mismatch",
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
    }
    for name, reason in both.items():
        with TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / "base.npz"
            tp_file = root / "tail.npz"
            copyfile(mdp.DEFAULT_POLICY_PATH, base)
            copyfile(tp.DEFAULT_TAIL_POLICY_PATH, tp_file)
            target = base if name.startswith("base") else tp_file
            suffix = name.split("-", 1)[1]
            if suffix == "missing":
                target.unlink()
            elif suffix == "corrupt":
                target.write_bytes(b"not an npz")
            elif suffix == "truncated":
                target.write_bytes(target.read_bytes()[: len(target.read_bytes()) // 2])
            elif suffix == "directory":
                target.unlink()
                target.mkdir()
            elif suffix == "wrong-schema":
                rewrite(target, schema_version=np.array("not_a_known_schema"))
            elif suffix == "wrong-rates":
                rewrite(target, bin_p_hit=np.array([float("nan")]))
            elif suffix == "oversize":
                with target.open("r+b") as stream:
                    stream.truncate(tp.MAX_ARTIFACT_BYTES + 1)
            elif suffix == "unrelated":
                tp.save_tail_policy(replace(tail, base_policy_sha256="a" * 64), target)
            elif suffix == "two-bin":
                freq, hits, doubles = [.25, .75], [.82, .95], [.50, .83]
                artifact = replace(
                    tail, boundaries=[.82], bin_freq=freq, bin_p_hit=hits, bin_p_both=doubles,
                    policy_table=tp.solve_emax_season_best(freq, hits, doubles).policy,
                    manifest=tp.tail_manifest(n_bins=2, hits=7, both=5, late_seed_days=8, bin_counts=[2, 6]),
                )
                tp.save_tail_policy(artifact, target)
            elif suffix == "malformed-manifest":
                rewrite(target, manifest=np.array(json.dumps(tail.manifest | {"bin_counts": [None]})))
            elif suffix == "missing-table":
                np.savez(target, boundaries=np.array([]), season_length=np.array(180))
            elif suffix in ("zero", "negative"):
                rewrite(target, season_length=np.array(0 if suffix == "zero" else -1))
                tp.save_tail_policy(replace(tail, base_policy_sha256=sha256(base.read_bytes()).hexdigest()),
                                    tp_file)
            elif suffix == "absent":
                base.unlink()
                tp_file.unlink()
            with (walk.offline(now), patch.object(mdp, "DEFAULT_POLICY_PATH", base),
                  patch.object(tp, "DEFAULT_TAIL_POLICY_PATH", tp_file),
                  patch.object(strategy, "_mdp_cache", None)):
                boundary = tp.LONGEST_STREAK_FROM
                with patch.object(strategy, "_load_mdp", wraps=strategy._load_mdp) as loader:
                    assert strategy.effective_pick_bar(boundary, str(day), True) == 0.0
                    loader.assert_not_called()
                    if name == "loader-error":
                        loader.side_effect = RuntimeError("artificial reader failure")
                    with patch.object(strategy, "longest_streak_action",
                                      wraps=strategy.longest_streak_action) as rule:
                        if name == "rule-error":
                            rule.side_effect = ArithmeticError("simulated rule error")
                        result = select(root, day, boundary)
                d = result.decision
                assert (result.action, d.policy_action, d.objective, d.source, d.tail_sha256,
                        d.days_effective) == ("single", "single", "longest_streak", "mdp", None, 1)
                assert result.pick_result.daily.double_down is None
                assert reason.lower() in d.degraded_reason.lower(), (name, reason, d.degraded_reason)
                assert d.degraded_reason.startswith("longest-streak rule unavailable (")
                assert d.degraded_reason.endswith("); forced single")
                print(name, d.degraded_reason)
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
