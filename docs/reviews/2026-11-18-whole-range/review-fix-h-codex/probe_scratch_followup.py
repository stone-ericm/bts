"""Reproduce the two missed changes and verify a minimal fix in scratch."""
from pathlib import Path
from shutil import copytree
import os
import subprocess
import sys
from tempfile import TemporaryDirectory

from probe_scratch import HERE, MUTATIONS, REPO


def ordering(kind):
    from datetime import date
    from unittest.mock import patch

    import bts
    from bts import strategy
    from probe_artifacts import select
    from probe_delivery import clock
    from scripts.audit.season_walk import walk

    print(bts.__file__, flush=True)
    policy = strategy._load_mdp()
    events = []
    regime = strategy._regime

    def dispatch(mdp, streak, text):
        events.append("dispatch")
        return regime(mdp, streak, text)

    def load():
        events.append("load")
        return policy

    day = date(2027, 9, 26)
    with (TemporaryDirectory() as temp, walk.offline(clock(day)),
          patch.object(strategy, "_regime", side_effect=dispatch),
          patch.object(strategy, "_load_mdp", side_effect=load)):
        decision = select(Path(temp), day, 57).decision
        assert decision.action == "double"
    print(kind, events, flush=True)
    assert events[0] == ("load" if kind == "read-before-regime" else "dispatch")
    if kind in {"original", "use-base-horizon"}:
        short = policy | {"season_length": 1, "policy_table": policy["policy_table"][:, :2]}
        with (TemporaryDirectory() as temp, walk.offline(clock(date(2027, 9, 23))),
              patch.object(strategy, "_load_mdp", return_value=short)):
            decision = select(Path(temp), date(2027, 9, 23), 57).decision
        expected = (1, "double") if kind == "use-base-horizon" else (4, "single")
        assert (decision.days_effective, decision.action) == expected
        print(kind, "short valid base, raw 4, effective", decision.days_effective,
              "action", decision.action, flush=True)


def run(kind, source):
    if kind == "fix-scalar":
        args = [str(HERE / "probe_paired_base.py"), "--fixed"]
    else:
        args = [__file__, "--worker", kind]
    result = subprocess.run([sys.executable, *args], cwd=REPO,
                            env=os.environ | {"PYTHONPATH": f"{source}:{REPO}"},
                            capture_output=True, text=True, timeout=10)
    print(result.stdout, end="", flush=True)
    if result.returncode:
        print(result.stderr, flush=True)
        raise AssertionError((kind, result.returncode))


def main():
    run("original", REPO / "src")
    for kind in ("read-before-regime", "use-base-horizon", "fix-scalar"):
        with TemporaryDirectory(prefix=f"{kind}-", dir=HERE) as temp:
            source = Path(temp) / "src"
            copytree(REPO / "src", source)
            if kind == "fix-scalar":
                name = "strategy.py"
                old = '        if int(policy["season_length"]) <= 0:'
                new = ('        if policy["policy_table"].ndim != 4:\n'
                       '            raise ValueError("malformed base policy table shape")\n' + old)
            else:
                name, old, new = MUTATIONS[kind]
            file = source / "bts" / name
            content = file.read_text()
            assert content.count(old) == 1
            file.write_text(content.replace(old, new, 1))
            run(kind, source)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--worker"]:
        ordering(sys.argv[2])
    else:
        main()
