"""Run the supplied round-one fixtures against the round-two revisions."""

from collections import Counter
import json
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "round1"))
import probe_support
from probe_support import BASE, archive, offline, python, scratch

HEAD = "aa61bb43"
probe_support.REPO = HERE.parents[1]
probe_support.HERE = HERE


def carbon(root):
    import probe_corpus_and_fold as core

    specs = core._build_deterministic_cases() + core._build_seeded_cases()
    requests = []
    for index, spec in enumerate(specs):
        directory = root / str(index)
        directory.mkdir()
        spec.builder(directory)
        requests.append(dict(picks_dir=str(directory), season=spec.season, today=spec.today))
    with (root / "requests.json").open("w") as handle:
        json.dump(requests, handle)
    return specs, requests


def carbon_oracle(req):
    from datetime import date
    import re

    root = Path(req["picks_dir"])
    limit = req["today"]
    prefix = str(req["season"])
    for path in sorted(root.glob(f"{prefix}-*.json")):
        day = path.stem
        if day > limit or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
            continue
        date.fromisoformat(day)
        pick = json.loads(path.read_text())
        if pick.get("result") in ("hit", "miss", "void"):
            continue
        record = root / day / "decision.json"
        if record.exists() and json.loads(record.read_text())["scoreable"]:
            if day < limit:
                return None
    # Fold terminal outcomes directly from the chronology.
    total, saver = 0, True
    for path in sorted(root.glob(f"{prefix}-*.json")):
        if path.stem > limit or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", path.stem):
            continue
        pick = json.loads(path.read_text())
        if pick.get("result") == "hit":
            slots = pick.get("slot_results")
            total += (sum(result == "hit" for result in slots.values()) if slots is not None
                      else (2 if pick.get("double_down") else 1))
        elif pick.get("result") == "miss":
            if 10 <= total <= 15 and saver:
                saver = False
            else:
                total = 0
    return [total, saver]


def codex(root):
    import corpus

    src = "src/bts/state/regenerate.py"
    path = archive("11059dd1", root / "legacy", src) / src
    legacy = ModuleType("early_regenerate")
    sys.modules[legacy.__name__] = legacy
    exec(compile(path.read_text(), str(path), "exec"), legacy.__dict__)
    histories = root / "histories"
    manifest = list(corpus.build(histories, legacy))
    outcomes = []
    for revision in (BASE, HEAD):
        source = archive(revision, root / revision, "src") / "src"
        result = python(source, corpus.__file__, "--worker", histories, timeout=35)
        print(result.stderr.strip())
        outcomes.append(json.loads(result.stdout))
    counts = Counter()
    with (HERE / "corpus-differences.jsonl").open("w") as ledger:
        for name, case, _ in manifest:
            base, head = (value[name] for value in outcomes)
            if isinstance(head, list):
                oracle = carbon_oracle(dict(picks_dir=str(histories / name), season=2026, today=corpus.TODAY))
                assert oracle == head, (name, oracle, head)
                counts["oracle verified"] += 1
            else:
                oracle = None
            if base == head:
                counts["equal"] += 1
                continue
            status = lambda v: "value" if isinstance(v, list) else "unavailable"
            counts[f"{status(base)} -> {status(head)}"] += 1
            print(json.dumps(dict(history=name, shape=case.name, origin=case.origin,
                                  base=base, head=head, oracle=oracle)), file=ledger)
    print(json.dumps(counts))


def worker(mode, path):
    with offline():
        if mode == "deliveries":
            import delivery_followup as followup

            probe = followup.delivery_probe
            probe.manual_run_commit_original = probe.manual_run_commit
            probe.manual_run_commit = followup.corrected_manual_commit
            probe.scenario_runs(path)
        elif mode == "directories":
            import probe_blocker_glob_and_parent as probe
            from bts import picks

            print(json.dumps(probe._run_suite("round2")))
            if hasattr(picks, "_fold_season"):
                with patch.object(picks, "_fold_season", probe._fixed_fold_season):
                    print(json.dumps(probe._run_suite("fix")))
        elif mode == "carbon":
            from bts import picks

            outcomes = []
            for req in json.loads(path.read_text()):
                try:
                    result = picks._replay_season_streak(Path(req["picks_dir"]), req["season"], req["today"])
                except Exception as error:
                    result = {"raised": type(error).__name__}
                outcomes.append(result)
            print(json.dumps(outcomes))


def main(mode):
    with scratch(mode) as tmp:
        root = Path(tmp)
        if mode == "codex":
            codex(root)
            return
        if mode == "carbon":
            specs, requests = carbon(root)
        observations = {}
        for revision in (BASE, HEAD):
            src = archive(revision, root / revision, "src") / "src"
            arg = root / ("requests.json" if mode == "carbon" else f"data-{revision}")
            result = python(src, __file__, "worker", mode, arg, timeout=45)
            if mode == "carbon":
                observations[revision] = json.loads(result.stdout)
            else:
                print(revision, result.stdout, sep="\n", end="")
        if mode == "carbon":
            counts = Counter()
            for spec, req, base, head in zip(specs, requests, observations[BASE], observations[HEAD], strict=True):
                transition = lambda value: "value" if isinstance(value, list) else "unavailable"
                counts["equal" if base == head else f"{transition(base)} -> {transition(head)}"] += 1
                if base != head:
                    oracle = carbon_oracle(req) if not isinstance(base, list) and isinstance(head, list) else None
                    print(json.dumps(dict(case=spec.name, base=base, head=head, oracle=oracle, note=spec.note)))
            print(json.dumps(counts))


if __name__ == "__main__":
    if sys.argv[1] == "worker":
        worker(sys.argv[2], Path(sys.argv[3]))
    else:
        main(sys.argv[1])
