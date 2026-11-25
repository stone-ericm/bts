"""Item 7 probe: run the full 2027 season walk end to end and verify against walk-ea-g-h.json.gz and walk-h2.json.gz."""
from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent


def main() -> int:
    from scripts.audit.season_walk import walk

    out_json = OUT_DIR / "walk-h2-review.json"
    rc = walk.main(["--out", str(out_json)])
    fresh = json.loads(out_json.read_text(encoding="utf-8"))

    results_dir = REPO / "scripts/audit/season_walk/results"
    with gzip.open(results_dir / "walk-ea-g-h.json.gz", "rt", encoding="utf-8") as f:
        baseline = json.load(f)
    with gzip.open(results_dir / "walk-h2.json.gz", "rt", encoding="utf-8") as f:
        committed = json.load(f)

    problems = []
    if rc != 0:
        problems.append(f"walk.main returned {rc}")
    if fresh["problems"] != [] or walk.check(fresh["records"]) != []:
        problems.append(f"walk problems: {fresh['problems']}")
    if fresh["runs"] != 5037 or len(fresh["records"]) != 5037:
        problems.append(f"unexpected run count: {fresh['runs']}")

    high_label = "dated/58/60/used/0.96"
    existing = [r for r in fresh["records"] if r["state"] != high_label]
    added = [r for r in fresh["records"] if r["state"] == high_label]

    if len(existing) != 4818 or existing != baseline["records"]:
        problems.append(f"existing records differ from walk-ea-g-h.json.gz (len={len(existing)})")
    if fresh["boundary"] != baseline["boundary"]:
        problems.append("boundary differs from walk-ea-g-h.json.gz")
    if fresh != committed:
        problems.append("fresh walk output differs from committed walk-h2.json.gz")

    counts = Counter(r["action"] for r in added)
    doubles = [r["day"] for r in added if r["action"] == "double"]
    skips = [r["day"] for r in added if r["action"] == "skip"]
    if dict(counts) != {"single": 207, "double": 3, "skip": 9}:
        problems.append(f"unexpected action counts: {counts}")
    if doubles != ["2027-09-24", "2027-09-25", "2027-09-26"]:
        problems.append(f"unexpected double dates: {doubles}")
    if len(skips) != 9 or not all(d > "2027-09-26" for d in skips):
        problems.append(f"unexpected skip dates: {skips}")

    summary = (
        f"rc={rc} runs={fresh['runs']} existing={len(existing)}==baseline({ existing == baseline['records'] }) "
        f"matches_committed={fresh == committed} added_counts={dict(counts)} doubles={doubles} "
        f"skips={skips[0]}..{skips[-1]} problems={len(problems)}\n"
    )
    (OUT_DIR / "item7_season_walk.log").write_text(summary, encoding="utf-8")
    print(summary, end="")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
