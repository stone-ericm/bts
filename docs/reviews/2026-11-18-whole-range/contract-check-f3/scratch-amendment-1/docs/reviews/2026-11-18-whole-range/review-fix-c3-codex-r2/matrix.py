"""Exhaust the amendment-3 cross product and record exact checked-in table coverage."""

from collections import Counter
from itertools import product
import json
from pathlib import Path

from compare_probes import HERE, offline, scratch
from probe_support import dump, pick_data, seed_streak, RUN


DECISIONS = ("absent", "skip", "other", "scoreable", "unreadable")
SIGNALS = ("none", "bluesky_posted", "bluesky_uri", "notification_sent", "notification_id",
           "delivery_attempted", "delivered_at")
STAMPS = ("absent", "required", "preview", "invalid")
STATES = ("absent", "unlocked", "locked", "unreadable")
MARKERS = ("none", "same", "tagless", "foreign")
WHEN = ("past", "today")

# Exact tuples in the explicit parametrized table (invalid scheduler variants
# share one axis cell). Other tests independently cover some of these factors.
TABLE = {
    ("absent", "none", "absent", "absent", "none"),
    ("absent", "none", "invalid", "absent", "none"),
    ("absent", "none", "required", "absent", "none"),
    ("absent", "none", "preview", "absent", "none"),
    ("absent", "none", "preview", "unlocked", "none"),
    ("skip", "none", "absent", "absent", "none"),
    ("other", "none", "absent", "unlocked", "none"),
    ("scoreable", "none", "preview", "absent", "none"),
    ("skip", "none", "preview", "absent", "same"),
    ("absent", "notification_id", "preview", "absent", "none"),
    ("absent", "delivery_attempted", "preview", "absent", "none"),
    *(("skip", signal, "absent", "absent", "none") for signal in SIGNALS[1:]),
    ("absent", "none", "preview", "locked", "none"),
    ("skip", "none", "absent", "locked", "none"),
    ("skip", "none", "absent", "unreadable", "none"),
    ("unreadable", "none", "preview", "absent", "none"),
}

# Additional exact cells in the named reconcile/delivery tests and the corpus.
OTHER = {
    ("absent", "none", "absent", "locked", "none", "past"),
    ("absent", "bluesky_posted", "absent", "absent", "none", "past"),
    ("absent", "bluesky_posted", "absent", "absent", "none", "today"),
    ("absent", "none", "absent", "absent", "none", "today"),
    *(("absent", "none", "absent", "absent", mark, horizon) for mark in MARKERS[1:] for horizon in WHEN),
    *(("absent", "bluesky_posted", "absent", "absent", mark, "past") for mark in ("same", "tagless")),
    ("absent", "delivery_attempted", "absent", "absent", "none", "past"),
    ("scoreable", "none", "absent", "absent", "none", "past"),
}


def oracle(decision, signal, stamp, state, marker, when):
    if decision == "unreadable":
        return "unreadable or invalid decision record"
    if marker in ("same", "tagless"):
        return "applied-date marker without a result"
    if when == "today":
        return None
    if decision == "scoreable":
        return "committed pick awaiting a result"
    if signal != "none":
        if decision in ("skip", "other"):
            return "skip record contradicts a delivery"
        return ("delivery attempt without a decision or result" if signal == "delivery_attempted"
                else "committed pick awaiting a result")
    if state == "locked":
        return "scheduler state marks the pick locked"
    if state == "unreadable":
        return "unreadable or invalid scheduler state"
    if decision in ("skip", "other") or stamp == "preview":
        return None
    return "no decision or preview stamp"


def measure(root, cell):
    from bts import picks
    from bts.contest_state import DecisionStreakState
    from bts.daily_decision import decision_path, write_decision

    decision, signal, stamp, state, marker, when = cell
    today = "2027-04-20"
    day = "2027-04-19" if when == "past" else today
    data = pick_data(day)
    model = DecisionStreakState(0, True, True, "model", "fresh", 0, True)
    if stamp != "absent":
        data["chosen_on"] = {} if stamp == "invalid" else picks.chosen_on_record(model, required=stamp == "required")
    if signal != "none":
        data[signal] = True if signal in ("bluesky_posted", "notification_sent", "delivery_attempted") else "synthetic"
    dump(root / f"{day}.json", data)
    if decision != "absent":
        assert write_decision(day, root, action="skip" if decision == "skip" else "single", source="matrix",
                              delivery_status="not_applicable", scoreable=decision == "scoreable", now=RUN)
        if decision == "unreadable":
            decision_path(day, root).write_text("{")
    if state != "absent":
        path = root / day / "scheduler_state.json"
        dump(path, {"date": day, "pick_locked": state == "locked"})
        if state == "unreadable":
            path.write_text("{")
    if marker != "none":
        seed_streak(root, markers=[day], season={"same": 2027, "tagless": None, "foreign": 2026}[marker])
    markers = [day] if marker in ("same", "tagless") else []
    replay, blocker = picks._fold_season(root, 2027, today, markers)
    assert picks._replay_season_streak(root, 2027, today) == replay
    if replay is not None:
        assert replay == (0, True) and blocker is None
    else:
        assert blocker[0] == day
    return blocker


if __name__ == "__main__":
    counts = Counter()
    with offline(), scratch("matrix") as directory, (HERE / "matrix.tsv").open("w") as ledger:
        print("decision\tsignal\tchosen_on\tscheduler\tmarker\tday\texpected\tobserved\ttest coverage", file=ledger)
        for index, cell in enumerate(product(DECISIONS, SIGNALS, STAMPS, STATES, MARKERS, WHEN)):
            expected = oracle(*cell)
            observed = measure(Path(directory) / str(index), cell)
            actual = observed[1] if observed else None
            coverage = "table" if cell[-1] == "past" and cell[:-1] in TABLE else "other known" if cell in OTHER else "no exact case identified"
            counts[coverage] += 1
            counts[actual or "skip"] += 1
            if expected != actual:
                counts["mismatch"] += 1
            print(*cell, expected or "skip", f"{observed[0]} ({actual})" if observed else "skip", coverage, sep="\t", file=ledger)
    print(json.dumps(counts, sort_keys=True))
