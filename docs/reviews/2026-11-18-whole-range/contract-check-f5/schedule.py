"""Read the actual installer lines and enumerate slots for ordinary and DST dates."""
from collections import Counter
from datetime import datetime, timedelta, timezone
import csv
import json
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")

def field(text, count):
    if text == "*":
        return range(count)
    if text.startswith("*/"):
        return range(0, count, int(text[2:]))
    if "-" in text:
        low, high = map(int, text.split("-"))
        return range(low, high + 1)
    return (int(text),)

def job(label):
    data = json.loads((HERE / "generated-crons.json").read_text())
    return next(line for line in data[label] if "run bts check-pick-entered" in line)

def slots(label):
    minute, hour, *_ = job(label).split()
    return [h * 60 + m for h in field(hour, 24) for m in field(minute, 60)]

def format_min(m):
    return f"{m // 60:02d}:{m % 60:02d}"

def inside(pitch, starts):
    return [m for m in starts if 5 < pitch - m <= 75]

if __name__ == "__main__":
    base, f5 = slots("base"), slots("f5")
    with (HERE / "pitch-grid.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("pitch", "base_count", "f5_count", "base_slots", "f5_slots"))
        for pitch in range(1440):
            b, f = inside(pitch, base), inside(pitch, f5)
            assert set(b).issubset(f)
            if pitch % 5 == 0:
                writer.writerow((format_min(pitch), len(b), len(f), " ".join(map(format_min, b)),
                                 " ".join(map(format_min, f))))
    for pitch in (0, 5, 10, 6 * 60 + 10, 10 * 60 + 5, 11 * 60, 11 * 60 + 1, 13 * 60 + 5,
                  19 * 60 + 10, 22 * 60 + 10, 23 * 60 + 59, 24 * 60 + 20):
        print(format_min(pitch), *[[format_min(m) for m in inside(pitch, s)] for s in (base, f5)])
    for step in (5, 1):
        gains = [p for p in range(0, 1440, step) if len(inside(p, f5)) > len(inside(p, base))]
        print("grid", step, "gain", len(gains), format_min(min(gains)), format_min(max(gains)),
              "distribution", [dict(sorted(Counter(len(inside(p, s)) for p in range(0, 1440, step)).items()))
                               for s in (base, f5)])
    for month, day in ((3, 14), (11, 7)):
        start = datetime(2027, month, day, tzinfo=ET).astimezone(timezone.utc)
        utc = [start + timedelta(minutes=15 * i) for i in range(25 * 4)]
        local = [stamp.astimezone(ET) for stamp in utc]
        for label in ("base", "f5"):
            plan = set(slots(label))
            launch = [s for s in local if s.date() == start.astimezone(ET).date() and s.hour * 60 + s.minute in plan]
            gaps = {(b.astimezone(timezone.utc) - a.astimezone(timezone.utc)).total_seconds()
                    for a, b in zip(launch, launch[1:])}
            print(label, launch[0].date(), len(launch), gaps,
                  [s.strftime("%H:%M %Z") for s in launch if s.hour < 4])
