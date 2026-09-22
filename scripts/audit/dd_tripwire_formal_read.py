import csv, json, numpy as np
from collections import OrderedDict
"""P-01 DD tripwire formal read: first n>=80 snapshot over the 7/12 recipe output
(scripts/audit/build_slot_dataset.py -> /tmp/slot_dataset_2026.csv). See
docs/audit/2026-09-22-dd-tripwire-formal-read.md."""
rows = list(csv.DictReader(open("/tmp/slot_dataset_2026.csv")))
dd = sorted([r for r in rows if r["slot"] == "double_down" and r["outcome"] in ("hit", "miss")], key=lambda r: r["date"])
pr = sorted([r for r in rows if r["slot"] == "pick" and r["outcome"] in ("hit", "miss")], key=lambda r: r["date"])

def pb_tail(ps, k, upper=False):
    dist = np.array([1.0])
    for p in ps:
        dist = np.convolve(dist, [1 - p, p])
    return float(dist[k:].sum()) if upper else float(dist[:k + 1].sum())

out = {}
def read(legs, label):
    n = len(legs); h = sum(r["outcome"] == "hit" for r in legs); ps = [float(r["p"]) for r in legs]
    real = h / n; stated = float(np.mean(ps)); gap = (stated - real) * 100
    tail = pb_tail(ps, h) if real < stated else pb_tail(ps, h, upper=True)
    print(f"{label:52s} {h}/{n} realized={real:.3f} stated={stated:.3f} shortfall={gap:+.1f}pp exact_tail={tail:.3f} span={legs[0]['date']}..{legs[-1]['date']}")
    out[label] = {"hits": h, "n": n, "realized": round(real, 4), "stated": round(stated, 4), "shortfall_pp": round(gap, 2), "exact_tail": round(tail, 4), "first": legs[0]["date"], "last": legs[-1]["date"]}
    return gap

cum = 0; cross = None; by_date = OrderedDict()
for r in dd:
    by_date.setdefault(r["date"], []).append(r)
for d, legs in by_date.items():
    cum += len(legs)
    if cross is None and cum >= 80:
        cross = d
print(f"FIRST DATE WITH CUMULATIVE DD LEGS >= 80: {cross} (cumulative at that date = {sum(len(v) for k, v in by_date.items() if k <= cross)})")
print("total graded DD legs (recipe, all pick files):", len(dd), "| dates with a graded DD leg:", len(by_date))
g = read([r for r in dd if r["date"] <= cross], f"P-01 FORMAL READ @ first n>=80 snapshot ({cross})")
print("  >= 10pp escalation trigger fired:", g >= 10.0)
read([r for r in dd if r["date"] <= "2026-07-12"], "anchor: DD thru 7/12 (doc: 25/42, .595 vs .734)")
read([r for r in dd if r["date"] <= "2026-08-09"], "anchor: DD thru 8/09 (doc: 39/57, .684 vs .740)")
read([r for r in dd if r["date"] <= "2026-09-13"], "season-end UPDATE: DD thru 9/13 (last contest day)")
read(dd, "all graded DD incl. private 9/14+ (research only)")
read([r for r in dd if "2026-08-10" <= r["date"] <= cross], "DD legs 8/10..snapshot (post-interim-look window)")
read([r for r in pr if r["date"] <= cross], f"primaries thru {cross}")
read([r for r in pr if r["date"] <= "2026-09-13"], "primaries thru 9/13")
print("monthly DD:", {m: f"{sum(r['outcome']=='hit' for r in dd if r['date'][5:7]==m)}/{sum(1 for r in dd if r['date'][5:7]==m)}" for m in sorted({r['date'][5:7] for r in dd})})
# inclusion audit: which files fed DD legs after 9/13 (private) and how many DD rows were excluded (unresolved/legacy)
dd_all = [r for r in rows if r["slot"] == "double_down"]
print("DD slot rows total:", len(dd_all), "| graded:", len(dd), "| excluded (unresolved/void/legacy):", len(dd_all) - len(dd))
json.dump({"snapshot_date": cross, "reads": out}, open("/tmp/p01_read.json", "w"), indent=1)
print("wrote /tmp/p01_read.json")
