"""Season per-slot dataset from pick files + per-miss DD forensics.

Slot grading mirrors the shipped per-slot predicted_vs_realized semantics:
slot_results when present; legacy single-pick day result = primary outcome;
legacy DD days without slot_results excluded (censoring).

Forensics PA reads go through the resumed-portion-aware reader per the
CLAUDE.md safety rule (never raw pd.read_parquet for scoring-adjacent PA).
"""
import csv
import json
import sys
from datetime import date
from pathlib import Path

REPO = Path("/home/bts/projects/bts")
PICKS = REPO / "data" / "picks"
OUT = Path("/tmp/slot_dataset_2026.csv")

rows = []
for p in sorted(PICKS.glob("2026-*.json")):
    if "." in p.stem:
        continue
    try:
        d = date.fromisoformat(p.stem)
    except ValueError:
        continue
    try:
        data = json.loads(p.read_text())
    except Exception as e:
        print(f"UNREADABLE {p.stem}: {e}", file=sys.stderr)
        continue
    result = data.get("result")
    slot_results = data.get("slot_results") or {}
    dd = data.get("double_down") or None

    def emit(slot_key, obj):
        if not obj or obj.get("p_game_hit") is None:
            return
        outcome = slot_results.get(slot_key)
        if outcome not in ("hit", "miss", "void"):
            if result in ("hit", "miss") and dd is None and slot_key == "pick":
                outcome = result  # legacy single-pick
            else:
                outcome = ""  # unresolved or legacy-DD (excluded)
        rows.append({
            "date": p.stem,
            "slot": slot_key,
            "batter_id": obj.get("batter_id"),
            "batter": obj.get("batter_name"),
            "p": round(float(obj["p_game_hit"]), 6),
            "outcome": outcome,
            "projected": obj.get("projected_lineup"),
            "game_pk": obj.get("game_pk"),
            "pitcher": obj.get("pitcher_name"),
            "day_result": result or "",
        })

    emit("pick", data.get("pick") or {})
    emit("double_down", dd)

with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {OUT} ({len(rows)} slot rows)")

graded = [r for r in rows if r["outcome"] in ("hit", "miss")]
for slot in ("pick", "double_down"):
    g = [r for r in graded if r["slot"] == slot]
    if not g:
        continue
    n = len(g)
    hits = sum(1 for r in g if r["outcome"] == "hit")
    mp = sum(r["p"] for r in g) / n
    print(f"{slot}: n={n} hits={hits} realized={hits/n:.4f} mean_p={mp:.4f} gap={mp - hits/n:+.4f}")

# --- forensics: every graded DD miss ---
dd_misses = [r for r in graded if r["slot"] == "double_down" and r["outcome"] == "miss"]
print(f"\nDD misses ({len(dd_misses)}):")
try:
    sys.path.insert(0, str(REPO / "src"))
    from bts.data.build import read_pa_for_bts_scoring
    pa = read_pa_for_bts_scoring(
        REPO / "data" / "processed" / "pa_2026.parquet",
        ["date", "batter_id", "game_pk", "is_hit", "lineup_position"],
    )
    pa["date_str"] = pa["date"].astype(str).str[:10]
    for r in dd_misses:
        sub = pa[(pa["date_str"] == r["date"]) & (pa["batter_id"] == r["batter_id"])]
        n_pa = len(sub)
        gpks = sorted(sub["game_pk"].unique().tolist()) if n_pa else []
        lineup = sorted(sub["lineup_position"].dropna().unique().tolist()) if n_pa else []
        hits_ev = int((sub["is_hit"] == 1).sum()) if n_pa else 0
        print(json.dumps({"date": r["date"], "batter": r["batter"], "pa": n_pa,
                          "hits": hits_ev, "lineup_pos": lineup, "game_pks": gpks,
                          "picked_game_pk": r["game_pk"], "p": r["p"],
                          "projected_at_delivery": r["projected"]}))
except Exception as e:
    print(f"PA forensics failed ({type(e).__name__}: {e}) — check column names", file=sys.stderr)
