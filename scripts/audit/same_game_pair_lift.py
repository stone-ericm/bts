"""Same-game pair correlation test for a both-must-hit double-down (2026-08-30 audit).

Run from the repo root: UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/audit/same_game_pair_lift.py
Results + interpretation: docs/audit/2026-08-30-same-game-pair-correlation.md

Question: for two hitters in the SAME game on OPPOSITE teams, is P(both get a hit)
above or below the independence product P(A hit) * P(B hit)?  Lift R = realized joint
hit rate / independence-predicted joint rate.  R > 1 => positive correlation (helps a
double-down); R < 1 => the "different game" rule has merit.

Marginals: leave-one-out batter-season game-hit rate, shrunk toward the league rate
(k pseudo-games), so a batter's own game never informs his own marginal.
Control: same-date, DIFFERENT-game pairs (should sit at R ~ 1 if marginals are unbiased).
CI: cluster bootstrap over dates.
"""
import sys, inspect
import numpy as np, pandas as pd
from pathlib import Path

SEASONS = [2019, 2021, 2022, 2023, 2024, 2025]
MIN_PA = 3          # starter-like appearance
MAX_LINEUP = 5      # top-of-order, where BTS picks live
K_SHRINK = 20
RNG = np.random.default_rng(42)

def load_pa(season):
    path = Path(f"data/processed/pa_{season}.parquet")
    try:
        from bts.data import build
        fn = getattr(build, "read_pa_for_bts_scoring", None)
        if fn is not None:
            sig = inspect.signature(fn)
            try:
                return fn(path) if len(sig.parameters) >= 1 else fn()
            except TypeError:
                pass
    except Exception as e:  # noqa
        print(f"  (bts reader unavailable for {season}: {e}; raw read)", file=sys.stderr)
    df = pd.read_parquet(path, columns=["game_pk", "date", "batter_id", "lineup_position", "is_home", "is_hit"])
    return df

frames = []
for s in SEASONS:
    df = load_pa(s)
    df = df[["game_pk", "date", "batter_id", "lineup_position", "is_home", "is_hit"]]
    g = (df.groupby(["game_pk", "date", "batter_id", "is_home"], as_index=False)
           .agg(hit=("is_hit", "max"), n_pa=("is_hit", "size"), lineup=("lineup_position", "min")))
    g["season"] = s
    frames.append(g)
bg = pd.concat(frames, ignore_index=True)
bg = bg[(bg.n_pa >= MIN_PA) & (bg.lineup <= MAX_LINEUP)].copy()
bg["hit"] = bg["hit"].astype(float)

# leave-one-out shrunk marginal per batter-season
grp = bg.groupby(["season", "batter_id"])["hit"]
S = grp.transform("sum"); N = grp.transform("size")
league = bg.groupby("season")["hit"].transform("mean")
bg["p"] = ((S - bg["hit"]) + K_SHRINK * league) / ((N - 1) + K_SHRINK)
bg = bg[N >= 10]   # need some history for the marginal
print(f"batter-games: {len(bg):,}  seasons={SEASONS}  mean hit={bg.hit.mean():.4f}  mean p={bg.p.mean():.4f}")

def pairs_within_game(bg, same_team):
    home = bg[bg.is_home == True]; away = bg[bg.is_home == False]  # noqa: E712
    if not same_team:
        m = home.merge(away, on=["game_pk", "date", "season"], suffixes=("_i", "_j"))
    else:
        parts = []
        for side in (home, away):
            mm = side.merge(side, on=["game_pk", "date", "season"], suffixes=("_i", "_j"))
            mm = mm[mm.batter_id_i < mm.batter_id_j]
            parts.append(mm)
        m = pd.concat(parts, ignore_index=True)
    return m

def pairs_cross_game(bg, n_target):
    # same date, different game: sample by pairing shuffled rows within date
    out = []
    for date, d in bg.groupby("date"):
        if d.game_pk.nunique() < 2:
            continue
        a = d.sample(frac=1.0, random_state=int(RNG.integers(1 << 31)))
        b = d.sample(frac=1.0, random_state=int(RNG.integers(1 << 31)))
        mm = pd.DataFrame({
            "date": date, "hit_i": a.hit.values, "p_i": a.p.values, "game_i": a.game_pk.values,
            "lineup_i": a.lineup.values, "hit_j": b.hit.values, "p_j": b.p.values,
            "game_j": b.game_pk.values, "lineup_j": b.lineup.values})
        mm = mm[mm.game_i != mm.game_j]
        out.append(mm)
    m = pd.concat(out, ignore_index=True)
    if len(m) > n_target:
        m = m.sample(n=n_target, random_state=7)
    return m

def lift(m):
    joint = (m.hit_i * m.hit_j).mean(); indep = (m.p_i * m.p_j).mean()
    return joint / indep, joint, indep

def boot_ci(m, reps=400):
    dates = m.date.values
    ud, inv = np.unique(dates, return_inverse=True)
    hh = (m.hit_i * m.hit_j).values; pp = (m.p_i * m.p_j).values
    # per-date sums
    num = np.bincount(inv, weights=hh, minlength=len(ud)); den = np.bincount(inv, weights=pp, minlength=len(ud))
    vals = []
    for _ in range(reps):
        w = RNG.multinomial(len(ud), np.full(len(ud), 1 / len(ud)))
        vals.append((w * num).sum() / (w * den).sum())
    return np.percentile(vals, [2.5, 97.5])

def report(label, m):
    R, joint, indep = lift(m)
    lo, hi = boot_ci(m)
    print(f"{label:<52} n={len(m):>9,}  joint={joint:.4f} indep={indep:.4f}  R={R:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    return R

opp = pairs_within_game(bg, same_team=False)
same = pairs_within_game(bg, same_team=True)
cross = pairs_cross_game(bg, n_target=max(len(opp), 400_000))

print("\n=== Lift R = P(both hit) / [P(A) P(B)]  (R>1: positive correlation helps a DD)")
R_cross = report("control: same date, DIFFERENT game", cross)
R_opp = report("SAME game, OPPOSITE teams (McNeil/Alexander case)", opp)
R_same = report("SAME game, SAME team", same)

print("\n=== Strata (same game, opposite teams)")
report("  both lineup <= 2 (leadoff/2-hole pairs)", opp[(opp.lineup_i <= 2) & (opp.lineup_j <= 2)])
report("  both p >= 0.65 (BTS-grade hitters)", opp[(opp.p_i >= 0.65) & (opp.p_j >= 0.65)])
report("  both p >= 0.70", opp[(opp.p_i >= 0.70) & (opp.p_j >= 0.70)])
for s in SEASONS:
    report(f"  season {s}", opp[opp.season == s])
print("\n=== Strata (control, same date different game)")
report("  both p >= 0.65", cross[(cross.p_i >= 0.65) & (cross.p_j >= 0.65)])

# decision translation
pM, pA, pAlt = 0.7428, 0.7252, 0.7041
rel = R_opp / R_cross
print(f"\n=== Decision: McNeil {pM:.3f} + Alexander {pA:.3f} (same game) vs McNeil + alt {pAlt:.3f} (different game)")
print(f"relative lift same/cross = {rel:.4f};  breakeven = {pAlt/pA:.4f}")
print(f"P(both) same-game pair  ≈ {rel * pM * pA:.4f}")
print(f"P(both) different-game  ≈ {pM * pAlt:.4f}")

print("\n=== Matched controls for the strata that moved")
report("  control lineup<=2 both (same date, diff game)", cross[(cross.lineup_i <= 2) & (cross.lineup_j <= 2)])
report("  control p>=0.70 both  (same date, diff game)", cross[(cross.p_i >= 0.70) & (cross.p_j >= 0.70)])
report("  same team, both lineup<=2", same[(same.lineup_i <= 2) & (same.lineup_j <= 2)])
report("  same team, both p>=0.65", same[(same.p_i >= 0.65) & (same.p_j >= 0.65)])
exact = opp[(opp.lineup_i == 2) & (opp.lineup_j == 1)]   # home 2-hole (McNeil) x away leadoff (Alexander)
report("  EXACT config: home #2 x away #1", exact)
report("  control for exact: cross pairs lineup (2,1)", cross[(cross.lineup_i == 2) & (cross.lineup_j == 1)])
# Bolte option: same team, leadoff + 2-hole
bolte = same[((same.lineup_i == 1) & (same.lineup_j == 2)) | ((same.lineup_i == 2) & (same.lineup_j == 1))]
report("  same team #1 x #2 (McNeil+Bolte config)", bolte)
