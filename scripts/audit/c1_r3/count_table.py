"""C1 rank 3, T3: the slot × home/away plate-appearance count table (registration §§1–2, §4).

N is a certified starting batter's scoring-definition PAs in the game (production's PA_ENDING_EVENTS), with the
resumed portion excluded (the feed's flag; see count_verify). Each table cell is a distribution over min(N, 8),
conditioned on N >= 1, with add-one smoothing over the categories 1..8. Category 8 holds N >= 8. The historical
overflow (N > 8) is reported, not dropped or renormalized away. N = 0 starters are counted and excluded from the
conditional. All 18 cells are kept, even when empty.
"""
from __future__ import annotations

from scripts.audit.c1_r3.count_meta import GameMeta

CAP = 8


def starter_counts(meta: GameMeta) -> list[dict]:
    rows = []
    for side, slots in meta.starters.items():
        for slot, pid in sorted(slots.items()):
            n = sum(1 for p in meta.pas if p.side == side and p.batter == pid and not p.resumed)
            rows.append({"game_pk": meta.game_pk, "slot": slot, "is_home": side == "home", "batter": pid, "n": n})
    return rows


def count_table(rows) -> dict:
    cells = {}
    for slot in range(1, 10):
        for home in (False, True):
            ns = [r["n"] for r in rows if r["slot"] == slot and r["is_home"] == home]
            counts = {str(k): sum(1 for n in ns if min(n, CAP) == k and n >= 1) for k in range(1, CAP + 1)}
            total = sum(counts.values())
            cells[f"{slot}|{'home' if home else 'away'}"] = {
                "counts": counts, "n_starts": len(ns), "n_zero_excluded": sum(1 for n in ns if n == 0),
                "overflow_gt8": sum(1 for n in ns if n > CAP),
                "p": {k: (c + 1) / (total + CAP) for k, c in counts.items()}}
    return {"cap": CAP, "smoothing": "add-one over categories 1..8", "conditioning": "N >= 1", "cells": cells}
