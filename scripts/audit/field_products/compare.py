"""W2.2 comparison (design r1 edits B-E1 and B-E6). E's usable contest-graded slots (May 1–July 3, daily corpus)
versus our ledger slots in the same fixed window, as POOLED SLOT RATIOS (hits / graded slots: prolific users and DD
days weigh more; not a mean-user skill estimate). Two tables: every observed date (union; each arm its own
denominator) and one frozen shared-date subset (dates with ≥1 usable graded slot in each arm). Whole dates are
resampled jointly for both arms, 10,000 times at seed 20261004, keeping every drawn date's rows and every repeat
(``mlb_benchmark.metrics.summary_bootstrap`` over per-date sums: a ratio of sums over drawn dates with repeats is
the ratio over the concatenated rows). Failed draws (an empty denominator) are counted and reported unavailable.
Intervals assume exchangeable date clusters; they do not cover cross-date dependence, observation selection or
missing histories. No causal value, test, skill rank or equivalence follows."""
from __future__ import annotations

from collections import Counter
from datetime import date

import pandas as pd

from scripts.audit.mlb_benchmark import metrics as m

PRIMARY = (date(2026, 5, 1), date(2026, 7, 3))
EXTENSION = (date(2026, 7, 4), date(2026, 9, 27))
SEED = 20261004
CONFIRMED_MATCHES = frozenset({"evidenced", "inferred"})   # the ledger's only grade-transferring links (memo §5)
EXACT = frozenset({"hit", "not_hit"})
# Daily username-keyed files carry no user id and no stored record witnesses which account each appended batch was
# fetched for (code search, review r1 F2): the E arm rests on the members' stable 5/01 usernames, unwitnessed per batch.
ATTRIBUTION_BASIS = "stable_5_01_username_unwitnessed"


def _iso(d) -> str | None:
    return None if _v(d) is None else pd.Timestamp(d).date().isoformat()


def _v(x):
    """A ledger cell with pandas' missing markers (None/NaN/NA) normalised to None."""
    return None if x is None or (not isinstance(x, str) and pd.isna(x)) else x


def ours_slots(ledger: pd.DataFrame, contest_slots: pd.DataFrame | None, start: date, end: date
               ) -> tuple[pd.DataFrame, dict]:
    """Our denominator: a committed_evidenced selection row, uniquely confirmed contest linkage (entry_status
    confirmed through an evidenced/inferred match, and exactly one contest slot carrying its selection_id whose grade
    equals the row's) and an exact hit/not_hit contest grade. Both DD legs are kept. The contest grade is used, never
    local or current-feed regrading. Every exclusion is counted by its first failing reason."""
    d = ledger["date"].map(_iso)
    undated = ledger[d.isna()]
    dd = d.fillna("")
    win = ledger[(dd >= start.isoformat()) & (dd <= end.isoformat())].copy()
    win["date"] = win["date"].map(_iso)
    sels = win[win["row_kind"] == "selection"]
    links = Counter(contest_slots["selection_id"].dropna()) if contest_slots is not None else None
    grade_of = (contest_slots.dropna(subset=["selection_id"]).groupby("selection_id")["slot_result"].first().to_dict()
                if contest_slots is not None else {})
    reasons: Counter = Counter()
    keep = []
    for r in sels.itertuples():
        commit, entry, match = _v(r.commit_status), _v(r.entry_status), _v(r.match)
        grade, sid = _v(r.contest_slot_grade_raw), _v(r.selection_id)
        if commit != "committed_evidenced":
            reasons[f"not_committed_evidenced:{commit}"] += 1
        elif entry != "confirmed" or match not in CONFIRMED_MATCHES:
            reasons[f"not_uniquely_confirmed:{entry}/{match}"] += 1
        elif links is not None and links.get(sid, 0) != 1:
            reasons[f"contest_link_not_unique:{links.get(sid, 0)}"] += 1
        elif links is not None and _v(grade_of.get(sid)) != grade:
            reasons["contest_grade_disagrees"] += 1
        elif grade not in EXACT:
            reasons[f"grade_not_exact:{grade}"] += 1
        else:
            keep.append(r.Index)
    inc = sels.loc[keep, ["date", "round_id", "slot", "selection_id", "unit_id", "batter_id", "game_pk", "match",
                          "contest_slot_grade_raw"]].copy()
    inc["hit"] = inc["contest_slot_grade_raw"] == "hit"
    inc["user_id"] = 0
    exc = {"window_rows": int(len(win)), "selection_rows": int(len(sels)),
           "undated_rows": dict(Counter(undated["row_kind"])),
           "non_selection_rows": dict(Counter(win.loc[win["row_kind"] != "selection", "row_kind"])),
           "reasons": dict(reasons), "included": int(len(inc)),
           "included_by_slot": dict(Counter(inc["slot"])), "included_by_match": dict(Counter(inc["match"])),
           "contest_cross_check": contest_slots is not None}
    return inc.reset_index(drop=True), exc


def e_arm(slots: pd.DataFrame, available: set[int], start: date, end: date) -> pd.DataFrame:
    """E's usable graded slots (exact hit/not_hit, status ok) of members with an available daily history; no
    filter on allocation (E_in_A members stay in the primary)."""
    if not len(slots):
        return pd.DataFrame(columns=["user_id", "date", "round_id", "pick_number", "hit", "attribution_basis"])
    d = pd.to_datetime(slots["pick_date"]).dt.date
    s = slots[slots["user_id"].isin(available) & slots["usable"] & (d >= start) & (d <= end)]
    out = s[["user_id", "round_id", "pick_number", "hit"]].copy()
    out.insert(1, "date", s["pick_date"].map(_iso))
    out["attribution_basis"] = ATTRIBUTION_BASIS
    return out.reset_index(drop=True)


def e_exclusions(slots: pd.DataFrame, available: set[int], start: date, end: date) -> dict:
    """E's window slots of available members that are NOT usable graded slots, counted by label or status
    (pending, Pass/void, other labels, conflicts, unresolved identity) — never silently dropped."""
    if not len(slots):
        return {"window_slots": 0, "usable_graded": 0, "excluded_by_label": {}, "excluded_by_status": {}}
    d = pd.to_datetime(slots["pick_date"]).dt.date
    s = slots[slots["user_id"].isin(available) & (d >= start) & (d <= end)]
    return {"window_slots": int(len(s)), "usable_graded": int(s["usable"].sum()),
            "excluded_by_label": dict(Counter(s.loc[(s["status"] == "ok") & ~s["graded"], "label"])),
            "excluded_by_status": dict(Counter(s.loc[s["status"] != "ok", "status"]))}


def _per_date(arm: pd.DataFrame, p: str) -> pd.DataFrame:
    g = arm.groupby("date")
    return pd.DataFrame({f"{p}_slots": g.size(), f"{p}_hits": g["hit"].sum().astype(int),
                         f"{p}_users": g["user_id"].nunique()})


def _ratio(p: str):
    def f(s: pd.DataFrame) -> float:
        n = s[f"{p}_slots"].sum()
        return float(s[f"{p}_hits"].sum() / n) if n > 0 else float("nan")
    return f


def _round_counts(arm: pd.DataFrame) -> dict:
    """Analytical round keys (user_id, round_id); rows without a round id and user-dates holding several round ids are
    counted, never folded into user-dates."""
    keyed = arm[arm["round_id"].notna()]
    per_date = keyed.groupby(["user_id", "date"])["round_id"].nunique()
    return {"rounds": int(len(keyed[["user_id", "round_id"]].drop_duplicates())),
            "round_id_missing": int(arm["round_id"].isna().sum()),
            "user_dates_with_multiple_rounds": int((per_date > 1).sum())}


def _arm_block(arm: pd.DataFrame, pd_tab: pd.DataFrame, p: str) -> dict:
    sub = arm[arm["date"].isin(pd_tab.index)]
    n = int(pd_tab[f"{p}_slots"].sum())
    basis = sorted(set(sub["attribution_basis"])) if "attribution_basis" in sub else []
    return {"dates_with_slots": int((pd_tab[f"{p}_slots"] > 0).sum()), "users": int(sub["user_id"].nunique()),
            **_round_counts(sub), "attribution_basis": ";".join(basis) if basis else None,
            "slots": n, "hits": int(pd_tab[f"{p}_hits"].sum()),
            "ratio": float(pd_tab[f"{p}_hits"].sum() / n) if n else None}


def _clean(iv: dict) -> dict:
    return {k: {kk: (None if isinstance(vv, float) and vv != vv else vv) for kk, vv in v.items()} for k, v in iv.items()}


def _coverage(e: pd.DataFrame, dates, n_members: int | None, calendar_dates: int | None) -> dict:
    users = int(e.loc[e["date"].isin(dates), "user_id"].nunique())
    return {"E_members": n_members, "E_users_contributing": users,
            "E_user_share": users / n_members if n_members else None, "window_calendar_dates": calendar_dates,
            "dates": int(len(dates)), "date_share_of_window": len(dates) / calendar_dates if calendar_dates else None}


def date_tables(e: pd.DataFrame, o: pd.DataFrame, n_resamples: int = 10_000, n_members: int | None = None,
                calendar_dates: int | None = None) -> dict:
    """The all-observed-date table (union of dates; each arm's own denominator) and the frozen shared-date table
    (dates with ≥1 usable graded slot in each arm) with joint whole-date percentile intervals. Each arm block reports
    its user, date, user-round and slot denominators; ``coverage`` relates them to E and the window calendar."""
    tab = _per_date(e, "E").join(_per_date(o, "ours"), how="outer").fillna(0).astype(int).sort_index()
    out = {}
    stats_all = {"E_ratio": _ratio("E"), "ours_ratio": _ratio("ours")}
    out["all_observed_dates"] = {
        "dates": int(len(tab)), "available": bool(len(tab)), "E": _arm_block(e, tab, "E"),
        "ours": _arm_block(o, tab, "ours"), "coverage": _coverage(e, tab.index, n_members, calendar_dates),
        "intervals": _clean(m.summary_bootstrap(tab, stats_all, n_resamples=n_resamples, seed=SEED)) if len(tab) else {}}
    shared = tab[(tab["E_slots"] > 0) & (tab["ours_slots"] > 0)]
    if not len(shared):
        out["shared_dates"] = {"dates": 0, "available": False, "date_list": [], "diff_ours_minus_E": None,
                               "E": None, "ours": None, "intervals": {},
                               "coverage": _coverage(e, [], n_members, calendar_dates)}
    else:
        eb, ob = _arm_block(e, shared, "E"), _arm_block(o, shared, "ours")
        stats = {**stats_all, "diff_ours_minus_E": lambda s: _ratio("ours")(s) - _ratio("E")(s)}
        out["shared_dates"] = {"dates": int(len(shared)), "available": True, "date_list": list(shared.index),
                               "E": eb, "ours": ob, "diff_ours_minus_E": ob["ratio"] - eb["ratio"],
                               "coverage": _coverage(e, shared.index, n_members, calendar_dates),
                               "intervals": _clean(m.summary_bootstrap(shared, stats, n_resamples=n_resamples,
                                                                       seed=SEED))}
    out["per_date"] = tab
    out["note"] = ("pooled slot ratios; joint whole-date resampling (exchangeable date clusters assumed; no "
                   "cross-date dependence, observation selection or missing-history adjustment); descriptive only")
    return out


def availability(members: pd.DataFrame, labels: pd.DataFrame, binding: pd.DataFrame, *, ownership_quarantined: set,
                 res, obs_stats: dict, start: date, end: date, windows: dict | None = None) -> pd.DataFrame:
    """Every E member: acquisition label, daily history availability (binding / quarantine), observed window activity
    (missing observations never establish that someone stopped or skipped), usable graded slots and DD counts."""
    from scripts.audit.field_products import picks as P
    graded = P.graded_summary(res.slots, start, end) if len(res.slots) else {}
    dd = P.dd_summary(res.rounds, start, end) if len(res.rounds) else {}
    lab = labels.set_index("user_id")["allocation"].to_dict()
    bnd = binding.set_index("user_id")
    rows = []
    for r in members.itertuples():
        uid = int(r.user_id)
        b = bnd.loc[uid]
        st = obs_stats.get(uid, {})
        if b["binding"] != "bound":
            hist = b["binding"]
        elif uid in ownership_quarantined:
            hist = "quarantined_ownership_conflict"
        elif not st.get("rows"):
            hist = "empty_files"
        else:
            hist = "available"
        g, k = graded.get(uid, {}), dd.get(uid, {})
        activity = ("not_assessable" if hist != "available"
                    else "observed" if k.get("rounds") else "none_observed_unknown")
        use = hist == "available"
        row = {"order": int(r.order), "user_id": uid, "allocation": lab.get(uid), "binding": b["binding"],
               "files": list(b["files"]), "daily_history": hist, "observation_rows": int(st.get("rows", 0)),
               "attribution_basis": ATTRIBUTION_BASIS if hist == "available" else None,
               "captures": int(st.get("captures", 0)), "first_capture": st.get("first_capture"),
               "last_capture": st.get("last_capture"), "window_activity": activity,
               "window_rounds": int(k.get("rounds", 0)) if use else 0,
               "window_pick_days": int(k.get("pick_days", 0)) if use else 0,
               "window_complete_rounds": int(k.get("complete_rounds", 0)) if use else 0,
               "window_dd_rounds": int(k.get("dd_rounds", 0)) if use else 0,
               "window_dd_frequency": k.get("dd_frequency") if use else None,
               "window_incomplete_rounds": int(k.get("incomplete_rounds", 0)) if use else 0,
               "window_graded_slots": int(g.get("graded_slots", 0)) if use else 0,
               "window_hits": int(g.get("hits", 0)) if use else 0,
               "window_hit_rate": g.get("hit_rate") if use else None,
               "window_excluded_slots": int(g.get("slots", 0) - g.get("graded_slots", 0)) if use else 0,
               "last_window_pick_date": None}
        if use and len(res.rounds):
            mine = res.rounds[res.rounds["user_id"] == uid]
            dts = pd.to_datetime(mine["pick_date"]).dt.date
            inwin = dts[(dts >= start) & (dts <= end)]
            row["last_window_pick_date"] = inwin.max().isoformat() if len(inwin) else None
        if windows is not None:
            w = windows.get(uid) if use else None
            row.update({"window_streak_status": w["status"] if w else "not_assessable",
                        "window_longest_exact": w["longest_exact"] if w else None,
                        "window_longest_lower_bound": w["lower_bound"] if w else None,
                        "window_streak_reasons": ";".join(w["reasons"]) if w else None})
        rows.append(row)
    return pd.DataFrame(rows)


def extension(fg_status: pd.DataFrame, slots: pd.DataFrame, n_members: int) -> dict:
    """The final-backfill extension, July 4–Sept 27, from usable final-grab histories of E∩(A∪B): labelled
    separately, never appended to the primary; its support is partly determined by final outcomes."""
    start, end = EXTENSION
    usable = set(fg_status.loc[fg_status["usable"], "user_id"].astype(int))
    arm = e_arm(slots, usable, start, end)
    per_user = arm.groupby("user_id").agg(slots=("hit", "size"), hits=("hit", "sum")).reset_index()
    n = int(len(arm))
    hist = Counter(fg_status["history"])
    return {"label": "final-backfill extension (E∩(A∪B) usable final-grab histories; restricted, partly "
                     "outcome-determined support; never appended to the primary curve)",
            "window": [start.isoformat(), end.isoformat()],
            "attribution_basis": "final_grab_user_id_keyed (identity.json parsed_sha256 verified)",
            "pooled": {"users": int(arm["user_id"].nunique()), "dates": int(arm["date"].nunique()),
                       "rounds": int(len(arm[["user_id", "round_id"]].drop_duplicates())), "slots": n,
                       "hits": int(arm["hit"].sum()), "ratio": float(arm["hit"].mean()) if n else None},
            "per_user": per_user.assign(hit_rate=per_user["hits"] / per_user["slots"]).to_dict("records"),
            "coverage": {"E_members": n_members, "usable": len(usable),
                         "usable_share_of_E": len(usable) / n_members if n_members else None,
                         "E_in_A": int((fg_status["allocation"] == "E_in_A").sum()),
                         "E_in_A_usable": int((fg_status["usable"] & (fg_status["allocation"] == "E_in_A")).sum()),
                         "budget_omissions": int(hist.get("budget_omission", 0)),
                         "fetch_or_parse_failures": int(hist.get("fetch_or_parse_failure", 0)
                                                        + hist.get("parsed_hash_mismatch", 0)),
                         "no_history": int(hist.get("no_history", 0)), "history": dict(hist),
                         "history_depth": fg_status.loc[fg_status["usable"], ["user_id", "first_pick_date",
                                                                              "last_pick_date", "n_rounds"]
                                                        ].to_dict("records")}}
