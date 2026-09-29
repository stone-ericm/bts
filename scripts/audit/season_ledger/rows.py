"""Day status, row kinds, commit / history status and the delivery predicate (spec §5, §9)."""
from __future__ import annotations

from .ids import utc_iso

SELECTION_ACTIONS = frozenset({"single", "double"})
ARCHIVE_KINDS = frozenset({"archive", "manual_archive", "repair_archive"})


def selection_id(date: str, slot: str, batter_id, game_pk) -> str:
    return f"{date}|{slot}|{batter_id}|{game_pk}"


def decision_names(decision: dict | None, slot: str) -> tuple | None:
    """The (batter_id, game_pk) a single/double decision names for `slot`, else None."""
    if decision is None or decision["action"] not in SELECTION_ACTIONS:
        return None
    if slot == "double_down" and decision["action"] != "double":
        return None
    return decision[f"{slot}_batter_id"], decision[f"{slot}_game_pk"]


def pick_delivery(pick: dict | None) -> tuple[bool | None, str]:
    """The pick-file branches of the §9 era predicate, in order. An attempt alone is not delivery."""
    if pick is None:
        return None, "no_pick_file"
    if pick["delivered_at"]:
        return True, "delivered_at"
    if pick["notification_sent"] and pick["notification_id"]:
        return True, "dm_notification"
    if pick["bluesky_posted"] and pick["bluesky_uri"]:
        return True, "public_post"
    if pick["bluesky_posted"]:
        return None, "bluesky_posted_without_uri"
    if pick["delivery_attempted"]:
        return None, "attempt_only"
    return None, "no_delivery_evidence"


def delivery(*, decision_status: str | None, pick: dict | None) -> tuple[bool | None, str, bool]:
    """§9 for one selection (Interpretation I4). `decision_status` is the delivery_status of a decision naming
    this selection (else None); `pick` the attached pick-file slot (else None). A decision `delivered`, then
    a positive pick-side signal, confirm delivery; only without either does `private_locked` read false and
    `locked_unconfirmed` null. The flag marks private/lock status beside a positive pick-side signal."""
    pick_ok, pick_basis = pick_delivery(pick)
    conflict = decision_status in ("private_locked", "locked_unconfirmed") and pick_ok is True
    if decision_status == "delivered":
        return True, "decision_delivered", conflict
    if pick_ok:
        return True, pick_basis, conflict
    if decision_status == "private_locked":
        return False, "decision_private_locked", conflict
    if decision_status == "locked_unconfirmed":
        return None, "decision_locked_unconfirmed", conflict
    return None, pick_basis, conflict


def commit_status(*, selection: tuple, slot: str, decision: dict | None, file_pick: dict | None,
                  pick_agrees: bool) -> tuple[str, str]:
    """§5 and Interpretation I3, independent of contest entry: commit evidence naming this selection versus
    evidence naming something else. A pick file's confirmed delivery counts for the selection only when the
    file's whole selection set agrees (Interpretation I14). The scheduler's committed_pick_written flag
    names no selection and is never commit evidence; neither is a contest match, a generic pick_locked flag
    or a delivery attempt."""
    this: list[str] = []
    other: list[str] = []
    named = decision_names(decision, slot)
    if named is not None:
        (this if named == selection else other).append(f"decision:{decision['delivery_status']}")
    if file_pick is not None:
        ok, basis = pick_delivery(file_pick)
        if ok:
            (this if pick_agrees else other).append(f"delivery:{basis}")
    if other:
        return "conflicted", ";".join(this + [f"other:{o}" for o in other])
    if this:
        return "committed_evidenced", ";".join(this)
    return "unconfirmed", "no_commit_evidence"


def history_status(thin: list[dict], retained: list[dict]) -> str:
    """§5 and Interpretation I5: known_incomplete when a thin observation (a lineup-evolution entry) names a
    (slot, batter, game) that no retained full record for the date names — that version's content is gone.
    A retained archive or version is an observation, not missing content. Never `complete`."""
    kept = {(r["slot"], r["batter_id"], r["game_pk"]) for r in retained}
    return "known_incomplete" if any((o["slot"], o["batter_id"], o["game_pk"]) not in kept for o in thin) else "unknown"


def _decision_cols(decision: dict | None) -> dict:
    if decision is None:
        return {}
    return {"action": decision["action"], "action_source_raw": decision["action_source_raw"],
            "action_source": decision["action_source"], "objective": decision["objective"],
            "degraded_reason": decision["degraded_reason"], "decision_streak": decision["streak"],
            "decision_state_source": decision["state_source"], "decision_state_status": decision["state_status"],
            "decision_obs_id": decision["obs_id"]}


def _day_row(date: str, kind: str, reason: str, *, decision=None, state=None) -> dict:
    row = {"row_id": f"day|{date}|{kind}", "row_kind": kind, "date": date, "slot": None, "selection_id": None,
           "reason": reason, "state_obs_id": state["obs_id"] if state else None,
           "scheduler_commit_flag": state["committed_pick_written"] if state else None, **_decision_cols(decision)}
    if decision is not None and decision["action"] == "skip":
        row.update(declined_batter_id=decision["primary_batter_id"], declined_game_pk=decision["primary_game_pk"])
    return row


def _selection_row(date, slot, selection, *, name, team, p, decision, pick, pick_view, pick_action, file_pick,
                   pick_agrees, pick_complete, state, finalization, history) -> dict:
    commit, basis = commit_status(selection=selection, slot=slot, decision=decision, file_pick=file_pick,
                                  pick_agrees=pick_agrees)
    named = decision_names(decision, slot) == selection
    confirmed, dbasis, conflict = delivery(
        decision_status=decision["delivery_status"] if (decision is not None and named) else None, pick=pick)
    locked = (state is not None and state["pick_locked"] and commit == "committed_evidenced"
              and finalization != "unresolved")
    sid = selection_id(date, slot, *selection)
    return {"row_id": sid, "row_kind": "selection", "date": date, "slot": slot, "selection_id": sid, "reason": None,
            "batter_id": selection[0], "batter_name": name, "team_at_pick": team, "game_pk": selection[1],
            "p_stated": p, "game_time": utc_iso(pick["game_time"]) if pick else None,
            "lineup_position": pick["lineup_position"] if pick else None,
            "projected_lineup": pick["projected_lineup"] if pick else None,
            "pitcher_id": pick["pitcher_id"] if pick else None, "finalization": finalization,
            "pick_view_batter_id": pick_view["batter_id"] if pick_view else None,
            "pick_view_game_pk": pick_view["game_pk"] if pick_view else None,
            "pick_view_action": pick_action if finalization == "unresolved" else None,
            "pick_file_complete": pick_complete,
            "commit_status": commit, "commit_basis": basis, "history_status": history,
            "scheduler_commit_flag": state["committed_pick_written"] if state else None,
            "pick_policy_objective": pick["pick_policy_objective"] if pick else None,
            "predicted_at": utc_iso(pick["run_time"]) if pick else None,
            "locked_at": utc_iso(state["pick_locked_at"]) if locked else None,
            "delivery_attempted": pick["delivery_attempted"] if pick else None, "delivery_attempted_at": None,
            "delivery_confirmed": confirmed, "delivery_basis": dbasis, "delivery_evidence_conflict": conflict,
            "delivered_at": utc_iso(pick["delivered_at"]) if pick else None,
            "game_eligibility": "unknown", "game_eligibility_at": None, "game_eligibility_basis": None,
            "pick_obs_id": pick["obs_id"] if pick else None,
            "pick_view_obs_id": pick_view["obs_id"] if pick_view else None,
            "state_obs_id": state["obs_id"] if state else None, **_decision_cols(decision)}


def day_rows(date: str, *, decision: dict | None, pick_rows: list[dict], pick_file: dict | None,
             state: dict | None, observations: list[dict], unusable: frozenset = frozenset()) -> list[dict]:
    """Spec §5 table. Declined skip candidates, scheduler intent, archives and lineup-evolution entries never
    become selections. `pick_file` is None when no pick file survives, else its `pick_file_state`. A decision
    and a surviving pick file agree only when the file parsed completely and names the identical selection set
    (Interpretation I14); only then are the file's facts attached. `unusable` names the other day-evidence kinds
    present for the date but quarantined: unknown content never reads as absent evidence, never establishes a
    skip, and withholds `known_incomplete` (Codex plan r3 #5)."""
    thin = [o for o in observations if o["source_kind"] == "lineup_evolution"]
    retained = [*pick_rows, *(o for o in observations if o["source_kind"] in ARCHIVE_KINDS)]
    unreadable_versions = bool(unusable & ARCHIVE_KINDS) or (pick_file is not None and not pick_file["complete"])
    history = "unknown" if unreadable_versions else history_status(thin, retained)
    by_slot = {r["slot"]: r for r in pick_rows}
    file_pick = pick_rows[0] if pick_rows else None
    complete = None if pick_file is None else pick_file["complete"]
    pick_action = None if not pick_file or not pick_file["slots"] else (
        "double" if "double_down" in pick_file["slots"] else "single")
    if "decision" in unusable:
        return [_day_row(date, "unfinalized_day", "decision_unusable", state=state)]
    if decision is not None and decision["action"] in SELECTION_ACTIONS:
        slots = ("primary", "double_down") if decision["action"] == "double" else ("primary",)
        chosen = [(slot, decision_names(decision, slot)) for slot in slots]
        agree = pick_file is None or (pick_file["complete"] and {(r["slot"], r["batter_id"], r["game_pk"])
                                                                  for r in pick_rows} == {(s, *sel) for s, sel in chosen})
        return [_selection_row(date, slot, sel, name=decision[f"{slot}_batter_name"], team=decision[f"{slot}_team"],
                               p=decision[f"{slot}_p_game_hit"], decision=decision,
                               pick=by_slot.get(slot) if agree else None,
                               pick_view=None if agree else by_slot.get(slot), pick_action=pick_action,
                               file_pick=file_pick, pick_agrees=agree, pick_complete=complete, state=state,
                               finalization="decision" if agree else "unresolved", history=history)
                for slot, sel in chosen]
    if decision is not None:   # action == "skip"
        if decision["scoreable"] is not False:
            return [_day_row(date, "unfinalized_day", "skip_decision_unexpected_shape", decision=decision, state=state)]
        file_fields = pick_file["file_fields"] if pick_file is not None else None
        if ((state is not None and state["committed_pick_written"]) or any(pick_delivery(p)[0] for p in pick_rows)
                or (file_fields is not None and pick_delivery(file_fields)[0])):
            return [_day_row(date, "unfinalized_day", "skip_decision_with_commit_evidence", decision=decision, state=state)]
        if pick_file is not None and file_fields is None:
            return [_day_row(date, "unfinalized_day", "skip_decision_with_unreadable_pick_file", decision=decision,
                             state=state)]
        if "scheduler_state" in unusable:
            return [_day_row(date, "unfinalized_day", "skip_decision_with_unusable_state", decision=decision, state=state)]
        return [_day_row(date, "skip_day", "decision_skip", decision=decision, state=state)]
    if pick_rows:
        return [_selection_row(date, p["slot"], (p["batter_id"], p["game_pk"]), name=p["batter_name"], team=p["team"],
                               p=p["p_game_hit"], decision=None, pick=p, pick_view=None, pick_action=pick_action,
                               file_pick=p, pick_agrees=True, pick_complete=complete, state=state,
                               finalization="pick_file_only", history=history) for p in pick_rows]
    if pick_file is not None:
        return [_day_row(date, "unfinalized_day", "pick_file_unparseable", state=state)]
    if state is not None and state["committed_pick_written"]:
        reason = "commit_flag_without_record"
    elif state is not None and state["final_skip_candidate_present"]:
        reason = "skip_intent_only"
    elif any(o["source_kind"] in ARCHIVE_KINDS for o in observations):
        reason = "archived_candidates_only"
    elif observations:
        reason = "lineup_evolution_only"
    elif state is not None:
        reason = "scheduler_ran_no_record"
    elif unusable:
        reason = "unusable_evidence_only"
    else:
        return [_day_row(date, "unobserved_day", "no_evidence")]
    return [_day_row(date, "unfinalized_day", reason, state=state)]
