"""W2.2 membership and identity (design r1 edits B-E1/B-E2).

E = the 310 distinct user ids of the frozen May 1 four-tab manifest, in recorded order (the analytical population).
A, B, E_in_A, E_unfetched and B_shortfall are acquisition labels of the 9/27 grab, never an analytical cohort.

Daily-corpus pick files are keyed by the sanitized username and carry no user_id; the only capture-time
username↔user_id evidence is the manifest's ``usernames_2026_05_01``. A member is bound to daily files only when its
5/01 name's sanitization belongs to no other manifest member AND every file stem that sanitizes to it is the member's
own raw name (written before the 2026-06-09 sanitizer) or its sanitized name. Otherwise the member is quarantined and
counted. A later username change or reuse by another account is undetectable from these files (stated limit),
except where it alters a settled slot (``picks.ownership_conflict_users``)."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from bts.leaderboard.storage import safe_filename_component

USABLE_STATUSES = frozenset({"success", "success_partial_lookup"})
FAILURE_STATUSES = frozenset({"http_error", "envelope_error", "parse_error", "success_with_unresolved", "aborted",
                              "in_flight"})


def load_manifest(path: Path) -> dict:
    raw = Path(path).read_bytes()
    doc = json.loads(raw)
    users = doc["users"]
    ids = [int(u["user_id"]) for u in users]
    if len(set(ids)) != len(ids):
        raise ValueError("manifest E: user ids are not distinct")
    if [int(u["order"]) for u in users] != list(range(1, len(users) + 1)):
        raise ValueError("manifest E: order is not 1..n in recorded sequence")
    if doc.get("n_users") is not None and int(doc["n_users"]) != len(users):
        raise ValueError("manifest E: n_users disagrees with the users list")
    members = pd.DataFrame({"order": [int(u["order"]) for u in users], "user_id": ids,
                            "usernames": [list(u.get("usernames_2026_05_01") or []) for u in users]})
    return {"members": members, "sha256": hashlib.sha256(raw).hexdigest(), "n": len(users),
            "definition": doc.get("definition"), "frozen_at": doc.get("frozen_at"),
            "source_fixture_sha256": doc.get("source_fixture_sha256")}


def allocation(manifest: dict, cohort_json: dict) -> tuple[pd.DataFrame, dict]:
    """Per-member acquisition label from the grab's cohort.json, re-derived from A and the manifest order (B = the
    first cohort_b E ids outside A) and checked against the recorded lists."""
    m = manifest["members"]
    a_set = {int(x) for x in cohort_json["A"]}
    b_rec = [int(x) for x in cohort_json["B"]]
    cohort_b = len(b_rec) + int(cohort_json.get("B_shortfall") or 0)
    b_re = [u for u in m["user_id"] if u not in a_set][:cohort_b]
    b_set = set(b_re)
    labels = m[["order", "user_id"]].copy()
    labels["allocation"] = ["E_in_A" if u in a_set else "B" if u in b_set else "E_unfetched" for u in m["user_id"]]
    rec_in_a = {int(x) for x in cohort_json.get("E_in_A", [])}
    rec_unf = {int(x) for x in cohort_json.get("E_unfetched", [])}
    checks = {"manifest_sha_matches_grab": cohort_json.get("early_cohort_sha256") == manifest["sha256"],
              "E_n_matches": int(cohort_json.get("E_n") or -1) == manifest["n"],
              "recomputed_allocation_matches": b_re == b_rec
              and rec_in_a == set(labels.loc[labels["allocation"] == "E_in_A", "user_id"])
              and rec_unf == set(labels.loc[labels["allocation"] == "E_unfetched", "user_id"]),
              "B_shortfall": int(cohort_json.get("B_shortfall") or 0), "cohort_b": cohort_b}
    return labels, checks


def _disk(p: Path) -> bytes:
    return Path(p).read_bytes()


def _sha_of(read, p: Path) -> str | None:
    try:
        return hashlib.sha256(read(p)).hexdigest()
    except FileNotFoundError:
        return None


def final_grab_status(labels: pd.DataFrame, identity: dict, grab_dir: Path, read=None) -> pd.DataFrame:
    """Per member: fetched or budget-omitted, the grab's profile status, history depth, and whether the id-keyed
    pick file is usable (a usable status, the file present and its bytes — ``read``, default the disk — hashing to
    identity.json's parsed_sha256)."""
    read = read or _disk
    rows = []
    for r in labels.itertuples():
        rec = identity.get(str(r.user_id))
        fetched = r.allocation != "E_unfetched"
        row = {"order": r.order, "user_id": r.user_id, "allocation": r.allocation, "fetched": fetched,
               "fetch_status": None, "http_status": None, "n_picks": 0, "n_rounds": 0, "first_pick_date": None,
               "last_pick_date": None, "skipped_unknown_round_predictions": 0, "parsed_path": None, "usable": False}
        if not fetched:
            history = "budget_omission"
        elif rec is None:
            history = "fetch_or_parse_failure"
            row["fetch_status"] = "missing_identity_record"
        else:
            row.update(fetch_status=rec.get("status"), http_status=rec.get("http_status"),
                       n_picks=int(rec.get("n_picks") or 0), n_rounds=int(rec.get("n_rounds") or 0),
                       first_pick_date=rec.get("first_pick_date"), last_pick_date=rec.get("last_pick_date"),
                       skipped_unknown_round_predictions=int(rec.get("skipped_unknown_round_predictions") or 0),
                       parsed_path=rec.get("parsed_path"))
            status = rec.get("status")
            if status == "success_no_history":
                history = "no_history"
            elif status in USABLE_STATUSES and rec.get("parsed_path"):
                p = Path(grab_dir) / rec["parsed_path"]
                if _sha_of(read, p) == rec.get("parsed_sha256"):
                    history, row["usable"] = ("usable_partial_lookup" if status == "success_partial_lookup"
                                              else "usable"), True
                else:
                    history = "parsed_hash_mismatch"
            elif status in USABLE_STATUSES:
                history = "no_parsed_picks"
            else:
                history = "fetch_or_parse_failure"
        row["history"] = history
        rows.append(row)
    return pd.DataFrame(rows)


def bind_daily_files(manifest: dict, stems: list[str]) -> tuple[pd.DataFrame, dict]:
    """Bind daily ``user_picks/<stem>.parquet`` files to E members by capture-time (5/01) username evidence only."""
    m = manifest["members"]
    keys_of = {int(r.user_id): {safe_filename_component(n) for n in r.usernames} for r in m.itertuples()}
    accept_of = {int(r.user_id): set(r.usernames) | keys_of[int(r.user_id)] for r in m.itertuples()}
    claimants: dict[str, set[int]] = {}
    for uid, keys in keys_of.items():
        for k in keys:
            claimants.setdefault(k, set()).add(uid)
    by_key: dict[str, list[str]] = {}
    for s in stems:
        by_key.setdefault(safe_filename_component(s), []).append(s)
    rows, file_state = [], {}
    for r in m.itertuples():
        uid = int(r.user_id)
        keys = keys_of[uid]
        cand = sorted(s for k in keys for s in by_key.get(k, []))
        if not keys:
            binding = "no_username_evidence"
        elif any(len(claimants[k]) > 1 for k in keys):
            binding = "quarantined_manifest_collision"
        elif any(s not in accept_of[uid] for s in cand):
            binding = "quarantined_foreign_name_collision"
        elif cand:
            binding = "bound"
        else:
            binding = "no_daily_file"
        for s in cand:
            file_state[s] = "bound" if binding == "bound" else "quarantined"
        rows.append({"order": r.order, "user_id": uid, "usernames_2026_05_01": ";".join(r.usernames),
                     "binding": binding, "files": list(cand), "n_files": len(cand)})   # a list: names may hold any char
    states = Counter(file_state.get(s, "not_E") for s in stems)
    counts = {"files_total": len(stems), "files_bound": states["bound"], "files_quarantined": states["quarantined"],
              "files_not_E": states["not_E"], "members": dict(Counter(row["binding"] for row in rows))}
    return pd.DataFrame(rows), counts


def raw_witness(grab_dir: Path, status: dict, user_id: int, obs_user: pd.DataFrame, read=None) -> tuple[dict, str]:
    """Round-completeness witnesses for one final-grab user (``picks.raw_profile_witness``), only from the archived
    raw profile response whose bytes hash to the grab's recorded ``archived_sha256`` for that user's single
    successful profile request, bound to the parsed file's single capture instant. Otherwise ({}, reason)."""
    import gzip

    from scripts.audit.field_products import picks as P
    read = read or _disk
    entries = [e for e in status.get("requests", []) if e.get("class") == "profile" and e.get("name") == str(user_id)]
    if len(entries) != 1:
        return {}, "no_unique_profile_request"
    e = entries[0]
    if e.get("http_status") != 200 or e.get("outcome") != "success":
        return {}, "request_not_success"
    try:
        raw = read(Path(grab_dir) / e["raw_path"])
    except FileNotFoundError:
        return {}, "archive_missing"
    try:
        body = gzip.decompress(raw)
    except OSError:
        return {}, "archive_not_gzip"
    if hashlib.sha256(body).hexdigest() != e.get("archived_sha256"):
        return {}, "archive_hash_mismatch"
    caps, files = obs_user["captured_at"].unique(), obs_user["file"].unique()
    if len(caps) != 1 or len(files) != 1:
        return {}, "parsed_capture_not_unique"
    return P.raw_profile_witness(body, user_id=user_id, file=str(files[0]), captured_at=caps[0]), "verified"
