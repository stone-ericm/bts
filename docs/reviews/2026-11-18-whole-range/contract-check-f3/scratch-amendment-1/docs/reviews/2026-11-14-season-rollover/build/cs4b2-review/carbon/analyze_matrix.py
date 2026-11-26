"""Normalize the two subprocess TemporaryDirectory prefixes in matrix_base.json and matrix_branch.json
and verify exact byte-for-byte / alert-for-alert equality across all 31,104 cells."""
from __future__ import annotations

import json
import re
from pathlib import Path

from probe_matrix import check_tag_matches_message, expected_refusal_tag

WORK = Path(__file__).resolve().parent
TMP_RE = re.compile(r"/private/var/folders/[^/]+/[^/]+/T/tmp[^/\"`\s]+|/var/folders/[^/]+/[^/]+/T/tmp[^/\"`\s]+")


def norm(obj):
    if isinstance(obj, str):
        return TMP_RE.sub("<TMP>", obj)
    if isinstance(obj, list):
        return [norm(x) for x in obj]
    if isinstance(obj, dict):
        return {k: norm(v) for k, v in obj.items()}
    return obj


def main() -> None:
    base = norm(json.loads((WORK / "matrix_base.json").read_text(encoding="utf-8")))
    branch = norm(json.loads((WORK / "matrix_branch.json").read_text(encoding="utf-8")))

    assert base["denied_network"] == []
    assert branch["denied_network"] == []
    assert base["audit_violations"] == []
    assert branch["audit_violations"] == []

    counts = {
        "total_cells": branch["cell_count"],
        "opt_dec_diff": 0,
        "opt_hlth_diff": 0,
        "sel_opt_diff": 0,
        "sel_req_diff": 0,
        "req_dec_identical": 0,
        "req_dec_changed": 0,
        "req_dec_accepted_branch": 0,
        "req_dec_loader_err_branch": 0,
        "req_dec_refused_branch": 0,
        "req_hlth_identical": 0,
        "req_hlth_changed": 0,
        "req_hlth_undated_auto_changed": 0,
        "req_hlth_refusal_appended": 0,
        "uncovered_or_legacy_accepted_live": 0,
        "unexplained_diffs": [],
    }

    for key, br_tuple in branch["results"].items():
        ba_tuple = base["results"][key]
        sw_k, man_k, auto_k, mod_k, sav_k = key.split("|")
        ba_opt_d, ba_opt_h, ba_sel_o, ba_sel_r, ba_req_d, ba_req_h = ba_tuple
        br_opt_d, br_opt_h, br_sel_o, br_sel_r, br_req_d, br_req_h = br_tuple

        if br_opt_d != ba_opt_d:
            counts["opt_dec_diff"] += 1
            counts["unexplained_diffs"].append(("opt_dec", key, ba_opt_d, br_opt_d))
        if br_opt_h != ba_opt_h:
            counts["opt_hlth_diff"] += 1
            counts["unexplained_diffs"].append(("opt_hlth", key, ba_opt_h, br_opt_h))
        if br_sel_o != ba_sel_o:
            counts["sel_opt_diff"] += 1
            counts["unexplained_diffs"].append(("sel_opt", key, ba_sel_o, br_sel_o))
        if br_sel_r != ba_sel_r:
            counts["sel_req_diff"] += 1
            counts["unexplained_diffs"].append(("sel_req", key, ba_sel_r, br_sel_r))

        if br_req_d == ba_req_d:
            counts["req_dec_identical"] += 1
        else:
            counts["req_dec_changed"] += 1

        if br_req_h == ba_req_h:
            counts["req_hlth_identical"] += 1
        else:
            counts["req_hlth_changed"] += 1

        if not br_sel_r["ok"]:
            counts["req_dec_loader_err_branch"] += 1
            if br_req_d != ba_req_d:
                counts["unexplained_diffs"].append(("loader_err_dec_changed", key, ba_req_d, br_req_d))
            expected_hlth = [{"level": "CRITICAL", "source": "contest_state", "message": br_sel_r["msg"]}]
            if br_req_h != {"ok": True, "val": expected_hlth}:
                counts["unexplained_diffs"].append(("loader_err_hlth_wrong", key, br_req_h))
            if br_req_h != ba_req_h:
                if auto_k not in ("undated", "undated_bad_v2_season"):
                    counts["unexplained_diffs"].append(("unexpected_loader_err_hlth_diff", key, ba_req_h, br_req_h))
                else:
                    counts["req_hlth_undated_auto_changed"] += 1
            continue

        tag = expected_refusal_tag(sw_k, man_k, auto_k, br_sel_r)
        if tag is None:
            counts["req_dec_accepted_branch"] += 1
            if not br_req_d["ok"] or br_req_d != ba_req_d:
                counts["unexplained_diffs"].append(("accepted_dec_diff", key, ba_req_d, br_req_d))
            if br_req_h != ba_req_h:
                counts["unexplained_diffs"].append(("accepted_hlth_diff", key, ba_req_h, br_req_h))
            if man_k != "dated_today" and not (
                sw_k == "automatic" and auto_k in ("auto_v2_covered", "recorded_at_only")
            ):
                counts["uncovered_or_legacy_accepted_live"] += 1
        else:
            counts["req_dec_refused_branch"] += 1
            if br_req_d["ok"]:
                counts["uncovered_or_legacy_accepted_live"] += 1
                counts["unexplained_diffs"].append(("should_have_refused", key, tag, br_req_d))
            else:
                if br_req_d["err"] != "ContestStateError" or not check_tag_matches_message(tag, br_req_d["msg"]):
                    counts["unexplained_diffs"].append(("wrong_refusal_msg", key, tag, br_req_d))
                refusal_alert = {"level": "CRITICAL", "source": "contest_state", "message": br_req_d["msg"]}
                expected_alerts = ba_req_h["val"] + [refusal_alert]
                if br_req_h != {"ok": True, "val": expected_alerts}:
                    counts["unexplained_diffs"].append(("wrong_hlth_refusal", key, expected_alerts, br_req_h))
                else:
                    counts["req_hlth_refusal_appended"] += 1
                occurrences = sum(1 for a in br_req_h["val"] if a["message"] == br_req_d["msg"])
                if occurrences != 1 or br_req_h["val"][-1] != refusal_alert:
                    counts["unexplained_diffs"].append(("refusal_not_once_and_last", key, br_req_h))

    for k in ("fresh_no_picks_dec", "stale_gap2_dec", "stale_gap2_hlth", "stale_miss_dec"):
        if branch["extra"][k] != base["extra"][k]:
            counts["unexplained_diffs"].append(("extra_parity", k, base["extra"][k], branch["extra"][k]))

    summary = {"counts": counts, "branch_extra": branch["extra"]}
    (WORK / "matrix_report_normalized.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    assert counts["opt_dec_diff"] == 0
    assert counts["opt_hlth_diff"] == 0
    assert counts["sel_opt_diff"] == 0
    assert counts["sel_req_diff"] == 0
    assert counts["uncovered_or_legacy_accepted_live"] == 0
    assert counts["unexplained_diffs"] == [], counts["unexplained_diffs"][:3]


if __name__ == "__main__":
    main()
