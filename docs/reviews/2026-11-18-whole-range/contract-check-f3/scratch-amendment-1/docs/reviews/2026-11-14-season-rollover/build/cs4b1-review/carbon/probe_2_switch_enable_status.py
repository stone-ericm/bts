"""Probe 2: Exhaustive adversarial audit of switch_state, season-contract enable, and season-contract status (Items 4 & 5)."""
from __future__ import annotations

import datetime as dt
import itertools as it
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ET = ZoneInfo("America/New_York")
NOW = dt.datetime(2027, 7, 15, 12, 0, 0, tzinfo=ET)
REAL_DATETIME = dt.datetime
ID_A = "a" * 32
ID_B = "b" * 32

AUDIT_EVENTS: list[tuple[str, str, str]] = []
WATCH_ROOT: Path | None = None


def _audit_hook(event: str, args: tuple) -> None:
    if event in {
        "socket.connect",
        "socket.connect_ex",
        "socket.getaddrinfo",
        "socket.gethostbyname",
        "socket.gethostbyname_ex",
        "socket.getnameinfo",
        "socket.sendto",
        "urllib.Request",
    }:
        raise RuntimeError(f"FORBIDDEN NETWORK AUDIT EVENT: {event} {args!r}")
    w = WATCH_ROOT
    if w is None:
        return
    w_str = str(w)
    if event == "open":
        path_arg, mode, flags = args[0], args[1], args[2]
        if isinstance(path_arg, (str, os.PathLike)):
            p_str = str(path_arg)
            if p_str.startswith(w_str):
                AUDIT_EVENTS.append(("open", os.path.relpath(p_str, w_str), f"mode={mode}|flags={flags}"))
    elif event in {"os.mkdir", "os.unlink", "os.remove", "os.chmod", "os.truncate"}:
        p_str = str(args[0])
        if p_str.startswith(w_str):
            AUDIT_EVENTS.append((event, os.path.relpath(p_str, w_str), ""))
    elif event == "os.rename":
        src, dst = str(args[0]), str(args[1])
        if src.startswith(w_str) or dst.startswith(w_str):
            s_rel = os.path.relpath(src, w_str) if src.startswith(w_str) else src
            d_rel = os.path.relpath(dst, w_str) if dst.startswith(w_str) else dst
            AUDIT_EVENTS.append(("os.rename", f"{s_rel}->{d_rel}", ""))


sys.addaudithook(_audit_hook)


def contract_doc(**overrides) -> dict:
    base = {
        "schema_version": "bts_season_contract_v1",
        "season": 2027,
        "status": "automatic",
        "activation_id": ID_A,
        "switched_at": "2027-07-15T16:00:00Z",
        "revoked_at": None,
        "note": None,
    }
    base.update(overrides)
    return base


def snapshot_tree(root: Path) -> dict[str, bytes | str]:
    out: dict[str, bytes | str] = {}
    if not root.exists():
        return out
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dp = Path(dirpath)
        rel_d = str(dp.relative_to(root))
        out[f"dir:{rel_d}"] = "dir"
        for name in sorted(dirnames + filenames):
            p = dp / name
            rel = str(p.relative_to(root))
            st = p.lstat()
            if os.path.islink(p):
                out[f"link:{rel}"] = os.readlink(p)
            elif os.path.isfile(p):
                try:
                    out[f"file:{rel}"] = p.read_bytes()
                except OSError as exc:
                    out[f"unreadable:{rel}"] = type(exc).__name__
    return out


def main() -> None:
    # 0. Check base tree exits 2 on unknown season-contract commands and touches nothing
    base_cmd = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from click.testing import CliRunner; from bts.cli import cli; "
        "r1 = CliRunner().invoke(cli, ['season-contract', 'status', '--picks-dir', sys.argv[2]]); "
        "r2 = CliRunner().invoke(cli, ['season-contract', 'enable', '--season', '2027', '--picks-dir', sys.argv[2]]); "
        "assert r1.exit_code == 2 and r2.exit_code == 2, (r1.exit_code, r2.exit_code)"
    )
    work = HERE / "_work_probe2"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    cold_base = work / "cold_base/picks"
    subprocess.run([sys.executable, "-c", base_cmd, str(HERE / "base_tree/src"), str(cold_base)], check=True, cwd=ROOT)
    assert not cold_base.parent.exists()

    # Load branch src
    sys.path.insert(0, str(ROOT / "src"))
    import pandas  # noqa: F401
    from click.testing import CliRunner
    import bts.cli as climod
    from bts.season_contract import switch_state, validate_contract, InvalidContract

    class FixedClock(REAL_DATETIME):
        current = NOW

        @classmethod
        def now(cls, tz=None):
            c = cls.current
            return c.astimezone(tz) if tz else c.replace(tzinfo=None)

    dt.datetime = FixedClock
    runner = CliRunner()

    global WATCH_ROOT
    findings_notes: list[str] = []

    # 1. Exhaustive switch file invalid cases + enable overwrite behavior (Item 4)
    invalid_fs_kinds = [
        "directory",
        "dangling_symlink",
        "symlink_loop",
        "unreadable_000",
        "not_utf8",
        "huge_int_top",
        "huge_int_field",
        "deep_array",
        "deep_object",
        "top_nan",
        "top_inf",
        "top_list",
        "top_null",
        "empty_file",
    ]
    fs_results = {}
    for kind in invalid_fs_kinds:
        cdir = work / f"fs_{kind}"
        picks = cdir / "picks"
        sdir = picks / "account_state"
        sdir.mkdir(parents=True)
        p = sdir / "season_contract.json"
        if kind == "directory":
            p.mkdir()
        elif kind == "dangling_symlink":
            p.symlink_to(sdir / "does_not_exist")
        elif kind == "symlink_loop":
            p.symlink_to(p)
        elif kind == "unreadable_000":
            p.write_text("{}", encoding="utf-8")
            p.chmod(0o000)
        elif kind == "not_utf8":
            p.write_bytes(b"\xff\xfe\xfd")
        elif kind == "huge_int_top":
            p.write_text("9" * 5000, encoding="utf-8")
        elif kind == "huge_int_field":
            p.write_text('{"season": ' + "9" * 5000 + "}", encoding="utf-8")
        elif kind == "deep_array":
            p.write_text("[" * 2500 + "0" + "]" * 2500, encoding="utf-8")
        elif kind == "deep_object":
            p.write_text('{"a":' * 2000 + "0" + "}" * 2000, encoding="utf-8")
        elif kind == "top_nan":
            p.write_text("NaN", encoding="utf-8")
        elif kind == "top_inf":
            p.write_text("Infinity", encoding="utf-8")
        elif kind == "top_list":
            p.write_text("[1, 2]", encoding="utf-8")
        elif kind == "top_null":
            p.write_text("null", encoding="utf-8")
        elif kind == "empty_file":
            p.write_text("", encoding="utf-8")

        try:
            st = switch_state(picks, NOW)
            assert st.state == "corrupt" and st.check == 1, f"{kind}: got {st}"
            r_status = runner.invoke(climod.cli, ["season-contract", "status", "--picks-dir", str(picks)])
            assert r_status.exit_code == 1 and "check 1:" in r_status.output

            r_en = runner.invoke(climod.cli, ["season-contract", "enable", "--season", "2027", "--picks-dir", str(picks), "--note", f"from_{kind}"])
            if kind == "directory":
                assert r_en.exit_code == 1 and "was NOT switched" in r_en.output
                assert p.is_dir() and not list(sdir.glob("*.tmp"))
            else:
                assert r_en.exit_code == 0, f"{kind} enable failed: {r_en.output}"
                assert not os.path.islink(p)
                written = json.loads(p.read_text(encoding="utf-8"))
                assert written["status"] == "automatic" and written["season"] == 2027 and written["note"] == f"from_{kind}"
                assert written["revoked_at"] is None
            fs_results[kind] = {"check": st.check, "enable_exit": r_en.exit_code}
        finally:
            if p.exists() and not p.is_symlink() and not p.is_dir():
                p.chmod(0o644)

    # 2. Duplicate keys and NaN in fields
    dup_cdir = work / "dup_nan"
    dup_picks = dup_cdir / "picks"
    (dup_picks / "account_state").mkdir(parents=True)
    sp = dup_picks / "account_state/season_contract.json"
    # Duplicate key where second wins invalid vs valid
    raw_valid_base = json.dumps(contract_doc())[:-1]
    sp.write_text(raw_valid_base + ', "season": true}', encoding="utf-8")
    assert switch_state(dup_picks, NOW).check == 3
    sp.write_text(raw_valid_base + ', "status": "revoked", "revoked_at": "2027-07-15T16:00:00Z"}', encoding="utf-8")
    assert switch_state(dup_picks, NOW).state == "revoked"

    # NaN in each field 2..8
    nan_field_checks = [
        ("schema_version", 2),
        ("season", 3),
        ("status", 4),
        ("activation_id", 5),
        ("switched_at", 6),
        ("revoked_at", 7),
        ("note", 8),
    ]
    for fname, expected_chk in nan_field_checks:
        doc = contract_doc()
        doc[fname] = "__NAN__"
        txt = json.dumps(doc).replace('"__NAN__"', "NaN")
        sp.write_text(txt, encoding="utf-8")
        st = switch_state(dup_picks, NOW)
        assert st.state == "corrupt" and st.check == expected_chk, f"NaN in {fname}: {st}"

    # 3. Missing each key + wrong types + all 21 pairs (i, j) for 2 <= i < j <= 8 across both seasons (2027 and 2026)
    missing_checks = [
        ("schema_version", "automatic", 2),
        ("season", "automatic", 3),
        ("status", "automatic", 4),
        ("activation_id", "automatic", 5),
        ("switched_at", "automatic", 6),
        ("switched_at", "revoked", 6),
        ("revoked_at", "automatic", 7),
        ("revoked_at", "revoked", 7),
    ]
    for fname, st_val, expected_chk in missing_checks:
        doc = contract_doc(status=st_val, revoked_at="2027-07-15T16:00:00Z")
        del doc[fname]
        sp.write_text(json.dumps(doc), encoding="utf-8")
        got = switch_state(dup_picks, NOW)
        assert got.state == "corrupt" and got.check == expected_chk, f"missing {fname} ({st_val}): {got}"
    # Missing note is VALID!
    doc_no_note = contract_doc()
    del doc_no_note["note"]
    sp.write_text(json.dumps(doc_no_note), encoding="utf-8")
    assert switch_state(dup_picks, NOW).state == "automatic"

    bad_mutations = {
        2: ("schema_version", "bts_season_contract_v2"),
        3: ("season", True),
        4: ("status", "enabled"),
        5: ("activation_id", "0" * 31 + "g"),
        6: ("switched_at", "2027-07-15T16:00:00"),
        7: ("revoked_at", "2027-07-15"),
        8: ("note", False),
    }
    pair_count = 0
    for i, j in it.combinations(range(2, 9), 2):
        for yr in (2027, 2026):
            doc = contract_doc(season=yr)
            doc[bad_mutations[i][0]] = bad_mutations[i][1]
            doc[bad_mutations[j][0]] = bad_mutations[j][1]
            sp.write_text(json.dumps(doc), encoding="utf-8")
            got = switch_state(dup_picks, NOW)
            assert got.state == "corrupt" and got.check == i, f"pair ({i},{j}) yr={yr}: got {got}"
            pair_count += 1

    # 4. Activation ID and timestamp edge cases
    bad_ids = [
        None, True, False, 0, 32, [], {}, "", "a" * 31, "a" * 33,
        "A" * 32, "a" * 31 + "F", "0x" + "a" * 30, "a" * 32 + "\n",
        " " + "a" * 31, "٠" * 32, "０" * 32,
    ]
    for bid in bad_ids:
        with subprocess.Popen(["true"]):  # noop placeholder? No, don't spawn processes in loop!
            pass
        break
    for bid in bad_ids:
        try:
            validate_contract(contract_doc(activation_id=bid))
            raise AssertionError(f"accepted bad id {bid!r}")
        except InvalidContract as exc:
            assert exc.check == 5, (bid, exc.check)

    ts_cases = [
        ("2027-07-15T16:00:00Z", True),
        ("2027-07-15T12:00:00-04:00", True),
        ("2027-07-15T21:30:00+05:30", True),
        ("2027-07-15T16:00:00.123456Z", True),
        ("2027-07-15T16:00:00", False),
        ("2027-07-15", False),
        ("2027-07-15Z", False),
        ("2027-07-15T16:00:00+00:00Z", False),
        ("", False),
        ("not-a-ts", False),
        (1234567890, False),
        (True, False),
    ]
    for ts_val, is_ok in ts_cases:
        for chk_no, fld, st_val in [(6, "switched_at", "automatic"), (7, "revoked_at", "revoked")]:
            doc = contract_doc(status=st_val, switched_at="2027-07-15T16:00:00Z", revoked_at="2027-07-15T16:00:00Z")
            doc[fld] = ts_val
            try:
                validate_contract(doc)
                assert is_ok, f"unexpectedly valid {fld}={ts_val!r}"
            except InvalidContract as exc:
                assert not is_ok and exc.check == chk_no, f"wrong failure {fld}={ts_val!r}: {exc.check}"

    # 5. Enable isolation, carry-over, freshness, no-op, and wrong-season WITH existing switch on disk!
    en_dir = work / "enable_iso"
    en_picks = en_dir / "picks"
    (en_picks / "account_state").mkdir(parents=True)
    # Seed switch on disk + sentinel files in every other slot, then run wrong --season 2026:
    # Verify ZERO opens (and zero lstats!) of season_contract.json or any state file!
    sp_en = en_picks / "account_state/season_contract.json"
    sp_en.write_text(json.dumps(contract_doc()), encoding="utf-8")
    lstat_touches = []
    orig_lstat = Path.lstat

    def spy_lstat(self_p, *a, **kw):
        if str(self_p).startswith(str(en_dir)):
            lstat_touches.append(str(self_p.relative_to(en_dir)))
        return orig_lstat(self_p, *a, **kw)

    AUDIT_EVENTS.clear()
    WATCH_ROOT = en_dir
    Path.lstat = spy_lstat
    try:
        r_wrong = runner.invoke(climod.cli, ["season-contract", "enable", "--season", "2026", "--picks-dir", str(en_picks)])
    finally:
        Path.lstat = orig_lstat
        WATCH_ROOT = None
    assert r_wrong.exit_code == 2 and "2027" in r_wrong.output
    assert AUDIT_EVENTS == [] and lstat_touches == [], f"premature state touch on wrong season: {AUDIT_EVENTS} {lstat_touches}"

    # Carry-over matrix:
    # - revoked 2027 + corrupt note=[] -> NOT carried (None)
    sp_en.write_text(json.dumps(contract_doc(status="revoked", revoked_at="2027-05-01T00:00:00Z", note=[])), encoding="utf-8")
    assert runner.invoke(climod.cli, ["season-contract", "enable", "--season", "2027", "--picks-dir", str(en_picks)]).exit_code == 0
    d1 = json.loads(sp_en.read_text())
    assert d1["revoked_at"] is None
    id1 = d1["activation_id"]

    # - repeated enable (no-op): zero writes, same activation_id, new --note ignored!
    AUDIT_EVENTS.clear()
    WATCH_ROOT = en_dir
    r_noop = runner.invoke(climod.cli, ["season-contract", "enable", "--season", "2027", "--picks-dir", str(en_picks), "--note", "ignored"])
    WATCH_ROOT = None
    assert r_noop.exit_code == 0 and "already switched" in r_noop.output
    write_evs = [ev for ev in AUDIT_EVENTS if ev[0] != "open" or "mode=r" not in ev[2]]
    assert write_evs == [], f"noop performed write/rename/mkdir: {write_evs}"
    d1_after = json.loads(sp_en.read_text())
    assert d1_after["activation_id"] == id1 and d1_after["note"] is None

    # - wholly valid same-season revoked -> carries revoked_at and issues fresh UUID!
    sp_en.write_text(json.dumps(contract_doc(status="revoked", activation_id=id1, revoked_at="2027-06-01T12:00:00Z")), encoding="utf-8")
    AUDIT_EVENTS.clear()
    WATCH_ROOT = en_dir
    r_re = runner.invoke(climod.cli, ["season-contract", "enable", "--season", "2027", "--picks-dir", str(en_picks), "--note", "re-enabled"])
    WATCH_ROOT = None
    assert r_re.exit_code == 0
    # Verify enable touched ONLY season_contract.json and its .tmp in account_state!
    touched_non_switch = [
        ev for ev in AUDIT_EVENTS
        if not (
            ev[1] == "picks/account_state"
            or "season_contract.json" in ev[1]
            or ev[1].endswith(".tmp")
        )
    ]
    assert touched_non_switch == [], f"enable touched other files: {touched_non_switch}"
    d2 = json.loads(sp_en.read_text())
    assert d2["activation_id"] != id1 and d2["revoked_at"] == "2027-06-01T12:00:00Z" and d2["note"] == "re-enabled"

    # - valid other-season revoked (2026) -> revoked_at NOT carried (None)!
    sp_en.write_text(json.dumps(contract_doc(season=2026, status="revoked", revoked_at="2026-09-01T12:00:00Z")), encoding="utf-8")
    assert runner.invoke(climod.cli, ["season-contract", "enable", "--season", "2027", "--picks-dir", str(en_picks)]).exit_code == 0
    d3 = json.loads(sp_en.read_text())
    assert d3["revoked_at"] is None and d3["activation_id"] not in {id1, d2["activation_id"]}

    # 6. Exhaustive `status` read-only, non-creation, section independence, and edge diagnostics (Item 5)
    # 6a: Non-existent picks dir and non-existent parent dir
    cold_status_dir = work / "nonexistent_parent/deep/picks"
    before_cold = snapshot_tree(work)
    AUDIT_EVENTS.clear()
    WATCH_ROOT = work
    r_cold = runner.invoke(climod.cli, ["season-contract", "status", "--picks-dir", str(cold_status_dir)])
    WATCH_ROOT = None
    assert r_cold.exit_code == 0
    assert AUDIT_EVENTS == [] and snapshot_tree(work) == before_cold
    for expected_line in [
        "ET date: 2027-07-15; year: 2027; inspected season: 2027",
        "switch: absent",
        "manual: absent",
        "automatic: absent",
        "saver: unavailable (absent); reads uninitialized for season 2027",
        "ledger: absent; rows: 0; bad lines: 0",
    ]:
        assert expected_line in r_cold.output, f"missing {expected_line!r} in:\n{r_cold.output}"

    # 6b: All 5 files broken simultaneously in various ways (symlink loop, dangling symlink, dir, 0o000, refused, non-UTF8, 5000-digit, deep JSON)
    broken_variants = [
        "dangling",
        "loop",
        "directory",
        "unreadable_000",
        "not_utf8",
        "huge_5000",
        "deep_3000",
        "semantic_refused",
    ]
    status_matrix_count = 0
    for v_kind in broken_variants:
        for target_slot in ["all", "switch", "manual", "automatic", "saver", "ledger"]:
            s_root = work / f"st_{v_kind}_{target_slot}"
            s_picks = s_root / "picks"
            s_acc = s_picks / "account_state"
            s_acc.mkdir(parents=True)
            # Seed valid baseline in all 5 slots first
            (s_acc / "season_contract.json").write_text(json.dumps(contract_doc()), encoding="utf-8")
            (s_acc / "contest_streak.manual.json").write_text(json.dumps({
                "schema_version": "bts_contest_streak_manual_v3",
                "effective_date": "2027-07-15", "season": 2027,
                "expires_at": "2027-07-16T03:59:59.999999Z",
                "active_streak": 3, "best_streak": 9, "saver_state": "active",
                "recorded_at": "2027-07-15T12:00:00Z", "source": "manual_cli",
            }), encoding="utf-8")
            (s_acc / "contest_streak.json").write_text(json.dumps({
                "schema_version": "bts_contest_streak_auto_v2",
                "active_streak": 7, "best_streak": 9, "source": "mlb_bts_profile",
                "source_date": "2027-07-14", "recorded_at": "2027-07-15T12:00:00Z",
                "season": 2027, "activation_id": ID_A,
            }), encoding="utf-8")
            (s_acc / "saver_state.json").write_text(json.dumps({"season": 2027, "state": "active"}), encoding="utf-8")
            (s_acc / "contest_ledger.jsonl").write_text(json.dumps({"recorded_at": "2027-07-15T12:00:00Z", "active_streak": 7}) + "\n", encoding="utf-8")

            slot_map = {
                "switch": "season_contract.json",
                "manual": "contest_streak.manual.json",
                "automatic": "contest_streak.json",
                "saver": "saver_state.json",
                "ledger": "contest_ledger.jsonl",
            }
            to_break = list(slot_map.keys()) if target_slot == "all" else [target_slot]
            chmod_paths = []
            for sl in to_break:
                fp = s_acc / slot_map[sl]
                fp.unlink()
                if v_kind == "dangling":
                    fp.symlink_to(s_acc / f"missing_{sl}")
                elif v_kind == "loop":
                    fp.symlink_to(fp)
                elif v_kind == "directory":
                    fp.mkdir()
                elif v_kind == "unreadable_000":
                    fp.write_text("{}", encoding="utf-8")
                    fp.chmod(0o000)
                    chmod_paths.append(fp)
                elif v_kind == "not_utf8":
                    fp.write_bytes(b"\x80\xff")
                elif v_kind == "huge_5000":
                    fp.write_text("9" * 5000, encoding="utf-8")
                elif v_kind == "deep_3000":
                    fp.write_text("[" * 3000 + "0" + "]" * 3000, encoding="utf-8")
                elif v_kind == "semantic_refused":
                    if sl == "switch":
                        fp.write_text(json.dumps(contract_doc(season=True)), encoding="utf-8")
                    elif sl == "manual":
                        fp.write_text(json.dumps({"schema_version": "bts_contest_streak_manual_v3", "effective_date": "bad"}), encoding="utf-8")
                    elif sl == "automatic":
                        fp.write_text(json.dumps({"active_streak": 58, "best_streak": 58}), encoding="utf-8")
                    elif sl == "saver":
                        fp.write_text(json.dumps({"season": 2027, "state": "bogus"}), encoding="utf-8")
                    elif sl == "ledger":
                        fp.write_text("\n[1,2]\n{bad\n", encoding="utf-8")

            try:
                before_snap = snapshot_tree(s_root)
                AUDIT_EVENTS.clear()
                WATCH_ROOT = s_root
                r_st = runner.invoke(climod.cli, ["season-contract", "status", "--picks-dir", str(s_picks), "--season", "0"])
                WATCH_ROOT = None
                after_snap = snapshot_tree(s_root)
                status_matrix_count += 1

                # Verify zero write/create/rename/unlink events!
                write_events = [
                    ev for ev in AUDIT_EVENTS
                    if ev[0] != "open" or any(m in ev[2] for m in ("mode=w", "mode=a", "mode=x", "+"))
                ]
                assert write_events == [], f"status wrote in {v_kind}/{target_slot}: {write_events}"
                assert before_snap == after_snap, f"status mutated tree in {v_kind}/{target_slot}"

                # Exit 1 ONLY when switch is corrupt!
                expected_exit = 1 if ("switch" in to_break) else 0
                assert r_st.exit_code == expected_exit, f"{v_kind}/{target_slot}: exit={r_st.exit_code}\n{r_st.output}"
                assert "Traceback" not in r_st.output

                # Every section must always appear in order!
                out_lines = r_st.output.splitlines()
                prefixes = ["ET date:", "switch:", "manual:", "automatic:", "saver:", "ledger:"]
                positions = []
                for pfx in prefixes:
                    idxs = [idx for idx, ln in enumerate(out_lines) if ln.startswith(pfx)]
                    assert len(idxs) == 1, f"missing or duplicate section {pfx!r} in {v_kind}/{target_slot}:\n{r_st.output}"
                    positions.append(idxs[0])
                assert positions == sorted(positions), f"sections out of order: {positions}"

                # Check refused automatic or invalid manual NEVER prints streak=0!
                if "manual" in to_break:
                    m_line = [ln for ln in out_lines if ln.startswith("manual:")][0]
                    assert "manual: invalid; unavailable:" in m_line and "streak=0" not in m_line
                if "automatic" in to_break:
                    a_line = [ln for ln in out_lines if ln.startswith("automatic:")][0]
                    assert "automatic: refused; unavailable:" in a_line and "streak=0" not in a_line
            finally:
                for cp in chmod_paths:
                    cp.chmod(0o644)

    # 6c: Check saver_state.json with boolean season (`{"state": "used", "season": true}`) and ledger recursion depths!
    edge_dir = work / "saver_bool_and_ledger_depth/picks"
    (edge_dir / "account_state").mkdir(parents=True)
    (edge_dir / "account_state/saver_state.json").write_text('{"state": "used", "season": true}', encoding="utf-8")
    r_bool_saver = runner.invoke(climod.cli, ["season-contract", "status", "--picks-dir", str(edge_dir)])
    saver_bool_line = [ln for ln in r_bool_saver.output.splitlines() if ln.startswith("saver:")][0]
    findings_notes.append(f"saver_bool_season_output: {saver_bool_line}")

    # Test ledger lines at depths 400..1100 to see if json.loads vs repr(dict) can ever disagree on RecursionError!
    depth_anomalies = []
    for depth in [400, 490, 495, 498, 500, 505, 800, 990, 1000, 1100]:
        nested = '{"active_streak": ' + "[" * depth + "0" + "]" * depth + "}\n"
        (edge_dir / "account_state/contest_ledger.jsonl").write_text(nested, encoding="utf-8")
        r_d = runner.invoke(climod.cli, ["season-contract", "status", "--picks-dir", str(edge_dir)])
        led_lines = [ln for ln in r_d.output.splitlines() if ln.startswith("ledger:")]
        if r_d.exit_code != 0 or len(led_lines) != 1:
            depth_anomalies.append((depth, r_d.exit_code, led_lines))

    shutil.rmtree(work, ignore_errors=True)
    summary = {
        "fs_invalid_cases": fs_results,
        "pairwise_check_combinations": pair_count,
        "status_broken_matrix_cases": status_matrix_count,
        "depth_anomalies": depth_anomalies,
        "notes": findings_notes,
    }
    (HERE / "probe2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    assert depth_anomalies == [], f"Ledger depth anomalies found: {depth_anomalies}"
    print("PROBE 2 SUMMARY:", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
