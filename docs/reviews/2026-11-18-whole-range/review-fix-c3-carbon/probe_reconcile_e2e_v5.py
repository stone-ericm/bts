"""Runner v5 for e2e & callers: tests concurrent reconciles both with fresh receipt dir
(demonstrating untouched receipt_io.ensure_dir FileExistsError race when two processes
create health_state/reconcile_receipts/<date> simultaneously) and with existing receipt dir
(asserting len(published_receipts) == 2). All other assertions unchanged."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
import time
from unittest.mock import patch

import probe_reconcile_e2e_v2 as v2
import probe_reconcile_e2e_v4 as v4

REVIEW_DIR = v2.REVIEW_DIR
REPO_ROOT = v2.REPO_ROOT
ET = v2.ET
RUN_ET = v2.RUN_ET


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] in ("--special", "--concurrent", "--callers"):
        return v2.main()

    sys.path.insert(0, str(REPO_ROOT / "src"))
    import bts.cli as cli_mod
    import bts.picks as picks_mod
    import bts.reconcile_receipt as rr_mod

    v2._install_no_network()
    out: dict = {}

    with TemporaryDirectory(dir=REVIEW_DIR, prefix="_e2e_tail_") as scratch_str:
        scratch = Path(scratch_str)

        # 4.4a Concurrent reconciles on fresh uncreated receipt dir (characterize receipt_io.ensure_dir race)
        p_conc_fresh = scratch / "i4_concurrent_fresh" / "picks"
        p_conc_fresh.mkdir(parents=True)
        v2._make_pick(picks_mod, p_conc_fresh, "2027-04-18", "hit", delivered_dm=True)
        v2._make_pick(picks_mod, p_conc_fresh, "2027-04-19", "miss", delivered_dm=True)
        picks_mod.save_streak(1, p_conc_fresh, saver_available=True, season=2027)

        pa = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--concurrent", str(p_conc_fresh), "350", "hit"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        time.sleep(0.05)
        pb = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--concurrent", str(p_conc_fresh), "0", "hit"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        oa, ea = pa.communicate(timeout=10)
        ob, eb = pb.communicate(timeout=10)
        assert pa.returncode == 0 and pb.returncode == 0
        ra, rb = json.loads(oa), json.loads(ob)
        assert ra["exit_code"] == 0 and rb["exit_code"] == 0
        s_fresh = json.loads((p_conc_fresh / "streak.json").read_text())
        assert (s_fresh["streak"], s_fresh["saver_available"], s_fresh["season"]) == (2, True, 2027)
        rec_fresh = rr_mod.discover(p_conc_fresh, "2027-04-20")
        if len(rec_fresh) == 1:
            assert "FileExistsError" in (ra["stderr"] + rb["stderr"])
        else:
            assert len(rec_fresh) == 2

        # 4.4b Concurrent reconciles when receipt date directory already exists -> MUST publish 2 receipts
        p_conc = scratch / "i4_concurrent_existing" / "picks"
        p_conc.mkdir(parents=True)
        (rr_mod.receipts_root(p_conc) / "2027-04-20").mkdir(parents=True)
        v2._make_pick(picks_mod, p_conc, "2027-04-18", "hit", delivered_dm=True)
        v2._make_pick(picks_mod, p_conc, "2027-04-19", "miss", delivered_dm=True)
        picks_mod.save_streak(1, p_conc, saver_available=True, season=2027)

        proc_a = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--concurrent", str(p_conc), "350", "hit"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        time.sleep(0.05)
        proc_b = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--concurrent", str(p_conc), "0", "hit"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        out_a, err_a = proc_a.communicate(timeout=10)
        out_b, err_b = proc_b.communicate(timeout=10)
        assert proc_a.returncode == 0 and proc_b.returncode == 0, (err_a, err_b)
        res_a, res_b = json.loads(out_a), json.loads(out_b)
        assert res_a["exit_code"] == 0 and res_b["exit_code"] == 0
        final_streak_doc = json.loads((p_conc / "streak.json").read_text())
        assert (final_streak_doc["streak"], final_streak_doc["saver_available"], final_streak_doc["season"]) == (2, True, 2027)
        published_receipts = rr_mod.discover(p_conc, "2027-04-20")
        assert len(published_receipts) == 2
        out["concurrent"] = {
            "fresh_dir_receipts": len(rec_fresh),
            "fresh_dir_stderr_combined": (ra["stderr"] + rb["stderr"]).strip(),
            "existing_dir_receipts": len(published_receipts),
            "proc_a": res_a["stdout"].strip(),
            "proc_b": res_b["stdout"].strip(),
        }

        # 4.5 Interrupt between correction writes and streak write, then retry
        p_int = scratch / "i4_interrupt" / "picks"
        p_int.mkdir(parents=True)
        v2._make_pick(picks_mod, p_int, "2027-04-16", None)
        v2._make_pick(picks_mod, p_int, "2027-04-18", "hit", delivered_dm=True)
        v2._make_pick(picks_mod, p_int, "2027-04-19", "miss", delivered_dm=True)
        picks_mod.save_streak(99, p_int, saver_available=False, season=2027)
        before_bytes = (p_int / "streak.json").read_bytes()

        with patch.object(picks_mod, "_write_streak", side_effect=KeyboardInterrupt("simulated mid-run interrupt")):
            r_int = v2._invoke_reconcile(cli_mod, picks_mod, rr_mod, p_int, feed_map={"2027-04-19": {"pick": "hit"}})
        assert r_int.exit_code != 0
        assert picks_mod.load_pick("2027-04-19", p_int).result == "hit"
        assert (p_int / "streak.json").read_bytes() == before_bytes
        int_receipts = rr_mod.discover(p_int, "2027-04-20")
        assert len(int_receipts) == 1 and int_receipts[0]["outcome"] == "raised" and int_receipts[0]["error"] == "KeyboardInterrupt"

        r_retry = v2._invoke_reconcile(cli_mod, picks_mod, rr_mod, p_int, feed_map={"2027-04-19": {"pick": "hit"}})
        assert r_retry.exit_code == 0 and "No scoring changes detected. Streak: 2" in r_retry.stdout
        rebuilt_int = json.loads((p_int / "streak.json").read_text())
        assert (rebuilt_int["streak"], rebuilt_int["saver_available"], rebuilt_int["season"]) == (2, True, 2027)
        out["interrupt_and_retry"] = "passed"

        # Item 5: Other callers parity (Base c2039541 vs HEAD c70effe9)
        base_extract = scratch / "base_c2039541"
        base_extract.mkdir()
        with subprocess.Popen(["git", "archive", "c2039541", "src"], cwd=REPO_ROOT, stdout=subprocess.PIPE) as arch:
            assert arch.stdout is not None
            with tarfile.open(fileobj=arch.stdout, mode="r|") as tf:
                tf.extractall(base_extract, filter="data")
            assert arch.wait() == 0

        w_base = scratch / "callers_base"
        w_head = scratch / "callers_head"
        w_base.mkdir()
        w_head.mkdir()
        p_base_callers = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--callers", str(base_extract / "src"), str(w_base)],
            capture_output=True, text=True, check=True, timeout=15,
        )
        p_head_callers = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--callers", str(REPO_ROOT / "src"), str(w_head)],
            capture_output=True, text=True, check=True, timeout=15,
        )
        snap_base = json.loads(p_base_callers.stdout)
        snap_head = json.loads(p_head_callers.stdout)
        assert snap_base == snap_head, f"Caller parity mismatch!\nBase: {snap_base}\nHead: {snap_head}"
        out["item5_caller_parity"] = {
            "equal": True,
            "checked": list(snap_head.keys()),
            "update_streak_steps_byte_identical": len(snap_head["update_streak_snapshots"]),
        }

    (REVIEW_DIR / "e2e_tail_results.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
