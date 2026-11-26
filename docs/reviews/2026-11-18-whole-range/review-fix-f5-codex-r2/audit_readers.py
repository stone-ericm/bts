"""Run the local audit acquisition with and without the new watchdog files."""
from pathlib import Path
from tempfile import TemporaryDirectory

from support import DAY, Replay, no_network, source
from scripts.audit.pvr_threshold_bootstrap import load_days
from scripts.audit.season_ledger.acquire import acquire
from scripts.audit.season_ledger.bundle import open_bundle


if __name__ == "__main__":
    source("f5")
    with TemporaryDirectory() as directory, no_network():
        root = Path(directory)
        world = Replay(root / "snapshot" / "data")
        world.commit_at("06:10", at="04:55")
        world.set_marker({"date": "2027-03-23", "status": "dm_failed"})
        opts = {"snapshot_root": root / "snapshot", "dates": [DAY], "fetch": lambda day: b'{"dates":[]}',
                "now_utc": lambda: "2027-03-24T16:00:00Z"}
        acquire(out_root=root / "before", **opts)
        score = root / "slate_slots.csv"
        score.write_text(f"outcome,date,p\nhit,{DAY},0.72\n")
        baseline_scores = load_days(score)
        checked = world.invoke("05:15")
        assert checked["receipt"]["early_cron_run"] is True
        assert world.marker.with_name("pick_entry_check.2027-03-23.json").is_file()
        acquire(out_root=root / "after", **opts)
        assert open_bundle(root / "before") == open_bundle(root / "after")
        assert load_days(score) == baseline_scores
        print("Season-ledger acquisition and bootstrap CSV reader were unchanged with a sealed early receipt and dated marker present.")
