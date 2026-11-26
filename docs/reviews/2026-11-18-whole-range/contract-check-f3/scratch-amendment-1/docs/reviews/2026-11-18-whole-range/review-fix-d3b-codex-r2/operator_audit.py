"""Check the new operator text against actual shimmed recovery behavior."""
from shared import ROOT, RECOVER, box, invoke, record

incident = (ROOT / "INCIDENT.md").read_text()
restore = incident.split("**Box/filesystem loss", 1)[1].split("## 5.", 1)[0]
pieces = ("first confirm the scheduler is stopped", "disabled, and keep it so through restore",
          "install-restic-hetzner.sh", "restic restore <SNAPSHOT_ID> --target /", "bts data sync-from-r2")
positions = [restore.index(piece) for piece in pieces]
assert positions == sorted(positions)
print("Restore hold precedes credentials, restic install, drills, file replacement and artifact sync.", flush=True)
claude = (ROOT / "CLAUDE.md").read_text()
assert "Recovery then reports that this attempt's record says committed." in claude
checklist = (ROOT / "docs/ops/2027-season-start.md").read_text()
b8c = next(line for line in checklist.splitlines() if line.startswith("| B8c |"))
assert b8c.index("boot link disabled through the switch") < b8c.index("deploy the approved current SHA")
assert b8c.index("Save and verify the crontab backup") < b8c.index("Confirm that attempt's PASS and HEAD")
print("Checklist keeps the old scheduler's boot link disabled on entry to normal boot.", flush=True)

assert "on recovery whenever the attempt record does not hold a verified 2xx" in incident
for label, initial, observed in (
    ("missing record", None, "no phase-2 record is present"),
    ("committed without page evidence", record("committed"), "record says committed"),
    ("different run", record(owner="832"), "different attempt owns"),
    ("matching begun, no response retained", record(), "DEPLOY FAILED at deploy-step-incomplete"),
    ("matching contained without 2xx", record("contained", tail="head verified verified unverified"), "containment re-checked"),
    ("matching contained with 2xx", record("contained"), "containment re-checked"),
):
    with box(initial) as path:
        state, _ = invoke(path, label, RECOVER)
        assert observed in state["out"]
        expected = label in ("matching begun, no response retained", "matching contained without 2xx")
        assert bool(state["posts"]) == expected
