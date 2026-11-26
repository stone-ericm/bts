"""Measure the contract's DM strings and the proposed replacements at the code's bound."""
from pathlib import Path

contract = Path("docs/reviews/2026-11-18-whole-range/kickoff-fix-e.md").read_text()
rows = {}
for line in contract.splitlines():
    cells = line.split(" | ")
    if len(cells) >= 4 and cells[0][2:] in {"S1", "S2", "S3", "S4", "S5", "S6", "M1", "M2", "M3", "M4", "M5"}:
        rows[cells[0][2:]] = (cells[1] if cells[2].startswith("unchanged") else cells[2]).strip("`")
date = "2027-12-31"
detail = "last accepted fetch " + "x" * 128 + ", about 17759548.98 hours ago"
envelope = "BTS health WARN attention:\n- [contest_state] "
sequences = ("S1 S2 S3 S4 S6 S5", "M1 M2 M3 M4 S6 M5")


def length(seq):
    return len(envelope + " ".join(rows[row] for row in seq.split()).format(D=date, detail=detail))


print("draft", [length(seq) for seq in sequences])
rows.update(
    S3="For a {D} slate, file different MLB values with bts set-contest-streak --date {D} --streak N --best-streak B --saver S --picks-dir DIR. Use the scheduler's absolute picks path for DIR.",
    S4="Write only on {D} ET for that slate. For an earlier slate, wait for MLB settlement or no entry and follow checklist B8b(1).",
    S6="If MLB shows this season's saver used, check bts saver-state --picks-dir DIR. For active or not_earned, run bts saver-state --use --picks-dir DIR. For uninitialized, follow checklist B8a.",
    M4="Only on {D} ET, file different MLB values with bts set-contest-streak --date {D} --streak N --best-streak B --saver S --picks-dir DIR. Use the scheduler's absolute picks path for DIR.",
)
print("replacement", [length(seq) for seq in sequences])
