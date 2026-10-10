"""The cap commit (apply ONLY after the fresh review's plain SIGN; its own short verbatim review follows):
CAP_H 165 -> 237.6901 (the lead's derivation 137.6901 + Eric's 100, the manager's ruling 2 of 2026-10-09), the ledger
docstring, the C1 cycle test pinned to the allowance row, the screen test that equated the stage-two row with the live cap.
Run from the worktree root. Every span must match exactly once; nothing is written otherwise."""
from pathlib import Path
import ast
edits = []
def patch(path, pairs):
    p = Path(path); t = p.read_text()
    for old, new, label in pairs:
        n = t.count(old); assert n == 1, f"{label}: {n} matches"
        t = t.replace(old, new); edits.append(label)
    if path.endswith(".py"):
        ast.parse(t)
    p.write_text(t)

patch("scripts/audit/c1/ledger.py", [
    ("The cycle cap is 100 CPU-hours with a stop-and-report at 50 (resumed only by Eric's written acknowledgement); Eric raised\nthe cap to 165 for the C2 framing screen's stage two (register row C2-framing-stage-two-cap, 2026-10-08).",
     "The cycle cap is 100 CPU-hours with a stop-and-report at 50 (resumed only by Eric's written acknowledgement); Eric raised\nthe cap to 165 for the C2 framing screen's stage two (register row C2-framing-stage-two-cap, 2026-10-08), and allowed\n100 more CPU-hours for the 2026 framing test (register row C2-framing-2026-allowance, 2026-10-09: his words\n\"100 cpu hours\"; the cap 237.6901 = the prior effective shared total 137.6901 + 100 is the lead's derivation).",
     "ledger docstring"),
    ("CAP_H = 165.0                  # Eric, register row C2-framing-stage-two-cap (2026-10-08): raised from 100",
     "CAP_H = 237.6901               # register row C2-framing-2026-allowance (2026-10-09): 137.6901 prior + Eric's 100 (was 165)",
     "CAP_H"),
])
patch("tests/scripts/test_c1_cycle.py", [
    ('''def test_the_cap_is_erics_ruled_165():
    """Eric raised the shared cap from 100 to 165 (register row C2-framing-stage-two-cap, 2026-10-08). The gate tests
    above are relative to the constant; this pins its value to his ruling, so a silent edit of either goes red."""
    from scripts.audit.c1 import admission
    cells = admission.row_cells((admission.REPO / "docs/audit/2026-09-22-exposure-register.md").read_text(),
                                "C2-framing-stage-two-cap")
    assert ledger.CAP_H == 165.0
    assert cells and cells[3].split()[:1] == ["Eric"]
    assert "RAISE the shared C1/C2 compute cap from 100 to 165 CPU-hours" in cells[2]''',
     '''def test_the_cap_is_the_allowance_rows_cap():
    """The shared cap follows Eric's latest ruling in the real register: the 2026 framing test's allowance row
    (C2-framing-2026-allowance, 2026-10-09; his words "100 cpu hours", the cap 137.6901 + 100 = 237.6901 the lead's
    derivation). The stage-two row (165, 2026-10-08) stays as history. The gate tests above are relative to the
    constant; this pins its value to the row, so a silent edit of either goes red."""
    from scripts.audit.c1 import admission
    from scripts.audit.c2_framing import f26
    register = (admission.REPO / "docs/audit/2026-09-22-exposure-register.md").read_text()
    allow = f26.allowance(register)
    assert allow is not None and allow["cap"] == ledger.CAP_H == 237.6901
    assert allow["budget"] == 12.0 and allow["first_unit_stop"] == 4.1
    cells = admission.row_cells(register, "C2-framing-stage-two-cap")
    assert cells and cells[3].split()[:1] == ["Eric"]
    assert "RAISE the shared C1/C2 compute cap from 100 to 165 CPU-hours" in cells[2]''',
     "c1 cycle cap test"),
])
patch("tests/scripts/c2_framing/test_screen.py", [
    ('''def test_the_launchers_cap_is_the_cap_in_erics_row():
    """The launcher's cap constant equals the cap Eric ruled (row C2-framing-stage-two-cap), read from the real register,
    and both are 165: a silent edit of either goes red."""
    assert S.stage_two_release((ROOT / S.REGISTER_REL).read_text()) == (S.ledger.CAP_H, 20.0) == (165.0, 20.0)''',
     '''def test_the_stage_two_row_still_reads_165():
    """The stage-two cap row (C2-framing-stage-two-cap, 2026-10-08) is history: it still reads 165 and 20 per seed. The
    launcher's cap constant now follows the 2026 framing test's allowance row (tests/scripts/test_c1_cycle.py), so
    stage two's own gate (`stage_two_allowed`, cap == CAP_H) is closed, as stage two is."""
    assert S.stage_two_release((ROOT / S.REGISTER_REL).read_text()) == (165.0, 20.0)
    assert S.ledger.CAP_H != 165.0''',
     "screen cap test"),
])
print("applied:", edits)
