"""Accept pytest's rewritten-assert output and finish the five C27 cases."""

import probe_mutations as probe


def detected(name):
    output = (probe.WORK / f"mutation-{name}.log").read_text()
    assert "E   assert " in output or "AssertionError" in output, name
    assert "ERROR collecting" not in output, name
    failures = [line for line in output.splitlines() if line.startswith("FAILED ")]
    assert failures, name
    print(name, "detected", len(failures), "failed assertions", flush=True)


for label in probe.CASES:
    if label in ("legacy-streak-used", "stored-text-unchanged"):
        try:
            probe.run_case(label)
        except AssertionError as error:
            assert error.args == (label,), error
    detected(label)
