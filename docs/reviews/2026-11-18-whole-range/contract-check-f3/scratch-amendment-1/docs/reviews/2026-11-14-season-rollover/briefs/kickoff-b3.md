# Kickoff B3: a third correction, found by the test gate (do it after B2, same handover)

The gate on the merged candidate failed one existing test outside the list I gave you: `tests/test_incident_2026_08_30.py::test_stale_refresh_result_never_sends_after_cutoff`. It fails on your branch alone (`75994d3`); it passes on `main`. That my list of "tests that must stay green" left this file out is my omission.

## What happens
In that scenario the pick is refused at the chokepoint after its cutoff and archived, so no pick file remains. Your change then (correctly, by the contract) treats the rest of the day as "no pick file, no skip": the extra attempts run, select nothing, and the new end-of-day alert is dispatched. The test reads the LAST alert dispatch (`mock_alert.call_args`) and expects the `late_delivery` refusal notice; it now sees `missed_pick`.

## Required
1. **The alert's wording must be true in this case too.** "No pick was selected today" is false when a pick was selected and then refused. Use one wording that is true for both: the day's checks and remaining attempts are over and no pick is on file for today. Keep it one alert, same source and level, same transport.
2. **Update the incident test without weakening it.** It must still assert: no pick DM; a CRITICAL `late_delivery` refusal notice was dispatched (search all dispatch calls, not only the last); the refused-delivery archive exists; the state's `pick_locked` is false and the refusal names Steven Kwan; "DELIVERY REFUSED" on stderr. Add: the day-end alert is dispatched after the refusal notice, exactly once.
3. **Run every test file that exercises the scheduler, not only my list.** Find them yourself (`grep -rl "bts.scheduler\|from bts import scheduler\|run_day\|run_single_check" tests`), run them all with the focused command form, and report the full list with per-file counts. Any other failure: report it; fix it only if it is the same cause (a test that assumed the day ends silently after a refusal or with no pick), and say what you changed and why for each.
4. State in the handover what an operator now receives, in order, on a day when a pick is refused after its cutoff and nothing later is selected.

Put B2 and B3 in one handover: `.codex-review/sr/handover-b2.md` (under 45 lines now), marker `SRB2-…` as before.
