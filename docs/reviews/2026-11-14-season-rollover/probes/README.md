# Reviewer probes kept as raw material for the season-rollover follow-up

Copied unchanged on 2026-11-14 from the reviewers' scratch directories (Codex round 2 from `/tmp`, Gemini round 2, the final Codex review). They were written against commit `dbb78b1` or `c339361` and hard-code scratch paths; they are evidence and a starting point for RED tests, not part of the test suite.

- `gemini_r2_season_start_notes.py`: last season's `contest_streak.json`, `saver_state.json` and `streak.json` read on 2027-03-24; first-day health alarms (S1, S2).
- `codex_r2_notes.py`: the same carry-over with the tracked policy (single at 0.79 and streak 0, skip at streak 18); the first-fetch health alarm (S1, S2).
- `codex_r2_weather.py`: a weather sidecar breaking feature computation (S3).
- `codex_r2_cold.py`, `codex_r2_cold_gate.py`, `gemini_r2_cold_start.py`: the cold start through training, selection and delivery on synthetic data.
- `codex_final_test_adversarial.py`: the final review's five probes, including the late-lineup day that ends with no pick (S6).
