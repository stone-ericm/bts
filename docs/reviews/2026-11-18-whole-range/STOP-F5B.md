# STOP: change set F5b (the entry watchdog every 15 minutes all day; early runs keep their own ledger), after round 2

Written 2026-11-26 by `bts-wr-fresh13`. **Not merged. `main` is unchanged** (the watchdog still runs `*/15 10-23`: a game before about 11:15 ET gets fewer entry checks, one before 10:05 none; BL-O4 stays open). Branch `fix/wr-f5b` (pushed): contract `kickoff-fix-f5b.md` with Amendments 1 and 2; round 1 `e66cc74c`, round 2 `88b0427f`; records. Under ruling R68 BL-O4 now goes to Eric.

## Verdicts
- Contract check: BUILD WITH THESE EDITS `WRCHKF5B-e6e039d6` (`contract-check-f5b/`).
- Round 1: Codex BLOCK `WRF5BREV1-9abd5a51` (B1 an early all-clear lost on a pitch-only correction, raw time strings; B2 receipt suppression with a 64 KiB bound and the response time; B3 malformed early marker fields trusted, a cutoff overflow); Carbon stopped (R70).
- **Round 2 (bounded, R70): Codex BLOCK `WRF5BREV2-4cc7ff3d`** (`review-fix-f5b-codex-r2/`). Every round-1 finding is fixed (eleven targeted cases red on `e66cc74c`, green on `88b0427f`); manual and original-hours runs byte for byte the base's (28 cases); 77 F5b and 1,255 existing tests. Two defects the change set introduced:
  1. **A slow live all-clear can hide the last early warning failure.** A manual recovery writes `confirmed`, then its all-clear takes 30 seconds; in between the pick is replaced and the 09:45 early warning for the replacement fails; the manual receipt's `published_at` (09:45:20) is after the early observation, so after a later historical displacement of the live marker the early CRITICAL is suppressed. Round 2's `published_at` rule (my ruling for B2) assumed milliseconds between a marker write and its receipt; a send sits in between. The reviewer's fix: record the live marker's write-completion time in the existing seal's metadata (live marker, receipt and seal bytes otherwise identical) and compare that.
  2. **An eligible early check can hang on a writerless units capture file** (`entry_receipt.load_units` opens it with a blocking open, in qualification and in receipt construction). The base exposes this only to an eligible live check (F1 documents it); F5b exposes it to scheduled pre-10:00 checks on slates with no original slot. The reviewer's fix: non-blocking open and a regular-file check in both early loads, the existing unavailable-capture handling.
  One text edit (the receipt guide's published-time caveat).
- Carbon round 2: stopped before its report (R70 item 5).
- Gate on `962a9b36` (04:57 to 05:26): 11433 passed , failing set exactly the pinned 53 (R71's known-flaky pair did not fail this time).

## The session's own mistakes on F5b
1. Round 2's B2 fix was my ruling (use `published_at`), chosen to avoid a new file; I wrote its limit as "milliseconds" without checking what runs between the marker write and the receipt's publication (the all-clear send).
2. The units-capture read was in round 1's code path for live runs; neither my contract nor my brief named every read an early run newly reaches (the contract check listed picks, decisions and markers only).

## Ways forward (for Eric, R68)
1. **Accept F5b as it stands, with both findings written as limits** (an early warning failure followed within one send of a live recovery and a later historical displacement can go unreported at end of day; a named pipe placed at a units capture path can hang an early check until the 14-minute run limit, as F1 already states for live checks). Merge on the gate.
2. **One more small follow-up, built on the reviewer's two fixes** (seal metadata for the live write time; non-blocking units reads in early runs), each RED first, one fresh Codex check limited to them.
3. **Accept BL-O4 as a written limit for 2027:** keep `*/15 10-23`; on a day whose first game is before about 11:15 ET the operator confirms the entry in the MLB app by hand (a checklist line, January with Eb); the next morning's ENTRY UNVERIFIABLE on such a day is expected.
