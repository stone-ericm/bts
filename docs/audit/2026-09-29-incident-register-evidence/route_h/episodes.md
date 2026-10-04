# W1.5 working file — candidate episodes (route H + docs + memory pointers; route R not yet run)

Legend: prelim disposition OI = observed_incident, DL = deployed_latent_defect, NM = near_miss_control_held, PS = pre_ship_exclusion, UC = unresolved_candidate, EX = out of scope. Tier A/B/P(pending). Evidence: MO machine obs, OR contemporaneous operator report (commit body/memo ≤48h), INF inference.

## Early season — manual-pick / Pi5-orchestrator / Fly / Hetzner cutover (3/29 → 4/14)
| id | date(s) | episode | cls | disp | tier | commits / sources |
|---|---|---|---|---|---|---|
| E01 | 3/30 | live predict used each team's own pitcher as opponent after games started | D | UC | P | 245a971 |
| E02 | 4/01 | check_hit failed on wrong game_pk/batter_id (cron grading) | G | UC | P | 50d4ee5 |
| E03 | 4/01 | live picks on a stale season parquet | D | UC | P | b6b1865 |
| E04 | 4/01 | orchestrator tier failures (Mac predict-json stdout, Alienware cp1252) | L | UC | P | 1929f5a cb4c1cc 64ae365 |
| E05 | 4/03 | first scheduler day: later lineup did not re-trigger predictions | D | UC | P | f5ce1ed |
| E06 | 4/03 | projected-lineup fallback included pinch-hit subs | D | UC | P | 515cfa3 |
| E07 | 4/04 | duplicate public "Streak reset to 0" reply | A | OI? | B | 619d109 |
| E08 | 3/29–3/30 | null game_pk pick files → reconcile HTTP 400 | G | UC | P | a6cae41 (ledger: 3/29–3/30 null game_pk) |
| E09 | 4/04 | Díaz pick never locked/posted: postponed game's projected batter held the gap check open | D | OI | A | 7638af7 6d36adb |
| E10 | 4/04 | restarted scheduler ignored an already-posted pick (replayed checks, no polling) | L | UC | P | 823ccb3 |
| E11 | 4/04 | 1am cron re-applied an already-graded result (streak 4 vs 2) | G | OI | A | 5e02676 |
| E12 | 4/05 | scorecard fielder labels | U | UC | B | 27be68e d99c5ef |
| E13 | 4/06 | scheduler could sleep past first pitch; restart past deadline skipped fallback post | D | UC | P | 0632976 a002b04 |
| E14 | 4/06 | dashboard "Final — pick missed" early on a cross-game double | U | UC | B | 0ba01e1 |
| E15 | 4/06–4/08 | cross-game double graded on one game (+1 not +2); NameError in polling after the fix | G | UC | P | 1d9c199 c7c2f73 |
| E16 | 4/08 | live prediction timed out scanning ~23K raw feeds | L | UC | P | e1aaf61 |
| E17 | ~4/10 | Fly shadow instance posted picks publicly as a 2nd bot | D | OI? | A | 5075fba |
| E18 | 4/11 | Hetzner cutover deploy pull / restart silently failing | S | UC | P | 870cede cda8c1c |
| E19 | 4/11–4/12 | every Hetzner cron job dead since cutover (dash, no `source`) | L | OI | A | d5c466c |
| E20 | 4/11–4/12 | two scheduler units running; orphan respawned after deploys | S/L | OI | A | 8193909 |
| E21 | 4/12 | fallback deadline landed 3 min after the DD's first pitch | D | OI | A | 6fd61f9 |
| E22 | 4/12 | confirmation counter logged "0 new" while lineups flipped the pick | U | OI | B | bd8bcd3 |
| E23 | 4/12 | shadow file overwritten by production pick (DISAGREES → AGREES) | R | OI | A | 634768e 80fd31f |
| E24 | 4/12 | no live scorecard for a cross-game double; header never rendered | U | OI | B | 6168e1d 1e4859d |
| E25 | 4/12 | fallback posted a stale cached pick (Donovan) over a fresher Turner | D | OI | A | dd25664 |
| E26 | 4/10–4/13 | in-place sort reordered the training frame; live model silently changed | D | DL | P | 6cd1761 |
| E27 | 4/14 | 3am pull/build/preview chain failed; blend 2–3 days stale | L | OI | A | 28732c2 |
| E28 | 4/14 | deploy pull failed on uv.lock drift; stale code while CI green | S | OI | A | 6bffd63 |

## Hetzner era, pre-journal (4/15 → 5/10)
| E29 | 4/15 | 02:00 reconcile reset the streak to 0 (walk broke on today's preview) | G | OI | A | 1d61908 |
| E30 | 4/15 | shadow-report mislabelled DD-aware rate as "Production P@1" | U | UC | B | f5e51dc |
| E31 | 4/22 | first deploy-branch push fired no workflow (paths filter) | S | OI | B | f6fcc16 |
| E32 | 4/22 | heartbeat false Healthchecks failures (cold-wake predict; polling sleep) | A | OI | A | 0830ef4 c43869d |
| E33 | 4/22 | result polling keyed to the primary game; slept through the DD game | L/G | OI | A | b681f8a |
| E34 | 4/22–4/23 | watchdog SIGABRT killed the scheduler every ~30 min (NRestarts=21) | L | OI | A | 684c160 |
| E35 | 4/23 | EOD clean exit → Restart=always relaunch loop (NRestarts 0→7 in 25 min) | L | OI | A | ddee3db |
| E36 | 4/24 | lineup-status cell blank while the picked team fielded | U | OI? | B | 140940b |
| E37 | 4/21?–4/28 | silent no-op deploys (workflow pulled origin/main) | S | OI | A | a5af424 |
| E38 | 4/28 | memory_growth false CRITICAL; one unexplained restart | A | OI | B | 4e221c8 |
| E39 | 4/29 | bpm feature slowed prediction >5 min → stale heartbeat → HC failure DM | L | OI | A | a512851 (PR #5) |
| E40 | 4/29–4/30 | bpm missing from inference: every tick crashed; 4/30 pick never locked/posted | D/L | OI | A | ee4190f |
| E41 | 4/30 | blend_training / bluesky_post health false positives | A | OI? | B | f0c3af9 |
| E42 | ~5/01 | realized_calibration daily CRITICAL on DD-attribution-biased signal; pooled across iterations | A | OI? | B | b08769d c5256e6 |
| E43 | 5/05–5/26 | postponed-game cluster: stale postponed pick stays locked; void grading; polling all-final gate; candidate filtering; lock-status fallback | D/G | OI | A | 7b701c9 (+PR #24/#25) a142c14 68e4cdb (PR #74) 0799bf2 abbfdc5 |
| E44 | 4/10–5/08 | shadow results never reconciled (27/29 unresolved) | R | OI | A | 8671709 (PR #56) |
| E45 | 5/08–5/11 | dashboard pill/banner/DD scorecard showing a wrong state | U | UC | B | 96b6e4f 5a1beb7 425f62f |
| E46 | 5/08 | AppleDouble `._*` files crashed realized_calibration | A/L | OI? | B | 52db4d7 |
| E47 | 5/09 | shadow generation without watchdog pings → restart loops | L | OI? | A | 150ed4c |
| E48 | 5/09 | deploy canary failed on the scheduler's auto-restart state | S | OI | B | ac4a959 (+ run 25614229839 failure) |
| E49 | 5/09 | restart loop after day completion (wake built on today's date) | L | OI? | A | 864d3aa |

## Journal era (5/11 →)
| E50 | 5/11–5/26 | live-forward stream integrity: stale snapshots, post-game writes, resolver failed state, provenance drift, pending rows | R | OI | A? | 09ee616 429a998 9340218 2efd40a c511a03 3710227 f03a98b |
| E51 | 5/15 | int(NaN) crash when pitcher_id missing (preview) | L | OI? | P | 285f3a5 (PR #98) |
| E52 | 5/21–5/27 | scheduler OOM-killed during inline shadow; escalation + memory alerts + restart_spike wrong advice | L/A | OI | A | 43ddf0c a3df746 3a8d1bd 279f190 ac09137 |
| E53 | 5/22 | Hoerner force-delivered before lineups (should_lock=False) | D | OI? | A | 324c4a2 (+5/28 audit memo) |
| E54 | 5/23 | DD-shortfall WARN labelled pair correlation | A | UC | B | 05d6f63 |
| E55 | 5/27 | dashboard /health timed out; restart needed | U/L | OI? | B | c127673 |
| E56 | 5/29–6/06 | contest state silently frozen at 7 while real streak 0; no alert | G/A | OI | A | 365ea40 (PR #142, spec d383db5) |
| E57 | 6/06–6/07 | EOD contest_state nightly false CRITICAL | A | OI | B | 58c9adc |
| E58 | 6/06–6/10 | fetch-contest-streak daily false "failed" DM | A | OI | B | 8cd7207 |
| E59 | 6/09–6/11 | fable5 audit latent batch (incl. E3 missed-pick alert never implemented; platoon_hr NaN in live inference; leaky shadow feature; no-games thrash) | many | DL | P | 925db1d ce5f51d aa6b7b1 0f2bc3c b302935 af08426 082e5b0 dfcf655 86ff4f1 c27480e 8550b80 63fcb67 736ea8f 6d929f8 0794116 e1594e3 a7663ca 3a6e48b 40a091c 8e867fc bb7ccc1 6471f40 d94a51f 88234ae |
| E60 | 6/10 | CI deploy gate blocked a deploy (lockfile/TZ) | S | OI | B | 8c93799 279d423 |
| E61 | 6/11 | delivered pick never entered; nothing alerted until 6/12 | E | OI | A | 4f13eb3 |
| E62 | 6/12 | entry check v1 false-flagged the entered pick (settled-only endpoint) → cron disabled | E/A | OI | B | 2d68102 a6ec548 720651a |
| E63 | 6/11–6/17 | decision/dashboard streak 10 vs real 8; contest fetch discarded current activeStreak / not_hit rows; false noon WARN | G | OI | A | 2b4ff1d 709e68b 9a3e8ed 4bdfdb8 56f9726 (PR #143) |
| E64 | 6/17 | live-forward resolver stalled on a suspended/resumed game (CRITICAL) | R/X | OI | A | 0dca845 02abfcc |
| E65 | 6/18 | correct skip day produced no signal | A | OI | B | 9dc8ee4 |
| E66 | 6/21 | GH #144: check-results/polling could score undelivered previews on skip days | G | UC | A | d929479 d6ae318 25aeaee 62cfe4e 7bd2934 add1b3f 5fa36c3 (issue #144) |
| E67 | 6/29–6/30 | suspended-game scoring used resumed-portion PA (scorer, resolver, shadow backfill) | G/R | DL | P | a364b11 5801002 6e3f626 83a5cad |
| E68 | 6/30 | 3am cron could publish the R2 manifest over a failed rebuild | S | DL | P | b0420a0 |
| E69 | 7/01–7/02 | skip-day dashboard contradictions (first live flip) | U | OI | B | 258aaa4 2c62d72 |
| E70 | 7/04 | deep authenticated scrape stopped (ToU) | — | EX | — | config decision |
| E71 | 7/06 | premature "not entered" DM on a deferred, undelivered DD | E/A | OI | A | af6329f 540b1ab |
| E72 | 7/08 | DD leg never entered after a single alert (cost +1) | E | OI | A | 8bceda1 a6ec548 |
| E73 | 7/09–7/10 | gpt-5.6-sol audit latent batch (F2 projected-DD early lock ×11 no harm; F3; F4; F6 realized_calibration dead in prod; F8; F10 derivation lost; F12 unit drift; F13 scoring race; F15; R4) | many | DL | P | b25f348 a349ac5 bc56439 a081bab 40c0094 a374fa4 3a09b2c 49f2538 cca6761 ede5c28 9643d83 0ea2132 6d74d39 |
| E74 | 7/10–8/09 | shadow result stranded a month; no alert | R/A | OI | A | 4f0257a 41b2bb1 9ebad52 |
| E75 | 7/12 | eve-of-break restart loop + 47-DM storm + confounded pvr CRITICAL + blind restart_spike | L/A | OI | A | 9551818 ec242da 230f65c |
| E76 | 7/12–7/13 | DD-leg shortfall invisible to monitors | A | OI | B | a275399 |
| E77 | 7/16 | singleton-slate gap: game moved up, lone check at first pitch, pick undelivered | D | OI | A | 7b70da7 (docs) — UNFIXED |
| E78 | 7/28 | stale preview on a skip day; scoreable gate held | D/G | NM | B | memory pointer; verify in data |
| E79 | 8/03 | unretried schedule discovery (3am data pull path) | X | DL | P | 950c081 |
| E80 | 8/09 | decision records lacked state provenance, dropped second candidate | R | DL | B | 5216ea1 |
| E81 | 8/09 | mdp_policy_alignment bin collapse WARN (chronic) | — | EX | — | W1.4b read, not an incident |
| E82 | 8/11 | MLB auth/login flap → failed cron + DM advising cookie re-capture; park_drag table stale | A/X | OI | A | 404358d |
| E83 | 8/13 | silent pass: Warmup classification-locked the undelivered pick; no missed-pick alert | D/A | OI | A | 1b50b78 224ddce |
| E84 | 8/27, 8/29, 8/30 | deferral abandoned enterable picks; 8/30 DM after cutoff; 10.6-min cached-feed sleeps; health read it healthy | D/L/A | OI | A | ac0ce8d 67338cd c0c0a97 0c8ed32 3697512 2ff2db9 314154d |
| E85 | 9/03 | reach-57 all-skip idle once 57 unreachable | P | OI | A | 0abf503 eb010fd |
| E86 | 9/14 | `pick_delivery = private` did not silence the entry-check cron | E/A | OI | B | config only (crontab edit) |
| E87 | 9/16 | `stale_pick_snapshot.*` recaptures in the official live-forward root (private mode) | R | UC | P | memory pointer |
| E88 | 5/01–7/04 | C-01 leaderboard parser stored active streak in all_season/all_time rows | R | OI | A | 44df03f |
| E89 | 5/10, 8/20 | C-03 reconcile applied post-cutoff MLB re-scoring to local files | G | OI | A | ce6676d (main only) |
| E90 | 6/05 | MacBook loss: `pooled_bins_run` profiles lost; shipped policy cannot be re-solved | S | OI | A | memory/CLAUDE.md; plan W0.1 |

## Latent / unfixed (contracts in design §10)
| L01 | grading Pass rules (zero AB/SF, DNP, suspended) | G | DL | A | ledger spec §12; rules pinned |
| L02 | same-day replay rollback in reconcile | G | DL | P | reconcile-cutoff r1 #3 |
| L03 | postponed cached-fallback delivery | D | DL | P | 7/10 F1 deferral |
| L04 | scoring crash/restart double application | G | DL | P | 7/09 deferral |
| L05 | 8/30 deferrals: delivery_unknown; feed-file validation/atomic downloads; in-transport deadline | D | DL | P | 8/30 memo |
| L06 | no independent day-outcome watchdog; no-games-day early return bypasses 5b/EOD health | A | DL | P | optimization-ideas 8/14 |
| L07 | C-03 best-effort residuals | G | DL | P | corrections C-03 |
| L08 | unretried Open-Meteo fetch | X | DL | B | 8/03 note |
| L09 | private_locked/locked_unconfirmed + failed decision write → no entry alert | E | DL | P | 7/06 memo |

## Added from the runtime-negative review and the docs-only sweep (slice E)
| E91 | 4/11–7/10 | ops state (data/picks markers, health_state) existed only on the box — no backup until restic (7/10) | S | DL | A | 78c424a (body) |
| E92 | 5/09–9/14 | live-forward capture idled at pending_pick on skip days (11 of 79 decision days) until D8 | R | DL | B | 155f1e7 (body) |
| E93 | 4/11–4/12 | 3am cron chain's log redirect captured only the last command (pull/build/sync output dropped) | A/S | DL | B | 715c932 (body) |
| E94 | 3/29–3/31 | first three days' pick files created retroactively on 3/31 (evidence provenance note) | S | NM? | B | 1c2efaa (body) |
| E95 | 5/08–5/09 | operator backfill on the box rewrote 28 shadow result files (state repair) | R | OI | A? | 0141a63 293ce51 |
| E96 | 5/09 | shadow prediction starved the systemd watchdog; shadow_model=false set on the box (5/09 shadow gap) | L/R | OI | A | d01839f 150ed4c |
| E97 | 5/10 | #16 live-forward official 5/10 artifact could not be captured | R | OI? | B | d0141db |
| E98 | 6/22 | a deploy on a live skip day muddied the skip-day read | S/R | OI | B | a151ebf |
| E99 | 6/29–7/01 | deploy SIGTERMs logged as unit failures; SuccessExitStatus=143 applied box-side (unit drift until repo sync) | A/S | OI | B | 7636c11 48e8485 |
| E100 | 6/11, 7/10 | test failures blocked deploys (gate held) | S | NM | B | df01882 6919e19 (+ runs 27321348039, 29114359533) |
| E101 | 8/30–9/03 | 8/30 fix rebased onto a stale base → wrong run_single_check ordering; caught before deploy | S | NM | B | 3ec2fc5 |
| E102 | 6/24 | local replay 15 vs contest 13 (local state authority) | G | OI | A | 6db921c (part of the E63 class) |
| EX2 | 5/23 → | MDP policy maps live picks to Q0 (bin collapse) | P | EX | — | d54bfad f23e9a8 510ae4e f23010f — belongs to W1.4b's owed bin-collapse read |
| EX3 | 4/14 | research-audit cloud fleet: silent retrieve failure left zombie boxes | — | EX | — | d3e5082 d868eff (not the production service) |

## Added from the diff-level negative review (N1/N2, adjudicated 9/29)
| E103 | 3/29–4/01 | local grading had no streak-saver rule (any miss reset to 0) | G | DL | B | 2be445e (could not fire: streak < 10) |
| E104 | 4/01–4/04 | orchestrator worker tiers ran whatever code was checked out (no pull before predict) | S | UC | P | 5207a09 |
| E105 | 4/03–5/xx | public-post / dashboard display defects (DD post omitted 2nd batter; DD inning label; today's row colour; locked-unposted PENDING badge) | U | UC | B | b430e45 d3e9337 c30138b 2ee1763 |
| E106 | 4/08–4/10 | O(n^2) dead code in the bullpen block slowed worker predictions (with E16) | L | UC | P | 947fce8 |
| E107 | →6/06 | Healthchecks ping URL hardcoded in the public repo's cron script; check rotated | S/A | OI? | B | 681eb8c |
| E108 | →6/11 | heartbeat watchdog wrote RUNNING unconditionally, masking wedged cascades (audit H5) | L/A | DL | A? | 18efce1 |
| E109 | →7/10 | deploy race: an in-flight run could ship a later, untested commit (reset to mutable origin/deploy) | S | DL | P | 30452eb |
| E110 | 7/04–7/09 | entry check v2 treated present_unverified as confirmed and masked a missing DD slot | E/A | DL? | A | e5ef7ca (with E72) |
| E111 | 6/10–6/18 | MDP saver input from an unsound model-saver proxy / ledger inference | P/G | DL | P | 810d7e0 815cf50 bf6ccae |
