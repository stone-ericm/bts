# 2026-08-09 — DD-band tripwire recompute (early look): gap collapsed, no escalation

Pre-registration (7/13 value-sensitivity doc): at season n≈80–90 DD legs,
recompute the 7/12 leg-gap measurement; if a ≥10pp shortfall holds, rerun
`scripts/audit/dd_p_policy_value_sensitivity.py` and take the re-solve
question seriously.

**This is an early look at n=57**, prompted by the 8/09 `mdp_policy_alignment`
quality-bin-collapse WARN (regime shift: rank-1 p's compressed below the
policy's lowest boundary after the ball deadening began reverting park-by-park
in late July). Recorded as a look; the formal n≈80–90 read still stands
(~23 more legs, ≈ early-mid September at current double cadence).

Measurement identical to 7/12 (all 2026 pick files, per-slot grading,
`slot_results` authoritative, voids/ungraded self-excluding; exact tail =
Poisson-binomial). Reproduction anchor matched the 7/12 doc exactly:
25/42 = .595 vs .734, tail .0349.

## Season-to-date (through 8/09)

| slice | legs | realized | stated | gap | exact tail |
|---|---|---|---|---|---|
| DD legs, season | 39/57 | .684 | .740 | **−5.6pp** | .207 |
| DD legs, thru 7/12 (anchor) | 25/42 | .595 | .734 | −13.9pp | .035 |
| DD legs, thru 7/27 | 33/51 | .647 | .739 | −9.2pp | .094 |
| **DD legs since 7/13** | **14/15** | **.933** | .757 | **+17.7pp** | .985 |
| primaries, season | 60/79 | .759 | .772 | −1.2pp | .443 |

Monthly DD: May 16/28 (.571), June 7/7, July 10/16 (.625), Aug 6/6.

## Verdict

**Tripwire does not fire — on both conditions.** n=57 is outside the 80–90
window, and the gap no longer approaches 10pp: since the 7/12 measurement the
fresh sample ran 14/15, exactly the "if luck, it regresses" branch the 7/12
doc pre-wrote. Season gap −5.6pp at tail p=.21 is unremarkable against the
+0.98pp (SE ~1.1pp) backtest reference. No value-sensitivity rerun, no
re-solve escalation from this lever. Monitor-not-surgery holds.

**The quality-bin-collapse WARN is a separate, real question** — it is about
the p *distribution* shifting below the solved policy boundaries
(.796–.841), not about calibration (primaries −1.2pp = calibrated). The
policy still functions in the compressed range (doubles at low streak, skips
at ≥8 sub-.796) but cannot discriminate within Q0. Disposition: any re-solve
should wait for the ball regime to stabilize — the park_drag observability
table (5 parks reverted, several still juiced, 3–4 juicing NOW as of 8/09)
is the completion signal for when post-regime data is trainable. Re-solving
mid-churn would fit boundaries to a moving distribution.

Reproduction: `tripwire.py` in the 8/09 session scratchpad (30 lines; reads
pick files + Poisson-binomial DP). The 7/12 dataset-builder semantics apply
unchanged.
