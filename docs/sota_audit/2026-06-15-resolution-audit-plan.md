# Top-pick resolution audit — exploration plan (2026-06-15)

**Goal:** Determine, at a deliberately LOW bar (any *potential* improvement, not just likely), whether a different model class / objective / signal can improve **top-pick resolution** for BTS — i.e., separate today's ~0.86 batter-days from other ~0.86 batter-days. If nothing can, prove the **data/label ceiling** cheaply.

**Reframe (Codex):** this is NOT "try another learner." It's "is there *any* conditional signal that separates same-p top picks?" — answerable cheaply on existing backtest data via **residual screens** + **oracle upper bounds**, BEFORE building anything. So: screen everything cheaply; build only survivors.

**Established diagnosis (this cycle):** weak resolution. Realized hit by rank: r1 .865, r2 .854, r3-5 ~.83 (CIs overlap). Spearman(pred,realized)=0.13. Top-40% region FLAT (lower-half .907 vs upper-half .896). Calibration of p is a no-op for the quantile-binned MDP. Feature-hunting exhausted (k_contact null + ~30 rejects); market, deviations, recalibration, finer-bins, DD-floor all rejected.

**Discipline (hard-won this session):** premise-check + history-check before any build; cheap decisive kill-test first; OOF / leave-one-season-out for every residual (a residual computed including the scored day is circular = leak); block-bootstrap by date for CIs; **Codex reviews methodology + results at each phase checkpoint** (catch leakage/overfit before trusting). Kill criterion default: **<+1pp top-1 lift (CI crossing 0) ⇒ no headroom.**

## Phase 0 — master oracle bounds (can settle the whole question)
Bounds on what ANY model class could extract from current data. Cheapest-first:
- **0a. Player random-effects (LOSO) re-rank** [profiles-only, leak-safe]: re-rank daily candidates by p + leave-one-season-out shrunk batter residual; does new top-1 beat p-only top-1? *(this file's first screen)*
- **0b. Realized-PA / lineup-slot oracle**: re-rank using realized PA count (upper bound on opportunity-count modeling). Needs PA-level p or est_pas swap.
- **0c. Realized-game-variable oracle**: add realized team-runs / starter-IP one at a time → max possible lift.
- **0d. Nearest-neighbor Bayes bound** (master): per top-15 candidate, local empirical hit rate from similar-pregame-state historical neighbors (heavy shrinkage, OOF); does picking by that beat picking by p? Needs the full feature frame.
- **KILL-GATE:** best-possible oracle on existing features < +1pp top-1 (within-day top-vs-median spread <2pp) ⇒ **data/label irreducibility proven → STOP, no builds.**

## Phase 1 — residual screens (shared OOF machinery; run for every signal; build only survivors)
OOF residual = hit − current_p among daily top-15; does signal X predict it (rank-IC / top-vs-bottom-half gap / AUC), kill at <+1pp / IC<0.03 / bootstrap p>0.10:
player/pitcher/bullpen/catcher random-effects; time-varying latent player state; richer continuous target (xwOBA-on-day); matchup-archetype cells (spray×pitcher×park/defense); weather-as-interaction; survival/first-passage + Poisson-binomial count reformulation; slate-level skip signal; double-down pair-correlation audit; calendar/regime.

## Phase 2 — build + rigorously evaluate survivors
Worktree + OOS + MC-bootstrap + leak guards + the escalated harness. Only signals that passed Phase 1 AND the Phase-0 bound showed headroom.

## Phase 3 — heavy model-class/objective builds (LAST, only if a screen shows the signal exists)
Decision-focused end-to-end P(57) training; learned matchup embeddings. Do NOT build a learner to extract signal the cheap screens say isn't there.

## Already-done / down-rank (history-check)
Market *player props* ≈ item-05 (tested, null residual r=−0.02); recalibration / finer-bins / rank-loss / DT-LR / 15+-blend = rejected; umpire = partly in shadow model.

## Honest prior
Everything points to Phase 0 showing the data ceiling. This plan mostly serves to *prove* that cheaply (or catch the rare real signal). Either outcome = a definitive low-cost answer.
