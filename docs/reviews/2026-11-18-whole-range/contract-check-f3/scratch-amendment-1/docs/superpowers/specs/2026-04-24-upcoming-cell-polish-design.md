# Upcoming-PA Cell Polish — design

**Date**: 2026-04-24
**Author**: Eric + Claude (brainstorm w/ visual companion)
**Scope**: `src/bts/web.py` — CSS only, placeholder branch of `_render_pa_cell`

## Problem

The upcoming-PA cell currently renders as a 100px dashed-border box with small bold text on white background — visually orphaned next to the rich filled cells (pitch grids, baserunning diamonds, green hit highlight, amber AB-pulse). Eric's words: "as is is bad."

Visual mockups compared three directions; Eric picked **A — Tinted Cell** (matches the existing green-hit / amber-AB color vocabulary).

Approved mockup: `.superpowers/brainstorm/53596-1777078046/content/direction-a-in-context.html` (verified the amber cells don't clash with the AB-pulse cell — the AB pulse uses a 2px solid amber border, the upcoming amber uses a 1px lighter-amber border, so they read as distinct urgency states).

## Decisions

| # | Question | Choice |
|---|---|---|
| 1 | Visual direction | **A — Tinted Cell** (background-color tint per state, matches existing scorecard vocabulary) |
| 2 | Per-state palette | Amber for ON DECK / IN THE HOLE (imminent), gray for distant, red for OUT, white for blank states |
| 3 | Border treatment | Solid 1px in tint family (drop the dashed gray border — that signaled "empty" but conflicts with the new "informational" semantic) |
| 4 | Text size/weight | Keep 12px / 600 (already shipped) |
| 5 | "Not in lineup" treatment | Same gray as "N batters away" — visually quiet (rare state) |

## Status → CSS mapping

| `lineup_status` | Background | Border | Text color |
|---|---|---|---|
| `on_deck` | `#fef3c7` (amber-100) | `1px solid #fcd34d` (amber-300) | `#92400e` (amber-800) |
| `in_hole` | `#fef9e2` (amber-50) | `1px solid #fde68a` (amber-200) | `#92400e` (amber-800) |
| `upcoming` (any N) | `#f9fafb` (gray-50) | `1px solid #e5e7eb` (gray-200) | `#6b7280` (gray-500) |
| `out_of_game` | `#fee2e2` (red-100) | `1px solid #fca5a5` (red-300) | `#991b1b` (red-800) |
| `not_in_lineup` | `#f9fafb` (gray-50) | `1px solid #e5e7eb` (gray-200) | `#6b7280` (gray-500) |
| `at_bat` / `pre_game` / `final` / `None` | `#fff` | `1px dashed #ccc` (current behavior) | n/a (no label rendered) |

## Architecture

Changes are local to the `if pa is None` branch of `_render_pa_cell` in `src/bts/web.py`. Move the inline `style` string out of a single static template into a per-status lookup. No new functions, no new files.

```python
# Pseudocode shape
PLACEHOLDER_STYLES = {
    "on_deck":      "background:#fef3c7;border:1px solid #fcd34d;color:#92400e;...",
    "in_hole":      "background:#fef9e2;border:1px solid #fde68a;color:#92400e;...",
    "upcoming":     "background:#f9fafb;border:1px solid #e5e7eb;color:#6b7280;...",
    "out_of_game":  "background:#fee2e2;border:1px solid #fca5a5;color:#991b1b;...",
    "not_in_lineup":"background:#f9fafb;border:1px solid #e5e7eb;color:#6b7280;...",
}
DEFAULT_PLACEHOLDER_STYLE = "border:1px dashed #ccc;color:#bbb;..."
```

Style lookup at render time picks the right palette from `lineup_status`. The label-text computation stays unchanged from the current implementation.

## Testing

**No test changes needed.** Existing assertions in `tests/test_web_render.py` check substring presence (`"ON DECK" in html`, `"5 batters away" in html`, etc.). Substring matching survives CSS changes; the assertions pass regardless of the surrounding `<td style="...">` contents.

Optional minor: one regression test that verifies a state-specific background color hex appears in the rendered HTML (e.g., `"#fef3c7" in html` when `lineup_status="on_deck"`). YAGNI for v1; only add if a future regression breaks the palette mapping.

## Out of scope

- Animation / pulse for ON DECK (deferred — could be polish if Eric finds it under-emphatic)
- Icon glyphs (no icon system in the dashboard today; would be net-new dependency)
- Color-blind accessibility audit (the dashboard is single-user / personal-use; out of scope for v1; if shared later, audit then)
- Mobile / narrow-screen rendering — cell is already fixed at 100px; no responsive concerns

## Rollback

Single-line revert: replace the per-status style lookup with the original single-style string. Tests still pass either way.
