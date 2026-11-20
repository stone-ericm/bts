"""Summary table formatting for experiment results."""

from __future__ import annotations


def format_phase1_table(
    results: list[dict],
    baseline_p1: float = 0.862,
    baseline_p57: float = 0.0891,
) -> str:
    """Format Phase 1 results as a summary table string."""
    header = (
        f"Phase 1 Results — {len(results)} experiments vs baseline "
        f"(P@1={baseline_p1:.1%}, P(57)={baseline_p57:.2%})\n"
    )
    sep = "─" * 72 + "\n"
    col_header = f"{'Experiment':<24} {'P@1 2024':>10} {'P@1 2025':>10} {'P(57) MDP':>11} {'Pass':>6}\n"

    rows = []
    for r in sorted(results, key=lambda x: x.get("passed", False), reverse=True):
        name = r["name"][:23]
        diff = r.get("diff", {})

        p1_2024 = diff.get("p_at_1_by_season", {}).get("2024", {}).get("delta")
        p1_2025 = diff.get("p_at_1_by_season", {}).get("2025", {}).get("delta")
        p57 = diff.get("p_57_mdp", {}).get("delta")

        def fmt_delta(d):
            if d is None:
                return "N/A".rjust(10)
            sign = "+" if d >= 0 else ""
            return f"{sign}{d:.1%}".rjust(10)

        status = "✓" if r.get("passed") else "✗"
        rows.append(
            f"{name:<24} {fmt_delta(p1_2024)} {fmt_delta(p1_2025)} "
            f"{fmt_delta(p57):>11} {status:>6}"
        )

    n_pass = sum(1 for r in results if r.get("passed"))
    footer = f"\nWinners: {n_pass}/{len(results)} passed screening"

    return header + sep + col_header + sep + "\n".join(rows) + "\n" + sep + footer


def format_phase2_log(selection_result: dict) -> str:
    """Format Phase 2 forward/backward log as a summary string."""
    lines = ["Phase 2 — Forward Stepwise Selection"]
    lines.append("─" * 60)

    for step in selection_result.get("forward_log", []):
        status = "✓ KEPT" if step["kept"] else "✗ DROP"
        lines.append(
            f"  +{step['name']:<20} P(57): {step['p57_before']:.4f} → "
            f"{step['p57_after']:.4f} ({step['delta']:+.4f})  {status}"
        )

    lines.append("")
    lines.append("Backward Elimination")
    lines.append("─" * 60)

    for step in selection_result.get("backward_log", []):
        status = "✓ KEPT" if step["kept"] else "✗ DROP"
        lines.append(
            f"  -{step['name']:<20} P(57) without: {step['p57_without']:.4f} "
            f"(Δ={step['delta']:+.4f})  {status}"
        )

    included = selection_result.get("included", [])
    final_p57 = selection_result.get("final_scorecard", {}).get("p_57_mdp", "N/A")
    lines.append("")
    lines.append(f"Final model: {included}")
    lines.append(f"Final P(57): {final_p57}")

    return "\n".join(lines)
