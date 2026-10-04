"""Frozen run identity for the one registered #87 execution (amendment A1/A4; review r1 required changes 1 and 10).

``REGISTRATION`` fixes the window, cohort snapshot, estimand parameters, cell test, FDR family, nomination thresholds,
both analysis streams, the lock/settlement/admission rules and the optional diagnostic before any outcome-bearing
execution. Its sha256 fingerprint is pinned beside it, so an edit to any frozen value is refused rather than silently
run. Registered mode accepts no parameter override; any override requires ``--exploratory``, which labels the run and
disables nomination. Nothing outcome-bearing runs until exposure row X-22 is published and ``X22_COMMIT`` names it.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
X22_COMMIT: str | None = None          # the register commit that publishes X-22; set only when X-22 is published
REGISTER_PATH = "docs/audit/2026-09-22-exposure-register.md"
DOCUMENTS = {"protocol": "docs/sota_audit/2026-05-10-leaderboard-mechanism-mining-prereg.md",
             "amendment": "docs/sota_audit/2026-10-04-mechanism-mining-amendment.md",
             "review_r1": "docs/audit/2026-10-04-mechanism-mining-codex-r1.md",
             "exposure_register": REGISTER_PATH}

REGISTRATION = {
    "schema": "mining87_registration_v1",
    "window": {"start": "2026-03-26", "end": "2026-07-03"},
    # the corpus is "captured 5/01 → 7/04"; the scraper stamps naive UTC, so the end of 7/04 US Eastern (EDT) is
    # 2026-07-05T04:00 — late-7/04 captures that settle 7/03 picks stay in; later appends are excluded and counted
    "public_capture_end_exclusive_utc_naive": "2026-07-05T04:00:00",
    "excluded_public_sources": ["final_grab_*"],
    "cohort": {"snapshot_file": "2026-07-04.parquet", "tab": "active_streak"},
    "cohorts": ["fixed_cohort", "all_tracked"],
    "decomposition_variables": [
        "cohort", "pick_number", "consensus_pick_share_bin", "production_p_game_hit_bin", "agreement_state",
        "production_batter_skill_quartile", "production_batter_skill_prior_pa_bin", "production_projected_lineup",
        "production_regime", "production_is_park_driven", "production_is_indoor", "production_weather_temp_bin",
        "consensus_model_rank_bin", "consensus_model_probability_bin"],
    "top_k": [1, 2, 5, 10],
    "bootstrap": {"expected_block_length": 7, "n_bootstrap": 2000, "seed": 20260510,
                  "resampling": "circular geometric blocks over ordered observed dates; whole dates with delta sums "
                                "and unit counts; replicate = sum/count; the frozen seed for every calculation"},
    "cell_test": "exact_one_sided_paired_sign_test for every testable cell (amended A4 extension; "
                 "sign-exchangeability assumed)",
    "fdr": {"methods": ["BH", "BY"], "q_threshold": 0.10, "min_resolved_disagreement_units": 15,
            "family": "every testable cell of the stream, both cohorts"},
    "nomination": {"min_resolved_disagreement_units": 30, "min_absolute_lift": 0.05},
    "conditions": ["c1_min_units", "c2_min_lift", "c3_bh", "c4_all_tracked_direction",
                   "c5_lock_available_mechanism"],
    "streams": ["primary", "tie_excluded_sensitivity"],
    "nominating_stream": "primary",
    "production_source": "accepted W1.1 ledger build (ACCEPTED.json, rules fingerprint 5e9d74f2…)",
    "lock_rule": "selection row, commit_status committed_evidenced, finalization decision|pick_file_only",
    "settlement_rule": "contest slot grade via evidenced unit_capture or inferred unique scheduled game; contest "
                       "slot row must carry the same selection, batter and game; void void; else unknown",
    "consensus_rule": "distinct users by batter_id; DD = mode excluding the chosen primary; original slot "
                      "denominator; ties lowest id, flagged direct/exposed/dependent; settlement from the legal id's "
                      "voters: one unit, agreeing settled labels, else unknown/pending",
    "surface_admission": "independent witness binding the file sha256 + candidate universe, lineup assumptions, "
                         "feature computation, prediction timestamp <= lock; selection-consistent; else "
                         "missing_surface",
    "served_slate_diagnostic": "computed, labelled unproven, outside the family and nomination stream",
}
REGISTRATION_FINGERPRINT = "70a45bfd33defe56fedcad04cac1258f23cbb8acb9eb9135a74c243bc0116b2e"

OVERRIDABLE = ("window_start", "window_end", "seed", "n_bootstrap", "expected_block_length", "snapshot_file")


class RegistrationError(RuntimeError):
    pass


def registration_fingerprint(registration: dict | None = None) -> str:
    reg = REGISTRATION if registration is None else registration
    return hashlib.sha256(json.dumps(reg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def check_registration() -> str:
    got = registration_fingerprint()
    if got != REGISTRATION_FINGERPRINT:
        raise RegistrationError(f"registration fingerprint {got[:12]}… differs from the pinned "
                                f"{REGISTRATION_FINGERPRINT[:12]}…: a frozen value changed")
    return got


@dataclass(frozen=True)
class RunParams:
    mode: str
    window_start: str
    window_end: str
    capture_end_exclusive: str
    snapshot_file: str
    cohort_tab: str
    top_k: tuple
    expected_block_length: int
    n_bootstrap: int
    seed: int
    min_n: int
    q_threshold: float
    mechanism_min_n: int
    mechanism_min_lift: float
    overrides: dict = field(default_factory=dict)

    @property
    def can_nominate(self) -> bool:
        return self.mode == "registered"


def resolve_params(overrides: dict, *, exploratory: bool) -> RunParams:
    given = {k: v for k, v in overrides.items() if v is not None}
    unknown = sorted(set(given) - set(OVERRIDABLE))
    if unknown:
        raise RegistrationError(f"unknown override(s) {unknown}")
    if given and not exploratory:
        raise RegistrationError(f"registered-mode parameter drift {sorted(given)}: registered runs use the frozen "
                                "values; pass --exploratory to run a labelled exploratory analysis that cannot "
                                "nominate")
    r = REGISTRATION
    base = {"window_start": r["window"]["start"], "window_end": r["window"]["end"], "seed": r["bootstrap"]["seed"],
            "n_bootstrap": r["bootstrap"]["n_bootstrap"],
            "expected_block_length": r["bootstrap"]["expected_block_length"],
            "snapshot_file": r["cohort"]["snapshot_file"]}
    base.update(given)
    return RunParams(mode="exploratory" if exploratory else "registered",
                     capture_end_exclusive=r["public_capture_end_exclusive_utc_naive"], cohort_tab=r["cohort"]["tab"],
                     top_k=tuple(r["top_k"]), min_n=r["fdr"]["min_resolved_disagreement_units"],
                     q_threshold=r["fdr"]["q_threshold"],
                     mechanism_min_n=r["nomination"]["min_resolved_disagreement_units"],
                     mechanism_min_lift=r["nomination"]["min_absolute_lift"], overrides=dict(given), **base)


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.strip()


def x22_gate() -> str:
    """Refuse unless X22_COMMIT is set, is an ancestor of HEAD, and this checkout's register carries the X-22 row."""
    if X22_COMMIT is None:
        raise SystemExit("X-22 gate: X22_COMMIT is unset (publish exposure row X-22 first)")
    head = git("rev-parse", "HEAD")
    if subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", X22_COMMIT, head]).returncode != 0:
        raise SystemExit(f"X-22 gate: {X22_COMMIT} is not an ancestor of HEAD {head[:7]}")
    if "| X-22 |" not in (REPO / REGISTER_PATH).read_text():
        raise SystemExit("X-22 gate: the register in this checkout has no X-22 row")
    return head


def document_hashes() -> dict:
    return {name: {"path": rel, "sha256": hashlib.sha256((REPO / rel).read_bytes()).hexdigest()}
            for name, rel in DOCUMENTS.items()}
