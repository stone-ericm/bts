#!/usr/bin/env python3
"""Prospective leaderboard-to-candidate join audit.

This tool is intentionally research-only. It reads a frozen live candidate
artifact plus captured BTS leaderboard parquet files, filters public pick
observations to a decision cutoff, joins public-pick consensus features onto
candidate rows, and writes validation artifacts only.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SCHEMA_VERSION = "leaderboard_candidate_join_audit_v1"
VALID_RESULTS = {"hit", "not_hit"}
PICK_NUMBERS = (1, 2)
USER_PICK_REQUIRED_COLUMNS = {
    "captured_at",
    "pick_date",
    "pick_number",
    "batter_id",
    "batter_name",
    "result",
}
SNAPSHOT_REQUIRED_COLUMNS = {
    "captured_at",
    "tab",
    "rank",
    "username",
    "streak",
}


@dataclass(frozen=True)
class CandidateArtifact:
    root: Path
    manifest: dict[str, Any]
    profiles: pd.DataFrame


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def parse_cutoff(raw: str | None) -> pd.Timestamp | None:
    if raw is None:
        return None
    ts = pd.Timestamp(raw)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def normalize_date_key(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_default))
    return path


def _profile_path_sort_key(key: str) -> tuple[int, int | str]:
    if str(key).isdigit():
        return (0, int(key))
    return (1, str(key))


def load_candidate_artifact(
    artifact_dir: Path,
    *,
    variants: tuple[str, ...] = ("production", "candidate"),
    dates: set[str] | None = None,
) -> CandidateArtifact:
    manifest = read_json(artifact_dir / "manifest.json")
    frames: list[pd.DataFrame] = []
    for variant in variants:
        paths = manifest.get("profile_paths", {}).get(variant, {})
        for key in sorted(paths, key=_profile_path_sort_key):
            if dates is not None and str(key) not in dates:
                continue
            path = artifact_dir / paths[key]
            frame = pd.read_parquet(path)
            if frame.empty:
                continue
            frame = frame.copy()
            frame["variant"] = variant
            frame["profile_key"] = str(key)
            frame["date"] = frame["date"].map(normalize_date_key)
            frames.append(frame)
    if not frames:
        raise ValueError(f"{artifact_dir} has no profile rows for variants={variants}")
    profiles = pd.concat(frames, ignore_index=True)
    required = {"variant", "date", "rank", "batter_id", "p_game_hit"}
    missing = sorted(required.difference(profiles.columns))
    if missing:
        raise ValueError(f"{artifact_dir} profile rows missing columns: {missing}")
    return CandidateArtifact(root=artifact_dir, manifest=manifest, profiles=profiles)


def load_user_picks(
    leaderboard_dir: Path,
    *,
    decision_cutoff: pd.Timestamp | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    files = sorted((leaderboard_dir / "user_picks").glob("*.parquet"))
    parts: list[pd.DataFrame] = []
    empty_files = 0
    for path in files:
        frame = pq.read_table(path).to_pandas()
        if frame.empty:
            empty_files += 1
            continue
        missing = sorted(USER_PICK_REQUIRED_COLUMNS.difference(frame.columns))
        if missing:
            raise ValueError(f"{path} missing leaderboard user-pick columns: {missing}")
        frame = frame.copy()
        frame["username"] = path.stem
        parts.append(frame)
    if not parts:
        return pd.DataFrame(), {
            "user_pick_files": len(files),
            "empty_user_pick_files": empty_files,
            "raw_rows": 0,
            "cutoff_rows": 0,
            "dedup_rows": 0,
        }
    raw = pd.concat(parts, ignore_index=True)
    raw["captured_at"] = pd.to_datetime(raw["captured_at"])
    raw["pick_date"] = pd.to_datetime(raw["pick_date"]).dt.strftime("%Y-%m-%d")
    cutoff_frame = raw
    if decision_cutoff is not None:
        cutoff_frame = raw[raw["captured_at"] <= decision_cutoff].copy()
    dedup = (
        cutoff_frame.sort_values("captured_at")
        .drop_duplicates(["username", "pick_date", "pick_number"], keep="last")
        .copy()
    )
    return dedup, {
        "user_pick_files": len(files),
        "empty_user_pick_files": empty_files,
        "raw_rows": int(len(raw)),
        "cutoff_rows": int(len(cutoff_frame)),
        "dedup_rows": int(len(dedup)),
        "decision_cutoff_applied": decision_cutoff is not None,
    }


def load_snapshot_cohort(
    leaderboard_dir: Path,
    *,
    cohort_as_of: pd.Timestamp | None,
    tab: str = "active_streak",
) -> tuple[set[str], dict[str, Any]]:
    snaps = sorted((leaderboard_dir / "leaderboard_snapshots").glob("*.parquet"))
    chosen_path: Path | None = None
    chosen_frame: pd.DataFrame | None = None
    for path in snaps:
        frame = pq.read_table(path).to_pandas()
        if frame.empty:
            continue
        missing = sorted(SNAPSHOT_REQUIRED_COLUMNS.difference(frame.columns))
        if missing:
            raise ValueError(f"{path} missing leaderboard snapshot columns: {missing}")
        frame["captured_at"] = pd.to_datetime(frame["captured_at"])
        if cohort_as_of is not None:
            frame = frame[frame["captured_at"] <= cohort_as_of].copy()
        frame = frame[frame["tab"] == tab].copy()
        if frame.empty:
            continue
        chosen_path = path
        chosen_frame = frame
    if chosen_frame is None:
        return set(), {
            "source": "leaderboard_snapshot",
            "snapshot_path": None,
            "snapshot_count": len(snaps),
            "tab": tab,
            "n_users": 0,
        }
    users = set(str(u) for u in chosen_frame["username"].dropna().astype(str))
    return users, {
        "source": "leaderboard_snapshot",
        "snapshot_path": str(chosen_path) if chosen_path is not None else None,
        "snapshot_count": len(snaps),
        "tab": tab,
        "n_users": len(users),
        "snapshot_captured_at_min": chosen_frame["captured_at"].min().isoformat(),
        "snapshot_captured_at_max": chosen_frame["captured_at"].max().isoformat(),
    }


def load_cohort_users(path: Path) -> tuple[set[str], dict[str, Any]]:
    payload = read_json(path)
    if isinstance(payload, list):
        users = set(str(x) for x in payload)
    else:
        users = set(str(x) for x in payload.get("users", []))
    return users, {
        "source": "cohort_users_json",
        "path": str(path),
        "n_users": len(users),
    }


def pick_feature_tables(
    picks: pd.DataFrame,
    *,
    users: set[str] | None,
    cohort_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if users is not None:
        cohort_picks = picks[picks["username"].isin(users)].copy()
    else:
        cohort_picks = picks.copy()
    if cohort_picks.empty:
        return pd.DataFrame(), pd.DataFrame(), {
            "cohort_name": cohort_name,
            "n_users_with_rows": 0,
            "n_rows": 0,
            "n_date_slots": 0,
        }

    feature_rows: list[dict[str, Any]] = []
    consensus_rows: list[dict[str, Any]] = []
    for (pick_date, pick_number), group in cohort_picks.groupby(["pick_date", "pick_number"]):
        group = group[group["batter_id"].notna()].copy()
        if group.empty:
            continue
        n_users = int(group["username"].nunique())
        counts = (
            group.groupby(["batter_id", "batter_name"], dropna=False)
            .agg(
                pick_count=("username", "nunique"),
                result_hit_count=("result", lambda s: int((s == "hit").sum())),
                result_not_hit_count=("result", lambda s: int((s == "not_hit").sum())),
            )
            .reset_index()
            .sort_values(["pick_count", "batter_id"], ascending=[False, True])
        )
        counts["public_pick_rank"] = range(1, len(counts) + 1)
        counts["public_pick_share"] = counts["pick_count"] / n_users
        top = counts.iloc[0]
        top_result = None
        if top["result_hit_count"] > top["result_not_hit_count"]:
            top_result = "hit"
        elif top["result_not_hit_count"] > top["result_hit_count"]:
            top_result = "not_hit"
        elif top["result_hit_count"] or top["result_not_hit_count"]:
            top_result = "tie"

        consensus_rows.append({
            "cohort": cohort_name,
            "date": str(pick_date),
            "pick_number": int(pick_number),
            "n_public_users": n_users,
            "consensus_batter_id": int(top["batter_id"]),
            "consensus_batter_name": top["batter_name"],
            "consensus_pick_count": int(top["pick_count"]),
            "consensus_pick_share": float(top["public_pick_share"]),
            "consensus_result": top_result,
        })
        for row in counts.itertuples(index=False):
            feature_rows.append({
                "cohort": cohort_name,
                "date": str(pick_date),
                "pick_number": int(pick_number),
                "batter_id": int(row.batter_id),
                "batter_name": row.batter_name,
                "n_public_users": n_users,
                "public_pick_count": int(row.pick_count),
                "public_pick_share": float(row.public_pick_share),
                "public_pick_rank": int(row.public_pick_rank),
                "is_public_consensus": int(row.public_pick_rank == 1),
                "consensus_batter_id": int(top["batter_id"]),
                "consensus_batter_name": top["batter_name"],
                "consensus_pick_share": float(top["public_pick_share"]),
                "consensus_result": top_result,
            })

    features = pd.DataFrame(feature_rows)
    consensus = pd.DataFrame(consensus_rows)
    return features, consensus, {
        "cohort_name": cohort_name,
        "n_users_with_rows": int(cohort_picks["username"].nunique()),
        "n_rows": int(len(cohort_picks)),
        "n_date_slots": int(len(consensus)),
    }


def join_features_for_cohort(
    profiles: pd.DataFrame,
    features: pd.DataFrame,
    *,
    cohort_name: str,
) -> pd.DataFrame:
    joined = profiles.copy()
    metric_columns = [
        "n_public_users",
        "public_pick_count",
        "public_pick_share",
        "public_pick_rank",
        "is_public_consensus",
        "consensus_batter_id",
        "consensus_batter_name",
        "consensus_pick_share",
        "consensus_result",
    ]
    if features.empty:
        for pick_number in PICK_NUMBERS:
            prefix = f"lb_{cohort_name}_slot{pick_number}_"
            for col in metric_columns:
                joined[prefix + col] = pd.NA
        joined[f"lb_{cohort_name}_max_public_pick_share"] = pd.NA
        joined[f"lb_{cohort_name}_max_public_pick_count"] = pd.NA
        return joined
    for pick_number in PICK_NUMBERS:
        slot_features = features[features["pick_number"] == pick_number].copy()
        keep = [
            "date",
            "batter_id",
            *metric_columns,
        ]
        slot_features = slot_features[keep]
        prefix = f"lb_{cohort_name}_slot{pick_number}_"
        slot_features = slot_features.rename(
            columns={col: prefix + col for col in keep if col not in {"date", "batter_id"}}
        )
        joined = joined.merge(slot_features, on=["date", "batter_id"], how="left")
    share_cols = [
        f"lb_{cohort_name}_slot{pick_number}_public_pick_share"
        for pick_number in PICK_NUMBERS
    ]
    count_cols = [
        f"lb_{cohort_name}_slot{pick_number}_public_pick_count"
        for pick_number in PICK_NUMBERS
    ]
    joined[f"lb_{cohort_name}_max_public_pick_share"] = joined[share_cols].max(axis=1)
    joined[f"lb_{cohort_name}_max_public_pick_count"] = joined[count_cols].max(axis=1)
    return joined


def build_comparison_units(
    production_profiles: pd.DataFrame,
    consensus: pd.DataFrame,
    *,
    cohort_name: str,
) -> pd.DataFrame:
    if consensus.empty:
        return pd.DataFrame()
    prod = production_profiles[
        (production_profiles["variant"] == "production")
        & (production_profiles["rank"].isin(PICK_NUMBERS))
    ].copy()
    if prod.empty:
        return pd.DataFrame()
    prod["pick_number"] = prod["rank"].astype(int)
    comp = prod.merge(
        consensus[consensus["cohort"] == cohort_name],
        on=["date", "pick_number"],
        how="inner",
        suffixes=("_ours", "_consensus"),
    )
    comp = comp[comp["consensus_result"].isin(VALID_RESULTS)].copy()
    comp = comp[comp["actual_hit"].notna()].copy()
    if comp.empty:
        return comp
    comp["our_hit"] = comp["actual_hit"].astype(bool).astype(int)
    comp["consensus_hit"] = (comp["consensus_result"] == "hit").astype(int)
    comp["delta"] = comp["consensus_hit"] - comp["our_hit"]
    comp["agree"] = (
        comp["batter_id"].astype("Int64") == comp["consensus_batter_id"].astype("Int64")
    )
    return comp


def contiguous_block_bootstrap_ci(
    units: pd.DataFrame,
    *,
    value_col: str = "delta",
    date_col: str = "date",
    n_bootstrap: int = 2000,
    expected_block_length: int = 7,
    seed: int = 20260509,
) -> dict[str, Any] | None:
    if units.empty or units[value_col].dropna().empty:
        return None
    by_date = (
        units.groupby(date_col)[value_col]
        .mean()
        .sort_index()
        .reset_index(name=value_col)
    )
    values = by_date[value_col].to_numpy(dtype=float)
    n_dates = len(values)
    if n_dates == 0:
        return None
    observed = float(units[value_col].mean())
    if n_dates == 1 or n_bootstrap <= 0:
        return {
            "kind": "contiguous_day_block_bootstrap",
            "n_bootstrap": int(n_bootstrap),
            "expected_block_length": int(expected_block_length),
            "n_days": int(n_dates),
            "mean": observed,
            "ci_lower": None,
            "ci_upper": None,
            "p_mean_le_zero": None,
        }
    rng = np.random.default_rng(seed)
    p_stop = 1.0 / max(1, int(expected_block_length))
    samples = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        drawn: list[float] = []
        while len(drawn) < n_dates:
            start = int(rng.integers(0, n_dates))
            block_len = int(rng.geometric(p_stop))
            for offset in range(block_len):
                drawn.append(values[(start + offset) % n_dates])
                if len(drawn) >= n_dates:
                    break
        samples[i] = float(np.mean(drawn[:n_dates]))
    return {
        "kind": "contiguous_day_block_bootstrap",
        "n_bootstrap": int(n_bootstrap),
        "expected_block_length": int(expected_block_length),
        "seed": int(seed),
        "n_days": int(n_dates),
        "mean": observed,
        "ci_lower": float(np.quantile(samples, 0.025)),
        "ci_upper": float(np.quantile(samples, 0.975)),
        "p_mean_le_zero": float((samples <= 0).mean()),
    }


def summarize_comparison(
    units: pd.DataFrame,
    *,
    expected_block_length: int,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    if units.empty:
        return {
            "n_units": 0,
            "n_disagreements": 0,
            "our_hit_rate": None,
            "consensus_hit_rate": None,
            "mean_delta": None,
            "disagreement_mean_delta": None,
            "bootstrap": None,
            "disagreement_bootstrap": None,
        }
    disagreements = units[~units["agree"]].copy()
    return {
        "n_units": int(len(units)),
        "n_disagreements": int(len(disagreements)),
        "agreement_rate": float(units["agree"].mean()),
        "our_hit_rate": float(units["our_hit"].mean()),
        "consensus_hit_rate": float(units["consensus_hit"].mean()),
        "mean_delta": float(units["delta"].mean()),
        "disagreement_our_hit_rate": (
            float(disagreements["our_hit"].mean()) if len(disagreements) else None
        ),
        "disagreement_consensus_hit_rate": (
            float(disagreements["consensus_hit"].mean()) if len(disagreements) else None
        ),
        "disagreement_mean_delta": (
            float(disagreements["delta"].mean()) if len(disagreements) else None
        ),
        "bootstrap": contiguous_block_bootstrap_ci(
            units,
            expected_block_length=expected_block_length,
            n_bootstrap=n_bootstrap,
            seed=seed,
        ),
        "disagreement_bootstrap": contiguous_block_bootstrap_ci(
            disagreements,
            expected_block_length=expected_block_length,
            n_bootstrap=n_bootstrap,
            seed=seed + 1,
        ) if len(disagreements) else None,
    }


def write_joined_profiles(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def build_audit(
    *,
    leaderboard_dir: Path,
    artifact_dir: Path,
    output_path: Path,
    joined_output_path: Path | None,
    decision_cutoff_iso: str | None,
    cohort_as_of_iso: str | None,
    cohort_users_json: Path | None,
    dates: set[str] | None,
    n_bootstrap: int,
    expected_block_length: int,
    seed: int,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or utc_now_iso()
    decision_cutoff = parse_cutoff(decision_cutoff_iso)
    cohort_as_of = parse_cutoff(cohort_as_of_iso) if cohort_as_of_iso else decision_cutoff

    artifact = load_candidate_artifact(artifact_dir, dates=dates)
    picks, pick_inventory = load_user_picks(
        leaderboard_dir,
        decision_cutoff=decision_cutoff,
    )
    all_users = set(picks["username"].dropna().astype(str)) if not picks.empty else set()
    if cohort_users_json is not None:
        fixed_users, fixed_cohort_meta = load_cohort_users(cohort_users_json)
    else:
        fixed_users, fixed_cohort_meta = load_snapshot_cohort(
            leaderboard_dir,
            cohort_as_of=cohort_as_of,
        )
    fixed_cohort_meta["cohort_as_of_iso"] = cohort_as_of_iso or decision_cutoff_iso

    all_features, all_consensus, all_meta = pick_feature_tables(
        picks,
        users=None,
        cohort_name="all_tracked",
    )
    fixed_features, fixed_consensus, fixed_meta = pick_feature_tables(
        picks,
        users=fixed_users,
        cohort_name="fixed_cohort",
    )
    joined = join_features_for_cohort(
        artifact.profiles,
        all_features,
        cohort_name="all_tracked",
    )
    joined = join_features_for_cohort(
        joined,
        fixed_features,
        cohort_name="fixed_cohort",
    )

    consensus = pd.concat(
        [frame for frame in (all_consensus, fixed_consensus) if not frame.empty],
        ignore_index=True,
    ) if (not all_consensus.empty or not fixed_consensus.empty) else pd.DataFrame()
    comparison_fixed = build_comparison_units(joined, consensus, cohort_name="fixed_cohort")
    comparison_all = build_comparison_units(joined, consensus, cohort_name="all_tracked")

    if joined_output_path is None:
        joined_output_path = output_path.with_suffix(".joined.parquet")
    write_joined_profiles(joined, joined_output_path)

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "production_deploy_claim": False,
        "research_only": True,
        "pre_lock_visibility_claim": decision_cutoff is not None,
        "decision_cutoff_iso": decision_cutoff_iso,
        "cohort_as_of_iso": cohort_as_of_iso or decision_cutoff_iso,
        "artifact_dir": str(artifact_dir),
        "leaderboard_dir": str(leaderboard_dir),
        "joined_profiles_path": str(joined_output_path),
        "artifact_manifest": {
            "schema_version": artifact.manifest.get("schema_version"),
            "run_kind": artifact.manifest.get("run_kind"),
            "candidate_name": artifact.manifest.get("candidate_name"),
            "baseline_name": artifact.manifest.get("baseline_name"),
            "dates": artifact.manifest.get("dates"),
            "top_n": artifact.manifest.get("top_n"),
            "generated_at": artifact.manifest.get("generated_at"),
        },
        "inventory": {
            "leaderboard": pick_inventory,
            "candidate_profile_rows": int(len(artifact.profiles)),
            "joined_profile_rows": int(len(joined)),
            "all_tracked_users_before_cutoff": int(len(all_users)),
        },
        "cohorts": {
            "all_tracked": {
                "source": "user_picks_before_cutoff",
                "n_users": int(len(all_users)),
                **all_meta,
            },
            "fixed_cohort": {
                **fixed_cohort_meta,
                **fixed_meta,
            },
        },
        "comparison": {
            "unit": "(pick_date, pick_number) resolved production rank 1/2 vs public consensus",
            "fixed_cohort": summarize_comparison(
                comparison_fixed,
                expected_block_length=expected_block_length,
                n_bootstrap=n_bootstrap,
                seed=seed,
            ),
            "all_tracked_diagnostic": summarize_comparison(
                comparison_all,
                expected_block_length=expected_block_length,
                n_bootstrap=n_bootstrap,
                seed=seed + 17,
            ),
        },
        "methodology_constraints": {
            "no_policy_edit_supported_by_this_artifact": True,
            "minimum_future_resolved_disagreement_units_for_first_eval": 30,
            "subgroup_reads_are_diagnostic_without_fdr_or_preregistration": True,
            "candidate_join_key": "date + batter_id",
            "candidate_join_key_caveat": (
                "Leaderboard user-pick rows include batter_id but not game_pk, so "
                "candidate popularity joins cannot disambiguate same-date "
                "doubleheader/game-level context without an additional source."
            ),
            "time_of_day_staleness_caveat": (
                "Pre-lock leaderboard consensus may predate lineup/scratch news; "
                "operational use requires a freshness gate relative to lineup and "
                "candidate-artifact generation."
            ),
            "falsification_rule": (
                "reject consensus-edge for production use if the forward fixed-cohort "
                "mean(consensus_hit - our_hit) is <= 0, or uncertainty clearly "
                "includes zero and no candidate-join mechanism is found"
            ),
            "day_block_bootstrap_expected_block_length": int(expected_block_length),
            "day_block_bootstrap_seed": int(seed),
        },
    }
    write_json(report, output_path)
    return report


def parse_dates(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    return {normalize_date_key(part.strip()) for part in raw.split(",") if part.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaderboard-dir", default="data/leaderboard")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--joined-output", default=None)
    parser.add_argument("--decision-cutoff-iso", default=None)
    parser.add_argument("--cohort-as-of-iso", default=None)
    parser.add_argument("--cohort-users-json", default=None)
    parser.add_argument("--dates", default=None, help="Comma-separated YYYY-MM-DD filter")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--expected-block-length", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260509)
    args = parser.parse_args()

    report = build_audit(
        leaderboard_dir=Path(args.leaderboard_dir),
        artifact_dir=Path(args.artifact_dir),
        output_path=Path(args.output),
        joined_output_path=Path(args.joined_output) if args.joined_output else None,
        decision_cutoff_iso=args.decision_cutoff_iso,
        cohort_as_of_iso=args.cohort_as_of_iso,
        cohort_users_json=Path(args.cohort_users_json) if args.cohort_users_json else None,
        dates=parse_dates(args.dates),
        n_bootstrap=args.n_bootstrap,
        expected_block_length=args.expected_block_length,
        seed=args.seed,
    )
    fixed = report["comparison"]["fixed_cohort"]
    print(f"wrote {args.output}")
    print(f"joined profiles: {report['joined_profiles_path']}")
    print(
        "fixed cohort comparison: "
        f"n={fixed['n_units']} disagreements={fixed['n_disagreements']} "
        f"mean_delta={fixed['mean_delta']}"
    )


if __name__ == "__main__":
    main()
