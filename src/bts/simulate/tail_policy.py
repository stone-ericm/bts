"""Tail policy: exact E[season-best] once streak 57 is unreachable.

Why this exists (2026-09-03): the reach-57 solver (``bts.simulate.mdp.solve_mdp``)
values EVERY action at exactly 0.0 in any state with ``streak + 2*days < 57``, so
``argmax`` falls to index 0 = skip and production idles for the rest of the season
— it did exactly that on 9/03 at streak 0 with 25 game dates left. The owner's
objective for that regime is to keep playing for the account's SEASON-BEST streak
and to stop only once the season best can no longer be beaten.

Design (two Codex adversarial rounds, ``docs/audit/2026-09-03-emax-tail-policy.md``):

* ``mdp_objective`` decides the regime from (streak, days) alone — BEFORE any
  artifact is consulted — so an artifact failure can never fall through to the
  0.80 heuristic and recreate skip-forever.
* The tail objective is exact expected season-best on the augmented state
  (streak s, running best m, days d, saver), a port of the validated solver in
  ``scripts/audit/skip_threshold_resolve.py::solve_emax``. It is NOT the frontier
  shortcut (extracting the policy at m == s), which Codex showed is not the policy
  the solver evaluated after a miss.
* Stop rule, explicit: skip iff ``min(target, s + 2d) <= m`` (no outcome can raise
  the season best). Elsewhere ties among exact maximisers prefer a play (single,
  then double) so a structural tie never idles the account.
* The policy ships as a SEPARATE versioned artifact (``mdp_tail_policy.npz``) with
  an exhaustive loader contract, hard-bound to the sha256 of the reach-57 policy it
  pairs with. A broken/absent tail resolves to ``forced_tail_action`` (skip iff the
  stop rule, else single), never to the old zero table.

The one-bin production build (late-season rates) is a variance-reduction choice
made by the owner after seeing the preview: the ~150 real late-season dates behind
the 24-seed profiles cannot support five quintiles (Codex r2: top-20% vs rest is
1.4 SE). It trades adaptive quality decisions for estimation stability.
"""
from __future__ import annotations

import functools
import hashlib
import io
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bts.simulate.mdp import ACTIONS

TARGET = 57
MAX_TAIL_DAYS = (TARGET - 1) // 2   # 28: the largest d with 0 + 2d < TARGET
OBJECTIVE_REACH57 = "reach57"
OBJECTIVE_TAIL = "emax_season_best"
TAIL_SCHEMA = "bts_tail_policy_v1"
DEFAULT_TAIL_POLICY_PATH = Path("data/models/mdp_tail_policy.npz")
# A small compressed npz can declare a huge table; bound the bytes BEFORE parsing
# so a malicious/corrupt artifact fails into the forced fallback, not an OOM.
MAX_ARTIFACT_BYTES = 2 * 1024 * 1024

# Manifest contract: provenance of the rates the table was solved from.
_MANIFEST_TYPES = {
    "profiles_root": str, "seed_dirs": int, "parquets": int, "parquets_sha256": str,
    "rows": int, "seasons": list, "late_phase_days": int, "late_seed_days": int,
    "late_distinct_dates": int, "pairing": str, "n_bins": int, "bin_counts": list,
    "hits": int, "both": int, "base_policy_path": str,
}


def tail_manifest(*, n_bins: int, hits: int, both: int, late_seed_days: int, **overrides) -> dict:
    """A manifest satisfying the contract, with placeholder provenance for tests
    and the required consistency fields filled from the given counts. The rebuild
    script builds the real one (same keys)."""
    m = {
        "profiles_root": "unspecified", "seed_dirs": 0, "parquets": 0,
        "parquets_sha256": "0" * 64, "rows": 0, "seasons": [], "late_phase_days": 30,
        "late_seed_days": int(late_seed_days), "late_distinct_dates": 0,
        "pairing": "first lower-ranked candidate in a different game (production rule)",
        "n_bins": int(n_bins), "bin_counts": ([int(late_seed_days)] if n_bins == 1 else []),
        "hits": int(hits), "both": int(both), "base_policy_path": "unspecified",
    }
    m.update(overrides)
    return m

_SAVER_ZONE = (10, 15)   # inclusive; mirrors transition_outcomes in mdp.py


class TailPolicyError(ValueError):
    """The tail artifact is missing, malformed, or violates its contract."""


# --- regime + stop rule (pure, artifact-free) -----------------------------------

def effective_days(days_remaining: int, season_length: int = 180) -> int:
    """Clamp the raw day count to [0, season_length] — the same normalisation the
    reach-57 lookup applies — so the regime predicate and the table index agree."""
    return max(0, min(int(days_remaining), int(season_length)))


def mdp_objective(streak: int, days_remaining: int, season_length: int = 180,
                  target: int = TARGET) -> str:
    """Which objective governs this state. Exact within the MDP model: every action
    consumes one day, the largest increment is 2, and the saver never adds hits, so
    57 is reachable iff ``streak + 2*d >= target`` (equality is NOT degenerate: the
    all-double path has positive probability). d <= 0 and streak >= target stay with
    the base table, whose lookup already returns skip for them."""
    d = effective_days(days_remaining, season_length)
    if d <= 0 or streak >= target:
        return OBJECTIVE_REACH57
    return OBJECTIVE_REACH57 if streak + 2 * d >= target else OBJECTIVE_TAIL


def effective_best(streak: int, best: int | None, target: int = TARGET) -> int:
    """m = min(target, max(streak, best)). A stale-low best can never sit below the
    live streak; an above-target best (a winner) is capped at the solver cap."""
    return min(int(target), max(int(streak), int(best if best is not None else 0)))


def can_beat_best(streak: int, best: int, days: int, target: int = TARGET) -> bool:
    """True iff some outcome in ``days`` picks can push the streak past ``best``."""
    if days <= 0:
        return False
    return min(int(target), int(streak) + 2 * int(days)) > int(best)


def forced_tail_action(streak: int, best: int, days: int, target: int = TARGET) -> str:
    """Artifact-independent fallback for the tail regime: the stop rule, else a
    single. Used when the tail artifact (or the base policy) is missing/invalid so a
    failure can never silently reproduce the all-skip zero table."""
    m = effective_best(streak, best, target)
    return "single" if can_beat_best(streak, m, days, target) else "skip"


# --- solver --------------------------------------------------------------------

@dataclass
class TailSolution:
    """value[s, m, d, saver]: optimal E[season best] with d days left, marginalised
    over the next day's quality bin; policy[s, m, d, saver, q] in {0 skip, 1 single,
    2 double}. Cells with m < s are computed but never consulted (lookup clamps)."""
    value: np.ndarray
    policy: np.ndarray
    target: int
    max_days: int


def _validate_rates(freq, p_hit, p_both) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    freq = np.asarray(freq, dtype=float).ravel()
    p_hit = np.asarray(p_hit, dtype=float).ravel()
    p_both = np.asarray(p_both, dtype=float).ravel()
    if not (freq.shape == p_hit.shape == p_both.shape) or freq.size == 0:
        raise ValueError("freq, p_hit, p_both must be equal-length non-empty arrays")
    if not (np.all(np.isfinite(freq)) and np.all(np.isfinite(p_hit)) and np.all(np.isfinite(p_both))):
        raise ValueError("rates must be finite")
    if np.any(freq < 0) or abs(float(freq.sum()) - 1.0) > 1e-9:
        raise ValueError(f"bin frequencies must be >= 0 and sum to 1 (got {freq.sum()!r})")
    if np.any(p_hit < 0) or np.any(p_hit > 1) or np.any(p_both < 0) or np.any(p_both > p_hit):
        raise ValueError("need 0 <= p_both <= p_hit <= 1 in every bin")
    return freq, p_hit, p_both


def solve_emax_season_best(freq, p_hit, p_both, *, target: int = TARGET,
                           max_days: int = MAX_TAIL_DAYS) -> TailSolution:
    """Exact backward induction for E[max(m, future streak)] on (s, m, d, saver, q).

    Port of ``scripts/audit/skip_threshold_resolve.solve_emax`` (validated 2026-06-29)
    with two deliberate differences in POLICY EXTRACTION only (values are identical):
    the explicit stop rule, and play-first tie-breaking among exact maximisers
    (single, then double, then skip). The audit solver's skip-first ``argmax`` is
    what idled the reach-57 table.
    """
    freq, ph, pb = _validate_rates(freq, p_hit, p_both)
    if target < 1 or max_days < 1:
        raise ValueError("target and max_days must be >= 1")
    T, D, nq = int(target), int(max_days), freq.size
    Sg, Mg = np.meshgrid(np.arange(T + 1), np.arange(T + 1), indexing="ij")
    value = np.zeros((T + 1, T + 1, D + 1, 2))
    value[:, :, 0, :] = Mg[:, :, None]          # no days left: the season best is m
    policy = np.zeros((T + 1, T + 1, D + 1, 2, nq), dtype=np.int8)
    s1 = np.minimum(Sg + 1, T); m1 = np.maximum(Mg, s1)
    s2 = np.minimum(Sg + 2, T); m2 = np.maximum(Mg, s2)
    zero = np.zeros_like(Sg)
    lo, hi = _SAVER_ZONE
    for d in range(1, D + 1):
        EV = value[:, :, d - 1, :]                                  # [s', m', saver']
        stop = (np.minimum(T, Sg + 2 * d) <= Mg)[:, :, None]         # stop rule, broadcast over q
        for sv in (0, 1):
            catch = (sv == 1) & (Sg >= lo) & (Sg <= hi)
            ev_miss = np.where(catch, EV[Sg, Mg, 0], EV[zero, Mg, sv])
            v_skip = np.broadcast_to(EV[Sg, Mg, sv][:, :, None], (T + 1, T + 1, nq))
            v_single = ph[None, None, :] * EV[s1, m1, sv][:, :, None] + (1 - ph)[None, None, :] * ev_miss[:, :, None]
            v_double = pb[None, None, :] * EV[s2, m2, sv][:, :, None] + (1 - pb)[None, None, :] * ev_miss[:, :, None]
            best = np.maximum(np.maximum(v_skip, v_single), v_double)
            act = np.where(v_single == best, 1, np.where(v_double == best, 2, 0)).astype(np.int8)
            act = np.where(stop, np.int8(0), act)
            # At stop cells every branch equals m by construction; take v_skip (== m
            # exactly) rather than a p*m + (1-p)*m sum that can differ by an ulp.
            chosen = np.where(act == 1, v_single, np.where(act == 2, v_double, v_skip))
            value[:, :, d, sv] = chosen @ freq
            policy[:, :, d, sv, :] = act
    return TailSolution(value=value, policy=policy, target=T, max_days=D)


# --- artifact ------------------------------------------------------------------

_REQUIRED_KEYS = (
    "schema_version", "objective", "policy_table", "boundaries", "bin_freq",
    "bin_p_hit", "bin_p_both", "target", "max_days", "base_policy_sha256",
    "manifest", "built_at", "solver",
)


@dataclass
class TailPolicy:
    objective: str
    policy_table: np.ndarray
    boundaries: list[float]
    bin_freq: list[float]
    bin_p_hit: list[float]
    bin_p_both: list[float]
    target: int
    max_days: int
    base_policy_sha256: str
    manifest: dict
    built_at: str
    solver: str
    schema_version: str = TAIL_SCHEMA
    sha256: str | None = None   # of the artifact file, set by load_tail_policy

    @property
    def n_bins(self) -> int:
        return len(self.bin_freq)


def _is_sha256(s: object) -> bool:
    return isinstance(s, str) and len(s) == 64 and all(c in "0123456789abcdef" for c in s)


def _stop_rule_violations(table: np.ndarray, target: int, max_days: int) -> np.ndarray:
    """Indices (s, m, d, saver, q) of consulted cells that break the stop/play
    partition: consulted = s <= m <= target, 1 <= d <= max_days, s + 2d < target."""
    T, D = target, max_days
    nq = table.shape[-1]
    shape = (T + 1, T + 1, D + 1, 2, nq)
    s = np.broadcast_to(np.arange(T + 1)[:, None, None, None, None], shape)
    m = np.broadcast_to(np.arange(T + 1)[None, :, None, None, None], shape)
    d = np.broadcast_to(np.arange(D + 1)[None, None, :, None, None], shape)
    consulted = (s <= m) & (d >= 1) & (s + 2 * d < T)
    stop = np.minimum(T, s + 2 * d) <= m
    is_skip = table == 0
    bad = consulted & (is_skip != stop)
    return np.argwhere(bad)


@functools.lru_cache(maxsize=8)
def _solve_cached(freq: tuple, p_hit: tuple, p_both: tuple, target: int, max_days: int) -> np.ndarray:
    return solve_emax_season_best(np.array(freq), np.array(p_hit), np.array(p_both),
                                  target=target, max_days=max_days).policy


def _validate_manifest(tp: TailPolicy) -> None:
    m = tp.manifest
    if not isinstance(m, dict):
        raise TailPolicyError("manifest must be a JSON object")
    for key, typ in _MANIFEST_TYPES.items():
        if key not in m:
            raise TailPolicyError(f"manifest missing key {key!r}")
        v = m[key]
        if typ is int and (type(v) is not int):
            raise TailPolicyError(f"manifest {key!r} must be an int (got {v!r})")
        if typ is not int and not isinstance(v, typ):
            raise TailPolicyError(f"manifest {key!r} must be {typ.__name__} (got {v!r})")
    if m["n_bins"] != tp.n_bins:
        raise TailPolicyError(f"manifest n_bins {m['n_bins']} != {tp.n_bins} bins in the artifact")
    n = m["late_seed_days"]
    if n <= 0 or m["hits"] < 0 or m["both"] < 0 or m["both"] > m["hits"] or m["hits"] > n:
        raise TailPolicyError("manifest counts inconsistent (need 0 <= both <= hits <= late_seed_days)")
    if len(m["bin_counts"]) != tp.n_bins or sum(int(c) for c in m["bin_counts"]) != n:
        raise TailPolicyError("manifest bin_counts must have one entry per bin and sum to late_seed_days")
    if tp.n_bins == 1:
        if abs(tp.bin_p_hit[0] - m["hits"] / n) > 1e-9 or abs(tp.bin_p_both[0] - m["both"] / n) > 1e-9:
            raise TailPolicyError("manifest hits/both do not reproduce the one-bin rates")


def validate_tail_policy(tp: TailPolicy, *, expected_base_sha: str | None = None) -> None:
    """Raise TailPolicyError unless every field satisfies the contract. Exhaustive
    over the table (195,112 action cells at target 57 — cheap), not sampled, AND
    the table must equal an exact re-solve from the embedded rates (Codex r3: a
    stop/play partition alone accepts any single/double substitution)."""
    if tp.schema_version != TAIL_SCHEMA:
        raise TailPolicyError(f"schema_version {tp.schema_version!r} != {TAIL_SCHEMA!r}")
    if tp.objective != OBJECTIVE_TAIL:
        raise TailPolicyError(f"objective {tp.objective!r} != {OBJECTIVE_TAIL!r}")
    if int(tp.target) != TARGET:
        raise TailPolicyError(f"target {tp.target!r} != {TARGET}")
    if int(tp.max_days) != MAX_TAIL_DAYS:
        raise TailPolicyError(f"max_days {tp.max_days!r} != {MAX_TAIL_DAYS} (the v1 horizon is exact)")
    nq = len(tp.bin_freq)
    if nq < 1 or not (len(tp.bin_p_hit) == len(tp.bin_p_both) == nq):
        raise TailPolicyError("bin_freq / bin_p_hit / bin_p_both must be equal-length and non-empty")
    freq = np.asarray(tp.bin_freq, float); ph = np.asarray(tp.bin_p_hit, float); pb = np.asarray(tp.bin_p_both, float)
    if not (np.all(np.isfinite(freq)) and np.all(np.isfinite(ph)) and np.all(np.isfinite(pb))):
        raise TailPolicyError("bin rates must be finite")
    if np.any(freq < 0) or abs(float(freq.sum()) - 1.0) > 1e-9:
        raise TailPolicyError(f"bin frequency must be >= 0 and sum to 1 (got {freq.tolist()})")
    if np.any(ph < 0) or np.any(ph > 1) or np.any(pb < 0) or np.any(pb > ph):
        raise TailPolicyError(f"need 0 <= p_both <= p_hit <= 1 (p_hit={ph.tolist()}, p_both={pb.tolist()})")
    b = np.asarray(tp.boundaries, float).ravel()
    if b.size != nq - 1 or not np.all(np.isfinite(b)) or (b.size > 1 and not np.all(np.diff(b) > 0)):
        raise TailPolicyError(f"boundaries must be {nq - 1} finite strictly-increasing floats (got {b.tolist()})")
    table = tp.policy_table
    if not isinstance(table, np.ndarray) or table.dtype != np.int8:
        raise TailPolicyError(f"policy_table dtype must be int8 (got {getattr(table, 'dtype', type(table))})")
    want = (tp.target + 1, tp.target + 1, tp.max_days + 1, 2, nq)
    if table.shape != want:
        raise TailPolicyError(f"policy_table shape {table.shape} != {want}")
    if table.min() < 0 or table.max() > 2:
        raise TailPolicyError("policy_table action out of range (must be 0 skip / 1 single / 2 double)")
    if np.any(table[:, :, 0] != 0):
        raise TailPolicyError("policy_table day 0 slice must be all skip (never consulted)")
    if not _is_sha256(tp.base_policy_sha256):
        raise TailPolicyError(f"base policy sha256 malformed: {tp.base_policy_sha256!r}")
    if expected_base_sha is not None and tp.base_policy_sha256 != expected_base_sha:
        raise TailPolicyError(
            f"base policy sha256 mismatch: artifact pairs with {tp.base_policy_sha256[:12]}, "
            f"current base policy is {expected_base_sha[:12]} — rebuild the tail")
    _validate_manifest(tp)
    for name in ("built_at", "solver"):
        if not isinstance(getattr(tp, name), str) or not getattr(tp, name):
            raise TailPolicyError(f"{name} must be a non-empty string")
    bad = _stop_rule_violations(table, tp.target, tp.max_days)
    if len(bad):
        first = tuple(int(x) for x in bad[0])
        raise TailPolicyError(
            f"stop rule violated at {len(bad)} consulted cells (first (s, m, d, saver, q) = {first}): "
            f"skip iff min(target, s + 2d) <= m")
    expected = _solve_cached(tuple(map(float, freq)), tuple(map(float, ph)), tuple(map(float, pb)),
                             int(tp.target), int(tp.max_days))
    if not np.array_equal(table, expected):
        n_diff = int((table != expected).sum())
        first = tuple(int(x) for x in np.argwhere(table != expected)[0])
        raise TailPolicyError(
            f"policy_table is not the exact E[season-best] policy for its embedded rates "
            f"({n_diff} cells differ; first (s, m, d, saver, q) = {first})")


def _scalar_str(arr) -> str:
    v = np.asarray(arr)
    if v.shape != () or v.dtype.kind not in ("U", "S"):
        raise TailPolicyError(f"expected a string scalar, got shape {v.shape} dtype {v.dtype}")
    v = v.item()
    return v.decode() if isinstance(v, bytes) else str(v)


def _scalar_int(arr, name: str) -> int:
    v = np.asarray(arr)
    if v.shape != () or v.dtype.kind not in ("i", "u"):
        raise TailPolicyError(f"{name} must be an integer scalar")
    return int(v.item())


def save_tail_policy(tp: TailPolicy, path: Path | str = DEFAULT_TAIL_POLICY_PATH,
                     *, validate: bool = True) -> Path:
    """Atomic ``savez_compressed`` (temp file + ``os.replace``, as MDPSolution.save).
    ``validate=False`` exists ONLY so tests can write deliberately broken artifacts."""
    if validate:
        validate_tail_policy(tp)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(
                f,
                schema_version=np.array(tp.schema_version),
                objective=np.array(tp.objective),
                policy_table=np.asarray(tp.policy_table),
                boundaries=np.asarray(tp.boundaries, dtype=float).ravel(),
                bin_freq=np.asarray(tp.bin_freq, dtype=float).ravel(),
                bin_p_hit=np.asarray(tp.bin_p_hit, dtype=float).ravel(),
                bin_p_both=np.asarray(tp.bin_p_both, dtype=float).ravel(),
                target=np.array(int(tp.target)),
                max_days=np.array(int(tp.max_days)),
                base_policy_sha256=np.array(str(tp.base_policy_sha256)),
                manifest=np.array(json.dumps(tp.manifest, sort_keys=True)),
                built_at=np.array(str(tp.built_at)),
                solver=np.array(str(tp.solver)),
            )
        try:
            mode = path.stat().st_mode & 0o777
        except FileNotFoundError:
            umask = os.umask(0)
            os.umask(umask)
            mode = 0o666 & ~umask
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_tail_policy(path: Path | str = DEFAULT_TAIL_POLICY_PATH, *,
                     expected_base_sha: str | None = None) -> TailPolicy:
    """Strict loader: no pickle, every key required, every field validated, the
    whole table checked against the stop rule, and (when given) hard-bound to the
    sha256 of the reach-57 policy it must pair with. Raises TailPolicyError on ANY
    problem — callers translate that into the forced fallback, never into the base
    table's zero region."""
    path = Path(path)
    if not path.exists():
        raise TailPolicyError(f"tail policy artifact missing: {path}")
    try:
        # ONE read: the sha256 recorded on the decision is the hash of exactly the
        # bytes the actions came from (no second open that a concurrent atomic
        # replace could retarget).
        raw = path.read_bytes()
        if len(raw) > MAX_ARTIFACT_BYTES:
            raise TailPolicyError(
                f"tail policy artifact too large ({len(raw)} bytes > {MAX_ARTIFACT_BYTES})")
        with np.load(io.BytesIO(raw), allow_pickle=False) as data:
            missing = [k for k in _REQUIRED_KEYS if k not in data.files]
            if missing:
                raise TailPolicyError(f"tail policy artifact missing key(s): {', '.join(missing)}")
            try:
                manifest = json.loads(_scalar_str(data["manifest"]))
            except (TypeError, ValueError) as exc:
                raise TailPolicyError(f"manifest is not valid JSON: {exc}") from exc
            tp = TailPolicy(
                schema_version=_scalar_str(data["schema_version"]),
                objective=_scalar_str(data["objective"]),
                policy_table=np.asarray(data["policy_table"]),
                boundaries=np.asarray(data["boundaries"], dtype=float).ravel().tolist(),
                bin_freq=np.asarray(data["bin_freq"], dtype=float).ravel().tolist(),
                bin_p_hit=np.asarray(data["bin_p_hit"], dtype=float).ravel().tolist(),
                bin_p_both=np.asarray(data["bin_p_both"], dtype=float).ravel().tolist(),
                target=_scalar_int(data["target"], "target"),
                max_days=_scalar_int(data["max_days"], "max_days"),
                base_policy_sha256=_scalar_str(data["base_policy_sha256"]),
                manifest=manifest,
                built_at=_scalar_str(data["built_at"]),
                solver=_scalar_str(data["solver"]),
                sha256=hashlib.sha256(raw).hexdigest(),
            )
    except TailPolicyError:
        raise
    except Exception as exc:   # zip/npz corruption, OSError, bad dtypes
        raise TailPolicyError(f"tail policy artifact unreadable at {path}: {exc!r}") from exc
    validate_tail_policy(tp, expected_base_sha=expected_base_sha)
    return tp


# --- lookup --------------------------------------------------------------------

def classify_bin(p_game_hit: float, boundaries: list[float]) -> int:
    q = 0
    for i, boundary in enumerate(boundaries):
        if p_game_hit >= boundary:
            q = i + 1
    return min(q, len(boundaries))


def lookup_tail_action(tail: TailPolicy, streak: int, best: int | None, days: int,
                       saver: bool, p_game_hit: float) -> str:
    """Production entry point for the tail regime. ``days`` must already be the
    effective (clamped) day count; the regime predicate guarantees days <= max_days
    for target 57, so exceeding the table horizon is a contract error, not a clamp."""
    if days <= 0 or streak >= tail.target:
        return "skip"
    if days > tail.max_days:
        raise TailPolicyError(
            f"days {days} beyond the tail horizon {tail.max_days}: state is not a tail state")
    m = effective_best(streak, best, tail.target)
    q = classify_bin(float(p_game_hit), tail.boundaries)
    return ACTIONS[int(tail.policy_table[int(streak), m, int(days), int(bool(saver)), q])]
