from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from bts import strategy
from bts.simulate import mdp, tail_policy
from bts.simulate.pooled_policy import compute_pooled_bins, build_pooled_policy
from bts.simulate.quality_bins import compute_bins, compute_bins_with_boundaries
from scratch import inject


@st.composite
def confidence_profile(draw):
    values = st.integers(1, 9).map(lambda value: value / 10)
    shape = draw(st.sampled_from(("constant", "outlier", "few", "distinct", "tiny")))
    if shape == "constant":
        return [draw(values)] * draw(st.integers(1, 12))
    if shape == "outlier":
        return [draw(values)] * draw(st.integers(2, 10)) + [draw(values)]
    if shape == "few":
        levels = draw(st.lists(values, min_size=1, max_size=3, unique=True))
        return draw(st.lists(st.sampled_from(levels), min_size=1, max_size=12))
    if shape == "distinct":
        return draw(st.lists(values, min_size=5, max_size=9, unique=True))
    return draw(st.lists(values, min_size=1, max_size=3))


def frame(values, offset=0):
    return pd.DataFrame([
        (f"2027-04-{offset + i + 1:02d}", rank, p, (i + rank) % 3 != 0, 1)
        for i, p in enumerate(values) for rank in (1, 2)
    ], columns=["date", "rank", "p_game_hit", "actual_hit", "seed"])


def probes(values, boundaries):
    return sorted({0.0, 1.0, *values, *boundaries,
                   *(np.nextafter(b, 0.0) for b in boundaries),
                   *(np.nextafter(b, 1.0) for b in boundaries)})


def lookup_indices(bins, p):
    """Expose the exact consulted column with one sentinel table per column."""
    width = len(bins.boundaries) + 1
    base = np.zeros((58, 2, 2, width), dtype=np.int8)
    tail = SimpleNamespace(
        policy_table=np.zeros((58, 58, 2, 2, width), dtype=np.int8),
        boundaries=bins.boundaries, target=57, max_days=1, base_policy_sha256="stub",
    )
    live = dict(policy_table=base, boundaries=bins.boundaries, season_length=1,
                tail=tail, base_sha256="stub")
    selections = [[], [], [], []]
    for q in range(width):
        base[..., q] = 1
        tail.policy_table[..., q] = 1
        actions = (
            mdp.lookup_action(base, bins.boundaries, 0, 1, True, p, 1),
            tail_policy.lookup_tail_action(tail, 0, 0, 1, True, p),
            strategy._mdp_action_from(live, p, 0, "2027-04-01", True),
            strategy._tail_action(live, p, 0, 1, True, 0)[0],
        )
        for found, action in zip(selections, actions):
            if action == "single":
                found.append(q)
        base[..., q] = 0
        tail.policy_table[..., q] = 0
    return [bins.classify(p), tail_policy.classify_bin(p, bins.boundaries),
            *(found[0] if len(found) == 1 else -1 for found in selections)]


@pytest.mark.parametrize("builder", [compute_bins, compute_pooled_bins])
@settings(max_examples=60, deadline=None, database=None)
@given(values=confidence_profile())
def test_table_width(builder, values):
    bins = builder(frame(values))
    from os import environ
    with pytest.MonkeyPatch.context() as mp:
        if environ.get("BTS_PROPTEST_MUTATE"):
            # A supported five-column control isolates the mutation from the known
            # dropped-bin failure in the original trainers.
            bins = compute_bins_with_boundaries(frame(values), bins.boundaries)
            assert len(bins.bins) == len(bins.boundaries) + 1
            inject(mp, mdp, "solve_mdp",
                   "policy_table=policy,", "policy_table=policy[..., :-1],")
        table = mdp.solve_mdp(bins, season_length=1).policy_table
        columns = {q for p in probes(values, bins.boundaries) for q in lookup_indices(bins, p)}
        assert table.shape[-1] == len(bins.boundaries) + 1
        assert columns <= set(range(table.shape[-1]))


@settings(max_examples=60, deadline=None, database=None)
@given(early=confidence_profile(), late=confidence_profile())
def test_phase_width(early, late):
    joined = pd.concat([frame(early), frame(late, 14)], ignore_index=True)
    solution = train_pooled_policy(joined, season_length=2, late_phase_days=len(late))
    assert solution.policy_table.shape[-1] == len(solution.quality_bins.boundaries) + 1


@settings(max_examples=60, deadline=None, database=None)
@given(values=confidence_profile())
def test_lookup_indices(values):
    from os import environ
    bins = compute_bins(frame(values))
    with pytest.MonkeyPatch.context() as mp:
        if environ.get("BTS_PROPTEST_MUTATE"):
            inject(mp, tail_policy, "classify_bin",
                   "if p_game_hit >= boundary:", "if p_game_hit < boundary:")
        previous = [-1] * 6
        for p in probes(values, bins.boundaries):
            current = lookup_indices(bins, p)
            assert all(low <= high <= len(bins.boundaries)
                       for low, high in zip(previous, current))
            assert min(current) >= 0
            previous = current


@settings(max_examples=60, deadline=None, database=None)
@given(values=confidence_profile())
def test_assignment(values):
    from os import environ
    for bins in (compute_bins(frame(values)), compute_pooled_bins(frame(values))):
        with pytest.MonkeyPatch.context() as mp:
            if environ.get("BTS_PROPTEST_MUTATE"):
                inject(mp, mdp, "lookup_action",
                       "if top_pick_confidence >= boundary:", "if top_pick_confidence > boundary:")
            for value in set(values):
                assigned, = [b.index for b in bins.bins if b.p_range[0] <= value <= b.p_range[1]]
                assert lookup_indices(bins, value) == [assigned] * 6


@settings(max_examples=60, deadline=None, database=None)
@given(values=confidence_profile())
def test_roundtrip(values):
    from os import environ
    boundaries = compute_bins(frame(values)).boundaries
    bins = compute_bins_with_boundaries(frame(values), boundaries)
    solution = mdp.solve_mdp(bins, season_length=1)
    # Distinguish neighbouring columns even when the optimizer selects identical actions.
    solution.policy_table[:] = np.arange(len(bins.bins)) % 3
    with TemporaryDirectory() as directory, pytest.MonkeyPatch.context() as mp:
        if environ.get("BTS_PROPTEST_MUTATE"):
            inject(mp, mdp, "load_policy",
                   'data["boundaries"].tolist(),', '(data["boundaries"] + 0.01).tolist(),')
        table, reloaded, days = mdp.load_policy(solution.save(Path(directory) / "policy.npz"))
        for p in probes(values, boundaries):
            assert mdp.lookup_action(table, reloaded, 55, days, True, p, days) == (
                mdp.lookup_action(solution.policy_table, boundaries, 55, 1, True, p, 1)
            )
