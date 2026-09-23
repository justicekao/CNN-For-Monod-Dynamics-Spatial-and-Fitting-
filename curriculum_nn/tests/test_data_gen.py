"""
Per docs/curriculum_nn_spec.md's suggested test: at minimum, generated
trajectories are finite, bounded, and reproduce known equilibria.
"""

import numpy as np
import pytest

from curriculum_nn.src.data_gen import (
    STAGE_0, STAGE_1, generate_dataset, generate_sample, sample_config, solve_equilibrium, split_by_tier,
)
from shared.monod_core import reaction_rhs


_STAGE_TIER_PAIRS = [(STAGE_0, tier) for tier in STAGE_0.tiers] + [(STAGE_1, tier) for tier in STAGE_1.tiers]


@pytest.mark.parametrize("stage,tier", _STAGE_TIER_PAIRS)
def test_sampled_config_has_a_genuine_equilibrium(stage, tier):
    """The whole point of solve_equilibrium(): (N_star, x_star, tox_star)
    must be an actual fixed point (dN=dx=dtox=0) of the sampled config."""
    S, M, T = tier
    rng = np.random.default_rng(42)
    config, N_star, x_star, tox_star = sample_config(S, M, T, rng, stage)
    rs = reaction_rhs(N_star, x_star, tox_star, None, config)
    np.testing.assert_allclose(rs.dN, 0.0, atol=1e-8)
    np.testing.assert_allclose(rs.dx, 0.0, atol=1e-8)
    if T:
        np.testing.assert_allclose(rs.dtox, 0.0, atol=1e-8)


@pytest.mark.parametrize("tier", STAGE_0.tiers)
def test_generated_trajectory_is_finite_and_positive(tier):
    S, M, T = tier
    rng = np.random.default_rng(7)
    sample = generate_sample(S, M, T, rng, STAGE_0, t_span=(0.0, 100.0), n_points=30)

    assert np.all(np.isfinite(sample.N))
    assert np.all(np.isfinite(sample.x))
    assert np.all(np.isfinite(sample.tox))
    assert np.all(sample.N > 0)
    assert np.all(sample.x > 0)
    if T:
        assert np.all(sample.tox > 0)
    assert sample.N.shape == (30, S)
    assert sample.x.shape == (30, M)
    assert sample.tox.shape == (30, T)


def test_perturbed_trajectory_relaxes_back_toward_its_equilibrium():
    """Since the config was solved so (N_star, x_star) is a genuine
    equilibrium and the perturbation is modest, the trajectory should end
    up closer to that equilibrium than it started (this is what makes the
    generated data "informative but not degenerate", per the spec)."""
    rng = np.random.default_rng(3)
    S, M, T = 2, 3, 0
    config, N_star, x_star, tox_star = sample_config(S, M, T, rng, STAGE_0)
    N0 = N_star * 2.0  # a real, deliberate perturbation
    from shared.monod_core import simulate_well_mixed
    result = simulate_well_mixed(config, N0=N0, x0=x_star, t_span=(0.0, 300.0), n_points=100)

    start_dist = np.abs(N0 - N_star).sum()
    end_dist = np.abs(result.N[-1] - N_star).sum()
    assert end_dist < start_dist


def test_generate_dataset_and_split_by_tier_cover_every_tier():
    rng = np.random.default_rng(11)
    dataset = generate_dataset(STAGE_0, n_per_tier=10, rng=rng, t_span=(0.0, 50.0), n_points=20)
    assert set(dataset.keys()) == set(STAGE_0.tiers)
    for tier in STAGE_0.tiers:
        assert len(dataset[tier]) == 10

    train, val, test = split_by_tier(dataset, val_frac=0.2, test_frac=0.2, rng=rng)
    for tier in STAGE_0.tiers:
        assert len(train[tier]) + len(val[tier]) + len(test[tier]) == 10
        assert len(val[tier]) == 2
        assert len(test[tier]) == 2


def test_sample_params_returns_expected_matrices():
    rng = np.random.default_rng(5)
    sample = generate_sample(3, 4, 1, rng, STAGE_1, t_span=(0.0, 50.0), n_points=10)
    params = sample.params()
    assert params["r"].shape == (3, 4)
    assert params["k"].shape == (3, 4)
    assert params["c"].shape == (3, 4)
    assert params["delta"].shape == (3,)
    assert params["m_supply"].shape == (4,)
    assert params["D_dilution"].shape == (4,)
    assert params["P"].shape == (3, 1)
    assert params["K_tox"].shape == (3, 1)
