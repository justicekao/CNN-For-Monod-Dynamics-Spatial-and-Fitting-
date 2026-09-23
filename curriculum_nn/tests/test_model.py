"""
The whole point of the bipartite message-passing architecture (see
model.py's docstring) is permutation EQUIVARIANCE: relabeling which
strain is "strain 0" vs "strain 1" is arbitrary, so the network's
predictions must permute along with the input, not change in value.
Separately, since it has no S/M/T baked into its weights, the identical
module instance must accept any (S, M, T) shape, including T=0.
"""

import numpy as np
import torch

from curriculum_nn.src.model import MonodParameterGNN, N_SUMMARY_POINTS, featurize_sample


def _random_batch(rng, B, S, M, T, n_points=20):
    N = rng.uniform(1.0, 5.0, size=(B, n_points, S))
    x = rng.uniform(0.5, 3.0, size=(B, n_points, M))
    tox = rng.uniform(0.2, 2.0, size=(B, n_points, T))
    sf, mf, tf = zip(*(featurize_sample(N[b], x[b], tox[b]) for b in range(B)))
    return (torch.tensor(np.stack(sf), dtype=torch.float32),
            torch.tensor(np.stack(mf), dtype=torch.float32),
            torch.tensor(np.stack(tf), dtype=torch.float32))


def test_forward_pass_shapes_for_various_dimensionalities():
    torch.manual_seed(0)
    model = MonodParameterGNN(embed_dim=16, n_rounds=2, hidden=32)
    rng = np.random.default_rng(0)

    for S, M, T in [(2, 3, 0), (2, 3, 1), (4, 6, 2)]:
        sf, mf, tf = _random_batch(rng, B=3, S=S, M=M, T=T)
        out = model(sf, mf, tf)
        assert out.r.shape == (3, S, M)
        assert out.k.shape == (3, S, M)
        assert out.c.shape == (3, S, M)
        assert out.delta.shape == (3, S)
        assert out.m_supply.shape == (3, M)
        assert out.D_dilution.shape == (3, M)
        if T:
            assert out.P.shape == (3, S, T)
            assert out.K_tox.shape == (3, S, T)
        else:
            assert out.P is None and out.K_tox is None
        assert torch.all(out.r >= 0) and torch.all(out.k > 0) and torch.all(out.c >= 0)
        assert torch.all(out.delta >= 0)


def test_predictions_are_permutation_equivariant_in_strain_order():
    torch.manual_seed(1)
    model = MonodParameterGNN(embed_dim=16, n_rounds=2, hidden=32)
    rng = np.random.default_rng(1)
    sf, mf, tf = _random_batch(rng, B=2, S=4, M=3, T=1)

    perm = torch.tensor([3, 1, 0, 2])
    out = model(sf, mf, tf)
    out_perm = model(sf[:, perm, :], mf, tf)

    torch.testing.assert_close(out_perm.r, out.r[:, perm, :], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(out_perm.delta, out.delta[:, perm], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(out_perm.P, out.P[:, perm, :], atol=1e-5, rtol=1e-4)


def test_predictions_are_permutation_equivariant_in_metabolite_order():
    torch.manual_seed(2)
    model = MonodParameterGNN(embed_dim=16, n_rounds=2, hidden=32)
    rng = np.random.default_rng(2)
    sf, mf, tf = _random_batch(rng, B=2, S=3, M=5, T=0)

    perm = torch.tensor([4, 2, 0, 3, 1])
    out = model(sf, mf, tf)
    out_perm = model(sf, mf[:, perm, :], tf)

    torch.testing.assert_close(out_perm.r, out.r[:, :, perm], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(out_perm.m_supply, out.m_supply[:, perm], atol=1e-5, rtol=1e-4)


def test_featurize_sample_is_invariant_to_n_points():
    """A trajectory summary shouldn't depend on how many timepoints it
    happened to be simulated with, since different samples/stages may use
    different n_points -- only the summary shape (N_SUMMARY_POINTS) must
    match so the model can consume samples generated with different
    settings in the same batch."""
    # log(N) LINEAR in time, so piecewise-linear interpolation reconstructs
    # the exact same values at any query point regardless of how many
    # source points span it -- isolating "does the normalized-time
    # resampling logic itself work" from "how much error does linear
    # interpolation of a CURVED function pick up at low resolution", which
    # is a separate (expected, real) numerical-approximation concern.
    t_fine = np.linspace(0, 10, 200)
    N_fine = np.exp(np.outer(0.3 * t_fine - 1.0, np.ones(2)))
    t_coarse = np.linspace(0, 10, 10)
    N_coarse = np.exp(np.outer(0.3 * t_coarse - 1.0, np.ones(2)))

    feat_fine, _, _ = featurize_sample(N_fine, np.ones((200, 1)), np.zeros((200, 0)))
    feat_coarse, _, _ = featurize_sample(N_coarse, np.ones((10, 1)), np.zeros((10, 0)))
    assert feat_fine.shape == feat_coarse.shape == (2, N_SUMMARY_POINTS)
    np.testing.assert_allclose(feat_fine, feat_coarse, atol=1e-8)
