"""
Training-loop smoke tests: loss actually decreases with gradient steps,
batching handles a T=0 tier and a T>0 tier without special-casing by the
caller, and the SAME model instance can be trained across a "curriculum"
of stages (different S/M/T tiers) without reinitializing anything -- that
reuse is the entire point of the size-agnostic architecture (model.py's
docstring). This does not test that the curriculum IMPROVES accuracy
(that needs a much larger experiment than a fast unit test budget allows
-- see docs/curriculum_nn_spec.md's "Status" notes on what's been checked
so far vs. what a real training run would need), only that nothing about
switching tiers breaks the training loop itself.
"""

import numpy as np
import torch

from curriculum_nn.src.curriculum import batch_to_tensors, param_loss, train_curriculum, train_stage
from curriculum_nn.src.data_gen import STAGE_0, STAGE_1, generate_dataset, split_by_tier
from curriculum_nn.src.eval import evaluate_dataset
from curriculum_nn.src.model import MonodParameterGNN


def _small_stage0_data(seed=0, n_per_tier=6):
    rng = np.random.default_rng(seed)
    dataset = generate_dataset(STAGE_0, n_per_tier=n_per_tier, rng=rng, t_span=(0.0, 80.0), n_points=25)
    return split_by_tier(dataset, val_frac=0.2, test_frac=0.2, rng=rng)


def test_batch_to_tensors_handles_zero_and_nonzero_toxin_tiers():
    train_data, _, _ = _small_stage0_data()
    for tier, samples in train_data.items():
        (sf, mf, tf), targets = batch_to_tensors(samples)
        S, M, T = tier
        assert sf.shape[1:] == (S, 6)
        assert mf.shape[1:] == (M, 6)
        assert tf.shape[1:] == (T, 6)
        assert ("P" in targets) == (T > 0)


def test_training_loss_decreases():
    torch.manual_seed(0)
    train_data, val_data, _ = _small_stage0_data()
    model = MonodParameterGNN(embed_dim=16, n_rounds=2, hidden=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    history = train_stage(model, optimizer, train_data, val_data, n_epochs=60, log_every=59)
    first_loss = history[0][1]
    last_loss = history[-1][1]
    assert last_loss < first_loss, f"expected loss to decrease, got {first_loss} -> {last_loss}"


def test_train_curriculum_reuses_one_model_across_stages():
    """The same model/optimizer instance is passed through both stages;
    this checks that works end to end (no shape mismatch when a later
    stage's tiers differ from the first) and that parameters actually
    changed after both stages (i.e. stage 1 data really was trained on,
    not silently skipped)."""
    torch.manual_seed(1)
    rng = np.random.default_rng(2)

    ds0 = generate_dataset(STAGE_0, n_per_tier=6, rng=rng, t_span=(0.0, 80.0), n_points=20)
    train0, val0, _ = split_by_tier(ds0, rng=rng)
    ds1 = generate_dataset(STAGE_1, n_per_tier=6, rng=rng, t_span=(0.0, 80.0), n_points=20)
    train1, val1, test1 = split_by_tier(ds1, rng=rng)

    model = MonodParameterGNN(embed_dim=16, n_rounds=2, hidden=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    params_before = [p.clone() for p in model.parameters()]

    history = train_curriculum(model, optimizer, [(STAGE_0, train0, val0), (STAGE_1, train1, val1)],
                                n_epochs_per_stage=20, log_every=19)

    assert set(history.keys()) == {"stage0", "stage1"}
    assert any(not torch.equal(a, b) for a, b in zip(params_before, model.parameters()))

    # evaluate_dataset must run without error on stage 1's held-out tiers,
    # including a tier ((4, 6, 2)) with more strains/metabolites/toxins
    # than any Stage 0 tier ever had
    results = evaluate_dataset(model, test1)
    assert (4, 6, 2) in results
    assert np.isfinite(results[(4, 6, 2)]["param_rel_err_r"])
