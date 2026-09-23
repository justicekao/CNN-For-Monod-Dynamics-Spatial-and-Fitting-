"""
curriculum.py -- batching, loss, and the curriculum training loop itself.

The curriculum IS the reuse of one model/optimizer across stages: because
MonodParameterGNN has no S/M/T baked into its weights (see model.py), the
"transfer" from an easy small-dimensionality stage to a harder
large-dimensionality one is literally just continuing to train the same
parameters on new (bigger) batches -- there is no separate
padding/fine-tuning machinery needed, which is exactly the case
docs/curriculum_nn_spec.md's architecture section argues for.

Batching is done PER TIER (all samples in one batch share the same
(S, M, T), since MonodParameterGNN's forward pass is written for one
fixed shape per call) rather than via padding+masking across tiers --
see data_gen.py's StageSpec, whose tiers are exactly the discrete
dimensionality buckets this assumes.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from .data_gen import Sample, StageSpec, Tier
from .model import MonodParameterGNN, featurize_sample

_CORE_KEYS = ["r", "k", "c", "delta", "m_supply", "D_dilution"]


def batch_to_tensors(samples: List[Sample]):
    """samples must all share one (S, M, T) tier."""
    feats = [featurize_sample(s.N, s.x, s.tox) for s in samples]
    strain_feats = torch.tensor(np.stack([f[0] for f in feats]), dtype=torch.float32)
    metab_feats = torch.tensor(np.stack([f[1] for f in feats]), dtype=torch.float32)
    toxin_feats = torch.tensor(np.stack([f[2] for f in feats]), dtype=torch.float32)

    targets = {
        "r": torch.tensor(np.stack([s.config.r for s in samples]), dtype=torch.float32),
        "k": torch.tensor(np.stack([s.config.k for s in samples]), dtype=torch.float32),
        "c": torch.tensor(np.stack([s.config.c for s in samples]), dtype=torch.float32),
        "delta": torch.tensor(np.stack([s.config.delta for s in samples]), dtype=torch.float32),
        "m_supply": torch.tensor(np.stack([s.config.m_supply for s in samples]), dtype=torch.float32),
        "D_dilution": torch.tensor(np.stack([s.config.D_dilution for s in samples]), dtype=torch.float32),
    }
    if samples[0].T:
        targets["P"] = torch.tensor(np.stack([s.config.P for s in samples]), dtype=torch.float32)
        targets["K_tox"] = torch.tensor(np.stack([s.config.K_tox for s in samples]), dtype=torch.float32)
    return (strain_feats, metab_feats, toxin_feats), targets


def param_loss(pred, targets: Dict[str, torch.Tensor], eps: float = 1e-6) -> torch.Tensor:
    """Sum of log-space MSE per matrix. Log-space (not raw MSE) because
    the matrices span very different scales (populations' delta ~ O(1),
    metabolite supply ~ O(1), toxin half-saturation ~ O(1) too here, but
    in general these need not share a scale) and because the model's own
    outputs are already positivity-constrained via softplus -- comparing
    in log-space keeps relative, not absolute, errors comparable across
    matrices of different magnitude."""
    def log_mse(p, t):
        return torch.mean((torch.log(p + eps) - torch.log(t + eps)) ** 2)

    loss = sum(log_mse(getattr(pred, k), targets[k]) for k in _CORE_KEYS)
    if pred.P is not None:
        loss = loss + log_mse(pred.P, targets["P"]) + log_mse(pred.K_tox, targets["K_tox"])
    return loss


def _mean_loss_over_tiers(model: MonodParameterGNN, data: Dict[Tier, List[Sample]]) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for samples in data.values():
            if not samples:
                continue
            (sf, mf, tf), targets = batch_to_tensors(samples)
            losses.append(param_loss(model(sf, mf, tf), targets).item())
    return float(np.mean(losses)) if losses else float("nan")


def train_stage(model: MonodParameterGNN, optimizer: torch.optim.Optimizer,
                 train_data: Dict[Tier, List[Sample]], val_data: Dict[Tier, List[Sample]],
                 n_epochs: int, log_every: int = 20) -> List[Tuple[int, float, float]]:
    """One epoch = one gradient step per tier (full-batch per tier --
    the per-tier dataset sizes used in this repo's tests/CLI default are
    small enough that mini-batching within a tier isn't needed; split a
    tier's sample list into chunks yourself first if you scale up
    n_per_tier enough that it stops fitting comfortably in memory/a
    single forward pass)."""
    history = []
    tiers = [t for t, s in train_data.items() if s]
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for tier in tiers:
            samples = train_data[tier]
            (sf, mf, tf), targets = batch_to_tensors(samples)
            optimizer.zero_grad()
            loss = param_loss(model(sf, mf, tf), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % log_every == 0 or epoch == n_epochs - 1:
            val_loss = _mean_loss_over_tiers(model, val_data)
            history.append((epoch, total_loss / max(len(tiers), 1), val_loss))
    return history


def train_curriculum(model: MonodParameterGNN, optimizer: torch.optim.Optimizer,
                      stage_datasets: List[Tuple[StageSpec, Dict[Tier, List[Sample]], Dict[Tier, List[Sample]]]],
                      n_epochs_per_stage: int, log_every: int = 20) -> Dict[str, List[Tuple[int, float, float]]]:
    """stage_datasets: [(stage, train_data, val_data), ...] IN CURRICULUM
    ORDER (small dimensionality first). The SAME model and optimizer are
    reused across every stage -- see this module's docstring."""
    history = {}
    for stage, train_data, val_data in stage_datasets:
        history[stage.name] = train_stage(model, optimizer, train_data, val_data, n_epochs_per_stage, log_every)
    return history
