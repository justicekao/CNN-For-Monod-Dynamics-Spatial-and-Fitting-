"""
eval.py -- the two metrics docs/curriculum_nn_spec.md's "Evaluation"
section calls for:

1. Parameter recovery error: relative L2 error per predicted matrix
   against known synthetic ground truth.
2. Trajectory reconstruction error: re-simulate with the PREDICTED
   parameters and compare to the input trajectory -- this catches "wrong
   parameters that still fit the data" (non-identifiability), which
   parameter error alone won't, per the spec's own reasoning.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List

import numpy as np
import torch

from shared.monod_core import simulate_well_mixed
from .curriculum import batch_to_tensors
from .data_gen import Sample, Tier
from .model import MonodParameterGNN, PredictedParams

_CORE_KEYS = ["r", "k", "c", "delta", "m_supply", "D_dilution"]


def parameter_recovery_error(pred: PredictedParams, targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
    def rel_err(p: torch.Tensor, t: torch.Tensor) -> float:
        p_np, t_np = p.detach().numpy(), t.detach().numpy()
        return float(np.linalg.norm(p_np - t_np) / (np.linalg.norm(t_np) + 1e-8))

    out = {k: rel_err(getattr(pred, k), targets[k]) for k in _CORE_KEYS}
    if pred.P is not None:
        out["P"] = rel_err(pred.P, targets["P"])
        out["K_tox"] = rel_err(pred.K_tox, targets["K_tox"])
    return out


def trajectory_reconstruction_rmse(pred: PredictedParams, sample: Sample, idx: int) -> float:
    """Re-simulate sample `idx` from ITS OWN initial condition but with the
    model's PREDICTED parameters, and compare the resulting log-population
    trajectory to the true one."""
    overrides = dict(
        r=pred.r[idx].detach().numpy(), k=pred.k[idx].detach().numpy(), c=pred.c[idx].detach().numpy(),
        delta=pred.delta[idx].detach().numpy(), m_supply=pred.m_supply[idx].detach().numpy(),
        D_dilution=pred.D_dilution[idx].detach().numpy(),
    )
    if pred.P is not None:
        overrides["P"] = pred.P[idx].detach().numpy()
        overrides["K_tox"] = pred.K_tox[idx].detach().numpy()
    predicted_config = replace(sample.config, **overrides)

    try:
        result = simulate_well_mixed(
            predicted_config, N0=sample.N[0], x0=sample.x[0], tox0=sample.tox[0],
            t_span=(float(sample.t[0]), float(sample.t[-1])), n_points=len(sample.t),
        )
    except RuntimeError:
        return float("inf")  # predicted params were bad enough to blow up integration

    true_log = np.log(np.maximum(sample.N, 1e-8))
    pred_log = np.log(np.maximum(result.N, 1e-8))
    return float(np.sqrt(np.mean((true_log - pred_log) ** 2)))


def evaluate_tier(model: MonodParameterGNN, samples: List[Sample]) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        (sf, mf, tf), targets = batch_to_tensors(samples)
        pred = model(sf, mf, tf)
        param_err = parameter_recovery_error(pred, targets)
    traj_rmse = float(np.mean([trajectory_reconstruction_rmse(pred, samples[i], i) for i in range(len(samples))]))
    return {**{f"param_rel_err_{k}": v for k, v in param_err.items()}, "trajectory_rmse_logspace": traj_rmse}


def evaluate_dataset(model: MonodParameterGNN, data: Dict[Tier, List[Sample]]) -> Dict[Tier, Dict[str, float]]:
    return {tier: evaluate_tier(model, samples) for tier, samples in data.items() if samples}
