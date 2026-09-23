"""
model.py -- a permutation-invariant, size-agnostic network that predicts
Monod parameter matrices from a trajectory.

Per docs/curriculum_nn_spec.md's architecture discussion, this implements
direction (1), the recommended default: "a set/graph-based architecture
over strains and resources as nodes, with edges = matrix entries." A
strain's identity as "strain 3" vs "strain 1" is arbitrary labeling, not
a real feature -- the network must give the same answer under any
permutation of strain order (and separately, of metabolite order). That
is what makes ONE network able to train on Stage 0 (S=2, M=3) and be
evaluated on Stage 1 (S=4, M=6) with no architecture change: it operates
on a variable-size SET of strain nodes and a variable-size SET of
metabolite (and toxin) nodes, joined by a complete bipartite graph whose
edges correspond exactly to the r/k/c (and P/K_tox) matrix entries.

This is a from-scratch bipartite message-passing network (plain
torch.nn, no graph library dependency) rather than a full graph-neural-
network toolkit, since the graph here is always a complete bipartite
graph (every strain connects to every metabolite/toxin) -- there's no
sparse/irregular connectivity that would justify the extra dependency.

Batching: a batch must share one (S, M, T) tier (see curriculum.py's
bucketing) -- the model itself has no S/M/T baked into its weights, only
into the shape of whatever batch it's called with.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

N_SUMMARY_POINTS = 6  # fixed-size per-node trajectory summary, any n_points


def _summarize_log_trajectory(traj: np.ndarray, n_summary: int = N_SUMMARY_POINTS) -> np.ndarray:
    """traj: (n_points, n_species) linear-space values -> (n_species,
    n_summary) log-space values resampled to n_summary points evenly
    spaced in NORMALIZED time, so the feature size doesn't depend on how
    many timepoints a particular trajectory happened to be simulated
    with."""
    n_points = traj.shape[0]
    src_t = np.linspace(0.0, 1.0, n_points)
    dst_t = np.linspace(0.0, 1.0, n_summary)
    log_traj = np.log(np.maximum(traj, 1e-12))
    return np.stack([np.interp(dst_t, src_t, log_traj[:, i]) for i in range(traj.shape[1])], axis=0)


def featurize_sample(N: np.ndarray, x: np.ndarray, tox: np.ndarray):
    """N: (n_points, S), x: (n_points, M), tox: (n_points, T) -> per-node
    feature arrays (S, N_SUMMARY_POINTS), (M, N_SUMMARY_POINTS), (T, N_SUMMARY_POINTS)."""
    strain_feats = _summarize_log_trajectory(N)
    metab_feats = _summarize_log_trajectory(x)
    toxin_feats = _summarize_log_trajectory(tox) if tox.shape[1] else np.zeros((0, N_SUMMARY_POINTS))
    return strain_feats, metab_feats, toxin_feats


def _mlp(in_dim: int, out_dim: int, hidden: int = 64) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))


@dataclass
class PredictedParams:
    r: torch.Tensor           # (B, S, M)
    k: torch.Tensor            # (B, S, M)
    c: torch.Tensor            # (B, S, M)
    delta: torch.Tensor         # (B, S)
    m_supply: torch.Tensor       # (B, M)
    D_dilution: torch.Tensor      # (B, M)
    P: Optional[torch.Tensor] = None       # (B, S, T)
    K_tox: Optional[torch.Tensor] = None    # (B, S, T)


class MonodParameterGNN(nn.Module):
    """Bipartite (strain <-> metabolite [<-> toxin]) message-passing
    network. `embed_dim` and `n_rounds` are the only size-independent
    hyperparameters -- everything else scales automatically with however
    many strain/metabolite/toxin nodes a given batch has."""

    def __init__(self, embed_dim: int = 32, n_rounds: int = 3, hidden: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_rounds = n_rounds

        self.strain_encoder = _mlp(N_SUMMARY_POINTS, embed_dim, hidden)
        self.metab_encoder = _mlp(N_SUMMARY_POINTS, embed_dim, hidden)
        self.toxin_encoder = _mlp(N_SUMMARY_POINTS, embed_dim, hidden)

        # one shared edge network per bipartite relation, reused every round
        self.sm_edge = _mlp(2 * embed_dim, embed_dim, hidden)   # message metabolite -> strain
        self.ms_edge = _mlp(2 * embed_dim, embed_dim, hidden)   # message strain -> metabolite
        self.st_edge = _mlp(2 * embed_dim, embed_dim, hidden)   # message toxin -> strain
        self.ts_edge = _mlp(2 * embed_dim, embed_dim, hidden)   # message strain -> toxin

        self.strain_update = _mlp(2 * embed_dim, embed_dim, hidden)
        self.metab_update = _mlp(2 * embed_dim, embed_dim, hidden)
        self.toxin_update = _mlp(2 * embed_dim, embed_dim, hidden)

        self.rkc_head = _mlp(2 * embed_dim, 3, hidden)
        self.delta_head = _mlp(embed_dim, 1, hidden)
        self.resource_head = _mlp(embed_dim, 2, hidden)  # m_supply, D_dilution
        self.toxin_head = _mlp(2 * embed_dim, 2, hidden)  # P, K_tox

    @staticmethod
    def _pairwise(h_a: torch.Tensor, h_b: torch.Tensor) -> torch.Tensor:
        """h_a: (B, A, d), h_b: (B, B_, d) -> (B, A, B_, 2d) via broadcasting."""
        B, A, d = h_a.shape
        _, Bn, _ = h_b.shape
        a_exp = h_a.unsqueeze(2).expand(B, A, Bn, d)
        b_exp = h_b.unsqueeze(1).expand(B, A, Bn, d)
        return torch.cat([a_exp, b_exp], dim=-1)

    def forward(self, strain_feats: torch.Tensor, metab_feats: torch.Tensor,
                toxin_feats: Optional[torch.Tensor] = None) -> PredictedParams:
        """strain_feats: (B, S, N_SUMMARY_POINTS), metab_feats: (B, M,
        N_SUMMARY_POINTS), toxin_feats: (B, T, N_SUMMARY_POINTS) or None/T=0."""
        h_S = self.strain_encoder(strain_feats)
        h_M = self.metab_encoder(metab_feats)
        has_toxin = toxin_feats is not None and toxin_feats.shape[1] > 0
        h_T = self.toxin_encoder(toxin_feats) if has_toxin else None

        for _ in range(self.n_rounds):
            msg_to_S = self.sm_edge(self._pairwise(h_S, h_M)).mean(dim=2)   # (B, S, d)
            msg_to_M = self.ms_edge(self._pairwise(h_M, h_S)).mean(dim=2)   # (B, M, d)
            if has_toxin:
                msg_to_S = msg_to_S + self.st_edge(self._pairwise(h_S, h_T)).mean(dim=2)
                msg_to_T = self.ts_edge(self._pairwise(h_T, h_S)).mean(dim=2)
                h_T = h_T + self.toxin_update(torch.cat([h_T, msg_to_T], dim=-1))

            h_S = h_S + self.strain_update(torch.cat([h_S, msg_to_S], dim=-1))
            h_M = h_M + self.metab_update(torch.cat([h_M, msg_to_M], dim=-1))

        rkc = torch.nn.functional.softplus(self.rkc_head(self._pairwise(h_S, h_M)))  # (B, S, M, 3)
        r, k, c = rkc[..., 0], rkc[..., 1] + 1e-2, rkc[..., 2]

        delta = torch.nn.functional.softplus(self.delta_head(h_S)).squeeze(-1)  # (B, S)
        resource = torch.nn.functional.softplus(self.resource_head(h_M))          # (B, M, 2)
        m_supply, D_dilution = resource[..., 0], resource[..., 1]

        P = K_tox = None
        if has_toxin:
            tox_out = torch.nn.functional.softplus(self.toxin_head(self._pairwise(h_S, h_T)))  # (B, S, T, 2)
            P, K_tox = tox_out[..., 0], tox_out[..., 1] + 1e-2

        return PredictedParams(r=r, k=k, c=c, delta=delta, m_supply=m_supply,
                                D_dilution=D_dilution, P=P, K_tox=K_tox)
