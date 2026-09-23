"""
train.py -- CLI entry point for the curriculum-trained Monod
parameter-fitting network.

    python -m curriculum_nn.src.train --stages stage0 stage1 --n-per-tier 40

Runs each requested stage IN ORDER on one model/optimizer (the
curriculum), then reports parameter-recovery and trajectory-
reconstruction error (eval.py) on each stage's held-out test split.

Also runs the actual curriculum-transfer check the spec asks for: before
touching Stage 1 data at all, it evaluates the Stage-0-only model
zero-shot on a Stage-1-sized tier it has never seen, then re-evaluates
that SAME tier after Stage 1 training -- this is the concrete "did
training on small systems help with a larger one" measurement, not just
an assertion that it should.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from .curriculum import train_curriculum
from .data_gen import STAGES, generate_dataset, split_by_tier
from .eval import evaluate_dataset, evaluate_tier
from .model import MonodParameterGNN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="+", default=["stage0", "stage1"], choices=list(STAGES.keys()))
    parser.add_argument("--n-per-tier", type=int, default=40)
    parser.add_argument("--epochs-per-stage", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--n-rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None, help="path to save trained weights (.pt)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    model = MonodParameterGNN(embed_dim=args.embed_dim, n_rounds=args.n_rounds)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    stage_specs = [STAGES[name] for name in args.stages]
    stage_datasets, test_datasets = [], {}
    for stage in stage_specs:
        dataset = generate_dataset(stage, args.n_per_tier, rng)
        train_data, val_data, test_data = split_by_tier(dataset, rng=rng)
        stage_datasets.append((stage, train_data, val_data))
        test_datasets[stage.name] = test_data

    # zero-shot check: if a later stage has a tier the FIRST stage never
    # trained on, measure the untrained-on-it error before that stage runs
    zero_shot_tier, zero_shot_samples, zero_shot_before = None, None, None
    if len(stage_specs) > 1:
        first_tiers = set(stage_specs[0].tiers)
        for tier in stage_specs[1].tiers:
            if tier not in first_tiers:
                zero_shot_tier = tier
                zero_shot_samples = test_datasets[stage_specs[1].name].get(tier) or \
                    generate_dataset(stage_specs[1], 10, rng)[tier]
                break

    for i, (stage, train_data, val_data) in enumerate(stage_datasets):
        print(f"=== Training {stage.name} (tiers: {stage.tiers}) ===")
        if i == 1 and zero_shot_tier is not None:
            zero_shot_before = evaluate_tier(model, zero_shot_samples)
            print(f"  [before {stage.name}] zero-shot on unseen tier {zero_shot_tier}: {zero_shot_before}")

        history = train_curriculum(model, optimizer, [(stage, train_data, val_data)], args.epochs_per_stage)
        for epoch, train_loss, val_loss in history[stage.name]:
            print(f"  epoch {epoch:4d}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        print(f"=== Evaluating {stage.name} on its held-out test split ===")
        for tier, metrics in evaluate_dataset(model, test_datasets[stage.name]).items():
            print(f"  tier {tier}: {metrics}")

    if zero_shot_tier is not None:
        zero_shot_after = evaluate_tier(model, zero_shot_samples)
        print(f"\n=== Curriculum-transfer check on tier {zero_shot_tier} ===")
        print(f"  before training on the stage containing it: {zero_shot_before}")
        print(f"  after:                                      {zero_shot_after}")

    if args.out:
        torch.save(model.state_dict(), args.out)
        print(f"saved model weights to {args.out}")


if __name__ == "__main__":
    main()
