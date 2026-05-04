#!/usr/bin/env python3
"""Filter near-zero (idle) transitions from OmniReset demo data.

After peg insertion succeeds the expert policy holds still until the episode
ends, contaminating ~44% of transitions with near-zero obs change.  The
discriminator in MPAIL2 then learns that "standing still" is expert behaviour,
giving the agent a free high reward for doing nothing.

Usage:
    python scripts/filter_idle_demos.py \
        --input  mpail2/train/isaac_franka/demos/omnireset_peg_state.pt \
        --output mpail2/train/isaac_franka/demos/omnireset_peg_state_filtered.pt \
        --thresh 0.005
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  type=Path,  required=True)
    parser.add_argument("--output", type=Path,  required=True)
    parser.add_argument(
        "--thresh", type=float, default=0.005,
        help="L2 norm threshold below which (obs→next_obs) is considered idle (default 0.005)",
    )
    args = parser.parse_args()

    data = torch.load(str(args.input), map_location="cpu", weights_only=False)
    demos = data["demonstrations"]
    metadata = data.get("metadata", {})

    # All obs groups should have the same N — filter on the primary "policy" key
    primary = "policy"
    d = demos[primary]          # (N, 2, obs_dim)
    diffs = (d[:, 1] - d[:, 0]).norm(dim=-1)   # (N,)

    keep = diffs >= args.thresh
    n_before, n_after = len(d), keep.sum().item()
    removed = n_before - n_after
    pct = removed / n_before * 100

    print(f"Transitions before : {n_before}")
    print(f"Threshold          : {args.thresh}")
    print(f"Removed (idle)     : {removed}  ({pct:.1f}%)")
    print(f"Transitions kept   : {n_after}")

    filtered_demos = {k: v[keep] for k, v in demos.items()}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"demonstrations": filtered_demos, "metadata": metadata}, str(args.output))
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
