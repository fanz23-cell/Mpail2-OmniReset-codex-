#!/usr/bin/env python3
"""Convert OmniReset demonstrations into the MPAIL transition format."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mpail2.envs.gym_mujoco.utils.data_utils import save_demonstrations


def _to_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return torch.as_tensor(np.asarray(value))


def _load_pt(path: Path) -> dict[str, Any]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a dictionary in {path}, got {type(data)}")
    return data


def _load_zarr(path: Path) -> dict[str, Any]:
    try:
        import zarr
    except ImportError as exc:
        raise ImportError("Reading .zarr demos requires `pip install zarr`.") from exc

    root = zarr.open(str(path), mode="r")
    return root


def _read_path(data: Any, dotted_path: str) -> Any:
    current = data
    for part in dotted_path.split("."):
        if current is None:
            break
        try:
            current = current[part]
        except Exception as exc:
            raise KeyError(f"Could not resolve path '{dotted_path}'") from exc
    return current


def _extract_transition_obs(data: Any, obs_path: str, next_obs_path: str | None) -> dict[str, torch.Tensor]:
    obs = _read_path(data, obs_path)
    if next_obs_path:
        next_obs = _read_path(data, next_obs_path)
    else:
        obs_tensor = _to_tensor(obs)
        if obs_tensor.shape[0] < 2:
            raise ValueError("Need at least 2 timesteps to build transitions.")
        return {"policy": torch.stack([obs_tensor[:-1], obs_tensor[1:]], dim=1).float()}

    obs_tensor = _to_tensor(obs)
    next_obs_tensor = _to_tensor(next_obs)
    if obs_tensor.shape != next_obs_tensor.shape:
        raise ValueError(
            f"obs and next_obs must have the same shape, got {tuple(obs_tensor.shape)} vs {tuple(next_obs_tensor.shape)}"
        )
    return {"policy": torch.stack([obs_tensor, next_obs_tensor], dim=1).float()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert OmniReset demos to MPAIL format.")
    parser.add_argument("input", type=Path, help="Input .pt or .zarr demo file")
    parser.add_argument("output", type=Path, help="Output .pt path for MPAIL")
    parser.add_argument(
        "--obs-path",
        default="data.obs.policy",
        help="Path to the expert observation array inside the source file",
    )
    parser.add_argument(
        "--next-obs-path",
        default="data.next_obs.policy",
        help="Path to the next-observation array; leave empty to infer with a 1-step shift from --obs-path",
    )
    args = parser.parse_args()

    loader = _load_zarr if args.input.suffix == ".zarr" else _load_pt
    data = loader(args.input)
    next_obs_path = args.next_obs_path or None
    demonstrations = _extract_transition_obs(data, args.obs_path, next_obs_path)

    metadata = {
        "source": str(args.input),
        "obs_path": args.obs_path,
        "next_obs_path": next_obs_path,
        "format": "mpail_transition_dict",
        "obs_key": "policy",
    }
    save_demonstrations(demonstrations, str(args.output), metadata=metadata)


if __name__ == "__main__":
    main()
