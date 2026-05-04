#!/usr/bin/env python3
"""Watch expert policy in Isaac Sim and record a video.

Usage:
    conda run -n mpail2-omnireset python scripts/watch_expert.py \
        --checkpoint mpail2/train/isaac_franka/demos/peg_expert_seed42.pt \
        --output /tmp/expert_demo.mp4
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, default=Path("/tmp/expert_demo.mp4"))
parser.add_argument("--num-steps", type=int, default=320)
parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch, gymnasium as gym, imageio
import torch.nn as nn
from isaaclab_tasks.utils import parse_env_cfg
import isaaclab_tasks, uwlab_tasks  # noqa

class _ObsNormalizer(nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        self.register_buffer("_mean", state_dict["actor_obs_normalizer._mean"])
        self.register_buffer("_std",  state_dict["actor_obs_normalizer._std"])
    def forward(self, x):
        return (x - self._mean) / (self._std + 1e-8)

class _Actor(nn.Module):
    def __init__(self, state_dict, device):
        super().__init__()
        self.norm = _ObsNormalizer(state_dict)
        weight_keys = sorted([k for k in state_dict if k.startswith("actor.") and k.endswith(".weight")],
                             key=lambda k: int(k.split(".")[1]))
        layers = []
        for i, wk in enumerate(weight_keys):
            o, inp = state_dict[wk].shape
            layers.append(nn.Linear(inp, o))
            if i < len(weight_keys) - 1:
                layers.append(nn.ELU())
        self.actor = nn.Sequential(*layers)
        self.actor.load_state_dict({k[len("actor."):]: v for k, v in state_dict.items() if k.startswith("actor.")})
        self.to(device)

    @torch.inference_mode()
    def forward(self, obs):
        return self.actor(self.norm(obs))

def main():
    device = args_cli.device or "cuda"
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    ckpt = torch.load(str(args_cli.checkpoint), map_location=device, weights_only=False)
    policy = _Actor(ckpt["model_state_dict"], device)
    policy.eval()
    print(f"[INFO] Loaded expert checkpoint iter={ckpt.get('iter','?')}")

    frames = []
    obs, _ = env.reset()
    obs_t = obs["policy"] if isinstance(obs, dict) else obs

    for step in range(args_cli.num_steps):
        actions = policy(obs_t)
        obs, _, terms, truncs, _ = env.step(actions)
        obs_t = obs["policy"] if isinstance(obs, dict) else obs

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        if (terms | truncs).any():
            print(f"  Episode ended at step {step}")
            obs, _ = env.reset()
            obs_t = obs["policy"] if isinstance(obs, dict) else obs

    env.close()
    if frames:
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(args_cli.output), frames, fps=10)
        print(f"Video saved → {args_cli.output}  ({len(frames)} frames)")
    else:
        print("[WARN] No frames captured")

if __name__ == "__main__":
    main()
    simulation_app.close()
