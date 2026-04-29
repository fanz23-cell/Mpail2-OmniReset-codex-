#!/usr/bin/env python3
"""Create the UWLab OmniReset peg-in-hole environment and step it once."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the OmniReset peg-in-hole environment.")
    parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true", default=True)

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        import gymnasium as gym
        import torch

        import uwlab_tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg

        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        env = gym.make(args.task, cfg=env_cfg, render_mode=None)
        try:
            env.reset()
            action_space = getattr(env, "single_action_space", None) or env.action_space
            zeros = torch.zeros((args.num_envs, *action_space.shape), device=args.device, dtype=torch.float32)
            env.step(zeros)
            print("omnireset smoke test ok")
        finally:
            env.close()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
