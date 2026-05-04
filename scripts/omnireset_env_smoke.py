#!/usr/bin/env python3
"""Create the UWLab OmniReset peg-in-hole environment and step it once."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the OmniReset peg-in-hole environment.")
    parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0")
    parser.add_argument("--num-envs", type=int, default=1)

    from isaaclab.app import AppLauncher

    # AppLauncher owns --headless and --device; do not add them manually
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import traceback
    _success = False
    try:
        import gymnasium as gym
        import torch

        import uwlab_tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg

        sim_device = args.device  # set by AppLauncher
        env_cfg = parse_env_cfg(args.task, device=sim_device, num_envs=args.num_envs)
        env = gym.make(args.task, cfg=env_cfg, render_mode=None)
        try:
            env.reset()
            action_space = getattr(env, "single_action_space", None) or env.action_space
            # action_space.shape may be () for Box spaces; use total_action_dim if available
            act_dim = getattr(env.unwrapped, "total_action_dim", None)
            if act_dim is None:
                act_dim = action_space.shape[-1] if action_space.shape else 1
            zeros = torch.zeros((args.num_envs, act_dim), device=sim_device, dtype=torch.float32)
            env.step(zeros)
            print("omnireset smoke test ok")
            _success = True
        finally:
            env.close()
    except Exception:
        traceback.print_exc()
    finally:
        simulation_app.close()
    if not _success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
