"""Learner preset for OmniReset state-based manipulation tasks."""

from __future__ import annotations

from dataclasses import dataclass, field

from mpail2.configs.cfgs import ObsNormalizerCfg
import mpail2.configs.defs as defs


@dataclass
class OmniResetStatePlannerConfig(defs.PlannerConfig):
    action_dim: int = 7
    seed: int = 0
    encoder_cfg: defs.MLPCoderConfig = field(
        default_factory=lambda: defs.MLPCoderConfig(
            obs_key="policy",
            input_dim=135,
        )
    )


@dataclass
class OmniResetStateLearnerConfig(defs.LearnerConfig):
    """State-based preset for OmniReset manager-based manipulation envs."""

    replay_size: int = 100_000
    replay_batch_size: int = 512
    use_terminations: bool = False
    obs_normalizer_cfg: ObsNormalizerCfg = field(
        default_factory=lambda: ObsNormalizerCfg(normalization_type="fixed")
    )
    planner_cfg: defs.PlannerConfig = field(default_factory=OmniResetStatePlannerConfig)
