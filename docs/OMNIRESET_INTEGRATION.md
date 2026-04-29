# OmniReset Integration Notes

This repo now includes a first-pass `mpail2` integration for the OmniReset peg-in-hole task from `UWLab`.

## What was added

- Registered `--env omnireset_peg` in `mpail2.train`.
- Added a state-based learner preset for OmniReset tasks.
- Added module auto-import support so `mpail2` can load external Isaac task packages such as `uwlab_tasks`.
- Added `scripts/convert_omnireset_demo.py` to convert OmniReset demonstration datasets into the MPAIL `.pt` transition format.
- Added `scripts/check_omnireset_readiness.py` plus conda env templates under `envs/` for faster host setup.

## Recommended environment layout

Do **not** try to share one Python environment across everything on day one.

- `mpail2-core`: Python 3.10 for the base repo, Gym, MuJoCo, and real-robot utilities.
- `mpail2-omnireset`: Python 3.11 for Isaac Sim 5.1 + `UWLab`, then install `mpail2` in editable mode there too.

Why:

- `mpail2` currently declares `>=3.10,<3.11`.
- `UWLab` documents Python 3.11 for Isaac Sim 5.1.
- `WheeledLab-research` is older and tied to a different Isaac stack.

## Suggested install order for OmniReset work

1. Install NVIDIA driver and confirm `nvidia-smi` works.
2. Create a fresh Python 3.11 / Isaac Sim 5.1 environment for `UWLab`.
3. Install `UWLab` editable packages into that environment.
4. Install this `mpail2` repo editable into the same OmniReset environment.
5. Collect or download OmniReset demos.
6. Convert demos:

```bash
python scripts/convert_omnireset_demo.py \
    /path/to/omnireset_demo.pt \
    mpail2/train/isaac_franka/demos/omnireset_peg_state.pt
```

7. Train:

```bash
python -m mpail2.train.train --env omnireset_peg --headless log.no_wandb=True
```

Use `runner.path_to_demonstrations=/abs/path/to/demo.pt` if you want the demo file somewhere else.

## Quick host sanity check

```bash
python scripts/check_omnireset_readiness.py
```

## Migration helpers

- `scripts/bootstrap_lab_omnireset.sh`: one-command setup for a fresh lab machine
- `scripts/omnireset_env_smoke.py`: instantiate the OmniReset peg-in-hole env once
- `scripts/run_lab_omnireset.sh`: launch the `mpail2` OmniReset training entry with low-memory defaults
