# IsaacLab SO-ARM100 / SO-ARM101

> PPO-trained manipulation policies for the SO-ARM100 and SO-ARM101 robot arms — Reach and Lift tasks, trained in Isaac Lab, exported as PyTorch JIT and ONNX.

[![Isaac Sim](https://img.shields.io/badge/IsaacSim-5.1.0-76B900.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.0-8A2BE2.svg)](https://isaac-sim.github.io/IsaacLab/main/index.html)
[![Python](https://img.shields.io/badge/python-3.11-3776AB.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

---

![rl-video-step-0](https://github.com/user-attachments/assets/890e3a9d-5cbd-46a5-9317-37d0f2511684)

---

## Tasks

| Task | SO-ARM100 | SO-ARM101 | Description |
|------|-----------|-----------|-------------|
| Reach | ✅ | ✅ | End-effector tracks a target pose in 3D space |
| Lift | ✅ | ✅ | Pick up a cube from the table |

Both robots, both tasks — 4 independently trained policies total.  
Policies are exported as `.pt` (PyTorch JIT) and `.onnx` after evaluation.

---

## Training details

| Property | Value |
|----------|-------|
| Algorithm | PPO via RSL-RL |
| Observation space | Joint positions, velocities, end-effector pose, target pose |
| Action space | Joint position targets (6 DOF) |
| Physics backend | Isaac Sim GPU pipeline |
| Export formats | `.pt` (JIT) + `.onnx` |

---

## Quick start

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/KyawLinnKhant/IsaacLab_SO-ARM100-101.git
cd IsaacLab_SO-ARM100-101
uv sync

# List all available environments
uv run list_envs

# Train — Reach
uv run train --task SO-ARM100-Reach-v0 --headless

# Train — Lift
uv run train --task SO-ARM100-Lift-Cube-v0 --headless

# Evaluate (with rendering)
uv run play --task SO-ARM100-Reach-Play-v0
uv run play --task SO-ARM100-Lift-Cube-Play-v0

# Baselines (sanity check)
uv run zero_agent --task SO-ARM100-Reach-Play-v0   # stays still
uv run random_agent --task SO-ARM100-Reach-Play-v0 # chaotic
```

---

## Project structure

```
src/isaac_so_arm101/
├── robots/
│   ├── trs_so100/     SO-ARM100 articulation config + USD/URDF
│   └── trs_so101/     SO-ARM101 articulation config + USD/URDF
├── tasks/
│   ├── reach/         Env definition, MDP terms, PPO config
│   └── lift/          Env definition, MDP terms, PPO config
└── scripts/
    ├── rsl_rl/        train.py · play.py · cli_args.py
    ├── list_envs.py
    ├── zero_agent.py
    └── random_agent.py
```

---

## Built with

- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) — simulation and RL framework
- [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim) — physics backend
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl) — PPO training library
- [SO-ARM100/SO-ARM101](https://github.com/TheRobotStudio/SO-ARM100) — robot hardware

---

## Author

**Kyaw Linn Khant** — Robotics & AI Engineer  
[Portfolio](https://kyawlinnkhant.github.io/my_portfolio/) · [LinkedIn](https://linkedin.com/in/kyawlinnkhant)
