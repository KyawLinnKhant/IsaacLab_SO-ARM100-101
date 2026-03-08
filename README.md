# IsaacLab SO-ARM100 / SO-ARM101

[![Isaac Sim](https://img.shields.io/badge/IsaacSim-5.1.0-76B900.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.0-8A2BE2.svg)](https://isaac-sim.github.io/IsaacLab/main/index.html)
[![Python](https://img.shields.io/badge/python-3.11-3776AB.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

Reinforcement learning environments for the SO-ARM100 and SO-ARM101 robot arms, built on top of NVIDIA Isaac Lab.

![rl-video-step-0](https://github.com/user-attachments/assets/890e3a9d-5cbd-46a5-9317-37d0f2511684)

---

## What's inside

Two task types, each supporting both robot variants:

| Task | SO-ARM100 | SO-ARM101 |
|------|-----------|-----------|
| Reach (end-effector pose tracking) | ✅ | ✅ |
| Lift (pick up a cube) | ✅ | ✅ |

Training is done via PPO using RSL-RL. Policies are exported as `.pt` and `.onnx` after evaluation.

---

## Setup

Install [uv](https://github.com/astral-sh/uv) first:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone and install:
```bash
git clone https://github.com/KyawLinnKhant/IsaacLab_SO-ARM100-101.git
cd IsaacLab_SO-ARM100-101
uv sync
```

---

## Usage

### List available environments
```bash
uv run list_envs
```

### Test with dummy agents
```bash
# Zero actions — robot stays still
uv run zero_agent --task SO-ARM100-Reach-Play-v0

# Random actions — chaotic baseline
uv run random_agent --task SO-ARM100-Reach-Play-v0
```

### Train
```bash
# Reach task
uv run train --task SO-ARM100-Reach-v0 --headless

# Lift task
uv run train --task SO-ARM100-Lift-Cube-v0 --headless
```

### Evaluate
```bash
# Reach
uv run play --task SO-ARM100-Reach-Play-v0

# Lift
uv run play --task SO-ARM100-Lift-Cube-Play-v0
```

---

## Project Structure
```
src/isaac_so_arm101/
├── robots/
│   ├── trs_so100/          ← SO-ARM100 articulation config + URDF
│   └── trs_so101/          ← SO-ARM101 articulation config + URDF
├── tasks/
│   ├── reach/              ← Reach task: env, MDP, PPO config
│   └── lift/               ← Lift task: env, MDP, PPO config
└── scripts/
    ├── rsl_rl/             ← train.py, play.py, cli_args.py
    ├── list_envs.py
    ├── zero_agent.py
    └── random_agent.py
```

---

## Built with

- **[Isaac Lab](https://isaac-sim.github.io/IsaacLab/)** — simulation and RL framework
- **[NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim)** — physics backend
- **[RSL-RL](https://github.com/leggedrobotics/rsl_rl)** — PPO training library
- **[SO-ARM100/SO-ARM101](https://github.com/TheRobotStudio/SO-ARM100)** — robot hardware platform

---

## License

BSD-3-Clause © 2026 Kyaw Linn Khant — see [LICENSE](LICENSE)
