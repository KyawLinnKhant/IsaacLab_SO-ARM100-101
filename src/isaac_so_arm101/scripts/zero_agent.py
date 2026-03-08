# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
parsed_args = parser.parse_args()

# launch omniverse app
sim_launcher = AppLauncher(parsed_args)
omni_app = sim_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaac_so_arm101.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    """Zero joint_actions agent with Isaac Lab environment."""
    # parse configuration
    environment_cfg = parse_env_cfg(
        parsed_args.task, device=parsed_args.device, num_envs=parsed_args.num_envs, use_fabric=not parsed_args.disable_fabric
    )
    # create environment
    env = gym.make(parsed_args.task, cfg=environment_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    # simulate environment
    while omni_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero joint_actions
            joint_actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            # apply joint_actions
            env.step(joint_actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    omni_app.close()