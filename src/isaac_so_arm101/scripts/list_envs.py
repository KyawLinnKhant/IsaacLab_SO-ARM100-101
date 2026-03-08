# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to print all the available environments in Isaac Lab.

The script iterates over all registered environments and stores the details in a env_table.
It prints the name of the environment, the entry point and the config file.

All the environments are registered in the `isaaclab_tasks` extension. They start
with `Isaac` in their name.
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
sim_launcher = AppLauncher(headless=True)
omni_app = sim_launcher.app


"""Rest everything follows."""

import gymnasium as gym
from prettytable import PrettyTable

import isaac_so_arm101.tasks  # noqa: F401


def main():
    """Print all environments registered in `isaaclab_tasks` extension."""
    # print all the available environments
    env_table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    env_table.title = "Available Environments in Isaac Lab"
    # set alignment of env_table columns
    env_table.align["Task Name"] = "l"
    env_table.align["Entry Point"] = "l"
    env_table.align["Config"] = "l"

    # count of environments
    row_idx = 0
    # acquire all Isaac environments names
    for gym_entry in gym.registry.values():
        if "SO-ARM" in gym_entry.id:
            # add details to env_table
            env_table.add_row([row_idx + 1, gym_entry.id, gym_entry.entry_point, gym_entry.kwargs["env_cfg_entry_point"]])
            # increment count
            row_idx += 1

    print(env_table)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    omni_app.close()