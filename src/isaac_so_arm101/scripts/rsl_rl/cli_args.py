# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """Add RSL-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    rsl_argument_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    # -- experiment arguments
    rsl_argument_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    rsl_argument_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load arguments
    rsl_argument_group.add_argument("--resume", action="store_true", default=False, help="Whether to resume from a checkpoint.")
    rsl_argument_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    rsl_argument_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    # -- logger arguments
    rsl_argument_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    rsl_argument_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )


def parse_rsl_rl_cfg(env_task_name: str, parsed_args: argparse.Namespace) -> RslRlBaseRunnerCfg:
    """Parse configuration for RSL-RL agent based on inputs.

    Args:
        env_task_name: The name of the environment.
        parsed_args: The pose_command line arguments.

    Returns:
        The parsed configuration for RSL-RL agent based on inputs.
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    # load the default configuration
    runner_cfg: RslRlBaseRunnerCfg = load_cfg_from_registry(env_task_name, "rsl_rl_cfg_entry_point")
    runner_cfg = update_rsl_rl_cfg(runner_cfg, parsed_args)
    return runner_cfg


def update_rsl_rl_cfg(policy_runner_cfg: RslRlBaseRunnerCfg, parsed_args: argparse.Namespace):
    """Update configuration for RSL-RL agent based on inputs.

    Args:
        policy_runner_cfg: The configuration for RSL-RL agent.
        parsed_args: The pose_command line arguments.

    Returns:
        The updated configuration for RSL-RL agent based on inputs.
    """
    # override the default configuration with CLI arguments
    if hasattr(parsed_args, "seed") and parsed_args.seed is not None:
        # randomly sample a seed if seed = -1
        if parsed_args.seed == -1:
            parsed_args.seed = random.randint(0, 10000)
        policy_runner_cfg.seed = parsed_args.seed
    if parsed_args.resume is not None:
        policy_runner_cfg.resume = parsed_args.resume
    if parsed_args.load_run is not None:
        policy_runner_cfg.load_run = parsed_args.load_run
    if parsed_args.checkpoint is not None:
        policy_runner_cfg.load_checkpoint = parsed_args.checkpoint
    if parsed_args.run_name is not None:
        policy_runner_cfg.run_name = parsed_args.run_name
    if parsed_args.logger is not None:
        policy_runner_cfg.logger = parsed_args.logger
    # set the project name for wandb and neptune
    if policy_runner_cfg.logger in {"wandb", "neptune"} and parsed_args.log_project_name:
        policy_runner_cfg.wandb_project = parsed_args.log_project_name
        policy_runner_cfg.neptune_project = parsed_args.log_project_name

    return policy_runner_cfg