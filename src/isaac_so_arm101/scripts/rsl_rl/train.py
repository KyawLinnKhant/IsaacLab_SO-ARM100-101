# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import isaac_so_arm101.scripts.rsl_rl.cli_args as cli_args # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
parsed_args, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if parsed_args.video:
    parsed_args.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
sim_launcher = AppLauncher(parsed_args)
omni_app = sim_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
rslrl_installed_ver = metadata.version("rsl-rl-lib")
if version.parse(rslrl_installed_ver) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{rslrl_installed_ver}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

import omni
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import isaac_so_arm101.tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(parsed_args.task, parsed_args.agent)
def main(environment_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, policy_runner_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    policy_runner_cfg = cli_args.update_rsl_rl_cfg(policy_runner_cfg, parsed_args)
    environment_cfg.scene.num_envs = parsed_args.num_envs if parsed_args.num_envs is not None else environment_cfg.scene.num_envs
    policy_runner_cfg.max_iterations = (
        parsed_args.max_iterations if parsed_args.max_iterations is not None else policy_runner_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    environment_cfg.seed = policy_runner_cfg.seed
    environment_cfg.sim.device = parsed_args.device if parsed_args.device is not None else environment_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if parsed_args.distributed and parsed_args.device is not None and "cpu" in parsed_args.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if parsed_args.distributed:
        environment_cfg.sim.device = f"cuda:{sim_launcher.local_rank}"
        policy_runner_cfg.device = f"cuda:{sim_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = policy_runner_cfg.seed + sim_launcher.local_rank
        environment_cfg.seed = seed
        policy_runner_cfg.seed = seed

    # specify directory for logging experiments
    experiment_log_dir = os.path.join("logs", "rsl_rl", policy_runner_cfg.experiment_name)
    experiment_log_dir = os.path.abspath(experiment_log_dir)
    print(f"[INFO] Logging experiment in directory: {experiment_log_dir}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    run_output_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from pose_command line: {run_output_dir}")
    if policy_runner_cfg.run_name:
        run_output_dir += f"_{policy_runner_cfg.run_name}"
    run_output_dir = os.path.join(experiment_log_dir, run_output_dir)

    # set the IO descriptors output directory if requested
    if isinstance(environment_cfg, ManagerBasedRLEnvCfg):
        environment_cfg.export_io_descriptors = parsed_args.export_io_descriptors
        environment_cfg.io_descriptors_output_dir = run_output_dir
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    environment_cfg.run_output_dir = run_output_dir

    # create isaac environment
    env = gym.make(parsed_args.task, cfg=environment_cfg, render_mode="rgb_array" if parsed_args.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new run_output_dir
    if policy_runner_cfg.resume or policy_runner_cfg.algorithm.class_name == "Distillation":
        checkpoint_path = get_checkpoint_path(experiment_log_dir, policy_runner_cfg.load_run, policy_runner_cfg.load_checkpoint)

    # wrap for video recording
    if parsed_args.video:
        recording_options = {
            "video_folder": os.path.join(run_output_dir, "videos", "train"),
            "step_trigger": lambda step: step % parsed_args.video_interval == 0,
            "video_length": parsed_args.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(recording_options, nesting=4)
        env = gym.wrappers.RecordVideo(env, **recording_options)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=policy_runner_cfg.clip_actions)

    # create policy_runner from rsl-rl
    if policy_runner_cfg.class_name == "OnPolicyRunner":
        policy_runner = OnPolicyRunner(env, policy_runner_cfg.to_dict(), run_output_dir=run_output_dir, device=policy_runner_cfg.device)
    elif policy_runner_cfg.class_name == "DistillationRunner":
        policy_runner = DistillationRunner(env, policy_runner_cfg.to_dict(), run_output_dir=run_output_dir, device=policy_runner_cfg.device)
    else:
        raise ValueError(f"Unsupported policy_runner class: {policy_runner_cfg.class_name}")
    # write git state to logs
    policy_runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if policy_runner_cfg.resume or policy_runner_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {checkpoint_path}")
        # load previously trained model
        policy_runner.load(checkpoint_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(run_output_dir, "params", "env.yaml"), environment_cfg)
    dump_yaml(os.path.join(run_output_dir, "params", "agent.yaml"), policy_runner_cfg)

    # run training
    policy_runner.learn(num_learning_iterations=policy_runner_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    omni_app.close()