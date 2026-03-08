# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

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
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
parsed_args, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if parsed_args.video:
    parsed_args.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
sim_launcher = AppLauncher(parsed_args)
omni_app = sim_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
import isaac_so_arm101.tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(parsed_args.task, parsed_args.agent)
def main(environment_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, policy_runner_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    env_task_name = parsed_args.task.split(":")[-1]
    train_env_name = env_task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    policy_runner_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(policy_runner_cfg, parsed_args)
    environment_cfg.scene.num_envs = parsed_args.num_envs if parsed_args.num_envs is not None else environment_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    environment_cfg.seed = policy_runner_cfg.seed
    environment_cfg.sim.device = parsed_args.device if parsed_args.device is not None else environment_cfg.sim.device

    # specify directory for logging experiments
    experiment_log_dir = os.path.join("logs", "rsl_rl", policy_runner_cfg.experiment_name)
    experiment_log_dir = os.path.abspath(experiment_log_dir)
    print(f"[INFO] Loading experiment from directory: {experiment_log_dir}")
    if parsed_args.use_pretrained_checkpoint:
        checkpoint_path = get_published_pretrained_checkpoint("rsl_rl", train_env_name)
        if not checkpoint_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif parsed_args.checkpoint:
        checkpoint_path = retrieve_file_path(parsed_args.checkpoint)
    else:
        checkpoint_path = get_checkpoint_path(experiment_log_dir, policy_runner_cfg.load_run, policy_runner_cfg.load_checkpoint)

    run_output_dir = os.path.dirname(checkpoint_path)

    # set the log directory for the environment (works for all environment types)
    environment_cfg.run_output_dir = run_output_dir

    # create isaac environment
    env = gym.make(parsed_args.task, cfg=environment_cfg, render_mode="rgb_array" if parsed_args.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if parsed_args.video:
        recording_options = {
            "video_folder": os.path.join(run_output_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": parsed_args.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(recording_options, nesting=4)
        env = gym.wrappers.RecordVideo(env, **recording_options)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=policy_runner_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {checkpoint_path}")
    # load previously trained model
    if policy_runner_cfg.class_name == "OnPolicyRunner":
        policy_runner = OnPolicyRunner(env, policy_runner_cfg.to_dict(), run_output_dir=None, device=policy_runner_cfg.device)
    elif policy_runner_cfg.class_name == "DistillationRunner":
        policy_runner = DistillationRunner(env, policy_runner_cfg.to_dict(), run_output_dir=None, device=policy_runner_cfg.device)
    else:
        raise ValueError(f"Unsupported policy_runner class: {policy_runner_cfg.class_name}")
    policy_runner.load(checkpoint_path)

    # obtain the trained inference_policy for inference
    inference_policy = policy_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_network = policy_runner.alg.inference_policy
    except AttributeError:
        # version 2.2 and below
        policy_network = policy_runner.alg.actor_critic

    # extract the obs_normalizer
    if hasattr(policy_network, "actor_obs_normalizer"):
        obs_normalizer = policy_network.actor_obs_normalizer
    elif hasattr(policy_network, "student_obs_normalizer"):
        obs_normalizer = policy_network.student_obs_normalizer
    else:
        obs_normalizer = None

    # export inference_policy to onnx/jit
    exported_policy_dir = os.path.join(os.path.dirname(checkpoint_path), "exported")
    export_policy_as_jit(policy_network, obs_normalizer=obs_normalizer, path=exported_policy_dir, filename="inference_policy.pt")
    export_policy_as_onnx(policy_network, obs_normalizer=obs_normalizer, path=exported_policy_dir, filename="inference_policy.onnx")

    sim_step_dt = env.unwrapped.step_dt

    # reset environment
    current_obs = env.get_observations()
    step_counter = 0
    # simulate environment
    while omni_app.is_running():
        loop_start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            joint_actions = inference_policy(current_obs)
            # env stepping
            current_obs, _, _, _ = env.step(joint_actions)
        if parsed_args.video:
            step_counter += 1
            # Exit the play loop after recording one video
            if step_counter == parsed_args.video_length:
                break

        # time delay for real-time evaluation
        remaining_dt = sim_step_dt - (time.time() - loop_start_time)
        if parsed_args.real_time and remaining_dt > 0:
            time.sleep(remaining_dt)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    omni_app.close()