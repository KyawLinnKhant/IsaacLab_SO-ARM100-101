# Copyright (c) 2026, Kyaw Linn Khant
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab.utils import configclass
from isaac_so_arm101.robots import ARM100_ARTICULATION_CFG, ARM101_ARTICULATION_CFG  # noqa: F401
from isaac_so_arm101.tasks.reach.reach_env_cfg import ArmReachEnvCfg

##
# Scene definition
##


@configclass
class Arm100ReachEnvCfg(ArmReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to franka
        self.scene.robot = ARM100_ARTICULATION_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["gripper"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["gripper"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["gripper"]

        # TODO: reorient pose_command target

        # override joint_actions
        self.joint_actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            scale=0.5,
            use_default_offset=True,
        )
        # override pose_command generator body
        # end-effector is along z-direction
        self.commands.ee_pose.body_name = ["gripper"]
        # self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)


@configclass
class Arm100ReachPlayEnvCfg(Arm100ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.inference_policy.enable_corruption = False


@configclass
class Arm101ReachEnvCfg(ArmReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to franka
        self.scene.robot = ARM101_ARTICULATION_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["gripper_link"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["gripper_link"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["gripper_link"]

        self.rewards.end_effector_orientation_tracking.weight = 0.0

        # override joint_actions
        self.joint_actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=0.5,
            use_default_offset=True,
        )
        # override pose_command generator body
        # end-effector is along z-direction
        self.commands.ee_pose.body_name = ["gripper_link"]
        # self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)


@configclass
class Arm101ReachPlayEnvCfg(Arm101ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.inference_policy.enable_corruption = False
