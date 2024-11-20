# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Universal Robots.

The following configuration parameters are available:

* :obj:`UR10_CFG`: The UR10 arm without a gripper.

Reference: https://github.com/ros-industrial/universal_robot
"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##


UR10_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/UniversalRobots/UR10/ur10_instanceable.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.712,
            "elbow_joint": 1.712,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
        },
        pos = (-0.2,0.0,1.0)
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
    },
)
"""Configuration of UR-10 arm using implicit actuator models."""


UR5_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="usd/ur5_with_gripper.usd",
        # usd_path = "/home/jm/workspace/ur5.usd",
        # usd_path = "usd/ur5_with_gripper_joint_disabled.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled = True,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.65,
            "elbow_joint": 1.5,
            "wrist_1_joint": -1.712,
            "wrist_2_joint": -1.712,
            "wrist_3_joint": 0.0,
            "left_inner_knuckle_joint" : 0.00033,
            "right_inner_knuckle_joint" : -0.00031,
            "left_inner_finger_joint" : 0.00051,
            "right_inner_finger_joint" : -0.00015,
            "right_outer_knuckle_joint" : -0.00034,
            "finger_joint" : 0.0
        },
        pos = (-0.2,0.0,1.0)
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_.*_joint"],
            velocity_limit=10.0,
            effort_limit=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
        "finger": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint", "right_outer_knuckle_joint"],
            velocity_limit=0.2,
            effort_limit=200.0,
            stiffness=2000.0,
            damping=2000.0,
        ),
    },
)