import torch
import numpy as np
from collections.abc import Sequence
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import sample_uniform
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from ..franka_manipulation import FrankaBaseEnv, FrankaBaseEnvCfg
from ..reward_utils.reward_utils import *

@configclass
class FrankaPickEnvCfg(FrankaBaseEnvCfg):
    table_size = (0.7, 0.7)

class FrankaPickEnv(FrankaBaseEnv):
    cfg: FrankaPickEnvCfg

    def __init__(self, cfg: FrankaPickEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _gripper_grasp_reward(self, action, obj_pos, obj_radius = 0, pad_success_thresh = 0, obj_reach_radius = 0, xz_thresh = 0, desired_gripper_effort=1, high_density=False, medium_density=False):
        pad_success_margin = 0.05
        x_z_success_margin = 0.005
        obj_radius = 0.015
        tcp = self.tcp
        left_pad = self.robot.data.body_state_w[:, self.finger_idx[0], 0:3]
        right_pad = self.robot.data.body_state_w[:, self.finger_idx[1], 0:3]
        delta_obj_y_left_pad = left_pad[:, 1] - obj_pos[:, 1]
        delta_obj_y_right_pad = obj_pos[:, 1] - right_pad[:, 1]

        right_caging_margin = torch.abs(
            torch.abs(obj_pos[:,1] - self.init_left_pad[:,1]) - pad_success_margin
        )
        left_caging_margin = torch.abs(
            torch.abs(obj_pos[:, 1] - self.init_right_pad[:,1] - pad_success_margin)
        )

        right_caging = tolerance(
            delta_obj_y_right_pad, 
            bounds = (obj_radius, pad_success_margin),
            margin = right_caging_margin,
            sigmoid="long_tail"
        )
        left_caging = tolerance(
            delta_obj_y_left_pad,
            bounds = (obj_radius, pad_success_margin),
            margin = left_caging_margin,
            sigmoid="long_tail"
        )

        y_caging = hamacher_product(left_caging, right_caging)

        tcp_xz = tcp.clone()
        tcp_xz[:,1] = 0.0
        obj_position_xz = obj_pos.clone()
        obj_position_xz[:, 1] = 0.0
        tcp_obj_dist_xz = torch.norm(tcp_xz - obj_position_xz, dim = 1)

        init_obj_xz = self.target_init_pos.clone()
        init_obj_xz[:, 1] = 0.0
        init_tcp_xz = self.init_tcp.clone()
        init_tcp_xz[:, 1] = 0.0
        tcp_obj_xz_margin = (
            torch.norm(init_obj_xz - init_tcp_xz, dim = 1) - x_z_success_margin
        )

        x_z_caging = tolerance(
            tcp_obj_dist_xz,
            bounds = (0, x_z_success_margin),
            margin = tcp_obj_xz_margin,
            sigmoid = "long_tail"
        )

        gripper_closed = self.actions[:, -1] <= 0
        caging = hamacher_product(y_caging, x_z_caging)

        gripping = torch.where(caging>0.97, gripper_closed, 0.0)
        caging_and_gripping = hamacher_product(caging, gripping)
        caging_and_gripping = (caging_and_gripping + caging)/2
        return caging_and_gripping


    def _get_rewards(self) -> torch.Tensor:
        self.compute_intermediate()
        obj = self.target_pos
        tcp_opened = self.actions[:, -1] >0

        tcp_to_target = torch.norm(self.tcp - obj, dim = 1)
        goal_to_target = torch.norm(obj - self.goal, dim = 1)
        inplace_margin = self.init_dist.clone()

        in_place = tolerance(
            goal_to_target,
            bounds=(0, 0.05),
            margin=inplace_margin,
            sigmoid="long_tail"
        )

        object_grasped = self._gripper_grasp_reward(self.actions, obj)
        inplace_and_object_grasped = hamacher_product(
            object_grasped, in_place
        )
        reward = inplace_and_object_grasped

        tcp_target_condition = torch.logical_and(torch.logical_and(tcp_to_target < 0.02, tcp_opened), obj[:,2]-0.01 > self.target_init_pos[:,2])
        goal_target_condition = goal_to_target < 0.05

        reward[tcp_target_condition] += 1.0 + 5.0*in_place[tcp_target_condition]
        reward[goal_target_condition] = 10.0
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.compute_intermediate()
        goal_to_target = torch.norm(self.tcp - self.goal, dim=1)
        tcp_to_target = torch.norm(self.tcp - self.target_pos, dim = 1)
        
        contacts_dones_condition = torch.any(torch.norm(self.sensor.data.net_forces_w[:, self.undesired_contact_body_ids, :], dim=-1) > 1e-3, dim = -1)
        dones = torch.logical_or(contacts_dones_condition, self.target_pos[:,-1]<0.8)
        dones = torch.logical_or(tcp_to_target>=1.0, dones)
        dones = torch.logical_or(goal_to_target>=0.7, dones)
        dones = torch.logical_or(goal_to_target<=0.05, dones)

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return dones, time_out
    
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self.init_left_pad = self.robot.data.body_state_w[:, self.finger_idx[0], 0:3]
        self.init_right_pad = self.robot.data.body_state_w[:, self.finger_idx[1], 0:3]
    
