import torch
import numpy as np
from collections.abc import Sequence
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import sample_uniform
from ..franka_manipulation import FrankaBaseEnv, FrankaBaseEnvCfg
from ..reward_utils.reward_utils import *


@configclass
class FrankaPushEnvCfg(FrankaBaseEnvCfg):
    table_size = (0.7, 0.7)

class FrankaPushEnv(FrankaBaseEnv):
    cfg: FrankaPushEnvCfg
    def __init__(self, cfg: FrankaPushEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
 

    def _get_rewards(self) -> torch.Tensor:
        self.update_target_pos()
        tcp_pos = torch.mean(self.robot.data.body_state_w[:, self.finger_idx, 0:3], dim = 1)
        tcp_to_target = torch.norm(tcp_pos - self.target_pos, dim = 1)
        goal_to_target = torch.norm(self.target_pos - self.goal, dim = 1)
        goal_to_target_init = self.init_dist.clone()
        
        reward = torch.zeros(self.num_envs, device = self.device, dtype = torch.float32)
        
        in_place = tolerance(
            goal_to_target, 
            bounds=(0, 0.05), 
            margin=goal_to_target_init, 
            sigmoid="long_tail")
        
        object_grasped = self._gripper_grasp_reward(
            self.actions,
            self.target_pos,
            obj_reach_radius=0.04,
            obj_radius=0.02,
            pad_success_thresh=0.05,
            xz_thresh=0.05,
            desired_gripper_effort=0.7,
            medium_density=True
        )
        
        reward = hamacher_product(object_grasped, in_place)
        tcp_target_condition = torch.logical_and(tcp_to_target < 0.02, self.actions[:,-1] >0)
        goal_target_condition = goal_to_target < 0.05
        
        reward[tcp_target_condition] += 1.0 + 5.0 * in_place[tcp_target_condition]
        reward[goal_target_condition] = 10.0
        
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        tcp_pos = torch.mean(self.robot.data.body_state_w[:, self.finger_idx, 0:3], dim = 1)
        self.update_target_pos()
        goal_to_target = torch.norm(tcp_pos - self.goal, dim=1)
        tcp_to_target = torch.norm(tcp_pos - self.target_pos, dim = 1)
        
        contacts_dones_condition = torch.any(torch.norm(self.sensor.data.net_forces_w[:, self.undesired_contact_body_ids, :], dim=-1) > 1e-3, dim = -1)
        dones = torch.logical_or(contacts_dones_condition, self.target_pos[:,-1]<0.8)
        dones = torch.logical_or(tcp_to_target>=1.0, dones)
        dones = torch.logical_or(goal_to_target>=0.7, dones)
        dones = torch.logical_or(goal_to_target<=0.01, dones)

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return dones, time_out
    
    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)

        goal_pose = self.update_goal_or_target(target_pos=self.target_pos.clone(), dz_range=(0.0, 0.0))
        self.goal[env_ids,:] = goal_pose[env_ids,:3].clone() # destination to arrive
        self.init_dist[env_ids] = torch.norm(self.target_pos[env_ids,:] - self.goal[env_ids,:], dim = 1)

        marker_locations = self.goal[env_ids,:]
        marker_orientations = torch.tensor([1, 0, 0, 0],dtype=torch.float32).repeat(len(env_ids),1).to(self.device)  
        marker_indices = torch.zeros((env_ids,), dtype=torch.int32)  
        self.target_marker.visualize(translations = marker_locations, orientations = marker_orientations, marker_indices = marker_indices)

