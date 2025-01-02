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
    use_visual_obs = False
    use_visual_marker = False
    num_observations = [3, 64, 64] if use_visual_obs else 27

class FrankaPushEnv(FrankaBaseEnv):
    cfg: FrankaPushEnvCfg
    def __init__(self, cfg: FrankaPushEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
    
    def _check_success(self) -> torch.Tensor:
        self.compute_intermediate()
        return torch.norm(self.target_pos - self.goal, dim = 1) < 0.05


    def _get_rewards(self) -> torch.Tensor:
        self.compute_intermediate()
        reward = torch.zeros((self.num_envs,), device=self.device)

        success_ids = self._check_success()
        reward[success_ids] = 2.25

        tcp_to_target_dist = self._tcp_to_target(return_dist=True)
        reaching_reward = 1 - torch.tanh(10.0 * tcp_to_target_dist[~success_ids])
        reward[~success_ids] += reaching_reward * 0.5  

        target_to_goal_dist = torch.norm(self.target_pos[~success_ids] - self.goal[~success_ids], dim=1)
        pushing_reward = 1 - torch.tanh(5.0 * target_to_goal_dist)
        reward[~success_ids] += pushing_reward * 0.75 

        return reward / 2.25

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
    
    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)

        tcp_to_goal = torch.zeros((len(env_ids),))
        goal_to_target = torch.zeros((len(env_ids),))

        for i, env_id in enumerate(env_ids):
            while True:
                goal_pose = self.update_goal_or_target(offset = self.on_table_pos, which = "goal", dx_range = (0.1, 0.2), dy_range = (-0.1, 0.1), dz_range = (0.0, 0.0))
                tcp_to_goal[i] = torch.norm(goal_pose[env_id,:3] - self.init_tcp[env_id,:])
                goal_to_target[i] = torch.norm(self.target_pos[env_id,:] - goal_pose[env_id,:3])
                if tcp_to_goal[i] > 0.1 and goal_to_target[i] > 0.1:
                    break
            self.goal[env_id,:] = goal_pose[env_id,:3].clone() # destination to arrive        
        self.init_dist[env_ids] = torch.norm(self.target_pos[env_ids,:] - self.goal[env_ids,:], dim = 1)

        marker_locations = self.goal
        marker_orientations = torch.tensor([1, 0, 0, 0],dtype=torch.float32).repeat(self.num_envs,1).to(self.device)  
        marker_indices = torch.zeros((self.num_envs,), dtype=torch.int32)  
        if self.cfg.use_visual_marker:
            self.marker.visualize(translations = marker_locations, orientations = marker_orientations, marker_indices = marker_indices)

