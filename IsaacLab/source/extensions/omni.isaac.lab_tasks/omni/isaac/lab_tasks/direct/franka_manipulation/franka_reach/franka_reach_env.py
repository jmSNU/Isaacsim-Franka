from typing import Sequence
import torch
from omni.isaac.lab.utils import configclass
from ..franka_manipulation import FrankaBaseEnv, FrankaBaseEnvCfg
from ..reward_utils.reward_utils import *

@configclass
class FrankaReachEnvCfg(FrankaBaseEnvCfg):
    episode_length_s = 5.0

class FrankaReachEnv(FrankaBaseEnv):
    cfg: FrankaReachEnvCfg

    def __init__(self, cfg: FrankaReachEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _get_rewards(self) -> torch.Tensor:
        tcp = torch.mean(self.robot.data.body_state_w[:, self.finger_idx, 0:3], dim = 1)
        tcp_to_goal = torch.norm(tcp - self.goal, dim = 1)
        in_place_margin = torch.norm(self.init_tcp - self.goal, dim = 1)
        in_place = tolerance(
            tcp_to_goal,
            bounds = (0, 0.05),
            margin = in_place_margin,
            sigmoid="long_tail"
        )
        return 10*in_place

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        tcp = torch.mean(self.robot.data.body_state_w[:, self.finger_idx, 0:3], dim = 1)
        tcp_to_goal = torch.norm(tcp - self.goal, dim = 1)
        
        contacts_dones_condition = torch.any(torch.norm(self.sensor.data.net_forces_w[:, self.undesired_contact_body_ids, :], dim=-1) > 1e-3, dim = -1)
        dones = torch.logical_or(contacts_dones_condition, self.target_pos[:,-1]<0.8)
        dones = torch.logical_or(tcp_to_goal>=1.0, dones)
        dones = torch.logical_or(tcp_to_goal<=0.05, dones)

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return dones, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)

        goal_pose = self.update_goal_or_target(offset=self.init_tcp.clone(), which = "goal", dz_range= (-0.5, 0.1))
        self.goal[env_ids,:] = goal_pose[env_ids,:3].clone() # destination to arrive
        
        marker_locations = self.goal
        marker_orientations = torch.tensor([1, 0, 0, 0],dtype=torch.float32).repeat(self.num_envs,1).to(self.device)  
        marker_indices = torch.zeros((self.num_envs,), dtype=torch.int32)  
        self.target_marker.visualize(translations = marker_locations, orientations = marker_orientations, marker_indices = marker_indices)
         
