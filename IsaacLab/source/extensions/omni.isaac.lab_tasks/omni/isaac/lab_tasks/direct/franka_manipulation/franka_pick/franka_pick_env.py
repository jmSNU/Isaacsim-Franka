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
    use_visual_obs = False
    use_visual_marker = False
    num_observations = [3, 64, 64] if use_visual_obs else 27

class FrankaPickEnv(FrankaBaseEnv):
    cfg: FrankaPickEnvCfg

    def __init__(self, cfg: FrankaPickEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _check_success(self) -> torch.Tensor:
        self.compute_intermediate()
        target_pos = self.target_pos.clone()
        goal_pos = self.goal.clone()
        dist = torch.norm(target_pos - goal_pos, dim = 1)
        return torch.where(dist<0.05, True, False)

    def _get_rewards(self) -> torch.Tensor:
        self.compute_intermediate()
        reward = torch.zeros((self.num_envs,), device = self.device)
        success_id = self._check_success()
        reward[success_id] = 1.0
        
        reach_mult = 0.1
        grasp_mult = 0.35
        lift_mult = 0.5
        hover_mult = 0.7
        
        tcp_to_target = self._tcp_to_target(return_dist=True)
        r_reach = (1 - torch.tanh(10.0*tcp_to_target)) * reach_mult

        r_grasp = self._check_grasp(self.actions, self.target_pos, 0.089, 0.12, 0.05).to(torch.int) * grasp_mult
        
        goal_z = self.goal[:,2].unsqueeze(1)
        target_z = self.target_pos[:,2].unsqueeze(1)
        z_dist = torch.norm(goal_z - target_z, dim = 1)
        r_lift = torch.where(
            r_grasp>0,
            grasp_mult + (1-torch.tanh(15.0*z_dist)) *(lift_mult - grasp_mult),
            0.0
            )
        
        goal_xy = self.goal[:,:2]
        target_xy = self.target_pos[:,:2]
        xy_dist = torch.norm(goal_xy - target_xy, dim = 1)
        r_hover = torch.where(xy_dist < 0.05, lift_mult, r_lift)
        r_hover += (1 - torch.tanh(10.0*xy_dist)) * (hover_mult - lift_mult)

        r_tot = torch.stack([r_reach, r_grasp, r_lift, r_hover]).t()
        reward += torch.max(r_tot, dim = 1)[0]

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.compute_intermediate()
        goal_to_target = torch.norm(self.tcp - self.goal, dim=1)
        tcp_to_target = torch.norm(self.tcp - self.target_pos, dim = 1)
        
        contacts_dones_condition = torch.any(torch.norm(self.sensor.data.net_forces_w[:, self.undesired_contact_body_ids, :], dim=-1) > 1e-3, dim = -1)
        dones = torch.logical_or(contacts_dones_condition, self.target_pos[:,-1]<0.8)
        dones = torch.logical_or(tcp_to_target>=1.5, dones)
        dones = torch.logical_or(goal_to_target>=1.5, dones)
        dones = torch.logical_or(self._check_success(), dones)

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return dones, time_out
    
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self.init_left_pad = self.robot.data.body_state_w[:, self.finger_idx[0], 0:3]
        self.init_right_pad = self.robot.data.body_state_w[:, self.finger_idx[1], 0:3]
    
