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
class FrankaLiftEnvCfg(FrankaBaseEnvCfg):
    episode_length_s = 10.0
    use_visual_obs = False
    use_visual_marker = False
    num_observations = [3, 64, 64] if use_visual_obs else 27

class FrankaLiftEnv(FrankaBaseEnv):
    cfg: FrankaLiftEnvCfg

    def __init__(self, cfg: FrankaLiftEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _check_success(self) -> torch.Tensor:
        self.compute_intermediate()
        target_height = self.target_pos[:,2]
        table_height = self.cfg.table_size[2]
        return target_height>table_height + 0.1
            
    def _get_rewards(self) -> torch.Tensor:
        self.compute_intermediate()
        reward = torch.zeros((self.num_envs,), device = self.device)
        success_ids = self._check_success()
        reward[success_ids] = 2.25
        
        dist = self._tcp_to_target(return_dist=True)
        reaching_reward = 1-torch.tanh(10.0*dist[~success_ids])
        reward[~success_ids] += reaching_reward
        
        reward[~success_ids] += torch.where(
            self._check_grasp(self.actions, self.target_pos, 0.089, 0.12, 0.05)[~success_ids],
            0.25,
            0.0
        )

        return reward/2.25

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.compute_intermediate()
        tcp_to_target = torch.norm(self.tcp - self.target_pos, dim = 1)
        
        contacts_dones_condition = torch.any(torch.norm(self.sensor.data.net_forces_w[:, self.undesired_contact_body_ids, :], dim=-1) > 1e-3, dim = -1)
        dones = torch.logical_or(contacts_dones_condition, self.target_pos[:,-1]<0.8)
        dones = torch.logical_or(tcp_to_target>=1.0, dones)
        dones = torch.logical_or(self._check_success(), dones)

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return dones, time_out
    
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self.init_left_pad = self.robot.data.body_state_w[:, self.finger_idx[0], 0:3]
        self.init_right_pad = self.robot.data.body_state_w[:, self.finger_idx[1], 0:3]
    
