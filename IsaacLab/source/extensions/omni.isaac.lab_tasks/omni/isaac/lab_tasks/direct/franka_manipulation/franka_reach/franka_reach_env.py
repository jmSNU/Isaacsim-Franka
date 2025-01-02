from typing import Sequence
import torch
from omni.isaac.lab.utils import configclass
from ..franka_manipulation import FrankaBaseEnv, FrankaBaseEnvCfg
from ..reward_utils.reward_utils import *

@configclass
class FrankaReachEnvCfg(FrankaBaseEnvCfg):
    episode_length_s = 5.0
    use_visual_obs = False
    use_visual_marker = False
    num_observations = [3, 64, 64] if use_visual_obs else 27

class FrankaReachEnv(FrankaBaseEnv):
    cfg: FrankaReachEnvCfg

    def __init__(self, cfg: FrankaReachEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)        

    def _check_success(self) -> torch.Tensor:
        self.compute_intermediate()
        return torch.norm(self.tcp - self.goal, dim = 1) < 0.05

    def _get_rewards(self) -> torch.Tensor:
        self.compute_intermediate()
        reward = torch.zeros((self.num_envs,), device=self.device)

        success_ids = self._check_success()
        reward[success_ids] = 2.25

        dist = torch.norm(self.tcp - self.goal, dim = 1)
        reaching_reward = 1 - torch.tanh(10.0 * dist[~success_ids])  #
        reward[~success_ids] += reaching_reward

        return reward / 2.25

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.compute_intermediate()
        tcp_to_goal = torch.norm(self.tcp - self.goal, dim = 1)
        
        contacts_dones_condition = torch.any(torch.norm(self.sensor.data.net_forces_w[:, self.undesired_contact_body_ids, :], dim=-1) > 1e-3, dim = -1)
        dones = torch.logical_or(tcp_to_goal>=1.0, contacts_dones_condition)
        dones = torch.logical_or(tcp_to_goal<=0.05, dones)

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return dones, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)

