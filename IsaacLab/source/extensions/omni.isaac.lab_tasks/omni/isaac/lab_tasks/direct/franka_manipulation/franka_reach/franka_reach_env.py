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
        self.compute_intermediate()
        tcp_to_goal = torch.norm(self.tcp - self.goal, dim = 1)
        
        in_place_margin = torch.norm(self.init_tcp - self.goal, dim = 1)
        in_place = tolerance(
            tcp_to_goal,
            bounds = (0, 0.05),
            margin = in_place_margin,
            sigmoid="long_tail"
        )
        return 10*in_place

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.compute_intermediate()
        tcp_to_goal = torch.norm(self.tcp - self.goal, dim = 1)
        
        contacts_dones_condition = torch.any(torch.norm(self.sensor.data.net_forces_w[:, self.undesired_contact_body_ids, :], dim=-1) > 1e-3, dim = -1)
        dones = torch.logical_or(contacts_dones_condition, self.target_pos[:,-1]<0.8)
        dones = torch.logical_or(tcp_to_goal>=1.0, dones)
        dones = torch.logical_or(tcp_to_goal<=0.05, dones)

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return dones, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)

