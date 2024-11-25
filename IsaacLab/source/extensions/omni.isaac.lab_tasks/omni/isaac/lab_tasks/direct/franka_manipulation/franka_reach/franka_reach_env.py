import torch
from omni.isaac.lab.utils import configclass
from ..franka_manipulation import FrankaBaseEnv, FrankaBaseEnvCfg
from ..reward_utils.reward_utils import *

@configclass
class FrankaReachEnvCfg(FrankaBaseEnvCfg):
    pass


class FrankaReachEnv(FrankaBaseEnv):
    cfg: FrankaReachEnvCfg

    def __init__(self, cfg: FrankaReachEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _get_rewards(self) -> torch.Tensor:
        tcp = torch.mean(self.robot.data.body_state_w[:, self.finger_idx, 0:3], dim = 1)
        self.update_target_pos()
        tcp_to_target = torch.norm(tcp-self.target_pos, dim = 1)
        in_place_margin = torch.norm(self.init_tcp - self.target_pos, dim = 1)
        in_place = tolerance(
            tcp_to_target,
            bounds = (0, 0.05),
            margin = in_place_margin,
        )

        return 10*in_place

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos = torch.mean(self.robot.data.body_state_w[:, self.finger_idx, 0:3], dim = 1)
        self.update_target_pos()
        distances = torch.norm(ee_pos - self.target_pos, dim=1)
        
        undesired_contact_body_ids,_ = self.sensor.find_bodies(['panda_link0', 'panda_link1', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link5', 'panda_link6', 'panda_link7'])
        dones = torch.empty((self.num_envs,), dtype = torch.bool, device = self.device)
        for env_id in range(self.num_envs):
            dones[env_id] = distances[env_id] <= 0.05 or distances[env_id] >= 1.5 or torch.any(torch.norm(self.sensor.data.net_forces_w[env_id, undesired_contact_body_ids, :], dim=-1) > 1e-3) or self.target_pos[env_id,-1] < 0.8

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return dones, time_out

    def update_target_pos(self):
        self.target_pos = self.target.data.root_state_w[:,:3].clone()
