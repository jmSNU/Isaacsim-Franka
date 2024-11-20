import torch
from omni.isaac.lab.utils import configclass
from ..franka_manipulation import FrankaBaseEnv, FrankaBaseEnvCfg

@configclass
class FrankaReachEnvCfg(FrankaBaseEnvCfg):
    pass


class FrankaReachEnv(FrankaBaseEnv):
    cfg: FrankaReachEnvCfg

    def __init__(self, cfg: FrankaReachEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _get_rewards(self) -> torch.Tensor:
        ee_pos = torch.mean(self.robot.data.body_state_w[:, self.finger_idx, 0:3], dim = 1)
        self.update_target_pos()
        dist = torch.norm(ee_pos - self.target_pos, dim=1)
        dist_xy = torch.norm(ee_pos[:,:2] - self.target_pos[:,:2], dim = 1)
        dist_z = torch.norm(ee_pos[:,-1] - self.target_pos[:,-1])
        reward = torch.where(dist_xy<0.06, -dist, - dist_xy - dist_z)
        reward += torch.where(dist <= 0.05, 50, 0)

        wrist_pos = self.robot.data.body_state_w[:,self.ee_idx,0:3]
        wrist_to_ee = ee_pos - wrist_pos
        wrist_to_ee = torch.nn.functional.normalize(wrist_to_ee, dim = 1)

        ee_to_target = self.target_pos - ee_pos
        ee_to_target = torch.nn.functional.normalize(ee_to_target,dim = 1)

        for env_id in range(self.num_envs):
            reward[env_id] += 10*torch.dot(wrist_to_ee[env_id], ee_to_target[env_id])

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos = torch.mean(self.robot.data.body_state_w[:, self.finger_idx, 0:3], dim = 1)
        self.update_target_pos()
        distances = torch.norm(ee_pos - self.target_pos, dim=1)
        
        undesired_contact_body_ids,_ = self.sensor.find_bodies(['base_link', 'shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link'])
        dones = torch.empty((self.num_envs,), dtype = torch.bool, device = self.device)
        for env_id in range(self.num_envs):
            dones[env_id] = distances[env_id] <= 0.01 or distances[env_id] >= 1.5 or torch.any(torch.norm(self.sensor.data.net_forces_w[env_id, undesired_contact_body_ids, :], dim=-1) > 1e-3) or self.target_pos[env_id,-1] < 0.8

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return dones, time_out

    def update_target_pos(self):
        self.target_pos = self.target.data.root_state_w[:,:3].clone()
