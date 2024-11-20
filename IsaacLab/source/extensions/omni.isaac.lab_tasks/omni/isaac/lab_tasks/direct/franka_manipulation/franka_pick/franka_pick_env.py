import torch
import numpy as np
from collections.abc import Sequence
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import sample_uniform
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from ..franka_manipulation import FrankaBaseEnv, FrankaBaseEnvCfg

@configclass
class FrankaPickEnvCfg(FrankaBaseEnvCfg):
    pass


class FrankaPickEnv(FrankaBaseEnv):
    cfg: FrankaPickEnvCfg

    def __init__(self, cfg: FrankaPickEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.goal = self.table_pos.clone()
        self.init_dist = torch.norm(self.target_pos - self.goal, dim = 1)
        self.target_marker = _define_markers() # visualize the target position  

    def _get_rewards(self) -> torch.Tensor:
        raise NotImplementedError

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
    
    def _reset_idx(self, env_ids: Sequence[int] | None):
        raise NotImplementedError

    def update_target_pos(self):
        self.target_pos = self.target.data.root_state_w[:,:3].clone()

    def update_goal_pose(self):
        target_pos = self.target_pos.clone()
        dx = (torch.rand(self.num_envs, 1) * 0.05 + 0.05) * (torch.randint(0, 2, (self.num_envs, 1)) * 2 - 1)
        # Generate random numbers between 0.1 and 0.2 or -0.2 and -0.1
        dy = (torch.rand(self.num_envs, 1) * 0.1 + 0.1) * (torch.randint(0, 2, (self.num_envs, 1)) * 2 - 1)

        x = target_pos[:, 0].unsqueeze(1) + dx.to(self.device)
        y = target_pos[:, 1].unsqueeze(1) + dy.to(self.device)
        z = target_pos[:, 2].unsqueeze(1)

        return torch.cat((x, y, z), dim=1)

def _generate_random_target_pos(num_envs, device, offset) -> torch.Tensor:
    """Generate random target positions with quaternion and velocities."""
    table_size = (0.7, 0.7)  # x, y size of the table
    x = torch.rand(num_envs, 1) * (table_size[0] / 4)
    y = torch.rand(num_envs, 1) * (table_size[1] / 2) - (table_size[1] / 4)
    z = torch.ones((num_envs, 1)) * 0.5

    quaternion = torch.tensor([0.7071, 0.0, 0.7071, 0.0]).repeat(num_envs, 1)
    translational_velocity = torch.zeros((num_envs, 3))
    rotational_velocity = torch.zeros((num_envs, 3))

    target_pos = torch.cat((x, y, z), dim=1) + offset
    combined_tensor = torch.cat((target_pos, quaternion, translational_velocity, rotational_velocity), dim=1)
    combined_tensor = combined_tensor.to(device)

    return combined_tensor

def _define_markers() -> VisualizationMarkers:
    """Define markers to visualize the target position."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/TargetMarkers",
        markers={
            "target": sim_utils.SphereCfg(  
                radius=0.05,  
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)), 
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)
