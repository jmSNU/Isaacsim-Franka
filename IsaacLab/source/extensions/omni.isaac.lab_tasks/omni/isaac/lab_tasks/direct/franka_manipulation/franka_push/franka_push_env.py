import torch
import numpy as np
from collections.abc import Sequence
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import sample_uniform
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from ..franka_manipulation import FrankaBaseEnv, FrankaBaseEnvCfg

@configclass
class FrankaPushEnvCfg(FrankaBaseEnvCfg):
    pass


class FrankaPushEnv(FrankaBaseEnv):
    cfg: FrankaPushEnvCfg

    def __init__(self, cfg: FrankaPushEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.goal = self.table_pos.clone()
        self.init_dist = torch.norm(self.target_pos - self.goal, dim = 1)
        self.target_marker = _define_markers() # visualize the target position  

    def _get_rewards(self) -> torch.Tensor:
        self.update_target_pos()
        tcp_pos = torch.mean(self.robot.data.body_state_w[:, self.hand_idx, 0:3], dim = 1)
        tcp_to_target = torch.norm(tcp_pos - self.target_pos, dim = 1)
        goal_to_target = torch.norm(self.target_pos - self.goal, dim = 1)
        goal_to_target_init = self.init_dist.clone()
        
        reward = torch.zeros(self.num_envs, device = self.device, dtype = torch.float32)

        def tolerance(x, bounds, margin, value_at_margin=0.1):
            lower, upper = bounds
            assert lower < upper and torch.all(margin > 0)
            
            # Check if x is within bounds
            in_bounds = torch.logical_and(lower <= x, x <= upper)
            
            if torch.all(margin == 0):
                value = torch.where(in_bounds, 1.0, 0.0)
            else:
                # Compute distance d and apply sigmoid function
                d = torch.where(x < lower, lower - x, x - upper) / margin
                sigmoid = lambda x, value_at_1: 1 / ((x * np.sqrt(1 / value_at_1 - 1))**2 + 1)
                value = torch.where(in_bounds, 1.0, sigmoid(d, value_at_margin))

            return value
        
        in_place = tolerance(goal_to_target, bounds=(0, 0.05), margin=goal_to_target_init)
        tcp_target_condition = tcp_to_target < 0.02
        goal_target_condition = goal_to_target < 0.05
        
        reward[tcp_target_condition] += 1.0 + 5.0 * in_place[tcp_target_condition]
        reward[goal_target_condition] = 10.0

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        tcp_pos = torch.mean(self.robot.data.body_state_w[:, self.hand_idx, 0:3], dim = 1)
        self.update_target_pos()
        goal_to_target = torch.norm(tcp_pos - self.goal, dim=1)
        tcp_to_target = torch.norm(tcp_pos - self.target_pos, dim = 1)
        
        undesired_contact_body_ids,_ = self.sensor.find_bodies(['base_link', 'shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link', 'flange', 'tool0', 'robotiq_base_link', 'gripper_center', 'left_outer_knuckle', 'left_inner_knuckle', 'right_inner_knuckle', 'right_outer_knuckle', 'left_outer_finger'])
        contacts_dones_condition = torch.any(torch.norm(self.sensor.data.net_forces_w[:, undesired_contact_body_ids, :], dim=-1) > 1e-3, dim = -1)
        dones = torch.logical_or(contacts_dones_condition, self.target_pos[:,-1]<0.8)
        dones = torch.logical_or(tcp_to_target>=1.0, dones)
        dones = torch.logical_or(goal_to_target>=0.7, dones)
        dones = torch.logical_or(goal_to_target<=0.01, dones)

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return dones, time_out
    
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        self.robot.reset(env_ids)
        self.camera.reset(env_ids)
        self.sensor.reset(env_ids)
        self.table.reset(env_ids)
        self.target.reset(env_ids)
        self.diff_ik_controller.reset(env_ids)
        self.diff_ik_controller_position.reset(env_ids)
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self.num_joints),
            self.device,
        )
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        spawn_pos = _generate_random_target_pos(self.num_envs, self.scene['robot'].device, offset = self.table_pos.cpu())
        self.target.write_root_state_to_sim(spawn_pos[env_ids,:], env_ids = env_ids)

        if self.cfg.enable_obstacle:
            obstacle_pos = spawn_pos + torch.tensor([0.0,0.15,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], device = self.device)
            self.obstacle.write_root_state_to_sim(obstacle_pos[env_ids,:], env_ids)
        self.target_pos[env_ids,:] = self.target.data.root_state_w[env_ids,:3].clone() # moving object's position

        goal_pose = self.update_goal_pose()
        self.goal[env_ids,:] = goal_pose[env_ids,:3].clone() # destination to arrive
        self.goal[env_ids,0] = torch.clamp(self.goal[env_ids,0], self.table_pos[env_ids,0] - self.cfg.table_size[0], self.table_pos[env_ids,0] + self.cfg.table_size[0])
        self.goal[env_ids,1] = torch.clamp(self.goal[env_ids,1], self.table_pos[env_ids,1] - self.cfg.table_size[1], self.table_pos[env_ids,1] + self.cfg.table_size[1])

        self.init_dist[env_ids] = torch.norm(self.target_pos[env_ids,:] - self.goal[env_ids,:], dim = 1)
        self.spawn_pos[env_ids,:] = self.target_pos[env_ids,:].clone() # original spawn position

        marker_locations = self.goal
        marker_orientations = torch.tensor([1, 0, 0, 0],dtype=torch.float32).repeat(self.num_envs,1).to(self.device)  
        marker_indices = torch.zeros((self.num_envs,), dtype=torch.int32)  
        self.target_marker.visualize(translations = marker_locations, orientations = marker_orientations, marker_indices = marker_indices)


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
