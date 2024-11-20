"""
Refer https://isaac-sim.github.io/IsaacLab/source/migration/migrating_from_isaacgymenvs.html
"""

from collections.abc import Sequence
import math
import torch
import numpy as np
from typing import List

from omni.isaac.lab_assets.unitree import UNITREE_A1_CFG
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import subtract_frame_transforms,sample_uniform
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from gymnasium.spaces.box import Box
from omni.isaac.lab.managers import SceneEntityCfg
import omni.isaac.core.utils.torch as torch_utils
from omni.isaac.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.sensors import ContactSensorCfg, ContactSensor,RayCasterCfg, RayCaster, patterns



def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


class EnvScene(InteractiveSceneCfg):
    # robot
    robot: ArticulationCfg = UNITREE_A1_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # lights
    light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # sensor
    sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", update_period=0.0, history_length=6, debug_vis=True
    )

    scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/trunk",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class UnitreeA1EnvCfg(DirectRLEnvCfg):
    # env  episode_length_s = dt * decimation * num_steps
    decimation = 2
    episode_length_s = 20.0
    action_scale = 1.0  # Scale for the CPG parameters
    num_actions = 7
    num_observations = 247  # Customize based on observations

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1/240, render_interval=decimation)

    scene : InteractiveSceneCfg = EnvScene(num_envs = 1024, env_spacing = 4.0)

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # reset
    joint_gears = [1.0] * 12

    # reward scales
    up_weight = 0.1
    heading_weight = 0.1
    actions_cost_scale = 0.5
    dof_vel_scale = 0.05
    energy_cost_scale = 0.01
    death_cost = -10.0
    alive_reward_scale = 0.5
    progress_reward_scale = 10.0
    undesired_contact_reward_scale = 10.0

    # observation scales
    angular_velocity_scale = 0.1
    dof_vel_scale = 0.2

    # dones
    termination_height = 0.03
    

class UnitreeA1Env(DirectRLEnv):
    cfg: UnitreeA1EnvCfg

    def __init__(self, cfg: UnitreeA1EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.action_scale = self.cfg.action_scale
        self.action_space = Box(
            low=np.array([0.1, 0.1, 0.1, 0.1, -0.1, -0.05, -0.05]), # sig1, sig2, k1, k2, A, B, A'
            high=np.array([2.0, 2.0, 1.0, 1.0, 0.1, 0.05, 0.05]),
        )

        self.dt = self.cfg.sim.dt
        self.num_legs = 4
        self.omega = 0.04
        self.num_neighbors = 2

        self.joint_gears = torch.tensor(self.cfg.joint_gears, dtype=torch.float32, device=self.sim.device)
        self.motor_effort_ratio = torch.ones_like(self.joint_gears, device=self.sim.device)
        self._joint_dof_idx, _ = self.robot.find_joints([".*"])

        self.potentials = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)
        self.prev_potentials = torch.zeros_like(self.potentials)
        self.targets = torch.tensor([1000, 0, 0], dtype=torch.float32, device=self.sim.device).repeat(
            (self.num_envs, 1)
        )
        self.targets += self.scene.env_origins
        self.start_rotation = torch.tensor([1, 0, 0, 0], device=self.sim.device, dtype=torch.float32)
        self.up_vec = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        self.heading_vec = torch.tensor([1, 0, 0], dtype=torch.float32, device=self.sim.device).repeat(
            (self.num_envs, 1)
        )
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))
        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        self.joint_ids, _ = self.robot.find_joints(['.*_joint']) # ['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint']
        self.body_ids, _ = self.robot.find_bodies(['.*']) # ['trunk', 'FL_hip', 'FR_hip', 'RL_hip', 'RR_hip', 'FL_thigh', 'FR_thigh', 'RL_thigh', 'RR_thigh', 'FL_calf', 'FR_calf', 'RL_calf', 'RR_calf', 'FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
        
        self.foot_names = ["FL","FR","RL","RR"]
        self.foot_ids, _ = self.robot.find_bodies([".*_foot"]) # ["FL","FR","RL","RR"]

        self.base_id, _ = self.robot.find_bodies("trunk")
        self.underisred_contact_body_ids, _ = self.sensors.find_bodies([".*_calf", ".*_thigh", "trunk"])

        ik_cfg = DifferentialIKControllerCfg(
            command_type="position",
            use_relative_mode=True, 
            ik_method="dls",
            ik_params = {
                "pinv": {"k_val": 1.0},
                "svd": {"k_val": 1.0, "min_singular_value": 1e-5},
                "trans": {"k_val": 1.0},
                "dls": {"lambda_val": 0.01},
            }
        )

        self.diff_ik_controller = DifferentialIKController(ik_cfg, num_envs=cfg.scene.num_envs, device=self.robot.device)

    def _setup_scene(self):
        self.robot = self.scene['robot']
        self.sensors = self.scene['sensor']
        self.scanner = self.scene['scanner']

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        # self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        low = torch.tensor(self.action_space.low, dtype=self.actions.dtype, device=self.device)
        high = torch.tensor(self.action_space.high, dtype=self.actions.dtype, device=self.device)

        actions = torch.clamp(self.actions, low, high)
        phases = self._compute_cpg_phases(actions)
        amplitude = actions[:,4:]

        target_positions = self._compute_target_positions(phases, amplitude) 
        joint_positions_des = torch.zeros((self.num_envs, 12), device = self.device)

        for id in range(self.num_legs):
            foot_id = self.foot_ids[id]
            joint_ids,_ = self.robot.find_joints([self.foot_names[id] + "_thigh_joint", self.foot_names[id] + "_hip_joint"])
    
            trunk_pos_w = self.robot.data.root_state_w[:, :3] # (N, 3)
            trunk_ori_w = self.robot.data.root_state_w[:, 3:7] # (N, 4)
            foot_pose_w = self.robot.data.body_state_w[:, foot_id, 0:7] # (N, 7)

            foot_pos_b, foot_quat_b = subtract_frame_transforms(
                trunk_pos_w, trunk_ori_w, foot_pose_w[:, :3], foot_pose_w[:,3:]
            ) # trunk_pos_b : (N,3), trunk_quad_b : (N,4)

            ik_commands = torch.zeros(self.scene.num_envs, self.diff_ik_controller.action_dim, device=self.robot.device)
            ik_commands[:] = target_positions[:,id, :] + foot_pos_b # (num_envs, 4, 3)

            self.diff_ik_controller.set_command(ik_commands, ee_quat = foot_quat_b)

            # Convert target positions to desired joint positions using IK
            joint_positions_des[:, joint_ids] = self._compute_joint_positions_from_targets(foot_id, joint_ids)

        # Set joint positions as the target for the robot
        self.robot.set_joint_position_target(joint_positions_des, joint_ids = self.joint_ids)


    def _compute_cpg_phases(self, actions: torch.Tensor) -> torch.Tensor:
        sigma1 = actions[:, 0]
        sigma2 = actions[:, 1]
        k = actions[:, 2:4]

        foot_ids, _ = self.sensors.find_bodies([".*_foot"])
        N_V = self.sensors.data.net_forces_w[:, foot_ids, 2]
        neighbors = torch.tensor([[1,2],[0,3],[0,3],[1,2]], device=self.device)

        for i in range(self.num_legs):
            NiV = N_V[:, i]
            cos_phi_i = torch.cos(self.phases[:, i]).to(actions.device)

            neighbor_sum = torch.zeros_like(NiV).to(self.device)

            for j in range(self.num_neighbors):
                neighbor_idx = neighbors[i, j]
                neighbor_sum += k[:, j] * N_V[:, neighbor_idx]

            # Calculate phase_dot
            phase_dot = self.omega - sigma1 * NiV * cos_phi_i + sigma2 * (1.0 / self.num_neighbors) * neighbor_sum * cos_phi_i
            self.phase_dot[:,i] = phase_dot
            self.phases[:, i] += phase_dot * self.dt

        return self.phases

    def _compute_joint_positions_from_targets(self, foot_id, joint_ids) -> torch.Tensor:

        current_joint_pos = self.robot.data.joint_pos[:, joint_ids] # self.robot.data.joint_pose : (N, num_joints) , current_joint_pos : (N, 3, 1)

        jacobian = self.robot.root_physx_view.get_jacobians()[:, foot_id, :, joint_ids] # (N, num_frames, 6, num_joints) -> (N, 17, 6, 18) : 12 for joints and 6 for (linear + angluar) velocity
        # jacobian : (N, 6, 3, 1)

        trunk_pos_w = self.robot.data.root_state_w[:, :3] # (N, 3)
        trunk_ori_w = self.robot.data.root_state_w[:, 3:7] # (N, 4)
        foot_pose_w = self.robot.data.body_state_w[:, foot_id, 0:7] # (N, 7)

        foot_pos_b, foot_quat_b = subtract_frame_transforms(
            trunk_pos_w, trunk_ori_w, foot_pose_w[:, :3], foot_pose_w[:,3:]
        ) # trunk_pos_b : (N,3), trunk_quad_b : (N,4)

        return self.diff_ik_controller.compute(foot_pos_b, foot_quat_b, jacobian, current_joint_pos)
    
    def _compute_target_positions(self, phases: torch.Tensor, amplitude : torch.Tensor) -> torch.Tensor:
        target_positions = torch.zeros((self.cfg.scene.num_envs, self.num_legs, 3), device=self.robot.device)

        for i in range(self.num_legs):
            target_positions[:, i, 0] = amplitude[:,1] * torch.cos(phases[:, i])
            target_positions[:, i, 1] = torch.where((phases[:, i] >= 0) & (phases[:, i] < torch.pi), amplitude[:,0] * torch.sin(phases[:, i]), amplitude[:,2] * torch.sin(phases[:, i]))
        return target_positions

    
    def _compute_intermediate_values(self):
        self.torso_position, self.torso_rotation = self.robot.data.root_pos_w, self.robot.data.root_quat_w
        self.velocity, self.ang_velocity = self.robot.data.root_lin_vel_w, self.robot.data.root_ang_vel_w
        self.dof_pos, self.dof_vel = self.robot.data.joint_pos, self.robot.data.joint_vel

        (
            self.up_proj,
            self.heading_proj,
            self.up_vec,
            self.heading_vec,
            self.vel_loc,
            self.angvel_loc,
            self.roll,
            self.pitch,
            self.yaw,
            self.angle_to_target,
            self.dof_pos_scaled,
            self.prev_potentials,
            self.potentials,
        ) = compute_intermediate_values(
            self.targets,
            self.torso_position,
            self.torso_rotation,
            self.velocity,
            self.ang_velocity,
            self.dof_pos,
            self.robot.data.soft_joint_pos_limits[0, :, 0],
            self.robot.data.soft_joint_pos_limits[0, :, 1],
            self.inv_start_rot,
            self.basis_vec0,
            self.basis_vec1,
            self.potentials,
            self.prev_potentials,
            self.cfg.sim.dt,
        )
    
    def _get_observations(self) -> dict:

        height_data = (
                self.scanner.data.pos_w[:, 2].unsqueeze(1) - self.scanner.data.ray_hits_w[..., 2] - 0.5
        ).clip(-1.0, 1.0)

        obs = torch.cat(
            (
                self.phases,
                self.phase_dot,
                self.torso_position[:, 2].view(-1, 1),
                self.vel_loc,
                self.angvel_loc * self.cfg.angular_velocity_scale,
                normalize_angle(self.yaw).unsqueeze(-1),
                normalize_angle(self.roll).unsqueeze(-1),
                normalize_angle(self.angle_to_target).unsqueeze(-1),
                self.actions,
                self.sensors.data.net_forces_w[:, :, 2],
                height_data
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        torque = self.robot.data.applied_torque
        total_reward = compute_rewards(
            torque,
            self.reset_terminated,
            self.cfg.up_weight,
            self.cfg.heading_weight,
            self.heading_proj,
            self.up_proj,
            self.dof_vel,
            self.dof_pos_scaled,
            self.potentials,
            self.prev_potentials,
            self.cfg.energy_cost_scale,
            self.cfg.dof_vel_scale,
            self.cfg.death_cost,
            self.cfg.alive_reward_scale,
            self.cfg.progress_reward_scale,
            self.motor_effort_ratio,
            self.sensors.data.net_forces_w_history,
            self.underisred_contact_body_ids,
            self.cfg.undesired_contact_reward_scale
        )
        return total_reward


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self.sensors.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self.base_id], dim=-1), dim=1)[0] > 1e-3, dim=1)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.phases = torch.randn(self.cfg.scene.num_envs, self.num_legs, device=self.device)
        self.phase_dot = torch.zeros_like(self.phases)
        self.robot.reset(env_ids)
        self.sensors.reset(env_ids)
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), len(self.joint_ids)),
            self.device,
        )
        joint_vel = joint_vel = torch.zeros_like(joint_pos)
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids = env_ids)

        to_target = self.targets[env_ids] - default_root_state[:, :3]
        to_target[:, 2] = 0.0
        self.potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.cfg.sim.dt

        self._compute_intermediate_values()


def compute_rewards(
    torque: torch.Tensor,
    reset_terminated: torch.Tensor,
    up_weight: float,
    heading_weight: float,
    heading_proj: torch.Tensor,
    up_proj: torch.Tensor,
    dof_vel: torch.Tensor,
    dof_pos_scaled: torch.Tensor,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
    energy_cost_scale: float,
    dof_vel_scale: float,
    death_cost: float,
    alive_reward_scale: float,
    progress_reward_scale : float,
    motor_effort_ratio: torch.Tensor,
    net_contact_forces : torch.Tensor,
    underisred_contact_body_ids : List[int],
    underisred_contact_reward_scale : float
):
    heading_weight_tensor = torch.ones_like(heading_proj) * heading_weight
    heading_reward = torch.where(heading_proj > 0.8, heading_weight_tensor, heading_weight * heading_proj / 0.8)

    # aligning up axis of robot and environment
    up_reward = torch.zeros_like(heading_reward)
    up_reward = torch.where(up_proj > 0.93, up_reward + up_weight, up_reward)

    # energy penalty for movement
    electricity_cost = torch.sum(
        torch.abs(torque * dof_vel * dof_vel_scale) * motor_effort_ratio.unsqueeze(0),
        dim=-1,
    )

    # undesired contact penalty
    is_contact = (
        torch.max(torch.norm(net_contact_forces[:, :, underisred_contact_body_ids,:], dim=-1), dim=1)[0] > 1.0
    )
    contacts = torch.sum(is_contact, dim=1)

    # dof at limit cost
    dof_at_limit_cost = torch.sum(dof_pos_scaled > 0.98, dim=-1)

    # reward for duration of staying alive
    alive_reward = torch.ones_like(potentials) * alive_reward_scale
    progress_reward = potentials - prev_potentials

    total_reward = (
        progress_reward * progress_reward_scale
        + alive_reward
        + up_reward
        + heading_reward
        - energy_cost_scale * electricity_cost
        - dof_at_limit_cost
        - contacts * underisred_contact_reward_scale
    )
    # adjust reward for fallen agents
    total_reward = torch.where(reset_terminated, torch.ones_like(total_reward) * death_cost, total_reward)
    
    return total_reward

    

@torch.jit.script
def compute_intermediate_values(
    targets: torch.Tensor,
    torso_position: torch.Tensor,
    torso_rotation: torch.Tensor,
    velocity: torch.Tensor,
    ang_velocity: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_lower_limits: torch.Tensor,
    dof_upper_limits: torch.Tensor,
    inv_start_rot: torch.Tensor,
    basis_vec0: torch.Tensor,
    basis_vec1: torch.Tensor,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
    dt: float,
):
    to_target = targets - torso_position
    to_target[:, 2] = 0.0

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2
    )

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position
    )

    dof_pos_scaled = torch_utils.maths.unscale(dof_pos, dof_lower_limits, dof_upper_limits)

    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    prev_potentials[:] = potentials
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    return (
        up_proj,
        heading_proj,
        up_vec,
        heading_vec,
        vel_loc,
        angvel_loc,
        roll,
        pitch,
        yaw,
        angle_to_target,
        dof_pos_scaled,
        prev_potentials,
        potentials,
    )
