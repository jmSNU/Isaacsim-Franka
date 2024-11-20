# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import numpy as np
from gymnasium.spaces.box import Box


import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensor, ContactSensorCfg, RayCaster, RayCasterCfg, patterns
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
from omni.isaac.lab_assets.anymal import ANYMAL_C_CFG  # isort: skip
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


@configclass
class AnymalCFlatEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4
    action_scale = 1.0
    num_actions = 7
    num_observations = 43
    num_states = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    # reward scales
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -2.5e-5
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.01
    feet_air_time_reward_scale = 0.5
    undersired_contact_reward_scale = -1.0
    flat_orientation_reward_scale = -5.0


@configclass
class AnymalCRoughEnvCfg(AnymalCFlatEnvCfg):
    # env
    num_observations = 235

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=9,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    # we add a height scanner for perceptive locomotion
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # reward scales (override from flat config)
    flat_orientation_reward_scale = 0.0


class AnymalCEnv(DirectRLEnv):
    cfg: AnymalCFlatEnvCfg | AnymalCRoughEnvCfg

    def __init__(self, cfg: AnymalCFlatEnvCfg | AnymalCRoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self.action_space = Box(
            low=np.array([0.2, 0.2, 0.1, 0.1, -0.2, -0.2, -0.2]),
            high=np.array([2.0, 2.0, 1.0, 1.0, 0.2, 0.2, 0.2]),
        )
        self._previous_actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)

        self.num_neighbors = 2
        self.num_legs = 4
        self.omega = np.pi*2
        self.dt = self.cfg.sim.dt

        self.joint_ids, joint_names = self._robot.find_joints([".*"], preserve_order = True) # ['LF_HAA', 'LH_HAA', 'RF_HAA', 'RH_HAA', 'LF_HFE', 'LH_HFE', 'RF_HFE', 'RH_HFE', 'LF_KFE', 'LH_KFE', 'RF_KFE', 'RH_KFE']
        self.body_ids, body_names = self._robot.find_bodies(['.*'], preserve_order = True) # ['base', 'LF_HIP', 'LH_HIP', 'RF_HIP', 'RH_HIP', 'LF_THIGH', 'LH_THIGH', 'RF_THIGH', 'RH_THIGH', 'LF_SHANK', 'LH_SHANK', 'RF_SHANK', 'RH_SHANK', 'LF_FOOT', 'LH_FOOT', 'RF_FOOT', 'RH_FOOT']

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._commands[:,0] = 1.0

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
            ]
        }
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*FOOT")
        self._underisred_contact_body_ids, _ = self._contact_sensor.find_bodies(".*THIGH")

        # Randomize robot friction
        env_ids = self._robot._ALL_INDICES
        mat_props = self._robot.root_physx_view.get_material_properties()
        mat_props[:, :, :2].uniform_(0.6, 0.8)
        self._robot.root_physx_view.set_material_properties(mat_props, env_ids.cpu())

        # Randomize base mass
        base_id, _ = self._robot.find_bodies("base")
        masses = self._robot.root_physx_view.get_masses()
        masses[:, base_id] += torch.zeros_like(masses[:, base_id]).uniform_(-5.0, 5.0)
        self._robot.root_physx_view.set_masses(masses, env_ids.cpu())

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        if isinstance(self.cfg, AnymalCRoughEnvCfg):
            # we add a height scanner for perceptive locomotion
            self._height_scanner = RayCaster(self.cfg.height_scanner)
            self.scene.sensors["height_scanner"] = self._height_scanner
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)



    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.cfg.action_scale * actions.clone()

    def _apply_action(self) -> None:
        low = torch.tensor(self.action_space.low, dtype=self.actions.dtype, device=self.device)
        high = torch.tensor(self.action_space.high, dtype=self.actions.dtype, device=self.device)

        actions = torch.clamp(self.actions, low, high)
        phases = self._compute_cpg_phases(actions)

        target_positions = self._compute_target_positions(phases) 
        joint_positions_des = torch.zeros((self.num_envs, 12), device = self.device)

        for id in range(self.num_legs):
            foot_id = self.foot_ids[id]
            joint_ids,_ = self.robot.find_joints([self.foot_names[id] + "_thigh_joint", self.foot_names[id] + "_calf_joint"])
    
            trunk_pos_w = self.robot.data.root_state_w[:, :3] # (N, 3)
            trunk_ori_w = self.robot.data.root_state_w[:, 3:7] # (N, 4)
            foot_pose_w = self.robot.data.body_state_w[:, foot_id, 0:7] # (N, 7)

            foot_pos_b, foot_quat_b = subtract_frame_transforms(
                trunk_pos_w, trunk_ori_w, foot_pose_w[:, :3], foot_pose_w[:,3:]
            ) # trunk_pos_b : (N,3), trunk_quad_b : (N,4)

            ik_commands = torch.zeros(self.scene.num_envs, self.diff_ik_controller.action_dim, device=self.robot.device)
            ik_commands[:] = target_positions[:,id, :] + foot_pose_w[:,:3] # (num_envs, 4, 3)

            self.diff_ik_controller.set_command(ik_commands, ee_quat = foot_pose_w[:,3:])

            # Convert target positions to desired joint positions using IK
            joint_positions_des[:, joint_ids] = self._compute_joint_positions_from_targets(foot_id, joint_ids)

        # Set joint positions as the target for the robot
        self.robot.set_joint_position_target(joint_positions_des, joint_ids = self.joint_ids)


    def _compute_cpg_phases(self, actions: torch.Tensor) -> torch.Tensor:
        sigma1 = actions[:, 0]
        sigma2 = actions[:, 1]
        k = actions[:, 2:4]

        foot_ids, _ = self._contact_sensor.find_bodies([".*FOOT"])
        N_V = self._contact_sensor.data.net_forces_w[:, foot_ids, 2]
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
            self.phases[:, i] += phase_dot * self.dt

        self.phases = torch.remainder(self.phases,2*np.pi)
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

    def _compute_target_positions(self, phases: torch.Tensor) -> torch.Tensor:
        # Convert phases to target positions based on the equations from the appendix
        target_positions = torch.zeros((self.cfg.scene.num_envs, self.num_legs, 3), device=self.robot.device)
        phases = torch.remainder(phases, torch.pi*2) 

        for i in range(self.num_legs):
            target_positions[:, i, 0] = self.B * torch.cos(phases[:, i])
            target_positions[:, i, 1] = torch.where((phases[:, i] >= 0) & (phases[:, i] < torch.pi),self.A * torch.sin(phases[:, i]), self.C * torch.sin(phases[:, i]))
        return target_positions

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        height_data = None
        if isinstance(self.cfg, AnymalCRoughEnvCfg):
            height_data = (
                self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
            ).clip(-1.0, 1.0)
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_lin_vel_b,
                    self._robot.data.root_ang_vel_b,
                    self._robot.data.projected_gravity_b,
                    self._commands,
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                    self._robot.data.joint_vel,
                    height_data,
                    self._actions,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # linear velocity tracking
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        # yaw rate tracking
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        # z velocity tracking
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        # angular velocity x/y
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        # feet air time
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (
            torch.norm(self._commands[:, :2], dim=1) > 0.1
        )
        # undersired contacts
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._underisred_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        # flat orientation
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "undesired_contacts": contacts * self.cfg.undersired_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self.phases = torch.randn(self.cfg.scene.num_envs, self.num_legs, device="cuda:0")

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)