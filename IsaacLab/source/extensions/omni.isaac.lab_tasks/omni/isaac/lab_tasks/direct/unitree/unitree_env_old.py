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
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg, mdp
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import subtract_frame_transforms,sample_uniform
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from gymnasium.spaces.box import Box
from omni.isaac.lab.managers import SceneEntityCfg
import omni.isaac.core.utils.torch as torch_utils
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.sensors import ContactSensorCfg, ContactSensor,RayCasterCfg, RayCaster, patterns
from scipy.integrate import solve_ivp

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



@configclass
class UnitreeA1EnvCfg(DirectRLEnvCfg):
    # env  episode_length_s = dt * decimation * num_steps
    decimation = 2
    episode_length_s = 10.0
    action_scale = 1.0  # Scale for the CPG parameters
    num_actions = 12
    num_observations = 16  # Customize based on observations

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1/100, render_interval=decimation)

    scene : InteractiveSceneCfg = EnvScene(num_envs = 64, env_spacing = 4.0)

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

    

class UnitreeA1Env(DirectRLEnv):
    cfg: UnitreeA1EnvCfg

    def __init__(self, cfg: UnitreeA1EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.action_scale = self.cfg.action_scale
        self.action_space = Box(
            low=np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.5, -1.5, -1.5, -1.5]), # mu_1, mu_2, mu_3, mu_4, omega_1, omega_2, omega_3, omega_4, psi_1, psi_2, psi_3, psi_4
            high=np.array([2.0, 2.0, 2.0, 2.0, 4.5, 4.5, 4.5, 4.5, 1.5, 1.5, 1.5, 1.5]),
        )

        self.dt = self.cfg.sim.dt
        self.num_legs = 4
        self.d_step = 0.15 # m
        self.h = 0.4 # m
        self.g_c = 0.1 # m
        self.g_p = 0.01 # m
        self.y0 = np.zeros((self.num_envs, 4 * self.num_legs))        

        self.joint_ids, _ = self.robot.find_joints(['.*_joint']) # ['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint']
        self.body_ids, _ = self.robot.find_bodies(['.*']) # ['trunk', 'FL_hip', 'FR_hip', 'RL_hip', 'RR_hip', 'FL_thigh', 'FR_thigh', 'RL_thigh', 'RR_thigh', 'FL_calf', 'FR_calf', 'RL_calf', 'RR_calf', 'FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
        
        self.foot_names = ["FL","FR","RL","RR"]
        self.foot_ids, _ = self.robot.find_bodies([".*_foot"]) # ["FL","FR","RL","RR"]

        self.base_id, _ = self.robot.find_bodies("trunk")

        ik_cfg = DifferentialIKControllerCfg(
            command_type="position",
            use_relative_mode=False, 
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
        target_positions = self.foot_position(actions)
        joint_positions_des = torch.zeros((self.num_envs, 12), device = self.device)

        trunk_pos_w = self.robot.data.root_state_w[:, :3] # (N, 3)
        trunk_ori_w = self.robot.data.root_state_w[:, 3:7] # (N, 4)

        for id in range(self.num_legs):
            foot_id = self.foot_ids[id]
            joint_ids,_ = self.robot.find_joints([self.foot_names[id] + "_calf_joint",self.foot_names[id] + "_thigh_joint", self.foot_names[id] + "_hip_joint"])
            foot_pose_w = self.robot.data.body_state_w[:, foot_id, 0:7] # (N, 7)

            foot_pos_b, foot_quat_b = subtract_frame_transforms(
                trunk_pos_w, trunk_ori_w, foot_pose_w[:, :3], foot_pose_w[:,3:]
            ) # trunk_pos_b : (N,3), trunk_quad_b : (N,4)

            ik_commands = torch.zeros(self.scene.num_envs, self.diff_ik_controller.action_dim, device=self.robot.device)
            ik_commands[:] = target_positions[:,id, :] # (num_envs, 4, 3)

            self.diff_ik_controller.set_command(ik_commands, ee_pos = foot_pos_b, ee_quat = foot_quat_b)

            # Convert target positions to desired joint positions using IK
            joint_positions_des[:, joint_ids] = self._compute_joint_positions_from_targets(foot_id, joint_ids, foot_pos_b, foot_quat_b)

        # Set joint positions as the target for the robot
        self.robot.set_joint_position_target(joint_positions_des, joint_ids = self.joint_ids)

    def foot_position(self, actions):
        num_envs = self.num_envs
        alpha = 150  
        n_oscillators = self.num_legs

        def oscillator_dynamics(t, y):
            dydt = np.zeros_like(y)
            for env in range(num_envs):
                for i in range(n_oscillators):
                    r_i, r_i_dot, theta_i, phi_i = y[env, 4*i:4*i+4]
                    mu_i = actions[env, i]
                    omega_i = actions[env, 4+i]
                    psi_i = actions[env, 8+i]

                    r_i_ddot = alpha * (alpha / 4 * (mu_i - r_i) - r_i_dot)
                    
                    dydt[env, 4*i] = r_i_dot  # dr_i/dt
                    dydt[env, 4*i+1] = r_i_ddot  # d^2r_i/dt^2
                    dydt[env, 4*i+2] = omega_i  # dtheta_i/dt
                    dydt[env, 4*i+3] = psi_i  # dphi_i/dt
            return dydt.flatten()  
        
        y0 = self.y0
        t_span = (self.sim.current_time, self.sim.current_time+self.dt)

        t_eval = np.linspace(t_span[0], t_span[1], 100)    

        sol = solve_ivp(
            lambda t, y: oscillator_dynamics(t, y.reshape(num_envs, 4 * n_oscillators)),
            t_span, y0.flatten(), t_eval=t_eval
        )

        final_values = sol.y[:, -1].reshape(num_envs, 4 * n_oscillators)        
        self.y0 = final_values

        positions = []
        for env in range(num_envs):
            env_positions = []
            for i in range(n_oscillators):
                r_i, _, theta_i, phi_i = final_values[env, 4*i:4*i+4]                
                x_i = -self.d_step * (r_i - 1) * np.cos(theta_i) * np.cos(phi_i)
                y_i = -self.d_step * (r_i - 1) * np.cos(theta_i) * np.sin(phi_i)
                if np.sin(theta_i) > 0:
                    z_i = -self.h + self.g_c * np.sin(theta_i)
                else:
                    z_i = -self.h + self.g_p * np.sin(theta_i)
                
                env_positions.append([x_i, y_i, z_i])
            positions.append(env_positions)
        
        return torch.tensor(np.array(positions), device = self.device)
    
    def _compute_joint_positions_from_targets(self, foot_id, joint_ids, foot_pos_b, foot_quat_b) -> torch.Tensor:
        current_joint_pos = self.robot.data.joint_pos[:, joint_ids] # self.robot.data.joint_pose : (N, num_joints) , current_joint_pos : (N, 3, 1)
        jacobian = self.robot.root_physx_view.get_jacobians()[:, foot_id, :, joint_ids] # (N, num_frames, 6, num_joints) -> (N, 17, 6, 18) : 12 for joints and 6 for (linear + angluar) velocity
        # jacobian : (N, 6, 3, 1)
        return self.diff_ik_controller.compute(foot_pos_b, foot_quat_b, jacobian, current_joint_pos)
    
    def _get_observations(self) -> dict:
        obs = torch.tensor(self.y0,dtype=torch.float)
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        torque = self.robot.data.applied_torque
        joint_ang_vel = self.robot.data.joint_vel
        vel = self.robot.data.root_vel_w
        lin_vel_x = vel[:, 0].unsqueeze(-1)
        lin_vel_y = vel[:, 1].unsqueeze(-1)
        lin_vel_z = vel[:, 2].unsqueeze(-1)
        ang_vel_xy = vel[:, 3:5]
        ang_vel_z = vel[:, -1].unsqueeze(-1)

        total_reward = 0.0

        def f(x):
            # assume that x is shape of (num_envs, 1)
            return torch.exp(-torch.norm(x, dim =1)**2/0.25)
        
        def joint_power(torque, joint_ang_vel):
            power = 0
            for i in range(torque.shape[0]):
                power += torch.dot(torque[i], joint_ang_vel[i])
            return power
        
        total_reward += 0.75*self.dt*f(self.command[:,0] - lin_vel_x) + 0.75*self.dt*f(self.command[:,1] - lin_vel_y) + 0.5*self.dt*f(self.command[:,2] - ang_vel_z) - 2*self.dt*torch.norm(lin_vel_z, dim=1) - 0.05*self.dt*torch.norm(ang_vel_xy, dim =1) 
        total_reward += 0.001*self.dt*joint_power(torque, joint_ang_vel)
        return total_reward


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self.sensors.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self.base_id], dim=-1), dim=1)[0] > 1e-3, dim=1) 
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.phases = torch.randn(self.cfg.scene.num_envs, self.num_legs, device="cuda:0")
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

        self.command = 2*torch.rand((self.num_envs, 3))-1
        self.command = self.command.to(self.device)
