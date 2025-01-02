import torch
import numpy as np
from collections.abc import Sequence
from omni.isaac.lab_assets.franka import FRANKA_PANDA_HIGH_PD_CFG
import omni.isaac.lab.sim as sim_utils
from gymnasium.spaces.box import Box
from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import subtract_frame_transforms, quat_mul, quat_from_euler_xyz
from omni.isaac.lab.sensors import CameraCfg, Camera, ContactSensorCfg, ContactSensor
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import sample_uniform
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
import cv2
from omni.isaac.lab.utils.noise.noise_cfg import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg
from omni.isaac.lab.envs import mdp
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from .reward_utils.reward_utils import *
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from pxr import UsdGeom, Gf
import omni.usd
import omni.isaac.lab.utils.math as math_utils

@configclass
class FrankaBaseEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 10.0
    action_scale = 0.5
    num_actions = 6+1 # (x, y, z, roll, pitch, yaw) + gripper

    use_visual_obs = True
    use_visual_marker = True
    num_observations = [3, 64, 64] if use_visual_obs else 27
    table_size = (0.7, 1.2, 1.0)

    # ground plane
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
    )

    # table
    table = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.MeshCuboidCfg(
            size = table_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,  
                kinematic_enabled = True # make table static
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled = True,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0))
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
    )

    # robot
    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    body_offset_pos = (-0.2, 0.0, 1.0)
    body_offset_rot = (1.0, 0.0, 0.0, 0.0)

    # objects
    target = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Target",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.2, 0, 1.0], rot=[0.7071, 0.7071, 0, 0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path = "usd/sugar_box.usd",
            rigid_props=RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=1,
                        max_angular_velocity=1000.0,
                        max_linear_velocity=1000.0,
                        max_depenetration_velocity=5.0,
                        disable_gravity=False,
                    ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled = True,
            ),
            visual_material_path = "sugar_box_texture.png"
        )
    )

    # sensors
    if use_visual_obs:
        camera = CameraCfg(
            prim_path="/World/envs/env_.*/Robot/camera",
            offset=CameraCfg.OffsetCfg(
                pos=(2.7, 0.0, 1.1), 
                rot=(0.0, -0.174, 0.0, 0.985), 
                convention="world"
            ),  
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=386.2455 / 128 * 20.955,  
                focus_distance=400.0, 
                horizontal_aperture=20.955,  
                clipping_range=(0.1, 1.0e5), 
            ),
            data_types=["rgb"], 
            width=num_observations[1], height=num_observations[2],  
        )

    sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", update_period=0.0, history_length=6, debug_vis=True
    )

    sim: SimulationCfg = SimulationCfg(dt=1/100, render_interval=decimation)

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=4, replicate_physics=True)

    # Domain Randomization
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
      noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
      bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
    )

    """
    Code : https://github.com/isaac-sim/IsaacLab/blob/e00d62561a1b4ab19985bc15b40ac77c10bd5e33/docs/source/migration/migrating_from_omniisaacgymenvs.rst#L852
    Methods : https://github.com/isaac-sim/IsaacLab/blob/main/source/extensions/omni.isaac.lab/omni/isaac/lab/envs/mdp/events.py
    """
    events = {
    "robot_joint_stiffness_and_damping": EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    ),
    "reset_gravity": EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="interval",
        is_global_time=True,
        interval_range_s=(36.0, 36.0),
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
            "operation": "add",
            "distribution": "gaussian",
        },
    ),
    "target_mass": EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("target"),
            "operation": "scale",
            "distribution": "uniform",
            "mass_distribution_params": (0.1, 1.0),
        }
    ),
    "table_material": EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("table"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    ),
    }

class FrankaBaseEnv(DirectRLEnv):
    cfg: FrankaBaseEnvCfg

    def __init__(self, cfg: FrankaBaseEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.action_scale = self.cfg.action_scale
        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # Body and Joint ids querying
        self.joint_ids, joint_names = self.robot.find_joints([".*"], preserve_order = True)  # ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7', 'panda_finger_joint1', 'panda_finger_joint2']
        self.body_ids, body_names = self.robot.find_bodies(['.*'], preserve_order = True) # ['panda_link0', 'panda_link1', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link5', 'panda_link6', 'panda_link7', 'panda_hand', 'panda_leftfinger', 'panda_rightfinger']
        self.num_joints = len(self.joint_ids)

        self.offset_pos = torch.tensor(self.cfg.body_offset_pos, device=self.device).repeat(self.num_envs, 1)
        self.offset_rot = torch.tensor(self.cfg.body_offset_rot, device=self.device).repeat(self.num_envs, 1)

        # For easy IDs indexing.
        self.robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
        self.robot_entity_cfg.resolve(self.scene)

        if self.robot.is_fixed_base:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0] - 1
        else:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0]

        self.ee_idx = self.robot_entity_cfg.body_ids[0] # 'panda_hand'
        self.finger_idx, _ = self.robot.find_bodies(['panda_leftfinger', 'panda_rightfinger'], preserve_order = True)
        self.gripper_joint_ids, _ = self.robot.find_joints(['panda_finger_joint1', 'panda_finger_joint2'], preserve_order = True)
        self.undesired_contact_body_ids, _ = self.robot.find_bodies(['panda_link0', 'panda_link1', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link5', 'panda_link6', 'panda_link7'])
        self.gripper_open_position = 0.0
        self.gripper_close_position = 20.0

        # IK solver setting
        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", 
            use_relative_mode=True, # (dx, dy, dz, droll, dpitch, dyaw)
            ik_method="dls",
            ik_params = {
                "pinv": {"k_val": 1.0},
                "svd": {"k_val": 1.0, "min_singular_value": 1e-5},
                "trans": {"k_val": 1.0},
                "dls": {"lambda_val": 0.01},
            },
        )
        self.diff_ik_controller = DifferentialIKController(ik_cfg, num_envs=cfg.scene.num_envs, device=self.scene["robot"].device)
        
        # Intialization
        self.init_tcp = torch.zeros((self.num_envs, 3), device = self.device, dtype = torch.float32)
        self.tcp = torch.zeros((self.num_envs, 3), device = self.device, dtype = torch.float32)

        self.table_pos = self.table.data.root_state_w[:,:3].clone()
        self.on_table_pos = self.table_pos.clone().cpu()
        self.on_table_pos[:, 2:3] += torch.ones((self.num_envs,1))*0.5
        self.target_pos = self.table_pos.clone()

        self.goal = self.table_pos.clone()
        self.init_dist = torch.norm(self.target_pos - self.goal, dim = 1)
        if self.cfg.use_visual_marker:
            self.marker = self._define_markers() # visualize the target position  

        self.tcp_marker = self._define_markers(radius=0.03, diffuse_color=(0.0, 0.0, 1.0))

    def _setup_scene(self):
        # Scene setting : robot, camera, table, walls, YCB Target
        self.table = RigidObject(self.cfg.table)
        self.robot = Articulation(self.cfg.robot)
        if self.cfg.use_visual_obs:
            self.camera = Camera(self.cfg.camera)
        self.sensor = ContactSensor(self.cfg.sensor)
        self.target = RigidObject(self.cfg.target)

        self.scene.articulations['robot'] = self.robot
        if self.cfg.use_visual_obs:
            self.scene.sensors['camera'] = self.camera
        self.scene.sensors['sensor'] = self.sensor
        self.scene.rigid_objects['target'] = self.target
        self.scene.rigid_objects['table'] = self.table

        # Terrain setting
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        processed_actions = actions[:,:-1] * self.action_scale
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        self.diff_ik_controller.set_command(processed_actions, ee_pos_curr, ee_quat_curr)

    def _apply_action(self):
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
        if ee_quat_curr.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self.diff_ik_controller.compute(ee_pos_curr, ee_quat_curr, jacobian, joint_pos)
        else:
            joint_pos_des = joint_pos.clone()
        self.robot.set_joint_position_target(joint_pos_des, self.robot_entity_cfg.joint_ids)

        raw_gripper_action = self.actions[:,-1]
        binary_mask = raw_gripper_action>0
        binary_mask = binary_mask.unsqueeze(1).expand(-1, 2)

        open_command = torch.zeros((self.num_envs, 2), device = self.device)
        open_command[:, 0] = self.gripper_open_position
        open_command[:, 1] = self.gripper_open_position

        close_command = torch.zeros((self.num_envs, 2), device = self.device)
        close_command[:, 0] = self.gripper_close_position
        close_command[:, 1] = self.gripper_close_position

        gripper_action = torch.where(binary_mask, open_command, close_command)
        self.robot.set_joint_position_target(gripper_action, joint_ids = self.gripper_joint_ids)

        marker_orientations = torch.tensor([1, 0, 0, 0],dtype=torch.float32).repeat(self.num_envs,1).to(self.device)  
        marker_indices = torch.zeros((self.num_envs,), dtype=torch.int32)  
        self.tcp_marker.visualize(translations = self.tcp, orientations = marker_orientations, marker_indices = marker_indices)
    
    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pose_w = self.robot.data.body_state_w[:, self.ee_idx, :7]
        root_pose_w = self.robot.data.root_state_w[:, :7]
        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        if self.offset_pos is not None:
            ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
                ee_pose_b, ee_quat_b, self.offset_pos, self.offset_rot
            )

        return ee_pose_b, ee_quat_b

    def _compute_frame_jacobian(self):
        jacobian = self.robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids]
        if self.offset_pos is not None:
            jacobian[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(self.offset_pos), jacobian[:, 3:, :])
            jacobian[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(self.offset_rot), jacobian[:, 3:, :])
        return jacobian

    def _get_observations(self) -> dict:
        self.compute_intermediate()
        if self.cfg.use_visual_obs:
            img = self.camera.data.output['rgb'] #(num_envs, H, W, 4)
            img = img[:,:,:,:-1]
            img = torch.permute(img,(0, 3, 1, 2))
            observations = {"policy": img}
        else:
            joint_pos_rel = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids] - self.robot.data.default_joint_pos[:, self.robot_entity_cfg.joint_ids]
            joint_vel_rel = self.robot.data.joint_vel[:, self.robot_entity_cfg.joint_ids] - self.robot.data.default_joint_vel[:, self.robot_entity_cfg.joint_ids]
            actions = self.actions
            target_pos = self.target_pos
            target_pos_b, _ = subtract_frame_transforms(
                self.robot.data.root_state_w[:, :3], self.robot.data.root_state_w[:, 3:7], target_pos
            )
            goal_pos = self.goal
            goal_pos_b, _ = subtract_frame_transforms(
                self.robot.data.root_state_w[:, :3], self.robot.data.root_state_w[:, 3:7], goal_pos
            )

            obs = torch.cat(
                (
                    joint_pos_rel,
                    joint_vel_rel,
                    actions,
                    target_pos_b,
                    goal_pos_b
                ),
                dim=-1,
            )
            observations = {"policy": torch.clamp(obs, -5.0, 5.0)}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        pass

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        pass
    
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        self.robot.reset(env_ids)
        if self.cfg.use_visual_obs:
            self.camera.reset(env_ids)
        self.sensor.reset(env_ids)
        self.table.reset(env_ids)
        self.target.reset(env_ids)
        self.diff_ik_controller.reset(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.15,
            0.15,
            (len(env_ids), self.num_joints),
            self.device,
        )

        joint_vel = torch.zeros_like(joint_pos)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        spawn_pos = self.update_goal_or_target(offset = self.table_pos.clone().cpu(), which = "target")
        self.target.write_root_state_to_sim(spawn_pos[env_ids,:], env_ids = env_ids)
        self.target_init_pos = spawn_pos[:,:3].clone()

        self.target_pos[env_ids,:] = self.target.data.root_state_w[env_ids,:3].clone()
        self.tcp[env_ids, :] = torch.mean(self.robot.data.body_state_w[:, self.finger_idx, 0:3], dim = 1)[env_ids,:]
        self.init_tcp[env_ids, :] = self.tcp[env_ids, :].clone() # (N, 3)
        
        tcp_to_goal = torch.zeros((len(env_ids),))
        goal_to_target = torch.zeros((len(env_ids),))

        for i, env_id in enumerate(env_ids):
            while True:
                goal_pose = self.update_goal_or_target(offset = self.on_table_pos, which = "goal", dx_range = (0.1, 0.2), dy_range = (-0.1, 0.1), dz_range = (0.05, 0.3))
                tcp_to_goal[i] = torch.norm(goal_pose[env_id,:3] - self.init_tcp[env_id,:])
                goal_to_target[i] = torch.norm(self.target_pos[env_id,:] - goal_pose[env_id,:3])
                if tcp_to_goal[i] > 0.15 and goal_to_target[i] > 0.15:
                    break
            self.goal[env_id,:] = goal_pose[env_id,:3].clone() # destination to arrive        
        self.init_dist[env_ids] = torch.norm(self.target_pos[env_ids,:] - self.goal[env_ids,:], dim = 1)

        self.compute_intermediate()

        marker_locations = self.goal
        marker_orientations = torch.tensor([1, 0, 0, 0],dtype=torch.float32).repeat(self.num_envs,1).to(self.device)  
        marker_indices = torch.zeros((self.num_envs,), dtype=torch.int32)  
        if self.cfg.use_visual_marker:
            self.marker.visualize(translations = marker_locations, orientations = marker_orientations, marker_indices = marker_indices)

        self.tcp_marker.visualize(translations = self.tcp, orientations = marker_orientations, marker_indices = marker_indices)

    def _check_grasp(self, action, obj_pos, obj_radius, pad_success_thresh, xz_thresh, desired_gripper_effort=1.0):
        self.compute_intermediate()
        tcp_pos = self.tcp.clone()
        left_pad = self.robot.data.body_state_w[:, self.finger_idx[0], 0:3]  # (N, 3)
        right_pad = self.robot.data.body_state_w[:, self.finger_idx[1], 0:3]  # (N, 3)

        # Calculate caging in the Y axis
        pad_y_lr = torch.stack((left_pad[:, 1], right_pad[:, 1]), dim=1)  # (N, 2)
        pad_to_obj_lr = torch.abs(pad_y_lr - obj_pos[:, 1].unsqueeze(-1))  # (N, 2)
        caging_lr = [
            tolerance(
                pad_to_obj_lr[:, i],
                bounds=(obj_radius, pad_success_thresh),
                margin=torch.abs(pad_to_obj_lr[:, i] - pad_success_thresh),
                sigmoid="long_tail",
            )
            for i in range(2)
        ]
        caging_y = hamacher_product(*caging_lr)

        # Calculate caging in the XZ plane
        xz = [0, 2]
        diff = torch.norm(tcp_pos[:, xz] - obj_pos[:, xz], dim=1) - xz_thresh
        margin = torch.where(diff>0, diff, 0)
        caging_xz = tolerance(
            torch.norm(tcp_pos[:, xz] - obj_pos[:, xz], dim=1),
            bounds=(0, xz_thresh),
            margin=margin,
            sigmoid="long_tail",
        )

        # Combine caging Y and XZ
        caging = hamacher_product(caging_y, caging_xz)

        # Check if the gripper is closed
        gripper_closed = torch.where(action[:, -1] < 0, desired_gripper_effort, 0.0)  # Gripper effort
        gripper_closed = gripper_closed / desired_gripper_effort

        # Gripping check: caging and gripper closed
        gripping = torch.where(caging > 0.97, gripper_closed, 0.0)

        # Grasp is successful if both caging and gripping are satisfied
        grasp_success = gripping > 0.5  # Threshold for successful grasp

        return grasp_success
    
    def compute_intermediate(self):
        self.target_pos = self.target.data.root_state_w[:,:3].clone()
        self.tcp = torch.mean(self.robot.data.body_state_w[:, self.finger_idx, :3], dim = 1)

    def update_goal_or_target(self, offset=None, which="target", dx_range=None, dy_range=None, dz_range=None):
        if offset is None:
            offset = torch.zeros((self.num_envs, 3), device=self.device)

        if which == "target":
            # x_half = self.cfg.table_size[0]/2 - 0.1
            # y_half = self.cfg.table_size[1]/2 - 0.1
            x_half = 0.1
            y_half = 0.1
            x = torch.rand(self.num_envs, 1) * 2*x_half - x_half
            y = torch.rand(self.num_envs, 1) * 2*y_half - y_half
            z = torch.ones((self.num_envs, 1)) * 0.5 

            quaternion = torch.tensor([0.7071, 0.0, 0.7071, 0.0]).repeat(self.num_envs, 1)
            translational_velocity = torch.zeros((self.num_envs, 3))
            rotational_velocity = torch.zeros((self.num_envs, 3))

            target_pos = torch.cat((x, y, z), dim=1) + offset
            combined_tensor = torch.cat((target_pos, quaternion, translational_velocity, rotational_velocity), dim=1)
            return combined_tensor.to(self.device)

        else:
            if dx_range is None:
                dx = torch.rand(self.num_envs, 1) * 0.1
            elif isinstance(dx_range, tuple):
                dx = torch.rand(self.num_envs, 1) * (dx_range[1] - dx_range[0]) + dx_range[0]
            else:  # Fixed value
                dx = torch.ones(self.num_envs, 1) * dx_range

            if dy_range is None:
                dy = (torch.rand(self.num_envs, 1) * 0.15 + 0.15) * (torch.randint(0, 2, (self.num_envs, 1)) * 2 - 1)
            elif isinstance(dy_range, tuple):
                dy = torch.rand(self.num_envs, 1) * (dy_range[1] - dy_range[0]) + dy_range[0]
            else:  # Fixed value
                dy = torch.ones(self.num_envs, 1) * dy_range

            if dz_range is None:
                dz = torch.rand(self.num_envs, 1) * 0.1
            elif isinstance(dz_range, tuple):
                dz = torch.rand(self.num_envs, 1) * (dz_range[1] - dz_range[0]) + dz_range[0]
            else:  # Fixed value
                dz = torch.ones(self.num_envs, 1) * dz_range

            x = offset[:, 0].unsqueeze(1) + dx
            y = offset[:, 1].unsqueeze(1) + dy
            z = offset[:, 2].unsqueeze(1) + dz

            output = torch.cat((x, y, z), dim=1).to(self.device)
            output[:, 0] = torch.clamp(output[:, 0], self.table_pos[:, 0] - self.cfg.table_size[0]/2, self.table_pos[:, 0] + self.cfg.table_size[0]/2)
            output[:, 1] = torch.clamp(output[:, 1], self.table_pos[:, 1] - self.cfg.table_size[1]/2, self.table_pos[:, 1] + self.cfg.table_size[1]/2)
            output[:, 2] = torch.clamp(output[:, 2], 1.0, 2.0)
            return output
        
    def _tcp_to_target(self, return_dist = False):
        self.compute_intermediate()
        tcp_pos = self.tcp.clone()
        target_pos = self.target_pos.clone()
        diff = target_pos - tcp_pos
        return torch.norm(diff, dim = 1) if return_dist else diff

    def _define_markers(self, radius = 0.05, diffuse_color = (1.0, 0.0, 0.0)) -> VisualizationMarkers:
        """Define markers to visualize the target position."""
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/TargetMarkers",
            markers={
                "target": sim_utils.SphereCfg(  
                    radius=radius,  
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=diffuse_color), 
                ),
            },
        )
        return VisualizationMarkers(marker_cfg)
