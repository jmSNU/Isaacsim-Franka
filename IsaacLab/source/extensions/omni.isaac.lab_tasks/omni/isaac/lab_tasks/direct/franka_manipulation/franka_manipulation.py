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


@configclass
class FrankaBaseEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 10.0
    action_scale = 0.5
    num_actions = 7+1 # (x, y, z, qw, qx, qy, qz) + gripper
    num_observations = [3, 64, 64] # rgb image
    enable_obstacle = False
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

    # objects
    target = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
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

    obstacle = RigidObjectCfg(
        prim_path ="/World/envs/env_.*/Obstacle",
        init_state = RigidObjectCfg.InitialStateCfg(pos=[0.2, 0, 1.0], rot=[0.7071, 0, 0.7071, 0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path = "usd/cracker_box.usd",
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
            visual_material_path = "cracker_box_texture.png"
        )
    )

    # sensors
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

        # For easy IDs indexing.
        self.robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["panda_hand"])
        self.robot_entity_cfg.resolve(self.scene)

        if self.robot.is_fixed_base:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0] - 1
        else:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0]

        self.ee_idx = self.robot_entity_cfg.body_ids[0] # 'panda_hand'
        self.finger_idx, _ = self.robot.find_bodies(['panda_leftfinger', 'panda_rightfinger'], preserve_order = True)
        self.pos_joint_ids, _ = self.robot.find_joints(['panda_joint1', 'panda_joint2', 'panda_joint3'], preserve_order = True)
        self.rot_joint_ids, _ = self.robot.find_joints(['panda_joint5', 'panda_joint6', 'panda_joint7'], preserve_order = True)
        self.gripper_joint_ids, _ = self.robot.find_joints(['panda_finger_joint1', 'panda_finger_joint2'], preserve_order = True)
        self.undesired_contact_body_ids, _ = self.robot.find_bodies(['panda_link0', 'panda_link1', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link5', 'panda_link6', 'panda_link7'])
        self.gripper_open_position = 0.0
        self.gripper_close_position = 20.0

        # IK solver setting
        ik_cfg = DifferentialIKControllerCfg(
            command_type="position", 
            use_relative_mode=False, # (x, y, z)
            ik_method="dls",
            ik_params = {
                "pinv": {"k_val": 1.0},
                "svd": {"k_val": 1.0, "min_singular_value": 1e-5},
                "trans": {"k_val": 1.0},
                "dls": {"lambda_val": 0.01},
            },
        )
        self.diff_ik_controller_position = DifferentialIKController(ik_cfg, num_envs=cfg.scene.num_envs, device=self.scene["robot"].device)

        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", 
            use_relative_mode=False, # (x, y, z, qw, qx, qy, qz)
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
        self.marker = self._define_markers() # visualize the target position  

    def _setup_scene(self):
        # Scene setting : robot, camera, table, walls, YCB Target
        self.table = RigidObject(self.cfg.table)
        self.robot = Articulation(self.cfg.robot)
        self.camera = Camera(self.cfg.camera)
        self.sensor = ContactSensor(self.cfg.sensor)
        self.target = RigidObject(self.cfg.target)
        if self.cfg.enable_obstacle:
            self.obstacle = RigidObject(self.cfg.obstacle)

        self.scene.articulations['robot'] = self.robot
        self.scene.sensors['camera'] = self.camera
        self.scene.sensors['sensor'] = self.sensor
        self.scene.rigid_objects['target'] = self.target
        if self.cfg.enable_obstacle:
            self.scene.rigid_objects['obstacle'] = self.obstacle
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
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        # clipping action
        # pos_low = torch.tensor(np.array([-0.1,-0.2,-0.2]), device = self.device)
        # pos_high = torch.tensor(np.array([0.2,0.2,0.2]), device = self.device)
        # roll_range = torch.tensor(np.array([-np.pi*5/180, np.pi*5/180]), device = self.device)
        # pitch_range = torch.tensor(np.array([-np.pi*5/180, np.pi*5/180]), device = self.device)
        # yaw_range = torch.tensor(np.array([-np.pi*5/180, np.pi*5/180]), device = self.device)
        # quat_low = quat_from_euler_xyz(roll_range[0], pitch_range[0], yaw_range[0])
        # quat_high = quat_from_euler_xyz(roll_range[1], pitch_range[1], yaw_range[1])

        # clipping ee position
        # low = self.table_pos + torch.tensor(np.array([-self.cfg.table_size[0]/2, -self.cfg.table_size[1]/2, 0.5]), device = self.device)
        # high = self.table_pos + torch.tensor(np.array([self.cfg.table_size[0]/2, self.cfg.table_size[1]/2, 1.5]), device = self.device)
        dummy_pos = torch.zeros(self.num_envs, 3, device = self.device, dtype = torch.float32)

        ee_pose_w = self.robot.data.body_state_w[:, self.ee_idx, 0:7]
        root_pose_w = self.robot.data.root_state_w[:, 0:7]

        # joint_action_pos = torch.clamp(self.actions[:,:3], pos_low, pos_high) + ee_pose_w[:,:3]   
        joint_action_pos = self.actions[:,:3]
        # joint_action_pos = torch.clamp(joint_action_pos, low, high)     
        joint_action_pos = torch.as_tensor(joint_action_pos, dtype = torch.float32, device = self.device)

        # joint_action_quat = quat_mul(torch.clamp(self.actions[:,3:-1], quat_low, quat_high), ee_pose_w[:,3:])
        joint_action_quat = self.actions[:,3:-1]
        joint_action_quat = joint_action_quat/torch.norm(joint_action_quat, dim = 1, keepdim = True)
        joint_action_quat = torch.as_tensor(joint_action_quat, dtype = torch.float32, device = self.device)

        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], joint_action_pos, joint_action_quat # change the coordinate from world to robot base
        )

        ik_commands = torch.zeros(self.num_envs, self.diff_ik_controller_position.action_dim, device=self.device)
        ik_commands[:] = ee_pos_b
        self.diff_ik_controller_position.set_command(ik_commands, ee_quat=ee_quat_b)

        ik_commands = torch.zeros(self.num_envs, self.diff_ik_controller.action_dim, device = self.device)
        ik_commands[:] = torch.cat([dummy_pos, ee_quat_b], dim = 1)
        self.diff_ik_controller.set_command(ik_commands)

        # Calculate joint_position from IK solver
        joint_positions = self._compute_joint_positions_from_targets(self.pos_joint_ids, self.diff_ik_controller_position)
        joint_orientation = self._compute_joint_positions_from_targets(self.rot_joint_ids, self.diff_ik_controller)

        # Apply actions to the robot
        self.robot.set_joint_position_target(joint_positions, joint_ids = self.pos_joint_ids)    
        self.robot.set_joint_position_target(joint_orientation, joint_ids = self.rot_joint_ids)

        # Gripper Action
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

    def _compute_joint_positions_from_targets(self,joint_ids, ik_controller) -> torch.Tensor:
        """Calculate the joint position by using IK solver. Refer to https://isaac-sim.github.io/IsaacLab/source/tutorials/05_controllers/run_diff_ik.html"""
        jacobian = self.robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, joint_ids]
        ee_pose_w = self.robot.data.body_state_w[:, self.ee_idx, 0:7]
        root_pose_w = self.robot.data.root_state_w[:, 0:7]
        joint_pos = self.robot.data.joint_pos[:, joint_ids]

        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        joint_pos_des = ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        return joint_pos_des

    def _get_observations(self) -> dict:
        img = self.camera.data.output['rgb'] #(num_envs, H, W, 4)
        img = img[:,:,:,:-1]
        observations = {"policy": img}
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
        self.camera.reset(env_ids)
        self.sensor.reset(env_ids)
        self.table.reset(env_ids)
        self.target.reset(env_ids)
        self.diff_ik_controller.reset(env_ids)
        self.diff_ik_controller_position.reset(env_ids)

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
                goal_pose = self.update_goal_or_target(offset = self.on_table_pos, which = "goal",dx_range = (0.1, 0.2), dy_range = (-0.1, 0.1), dz_range = (0.05, 0.3))
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
        self.marker.visualize(translations = marker_locations, orientations = marker_orientations, marker_indices = marker_indices)

    """ Reference : https://github.com/Farama-Foundation/Metaworld/blob/cca35cff0ec62f1a18b11440de6b09e2d10a1380/metaworld/envs/mujoco/sawyer_xyz/sawyer_xyz_env.py#L699 """
    def _gripper_grasp_reward(self, action, obj_pos, obj_radius, pad_success_thresh, obj_reach_radius, xz_thresh, desired_gripper_effort = 1.0, high_density = False, medium_density = False):
        if high_density and medium_density:
            raise ValueError("Can only be either high_density or medium_density")
        
        tcp_pos = torch.mean(self.robot.data.body_state_w[:, self.finger_idx, 0:3], dim = 1) # (N, 3)
        left_pad = self.robot.data.body_state_w[:, self.finger_idx[0], 0:3] # (N, 3)
        rigth_pad = self.robot.data.body_state_w[:, self.finger_idx[1], 0:3] # (N, 3)

        pad_y_lr = torch.stack((left_pad[:, 1], rigth_pad[:, 1]), dim = 1) # (N, 2)
        pad_to_obj_lr = torch.abs(pad_y_lr - obj_pos[:, 1].unsqueeze(-1)) #(N, 2)
        pad_to_objinit_lr = torch.abs(pad_y_lr - self.target_init_pos[:,1].unsqueeze(-1))

        caging_lr_margin = torch.abs(pad_to_objinit_lr - pad_success_thresh)
        caging_lr = [
            tolerance(
                pad_to_obj_lr[:, i],
                bounds = (obj_radius, pad_success_thresh),
                margin = caging_lr_margin[:, i],
                sigmoid="long_tail"
            )
            for i in range(2)
        ]

        caging_y = hamacher_product(*caging_lr)

        xz = [0,2]
        caging_xz_margin = torch.norm(self.target_init_pos[:,xz] - self.init_tcp[:, xz], dim = 1)
        caging_xz_margin -= xz_thresh
        caging_xz = tolerance(
            torch.norm(tcp_pos[:, xz] - obj_pos[:, xz]),
            bounds = (0, xz_thresh),
            margin = caging_xz_margin,
            sigmoid="long_tail"
        )
        
        gripper_closed = torch.where(action[:, -1] < 0, desired_gripper_effort, 0.0)  
        gripper_closed = gripper_closed / desired_gripper_effort
        caging = hamacher_product(caging_y, caging_xz)
        gripping = torch.where(caging>0.97, gripper_closed, 0.0) 
        caging_and_gripping = hamacher_product(caging, gripping)

        if high_density:
            caging_and_gripping = (caging_and_gripping + caging) / 2
        if medium_density:
            tcp_to_obj = torch.norm(obj_pos - tcp_pos, dim = 1)
            tcp_to_obj_init = torch.norm(self.target_init_pos - self.init_tcp, dim = 1)
            reach_margin = torch.abs(tcp_to_obj_init - obj_reach_radius)
            reach = tolerance(
                tcp_to_obj,
                bounds=(0, obj_reach_radius),
                margin=reach_margin,
                sigmoid="long_tail"
            )
            caging_and_gripping = (caging_and_gripping + reach.to(torch.float32)) / 2

        return caging_and_gripping
    
    def compute_intermediate(self):
        self.target_pos = self.target.data.root_state_w[:,:3].clone()
        self.tcp = torch.mean(self.robot.data.body_state_w[:, self.finger_idx, :3], dim = 1)

    def update_goal_or_target(self, offset=None, which="target", dx_range=None, dy_range=None, dz_range=None):
        if offset is None:
            offset = torch.zeros((self.num_envs, 3), device=self.device)

        if which == "target":
            x_half = self.cfg.table_size[0]/2 - 0.05
            y_half = self.cfg.table_size[1]/2 - 0.05
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
    
    def _define_markers(self) -> VisualizationMarkers:
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
