import gymnasium as gym
from . import franka_manipulation
from .franka_reach.franka_reach_env import FrankaReachEnv, FrankaReachEnvCfg
from .franka_reach import reach_agents
from .franka_push.franka_push_env import FrankaPushEnv, FrankaPushEnvCfg
from .franka_push import push_agents
from .franka_pick.franka_pick_env import FrankaPickEnv, FrankaPickEnvCfg
from .franka_pick import pick_agents

##
# Register Gym environments.
##
gym.register(
    id="Isaac-Franka-Reach-v0",
    entry_point="omni.isaac.lab_tasks.direct.franka_manipulation.franka_reach.franka_reach_env:FrankaReachEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaReachEnvCfg,
        "rl_games_cfg_entry_point": f"{reach_agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point" : f"{reach_agents.__name__}:sb3_sac_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{reach_agents.__name__}.rsl_rl_ppo_cfg:frankaPPORunnerCfg"
    },
)

gym.register(
    id="Isaac-Franka-Push-v0",
    entry_point="omni.isaac.lab_tasks.direct.franka_manipulation.franka_push.franka_push_env:FrankaPushEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaPushEnvCfg,
        "rl_games_cfg_entry_point": f"{push_agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point" : f"{push_agents.__name__}:sb3_sac_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{push_agents.__name__}.rsl_rl_ppo_cfg:frankaPPORunnerCfg"
    },
)

gym.register(
    id="Isaac-Franka-Pick-v0",
    entry_point="omni.isaac.lab_tasks.direct.franka_manipulation.franka_pick.franka_pick_env:FrankaPickEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaPickEnvCfg,
        "rl_games_cfg_entry_point": f"{pick_agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point" : f"{pick_agents.__name__}:sb3_sac_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{pick_agents.__name__}.rsl_rl_ppo_cfg:frankaPPORunnerCfg"
    },
)


