import gymnasium as gym

from . import agents
from .unitree_env import UnitreeA1Env, UnitreeA1EnvCfg

##
# Register Gym environments.
##
gym.register(
    id="Isaac-Unitree-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.unitree:UnitreeA1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": UnitreeA1EnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point" : f"{agents.__name__}:sb3_sac_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeFlatPPORunnerCfg"
    },
)

