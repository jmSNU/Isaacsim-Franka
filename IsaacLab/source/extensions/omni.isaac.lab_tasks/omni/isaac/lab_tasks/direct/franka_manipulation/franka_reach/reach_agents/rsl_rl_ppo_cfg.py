# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class UR5PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 500
    save_interval = 50
    experiment_name = "ur5-direct"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef= 0.3999,
        use_clipped_value_loss=True,
        clip_param= 0.2865,
        entropy_coef=0.0548,
        num_learning_epochs=20,
        num_mini_batches=193,
        learning_rate= 0.0153,
        schedule="none",
        gamma=0.9657,
        lam=0.8543,
        desired_kl=0.01,
        max_grad_norm=9.4229,
    )

    logger = "wandb"


