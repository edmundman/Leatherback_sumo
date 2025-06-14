# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
All Gymnasium environments are registered here.
"""

import gymnasium as gym

# --- Register the original waypoint task ---
gym.register(
    id="Template-Leatherback-Direct-v0",
    entry_point="Leatherback.tasks.direct.leatherback.leatherback_env:LeatherbackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "Leatherback.tasks.direct.leatherback.leatherback_env:LeatherbackEnvCfg",
        "skrl_cfg_entry_point": "Leatherback.tasks.direct.leatherback.agents:skrl_ppo_cfg.yaml",
    },
)

# --- Register the new sumo task ---
gym.register(
    id="Isaac-Leatherback-Sumo-v0",
    entry_point="Leatherback.tasks.direct.sumo.sumo_env:LeatherbackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "Leatherback.tasks.direct.sumo.sumo_env:LeatherbackSumoEnvCfg",
        "skrl_ippo_cfg_entry_point": "Leatherback.tasks.direct.sumo.agents:skrl_ippo_cfg.yaml",
    },
)