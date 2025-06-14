# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the leatherback robot."""

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# Define a base configuration that only contains the actuators.
# The spawn and init_state info is now handled by the main USD file.
BASE_LEATHERBACK_CFG = ArticulationCfg(
    actuators={
        "throttle": ImplicitActuatorCfg(
            joint_names_expr=["Wheel.*"],
            effort_limit=40000.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=100000.0,
        ),
        "steering": ImplicitActuatorCfg(
            joint_names_expr=["Knuckle__Upright__Front.*"],
            effort_limit=40000.0,
            velocity_limit=100.0,
            stiffness=1000.0,
            damping=0.0,
        ),
    },
)

# Create copies for each robot. This is good practice in case you
# want to give them different properties later.
LEATHERBACK_1_CFG = BASE_LEATHERBACK_CFG.copy()
LEATHERBACK_2_CFG = BASE_LEATHERBACK_CFG.copy()
