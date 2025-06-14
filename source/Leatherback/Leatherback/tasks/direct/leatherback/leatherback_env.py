from __future__ import annotations

import torch
from collections.abc import Sequence
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .leatherback import LEATHERBACK_1_CFG, LEATHERBACK_2_CFG

@configclass
class LeatherbackSumoEnvCfg(DirectMARLEnvCfg):
    # Path to the pre-built USD stage for the sumo environment
    # This path is relative to the leatherback_env.py file
    usd_path = "custom_assets/sumo_leatherback.usd"

    decimation = 4
    episode_length_s = 30.0
    action_space = 2
    observation_space = 10 # rel_pos (2), rel_heading (2), own_vel (3), opponent_vel (3) = 10
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=decimation)

    # Define agent groups for the two robots
    agent_groups = ["leatherback_1", "leatherback_2"]
    # Assign the articulation assets to the agent groups
    asset_groups = {"leatherback_1": ["robot_1"], "leatherback_2": ["robot_2"]}

    # Define the articulation assets by pointing to prims in the USD
    robot_1: ArticulationCfg = LEATHERBACK_1_CFG.replace(prim_path="/World/envs/env_.*/leatherback")
    robot_2: ArticulationCfg = LEATHERBACK_2_CFG.replace(prim_path="/World/envs/env_.*/leatherback2")

    # Define the dohyo border prim path from your USD
    dohyo_border_prim_path = "/World/envs/env_.*/Dohyo/WhiteBorder"

    # Scene configuration
    env_spacing = 5.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024, env_spacing=env_spacing, replicate_physics=True
    )


class LeatherbackEnv(DirectMARLEnv):
    cfg: LeatherbackSumoEnvCfg

    def __init__(self, cfg: LeatherbackSumoEnvCfg, render_mode: str | None = None, **kwargs):
        # Resolve the full path to the USD file
        self.cfg.usd_path = os.path.join(os.path.dirname(__file__), self.cfg.usd_path)
        super().__init__(cfg, render_mode, **kwargs)

        # Agent-specific information
        self.robot1_map = self.agent_manager.get_agent_indices("leatherback_1")
        self.robot2_map = self.agent_manager.get_agent_indices("leatherback_2")

        # Environment parameters
        self.dohyo_radius = 1.5  # Adjust if your dohyo radius is different
        self.win_reward = 100.0
        self.lose_penalty = -100.0
        self.push_reward = 1.0
        self.alive_reward = 0.1

    def _setup_scene(self):
        # -- Load the entire stage from the specified USD file
        sim_utils.open_usd(self.cfg.usd_path)

        # -- Get handles to the assets defined in the USD
        self.robot_1 = Articulation(self.cfg.robot_1)
        self.robot_2 = Articulation(self.cfg.robot_2)
        self.dohyo_border = RigidObject(
            RigidObjectCfg(prim_path=self.cfg.dohyo_border_prim_path)
        )

        # -- Clone environments and add assets to the scene
        # We set copy_from_source=True because the source (/World/Dohyo, /World/leatherback, etc.)
        # now comes from the loaded USD file.
        self.scene.clone_environments(copy_from_source=True)
        self.scene.articulations["robot_1"] = self.robot_1
        self.scene.articulations["robot_2"] = self.robot_2
        self.scene.rigid_objects["dohyo_border"] = self.dohyo_border

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]):
        # Actions are dictionaries with agent groups as keys
        actions_robot1 = actions["leatherback_1"]
        actions_robot2 = actions["leatherback_2"]

        # Apply actions to robot 1
        self.robot_1.set_joint_velocity_target(actions_robot1[:, 0].repeat(1,4), joint_regex="Wheel.*")
        self.robot_1.set_joint_position_target(actions_robot1[:, 1].repeat(1,2), joint_regex="Knuckle.*")

        # Apply actions to robot 2
        self.robot_2.set_joint_velocity_target(actions_robot2[:, 0].repeat(1,4), joint_regex="Wheel.*")
        self.robot_2.set_joint_position_target(actions_robot2[:, 1].repeat(1,2), joint_regex="Knuckle.*")

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        # Observations for Robot 1
        pos_1 = self.robot_1.data.root_pos_w
        pos_2 = self.robot_2.data.root_pos_w
        relative_pos_to_2 = pos_2 - pos_1
        obs_robot1 = torch.cat([
            relative_pos_to_2[:, :2],
            torch.cos(self.robot_1.data.heading_w).unsqueeze(1),
            torch.sin(self.robot_1.data.heading_w).unsqueeze(1),
            self.robot_1.data.root_lin_vel_b,
            self.robot_2.data.root_lin_vel_b
        ], dim=-1)

        # Observations for Robot 2
        relative_pos_to_1 = pos_1 - pos_2
        obs_robot2 = torch.cat([
            relative_pos_to_1[:, :2],
            torch.cos(self.robot_2.data.heading_w).unsqueeze(1),
            torch.sin(self.robot_2.data.heading_w).unsqueeze(1),
            self.robot_2.data.root_lin_vel_b,
            self.robot_1.data.root_lin_vel_b
        ], dim=-1)

        return {
            "leatherback_1": {"policy": obs_robot1},
            "leatherback_2": {"policy": obs_robot2},
        }

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        pos_1 = self.robot_1.data.root_pos_w
        pos_2 = self.robot_2.data.root_pos_w
        dist_from_center_1 = torch.norm(pos_1[:, :2] - self.scene.env_origins[:,:2], dim=1)
        dist_from_center_2 = torch.norm(pos_2[:, :2] - self.scene.env_origins[:,:2], dim=1)

        robot1_out = (dist_from_center_1 > self.dohyo_radius)
        robot2_out = (dist_from_center_2 > self.dohyo_radius)

        rewards_1 = torch.full_like(dist_from_center_1, self.alive_reward)
        rewards_2 = torch.full_like(dist_from_center_2, self.alive_reward)

        rewards_1 += torch.where(robot2_out & ~robot1_out, self.win_reward, 0.0)
        rewards_2 += torch.where(robot1_out & ~robot2_out, self.win_reward, 0.0)
        rewards_1 += torch.where(robot1_out, self.lose_penalty, 0.0)
        rewards_2 += torch.where(robot2_out, self.lose_penalty, 0.0)

        rewards_1 += self.push_reward * (dist_from_center_2 / self.dohyo_radius)
        rewards_2 += self.push_reward * (dist_from_center_1 / self.dohyo_radius)

        return {"leatherback_1": rewards_1, "leatherback_2": rewards_2}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos_1 = self.robot_1.data.root_pos_w
        pos_2 = self.robot_2.data.root_pos_w
        dist_from_center_1 = torch.norm(pos_1[:, :2] - self.scene.env_origins[:,:2], dim=1)
        dist_from_center_2 = torch.norm(pos_2[:, :2] - self.scene.env_origins[:,:2], dim=1)

        robot1_out = (dist_from_center_1 > self.dohyo_radius)
        robot2_out = (dist_from_center_2 > self.dohyo_radius)
        time_out = self.episode_length_buf >= self.max_episode_length

        dones = robot1_out | robot2_out | time_out
        return dones, dones

    def _reset_idx(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        
        # Define initial root states for reset
        # Robot 1 starts at (0, 0.5, z)
        root_state_1 = self.robot_1.data.default_root_state[env_ids]
        root_state_1[:, 1] = 0.5
        root_state_1[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device) # Reset orientation
        
        # Robot 2 starts at (0, -0.5, z)
        root_state_2 = self.robot_2.data.default_root_state[env_ids]
        root_state_2[:, 1] = -0.5
        root_state_2[:, 3:7] = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self.device) # Reset orientation (180 deg turn)

        # Write reset states to the simulation
        self.robot_1.write_root_state_to_sim(root_state_1, env_ids)
        self.robot_2.write_root_state_to_sim(root_state_2, env_ids)
        
        # Reset joint states
        default_joint_pos = self.robot_1.data.default_joint_pos[env_ids]
        default_joint_vel = self.robot_1.data.default_joint_vel[env_ids]
        self.robot_1.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)
        self.robot_2.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)

        # Reset episode timer
        self.episode_length_buf[env_ids] = 0
