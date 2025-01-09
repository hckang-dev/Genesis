import argparse
import os
import pickle
import shutil

import torch
import numpy as np
from go2_env import Go2Env
from rsl_rl.runners import OnPolicyRunner
import genesis as gs

def get_train_cfg(exp_name, max_iterations):
    return {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "max_iterations": max_iterations,
            "num_steps_per_env": 24,
            "save_interval": 100,
        },
        "seed": 1,
    }

def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        # joint/link names
        "default_joint_angles": {  # [rad]
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "dof_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        # PD
        "kp": 20.0,
        "kd": 0.5,
        # termination
            "termination_if_no_progress": 1.0,  # 일정 시간 동안 전진 속도가 낮으면 종료 (m/s)
            "termination_if_no_contact": 0.1,  # 다리가 지면에서 떨어진 시간 (초)
            "termination_if_collision": True,  # 환경과 충돌하면 종료
        # base pose
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 45,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "reward_scales": {
            "path_tracking": 1.0,
            "action_smoothness": -0.005,
        },
    }
    path_cfg = {
        "waypoints": np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ]),
        "tolerance": 0.1,
    }
    return env_cfg, obs_cfg, reward_cfg, path_cfg

class PathTrackingEnv(Go2Env):
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, path_cfg, device="mps"):
        super().__init__(num_envs, env_cfg, obs_cfg, reward_cfg, {}, device=device)
        self.waypoints = torch.tensor(path_cfg["waypoints"], device=self.device)
        self.tolerance = path_cfg["tolerance"]
        self.current_waypoint_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def step(self, actions):
        obs, _, rew, reset, extras = super().step(actions)
        self._update_waypoints()
        return obs, None, self._compute_path_reward(), reset, extras

    def _update_waypoints(self):
        distances = torch.norm(self.base_pos[:, :2] - self.waypoints[self.current_waypoint_idx], dim=1)
        waypoint_reached = distances < self.tolerance
        self.current_waypoint_idx += waypoint_reached.long()
        self.current_waypoint_idx.clamp_(max=len(self.waypoints) - 1)

    def _compute_path_reward(self):
        distances = torch.norm(self.base_pos[:, :2] - self.waypoints[self.current_waypoint_idx], dim=1)
        path_tracking_reward = torch.exp(-distances)
        return path_tracking_reward * self.reward_cfg["reward_scales"]["path_tracking"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="path_tracking")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=100)
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, path_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env = PathTrackingEnv(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, path_cfg=path_cfg
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="mps:0")

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, path_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

if __name__ == "__main__":
    main()
