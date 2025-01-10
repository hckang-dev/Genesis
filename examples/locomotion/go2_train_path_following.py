import argparse
import os
import pickle
import shutil

from go2_train import WalkingEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower
import torch
import math
import numpy as np

def get_train_cfg(exp_name, max_iterations):

    train_cfg_dict = {
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
        "init_member_classes": {},
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
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 24,
            "policy_class_name": "ActorCritic",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 100,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 3,
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
        "termination_if_roll_greater_than": 60,  # degree
        "termination_if_pitch_greater_than": 60,
        "termination_if_base_height_less_than": 0.1,
        "termination_if_x_greater_than": 2.0,
        "termination_if_y_greater_than": 2.0,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "at_target_threshold": 0.1,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 13,
        "obs_scales": {
            "rel_pos": 1 / 3.0,
            "lin_vel": 1 / 3.0,
            "ang_vel": 1 / 3.14159,
        },
    }
    reward_cfg = {
        "reward_scales": {
            "target": 10.0,
            "smooth": -1e-4,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "pos_x_range": [-1.0, 1.0],
        "pos_y_range": [-1.0, 1.0],
        "pos_z_range": [0.2, 0.2],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg
class PathFollowingEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, device="mps"):
        self.device = torch.device(device)

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequence on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]
        # # create scene
        # self.scene = gs.Scene(
        #     sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
        #     viewer_options=gs.options.ViewerOptions(
        #         max_FPS=int(0.5 / self.dt),
        #         camera_pos=(-2.0, 1.0, 2.0),
        #         camera_lookat=(0.0, 0.0, 0.5),
        #         camera_fov=40,
        #     ),
        #     vis_options=gs.options.VisOptions(n_rendered_envs=1),
        #     rigid_options=gs.options.RigidOptions(
        #         dt=self.dt,
        #         constraint_solver=gs.constraint_solver.Newton,
        #         enable_collision=True,
        #         enable_joint_limit=True,
        #     ),
        #     show_viewer=show_viewer,
        #     show_FPS=False
        # )

        # # add plain
        # self.scene.add_entity(
        #     gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True
        #     )
        # )
        
        # self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        # self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        # self.inv_base_init_quat = inv_quat(self.base_init_quat)
        # self.robot = self.scene.add_entity(
        #     gs.morphs.URDF(
        #         file="urdf/go2/urdf/go2.urdf",
        #         pos=self.base_init_pos.cpu().numpy(),
        #         quat=self.base_init_quat.cpu().numpy(),
        #     ),
        # )
        
        # # build
        # self.scene.build(n_envs=num_envs)

        # # names to indices
        # self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]

        # # PD control parameters
        # self.robot.set_dofs_kp([self.env_cfg["kp"]] * 12, self.motor_dofs)
        # self.robot.set_dofs_kv([self.env_cfg["kd"]] * 12, self.motor_dofs)

        # # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # # initialize buffers
        # self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        # self.last_base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        # self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        # self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        # self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(
        #     self.num_envs, 1
        # )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        # self.dofs = torch.zeros((self.num_envs, 12), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.last_base_pos = torch.zeros_like(self.base_pos)

        # self.default_dof_pos = torch.tensor(
        #     [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]],
        #     device=self.device,
        #     dtype=gs.tc_float,
        # )
        self.extras = dict()  # extra information for logging
        log_dir = f"logs/go2-joystick"
        env_cfg2, obs_cfg2, reward_cfg2, command_cfg2, train_cfg2 = pickle.load(open(f"logs/go2-joystick/cfgs.pkl", "rb"))
        self.loco_env = WalkingEnv(
            num_envs=num_envs,
            env_cfg=env_cfg2,
            obs_cfg=obs_cfg2,
            reward_cfg=reward_cfg2,
            command_cfg=command_cfg2,
            show_viewer=True
        )
        
        reward_cfg["reward_scales"] = {}
        runner = OnPolicyRunner(self.loco_env, train_cfg2, log_dir, device="mps:0")
        resume_path = os.path.join(log_dir, f"model_1000.pt")
        runner.load(resume_path)
        self.policy = runner.get_inference_policy(device="mps:0")
        self.loco_env.obs_buf, _ = self.loco_env.reset()
        

    def _resample_commands(self, envs_idx):
        # Get the current position of the robot
        current_pos = self.loco_env.base_pos[envs_idx]

        # Generate random angles for directions (in radians)
        random_angles = gs_rand_float(0, 2 * math.pi, (len(envs_idx),), self.device)

        # Calculate new positions with a fixed distance of 1m
        offset_x = torch.cos(random_angles)
        offset_y = torch.sin(random_angles)

        # New positions are 1m away from the current position in random directions
        new_pos_x = current_pos[:, 0] + offset_x
        new_pos_y = current_pos[:, 1] + offset_y

        # Update commands with the new target positions
        self.commands[envs_idx, 0] = new_pos_x
        self.commands[envs_idx, 1] = new_pos_y
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["pos_z_range"], (len(envs_idx),), self.device)
        if self.loco_env.target is not None:
            self.loco_env.target.set_pos(self.commands[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        
    def step(self, actions): # actions: lin_vel_x, lin_vel_y, ang_vel
        with torch.no_grad():

            self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
            exec_actions = self.last_actions if self.simulate_action_latency else self.actions
            control_actions = self.policy(self.loco_env.obs_buf)
            self.loco_env.commands = exec_actions
            self.loco_env.step(control_actions)
            
            # update buffers
            self.episode_length_buf += 1
            self.last_base_pos[:] = self.base_pos[:]
            self.base_pos[:] = self.loco_env.robot.get_pos()
            self.rel_pos = self.commands - self.base_pos
            self.last_rel_pos = self.commands - self.last_base_pos
            self.base_quat[:] = self.loco_env.robot.get_quat()
            self.base_euler = quat_to_xyz(
                transform_quat_by_quat(torch.ones_like(self.base_quat) * self.loco_env.inv_base_init_quat, self.base_quat)
            )
            # resample commands
            envs_idx = (
                (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
                .nonzero(as_tuple=False)
                .flatten()
            )
            self._resample_commands(envs_idx)

            # check termination and reset
            self.reset_buf = self.episode_length_buf > self.max_episode_length
            self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
            self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
            self.reset_buf |= torch.abs(self.base_pos[:, 2]) < self.env_cfg["termination_if_base_height_less_than"]
            self.reset_buf |= torch.abs(self.rel_pos[:, 0]) > self.env_cfg["termination_if_x_greater_than"]
            self.reset_buf |= torch.abs(self.rel_pos[:, 1]) > self.env_cfg["termination_if_y_greater_than"]
            time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
            self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
            self.extras["time_outs"][time_out_idx] = 1.0

            self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

            # compute reward
            self.rew_buf[:] = 0.0
            for name, reward_func in self.reward_functions.items():
                rew = reward_func() * self.reward_scales[name]
                self.rew_buf += rew
                self.episode_sums[name] += rew

            # compute observations
            self.obs_buf = torch.cat(
                [
                    torch.clip(self.rel_pos * self.obs_scales["rel_pos"], -1, 1), # 3
                    self.loco_env.base_quat, # 4
                    torch.clip(self.loco_env.base_lin_vel * self.obs_scales["lin_vel"], -1, 1), # 3
                    torch.clip(self.loco_env.base_ang_vel * self.obs_scales["ang_vel"], -1, 1), # 3
                ],
                axis=-1,
            )


            return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras
    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return
        
        # reset dofs
        # self.robot.set_dofs_position(
        #     position=self.loco_env.dof_pos[envs_idx],
        #     dofs_idx_local=self.loco_env.motor_dofs,
        #     zero_velocity=True,
        #     envs_idx=envs_idx,
        # )

        # reset base
        self.base_pos[envs_idx] = self.loco_env.base_init_pos
        self.base_quat[envs_idx] = self.loco_env.base_init_quat.reshape(1, -1)
        # self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        # self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        # self.base_lin_vel[envs_idx] = 0
        # self.base_ang_vel[envs_idx] = 0
        # self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)
        self.loco_env.reset_idx(envs_idx)

    def reset(self):
        self.loco_env.reset()
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

   
    # ------------ reward functions----------------
    def _reward_target(self):
        # Compute the Euclidean distance between the robot's current position and the target position
        distance_to_target = torch.norm(self.rel_pos, dim=1)

        # Reward is higher for being closer to the target, penalize being further away
        target_reward = -distance_to_target

        # Optional: Add a bonus for staying within a threshold distance of the target
        at_target_bonus = torch.zeros_like(target_reward)
        at_target_bonus[distance_to_target < self.env_cfg["at_target_threshold"]] = 1.0

        # Combine the rewards
        total_reward = target_reward + at_target_bonus

        return total_reward

    def _reward_smooth(self):
        smooth_rew = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        return smooth_rew
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-path-following")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=1000)
    args = parser.parse_args()

    gs.init(logging_level="error")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env = PathFollowingEnv(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=True
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="mps:0")

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    def learn():
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    gs.tools.run_in_another_thread(fn=learn, args=[])
    env.loco_env.scene.viewer.start()



if __name__ == "__main__":
    main()

"""
# training
python examples/locomotion/go2_train.py
"""
