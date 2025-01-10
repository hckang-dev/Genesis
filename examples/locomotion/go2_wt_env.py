import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
import numpy as np

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class Go2WalkingTargetEnv:
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

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(-2.0, 1.0, 2.0),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=1),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
            show_FPS=False
        )

        # add plain
        self.scene.add_entity(
            gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True
            )
        )
        # # add target
        # if self.env_cfg["visualize_target"]:
        #     self.target = self.scene.add_entity(
        #         morph=gs.morphs.Box(
        #             size=(0.05, 0.05, 0.05),
        #             fixed=True,
        #             collision=False,
        #         ),
        #         surface=gs.surfaces.Rough(
        #             diffuse_texture=gs.textures.ColorTexture(
        #                 color=(1.0, 0.5, 0.5),
        #             ),
        #         ),
        #     )
        # else:
        #     self.target = None
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )
        
        # build
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motor_dofs)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motor_dofs)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.rel_pos = torch.zeros_like(self.base_pos)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.last_base_pos = torch.zeros_like(self.base_pos)
        self.extras = dict()  # extra information for logging

    def _resample_commands(self, envs_idx):
        pass
        
    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.last_base_pos[:] = self.base_pos[:]
        self.base_pos[:] = self.robot.get_pos()
        self.rel_pos[:] = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)
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
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.actions,  # 12
                
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.last_base_pos[envs_idx] = self.base_init_pos
        self.rel_pos = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
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

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None
    
    def get_robot_pose(self):
        """
        Returns the robot's current position and orientation (quaternion).
        """
        position = self.robot.get_pos()  # [num_envs, 3]
        orientation = self.robot.get_quat()  # [num_envs, 4]
        return position, orientation

    def get_robot_velocity(self):
        """
        Returns the robot's linear and angular velocities in the local frame.
        """
        linear_velocity = self.robot.get_vel()
        angular_velocity = self.robot.get_ang()
        return linear_velocity, angular_velocity

    def get_robot_direction(self):
        """
        Computes and returns the robot's forward (face_dir), lateral (side_dir), and upward (up_dir) direction vectors.
        """
        _, orientation = self.get_robot_pose()

        # Forward direction (x-axis)
        face_dir = transform_by_quat(torch.tensor([1.0, 0.0, 0.0], device=self.device), orientation)

        # Lateral direction (y-axis)
        side_dir = transform_by_quat(torch.tensor([0.0, 1.0, 0.0], device=self.device), orientation)

        # Upward direction (z-axis)
        up_dir = transform_by_quat(torch.tensor([0.0, 0.0, 1.0], device=self.device), orientation)

        return face_dir, side_dir, up_dir
    def get_target_direction(self):
        """
        Computes and returns the normalized direction vector from the robot's position to the target.
        """
        # 현재 로봇 위치와 목표 위치
        current_pos = self.base_pos  # [num_envs, 3]
        target_pos = self.commands  # [num_envs, 3]

        # 방향 벡터 계산
        direction_vector = target_pos - current_pos  # [num_envs, 3]

        # 방향 벡터를 단위 벡터로 정규화
        direction_norm = torch.norm(direction_vector[:, :2], dim=1, keepdim=True) + 1e-6  # Avoid division by zero
        normalized_direction = direction_vector[:, :2] / direction_norm  # Only x, y components are normalized

        return normalized_direction  # [num_envs, 2]
    def get_energy_reward(self):
        joint_velocities = self.robot.get_dofs_velocity(self.motor_dofs)
        energy_consumption = torch.sum(torch.abs(self.actions) * torch.abs(joint_velocities), dim=1)
        return -energy_consumption
    
    def _reward_goal(self):
        """
        Computes the reward for the robot based on its alignment with the target direction,
        maintaining stability, and minimizing energy consumption.
        """
        # Get robot state information
        COM_pos, COM_quat = self.get_robot_pose()
        COM_vel, COM_ang = self.get_robot_velocity()
        face_dir, side_dir, up_dir = self.get_robot_direction()

        # Compute the normalized target direction
        target_direction = self.get_target_direction()  # [num_envs, 2]

        # Current face direction (x-axis of the robot) projected onto the ground plane
        face_direction_ground = face_dir[:, :2]  # [num_envs, 2]
        face_direction_norm = torch.norm(face_direction_ground, dim=1, keepdim=True) + 1e-6  # Avoid division by zero
        normalized_face_direction = face_direction_ground / face_direction_norm  # Normalize to unit vector

        # Reward for aligning the robot's face direction with the target direction
        alignment_reward = torch.sum(normalized_face_direction * target_direction, dim=1)  # Dot product measures alignment

        # Reward for maintaining height
        target_height = self.base_init_pos[2]  # Maintain the initial height
        height_reward = -torch.abs(COM_pos[:, 2] - target_height)  # Penalize deviations from initial height

        # Reward for stability (staying upright)
        target_up = torch.tensor([0, 0, 1], device=self.device, dtype=gs.tc_float)  # Target upright direction
        stability_reward = -torch.norm(up_dir - target_up, dim=1)  # Penalize deviation from upright orientation

        # Reward for minimizing angular velocity
        ang_velocity_penalty = -torch.norm(COM_ang, dim=1)  # Penalize excessive angular velocity

        # Total reward
        alpha_align = 2.0
        alpha_height = 1.0
        alpha_stability = 1.0
        alpha_ang = 0.5

        r = (alpha_align * alignment_reward +
            alpha_height * height_reward +
            alpha_stability * stability_reward +
            alpha_ang * ang_velocity_penalty)

        # Add energy penalty
        r_energy = self.get_energy_reward()
        return r + r_energy
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)
    
    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])
    def _reward_rotation(self):
        # Convert angular velocity to degrees per second
        base_ang_vel_deg = self.base_ang_vel * (180 / math.pi)
        desired_angular_velocity_deg = torch.tensor(
            [0.0, 0.0, self.command_cfg["target_yaw_rate"] * (180 / math.pi)],
            device=self.device,
        )
        ang_vel_error = torch.norm(base_ang_vel_deg - desired_angular_velocity_deg, dim=1)
        return -ang_vel_error

