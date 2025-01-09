import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
import numpy as np

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class Go2Env:
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
            renderer=gs.renderers.RayTracer(
                env_surface=gs.surfaces.Emission(
                    emissive_texture=gs.textures.ImageTexture(
                        image_path="textures/indoor_bright.png",
                    ),
                ),
                env_radius=15.0,
                env_euler=(0, 0, 180),
                lights=[
                    {"pos": (0.0, 0.0, 10.0), "radius": 3.0, "color": (15.0, 15.0, 15.0)},
                ],
            ),
            show_FPS=False
        )

        # add plain
        self.scene.add_entity(
            gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True
            )
        )
        # self.scene.add_entity(  
        #     morph=gs.morphs.Plane(
        #         pos=(0.0, 0.0, 0.0),
        #     ),
        #     surface=gs.surfaces.Aluminium(
        #         ior=10.0,
        #     ),
        # )
        # add robot
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
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.extras = dict()  # extra information for logging

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
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
        # self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        # self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

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
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        # 🛠️ Camera Follow Logic 🛠️
        robot_position = self.base_pos[0].cpu().numpy()  # 첫 번째 로봇의 위치 사용
        camera_offset = np.array([2.0, 2.0, 1.5])  # 카메라 위치 오프셋
        camera_position = robot_position + camera_offset
        camera_lookat = robot_position + np.array([0.0, 0.0, 0.0])  # 약간 위쪽을 바라보게 설정
    
        # self.scene.viewer.set_camera_pose(pos=camera_position, lookat=camera_lookat)
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

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])
    def _reward_pose_match(self):
        """목표 자세와의 오차를 보상으로 반환합니다."""
        target_pose = torch.tensor([
            0.0, 0.0, -0.5,  # FR_hip_joint
            0.0, 0.0, -0.5,  # FL_hip_joint
            0.5, -0.5, 0.0,  # RR_hip_joint
            0.5, -0.5, 0.0   # RL_hip_joint
        ], device=self.device)
        
        pose_error = torch.sum(torch.square(self.dof_pos - target_pose), dim=1)
        return torch.exp(-pose_error / self.reward_cfg["tracking_sigma"])
    def _reward_pose_stability(self):
        """
        자세 안정성 보상.
        - 로봇의 각 관절이 빠르게 움직이거나 진동하는 것을 방지합니다.
        - 관절의 속도가 작을수록 높은 보상을 제공합니다.
        """
        # 관절 속도의 절댓값 합을 기반으로 보상 계산
        stability_cost = torch.sum(torch.abs(self.dof_vel), dim=1)
        
        # 안정성 보상 (낮은 관절 속도 -> 높은 보상)
        return torch.exp(-stability_cost / self.reward_cfg["tracking_sigma"])
    
    def _reward_energy_efficiency(self):
        """
        에너지 효율성 보상.
        - 관절에 가해진 힘(force)와 속도의 곱을 기반으로 에너지 사용량을 측정합니다.
        - 에너지 사용량이 낮을수록 높은 보상을 제공합니다.
        """
        # DOF별 힘(force)와 속도 계산
        dof_force = self.robot.get_dofs_force(self.motor_dofs)  # 관절에 가해진 힘
        energy_cost = torch.sum(torch.abs(dof_force * self.dof_vel), dim=1)
        
        # 에너지 효율성 보상 (낮은 에너지 사용량 -> 높은 보상)
        return torch.exp(-energy_cost / self.reward_cfg["tracking_sigma"])
    def _reward_foot_impact(self):
        """
        Penalize high foot impact forces to encourage gentle foot placement.
        """
        # 로봇과 바닥(plane) 사이의 접촉 정보 가져오기
        contacts = self.robot.get_contacts(with_entity=None)
        
        # 필요한 키가 모두 존재하는지 확인
        if not all(key in contacts for key in ["force_a", "force_b", "valid_mask"]):
            return torch.zeros(self.num_envs, device=self.device)
        
        # force_a, force_b, valid_mask를 안전하게 텐서로 변환
        force_a = torch.tensor(contacts["force_a"], device=self.device, dtype=torch.float32)
        force_b = torch.tensor(contacts["force_b"], device=self.device, dtype=torch.float32)
        valid_mask = torch.tensor(contacts["valid_mask"], device=self.device, dtype=torch.bool)
        
        if force_a.numel() == 0 or force_b.numel() == 0 or valid_mask.numel() == 0:
            return torch.zeros(self.num_envs, device=self.device)
        
        # 유효한 접촉 포인트만 필터링
        valid_mask = valid_mask & (force_a.shape[1] > 0) & (force_b.shape[1] > 0)
        
        # 접촉 힘의 크기 계산
        impact_forces = torch.where(
            valid_mask,
            torch.norm(force_a, dim=-1) + torch.norm(force_b, dim=-1),
            torch.zeros_like(valid_mask, dtype=torch.float32)
        )
        
        # 각 환경(env)별로 충격 합산
        total_impact = torch.sum(impact_forces, dim=-1)
        
        # 충격을 보상으로 변환 (충격이 낮을수록 높은 보상)
        return torch.exp(-total_impact / self.reward_cfg["tracking_sigma"])

    
    def _reward_smooth_motion(self):
        """
        Penalize abrupt changes in joint velocities for smooth motion.
        """
        dof_vel_change = torch.sum(torch.square(self.dof_vel - self.last_dof_vel), dim=1)
        return torch.exp(-dof_vel_change / self.reward_cfg["tracking_sigma"])
    def _reward_speed(self):
        """
        최대한 빠른 속도로 전진하도록 보상을 부여합니다.
        """
        forward_speed = self.base_lin_vel[:, 0]  # x축 속도
        reward = torch.clip(forward_speed, min=0.0)  # 음수 보상 방지
        
        # 속도가 높을수록 보상을 기하급수적으로 증가
        reward = torch.square(reward)  # 속도가 2배면 보상은 4배
        
        return reward
    def _reward_stability(self):
        """
        로봇의 기울기가 과도하게 커지는 것을 방지합니다.
        """
        roll = torch.abs(self.base_euler[:, 0])  # x축 회전 각도
        pitch = torch.abs(self.base_euler[:, 1])  # y축 회전 각도
        
        # 기울기가 작을수록 보상
        reward = torch.exp(-roll) * torch.exp(-pitch)
        return reward