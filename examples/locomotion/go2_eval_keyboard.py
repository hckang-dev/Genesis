import argparse
import os
import pickle
import threading

import torch
from pynput import keyboard
from go2_env import Go2Env
from rsl_rl.runners import OnPolicyRunner
from genesis.utils.geom import transform_by_quat

import genesis as gs
import numpy as np
import time

def update_camera(scene, robot_pos):
    """Updates the camera position to follow the robot"""
    if not scene.viewer:
        return
    
    robot_position = robot_pos[0].cpu().numpy()  # 첫 번째 로봇의 위치 사용
    camera_offset = np.array([2.0, -2.0, 1.5])  # 카메라 위치 오프셋
    camera_position = robot_position + camera_offset
    camera_lookat = robot_position + np.array([0.0, 0.0, 0.5])  # 약간 위쪽을 바라보게 설정

    scene.viewer.set_camera_pose(pos=camera_position, lookat=camera_lookat)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("--ckpt", type=int, default=100)
    args = parser.parse_args()

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}
    
    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="mps:0")
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="mps:0")

    # Shared variables for keyboard input
    control_command = torch.zeros((1, 3), device="mps:0")
    press_start_time = {"up": None, "down": None, "left": None, "right": None}
    lock = threading.Lock()

    def on_press(key):
        try:
            with lock:
                current_time = time.time()
                if key == keyboard.Key.up:
                    if press_start_time["up"] is None:
                        press_start_time["up"] = current_time
                    duration = current_time - press_start_time["up"]
                    control_command[0, 0] = min(0.5 + duration, 1.0)  # Increase lin_vel_x
                elif key == keyboard.Key.down:
                    if press_start_time["down"] is None:
                        press_start_time["down"] = current_time
                    duration = current_time - press_start_time["down"]
                    control_command[0, 0] = max(-0.5 - duration, -1.0)  # Decrease lin_vel_x
                elif key == keyboard.Key.left:
                    if press_start_time["left"] is None:
                        press_start_time["left"] = current_time
                    duration = current_time - press_start_time["left"]
                    control_command[0, 1] = min(0.5 + duration, 1.0)  # Increase lin_vel_y
                elif key == keyboard.Key.right:
                    if press_start_time["right"] is None:
                        press_start_time["right"] = current_time
                    duration = current_time - press_start_time["right"]
                    control_command[0, 1] = max(-0.5 - duration, -1.0)  # Decrease lin_vel_y
                elif hasattr(key, 'char') and key.char == 'j':
                    if press_start_time["j"] is None:
                        press_start_time["j"] = current_time
                    duration = current_time - press_start_time["j"]
                    control_command[0, 2] = max(-0.5 - duration, 1.0)  # Decrease ang_vel
                elif hasattr(key, 'char') and key.char == 'k':
                    if press_start_time["k"] is None:
                        press_start_time["k"] = current_time
                    duration = current_time - press_start_time["k"]
                    control_command[0, 2] = min(0.5 + duration, -1.0)  # Increase ang_vel
        except Exception as e:
            pass

    def on_release(key):
        try:
            with lock:
                if key == keyboard.Key.up:
                    control_command[0, 0] = 0.0  # Reset lin_vel_x
                    press_start_time["up"] = None
                elif key == keyboard.Key.down:
                    control_command[0, 0] = 0.0  # Reset lin_vel_x
                    press_start_time["down"] = None
                elif key == keyboard.Key.left:
                    control_command[0, 1] = 0.0  # Reset lin_vel_y
                    press_start_time["left"] = None
                elif key == keyboard.Key.right:
                    control_command[0, 1] = 0.0  # Reset lin_vel_y
                    press_start_time["right"] = None
                elif key.char == 'j':
                    control_command[0, 2] = 0.0  # Reset ang_vel
                    press_start_time["j"] = None
                elif key.char == 'k':
                    control_command[0, 2] = 0.0  # Reset ang_vel
                    press_start_time["k"] = None
        except Exception as e:
            pass


    def update_camera(scene, robot):
        """Updates the camera position to follow the robot"""
        if not scene.viewer:
            return

        # Get robot position
        robot_pos = robot.get_pos()[0]

        # Camera position relative to robot
        offset_x = -4.0  # centered horizontally
        offset_y = 0.0  # 4 units behind (in Y axis)
        offset_z = 2.0  # 2 units above

        camera_pos = (
            float(robot_pos[0] + offset_x),
            float(robot_pos[1] + offset_y),
            float(robot_pos[2] + offset_z)
        )
        # Look target slightly ahead of the robot along the x-axis
        lookat_pos = (
            float(robot_pos[0] + 1.0),  # 1 unit ahead in X axis
            float(robot_pos[1]),
            float(robot_pos[2])
        )
        # Update camera position and look target
        scene.viewer.set_camera_pose(pos=camera_pos, lookat=lookat_pos)


    def step(obs):
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()

        with torch.no_grad():
            while True:
                actions = policy(obs)
                env.commands[0, 0] = control_command[0, 0] 
                env.commands[0, 1] = control_command[0, 1]  
                env.commands[0, 2] = control_command[0, 2]  
                obs, _, _, _, _ = env.step(actions)
                update_camera(env.scene, env.robot)

    obs, _ = env.reset()
    gs.tools.run_in_another_thread(fn=step, args=[obs])
    env.scene.viewer.start()

if __name__ == "__main__":
    main()

"""
# evaluation with keyboard control
python examples/locomotion/go2_eval_keyboard.py -e go2-walking --ckpt 100
"""
