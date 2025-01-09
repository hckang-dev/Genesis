import genesis as gs
import numpy as np
import torch
########################## init ##########################
gs.init(backend=gs.metal)

########################## create a scene ##########################
scene =gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.02, substeps=2),
    viewer_options=gs.options.ViewerOptions(
        max_FPS=int(0.5 / 0.02),
        camera_pos=(-2.0, 1.0, 2.0),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
    ),
    vis_options=gs.options.VisOptions(n_rendered_envs=1),
    rigid_options=gs.options.RigidOptions(
        dt=0.02,
        constraint_solver=gs.constraint_solver.Newton,
        enable_collision=True,
        enable_joint_limit=True,
    ),
    show_viewer=True,
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

########################## entities ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)
base_init_pos = torch.tensor([0.0, 0.0, 0.42], device="mps")
base_init_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device="mps")

robot = scene.add_entity(
    gs.morphs.URDF(
        file="urdf/go2/urdf/go2.urdf",
        pos=base_init_pos.cpu().numpy(),
        quat=base_init_quat.cpu().numpy()
    )
)
kp = 20
kd = 0.5
num_actions = 12
########################## build ##########################
scene.build()


dof_names = [
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
]
motor_dofs = [robot.get_joint(name).dof_idx_local for name in dof_names]
robot.set_dofs_kp(
    [kp] * num_actions, motor_dofs
)
robot.set_dofs_kv(
    [kd] * num_actions, motor_dofs
)
qpos_start = np.zeros(robot.n_qs)  # Initial neutral position
qpos_goal = np.zeros(robot.n_qs)  # Target position
qpos_goal[0] = 0.2  # Example: Move one joint to 0.2 radian


def simulate():
    def action():        
        waypoints = robot.plan_path(
            qpos_goal=qpos_goal,
            qpos_start=qpos_start,
            planner="EST",
            timeout=10.0,
            smooth_path=True,
            num_waypoints=50,
            ignore_collision=True,
            ignore_joint_limit=False,
        )
        # execute the planned path
        for waypoint in waypoints:
            robot.control_dofs_position(waypoint)
            scene.step()

        # allow robot to reach the last waypoint
        for i in range(100):
            scene.step()


        print("Action sequence complete.")

    while(True):
        action()
        scene.reset()

gs.tools.run_in_another_thread(
    fn=simulate, 
    args=[])

scene.viewer.start()