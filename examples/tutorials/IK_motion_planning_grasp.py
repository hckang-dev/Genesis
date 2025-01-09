import genesis as gs
import numpy as np

########################## init ##########################
gs.init(backend=gs.metal)

########################## create a scene ##########################
scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=30,
        max_FPS=60,
    ),
    sim_options=gs.options.SimOptions(
        dt=0.01,
    ),
    show_viewer=True,
    show_FPS=False
)
########################## entities ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)
cube = scene.add_entity(
    gs.morphs.Box(
        size=(0.04, 0.04, 0.04),
        pos=(0.65, 0.0, 0.02),
    )
)
franka = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
)
########################## build ##########################
scene.build()

def simulate():
    def action():
        motors_dof = np.arange(7)
        fingers_dof = np.arange(7, 9)

        # set control gains
        franka.set_dofs_kp(
            np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
        )
        franka.set_dofs_kv(
            np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
        )
        franka.set_dofs_force_range(
            np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
        )

        end_effector = franka.get_link("hand")

        # move to pre-grasp pose
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=np.array([0.65, 0.0, 0.25]),
            quat=np.array([0, 1, 0, 0]),
        )
        # gripper open pos
        qpos[-2:] = 0.04
        path = franka.plan_path(
            qpos_goal=qpos,
            num_waypoints=200,  # 2s duration
        )
        # execute the planned path
        for waypoint in path:
            franka.control_dofs_position(waypoint)
            scene.step()

        # allow robot to reach the last waypoint
        for i in range(100):
            scene.step()

        # reach
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=np.array([0.65, 0.0, 0.130]),
            quat=np.array([0, 1, 0, 0]),
        )
        print(qpos)
        franka.control_dofs_position(qpos[:-2], motors_dof)
        for i in range(100):
            scene.step()

        # grasp
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_force(np.array([-0.6, -0.6]), fingers_dof)

        for i in range(100):
            scene.step()

        # lift
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=np.array([0.65, 0.0, 0.28]),
            quat=np.array([0, 1, 0, 0]),
        )
        print(qpos)
        franka.control_dofs_position(qpos[:-2], motors_dof)
        for i in range(100):
            scene.step()

        target_qpos = np.array([1.7627, -1.4631, -1.1437, -1.4283, -1.4417, 1.1242, 1.2048, 0.0332, 0.0332])
        target_qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=np.array([0.3, -0.3, 0.28]),
            quat=np.array([0, 1, 0, 0]),
        )

        return_path = franka.plan_path(
            qpos_goal=target_qpos,
            num_waypoints=50, 
            ignore_collision=True
        )
        # execute the planned path
        for waypoint in return_path:
            franka.control_dofs_position(waypoint[:-2], motors_dof)
            scene.step()

        # reach 
        for i in range(100):
            scene.step()

        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=np.array([0.3, -0.3, 0.130]),
            quat=np.array([0, 1, 0, 0]),
        )

        franka.control_dofs_position(qpos[:-2], motors_dof)
        for i in range(100):
            scene.step()


        # release
        franka.control_dofs_position([0.04, 0.04], fingers_dof)
        for i in range(100):
            scene.step()

        # comeback
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=np.array([0.3, -0.3, 0.28]),
            quat=np.array([0, 1, 0, 0]),
        )
        
        franka.control_dofs_position(qpos[:-2], motors_dof)
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