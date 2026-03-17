#!/usr/bin/env python3
from __future__ import annotations
import os, sys, time
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from scipy.spatial.transform import Rotation as R

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.dls_velocity_commander import DLSVelocityCommander
from utils.gripper_commands.franka_gripper import FrankaGripperController


# -------------------------------------------------------
def move_and_wait(robot, executor, pos, quat, name):
    print(name)
    robot.set_target(pos, quat)
    robot.reset_goal_reached()

    while rclpy.ok():
        executor.spin_once(timeout_sec=0.01)
        if robot.goal_reached():
            break

    robot.publish_zero_velocity()
    time.sleep(0.15)


# -------------------------------------------------------
def main(args=None):

    rclpy.init(args=args)

    gripper = FrankaGripperController()
    gripper.open_gripper(width=0.08)
    time.sleep(1.0)

    robotB = DLSVelocityCommander(
        robot_id="robotB",
        base_link="fr3_link0",
        tip_link="fr3_link8",
        joint_names=[
            "fr3_joint1","fr3_joint2","fr3_joint3",
            "fr3_joint4","fr3_joint5","fr3_joint6","fr3_joint7",
        ],
        target_pos=[0,0,0],
        target_quat=[0,0,0,1],
        joint_state_topic="/NS_1/franka/joint_states",
        velocity_command_topic="/NS_1/joint_velocity_controller/commands",
        robot_description_topic="/NS_1/robot_description",
        ee_pose_topic=None,
        ee_pose_is_stamped=False,
        max_cartesian_vel=0.08,
        max_angular_vel=1,
        dt=0.01,
        damping=0.03,
    )

    executor = MultiThreadedExecutor()
    executor.add_node(robotB)

    # -------------------------------------------------------
    # USER INPUT
    # -------------------------------------------------------

    # START_POS  = [0.5603152550301461, -0.04328130682363318, 0.2188000931191316] #0.2178000931191316
    # START_QUAT = [0.9040584664694167, -0.4265039161201177, -0.026731102818722683, 0.0076254102358457206]

    START_POS  = [0.6481336480188284, 0.07694619616443668, 0.18092825074622304]
    START_QUAT = [ 0.9145697094038896, -0.40410455661455186, 0.01426581779529769, -0.007631540268703725]

    # OSC_POS = np.array([0.00, -0.001, 0.0])
    # OSC_ROT = np.array([0.01, 0.0, 0.02])

    OSC_POS = np.array([0.00, -0.00, 0.0001])
    OSC_ROT = np.array([0.0, 0.0, 0.2])

    PULL_Z_PER_CYCLE = 0.0002 # 0.2 mm per cycle
    N_CYCLES = 75

    FINAL_PULL_Z = 0.05      # 20 mm final unplug

    # -------------------------------------------------------

    try:

        move_and_wait(robotB, executor, START_POS, START_QUAT, "GO TO START")

        print("Closing gripper")
        gripper.close_gripper(width=0.04, force=50.0)
        time.sleep(1.0)

        base_pos = np.array(START_POS)
        base_R   = R.from_quat(START_QUAT)

        # -------- OSCILLATION LOOP --------
        for k in range(N_CYCLES):

            print(f"Oscillation cycle {k+1}/{N_CYCLES}")

            center_pos = base_pos + np.array([0,0,k*PULL_Z_PER_CYCLE])
            center_R   = base_R

            move_and_wait(
                robotB, executor,
                (center_pos + OSC_POS).tolist(),
                (center_R * R.from_rotvec(OSC_ROT)).as_quat().tolist(),
                "OSC +"
            )

            move_and_wait(
                robotB, executor,
                center_pos.tolist(),
                center_R.as_quat().tolist(),
                "CENTER"
            )

            move_and_wait(
                robotB, executor,
                (center_pos - OSC_POS).tolist(),
                (center_R * R.from_rotvec(-OSC_ROT)).as_quat().tolist(),
                "OSC -"
            )

            move_and_wait(
                robotB, executor,
                center_pos.tolist(),
                center_R.as_quat().tolist(),
                "CENTER"
            )

        # -------- FINAL PULL --------
        print("FINAL UNPLUG PULL")

        final_pos = base_pos + np.array([0,0,FINAL_PULL_Z])

        move_and_wait(
            robotB,
            executor,
            final_pos.tolist(),
            base_R.as_quat().tolist(),
            "FINAL PULL"
        )

        print("UNPLUG COMPLETE")

    finally:
        robotB.publish_zero_velocity()
        robotB.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
