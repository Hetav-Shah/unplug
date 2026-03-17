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
def main(args=None):
    rclpy.init(args=args)

    # ---------------- GRIPPER ----------------
    gripper = FrankaGripperController()
    gripper.open_gripper(width=0.08)
    time.sleep(1.0)

    # ---------------- ROBOT ----------------
    robotB = DLSVelocityCommander(
        robot_id="robotB",
        base_link="fr3_link0",
        tip_link="fr3_link8",
        joint_names=[
            "fr3_joint1","fr3_joint2","fr3_joint3",
            "fr3_joint4","fr3_joint5","fr3_joint6","fr3_joint7",
        ],
        target_pos=[0.0,0.0,0.0],
        target_quat=[0.0,0.0,0.0,1.0],
        joint_state_topic="/NS_1/franka/joint_states",
        velocity_command_topic="/NS_1/joint_velocity_controller/commands",
        robot_description_topic="/NS_1/robot_description",
        ee_pose_topic=None,
        ee_pose_is_stamped=False,
        max_cartesian_vel=0.2,
        max_angular_vel=0.2,
        dt=0.01,
        damping=0.03,
    )

    executor = MultiThreadedExecutor()
    executor.add_node(robotB)

    # -------------------------------------------------------
    # Move helper
    # -------------------------------------------------------
    def move_and_wait(pos, quat, name, timeout=6.0):
        robotB.get_logger().info(f"Moving to {name}")
        robotB.set_target(pos, quat)
        robotB.reset_goal_reached()

        start = time.time()
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.01)

            if robotB.goal_reached():
                break

            if time.time() - start > timeout:
                robotB.get_logger().warn(f"Timeout at {name}")
                break

    # ---------------- TASK PARAMETERS ----------------
    CAP_POS  = [0.4209066790, -0.2864754995, 0.3820524258]
    CAP_QUAT = [0.9222371318, -0.3865121441, -0.0060582147, 0.0070945918]

    try:

        # ---------------------------------------------------
        # MOVE TO CAP
        # ---------------------------------------------------
        move_and_wait(CAP_POS, CAP_QUAT, "CAP_POSE")

        robotB.publish_zero_velocity()
        gripper.close_gripper(width=0.04, force=20.0)
        time.sleep(0.6)

        current_quat = CAP_QUAT.copy()

        move_and_wait(CAP_POS, current_quat, f"ROTATE")

        robotB.publish_zero_velocity()
        gripper.open_gripper(width=0.08)
        time.sleep(0.4)

        move_and_wait(CAP_POS, CAP_QUAT, f"RESET")

        robotB.publish_zero_velocity()
        gripper.close_gripper(width=0.04, force=20.0)
        time.sleep(0.4)

        current_quat = CAP_QUAT.copy()

        # ---------------------------------------------------
        # LIFT CAP
        # ---------------------------------------------------
        LIFT_POS = [
            CAP_POS[0],
            CAP_POS[1],
            CAP_POS[2] + 0.15
        ]

        move_and_wait(LIFT_POS, CAP_QUAT, "LIFT")

        # ---------------------------------------------------
        # FINAL DROP POSITION (NEW STEP)
        # ---------------------------------------------------
        DROP_POS = [0.515845601865624, 0.013748501468951086, 0.12446347175304144]

        DROP_QUAT = [0.9270424064745059, -0.3746291962130282, -0.010416844032273803, 0.011697491318436636]

        move_and_wait(DROP_POS, DROP_QUAT, "DROP_POSITION")

        # ---------------------------------------------------
        # OPEN GRIPPER
        # ---------------------------------------------------
        robotB.publish_zero_velocity()
        gripper.open_gripper(width=0.08)
        time.sleep(1.0)

    finally:
        robotB.publish_zero_velocity()
        robotB.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
