#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from scipy.spatial.transform import Rotation as R
from numpy.linalg import pinv, eig

from geometry_msgs.msg import WrenchStamped

# ------------------------------------------------------------
# Fix Python path so utils folder is visible
# ------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.dls_velocity_commander import DLSVelocityCommander
from utils.gripper_commands.franka_gripper import FrankaGripperController


# ------------------------------------------------------------
# Wrench Monitor
# ------------------------------------------------------------
class WrenchMonitor:

    def __init__(self):

        self.force = np.zeros(3)
        self.torque = np.zeros(3)

    def callback(self, msg: WrenchStamped):

        self.force = np.array([
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z
        ])

        self.torque = np.array([
            msg.wrench.torque.x,
            msg.wrench.torque.y,
            msg.wrench.torque.z
        ])


# ------------------------------------------------------------
# Move helper
# ------------------------------------------------------------
def move_and_wait(robot, executor, pos, quat):

    robot.set_target(pos, quat)
    robot.reset_goal_reached()

    while rclpy.ok():
        executor.spin_once(timeout_sec=0.01)
        if robot.goal_reached():
            break

    robot.publish_zero_velocity()
    time.sleep(0.3)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():

    rclpy.init()

    # --------------------------------------------------------
    # Initialize gripper
    # --------------------------------------------------------
    gripper = FrankaGripperController()

    gripper.open_gripper(width=0.08)
    time.sleep(1)

    # --------------------------------------------------------
    # Force monitor node
    # --------------------------------------------------------
    wrench_monitor = WrenchMonitor()

    monitor_node = rclpy.create_node("wrench_monitor")

    monitor_node.create_subscription(
        WrenchStamped,
        "/NS_1/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame",
        wrench_monitor.callback,
        10
    )

    # --------------------------------------------------------
    # Robot controller
    # --------------------------------------------------------
    robot = DLSVelocityCommander(
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
        max_cartesian_vel=0.05,
        max_angular_vel=1,
        dt=0.01,
        damping=0.01,
    )

    executor = MultiThreadedExecutor()

    executor.add_node(robot)
    executor.add_node(monitor_node)

    # --------------------------------------------------------
    # Start pose
    # --------------------------------------------------------
    START_POS  = [0.5603, -0.0432, 0.2178]

    START_QUAT = [
        0.90405,
        -0.42650,
        -0.02673,
        0.00762
    ]

    print("Moving to start pose")

    move_and_wait(robot, executor, START_POS, START_QUAT)

    print("Closing gripper")

    gripper.close_gripper(width=0.04, force=25)

    time.sleep(1)

    base_pos = np.array(START_POS)
    base_R = R.from_quat(START_QUAT)

    # --------------------------------------------------------
    # Probing parameters
    # --------------------------------------------------------
    Ax = 0.001
    Ay = 0.001
    Az = 0.001

    frequency = 2.0

    force_threshold = 12.0
    amplitude_growth = 1.002
    amplitude_decay = 0.8

    probe_duration = 8.0

    # --------------------------------------------------------
    # Data buffers
    # --------------------------------------------------------
    data_X = []
    data_F = []

    start_time = time.time()

    print("Starting probing")

    # --------------------------------------------------------
    # Probing loop
    # --------------------------------------------------------
    while rclpy.ok():

        executor.spin_once(timeout_sec=0.01)

        t = time.time() - start_time

        dx = Ax*np.sin(2*np.pi*frequency*t)
        dy = Ay*np.sin(2*np.pi*frequency*t + np.pi/2)
        dz = Az*np.sin(2*np.pi*frequency*t + np.pi)

        pos = base_pos + np.array([dx,dy,dz])

        robot.set_target(pos.tolist(), START_QUAT)

        F = wrench_monitor.force
        T = wrench_monitor.torque

        # ----------------------------------------------
        # Adaptive amplitude control
        # ----------------------------------------------
        if abs(F[0]) > force_threshold:
            Ax *= amplitude_decay
        else:
            Ax *= amplitude_growth

        if abs(F[1]) > force_threshold:
            Ay *= amplitude_decay
        else:
            Ay *= amplitude_growth

        if abs(F[2]) > force_threshold:
            Az *= amplitude_decay
        else:
            Az *= amplitude_growth

        # ----------------------------------------------
        # Log data
        # ----------------------------------------------
        data_X.append([dx,dy,dz])
        data_F.append(F.tolist())

        if t > probe_duration:
            break

    print("Probing finished")

    robot.publish_zero_velocity()

    # --------------------------------------------------------
    # Estimate stiffness matrix
    # --------------------------------------------------------
    X = np.array(data_X).T
    F = np.array(data_F).T

    K = F @ pinv(X)

    print("\nEstimated stiffness matrix:")
    print(K)

    # --------------------------------------------------------
    # Eigen analysis
    # --------------------------------------------------------
    eigvals, eigvecs = eig(K)

    idx = np.argmin(eigvals)

    axis = eigvecs[:,idx]

    axis = axis / np.linalg.norm(axis)

    print("\nAxis of least resistance:")
    print(axis)

    print("\nEigenvalues:")
    print(eigvals)

    robot.destroy_node()
    monitor_node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()