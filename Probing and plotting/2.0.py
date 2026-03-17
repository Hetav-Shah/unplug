#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
import numpy as np
import rclpy

from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import WrenchStamped

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.dls_velocity_commander import DLSVelocityCommander
from utils.gripper_commands.franka_gripper import FrankaGripperController


# ==========================================================
# Wrench Monitor
# ==========================================================

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

    def force_mag(self):
        return np.linalg.norm(self.force)

    def torque_mag(self):
        return np.linalg.norm(self.torque)


# ==========================================================
# Exploration Velocity Generator
# ==========================================================

class VelocityExplorer:

    def __init__(self):

        self.vel_xy = 4
        self.vel_z = 4

        self.freq_xy = 2.0
        self.freq_z = 3.0

    def velocity(self, t):

        vx = self.vel_xy * np.sin(2*np.pi*self.freq_xy*t)
        vy = self.vel_xy * np.cos(2*np.pi*self.freq_xy*t)
        vz = self.vel_z * np.sin(2*np.pi*self.freq_z*t)

        return np.array([vx, vy, vz])


# ==========================================================
# Force Projection
# ==========================================================

def project_velocity(v_cmd, force):

    f_mag = np.linalg.norm(force)

    if f_mag < 0.5:
        return v_cmd

    f_hat = force / f_mag

    v_safe = v_cmd - np.dot(v_cmd, f_hat) * f_hat

    return v_safe


# ==========================================================
# Move Helper
# ==========================================================

def move_and_wait(robot, executor, pos, quat):

    robot.set_target(pos, quat)
    robot.reset_goal_reached()

    while rclpy.ok():

        executor.spin_once(timeout_sec=0.01)

        if robot.goal_reached():
            break

    robot.publish_zero_velocity()

    time.sleep(0.5)


# ==========================================================
# Main
# ==========================================================

def main():

    rclpy.init()

    wrench = WrenchMonitor()

    node = rclpy.create_node("probe_wrench_monitor")

    node.create_subscription(
        WrenchStamped,
        "/NS_1/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame",
        wrench.callback,
        10
    )

    gripper = FrankaGripperController()

    robot = DLSVelocityCommander(

        robot_id="robotB",
        base_link="fr3_link0",
        tip_link="fr3_link8",

        joint_names=[
            "fr3_joint1","fr3_joint2","fr3_joint3",
            "fr3_joint4","fr3_joint5","fr3_joint6","fr3_joint7"
        ],

        target_pos=[0,0,0],
        target_quat=[0,0,0,1],

        joint_state_topic="/NS_1/franka/joint_states",
        velocity_command_topic="/NS_1/joint_velocity_controller/commands",
        robot_description_topic="/NS_1/robot_description",

        max_cartesian_vel=0.01,
        dt=0.01,
        damping=0.01
    )

    executor = MultiThreadedExecutor()

    executor.add_node(robot)
    executor.add_node(node)

    START_POS = np.array([0.5603, -0.0432, 0.1778])

    START_QUAT = [
        0.904058,
        -0.426503,
        -0.026731,
        0.007625
    ]

    print("Opening gripper")

    gripper.open_gripper(width=0.08)

    time.sleep(1)

    print("Moving to probing pose")

    move_and_wait(robot, executor, START_POS.tolist(), START_QUAT)

    print("Closing gripper")

    gripper.close_gripper(width=0.01, force=25)

    time.sleep(1)

    explorer = VelocityExplorer()

    pos = START_POS.copy()

    dt = 0.01

    start_time = time.time()

    max_disp = 0.001

    print("Starting probing")

    try:

        while rclpy.ok():

            executor.spin_once(timeout_sec=0.01)

            if wrench.force_mag() > 10 or wrench.torque_mag() > 2:

                print("Safety threshold exceeded")
                break

            t = time.time() - start_time

            v_cmd = explorer.velocity(t)

            v_safe = project_velocity(v_cmd, wrench.force)

            pos = pos + v_safe * dt

            delta = pos - START_POS

            delta = np.clip(delta, -max_disp, max_disp)

            pos = START_POS + delta

            robot.set_target(
                pos.tolist(),
                START_QUAT
            )

    finally:

        robot.publish_zero_velocity()

        robot.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":
    main()