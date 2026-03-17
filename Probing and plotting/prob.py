#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
import numpy as np
import rclpy

from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import WrenchStamped
from scipy.spatial.transform import Rotation as R

# ----------------------------------------------------------
# Fix path for utils
# ----------------------------------------------------------

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
# Safety Monitor
# ==========================================================

class SafetyMonitor:

    def __init__(self, wrench):

        self.wrench = wrench

        self.force_stop = 10.0
        self.torque_stop = 2.0

        self.force_warn = 7.0
        self.torque_warn = 1.5

    def stop(self):

        if self.wrench.force_mag() > self.force_stop:
            return True

        if self.wrench.torque_mag() > self.torque_stop:
            return True

        return False

    def warning(self):

        if self.wrench.force_mag() > self.force_warn:
            return True

        if self.wrench.torque_mag() > self.torque_warn:
            return True

        return False


# ==========================================================
# Exploration Motion Generator
# ==========================================================

class ExplorationMotion:

    def __init__(self):

        self.xy_amp = 0.0025
        self.z_amp = 0.0015

        self.xy_freq = 2.0
        self.z_freq = 3.0

        self.last_jerk = 0
        self.jerk_interval = 3.0
        self.jerk_mag = 0.0002

    def compute(self, t, warning):

        amp_xy = self.xy_amp
        amp_z = self.z_amp

        if warning:
            amp_xy *= 0.5
            amp_z *= 0.5

        x = amp_xy * np.sin(2*np.pi*self.xy_freq*t)
        y = amp_xy * np.cos(2*np.pi*self.xy_freq*t)
        z = amp_z * np.sin(2*np.pi*self.z_freq*t)

        jerk = 0

        if t - self.last_jerk > self.jerk_interval:
            jerk = self.jerk_mag
            self.last_jerk = t

        return np.array([x, y, z + jerk])


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

    safety = SafetyMonitor(wrench)

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

    START_POS = [0.5603, -0.0432, 0.1778]

    START_QUAT = [
        0.904058,
        -0.426503,
        -0.026731,
        0.007625
    ]

    base_pos = np.array(START_POS)
    base_R = R.from_quat(START_QUAT)

    print("Opening gripper")

    gripper.open_gripper(width=0.08)

    time.sleep(1.0)

    print("Moving to probing pose")

    move_and_wait(robot, executor, START_POS, START_QUAT)

    print("Closing gripper")

    gripper.close_gripper(width=0.01, force=50)

    time.sleep(1.0)

    print("Starting probing motion")

    motion = ExplorationMotion()

    start_time = time.time()

    try:

        while rclpy.ok():

            executor.spin_once(timeout_sec=0.01)

            if safety.stop():
                print("Safety threshold exceeded — stopping probe")
                break

            t = time.time() - start_time

            delta = motion.compute(t, safety.warning())

            pos = base_pos + delta

            robot.set_target(
                pos.tolist(),
                base_R.as_quat().tolist()
            )

    finally:

        robot.publish_zero_velocity()

        robot.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":
    main()