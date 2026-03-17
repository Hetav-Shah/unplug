#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
import numpy as np
import rclpy

from rclpy.executors import MultiThreadedExecutor
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import WrenchStamped

# ----------------------------------------------------------
# Fix Python path for utils
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
# Move helper (blocking)
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
# Force Monitor + Offset Calibration
# ==========================================================

class ForceMonitor:

    def __init__(self):

        self.force = np.zeros(3)
        self.torque = np.zeros(3)

        self.offset_force = np.zeros(3)
        self.offset_torque = np.zeros(3)

        self.samples = []
        self.offset_ready = False

    def callback(self, msg: WrenchStamped):

        raw_force = np.array([
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z
        ])

        raw_torque = np.array([
            msg.wrench.torque.x,
            msg.wrench.torque.y,
            msg.wrench.torque.z
        ])

        if not self.offset_ready:

            self.samples.append((raw_force, raw_torque))

            if len(self.samples) > 200:

                forces = [s[0] for s in self.samples]
                torques = [s[1] for s in self.samples]

                self.offset_force = np.mean(forces, axis=0)
                self.offset_torque = np.mean(torques, axis=0)

                self.offset_ready = True

                print("Wrench offset calibrated")

        else:

            self.force = raw_force - self.offset_force
            self.torque = raw_torque - self.offset_torque


# ==========================================================
# Safety Monitor
# ==========================================================

class SafetyMonitor:

    def __init__(self, monitor):

        self.monitor = monitor

        self.force_limit = 10.0
        self.torque_limit = 2.0

    def safe(self):

        f = np.linalg.norm(self.monitor.force)
        t = np.linalg.norm(self.monitor.torque)

        if f > self.force_limit:
            return False

        if t > self.torque_limit:
            return False

        return True


# ==========================================================
# Axis Prober
# ==========================================================

class AxisProber:

    def __init__(self, robot, executor, monitor, safety):

        self.robot = robot
        self.executor = executor
        self.monitor = monitor
        self.safety = safety

        self.step = 0.00005
        self.max_disp = 0.0008

    def probe_axis(self, axis, base_pos, base_R):

        print(f"\nProbing axis {axis}")

        displacement = 0.0

        force_history = []

        pos = base_pos.copy()

        while displacement < self.max_disp:

            self.executor.spin_once(timeout_sec=0.01)

            if not self.safety.safe():

                print("Safety limit reached")
                break

            if axis == "X":
                pos += np.array([self.step,0,0])

            elif axis == "Y":
                pos += np.array([0,self.step,0])

            elif axis == "Z":
                pos += np.array([0,0,self.step])

            self.robot.set_target(pos.tolist(), base_R.as_quat().tolist())

            force_mag = np.linalg.norm(self.monitor.force)

            force_history.append(force_mag)

            displacement += self.step

        self.robot.publish_zero_velocity()

        if len(force_history) < 2:
            return None

        delta_force = force_history[-1] - force_history[0]

        stiffness = delta_force / displacement

        return stiffness


# ==========================================================
# Analyzer
# ==========================================================

class ResistanceAnalyzer:

    def analyze(self, stiffness):

        valid = {k:v for k,v in stiffness.items() if v is not None}

        print("\nDirectional stiffness")

        for axis,val in valid.items():
            print(axis,"=",val)

        best = min(valid, key=lambda a: valid[a])

        print("\nEstimated separation axis:",best)


# ==========================================================
# Main
# ==========================================================

def main():

    rclpy.init()

    gripper = FrankaGripperController()

    monitor = ForceMonitor()

    node = rclpy.create_node("force_monitor")

    node.create_subscription(
        WrenchStamped,
        "/NS_1/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame",
        monitor.callback,
        10
    )

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

        max_cartesian_vel=0.015,

        dt=0.01,
        damping=0.01
    )

    executor = MultiThreadedExecutor()

    executor.add_node(robot)
    executor.add_node(node)

    safety = SafetyMonitor(monitor)

    START_POS = np.array([0.5603,-0.0432,0.2178])

    START_QUAT = [
        0.904058,
        -0.426503,
        -0.026731,
        0.007625
    ]

    base_R = R.from_quat(START_QUAT)

    print("Moving to probing pose")

    move_and_wait(robot, executor, START_POS, START_QUAT)

    print("Calibrating wrench")

    while not monitor.offset_ready:
        executor.spin_once(timeout_sec=0.01)

    print("Closing gripper")

    gripper.close_gripper(width=0.04, force=25)

    time.sleep(1)

    prober = AxisProber(robot, executor, monitor, safety)

    analyzer = ResistanceAnalyzer()

    axes = ["X","Y","Z"]

    stiffness_results = {}

    for axis in axes:

        k = prober.probe_axis(axis, START_POS.copy(), base_R)

        stiffness_results[axis] = k

        time.sleep(1)

    analyzer.analyze(stiffness_results)

    robot.destroy_node()
    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()