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

# ---------------------------------------------------------
# Fix python path for utils
# ---------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.dls_velocity_commander import DLSVelocityCommander
from utils.gripper_commands.franka_gripper import FrankaGripperController


# =========================================================
# Move helper (blocking)
# =========================================================

def move_and_wait(robot, executor, pos, quat):

    robot.set_target(pos, quat)
    robot.reset_goal_reached()

    while rclpy.ok():

        executor.spin_once(timeout_sec=0.01)

        if robot.goal_reached():
            break

    robot.publish_zero_velocity()

    time.sleep(0.5)


# =========================================================
# Force monitor with offset calibration
# =========================================================

class ForceMonitor:

    def __init__(self):

        self.force = np.zeros(3)
        self.torque = np.zeros(3)

        self.offset_force = np.zeros(3)
        self.offset_torque = np.zeros(3)

        self.samples = []
        self.offset_ready = False

        self.soft_limit = 8.0
        self.hard_limit = 12.0

        self.prev_force_mag = 0

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

    def safety_state(self):

        f_mag = np.linalg.norm(self.force)

        dF = f_mag - self.prev_force_mag
        self.prev_force_mag = f_mag

        if f_mag > self.hard_limit:
            return "STOP"

        if f_mag > self.soft_limit:
            return "SLOW"

        if dF > 3:
            return "BLOCKED"

        return "OK"


# =========================================================
# Axis probing
# =========================================================

class AxisProber:

    def __init__(self, robot, executor, monitor):

        self.robot = robot
        self.executor = executor
        self.monitor = monitor

        self.base_amplitude = 0.0003
        self.frequency = 1.5

    def probe(self, axis, base_pos, base_R, duration=4):

        print(f"\n--- Probing {axis} ---")

        amplitude = self.base_amplitude

        samples = []

        start = time.time()

        while rclpy.ok():

            self.executor.spin_once(timeout_sec=0.01)

            if not self.monitor.offset_ready:
                continue

            t = time.time() - start

            if t > duration:
                break

            state = self.monitor.safety_state()

            if state == "STOP":
                print("Force limit reached")
                break

            if state == "SLOW":
                amplitude *= 0.7

            if state == "BLOCKED":
                print("Axis constrained")
                break

            osc = amplitude * np.sin(2*np.pi*self.frequency*t)

            motion = np.zeros(3)

            quat = base_R.as_quat()

            if axis == "X":
                motion = np.array([osc,0,0])

            elif axis == "Y":
                motion = np.array([0,osc,0])

            elif axis == "Z":
                motion = np.array([0,0,osc])

            elif axis == "RX":
                rot = R.from_rotvec([osc,0,0])
                quat = (base_R * rot).as_quat()

            elif axis == "RY":
                rot = R.from_rotvec([0,osc,0])
                quat = (base_R * rot).as_quat()

            elif axis == "RZ":
                rot = R.from_rotvec([0,0,osc])
                quat = (base_R * rot).as_quat()

            pos = base_pos + motion

            self.robot.set_target(pos.tolist(), quat.tolist())

            samples.append([
                self.monitor.force.copy(),
                self.monitor.torque.copy()
            ])

        self.robot.publish_zero_velocity()

        return samples


# =========================================================
# Resistance analysis
# =========================================================

class ResistanceAnalyzer:

    def __init__(self):

        self.results = {}

    def analyze(self, axis, samples):

        if len(samples) == 0:
            print("No samples collected")
            self.results[axis] = None
            return

        forces = []
        torques = []

        for f,t in samples:

            forces.append(np.linalg.norm(f))
            torques.append(np.linalg.norm(t))

        self.results[axis] = {

            "mean_force": np.mean(forces),
            "peak_force": np.max(forces),
            "mean_torque": np.mean(torques),
            "peak_torque": np.max(torques)

        }

    def print_results(self):

        print("\n========== RESULTS ==========\n")

        for axis,data in self.results.items():

            if data is None:
                print(axis,"→ no data")
                continue

            print(axis)

            print("mean force :",round(data["mean_force"],3),"N")
            print("peak force :",round(data["peak_force"],3),"N")

            print("mean torque:",round(data["mean_torque"],3),"Nm")
            print("peak torque:",round(data["peak_torque"],3),"Nm")

            print()

        valid = {k:v for k,v in self.results.items() if v}

        best = min(valid, key=lambda a: valid[a]["mean_force"])

        print("Least resistance axis:",best)


# =========================================================
# MAIN
# =========================================================

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

    START_POS = [0.5603,-0.0432,0.2178]

    START_QUAT = [
        0.904058,
        -0.426503,
        -0.026731,
        0.007625
    ]

    base_R = R.from_quat(START_QUAT)

    print("Moving to probing pose")

    move_and_wait(robot, executor, START_POS, START_QUAT)

    print("Calibrating wrench offset")

    while not monitor.offset_ready:
        executor.spin_once(timeout_sec=0.01)

    print("Closing gripper")

    gripper.close_gripper(width=0.04, force=25)

    time.sleep(1)

    prober = AxisProber(robot, executor, monitor)

    analyzer = ResistanceAnalyzer()

    axes = ["X","Y","Z","RX","RY","RZ"]

    for axis in axes:

        samples = prober.probe(axis, np.array(START_POS), base_R)

        analyzer.analyze(axis, samples)

        time.sleep(1)

    analyzer.print_results()

    robot.destroy_node()
    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()