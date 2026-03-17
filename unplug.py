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
# Move helper
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
# Wrench Monitor
# =========================================================

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

            if len(self.samples) > 500:

                forces = [s[0] for s in self.samples]
                torques = [s[1] for s in self.samples]

                self.offset_force = np.mean(forces, axis=0)
                self.offset_torque = np.mean(torques, axis=0)

                self.offset_ready = True
                print("Wrench offset calibrated")

        else:

            self.force = raw_force - self.offset_force
            self.torque = raw_torque - self.offset_torque


# =========================================================
# Axis Prober
# =========================================================

class AxisProber:

    def __init__(self, robot, executor, monitor):

        self.robot = robot
        self.executor = executor
        self.monitor = monitor

        self.contact_threshold = 2.0
        self.force_limit = 5.0
        self.torque_limit = 2.0

        self.translation_probe = 0.002
        self.rotation_probe = 0.05

        self.frequency = 2.0
        self.approach_speed = 0.0004


    def probe(self, axis, base_pos, base_R, duration=4):

        print(f"\n--- Probing {axis} ---")

        samples = []

        pos = base_pos.copy()
        quat = base_R.as_quat()

        # ------------------------------
        # CONTACT SEARCH
        # ------------------------------

        print("Searching for contact")

        while rclpy.ok():

            self.executor.spin_once(timeout_sec=0.01)

            F = self.monitor.force
            force_mag = np.linalg.norm(F)

            if force_mag > self.contact_threshold:

                print("Contact detected:", round(force_mag,2))
                break

            motion = np.zeros(3)

            if axis == "X":
                motion = np.array([self.approach_speed,0,0])

            elif axis == "Y":
                motion = np.array([0,self.approach_speed,0])

            elif axis == "Z":
                motion = np.array([0,0,self.approach_speed])

            pos = pos + motion*0.01

            self.robot.set_target(pos.tolist(), quat.tolist())

        self.robot.publish_zero_velocity()
        time.sleep(0.2)

        # ------------------------------
        # MICRO PROBING
        # ------------------------------

        print("Micro probing")

        start = time.time()

        while rclpy.ok():

            self.executor.spin_once(timeout_sec=0.01)

            t = time.time() - start

            if t > duration:
                break

            osc = np.sin(2*np.pi*self.frequency*t)

            motion = np.zeros(3)
            quat = base_R.as_quat()

            if axis == "X":
                motion = np.array([self.translation_probe*osc,0,0])

            elif axis == "Y":
                motion = np.array([0,self.translation_probe*osc,0])

            elif axis == "Z":
                motion = np.array([0,0,self.translation_probe*osc])

            elif axis == "RX":
                rot = R.from_rotvec([self.rotation_probe*osc,0,0])
                quat = (base_R * rot).as_quat()

            elif axis == "RY":
                rot = R.from_rotvec([0,self.rotation_probe*osc,0])
                quat = (base_R * rot).as_quat()

            elif axis == "RZ":
                rot = R.from_rotvec([0,0,self.rotation_probe*osc])
                quat = (base_R * rot).as_quat()

            pos_cmd = pos + motion

            self.robot.set_target(pos_cmd.tolist(), quat.tolist())

            F = self.monitor.force
            T = self.monitor.torque

            if np.linalg.norm(F) > self.force_limit:

                print("Force safety stop:", round(np.linalg.norm(F),2))
                break

            if np.linalg.norm(T) > self.torque_limit:

                print("Torque safety stop:", round(np.linalg.norm(T),2))
                break

            samples.append([F.copy(), T.copy()])

        self.robot.publish_zero_velocity()

        return samples


# =========================================================
# Analyzer
# =========================================================

class ResistanceAnalyzer:

    def __init__(self):

        self.results = {}

    def analyze(self, axis, samples):

        if len(samples) == 0:

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

        trans_axes = ["X","Y","Z"]
        rot_axes = ["RX","RY","RZ"]

        valid_trans = {k:v for k,v in self.results.items()
                       if k in trans_axes and v is not None}

        valid_rot = {k:v for k,v in self.results.items()
                     if k in rot_axes and v is not None}

        if valid_trans:

            best_trans = min(
                valid_trans,
                key=lambda a: valid_trans[a]["mean_force"]
            )

            print("Best translation axis:", best_trans)

        if valid_rot:

            best_rot = min(
                valid_rot,
                key=lambda a: valid_rot[a]["mean_torque"]
            )

            print("Best rotation axis:", best_rot)


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

    START_POS = [0.5603,-0.0432,0.1808]

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