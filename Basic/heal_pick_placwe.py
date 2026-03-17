#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

# ---------------------------------------------------------------------
# Make RC-DS root importable
# ---------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.dls_velocity_commander import DLSVelocityCommander
from utils.gripper_commands.heal_dynamixel_gripper import GripperController


# =====================================================================
# 3-POSE TRANSPORT SUPERVISOR
# =====================================================================

class HealThreePose(Node):

    def __init__(self):
        super().__init__("heal_three_pose_transport")

        # -------------------------
        # Gripper
        # -------------------------
        self.gripper = GripperController(verbose=True)

        # -------------------------
        # IK Controller (UNCHANGED)
        # -------------------------
        self.ik = DLSVelocityCommander(
            robot_id="heal",
            base_link="base_link",
            tip_link="end-effector",
            joint_names=["joint1","joint2","joint3","joint4","joint5","joint6"],
            target_pos=[0,0,0],
            target_quat=[0,0,0,1],
            joint_state_topic="/joint_states",
            velocity_command_topic="/velocity_controller/commands",
            robot_description_topic="/robot_description",
            ee_pose_topic=None,
            ee_pose_is_stamped=False,
            max_cartesian_vel=0.05,
            max_angular_vel=0.15,
            dt=0.01,
            damping=0.1,
        )

        # -------------------------
        # THREE POSITIONS ONLY
        # -------------------------

        # POS1 — approach / start
        self.pos1 = (
             [0.25888081215127906, 0.424028743866825, 0.20738242329963413 ],
            [0.5039159549861177, 0.4547362042747619, -0.4397693428688886, 0.5881212629253229],
        )

        # POS2 — place location
        self.pos2 = (
             [0.25888081215127906, 0.424028743866825, 0.10738242329963413 ],
            [0.5039159549861177, 0.4547362042747619, -0.4397693428688886, 0.5881212629253229],
        )

        # POS3 — lift after opening
        self.pos3 = (
             [0.25888081215127906, 0.424028743866825, 0.50738242329963413 ],
            [0.5039159549861177, 0.4547362042747619, -0.4397693428688886, 0.5881212629253229],
        )

        # -------------------------
        # State machine
        # -------------------------
        self.state = "MOVE_TO_POS1"
        self.wait_until = None

        self.get_logger().info("HEAL 3-Pose Transport started")

        self.timer = self.create_timer(0.05, self._tick)

    # ------------------------------------------------------------------
    def set_target(self, pos, quat):

        self.ik.target_pos_kdl.x(pos[0])
        self.ik.target_pos_kdl.y(pos[1])
        self.ik.target_pos_kdl.z(pos[2])

        self.ik.target_quat_xyzw[:] = quat
        self.ik.target_rot_kdl = self.ik.target_rot_kdl.Quaternion(*quat)

        self.ik.target_frame = self.ik.target_frame.Identity()
        self.ik.target_frame.p = self.ik.target_pos_kdl
        self.ik.target_frame.M = self.ik.target_rot_kdl

    # ------------------------------------------------------------------
    def _tick(self):

        if not self.ik._kdl_ready:
            return

        now = time.time()

        # -------------------------
        if self.state == "MOVE_TO_POS1":
            self.set_target(*self.pos1)

            if self.ik.has_converged():
                self.get_logger().info("Reached POS1")
                self.state = "MOVE_TO_POS2"

        # -------------------------
        elif self.state == "MOVE_TO_POS2":
            self.set_target(*self.pos2)

            if self.ik.has_converged():
                self.get_logger().info("Reached POS2 → opening gripper")
                self.ik.publish_zero_velocity()
                self.gripper.open_gripper()

                self.wait_until = now + 0.7
                self.state = "WAIT_AFTER_OPEN"

        # -------------------------
        elif self.state == "WAIT_AFTER_OPEN":
            if now > self.wait_until:
                self.state = "MOVE_TO_POS3"

        # -------------------------
        elif self.state == "MOVE_TO_POS3":
            self.set_target(*self.pos3)

            if self.ik.has_converged():
                self.state = "DONE"

        # -------------------------
        elif self.state == "DONE":
            self.ik.publish_zero_velocity()
            self.get_logger().info("Transport complete")
            self.destroy_timer(self.timer)


# =====================================================================
def main():
    rclpy.init()

    supervisor = HealThreePose()

    executor = MultiThreadedExecutor()
    executor.add_node(supervisor)
    executor.add_node(supervisor.ik)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            supervisor.ik.publish_zero_velocity()
        except Exception:
            pass
        supervisor.destroy_node()
        supervisor.ik.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
