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
# Pick & Place Supervisor
# =====================================================================

class HealPickPlace(Node):
    def __init__(self):
        super().__init__("heal_pick_place_dls")

        # -------------------------
        # Gripper (UNCHANGED)
        # -------------------------
        self.gripper = GripperController(verbose=True)

        # -------------------------
        # IK Controller (UNCHANGED)
        # -------------------------
        self.ik = DLSVelocityCommander(
            robot_id="heal",
            base_link="base_link",
            tip_link="end-effector",
            joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
            target_pos=[0.0, 0.0, 0.0],      # overwritten by states
            target_quat=[0.0, 0.0, 0.0, 1.0],
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
        # TASK POSES
        # -------------------------
        self.pick_above = (
            [0.25888081215127906, 0.424028743866825, 0.20738242329963413 ],
            [0.5039159549861177, 0.4547362042747619, -0.4397693428688886, 0.5881212629253229],
        )

        self.pick = (
            [0.25888081215127906, 0.424028743866825, 0.10038242329963413 ],
            [0.5039159549861177, 0.4547362042747619, -0.4397693428688886, 0.5881212629253229],
        )

        self.place = (
           [-0.37005747830386804,0.35563428070427844, 0.24978147655366245 ],
           [0.12566453023829227, 0.6910997707373016, -0.026881947845148975, 0.7112432028546136],
        )

        # -------------------------
        # State machine
        # -------------------------
        self.state = "APPROACH_PICK"
        self.get_logger().info("HEAL Pick–Place started")

        self.timer = self.create_timer(0.05, self._tick)

    # ------------------------------------------------------------------
    # Helper
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
    # State Machine Tick
    # ------------------------------------------------------------------
    def _tick(self):
        if not self.ik._kdl_ready:
            return

        # -------------------------
        if self.state == "APPROACH_PICK":
            self.set_target(*self.pick_above)
            if self.ik.has_converged():
                self.get_logger().info("Reached above pick")
                self.state = "DESCEND_PICK"

        # -------------------------
        elif self.state == "DESCEND_PICK":
            self.set_target(*self.pick)
            if self.ik.has_converged():
                self.get_logger().info("At pick pose")
                self.state = "CLOSE_GRIPPER"

        # -------------------------
        elif self.state == "CLOSE_GRIPPER":
            self.ik.publish_zero_velocity()
            time.sleep(0.2)
            self.gripper.close_gripper()
            time.sleep(0.5)
            self.state = "LIFT"

        # -------------------------
        elif self.state == "LIFT":
            self.set_target(*self.pick_above)
            if self.ik.has_converged():
                self.get_logger().info("Object lifted")
                self.state = "MOVE_TO_PLACE"

        # -------------------------
        elif self.state == "MOVE_TO_PLACE":
            self.set_target(*self.place)
            if self.ik.has_converged():
                self.get_logger().info("At place pose")
                self.state = "OPEN_GRIPPER"

        # -------------------------
           # elif self.state == "OPEN_GRIPPER":
            #    self.ik.publish_zero_velocity()
            #    time.sleep(0.2)
             #   self.gripper.open_gripper()
            #    self.state = "DONE"

        # -------------------------
        elif self.state == "DONE":
            self.ik.publish_zero_velocity()
            self.get_logger().info("Pick–Place complete")
            self.destroy_timer(self.timer)


# =====================================================================
# main
# =====================================================================

def main():
    rclpy.init()
    
    supervisor = HealPickPlace()
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