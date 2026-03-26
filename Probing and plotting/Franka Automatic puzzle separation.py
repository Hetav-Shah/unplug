#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import csv
import time
import numpy as np
import rclpy

from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import WrenchStamped
from scipy.spatial.transform import Rotation as R

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
        self.force        = np.zeros(3)
        self.torque       = np.zeros(3)
        self._bias_fz     = 0.0
        self._bias_torque = np.zeros(3)   # full torque bias for magnitude

    def tare(self):
        """Tare fz and full torque vector. Call once when robot is fully still."""
        self._bias_fz     = self.force[2]
        self._bias_torque = self.torque.copy()
        print(f"[Tare] fz_bias={self._bias_fz:.3f} N  "
              f"torque_bias={self._bias_torque}")

    @property
    def fz(self):
        return self.force[2] - self._bias_fz

    @property
    def tz(self):
        return self.torque[2] - self._bias_torque[2]

    @property
    def torque_mag(self):
        """Total torque magnitude with bias removed.
        More robust than tz alone — catches resistance in any rotational axis."""
        return float(np.linalg.norm(self.torque - self._bias_torque))

    def callback(self, msg: WrenchStamped):
        self.force  = np.array([msg.wrench.force.x,
                                 msg.wrench.force.y,
                                 msg.wrench.force.z])
        self.torque = np.array([msg.wrench.torque.x,
                                 msg.wrench.torque.y,
                                 msg.wrench.torque.z])


# ==========================================================
# Data Logger
# ==========================================================

class DataLogger:
    STATE_NAMES = {1: "PULL", 2: "CCW", 3: "CW"}

    def __init__(self, filepath: str):
        self.filepath    = filepath
        self.start_time  = time.time()
        self._file       = open(filepath, "w", newline="")
        self._writer     = csv.writer(self._file)
        self._writer.writerow([
            "timestamp_s", "state", "pos_z_m",
            "rot_angle_deg", "fz_N", "tz_Nm", "torque_mag_Nm",
            "v_z_mms", "v_rz_degs", "event"
        ])
        self._rot_origin = None
        print(f"[Logger] {filepath}")

    def _angle_deg(self, rot: R) -> float:
        if self._rot_origin is None:
            self._rot_origin = rot
            return 0.0
        return float(np.rad2deg((rot * self._rot_origin.inv()).magnitude()))

    def log(self, state: int, pos_z: float, rot: R,
            fz: float, tz: float, tmag: float = 0.0,
            v_z: float = 0.0, v_rz: float = 0.0,
            event: str = ""):
        t    = round(time.time() - self.start_time, 4)
        name = self.STATE_NAMES.get(state, str(state))
        ang  = round(self._angle_deg(rot), 4)
        self._writer.writerow([
            t, name, round(pos_z, 6), ang,
            round(fz, 4), round(tz, 4), round(tmag, 4),
            round(v_z * 1000, 4), round(np.rad2deg(v_rz), 4),
            event
        ])
        self._file.flush()
        if event:
            print(f"\n[{t:7.2f}s] [{name}] z={pos_z:.4f}m ang={ang:.2f}deg "
                  f"fz={fz:.3f}N tz={tz:.3f}Nm  *** {event} ***")
        else:
            print(f"[{t:7.2f}s] [{name}] z={pos_z:.4f}m ang={ang:.2f}deg "
                  f"fz={fz:.3f}N tz={tz:.3f}Nm tmag={tmag:.3f}Nm "
                  f"vz={v_z*1000:.4f}mm/s vrz={np.rad2deg(v_rz):.4f}deg/s",
                  end="\r")

    def close(self):
        self._file.close()
        print(f"\n[Logger] Saved → {self.filepath}")


# ==========================================================
# Helpers
# ==========================================================

def hard_stop(robot, executor, settle_sec=1.0):
    robot.publish_zero_velocity()
    deadline = time.time() + settle_sec
    while time.time() < deadline:
        executor.spin_once(timeout_sec=0.01)


def settle_and_tare(wrench, executor, settle_sec=3.0):
    print(f"[Tare] Settling {settle_sec}s...")
    deadline = time.time() + settle_sec
    while time.time() < deadline:
        executor.spin_once(timeout_sec=0.01)
    wrench.tare()
    print(f"[Tare] fz={wrench.fz:.4f} N  torque_mag={wrench.torque_mag:.4f} Nm  (~0 expected)")


# ==========================================================
# MAIN
# ==========================================================

def main():

    rclpy.init()

    wrench = WrenchMonitor()
    node   = rclpy.create_node("admittance_game_node")

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
        target_pos=[0, 0, 0],
        target_quat=[0, 0, 0, 1],
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

    START_POS  = np.array([0.5603, -0.0432, 0.2408])
    START_QUAT = [0.904058, -0.426503, -0.026731, 0.007625]

    # ---------------- SETUP ----------------

    gripper.open_gripper(width=0.08)
    time.sleep(1)

    robot.set_target(START_POS.tolist(), START_QUAT)
    while rclpy.ok():
        executor.spin_once(timeout_sec=0.01)
        if robot.goal_reached():
            break

    hard_stop(robot, executor, settle_sec=2.0)
    for _ in range(100):
        executor.spin_once(timeout_sec=0.01)

    gripper.close_gripper(width=0.04, force=25)
    time.sleep(1.5)

    # Tare ONCE after gripper close — fixed for entire run
    settle_and_tare(wrench, executor, settle_sec=2.0)

    # ---------------- LOGGER ----------------

    log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"franka_run_{int(time.time())}.csv"
    )
    logger = DataLogger(log_path)

    # ---------------- ADMITTANCE PARAMETERS ----------------
    #
    # Admittance law:
    #   v_z  = V_NOM_Z  + K_Z  * fz        → pull up, fz negative on resistance
    #   v_rz = V_NOM_RZ - K_RZ * torque_mag → rotate, torque_mag always positive
    #
    # When force is zero  → moves at nominal speed
    # When force is high  → velocity reduces, eventually reverses (natural retreat)
    # No thresholds. No spike detection. Physics does the work.

    V_NOM_Z   =  0.00001      # nominal Z pull speed [m/s]
    K_Z       =  0.000005     # Z admittance gain [m/s per N]
                               # barrier at 20N → v_z = 0.00001 - 0.000005*20 = -0.00009 (retreats)

    V_NOM_RZ  =  0.00001      # nominal rotation speed [rad/s]
    K_RZ      =  0.000005     # rotation admittance gain [rad/s per Nm]

    V_MAX_Z   =  0.00001      # clamp max upward [m/s]
    V_MIN_Z   = -0.00005      # clamp max retreat [m/s]
    V_MAX_RZ  =  0.00001      # clamp max rotation [rad/s]
    V_MIN_RZ  = -0.00005      # clamp max rotation retreat [rad/s]

    # How long to stay in each rotation phase before switching back to pull.
    # Since admittance handles resistance naturally, we just time-box rotation.
    # If tz resistance is high → robot naturally slows/retreats within the phase.
    # Rotation uses same stall logic as Z pull — no spike threshold needed.
    # torque_mag admittance naturally slows rotation when resistance builds.

    PULL_GOAL_Z = START_POS[2] + 0.010   # 10 mm goal

    dt = 0.01

    # ---------------- STATE DEFINITIONS ----------------

    STATE_PULL = 1
    STATE_CCW  = 2
    STATE_CW   = 3

    rot_sequence = [STATE_CCW, STATE_CW]
    rot_index    = 0

    pos           = START_POS.copy()
    rot           = R.from_quat(START_QUAT)
    current_state = STATE_PULL

    # Track Z progress to decide when to switch to rotation.
    # If Z admittance makes no progress for STALL_SEC → switch to rotation.
    STALL_SEC     = 5.0        # seconds without Z progress → stalled at barrier
    STALL_MIN_MM  = 0.01       # minimum progress to not be considered stalled [mm]

    print("Admittance control. No thresholds. Force drives velocity directly.")
    print(f"v_z  = {V_NOM_Z*1000:.4f} mm/s + {K_Z*1000:.4f} * fz")
    print(f"v_rz = {np.rad2deg(V_NOM_RZ):.4f} deg/s - {np.rad2deg(K_RZ):.4f} * tz")
    print(f"Sequence: PULL → CCW → PULL → CW → PULL → CCW ...\n")

    # ==================================================
    # STATE MACHINE
    # ==================================================

    try:
        while rclpy.ok():

            # --------------------------------------------------
            # STATE 1 — Z PULL (admittance)
            # v_z = V_NOM_Z + K_Z * fz  # fz negative on resistance → reduces v_z
            # Switches to rotation when stalled (no Z progress)
            # --------------------------------------------------
            if current_state == STATE_PULL:

                print("[1-PULL ] Entered. Re-taring...")
                settle_and_tare(wrench, executor, settle_sec=3.0)
                logger.log(1, pos[2], rot, wrench.fz, wrench.tz, wrench.torque_mag,
                           event="state_1_entered")

                stall_timer_start = time.time()
                z_at_stall_check  = pos[2]

                print(f"[1-PULL ] Pulling. Goal z={PULL_GOAL_Z:.4f} m")

                while rclpy.ok():
                    executor.spin_once(timeout_sec=0.01)

                    # Goal reached
                    if pos[2] >= PULL_GOAL_Z:
                        logger.log(1, pos[2], rot, wrench.fz, wrench.tz, wrench.torque_mag,
                                   event="GOAL_REACHED")
                        print(f"\n[1-PULL ] Goal reached! z={pos[2]:.4f} m")
                        hard_stop(robot, executor)
                        logger.close()
                        return

                    fz = wrench.fz
                    tz = wrench.tz

                    # Admittance law — fz directly modulates v_z
                    v_z = V_NOM_Z + K_Z * fz  # fz negative on resistance → reduces v_z
                    v_z = float(np.clip(v_z, V_MIN_Z, V_MAX_Z))

                    # Stall detection — if no Z progress for STALL_SEC → barrier
                    if time.time() - stall_timer_start > STALL_SEC:
                        progress_mm = (pos[2] - z_at_stall_check) * 1000
                        if progress_mm < STALL_MIN_MM:
                            logger.log(1, pos[2], rot, fz, tz, wrench.torque_mag,
                                       event=f"STALLED_progress={progress_mm:.4f}mm_switching_rotation")
                            print(f"\n[1-PULL ] Stalled. Progress={progress_mm:.4f}mm "
                                  f"in {STALL_SEC}s → switching to rotation")
                            hard_stop(robot, executor, settle_sec=0.5)
                            current_state = rot_sequence[rot_index % 2]
                            rot_index    += 1
                            print(f"[1-PULL ] → state {current_state} "
                                  f"({'CCW' if current_state == STATE_CCW else 'CW'})\n")
                            break
                        # Reset stall check
                        stall_timer_start = time.time()
                        z_at_stall_check  = pos[2]

                    logger.log(1, pos[2], rot, fz, tz, wrench.torque_mag, v_z=v_z)

                    pos[2] += v_z * dt
                    robot.set_target(pos.tolist(), rot.as_quat().tolist())

            # --------------------------------------------------
            # STATE 2 — ROTATE CCW (admittance)
            # v_rz = -V_NOM_RZ + K_RZ * tz  (negative = CCW)
            # Z stays at current position — admittance handles any fz
            # coupling by letting the controller absorb small forces.
            # --------------------------------------------------
            elif current_state == STATE_CCW:

                print("[2-CCW  ] Entered. Re-taring...")
                settle_and_tare(wrench, executor, settle_sec=3.0)
                logger.log(2, pos[2], rot, wrench.fz, wrench.tz, wrench.torque_mag,
                           event="state_2_entered")

                frozen_pos        = pos.copy()
                stall_timer_start = time.time()
                angle_at_stall    = R.from_quat(rot.as_quat())

                print(f"[2-CCW  ] Rotating CCW. Z fixed at {frozen_pos[2]:.4f} m. "
                      f"Stall detection: {STALL_SEC}s / {STALL_MIN_MM:.3f} deg")

                while rclpy.ok():
                    executor.spin_once(timeout_sec=0.01)

                    fz   = wrench.fz
                    tz   = wrench.tz
                    tmag = wrench.torque_mag

                    # Admittance — torque_mag reduces v_rz (always positive, slows rotation)
                    # CCW = negative direction
                    v_rz = -(V_NOM_RZ - K_RZ * tmag)
                    v_rz = float(np.clip(v_rz, V_MIN_RZ, V_MAX_RZ))

                    # Stall detection — same logic as Z pull
                    # If no angular progress for STALL_SEC → barrier → switch to pull
                    if time.time() - stall_timer_start > STALL_SEC:
                        angle_progress = np.rad2deg(
                            (rot * angle_at_stall.inv()).magnitude()
                        )
                        if angle_progress < STALL_MIN_MM:
                            logger.log(2, frozen_pos[2], rot, fz, tz, tmag,
                                       event=f"STALLED_ang={angle_progress:.4f}deg_switching_pull")
                            print(f"\n[2-CCW  ] Stalled. Progress={angle_progress:.4f}deg "
                                  f"in {STALL_SEC}s → switching to pull")
                            hard_stop(robot, executor, settle_sec=0.5)
                            current_state = STATE_PULL
                            break
                        stall_timer_start = time.time()
                        angle_at_stall    = R.from_quat(rot.as_quat())

                    logger.log(2, frozen_pos[2], rot, fz, tz, tmag, v_rz=v_rz)

                    dtheta = v_rz * dt
                    rot    = rot * R.from_rotvec([0, 0, dtheta])
                    robot.set_target(frozen_pos.tolist(), rot.as_quat().tolist())

            # --------------------------------------------------
            # STATE 3 — ROTATE CW (admittance)
            # Same as CCW but positive direction.
            # --------------------------------------------------
            elif current_state == STATE_CW:

                print("[3-CW   ] Entered. Re-taring...")
                settle_and_tare(wrench, executor, settle_sec=3.0)
                logger.log(3, pos[2], rot, wrench.fz, wrench.tz, wrench.torque_mag,
                           event="state_3_entered")

                frozen_pos        = pos.copy()
                stall_timer_start = time.time()
                angle_at_stall    = R.from_quat(rot.as_quat())

                print(f"[3-CW   ] Rotating CW. Z fixed at {frozen_pos[2]:.4f} m. "
                      f"Stall detection: {STALL_SEC}s / {STALL_MIN_MM:.3f} deg")

                while rclpy.ok():
                    executor.spin_once(timeout_sec=0.01)

                    fz   = wrench.fz
                    tz   = wrench.tz
                    tmag = wrench.torque_mag

                    # Admittance — torque_mag reduces v_rz, CW = positive
                    v_rz = +(V_NOM_RZ - K_RZ * tmag)
                    v_rz = float(np.clip(v_rz, V_MIN_RZ, V_MAX_RZ))

                    # Stall detection — same logic as Z pull
                    if time.time() - stall_timer_start > STALL_SEC:
                        angle_progress = np.rad2deg(
                            (rot * angle_at_stall.inv()).magnitude()
                        )
                        if angle_progress < STALL_MIN_MM:
                            logger.log(3, frozen_pos[2], rot, fz, tz, tmag,
                                       event=f"STALLED_ang={angle_progress:.4f}deg_switching_pull")
                            print(f"\n[3-CW   ] Stalled. Progress={angle_progress:.4f}deg "
                                  f"in {STALL_SEC}s → switching to pull")
                            hard_stop(robot, executor, settle_sec=0.5)
                            current_state = STATE_PULL
                            break
                        stall_timer_start = time.time()
                        angle_at_stall    = R.from_quat(rot.as_quat())

                    logger.log(3, frozen_pos[2], rot, fz, tz, tmag, v_rz=v_rz)

                    dtheta = v_rz * dt
                    rot    = rot * R.from_rotvec([0, 0, dtheta])
                    robot.set_target(frozen_pos.tolist(), rot.as_quat().tolist())

    finally:
        robot.publish_zero_velocity()
        logger.close()
        robot.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()