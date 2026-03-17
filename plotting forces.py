#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped

import matplotlib.pyplot as plt
from collections import deque
import time


class FrankaForcePlotter(Node):

    def __init__(self):

        super().__init__('franka_force_plotter')

        # Subscribe directly to Franka wrench topic
        self.subscription = self.create_subscription(
            WrenchStamped,
            '/NS_1/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame',
            self.callback,
            10
        )

        self.get_logger().info("Franka Force Plotter Started")

        # Data buffers
        self.max_points = 200
        self.fx = deque(maxlen=self.max_points)
        self.fy = deque(maxlen=self.max_points)
        self.fz = deque(maxlen=self.max_points)
        self.t = deque(maxlen=self.max_points)

        self.start_time = time.time()

        # Plot setup
        plt.ion()

        self.fig, self.ax = plt.subplots()
        plt.show(block=False)

        self.line_fx, = self.ax.plot([], [], label="Fx")
        self.line_fy, = self.ax.plot([], [], label="Fy")
        self.line_fz, = self.ax.plot([], [], label="Fz")

        self.ax.set_title("Franka Force (Fx Fy Fz)")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Force (N)")
        self.ax.legend()
        self.ax.grid(True)

    def callback(self, msg):

        t = time.time() - self.start_time

        fx = msg.wrench.force.x
        fy = msg.wrench.force.y
        fz = msg.wrench.force.z

        # store values
        self.t.append(t)
        self.fx.append(fx)
        self.fy.append(fy)
        self.fz.append(fz)

        self.update_plot()

    def update_plot(self):

        self.line_fx.set_data(self.t, self.fx)
        self.line_fy.set_data(self.t, self.fy)
        self.line_fz.set_data(self.t, self.fz)

        self.ax.relim()
        self.ax.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def main():

    rclpy.init()

    node = FrankaForcePlotter()

    try:
        while rclpy.ok():

            rclpy.spin_once(node, timeout_sec=0.1)

            plt.pause(0.01)

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()