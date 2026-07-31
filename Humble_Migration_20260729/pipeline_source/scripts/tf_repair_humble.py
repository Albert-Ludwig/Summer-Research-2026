#!/usr/bin/env python3
"""Publish the minimal J100 simulation TF only when the native TF is absent."""

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from tf2_msgs.msg import TFMessage


ROBOT_NS = "/cpr_j100_0001"


class J100TfRepair(Node):
    def __init__(self) -> None:
        super().__init__("j100_tf_repair_humble_sim")

        self.tf_global = self.create_publisher(TFMessage, "/tf", 100)
        self.tf_namespaced = self.create_publisher(
            TFMessage, f"{ROBOT_NS}/tf", 100
        )

        static_qos = QoSProfile(depth=10)
        static_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        static_qos.reliability = ReliabilityPolicy.RELIABLE
        self.tf_static_global = self.create_publisher(
            TFMessage, "/tf_static", static_qos
        )
        self.tf_static_namespaced = self.create_publisher(
            TFMessage, f"{ROBOT_NS}/tf_static", static_qos
        )

        self.create_subscription(
            Odometry,
            f"{ROBOT_NS}/platform/odom",
            self.odom_callback,
            50,
        )
        self.create_timer(1.0, self.publish_static)
        self.publish_static()

    def static_transform(
        self, parent: str, child: str, z: float = 0.0
    ) -> TransformStamped:
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = parent
        transform.child_frame_id = child
        transform.transform.translation.z = z
        transform.transform.rotation.w = 1.0
        return transform

    def publish_static(self) -> None:
        message = TFMessage()
        message.transforms = [
            self.static_transform("base_link", "chassis_link"),
            self.static_transform("chassis_link", "lidar2d_0_laser", 0.25),
        ]
        self.tf_static_global.publish(message)
        self.tf_static_namespaced.publish(message)

    def odom_callback(self, message: Odometry) -> None:
        transform = TransformStamped()
        transform.header.stamp = message.header.stamp
        transform.header.frame_id = message.header.frame_id or "odom"
        transform.child_frame_id = message.child_frame_id or "base_link"
        transform.transform.translation.x = message.pose.pose.position.x
        transform.transform.translation.y = message.pose.pose.position.y
        transform.transform.translation.z = message.pose.pose.position.z
        transform.transform.rotation = message.pose.pose.orientation
        output = TFMessage(transforms=[transform])
        self.tf_global.publish(output)
        self.tf_namespaced.publish(output)


def main() -> None:
    rclpy.init()
    node = J100TfRepair()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
