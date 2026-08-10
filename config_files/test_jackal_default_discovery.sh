#!/usr/bin/env bash

set -e

source /opt/ros/humble/setup.bash

unset FASTRTPS_DEFAULT_PROFILES_FILE
unset FASTDDS_DEFAULT_PROFILES_FILE
unset ROS_LOCALHOST_ONLY
unset ROS_AUTOMATIC_DISCOVERY_RANGE
unset ROS_DISCOVERY_SERVER
unset CYCLONEDDS_URI

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID="${1:-0}"

ros2 daemon stop >/dev/null 2>&1 || true
ros2 daemon start

echo "Using default Fast DDS discovery on ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
echo "Waiting 15 seconds for discovery..."
sleep 15

echo "=== ROS 2 nodes ==="
ros2 node list

echo "=== ROS 2 topics ==="
ros2 topic list -t
