#!/usr/bin/env bash

set -e

source /opt/ros/humble/setup.bash

unset ROS_DOMAIN_ID
unset ROS_LOCALHOST_ONLY
unset ROS_AUTOMATIC_DISCOVERY_RANGE
unset ROS_DISCOVERY_SERVER
unset CYCLONEDDS_URI

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

ROBOT_IP="${1:-192.168.131.1}"
HOST_IP="${2:-}"

if [[ -z "$HOST_IP" ]] && command -v ip >/dev/null 2>&1; then
  HOST_IP="$(ip route get "$ROBOT_IP" | sed -n 's/.* src \([^ ]*\).*/\1/p' | head -n 1)"
fi

if [[ -z "$HOST_IP" ]]; then
  HOST_IP="$(hostname -I | tr ' ' '\n' | grep '^192\.168\.131\.' | head -n 1)"
fi
TEMPLATE="/workspace/config_files/fastdds_robot_template.xml"
PROFILE="/tmp/fastdds_robot_wired_bridge.xml"

if [[ -z "$HOST_IP" ]]; then
  echo "ERROR: Could not determine the container source address for $ROBOT_IP." >&2
  exit 1
fi

sed \
  -e "s/ROBOT_IP/${ROBOT_IP}/g" \
  -e "s/HOST_IP/${HOST_IP}/g" \
  "$TEMPLATE" > "$PROFILE"

export FASTRTPS_DEFAULT_PROFILES_FILE="$PROFILE"

echo "Container source IP: $HOST_IP"
echo "Jackal IP: $ROBOT_IP"
echo "Fast DDS profile: $FASTRTPS_DEFAULT_PROFILES_FILE"

if grep -qE 'HOST_IP|ROBOT_IP' "$PROFILE"; then
  echo "ERROR: Fast DDS profile still contains unresolved placeholders." >&2
  exit 1
fi

echo "=== Fast DDS addresses ==="
grep -nE '<address>|<port>' "$PROFILE"

ros2 daemon stop >/dev/null 2>&1 || true
ros2 daemon start

echo "Waiting 15 seconds for discovery..."
sleep 15

echo "=== ROS 2 nodes ==="
ros2 node list

echo "=== ROS 2 topics ==="
ros2 topic list -t
