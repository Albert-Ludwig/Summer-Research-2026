#!/usr/bin/env bash

echo "=== Host ==="
hostname
uname -a

echo "=== OS ==="
cat /etc/os-release

echo "=== IPv4 interfaces ==="
ip -4 -o addr

echo "=== Installed ROS distributions ==="
find /opt/ros -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort

echo "=== Environment ==="
env | grep -E '^(ROS|RMW|FAST|CYCLONE)' | sort || true

echo "=== UDP port 11811 ==="
ss -lun | grep ':11811' || true

echo "=== ROS and discovery processes ==="
ps aux | grep -Ei 'fastdds|discovery|ros2|rosmaster|clearpath|jackal' | grep -v grep || true

echo "=== Relevant running services ==="
systemctl --type=service --state=running --no-pager 2>/dev/null |
  grep -Ei 'ros|dds|clearpath|jackal' || true

for setup_file in /opt/ros/*/setup.bash; do
  [[ -f "$setup_file" ]] || continue
  distro="$(basename "$(dirname "$setup_file")")"

  echo "=== ROS $distro nodes ==="
  bash -lc "source '$setup_file'; timeout 15 ros2 node list" || true

  echo "=== ROS $distro topics ==="
  bash -lc "source '$setup_file'; timeout 15 ros2 topic list -t" || true
done
