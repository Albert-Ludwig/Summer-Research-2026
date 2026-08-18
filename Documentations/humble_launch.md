# Humble Real Jackal Launch

## Runtime

```text
ROS: ROS 2 Humble
Container: jackal_robohub
Network: host
GPU: NVIDIA RTX 4070
WSL maximum memory: 6 GB
VNC: http://127.0.0.1:6084/vnc.html?autoconnect=1&resize=scale
```

## One-Command Start

Run in Windows PowerShell:

```powershell
python "C:\Users\Administrator\Documents\Summer Research 2026\Documentations\run_jackal_robohub.py"
```

The Python launcher starts:

```text
Ubuntu-22.04 WSL Docker
jackal_robohub
Jackal onboard localization through SSH and nahl_ws
RealSense people detection
SocialNavDiffusion and acados
Nav2 Goal bridge
/debug_cmd_vel -> /jackal1/cmd_vel adapter
RViz
```

Required before starting:

```text
Jackal workspace: /home/administrator/nahl_ws
Onboard ROS package: jackal_nav
Onboard map YAML: /home/administrator/nahl_ws/maps/final.yaml
Map topic: /jackal1/map
```

The teammate's tuned parameter files are used without modification.
The launcher sources `nahl_ws/install/setup.bash` and starts localization on the Jackal over SSH.
`jackal_nav` is not installed or launched inside the container.
The adapter starts only after localization, the policy, and the people detector are ready.
Default limits match the teammate policy: `1.0 m/s` linear,
`3.14 rad/s` angular, `1.5 m/s^2` linear acceleration, and
`3.14 rad/s^2` angular acceleration.
The launcher does not open an external browser by default.
PyTorch compilation is limited to one worker to reduce startup memory and CPU peaks.

In VS Code, run `Simple Browser: Show` and enter:

```text
http://127.0.0.1:6084/vnc.html?autoconnect=1&resize=scale
```

When `[READY] Real Jackal stack is running` appears, use RViz `Nav2 Goal`.
Press `Ctrl+C` once in PowerShell to stop the complete stack.

## One-Time Map Creation

Run onboard the Jackal before the first localization test:

```bash
cd ~/nahl_ws
source install/setup.bash
ros2 launch nav2_slam_toolbox slam.launch.py
```

Save from a second onboard shell after scanning the environment:

```bash
cd ~/nahl_ws
source install/setup.bash
mkdir -p ~/maps
cd ~/maps
ros2 run nav2_map_server map_saver_cli -f robohub_map \
  --ros-args -r map:=/jackal1/map
```

Optional map override:

```powershell
python "C:\Users\Administrator\Documents\Summer Research 2026\Documentations\run_jackal_robohub.py" --map-file /home/administrator/maps/robohub_map.yaml --map-topic /jackal1/map
```

## Launcher File

```text
Windows:
C:\Users\Administrator\Documents\Summer Research 2026\Documentations\run_jackal_robohub.py

Container mount:
/workspace/Documentations/run_jackal_robohub.py
```

This launcher does not start real-time SLAM, Gazebo, HuNav, simulation people, the Nav2 controller, or monitoring.
