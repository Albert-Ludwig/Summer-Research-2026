# Current Gazebo–Nav2 Runtime Interface Baseline for HuNavSim Integration

## 1. Objective

The purpose of this record is to document the currently working runtime interfaces of the Clearpath Jackal simulation and Nav2 stack before integrating HuNavSim pedestrians into the simulator environment.

The immediate goals are to:

1. Identify where laser scan, map, navigation goal, and velocity command topics are published and consumed.
2. Establish a working baseline for the current Gazebo–Nav2 system.
3. Identify the ROS interfaces that HuNavSim will need to interact with during later integration.
4. Prepare for testing alternative Gazebo worlds and, subsequently, dynamic pedestrian simulation.

> **Scope note:** This document records the runtime topic and TF interfaces verified through ROS 2 command-line inspection. Source-code-level review of launch files, parameter files, and HuNavSim porting changes remains a follow-up task.

---

## 2. Current Simulation Stack

| Component | Current Setup |
|---|---|
| Robot platform | Clearpath Jackal J100 |
| ROS distribution | ROS 2 Jazzy |
| Simulator | Gazebo Harmonic through `clearpath_gz` |
| Navigation framework | Nav2 |
| Mapping / SLAM | `slam_toolbox` |
| Robot namespace | `/cpr_j100_0001` |
| Current validation status | Full SLAM/Nav2 runtime interfaces detected |

### Namespace Observation

The major ROS topics used by the robot stack are under:

```text
/cpr_j100_0001
```

For example:

```text
/cpr_j100_0001/map
/cpr_j100_0001/goal_pose
/cpr_j100_0001/cmd_vel_nav
/cpr_j100_0001/platform/cmd_vel
/cpr_j100_0001/sensors/lidar2d_0/scan
```

However, the TF frame names observed in the runtime transform message are **not** namespaced:

```text
map → odom
```

This distinction is important for future HuNavSim configuration:

- ROS topic parameter example: `/cpr_j100_0001/goal_pose`
- TF global frame parameter example: `map`

---

## 3. LiDAR / Sensor Data Interface

### Verified ROS Interface

| Item | Verified Result |
|---|---|
| Laser scan topic | `/cpr_j100_0001/sensors/lidar2d_0/scan` |
| Message type | `sensor_msgs/msg/LaserScan` |
| Publisher | `/cpr_j100_0001/sensors/lidar2d_0_gz_bridge` |
| Active subscribers | `rviz2`, `slam_toolbox`, `local_costmap`, `global_costmap`, `collision_monitor` |

### Data Flow

```text
Gazebo LiDAR Sensor
        ↓
lidar2d_0_gz_bridge
        ↓
/cpr_j100_0001/sensors/lidar2d_0/scan
        ├── slam_toolbox
        ├── Nav2 local_costmap
        ├── Nav2 global_costmap
        ├── collision_monitor
        └── RViz
```

### Conclusion

The currently working navigation configuration directly consumes:

```text
/cpr_j100_0001/sensors/lidar2d_0/scan
```

An additional relay to `/scan` is **not required** for this working baseline.

This is relevant to HuNavSim because inserted pedestrians should eventually appear as dynamic obstacles in the simulated LiDAR stream and consequently affect Nav2 costmaps and collision monitoring.

---

## 4. Map Generation and Global Frame Interface

### Verified ROS Interface

| Item | Verified Result |
|---|---|
| Map topic | `/cpr_j100_0001/map` |
| Message type | `nav_msgs/msg/OccupancyGrid` |
| Publisher | `/cpr_j100_0001/slam_toolbox` |
| Subscribers | `rviz2`, `slam_toolbox`, `local_costmap`, `global_costmap` |
| Verified transform | `map → odom` |

### Map Data Flow

```text
/cpr_j100_0001/sensors/lidar2d_0/scan
        ↓
slam_toolbox
        ├── /cpr_j100_0001/map
        └── TF transform: map → odom
```

### TF Observation

A sampled message from:

```bash
ros2 topic echo /cpr_j100_0001/tf --once
```

contained:

```yaml
frame_id: map
child_frame_id: odom
```

### Conclusion

In the current simulation baseline:

- `slam_toolbox` generates the 2D occupancy grid map.
- The map is published on `/cpr_j100_0001/map`.
- `slam_toolbox` also supplies the global transform `map → odom`.
- The TF global frame relevant to future HuNavSim configuration is `map`, not `/cpr_j100_0001/map`.

---

## 5. Navigation Goal Input Interface

### Verified ROS Interface

| Item | Verified Result |
|---|---|
| Navigation goal topic | `/cpr_j100_0001/goal_pose` |
| Message type | `geometry_msgs/msg/PoseStamped` |
| Subscriber | `/cpr_j100_0001/bt_navigator` |
| Publisher behavior | A goal publisher is expected when a navigation goal is sent from RViz |

### Navigation Goal Flow

```text
RViz: 2D Nav Goal
        ↓
/cpr_j100_0001/goal_pose
        ↓
bt_navigator
        ↓
Nav2 planning and control pipeline
```

### Conclusion

The topic:

```text
/cpr_j100_0001/goal_pose
```

is the current navigation goal input interface for Nav2. It is a candidate topic for any HuNavSim option that starts pedestrian movement when the robot receives a navigation goal.

---

## 6. Robot Velocity Command Pipeline

### Verified Velocity Interfaces

| Stage | Topic | Publisher(s) | Subscriber(s) | Function |
|---|---|---|---|---|
| Nav2 navigation velocity output | `/cpr_j100_0001/cmd_vel_nav` | `controller_server`, `behavior_server` | `velocity_smoother` | Raw navigation / behavior velocity command from Nav2 |
| Smoothed velocity command | `/cpr_j100_0001/cmd_vel_smoothed` | `velocity_smoother` | `collision_monitor` | Velocity command after smoothing |
| Safety-filtered command input to mux | `/cpr_j100_0001/cmd_vel` | `collision_monitor`, `cmd_vel_bridge`, `docking_server` | `twist_mux` | Velocity stream considered by the command multiplexer |
| Final platform velocity input | `/cpr_j100_0001/platform/cmd_vel` | `twist_mux` | `platform_velocity_controller`, `cmd_vel_bridge` | Final motion command sent toward the simulated Jackal platform |

### Velocity Command Flow

```text
controller_server / behavior_server
        ↓
/cpr_j100_0001/cmd_vel_nav
        ↓
velocity_smoother
        ↓
/cpr_j100_0001/cmd_vel_smoothed
        ↓
collision_monitor
        ↓
/cpr_j100_0001/cmd_vel
        ↓
twist_mux
        ↓
/cpr_j100_0001/platform/cmd_vel
        ↓
platform_velocity_controller
        ↓
Simulated Jackal motion in Gazebo
```

### Conclusion

Nav2 does not command the simulated platform directly. The navigation velocity passes through:

1. Nav2 control / behavior output,
2. velocity smoothing,
3. collision monitoring,
4. velocity multiplexing,
5. the platform velocity controller.

This pipeline is important for HuNavSim integration because simulated pedestrians may alter LiDAR obstacle observations, which may subsequently affect costmaps, planner/controller decisions, and collision-monitor behavior.

---

## 7. Baseline Interface Summary

| Function | Current Interface | Responsible Node(s) | Relevance to HuNavSim Integration |
|---|---|---|---|
| Robot namespace | `/cpr_j100_0001` | Clearpath simulation configuration | HuNav-related ROS topic configuration must account for this namespace |
| LiDAR scan | `/cpr_j100_0001/sensors/lidar2d_0/scan` | `lidar2d_0_gz_bridge` | Pedestrians should become observable as dynamic obstacles |
| Map generation | `/cpr_j100_0001/map` | `slam_toolbox` | Existing mapping/navigation reference for simulator tests |
| Global TF frame | `map → odom` | `slam_toolbox` | Candidate HuNav global reference frame: `map` |
| Navigation goal input | `/cpr_j100_0001/goal_pose` | RViz → `bt_navigator` | Candidate trigger topic for pedestrian movement |
| Nav2 velocity output | `/cpr_j100_0001/cmd_vel_nav` | `controller_server`, `behavior_server` | Original navigation motion output |
| Smoothed velocity | `/cpr_j100_0001/cmd_vel_smoothed` | `velocity_smoother` | Intermediate processed velocity |
| Safety-filtered velocity | `/cpr_j100_0001/cmd_vel` | `collision_monitor` / command sources | Collision-aware command path |
| Platform velocity input | `/cpr_j100_0001/platform/cmd_vel` | `twist_mux` | Final command executed by the simulated robot |

---

## 8. Preliminary HuNavSim-Relevant Parameter Mapping

Based on the current runtime interfaces, the following parameter values are likely candidates for future HuNavSim wrapper configuration:

```yaml
navgoal_topic: /cpr_j100_0001/goal_pose
global_frame_to_publish: map
```

### Still To Be Confirmed

The following cannot yet be finalized from topic inspection alone:

```yaml
robot_name: <Gazebo entity/model name to be confirmed>
```

The Gazebo model/entity name of the Jackal must be confirmed during the simulator/wrapper integration stage.

---

## 9. Issues Identified and Resolved During Baseline Inspection

### Topic Configuration Issue

Earlier difficulties were caused by using or checking an incorrect scan/topic path. The currently working configuration directly uses:

```text
/cpr_j100_0001/sensors/lidar2d_0/scan
```

rather than a relayed root-level topic:

```text
/scan
```

### Namespace Clarification

The correct ROS topic paths are namespaced, for example:

```text
/cpr_j100_0001/map
/cpr_j100_0001/goal_pose
```

Querying root-level topics such as `/map` or `/goal_pose` does not represent the actual current stack configuration.

### TF Clarification

Although ROS topics are namespaced, the verified transform frames are:

```text
map
odom
```

not:

```text
cpr_j100_0001/map
cpr_j100_0001/odom
```

---

## 10. Next Steps

### Step 1: Validate a Different Gazebo World

Test whether the same Jackal + SLAM + Nav2 stack continues to work in an alternative Gazebo environment, such as an office-style world.

Validation criteria:

- Jackal spawns correctly.
- LiDAR topic remains available and subscribed by SLAM/Nav2.
- `slam_toolbox` publishes a map for the new environment.
- The `map → odom` transform is available.
- A navigation goal sent from RViz leads to robot motion.
- The velocity command pipeline remains intact.

### Step 2: Review Source/Launch Configuration

Identify the launch files and YAML parameter files responsible for:

- selecting the Gazebo world;
- configuring the LiDAR scan topic;
- configuring SLAM Toolbox map/frame inputs and outputs;
- configuring Nav2 velocity remappings;
- setting the robot namespace.

### Step 3: Investigate HuNavSim Porting Requirements

Prepare a compatibility analysis for integrating:

- `robotics-upo/hunav_sim`
- `robotics-upo/hunav_gazebo_fortress_wrapper` at `v2.0`

into the current:

```text
ROS 2 Jazzy + Gazebo Harmonic + Clearpath Jackal/Nav2
```

environment.

Expected investigation areas include:

- Gazebo Fortress-to-Harmonic plugin/API differences;
- ROS 2 Humble-to-Jazzy package and launch compatibility;
- insertion of pedestrian actors into the selected Gazebo world;
- connection of pedestrian simulation to the existing robot navigation baseline.

---

## 11. Working Baseline Conclusion

The current Clearpath Jackal simulation stack is operating under the ROS namespace `/cpr_j100_0001` on ROS 2 Jazzy and Gazebo Harmonic. Gazebo laser scan data is directly consumed by SLAM Toolbox and Nav2 obstacle-processing components through `/cpr_j100_0001/sensors/lidar2d_0/scan`, without requiring a `/scan` relay. SLAM Toolbox publishes the map on `/cpr_j100_0001/map` and provides the `map → odom` transform. Navigation goals enter Nav2 through `/cpr_j100_0001/goal_pose`. Robot motion commands pass from Nav2 through velocity smoothing, collision monitoring, and velocity multiplexing before reaching the simulated platform through `/cpr_j100_0001/platform/cmd_vel`.

This runtime baseline establishes the main ROS interfaces that must be preserved and/or referenced when adding HuNavSim pedestrians to the Gazebo Harmonic simulation environment.
