# Gazebo-Nav2 Baseline & interface record

## 1. Tested System

| Item                     | Verified Setup                     |
| ------------------------ | ---------------------------------- |
| Robot                    | Clearpath Jackal J100              |
| ROS 2 Distribution       | Jazzy                              |
| Simulator                | Gazebo Harmonic via `clearpath_gz` |
| Mapping                  | `slam_toolbox`                     |
| Navigation               | Nav2                               |
| Robot Namespace          | `/cpr_j100_0001`                   |
| Alternative World Tested | `office`                           |

## 2. Verified Runtime Interfaces

| Function                     | Topic / Frame                           | Responsible Node(s)                                | Interpretation                                           |
| ---------------------------- | --------------------------------------- | -------------------------------------------------- | -------------------------------------------------------- |
| LiDAR scan                   | `/cpr_j100_0001/sensors/lidar2d_0/scan` | `lidar2d_0_gz_bridge`                              | Direct input to SLAM and Nav2; no `/scan` relay required |
| Occupancy map                | `/cpr_j100_0001/map`                    | `slam_toolbox`                                     | 2D map consumed by RViz and costmaps                     |
| Global TF                    | `map -> odom`                           | `slam_toolbox` via `/cpr_j100_0001/tf`             | TF frames are not namespaced                             |
| Navigation goal              | `/cpr_j100_0001/goal_pose`              | `bt_navigator` subscribes                          | RViz/Nav2 goal interface                                 |
| Nav2 velocity output         | `/cpr_j100_0001/cmd_vel_nav`            | `controller_server`, `behavior_server`             | Initial navigation velocity command                      |
| Smoothed velocity            | `/cpr_j100_0001/cmd_vel_smoothed`       | `velocity_smoother`                                | Smoothed navigation velocity                             |
| Safety-filtered command path | `/cpr_j100_0001/cmd_vel`                | `collision_monitor` output; `twist_mux` subscribes | Velocity path before platform actuation                  |
| Final platform velocity      | `/cpr_j100_0001/platform/cmd_vel`       | `twist_mux`                                        | Command executed by simulated Jackal                     |

HuNavSim-relevant runtime interfaces:

```yaml
navgoal_topic: /cpr_j100_0001/goal_pose
global_frame_to_publish: map
```

## 3. Verified Runtime Flow

### Mapping and Obstacle Input

```text
Gazebo LiDAR
  -> /cpr_j100_0001/sensors/lidar2d_0/scan
      -> slam_toolbox
      -> local_costmap
      -> global_costmap
      -> collision_monitor
      -> RViz
```

### Map and Transform Output

```text
slam_toolbox
  -> /cpr_j100_0001/map
  -> TF: map -> odom
```

### Navigation Command Pipeline

```text
RViz Nav2 Goal
  -> /cpr_j100_0001/goal_pose
  -> bt_navigator
  -> controller_server / behavior_server
  -> /cpr_j100_0001/cmd_vel_nav
  -> velocity_smoother
  -> /cpr_j100_0001/cmd_vel_smoothed
  -> collision_monitor
  -> /cpr_j100_0001/cmd_vel
  -> twist_mux
  -> /cpr_j100_0001/platform/cmd_vel
  -> Gazebo Jackal motion
```

## 4. `office` World Validation Results

The Gazebo environment was changed to the `office` world while preserving the corrected LiDAR topic configuration:

```text
/cpr_j100_0001/sensors/lidar2d_0/scan
```

| Validation Item                                                   | Result       |
| ----------------------------------------------------------------- | ------------ |
| Jackal spawned in `office` world                                  | Passed       |
| RViz displayed robot, LaserScan, map, and costmap layers          | Passed       |
| `slam_toolbox` published `/cpr_j100_0001/map`                     | Passed       |
| LiDAR was consumed by SLAM, costmaps, collision monitor, and RViz | Passed       |
| `controller_server` lifecycle state                               | `active [3]` |
| `bt_navigator` lifecycle state                                    | `active [3]` |
| `velocity_smoother` lifecycle state                               | `active [3]` |
| Navigation goal interface available                               | Passed       |
| Nav2 velocity chain active                                        | Passed       |
| Non-zero final platform velocity observed                         | Passed       |

### Final Platform Velocity Sample

Topic:

```text
/cpr_j100_0001/platform/cmd_vel
```

```yaml
frame_id: base_link
twist:
  linear:
    x: 0.07967802882194519
  angular:
    z: 0.18409816920757294
```

The non-zero platform command confirms that Nav2 motion commands reached the simulated Jackal platform in the `office` environment.

## 5. Runtime Note

During final velocity topic inspection, the following Fast DDS shared-memory transport warning appeared:

```text
[RTPS_TRANSPORT_SHM Error] Failed init_port fastdds_port7030: open_and_lock_file failed
```

It did not prevent receipt of the final non-zero platform velocity command and did not block the validation result.

## 6. Source and Configuration References

### 6.1 Gazebo World Assets

The Gazebo world assets provided by the Clearpath simulator package are located under:

```text
/opt/ros/jazzy/share/clearpath_gz/worlds/
```

The identified world files are:

```text
construction.sdf
office.sdf
orchard.sdf
pipeline.sdf
solar_farm.sdf
warehouse.sdf
```

The alternative environment validated in the current simulation test corresponds to:

```text
/opt/ros/jazzy/share/clearpath_gz/worlds/office.sdf
```

The relevant Gazebo launch files were located under:

```text
/opt/ros/jazzy/share/clearpath_gz/launch/
```

including:

```text
gz_sim.launch.py
robot_spawn.launch.py
simulation.launch.py
```

### 6.2 Jackal J100 Navigation Configuration Files

The Clearpath navigation-demo configuration directory for the Jackal J100 is:

```text
/opt/ros/jazzy/share/clearpath_nav2_demos/config/j100/
```

The identified J100 configuration files are:

```text
slam.yaml
nav2.yaml
localization.yaml
```

The related launch files are located under:

```text
/opt/ros/jazzy/share/clearpath_nav2_demos/launch/
```

including:

```text
slam.launch.py
nav2.launch.py
localization.launch.py
```

### 6.3 SLAM Frame Configuration

The SLAM frame parameters are defined in:

```text
/opt/ros/jazzy/share/clearpath_nav2_demos/config/j100/slam.yaml
```

The identified parameters are:

```yaml
odom_frame: odom
map_frame: map
base_frame: base_link
scan_topic: /scan
```

These frame parameters are consistent with the runtime transform previously observed:

```text
map -> odom
```

### 6.4 Effective LiDAR Topic Selection

The base J100 SLAM parameter file contains:

```yaml
scan_topic: /scan
```

The following Clearpath launch files define the effective scan-topic selection logic:

```text
/opt/ros/jazzy/share/clearpath_nav2_demos/launch/slam.launch.py
/opt/ros/jazzy/share/clearpath_nav2_demos/launch/nav2.launch.py
/opt/ros/jazzy/share/clearpath_nav2_demos/launch/localization.launch.py
```

These launch files contain default logic equivalent to:

```python
if len(eval_scan_topic) == 0:
    eval_scan_topic = f'/{namespace}/sensors/lidar2d_0/scan'
```

For the current robot namespace:

```text
/cpr_j100_0001
```

the effective default scan topic resolves to:

```text
/cpr_j100_0001/sensors/lidar2d_0/scan
```

This matches the LiDAR topic observed during successful SLAM and Nav2 runtime operation. The validated configuration does not require an additional root-level relay topic:

```text
/scan
```

because the Clearpath launch logic can directly select the namespaced LiDAR topic:

```text
/cpr_j100_0001/sensors/lidar2d_0/scan
```

### 6.5 Nav2 Velocity Processing Configuration

The J100 Nav2 parameter file is located at:

```text
/opt/ros/jazzy/share/clearpath_nav2_demos/config/j100/nav2.yaml
```

The following velocity-processing components were identified in this file:

```text
velocity_smoother
collision_monitor
```

The collision-monitor configuration includes:

```yaml
base_frame_id: "base_link"
odom_frame_id: "odom"
cmd_vel_in_topic: "cmd_vel_smoothed"
```

This configuration is consistent with the runtime velocity-processing section previously observed:

```text
/cpr_j100_0001/cmd_vel_nav
  -> velocity_smoother
  -> /cpr_j100_0001/cmd_vel_smoothed
  -> collision_monitor
```

## 7. Source trace addendum velocity namespace

### 7.1 Collision-Monitor Velocity Output

The J100 Nav2 parameter file:

```text
/opt/ros/jazzy/share/clearpath_nav2_demos/config/j100/nav2.yaml
```

defines the collision-monitor velocity topics as:

```yaml
cmd_vel_in_topic: "cmd_vel_smoothed"
cmd_vel_out_topic: "cmd_vel"
```

This identifies the configuration source for the verified runtime segment:

```text
/cpr_j100_0001/cmd_vel_smoothed
  → collision_monitor
  → /cpr_j100_0001/cmd_vel
```

### 7.2 Robot Namespace Source

The current robot namespace is defined in:

```text
/home/ubuntu/clearpath/robot.yaml
```

```yaml
serial_number: j100-0001
namespace: cpr_j100_0001
```

The generated platform and sensor configuration files consistently use this namespace, including:

```text
/home/ubuntu/clearpath/platform/config/twist_mux.yaml
/home/ubuntu/clearpath/platform/launch/platform-service.launch.py
/home/ubuntu/clearpath/sensors/config/lidar2d_0.yaml
/home/ubuntu/clearpath/sensors/launch/lidar2d_0.launch.py
```

### 7.3 Generated LiDAR Topic Configuration

The generated LiDAR configuration file:

```text
/home/ubuntu/clearpath/sensors/config/lidar2d_0.yaml
```

defines the Gazebo scan topic as:

```yaml
gz_topic_name: /cpr_j100_0001/sensors/lidar2d_0/scan
```

This is consistent with the namespaced LiDAR topic used by the validated SLAM and Nav2 runtime stack.

### 7.4 ROS–Gazebo Velocity Bridge Location

The generated platform launch file:

```text
/home/ubuntu/clearpath/platform/launch/platform-service.launch.py
```

contains velocity bridge references between the ROS 2 command topic and the Gazebo robot model command topic:

```text
cpr_j100_0001/cmd_vel
/model/cpr_j100_0001/robot/cmd_vel
```

This identifies the generated platform launch file as part of the velocity-command interface between ROS 2 and the simulated Jackal model.

### 7.5. Conclusion table

| Confirmed Item                       | Result                                                           | Source Location                                                     |
| ------------------------------------ | ---------------------------------------------------------------- | ------------------------------------------------------------------- |
| Collision-monitor input topic        | `cmd_vel_smoothed`                                               | `/opt/ros/jazzy/share/clearpath_nav2_demos/config/j100/nav2.yaml`   |
| Collision-monitor output topic       | `cmd_vel`                                                        | `/opt/ros/jazzy/share/clearpath_nav2_demos/config/j100/nav2.yaml`   |
| Robot namespace                      | `cpr_j100_0001`                                                  | `/home/ubuntu/clearpath/robot.yaml`                                 |
| Robot serial number                  | `j100-0001`                                                      | `/home/ubuntu/clearpath/robot.yaml`                                 |
| Gazebo LiDAR scan topic              | `/cpr_j100_0001/sensors/lidar2d_0/scan`                          | `/home/ubuntu/clearpath/sensors/config/lidar2d_0.yaml`              |
| ROS–Gazebo command bridge references | `cpr_j100_0001/cmd_vel` and `/model/cpr_j100_0001/robot/cmd_vel` | `/home/ubuntu/clearpath/platform/launch/platform-service.launch.py` |
