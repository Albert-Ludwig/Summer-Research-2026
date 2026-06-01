# Gazebo–Nav2 Baseline and `office` World Validation Record

## 1. Tested System

| Item | Verified Setup |
|---|---|
| Robot | Clearpath Jackal J100 |
| ROS 2 Distribution | Jazzy |
| Simulator | Gazebo Harmonic via `clearpath_gz` |
| Mapping | `slam_toolbox` |
| Navigation | Nav2 |
| Robot Namespace | `/cpr_j100_0001` |
| Alternative World Tested | `office` |

## 2. Verified Runtime Interfaces

| Function | Topic / Frame | Responsible Node(s) | Interpretation |
|---|---|---|---|
| LiDAR scan | `/cpr_j100_0001/sensors/lidar2d_0/scan` | `lidar2d_0_gz_bridge` | Direct input to SLAM and Nav2; no `/scan` relay required |
| Occupancy map | `/cpr_j100_0001/map` | `slam_toolbox` | 2D map consumed by RViz and costmaps |
| Global TF | `map → odom` | `slam_toolbox` via `/cpr_j100_0001/tf` | TF frames are not namespaced |
| Navigation goal | `/cpr_j100_0001/goal_pose` | `bt_navigator` subscribes | RViz/Nav2 goal interface |
| Nav2 velocity output | `/cpr_j100_0001/cmd_vel_nav` | `controller_server`, `behavior_server` | Initial navigation velocity command |
| Smoothed velocity | `/cpr_j100_0001/cmd_vel_smoothed` | `velocity_smoother` | Smoothed navigation velocity |
| Safety-filtered command path | `/cpr_j100_0001/cmd_vel` | `collision_monitor` output; `twist_mux` subscribes | Velocity path before platform actuation |
| Final platform velocity | `/cpr_j100_0001/platform/cmd_vel` | `twist_mux` | Command executed by simulated Jackal |

## 3. Verified Runtime Flow

### Mapping and Obstacle Input

```text
Gazebo LiDAR
  → /cpr_j100_0001/sensors/lidar2d_0/scan
      → slam_toolbox
      → local_costmap
      → global_costmap
      → collision_monitor
      → RViz
```

### Map and Transform Output

```text
slam_toolbox
  → /cpr_j100_0001/map
  → TF: map → odom
```

### Navigation Command Pipeline

```text
RViz Nav2 Goal
  → /cpr_j100_0001/goal_pose
  → bt_navigator
  → controller_server / behavior_server
  → /cpr_j100_0001/cmd_vel_nav
  → velocity_smoother
  → /cpr_j100_0001/cmd_vel_smoothed
  → collision_monitor
  → /cpr_j100_0001/cmd_vel
  → twist_mux
  → /cpr_j100_0001/platform/cmd_vel
  → Gazebo Jackal motion
```

## 4. `office` World Validation Results

The Gazebo environment was changed to the `office` world while preserving the corrected LiDAR topic configuration:

```text
/cpr_j100_0001/sensors/lidar2d_0/scan
```

| Validation Item | Result |
|---|---|
| Jackal spawned in `office` world | Passed |
| RViz displayed robot, LaserScan, map, and costmap layers | Passed |
| `slam_toolbox` published `/cpr_j100_0001/map` | Passed |
| LiDAR was consumed by SLAM, costmaps, collision monitor, and RViz | Passed |
| `controller_server` lifecycle state | `active [3]` |
| `bt_navigator` lifecycle state | `active [3]` |
| `velocity_smoother` lifecycle state | `active [3]` |
| Navigation goal interface available | Passed |
| Nav2 velocity chain active | Passed |
| Non-zero final platform velocity observed | Passed |

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

## 5. Key Conclusions

1. The working LiDAR interface is `/cpr_j100_0001/sensors/lidar2d_0/scan`; no root-level `/scan` relay is needed.
2. `slam_toolbox` publishes the map on `/cpr_j100_0001/map` and provides `map → odom`.
3. Nav2 receives goals through `/cpr_j100_0001/goal_pose`.
4. The verified motion-command pipeline is:

   ```text
   cmd_vel_nav → cmd_vel_smoothed → cmd_vel → platform/cmd_vel
   ```

5. The current ROS 2 Jazzy + Gazebo Harmonic + Jackal + SLAM Toolbox + Nav2 stack operates successfully in the `office` Gazebo world.
6. Verified HuNavSim-relevant interfaces are:

   ```yaml
   navgoal_topic: /cpr_j100_0001/goal_pose
   global_frame_to_publish: map
   ```

## 6. Runtime Note

During final velocity topic inspection, the following Fast DDS shared-memory transport warning appeared:

```text
[RTPS_TRANSPORT_SHM Error] Failed init_port fastdds_port7030: open_and_lock_file failed
```

It did not prevent receipt of the final non-zero platform velocity command and did not block the validation result.
