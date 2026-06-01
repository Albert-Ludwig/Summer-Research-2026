# Operation Notes

## Launch the environment

### structure:

```
Terminal 0: for testing.
Terminal 1: for Gazebo.
Terminal 2: for Nav2.
Terminal 3: for SLAM.
Terminal 4: for RViz.
```

#### Terminal 0: only run the test command.

```Bash
unset ROS_LOCALHOST_ONLY
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
source /opt/ros/jazzy/setup.bash

pkill -f slam_toolbox
pkill -f nav2
pkill -f lifecycle_manager
pkill -f rviz2
pkill -f clearpath_gz
pkill -f gz
pkill -f gazebo

ros2 daemon stop
ros2 daemon start
```

check:

```Bash
ros2 node list
```

ideal case is no **exist of old/cpr_j100_0001/...** node.

#### Terminal 1: Gazebo

```Bash
unset ROS_LOCALHOST_ONLY
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
source /opt/ros/jazzy/setup.bash

ros2 launch clearpath_gz simulation.launch.py \
  setup_path:=$HOME/clearpath
```

After open the Gazebo, click play, and refill the topic as

```Bash
/cpr_j100_0001/cmd_vel
```

make this terminal as deamon, keep it running.

#### Terminal 0: check the Gazebo

check the topic:

```Bash
ros2 topic list | sort | grep -E "clock|scan|lidar|odom|joint|cmd_vel|tf|imu"
```

1. must to see all of the following topic:

```Bash
/clock
/cpr_j100_0001/cmd_vel
/cpr_j100_0001/platform/odom
/cpr_j100_0001/platform/odom/filtered
/cpr_j100_0001/sensors/lidar2d_0/scan
/cpr_j100_0001/tf
/cpr_j100_0001/tf_static
```

2. check odom->base_link:

```Bash
ros2 run tf2_ros tf2_echo odom base_link \
  --ros-args \
  -r /tf:=/cpr_j100_0001/tf \
  -r /tf_static:=/cpr_j100_0001/tf_static
```

if the transform is publishing continuously, means the **_odom → base_link ✅_**. After that press the ctrl+c.

3. check the lidar:

```Bash
ros2 topic echo /cpr_j100_0001/sensors/lidar2d_0/scan --once
```

if the LaserScan message is printed, means the **_lidar2d_0/scan ✅_**.

#### Terminal 2: Nav2:

This terminal is only for running the Nav2, which is a deamon process.

```Bash
unset ROS_LOCALHOST_ONLY
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
source /opt/ros/jazzy/setup.bash

ros2 launch clearpath_nav2_demos nav2.launch.py \
  setup_path:=$HOME/clearpath \
  use_sim_time:=true
```

#### Terminal 0: check Nav2:

1. check the Nav 2 nodes:

```Bash
ros2 node list | grep -E "planner|controller|bt_navigator|local_costmap|global_costmap|lifecycle"
```

should be able to see the following nodes:

```
planner、controller、costmap、lifecycle manager
```

2. check the local costmap, whether it can receive the signal from lidar:

```Bash
ros2 topic info /cpr_j100_0001/sensors/lidar2d_0/scan -v
```

It should be able to see:

```
Node name: local_costmap
Node namespace: /cpr_j100_0001/local_costmap
```

if is, then **_Nav2 local costmap ✅_**.

#### Terminal 3: SLAM:

This deamon terminal is only for running the SLAM.

```Bash
unset ROS_LOCALHOST_ONLY
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
source /opt/ros/jazzy/setup.bash

ros2 launch clearpath_nav2_demos slam.launch.py \
  setup_path:=$HOME/clearpath \
  use_sim_time:=true \
  scan_topic:=/cpr_j100_0001/sensors/lidar2d_0/scan
```

#### Terminal 0: check SLAM:

1. check the SLAM node:

```Bash
ros2 node list | grep -E "slam|toolbox"
```

should be able to see like:

```
/cpr_j100_0001/slam_toolbox
```

2. Check whether the slam_toolbox can subscribe the lidar:

```Bash
ros2 topic info /cpr_j100_0001/sensors/lidar2d_0/scan -v
```

It should be able to see:

```
Node name: slam_toolbox
Node namespace: /cpr_j100_0001
```

3. Check the map topic:

```Bash
ros2 topic info /cpr_j100_0001/map -v
```

Wish to see:

```
Publisher count: 1
```

4. Check map->odom:

```Bash
ros2 run tf2_ros tf2_echo map odom \
  --ros-args \
  -r /tf:=/cpr_j100_0001/tf \
  -r /tf_static:=/cpr_j100_0001/tf_static
```

if the terminal is printing continuously, means the **_map → odom ✅_**. After that press the ctrl+c.

#### Terminal 4: RViz:

This deamon terminal is only for running the RViz.

```Bash
unset ROS_LOCALHOST_ONLY
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
source /opt/ros/jazzy/setup.bash

ros2 launch clearpath_viz view_navigation.launch.py \
  setup_path:=$HOME/clearpath \
  namespace:=/cpr_j100_0001 \
  use_sim_time:=true
```

Set up the Rviz for

```
Global Options → Fixed Frame = map
```

if the warning happens (like map becomes red) then change it to:

```
Fixed Frame = odom
```

and back to terminal 0 to check:

```Bash
ros2 topic info /cpr_j100_0001/map -v
ros2 run tf2_ros tf2_echo map odom \
  --ros-args \
  -r /tf:=/cpr_j100_0001/tf \
  -r /tf_static:=/cpr_j100_0001/tf_static
```

### try to run the Nav 2 Goal:

when RViz shows:

```
Global Status: Ok
Fixed Frame: map
RobotModel: Ok
LaserScan: Ok
Map: Ok
```

then run.

If no reaction of Nav2 then:

#### Terminal 0 check action server

```Bash
ros2 action list | grep navigate
```

should be able to see:

```
/cpr_j100_0001/navigate_to_pose
```

#### Terminal 0 check whether the Nav2 can send the speed command to Gazebo:

```Bash
ros2 topic echo /cpr_j100_0001/cmd_vel
```

and click Nav2 Goal again.

Case A: cmd_vel is printing but the cart is not moving:
check

```Bash
ros2 topic info /cpr_j100_0001/cmd_vel -v
ros2 topic info /cpr_j100_0001/platform/cmd_vel -v
```

Case B: not printing cmd_vel:
means the Nav2Goal is not working, check the Nav2 log in terminal 2.
check:

```Bash
ros2 lifecycle nodes
ros2 node list | grep -E "bt_navigator|controller|planner|behavior|waypoint|smoother"
```
