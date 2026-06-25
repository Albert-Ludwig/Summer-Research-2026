# Record for script & YAML file

## Original human model YAML

This is the original YAML file for the human model.
Only `agent2` is enabled.
`agent1` and `agent3` are kept in the file, but they are commented out in the `agents` list.

```yaml
hunav_loader:
  ros__parameters:
    yaml_base_name: cafe_agents
    simulator: Gazebo Fortress
    map: cafe
    publish_people: true
    global_goals:
      1:
        x: 2.080
        y: 5.605
      2:
        x: -2.934
        y: 2.878
      3:
        x: -3.402
        y: -3.457
      4:
        x: -3.373
        y: -8.962
      5:
        x: 0.262
        y: -4.972
      6:
        x: 2.629
        y: -1.363
      7:
        x: 0.069
        y: 1.942
      8:
        x: -2.107
        y: -1.363
    agents:
      #- agent1
      - agent2
      #- agent3
    agent1:
      id: 1
      group_id: -1
      skin: 4
      max_vel: 1.6000000000000001
      radius: 0.4
      goal_radius: 0.3
      cyclic_goals: true
      init_pose:
        x: 2.216
        y: 5.987
        z: 1.250
        h: -2.089
      behavior:
        type: Regular
        configuration: 0
        goal_force_factor: 2.0
        obstacle_force_factor: 10.0
        social_force_factor: 5.0
        other_force_factor: 20.0
      goals:
        - 1
        - 3
    agent2:
      id: 2
      group_id: -1
      skin: 3
      max_vel: 1.8
      radius: 0.4
      goal_radius: 0.3
      cyclic_goals: true
      init_pose:
        x: -2.793
        y: 3.261
        z: 1.250
        h: -1.614
      behavior:
        type: Regular
        configuration: 2
        goal_force_factor: 2.0
        obstacle_force_factor: 3.1
        social_force_factor: 5.7
        other_force_factor: 20.0
      goals:
        - 2
        - 4
    agent3:
      id: 3
      group_id: -1
      skin: 1
      max_vel: 1.5
      radius: 0.4
      goal_radius: 0.3
      cyclic_goals: true
      init_pose:
        x: 0.289
        y: -8.323
        z: 1.250
        h: 1.585
      behavior:
        type: Regular
        configuration: 0
        goal_force_factor: 2.0
        obstacle_force_factor: 10.0
        social_force_factor: 5.0
        other_force_factor: 20.0
      goals:
        - 5
        - 6
        - 7
        - 8
```

## HuNavSim YAML version map

This section records the YAML files used by HuNavSim.

### Active YAML

```text
office_random_3_agents.yaml
```

Meaning:

- Active launch config.
- Used by Terminal 1.
- Currently contains the 3-agent v1 config.
- Active agents: `agent1`, `agent2`, `agent3`.

Terminal 1 should use:

```bash
configuration_file:=office_random_3_agents.yaml
```

### Backup YAML files

`original.yaml`

- Original backup.
- Do not edit.
- Use only for rollback.
- Only the original active agent was enabled.

Path:

```bash
~/hunav_jazzy_ws/install/hunav_gazebo_fortress_wrapper/share/hunav_gazebo_fortress_wrapper/scenarios/yaml_backups_6_15/original.yaml
```

`6.15 v1 - cafe_agents_3_agents.yaml`

- Version `6.15 v1`.
- Based on `original.yaml`.
- Enables `agent1`, `agent2`, and `agent3`.
- Copied into `office_random_3_agents.yaml`.

Path:

```bash
~/hunav_jazzy_ws/install/hunav_gazebo_fortress_wrapper/share/hunav_gazebo_fortress_wrapper/scenarios/yaml_backups_6_15/6.15 v1 - cafe_agents_3_agents.yaml
```

### Old launch YAML

`cafe_agents.yaml`

- Old original launch config name.
- Used by the old Terminal 1 command.
- Not used for the current 3-agent test.
- It is a symlink to the source scenario YAML.

Path:

```bash
~/hunav_jazzy_ws/install/hunav_gazebo_fortress_wrapper/share/hunav_gazebo_fortress_wrapper/scenarios/cafe_agents.yaml
```

Symlink target:

```bash
~/hunav_jazzy_ws/src/hunav_gazebo_fortress_wrapper/scenarios/cafe_agents.yaml
```

### Rule

For current 3-agent testing:

```bash
configuration_file:=office_random_3_agents.yaml
```

For original behavior rollback:

```bash
configuration_file:=cafe_agents.yaml
```

Or restore:

```text
original.yaml -> office_random_3_agents.yaml
```

## office_random_3_agents.yaml

This YAML enables `agent1`, `agent2`, and `agent3`.

```yaml
hunav_loader:
  ros__parameters:
    yaml_base_name: cafe_agents
    simulator: Gazebo Fortress
    map: cafe
    publish_people: true
    global_goals:
      1:
        x: 2.080
        y: 5.605
      2:
        x: -2.934
        y: 2.878
      3:
        x: -3.402
        y: -3.457
      4:
        x: -3.373
        y: -8.962
      5:
        x: 0.262
        y: -4.972
      6:
        x: 2.629
        y: -1.363
      7:
        x: 0.069
        y: 1.942
      8:
        x: -2.107
        y: -1.363
    agents:
      - agent1
      - agent2
      - agent3
    agent1:
      id: 1
      group_id: -1
      skin: 4
      max_vel: 1.6000000000000001
      radius: 0.4
      goal_radius: 0.3
      cyclic_goals: true
      init_pose:
        x: 2.216
        y: 5.987
        z: 1.250
        h: -2.089
      behavior:
        type: Regular
        configuration: 0
        goal_force_factor: 2.0
        obstacle_force_factor: 10.0
        social_force_factor: 5.0
        other_force_factor: 20.0
      goals:
        - 1
        - 3
    agent2:
      id: 2
      group_id: -1
      skin: 3
      max_vel: 1.8
      radius: 0.4
      goal_radius: 0.3
      cyclic_goals: true
      init_pose:
        x: -2.793
        y: 3.261
        z: 1.250
        h: -1.614
      behavior:
        type: Regular
        configuration: 2
        goal_force_factor: 2.0
        obstacle_force_factor: 3.1
        social_force_factor: 5.7
        other_force_factor: 20.0
      goals:
        - 2
        - 4
    agent3:
      id: 3
      group_id: -1
      skin: 1
      max_vel: 1.5
      radius: 0.4
      goal_radius: 0.3
      cyclic_goals: true
      init_pose:
        x: 0.289
        y: -8.323
        z: 1.250
        h: 1.585
      behavior:
        type: Regular
        configuration: 0
        goal_force_factor: 2.0
        obstacle_force_factor: 10.0
        social_force_factor: 5.0
        other_force_factor: 20.0
      goals:
        - 5
        - 6
        - 7
        - 8
```

## Launch file change for 3-agent YAML

Only change Terminal 1 in the launch file.
Keep all other terminals unchanged.

Use this Terminal 1 command:

```bash
source /opt/ros/jazzy/setup.bash
cd ~/hunav_jazzy_ws
source install/setup.bash

export HUNAV_SHARE="$HOME/hunav_jazzy_ws/install/hunav_gazebo_fortress_wrapper/share/hunav_gazebo_fortress_wrapper"
export HUNAV_LIB="$HOME/hunav_jazzy_ws/install/hunav_gazebo_fortress_wrapper/lib"

export GZ_SIM_SYSTEM_PLUGIN_PATH="$HUNAV_LIB${GZ_SIM_SYSTEM_PLUGIN_PATH:+:$GZ_SIM_SYSTEM_PLUGIN_PATH}"
export GZ_SIM_RESOURCE_PATH="$HUNAV_SHARE/worlds:/opt/ros/jazzy/share/clearpath_gz/worlds:/opt/ros/jazzy/share/clearpath_gz/meshes:/opt/ros/jazzy/share${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"
export GAZEBO_RESOURCE_PATH="$HUNAV_SHARE/worlds:/opt/ros/jazzy/share/clearpath_gz/worlds:/opt/ros/jazzy/share/clearpath_gz/meshes:/opt/ros/jazzy/share${GAZEBO_RESOURCE_PATH:+:$GAZEBO_RESOURCE_PATH}"

rm -f run_hunav_cold_start.log

ros2 launch hunav_gazebo_fortress_wrapper simulation_fortress.launch.py \
  environment_name:=office_no_sensors \
  configuration_file:=office_random_3_agents.yaml \
  robot_name:=cpr_j100_0001/robot \
  use_gazebo_obs:=false \
  use_navgoal_to_start:=false \
  global_frame_to_publish:=map \
  ignore_models:='ground_plane sun charge_dock office link visual collision surface base_link fixed_joint lump lidar chassis fender bracket sensor wheel cpr_j100_0001 robot' \
  update_rate:=20.0 \
  verbose:=false \
  2>&1 | tee run_hunav_cold_start.log
```

## office_2_agents.yaml

This YAML enables `agent1` and `agent2` in the office map.

```yaml
hunav_loader:
  ros__parameters:
    yaml_base_name: office_2_agents
    simulator: Gazebo Fortress
    map: office
    publish_people: true
    global_goals:
      1:
        x: -3.000
        y: 1.500
      2:
        x: 3.000
        y: 1.500
      3:
        x: 3.000
        y: -1.500
      4:
        x: -3.000
        y: -1.500
    agents:
      - agent1
      - agent2
    agent1:
      id: 1
      group_id: -1
      skin: 4
      max_vel: 0.45
      radius: 0.4
      goal_radius: 0.35
      cyclic_goals: true
      init_pose:
        x: -3.000
        y: 1.500
        z: 1.250
        h: 0.000
      behavior:
        type: Regular
        configuration: 0
        goal_force_factor: 2.0
        obstacle_force_factor: 10.0
        social_force_factor: 5.0
        other_force_factor: 20.0
      goals:
        - 2
        - 1
    agent2:
      id: 2
      group_id: -1
      skin: 3
      max_vel: 0.45
      radius: 0.4
      goal_radius: 0.35
      cyclic_goals: true
      init_pose:
        x: 3.000
        y: -1.500
        z: 1.250
        h: 3.142
      behavior:
        type: Regular
        configuration: 0
        goal_force_factor: 2.0
        obstacle_force_factor: 10.0
        social_force_factor: 5.0
        other_force_factor: 20.0
      goals:
        - 4
        - 3
```
