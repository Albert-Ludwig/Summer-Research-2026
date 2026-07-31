# Humble Final Integration Test Report

Date: 2026-07-30

## Result

The ROS 2 Humble simulation stack passed both the no-goal safety test and an
explicit-goal end-to-end test. The tested chain was:

Gazebo Fortress + HuNav -> simulated Clearpath J100 -> odom/lidar/TF -> SLAM
-> SocialNavDiffusion GPU inference -> acados projection -> simulated cmd_vel.

The test used simulation isolation:

```text
ROS_DOMAIN_ID=73
ROS_LOCALHOST_ONLY=1
ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
FASTDDS_BUILTIN_TRANSPORTS=UDPv4
```

No real Jackal was connected. No command was sent outside the isolated
simulation domain.

## Delivered Files

- `run_final_social_nav_test_humble.py`: Humble one-click launcher and validator.
- `scripts/tf_repair_humble.py`: optional TF fallback; it was not needed in the
  successful run because native `odom -> base_link` worked.
- `config/clearpath_humble/robot.yaml`: isolated simulated J100 configuration.
- `experiment_setup/hunav/worlds/office_no_sensors.sdf`: Fortress-compatible
  office world.
- `experiment_setup/hunav/scenarios/office_2_agents_humble.yaml`: HuNav v1
  two-agent scenario.
- `HUMBLE_FINAL_TEST_REPORT.md`: this report.

The existing Jazzy launcher was not overwritten or modified.

## Minimal Compatibility Patches

Two HuNav patches are preserved under
`/workspace/Humble_Migration_20260729/hunav_humble_patches/`.

1. `simulation_fortress.launch.py`
   - Passes `plugin_position` as an integer `ParameterValue`.
   - Required because the v1 plugin declares an integer ROS parameter.

2. `hunav_agent_manager/bt_node.cpp`
   - Ignores an empty `/compute_agents` startup request.
   - Fixes a race where the manager initialized with zero agents before Gazebo
     actors were ready, later returned unnamed agents, and crashed the plugin.

Launcher fixes included:

- Humble setup paths and the independent Humble venv/acados environment.
- Fortress `ign gazebo` / `ign topic` and Humble bridge message syntax.
- Clearpath `setup_path` with the required trailing slash.
- `/people` validation after the J100 model exists.
- Correct explicit-goal YAML.
- Fail-fast process, topic, TF, and command checks.
- Process-group-only cleanup and single-Ctrl+C shutdown.
- Automated goal verification for nonzero simulated command, acados status 0,
  completed warm-up, and exactly one model load.

No SocialNavDiffusion model, checkpoint, policy algorithm, or controller
parameter was changed.

## Sources And Dependencies

### Source repositories

| Component | Source | Revision |
|---|---|---|
| HuNavSim | `https://github.com/robotics-upo/hunav_sim.git` | branch `v1.0-humble`, commit `d97ac2c96b5de1ef9cd8835f99718504a4a005ae` |
| Fortress wrapper | `https://github.com/robotics-upo/hunav_gazebo_fortress_wrapper.git` | branch `v1.0-humble`, commit `e6160a9d8a91f2ee4fde39ff5879507acec17cd8` |
| LightSFM | `https://github.com/robotics-upo/lightsfm.git` | commit `b30327cca189af2fb90443a5d0040cceb46d7195` |
| SocialNavDiffusion | `/workspace/SocialNavDiffusion_Inference` | mounted project source and checkpoint |
| ROS wrapper | `/home/ubuntu/waterloo_jackal_pipeline_repo` | migrated Humble source |
| acados | `/workspace/Humble_Migration_20260729/acados_source` | copied and rebuilt locally |

The migrated pipeline, `people_msgs`, and acados copies do not contain `.git`
metadata in the container. Their exact original commits cannot be re-verified
from this container. This is a provenance limitation, not a runtime blocker.

### APT sources and versions

Clearpath packages came from:

```text
deb https://packages.clearpathrobotics.com/stable/ubuntu jammy main
```

Installed versions used by the successful run:

```text
ros-humble-clearpath-simulator  1.3.3-1jammy.20260708.212659
ros-humble-clearpath-nav2-demos 1.0.0-1jammy.20260612.221712
ros-humble-clearpath-viz        1.2.1-1jammy.20260626.095646
ros-humble-slam-toolbox         2.6.10-1jammy.20260612.214828
ros-humble-nav2-behavior-tree   1.1.20-1jammy.20260607.075949
ros-humble-behaviortree-cpp     4.9.0-1jammy.20260605.154429
ros-humble-tf-transformations   1.1.1-1jammy.20260226.033650
python3-rich                    11.2.0-1
```

Platform:

```text
Ubuntu 22.04.5 LTS
ROS_DISTRO=humble
Gazebo Fortress 6.18.0
PyTorch 2.12.1+cu130
GPU: NVIDIA GeForce RTX 4070 Laptop GPU
```

## Build Commands

Representative commands used for the final Humble build:

```bash
source /opt/ros/humble/setup.bash

cd /home/ubuntu/hunav_humble_ws
CMAKE_BUILD_PARALLEL_LEVEL=2 colcon build --symlink-install \
  --packages-select people_msgs hunav_msgs hunav_agent_manager \
  hunav_gazebo_fortress_wrapper \
  --cmake-args -DCMAKE_BUILD_TYPE=Release

cd /home/ubuntu/waterloo_jackal_pipeline_repo
source /home/ubuntu/hunav_humble_ws/install/setup.bash
colcon build --symlink-install --packages-select social_nav_diffusion_ros
```

The HuNav rebuild was limited to two parallel jobs because a prior unrestricted
build exhausted container memory. The final builds completed successfully.

## Exact Validation Commands

No-goal safety run:

```bash
cd /home/ubuntu/waterloo_jackal_pipeline_repo
python3 run_final_social_nav_test_humble.py \
  --no-rviz \
  --validation-seconds 20
```

Final explicit-goal run:

```bash
cd /home/ubuntu/waterloo_jackal_pipeline_repo
python3 run_final_social_nav_test_humble.py \
  --no-rviz \
  --goal 1.0 0.0 \
  --validation-seconds 10
```

Final run logs:

```text
/tmp/social_nav_humble_logs/run_20260730_195917
```

Normal persistent use:

```bash
cd /home/ubuntu/waterloo_jackal_pipeline_repo
python3 run_final_social_nav_test_humble.py
```

The default command does not publish a goal. Use RViz or `--goal X Y` only for
the isolated simulation.

## Process Results

| Process group | Required alive check | Result |
|---|---:|---|
| HuNav + Gazebo Fortress | 20 s | PASS |
| `/clock` bridge | 3 s | PASS |
| Clearpath J100 spawn stack | 20 s | PASS |
| SLAM | 10 s | PASS |
| SocialNavDiffusion wrapper | 15 s plus final observation | PASS |

All critical processes stayed alive through the final observation period.
Launcher cleanup then sent SIGINT only to launcher-owned process groups. No
simulation, HuNav, SLAM, or policy process remained afterward.

## Topic And TF Results

| Check | Result |
|---|---|
| `/clock` data | PASS |
| `/people` data, two HuNav agents | PASS |
| Gazebo model `cpr_j100_0001/robot` | PASS |
| `/cpr_j100_0001/platform/odom` | PASS |
| `/cpr_j100_0001/platform/odom/filtered` | PASS |
| `/cpr_j100_0001/sensors/lidar2d_0/scan` | PASS |
| `/cpr_j100_0001/map` | PASS |
| `odom -> base_link` | PASS, native TF |
| `map -> base_link` | PASS |
| `/social_nav_diffusion/policy_debug` | PASS |
| simulated `/cpr_j100_0001/cmd_vel` subscriber | PASS, count 1 |

Adaptive TF repair was not used.

## Policy And Safety Results

Before a goal:

```text
13 command samples checked
13 zero commands
nonzero commands: 0
```

After the explicit isolated simulation goal:

```text
checkpoint/model load records: 1
device: cuda
policy warm-up: completed in 20.666 s
first logged online command: v=0.375, w=-0.026
automatically verified cmd_vel: v=1.000, w=-0.038
acados projection: repeated status=0
acados solve time: approximately 3-9 ms
steady diffusion total time: approximately 0.43-0.59 s
```

The goal arrived during compile warm-up, was queued without movement, and was
accepted only after warm-up completed. This confirms the required behavior:
zero command before an explicit goal and policy control only afterward.

## Non-Blocking Warnings

- The Clearpath IMU filter reported that it was still waiting for
  `imu/data_raw`; required odom, filtered odom, lidar, TF, and SLAM checks still
  passed. The IMU namespace should be audited before a sensor-fusion study.
- SLAM Toolbox reported occasional lidar queue saturation.
- HuNav reported unknown geometry for some visual/mesh entities while obstacle
  processing continued. The two-agent data and plugin remained healthy.
- The wrapper reports a timing mismatch between `policy.time_step=0.25 s` and
  `diffusion_inference_period_sec=0.10 s`. It is an existing policy
  configuration warning and was not changed.
- Integrated steady inference was slower than the earlier offline compiled
  benchmark. It is functional but should be profiled before real-robot use.
- RViz was intentionally skipped in automated validation; it is optional.

## Remaining Blockers

There is no blocker for the requested Humble end-to-end simulation workflow.

This result does not authorize real-Jackal control. Real robot network,
`ROS_DOMAIN_ID`, namespace, QoS, TF, emergency stop, command limits, and operator
handover still require a separate controlled validation.

## Final Status

PASS: Humble end-to-end simulation stack is running and all required checks pass.
