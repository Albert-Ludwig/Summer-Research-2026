# RoboHub Test Steps

1. Stop the Jazzy simulation before the real test:

```powershell
docker stop ros_vnc_jazzy_gpu_full
```

2. Turn on the Jackal, connect the controller and Ethernet cable, and confirm the emergency stop works.

3. Set the Windows wired IPv4 address to `192.168.131.101/24`, then check:

```powershell
ping 192.168.131.1
ssh administrator@192.168.131.1
```

4. Start and enter the Humble container:

```powershell
wsl -d Ubuntu-22.04
```

```bash
sudo service docker start
docker start jackal_robohub
docker exec -it jackal_robohub bash
```

5. Load Ethernet DDS and verify the robot:

```bash
source /opt/ros/humble/setup.bash
cd /workspace/config_files
source ./ros_ethernet.env
bash ./test_jackal_discovery.sh 192.168.131.1 192.168.131.101
```

6. Verify `/jackal1/platform/odom/filtered`, LiDAR, IMU, `/jackal1/tf`, and `/jackal1/tf_static`.

7. Load the policy environment:

```bash
source /home/ubuntu/hunav_humble_ws/install/setup.bash
source /home/ubuntu/waterloo_jackal_pipeline_repo/install/setup.bash
source /home/ubuntu/social_nav_diffusion_humble_venv/bin/activate
export ACADOS_SOURCE_DIR=/home/ubuntu/acados
export LD_LIBRARY_PATH=/home/ubuntu/acados/lib:$LD_LIBRARY_PATH
```

8. This original live-SLAM procedure is retired. Use the saved-map localization workflow recorded below.

9. Send one RViz goal and verify `/debug_cmd_vel`, `policy_debug`, TF, SLAM, GPU inference, and acados status `0`.

10. Do not enable `/jackal1/cmd_vel` until all debug checks pass. Current real motion remains disabled.

## 2026-08-06 Progress Record

### Verified

- Container `jackal_robohub` is running with an RTX 4070 Laptop GPU.
- Ethernet and DDS discovery work between the Humble container and Jackal `192.168.131.1`.
- Jackal topics and nodes are visible under `/jackal1`.
- `/jackal1/platform/odom/filtered`: about 49.9 Hz.
- `/jackal1/sensors/lidar3d_0/scan`: about 9.9 Hz.
- `/jackal1/sensors/imu_0/data`: about 50 Hz.
- `/jackal1/tf`: about 116 Hz.
- `/jackal1/platform/emergency_stop`: `false`.
- `odom -> base_link` works after remapping `/tf` to `/jackal1/tf` and `/tf_static` to `/jackal1/tf_static`.
- SLAM generated `/map` successfully. The map resolution was 0.05 m.
- `/jackal1/cmd_vel` uses `geometry_msgs/msg/Twist` and had zero publishers during the test.

### Safety and Interface Fixes

- `disable_policy_command_publish` now creates no command publisher and sends no zero command when disabled.
- `jackal_twist_adapter` converts policy `TwistStamped` to the Jackal `Twist` interface.
- The adapter default input was changed to `/debug_cmd_vel`.
- Adapter output remains disabled by default with `enable_output=false`.
- The wrapper was rebuilt successfully in the Humble container.
- The two adapter behavior tests passed.

### Not Run

- SocialNavDiffusion policy was not launched against the connected Jackal.
- No real `/jackal1/cmd_vel` command was published.
- The reason is that the adapter and policy safety path must be reviewed before any output-enabled test.
- Full package style tests still fail because of pre-existing flake8 and pep257 issues across the repository.

### Cleanup

- Removed the unused, unreferenced duplicate `config_files/ros_robot 1.env`.
- Kept all other `config_files` because they are used by Ethernet, localhost, Docker startup, DDS diagnostics, firewall setup, or fallback connection workflows.

### Next Step

1. Re-enter `jackal_robohub` and source Humble, the DDS Ethernet file, the pipeline workspace, the venv, and acados.
2. Start SLAM and verify `/map` and the remapped TF chain.
3. Run the policy only in a fully disabled-output test and verify no command publisher exists.
4. Test the adapter with `enable_output=false` first.
5. Do not set `enable_output=true` or publish to `/jackal1/cmd_vel` until the output-enabled procedure is explicitly reviewed.

### Follow-up Result

- Synced the corrected adapter files into the Humble container workspace.
- `colcon build --symlink-install --packages-select social_nav_diffusion_ros`: passed.
- `python3 -m pytest -q test/test_jackal_twist_adapter.py`: 2 passed.
- Installed executables include `jackal_twist_adapter`, `policy_cmd_vel_node`, `social_nav_diffusion_node`, and `nav2_goal_to_pose_bridge`.
- Adapter default input is now `/debug_cmd_vel`.
- Adapter default real output remains disabled.
- Full lint tests remain blocked by pre-existing repository-wide style warnings; no style cleanup was included in this fix.
- A safe placeholder-node test with `disable_policy_command_publish=true` passed.
- The node log confirmed: `command publication disabled; no cmd_vel publisher created`.
- The test also confirmed that `/home/ubuntu/hunav_humble_ws/install/setup.bash` must be sourced before the pipeline workspace; otherwise `people_msgs` is missing.

## 2026-08-06 Connected Jackal Test

### Connection Order

- Followed the wired order in `config_files/Jackal Connection 1.txt`.
- Windows Ethernet address: `192.168.131.101/24`.
- Windows and container ping to `192.168.131.1`: passed with 0% loss.
- Container route used `eth0` with source `192.168.131.101`.
- TCP port 22 did not respond during this run, so SSH login was not verified.
- Fast DDS discovery still passed and all `/jackal1` ROS nodes and topics were visible.

### Live Input And SLAM

- Emergency stop topic reported `false`.
- Jackal remained stationary throughout the test.
- SLAM produced a `466 x 494` map at `0.05 m` resolution.
- `map -> base_link` was available through the Jackal TF remaps.
- `/jackal1/cmd_vel` remained `geometry_msgs/msg/Twist` with zero publishers.

### Disabled-Output GPU Policy Test

- Full diffusion model loaded on the RTX 4070.
- Checkpoint and normalization files loaded successfully.
- acados solver built and projection repeatedly returned `status=0`.
- Policy warm-up took `13.094 s` on the first run and `14.661 s` on the controlled run.
- DDIM sampling was about `0.166-0.232 s` for five trajectories.
- Total prediction time was about `0.253-0.327 s`.
- acados solve time was about `3.8-8.7 ms`.
- Policy ran with `disable_policy_command_publish=true` and created no command publisher.
- Predicted and projected `nav_msgs/msg/Path` publishers were present.
- Real Jackal odom stayed near zero and no real command was published.

### Issues Found

- SSH port 22 was closed or filtered even though ping and ROS DDS worked.
- Policy goal tolerance was `0.35 m`, while bridge tolerance was `0.20 m`.
- Policy time step was `0.25 s`, while diffusion inference period was `0.10 s`.
- A timeout during an active inference caused a shutdown race: an inference thread tried to publish a trajectory after node destruction.
- The complete runtime library path must retain the HuNav `people_msgs` library and prepend acados; overwriting `LD_LIBRARY_PATH` breaks ROS or people message typesupport.

### End State

- Policy, SLAM, and adapter test processes were stopped.
- `jackal_robohub` remains running.
- `/jackal1/cmd_vel` publisher count is `0`.
- Do not enable real output until the shutdown race and parameter mismatches are reviewed.

## 2026-08-06 Wrapper Fix And Regression

### Fixes

- Set policy `goal_tolerance` to `0.20 m`, matching the goal bridge.
- Set `diffusion_inference_period_sec` to `0.25 s`, matching the model time step.
- Kept `disable_policy_command_publish=true`; the disabled policy creates no command publisher.
- Preserved the existing ROS and HuNav library paths while prepending the acados library path.
- Added orderly policy shutdown: timers stop, goals are invalidated, the inference worker is joined, and a finishing worker cannot publish after node destruction.
- Added focused shutdown tests in `test/test_policy_shutdown.py`.

### Verification

- Rebuilt `social_nav_diffusion_ros` successfully with `colcon build --symlink-install`.
- Focused tests passed: `4 passed` (`test_jackal_twist_adapter.py` and `test_policy_shutdown.py`).
- Live disabled-output warm-up completed in `13.683 s`.
- Live diffusion predictions took about `0.255-0.312 s`; acados projection returned `status=0` in about `4-6 ms`.
- Sent SIGINT during active inference. No `InvalidHandle`, destruction error, traceback, or post-destruction publish occurred.
- Policy, SLAM, and adapter processes were stopped after the regression test.
- Final `/jackal1/cmd_vel` publisher count: `0`.
- Final odom: linear and angular velocity approximately zero.
- Container `jackal_robohub` remains running.

### Remaining Item

- SSH port 22 is still unavailable, although Ethernet ping and ROS 2 DDS communication work. This does not block the current ROS topic test path.
- Real command output remains disabled. Do not enable the adapter output until a supervised motion-test procedure is approved.

## 2026-08-06 Humble Simulation Ready

### Restoration

- The Humble simulation launcher used at this stage was later retired after the real RoboHub launcher replaced it.
- Simulation uses `/home/ubuntu/clearpath_sim/robot.yaml`; the real `/home/ubuntu/clearpath/robot.yaml` was not overwritten.
- Restored the pinned HuNav Fortress sources and built `people_msgs`, `hunav_msgs`, `hunav_agent_manager`, and `hunav_gazebo_fortress_wrapper` successfully.
- Simulation uses `ROS_DOMAIN_ID=73`; the real Jackal remains on domain 0.
- Disabled `ROS_LOCALHOST_ONLY` because it prevented node discovery in the host-network container.
- The Humble scenario now has two stationary people at `(1.5, 0.5)` and `(1.5, -0.5)`.

### Command Interface

- The policy publishes `TwistStamped` on `/cpr_j100_0001/cmd_vel_stamped`.
- The simulation-only adapter converts it to `Twist` on `/cpr_j100_0001/cmd_vel`.
- Added explicit simulation parameters to bypass the real emergency-stop input and continuously publish zero while idle or timed out.
- Real adapter defaults remain unchanged: output disabled, emergency-stop clearance required, and no continuous-zero mode.
- Focused wrapper tests passed: `6 passed`.
- ROS CLI launch checks now use direct DDS discovery with `--no-daemon` to avoid stale graph-cache failures.

### Running Test

- Launcher PID: `27125` in container `jackal_robohub`.
- Runtime log directory: `/tmp/social_nav_humble_logs/run_20260806_201955`.
- GPU: NVIDIA GeForce RTX 4070 Laptop GPU; PyTorch CUDA is available.
- HuNav people, odom, filtered odom, lidar, SLAM map, and all required TF checks passed.
- Diffusion warm-up passed; acados projection returned `status=0`.
- No-goal command safety passed with seven zero samples.
- `/cpr_j100_0001/cmd_vel` is exactly `geometry_msgs/msg/Twist` and has the expected Clearpath `twist_mux` subscriber.
- RViz is running and the launcher reports `[ready] Stack is running`.
- No automatic goal was sent. Use RViz `Nav2 Goal` to begin the simulation test.
- This run is isolated simulation and does not publish to `/jackal1/cmd_vel`.

## 2026-08-06 Real-Time RoboHub Perception Test

### Active Safe Stack

- Uses the real Jackal `/jackal1` topics on `ROS_DOMAIN_ID=0`.
- `slam_toolbox` builds the current RoboHub map from `/jackal1/sensors/lidar3d_0/scan`; no saved map is loaded.
- `rgbd_people_detector` uses the real RealSense color and aligned-depth streams and publishes `people_msgs/msg/People` on `/people` in frame `map`.
- The detector runs on CUDA and publishes `/people_detector/status` and `/people_detector/markers`.
- SocialNavDiffusion loads the real checkpoint, completes GPU warm-up, and publishes only `/debug_cmd_vel` plus debug paths/markers.
- HuNav simulated people are not launched.
- The real `/jackal1/cmd_vel` has zero publishers. The real-output adapter is not running.
- RViz runs separately in the VNC desktop at `http://127.0.0.1:6083`.

### Verified Runtime

- `/map`: one publisher.
- `/people`: one publisher and one policy subscriber.
- Detector status: ready on CUDA, about 69 ms per frame during the check.
- Current sample contained zero people because nobody was visible to the camera during the check.
- `/debug_cmd_vel`: one publisher; zero command before a goal.
- `/jackal1/cmd_vel`: zero publishers and one `twist_mux` subscriber.
- Diffusion warm-up: 11.062 s; acados projection status `0`.
- GPU: RTX 4070 Laptop GPU, about 856 MiB used after warm-up.

### Runtime Stability

- WSL was automatically stopping when no persistent Windows-side WSL client remained. That stopped Docker and restarted the container even though ROS had not failed.
- The temporary `memory=11GB` and `swap=8GB` limits were removed after they made Windows short on memory. `%USERPROFILE%\.wslconfig` now uses the original mirrored-network settings and WSL default memory limits.
- During testing, keep WSL alive from PowerShell with:

```powershell
Start-Process "$env:SystemRoot\System32\wsl.exe" -ArgumentList '-d Ubuntu-22.04 --exec sleep infinity' -WindowStyle Hidden
```

### Next Safe Check

1. Stand 1-5 m in front of the RealSense camera and confirm `/people_detector/status` reports at least one detection and `/people` is nonempty.
2. In RViz, visualize `/map`, `/people_detector/markers`, `/social_nav_diffusion/goal_path`, and the predicted/projected trajectory paths.
3. Send one goal and assess `/debug_cmd_vel` and the debug trajectories while the robot remains disconnected from real command output.
4. Do not start `jackal_twist_adapter` with output enabled until perception and debug navigation are validated.

### Restarted Test State

- Windows, WSL, Docker, and `jackal_robohub` restarted successfully.
- WSL uses about `3.9/7.4 GiB` after model warm-up; swap usage is zero.
- Real Jackal camera, aligned depth, LiDAR, odom, TF, and emergency-stop topics are online.
- Emergency stop reported `false`.
- Diffusion warm-up completed in `17.939 s`; acados projection returned `status=0`.
- `/map`, `/people`, `/debug_cmd_vel`, and RViz are running.
- The people detector is ready on CUDA; the latest frame had zero detections.
- `/debug_cmd_vel` is zero before a goal.
- `/jackal1/cmd_vel` still has zero publishers; real motion remains disabled.

## 2026-08-06 One-Command Real Jackal Launcher

- Canonical launcher: `Documentations/run_jackal_robohub.py`.
- The superseded `run_final_social_nav_test_humble.py` launcher copies were removed from `Documentations`, the migration source, and the container pipeline root.
- Run it from Windows PowerShell; the same mounted file controls the container side.
- It starts WSL Docker, `jackal_robohub`, real-time SLAM, RGB-D people detection, SocialNavDiffusion, the real Jackal command adapter, and RViz.
- The command adapter starts only after the SLAM sensor, people detector, and diffusion policy report ready.
- Real command limits default to `0.05 m/s` linear and `0.1 rad/s` angular.
- `Ctrl+C` stops the policy first, allows the adapter watchdog to publish zero, and then stops the adapter and RViz.
- No Gazebo, HuNav simulation, rosbag, or monitor process is started.
- WSL is limited to `6GB` RAM and `2GB` swap in `%USERPROFILE%/.wslconfig`.
- Container supervisor manages noVNC on local port `6084`; the launcher provides a local-only fallback if the supervisor service is unavailable. Port `6083` conflicts with the WSL host-network forwarding process.
- External browser launch is disabled by default; use VS Code `Simple Browser: Show` with the noVNC URL.
- A `4GB` WSL cap was tested and rejected: the complete stack reached about `3.97GB`, caused severe pressure, and WSL exited. The cap was restored to `6GB`, leaving about `9GB` for Windows.
- `TORCHINDUCTOR_COMPILE_THREADS`, `MAX_JOBS`, `OMP_NUM_THREADS`, and `MALLOC_ARENA_MAX` are limited to `2` to reduce startup memory and CPU peaks.
- Syntax, host command expansion, container command expansion, and all required container paths passed dry-run checks.
- The container was stopped and WSL was shut down after verification; no real ROS nodes or command publisher were started.

Windows command:

```powershell
python "C:\Users\Administrator\Documents\Summer Research 2026\Documentations\run_jackal_robohub.py"
```

## 2026-08-06 Staged Launcher And Successful Restart

### Launcher Update

- Updated `Documentations/run_jackal_robohub.py` to start one stage at a time.
- Startup order is: clock check, noVNC, SLAM, valid `/map`, valid `map -> base_link` TF, GPU policy, RGB-D people detector, command adapter, then RViz.
- A failed clock, map, or TF check prevents the GPU policy and real command adapter from starting.
- The policy loads before the people detector to reduce the startup memory peak.
- Real speed limits remain `0.05 m/s` linear and `0.1 rad/s` angular.
- Windows time sync is attempted but may warn because `W32Time` is stopped.
- Optional Jackal SSH time sync uses the temporary `JACKAL_SSH_PASSWORD` environment variable. No password is stored in source files.

### ROS Launch Update

- Added independent `start_slam`, `start_people_detector`, `start_policy`, `start_goal_bridge`, and `start_rviz` arguments to `jackal_realtime_social_nav_debug.launch.py`.
- Synced the updated launch file into `/home/ubuntu/waterloo_jackal_pipeline_repo/launch/`.
- The install tree uses symlink-install, so no rebuild was required.
- Humble `ros2 launch --show-args` recognized all new arguments.

### Time Fix

- Windows matched external standard time.
- Jackal was about `28.75 s` ahead, which caused LiDAR messages to fall outside the TF cache and prevented SLAM.
- SSH at `administrator@192.168.131.1` became reachable.
- Jackal reported `System clock synchronized: no` and inactive NTP.
- Jackal time was corrected over SSH. The launcher's ROS timestamp check then passed at `-0.993 s`, within the `2.0 s` limit.

### Successful Full Start

- noVNC ready on `127.0.0.1:6084`.
- SLAM registered the real LiDAR.
- `/map` was valid at `441 x 502` cells.
- `map -> base_link` TF passed.
- SocialNavDiffusion reported `Policy ready for real goals`.
- The CUDA RGB-D detector reported ready.
- The Jackal command adapter reported `Jackal output ENABLED`.
- RViz initialized OpenGL and the launcher reached `[READY] Real Jackal stack is running`.
- No Nav2 Goal was sent during this verification.

### End State

- Sent SIGINT to the launcher and allowed its ordered shutdown to run.
- Stopped container `jackal_robohub`.
- Shut down WSL to release memory.
- No launcher, ROS node, or real command adapter is intentionally left running.

### Tomorrow

1. Connect Jackal power, controller, emergency stop, and Ethernet.
2. In Windows PowerShell, set `JACKAL_SSH_PASSWORD` for the current session without saving it in Git.
3. Run:

```powershell
python "C:\Users\Administrator\Documents\Summer Research 2026\Documentations\run_jackal_robohub.py"
```

4. Wait for `[READY] Real Jackal stack is running`.
5. Open VS Code Simple Browser at:

```text
http://127.0.0.1:6084/vnc.html?autoconnect=1&resize=scale
```

6. Confirm the physical area is clear, then send one RViz `Nav2 Goal`.
7. Press `Ctrl+C` once in the launcher PowerShell when finished.

## 2026-08-10 Saved-Map Localization Update

- Teammate confirmed the same Jackal topics and provided these workflows:
  - Mapping: `ros2 launch nav2_slam_toolbox slam.launch.py`
  - Save map: `ros2 run nav2_map_server map_saver_cli -f <mapname> --ros-args -r map:=/jackal1/map`
  - Localization: `ros2 launch jackal_nav localisation.launch.py map:=<map.yaml>`
- Do not modify the teammate's tuned parameter files.
- Updated `Documentations/run_jackal_robohub.py` to use saved-map localization instead of starting the custom real-time SLAM stage.
- Default map: `/workspace/maps/robohub_map.yaml`.
- Default map topic: `/jackal1/map`.
- Startup now waits for a valid saved map and localization TF before loading SocialNavDiffusion or enabling the real command adapter.
- RViz starts before the TF gate so `2D Pose Estimate` can be used when AMCL needs an initial pose.
- Our policy launch remaps its existing `/map` subscription to the configured map topic; no parameter YAML was changed.

### Current Blockers

- `nav2_map_server` is installed in `jackal_robohub`.
- `nav2_slam_toolbox` is not installed in `jackal_robohub`.
- `jackal_nav` is not installed in `jackal_robohub`.
- No saved RoboHub map currently exists under `/workspace/maps`.
- Obtain/install the teammate's configured `nav2_slam_toolbox` and `jackal_nav` packages without editing their parameters, then create or copy the saved map.

### Directory Cleanup

- Inspected `pipeline_source/launch`; all five launch files remain in use or are retained as explicit simulation, debug, adapter, or real-time SLAM fallbacks.
- Removed three unused migration duplicates:
  - `pipeline_source/run_final_social_nav_test.py`
  - `pipeline_source/Documentations/run_final_social_nav_test.py`
  - `pipeline_source/Documentations/final_social_nav_test_steps.md`
- Removed the now-empty `pipeline_source/Documentations` directory.
- Kept the canonical Jazzy files under the top-level `Documentations` directory.
- Verified that no workspace file references the removed paths.
- Removed 15 project-level Python cache directories and 12 `.DS_Store` files.
- Removed the generated top-level ROS `log/` directory and browser `debug.log`.
- Verified that no project-level cache, temporary log, PID, or editor backup remains outside `.venv`.
- Kept `.venv`, checkpoints, `c_generated_proj`, and `acados_proj_ocp.json` because they are environments or runtime assets rather than temporary clutter.

## 2026-08-10 Live SLAM Retirement

- Deleted the custom `pipeline_source/launch/jackal_slam.launch.py`.
- Removed the `start_slam` argument and custom SLAM include from `jackal_realtime_social_nav_debug.launch.py`.
- Removed obsolete `start_slam:=false` arguments from `run_jackal_robohub.py`.
- Removed this wrapper's unused `slam_toolbox` runtime dependency declaration.
- Did not uninstall the system `slam_toolbox`; the teammate's `nav2_slam_toolbox` workflow may still depend on it.
- The supported real workflow is now saved-map localization through `jackal_nav localisation.launch.py`.
- Until `jackal_nav`, `nav2_slam_toolbox`, and a saved map are available, the real navigation stack cannot build a map or localize.

## 2026-08-10 Onboard Localization Correction

- Teammate clarified that `nav2_slam_toolbox`, `jackal_nav`, and all tuned parameters already exist onboard the Jackal in `/home/administrator/nahl_ws`.
- The earlier container package checks were based on the wrong architecture. These packages do not need to be downloaded, copied, installed, configured, or edited in `jackal_robohub`.
- Updated `run_jackal_robohub.py` to SSH to `administrator@192.168.131.1`, source `nahl_ws/install/setup.bash`, verify the onboard map, and start `jackal_nav localisation.launch.py` on the Jackal.
- The container now starts only the map/TF gates, RViz, SocialNavDiffusion, RealSense people detector, command adapter, and supporting bridge.
- `Ctrl+C` stops the container policy/adapter and the launcher-managed onboard localization process.
- Default onboard map path: `/home/administrator/maps/robohub_map.yaml`.
- First-time mapping remains a supervised onboard operation because the Jackal must be driven through the environment before the map saver is run.
- No teammate package or parameter file was modified.

## 2026-08-10 Live Jackal Connection And Localization Test

- Ethernet is connected with laptop IP `192.168.131.101` and Jackal IP `192.168.131.1`.
- SSH login works as `administrator@192.168.131.1`; robot hostname is `jackal1`.
- Confirmed robot configuration: J100 serial `j100-0304`, namespace `jackal1`, ROS domain `0`.
- Confirmed onboard workspace `/home/administrator/nahl_ws` and package `jackal_nav`.
- The installed SLAM package is named `slam_toolbox`; `nav2_slam_toolbox` is not an indexed ROS package. A standalone `/home/administrator/nav2_slam_toolbox/slam.launch.py` file also exists. Do not alter teammate files.
- Found multiple existing map pairs under `/home/administrator/nahl_ws/maps`.
- Latest pair is `final.yaml + final.pgm`, updated 2026-08-07. It loads successfully as a `789 x 673` map at `0.1 m/cell`.
- Updated launcher default map to `/home/administrator/nahl_ws/maps/final.yaml`.
- Confirmed live odom, TF, LiDAR, color image, and command topics under `/jackal1`.
- RealSense depth, alignment, and synchronization were disabled at runtime. The launcher now enables them through ROS parameters without editing any parameter file.
- Verified live RGB and aligned-depth messages; aligned depth is `640 x 480` in `camera_0_color_optical_frame`.
- Updated onboard SSH commands to source `/etc/clearpath/setup.bash` before `nahl_ws/install/setup.bash`.
- Changed clock synchronization to use WSL time instead of Windows time. Verified ROS skew after sync: about `-0.095 s`.
- Fixed the map gate to subscribe with reliable, transient-local QoS. The container then received `/jackal1/map` successfully.
- Fixed noVNC to listen on `0.0.0.0:6084`; Windows `127.0.0.1:6084` is reachable.
- The launcher kills the container's unnecessary update notifier and screensaver processes to reduce load.
- Increased the initial localization wait to 600 seconds.
- The live test reached RViz with the saved map loaded, but no `2D Pose Estimate` was sent during the 600-second window. The launcher therefore stopped before Diffusion, people detection, or the command adapter started.
- Fixed an SSH startup race by waiting for the real onboard launch log marker before continuing. Remote commands are Base64-encoded so Windows/WSL cannot expand Bash PID expressions.
- Fixed shutdown to terminate the complete onboard localization process group. A start/stop regression test confirmed that no `localisation.launch.py`, map server, AMCL, or localization lifecycle manager process remains.
- No navigation goal was sent and no real motion command was enabled during these checks. The `jackal_robohub` container remains available, but RViz and all test ROS processes are stopped.

## 2026-08-10 RViz Topic Correction

- RViz `Global Status` and `RobotModel` remained in error because AMCL had not received a `2D Pose Estimate`, so `map -> odom -> base_link` did not exist yet.
- The map itself was received and rendered; `/jackal1/map` was not the root problem.
- Clearpath's default RViz config subscribed to `sensors/lidar2d_0/scan`, while this Jackal publishes `/jackal1/sensors/lidar3d_0/scan`.
- Added `config_files/jackal_robohub_navigation.rviz` without modifying the installed Clearpath config.
- Updated `run_jackal_robohub.py` to load the project RViz config with the `/jackal1` namespace and `/jackal1/tf` remaps.
- On the next test, set the robot's real pose and heading with RViz `2D Pose Estimate`; the launcher will continue only after `map -> base_link` becomes valid.

## 2026-08-10 Full Real Jackal Test Ready

- Started the complete real workflow from an independent PowerShell window so WSL remains alive during the supervised test.
- Corrected the project RViz LiDAR topic to `/jackal1/sensors/lidar3d_0/scan`; map, RobotModel, and TF became valid after `2D Pose Estimate`.
- Found that the RGB-D detector's single-thread executor blocked sensor callbacks during inference, causing stale frames and TF extrapolation failures.
- Updated only the project `rgbd_people_detector` wrapper to cache sensor messages in one callback group and process the latest frame in a separate inference callback group with a two-thread executor.
- Rebuilt only `social_nav_diffusion_ros`; teammate Nav2, AMCL, map, parameter files, checkpoint, and model source were not modified.
- Policy warm-up passed on CUDA in `11.643 s`; acados projection returned `status=0`.
- RGB-D detector now publishes `/people_detector/status` with `ready=true` and `/people` in frame `map`; measured inference was about `139.71 ms` with no person visible.
- The Jackal adapter is enabled with limits `0.05 m/s` linear and `0.1 rad/s` angular. Before a goal it publishes zero because the policy input command has timed out, which is expected.
- Current full stack is running. Use RViz `Nav2 Goal`; stop the complete workflow with `Ctrl+C` in the independent PowerShell launcher window.

## 2026-08-10 Resource Reduction And Motion Fix

- Confirmed that saved-map localization, map server, AMCL, LiDAR, camera drivers, and robot base drivers already run onboard Jackal.
- Jackal onboard hardware is an Intel i5-4570TE with 4 CPUs, 7.7 GiB RAM, and no NVIDIA GPU. Diffusion and CUDA RGB-D detection cannot be moved there without losing real-time performance and risking localization stability.
- Before optimization, Windows had about 15.2 GiB physical RAM, WSL used about 5.5 GiB, and the test container used about 5.16 GiB.
- Reduced RViz from 30 FPS to 10 FPS, RGB-D inference from 5 Hz to about 3 Hz, and Torch/OMP build threads from 2 to 1.
- The persistent approximately 630 MiB Torch compile worker disappeared. With the full optimized stack ready, WSL reported about 3.7 GiB used and 1.7 GiB available.
- RViz CPU usage dropped from about 200% to about 75%; RGB-D detector CPU dropped from about 109% to about 86%.
- Fixed Humble `NavigateToPose.Result` compatibility: Humble has only `std_msgs/Empty result`, not `NONE`, `error_code`, or `error_msg`.
- The earlier goal did reach Diffusion and generated nonzero commands, but the real adapter limit of 0.05 m/s was likely below the Jackal's practical starting range and intermittent stale `/people` frames also forced zeros.
- Increased supervised real limits to 0.10 m/s linear and 0.20 rad/s angular. Emergency-stop gating and the command watchdog remain enabled.
- Added adapter logs for emergency-stop clear and forwarded velocity. Adapter unit tests pass: 4 passed.
- The optimized full stack is currently running, but no new `Nav2 Goal` has been received after the restart. The current forwarded command is correctly zero until a new goal is submitted.

## 2026-08-10 Low-Memory Live LiDAR Avoidance

- Kept SocialNavDiffusion as the planner; no Nav2 controller or extra process was added.
- The existing policy process now subscribes to `/jackal1/sensors/lidar3d_0/scan`, voxelizes at most 1,000 current-frame points, and fuses them with the saved-map occupancy used by Diffusion and acados.
- The existing Jackal adapter now fails closed when LiDAR is missing or older than `0.4 s`, stops inside `0.55 m`, and slows between `0.55 m` and `1.0 m`.
- The temporary first-run limits of `0.2 m/s` linear and `0.4 rad/s` angular were removed. The one-command launcher again uses the validated `angular_half_eval.yaml` limits: `1.0 m/s` linear and `pi/2 rad/s` angular.
- No model, checkpoint, teammate Nav2 parameter, or onboard package was modified.
- Wrapper build passed. Focused regression: `16 passed`.
- Adapter RSS changed from `55,816 KB` to `56,584 KB`: about `0.75 MB` added RAM. The policy retains only the latest bounded scan, not scan history.
- Before the next motion test, use RViz with output disabled to confirm the table leg appears in the live LaserScan. Then test with a large soft obstacle before testing a thin table leg or a person.
- No policy, detector, adapter, RViz, or real command process was left running after verification.

## 2026-08-10 Stable Real Navigation Baseline 1

- Launched the complete workflow with `run_jackal_robohub.py` after setting the initial pose in RViz.
- CUDA Diffusion warm-up passed, acados returned `status=0`, YOLO published `/people`, and live LiDAR ran at about `10.8 Hz`.
- Confirmed policy and adapter limits are `1.0 m/s` linear and `pi/2 rad/s` angular.
- Sent a real RViz Nav2 Goal. The Jackal moved correctly and reached the expected position.
- Stopped by publishing zero velocity and shutting down policy, adapter, detector, RViz, and onboard localization. No test process remained.
- Known behavior for the next tuning pass: obstacle avoidance can produce circling and goal overshoot. This baseline is preserved before any tuning changes.
