# Humble Launch

## Current Container

```text
Container: ros_vnc_humble_gpu_full
Image: tiryoh/ros2-desktop-vnc:humble
VNC: http://localhost:6083
Workspace mount: C:\Users\Administrator\Documents\Summer Research 2026 -> /workspace
Restart policy: unless-stopped
```

The old container creation PowerShell script is no longer required for normal
startup. The Humble container has already been created and configured.

## Humble Launcher Location

Active file inside the container:

```text
/home/ubuntu/waterloo_jackal_pipeline_repo/run_final_social_nav_test_humble.py
```

Windows main copy:

```text
C:\Users\Administrator\Documents\Summer Research 2026\Documentations\run_final_social_nav_test_humble.py
```

Migration/audit backup:

```text
C:\Users\Administrator\Documents\Summer Research 2026\Humble_Migration_20260729\pipeline_source\run_final_social_nav_test_humble.py
```

## Start The Container

Run in Windows PowerShell:

```powershell
docker start ros_vnc_humble_gpu_full
docker ps --filter "name=ros_vnc_humble_gpu_full"
```

Optional GPU check:

```powershell
docker exec ros_vnc_humble_gpu_full nvidia-smi
```

Open the container terminal:

```powershell
docker exec -it ros_vnc_humble_gpu_full bash
```

The container desktop is available at:

```text
http://localhost:6083
```

## Start The Humble Simulation

Run inside the container:

```bash
cd /home/ubuntu/waterloo_jackal_pipeline_repo
python3 run_final_social_nav_test_humble.py
```

This starts Gazebo Fortress, HuNav, the simulated J100, SLAM,
SocialNavDiffusion, and RViz. It does not send a goal automatically.

Start without RViz:

```bash
cd /home/ubuntu/waterloo_jackal_pipeline_repo
python3 run_final_social_nav_test_humble.py --no-rviz
```

Run an isolated simulation goal test:

```bash
cd /home/ubuntu/waterloo_jackal_pipeline_repo
python3 run_final_social_nav_test_humble.py \
  --no-rviz \
  --goal 1.0 0.0
```

The goal option is for the isolated Gazebo simulation only. It is not approved
for a real Jackal.

## One Command From PowerShell

Start the container and launch the simulation directly:

```powershell
docker start ros_vnc_humble_gpu_full
docker exec -it -w /home/ubuntu/waterloo_jackal_pipeline_repo ros_vnc_humble_gpu_full python3 run_final_social_nav_test_humble.py
```

## Stop

Press `Ctrl+C` once to stop the simulation and clean up launcher-owned
processes.

To stop the container without deleting it:

```powershell
docker stop ros_vnc_humble_gpu_full
```

Do not delete the container, image, workspace mount, or Docker volumes.
