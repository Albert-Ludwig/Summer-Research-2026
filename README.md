# Autonomous Robotics Research - Waterloo Rising Stars 2026

## Project Overview

This repository hosts the research materials, algorithms, and documentation developed during the **2026 Waterloo Engineering Rising Stars Summer Research Fellowship**. The research focuses on the field of autonomous robotics, specifically targeting control systems and navigation strategies for mobile robotic platforms.

## Context and Fellowship

- **Program**: [Waterloo Engineering Rising Stars Program](https://uwaterloo.ca/engineering/waterloo-engineering-rising-stars-fellowship-program)
- **Timeline**: May 10, 2026 – August 22, 2026
- **Institution**: University of Waterloo, Faculty of Engineering
- **Research Group**: Conducted within the research group of Dr. Stephen L. Smith.

## Supervision

The research was conducted under the supervision of **Dr. Stephen L. Smith**, Professor in the Department of Electrical and Computer Engineering at the University of Waterloo and Canada Research Chair in Control Systems for Mobile Robots.

## Researcher

**Johnson Haoran Ji**

- Research Fellow, University of Waterloo (Rising Stars 2026)
- Mechatronics Engineering, McMaster University

## Research Project

This is the repositry made by Christian Schaible, who is a PhD student at the University of Waterloo. The project is focused on developing a novel approach for social navigation in autonomous robots using diffusion models. The goal is to enable robots to navigate complex environments while considering social norms and human interactions.

This is the project that I worked on during my fellowship, and it is a continuation of the work done by Christian Schaible. The original repository can be found at
https://github.com/schaiblc/SocialNavDiffusion_Inference.git

## Environment And Rebuild Reference

This repository uses two separate GPU environments. Jazzy is the validated
simulation environment. Humble is the validated real-Jackal environment. Do
not copy `build/`, `install/`, `log/`, virtual environments, or compiled ROS
binaries between them.

### Shared Host And Assets

```text
Host: Windows with WSL 2 and NVIDIA GPU support
GPU: NVIDIA GeForce RTX 4070 Laptop GPU
Windows project path: C:\Users\Administrator\Documents\Summer Research 2026
Container project mount: /workspace
SocialNavDiffusion source: /workspace/SocialNavDiffusion_Inference
acados commit: dab96fc9b8ad486af8166331259834b33e93de37
```

Required model assets:

```text
/workspace/SocialNavDiffusion_Inference/ckpt_step478000_SOCIAL_NORMS8.pt
/workspace/SocialNavDiffusion_Inference/norm_stats_SOCIAL_NORMS8.npy
/workspace/SocialNavDiffusion_Inference/yolo11n.pt
```

Store large model files with Git LFS, a GitHub Release, or a separate controlled
download. Keep their SHA256 checksums in Git. Never commit API keys, passwords,
SSH private keys, Docker volumes, or WSL VHDX files.

### ROS 2 Jazzy Simulation

```text
Container: ros_vnc_jazzy_gpu_full
Image: ros_vnc_jazzy_with_pipeline:gpu-base
Base image family: tiryoh/ros2-desktop-vnc:jazzy
Container OS: Ubuntu 24.04 family
ROS: ROS 2 Jazzy
Python: 3.12
GPU access: required
HuNav workspace: /home/ubuntu/hunav_jazzy_ws
Pipeline workspace: /home/ubuntu/waterloo_jackal_pipeline_repo
Model venv: /workspace/SocialNavDiffusion_Inference/.venv
acados: /home/ubuntu/acados
Scenario YAML: office_2_agents.yaml
```

Dependency records:

```text
SocialNavDiffusion_Inference/requirements_inference.txt
SocialNavDiffusion_Inference/requirements_ros_wrapper_v1_fake_odom_success.txt
Documentations/Config&script.md
Documentations/final_social_nav_test_steps.md
```

Environment setup:

```bash
source /opt/ros/jazzy/setup.bash
source /home/ubuntu/hunav_jazzy_ws/install/setup.bash
cd /home/ubuntu/waterloo_jackal_pipeline_repo
source install/setup.bash

export ACADOS_SOURCE_DIR=/home/ubuntu/acados
export LD_LIBRARY_PATH=/home/ubuntu/acados/lib:$LD_LIBRARY_PATH
```

Validated simulation launcher:

```bash
cp /workspace/Documentations/run_final_social_nav_test.py \
  /home/ubuntu/waterloo_jackal_pipeline_repo/run_final_social_nav_test.py
cd /home/ubuntu/waterloo_jackal_pipeline_repo
python3 run_final_social_nav_test.py
```

### ROS 2 Humble Real Jackal

```text
WSL distribution: Ubuntu-22.04
Container: jackal_robohub
Base image family: tiryoh/ros2-desktop-vnc:humble
Container OS: Ubuntu 22.04
ROS: ROS 2 Humble
Python: 3.10
PyTorch: 2.12.1+cu130
PyTorch CUDA runtime: 13.0
Container network: host
GPU access: all NVIDIA GPUs
Security option: seccomp=unconfined
Shared memory target: 4 GB
VNC: http://127.0.0.1:6084/vnc.html?autoconnect=1&resize=scale
HuNav message workspace: /home/ubuntu/hunav_humble_ws
Pipeline workspace: /home/ubuntu/waterloo_jackal_pipeline_repo
Model venv: /home/ubuntu/social_nav_diffusion_humble_venv
acados: /home/ubuntu/acados
Persistent Humble source: /workspace/Humble_Migration_20260729/pipeline_source
```

Current WSL resource configuration:

```ini
[wsl2]
memory=6GB
swap=2GB
vmIdleTimeout=28800000
networkingMode=mirrored
dnsTunneling=true
firewall=true
autoProxy=true
```

Dependency records:

```text
Humble_Migration_20260729/pipeline_source/requirements.txt
SocialNavDiffusion_Inference/requirements_inference.txt
Documentations/humble_launch.md
config_files/fastdds_robot_wired.xml
config_files/jackal_robohub_navigation.rviz
```

Environment setup:

```bash
source /opt/ros/humble/setup.bash
source /home/ubuntu/hunav_humble_ws/install/setup.bash
source /home/ubuntu/waterloo_jackal_pipeline_repo/install/setup.bash
source /home/ubuntu/social_nav_diffusion_humble_venv/bin/activate

export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export FASTRTPS_DEFAULT_PROFILES_FILE=/workspace/config_files/fastdds_robot_wired.xml
export ACADOS_SOURCE_DIR=/home/ubuntu/acados
export LD_LIBRARY_PATH=/home/ubuntu/acados/lib:$LD_LIBRARY_PATH
```

The Jackal provides localization from its onboard workspace:

```text
SSH target: administrator@192.168.131.1
Onboard workspace: /home/administrator/nahl_ws
Onboard package: jackal_nav
Default map: /home/administrator/nahl_ws/maps/final.yaml
Map topic: /jackal1/map
Robot namespace: /jackal1
```

Do not modify the teammate's tuned onboard parameter files during an environment
rebuild.

Validated Windows launcher:

```powershell
python "C:\Users\Administrator\Documents\Summer Research 2026\Documentations\run_jackal_robohub.py"
```

### Rebuild Rules

1. Recreate Jazzy and Humble independently with their matching Ubuntu, ROS, and
   Python versions.
2. Mount this GitHub repository at `/workspace`.
3. Create a fresh virtual environment for each ROS/Python version.
4. Install the corresponding pinned requirements and ROS dependencies.
5. Clone acados at the recorded commit and rebuild it inside each environment.
6. Build HuNav messages and the pipeline with the matching ROS distribution.
7. Restore model assets and verify their checksums.
8. Verify `nvidia-smi`, `torch.cuda.is_available()`, acados shared libraries,
   and `ros2 pkg executables social_nav_diffusion_ros` before testing.
9. Test offline inference and simulation before enabling real velocity output.

The launch scripts start existing containers; they do not recreate a missing
container. Before deleting a working container, record its exact image and
runtime settings:

```bash
docker inspect <container> --format 'Image={{.Config.Image}} ImageID={{.Image}}'
docker inspect <container> --format '{{json .HostConfig}}'
docker inspect <container> --format '{{json .Mounts}}'
```
