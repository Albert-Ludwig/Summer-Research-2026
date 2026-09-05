# Autonomous Robotics Research - Waterloo Rising Stars 2026

## Project Overview

This repository contains research materials, algorithms, environment records,
and launch tools developed during the **2026 Waterloo Engineering Rising Stars
Summer Research Fellowship**. The project studies diffusion-model social
navigation and control for Clearpath mobile robots.

## Fellowship

- **Program**: [Waterloo Engineering Rising Stars Program](https://uwaterloo.ca/engineering/waterloo-engineering-rising-stars-fellowship-program)
- **Timeline**: May 10, 2026 - August 22, 2026
- **Institution**: University of Waterloo, Faculty of Engineering
- **Research group**: Dr. Stephen L. Smith's research group
- **Researcher**: Johnson Haoran Ji, McMaster University

The work continues Christian Schaible's SocialNavDiffusion research. The
original inference repository is:

```text
https://github.com/schaiblc/SocialNavDiffusion_Inference.git
```

## Environment And Rebuild Reference

The password for JACKAL SSH is: **clearpath**

Quick index:

- [Recovery scope](#recovery-scope)
- [Required model assets](#required-model-assets)
- [Humble real Jackal environment](#humble-real-jackal-environment)
- [Physical Jackal dependency](#physical-jackal-dependency)
- [Jazzy simulation environment](#jazzy-simulation-environment)
- [Rebuild verification](#rebuild-verification)
- [Backup checklist](#backup-checklist)

This is the recovery manual for both validated environments:

| Environment  | Purpose                              | Docker engine                       |
| ------------ | ------------------------------------ | ----------------------------------- |
| ROS 2 Jazzy  | HuNav, Gazebo, and Jackal simulation | Docker Desktop                      |
| ROS 2 Humble | Real Jackal and RoboHub testing      | Native Docker in `Ubuntu-22.04` WSL |

Never copy `build/`, `install/`, `log/`, virtual environments, generated
acados libraries, or compiled ROS binaries between Jazzy and Humble.

### Recovery Scope

The tracked source and this guide can rebuild the laptop-side Humble
environment. The following items are external and must be preserved separately:

1. The teammate-managed `nahl_ws` and maps on the physical Jackal.
2. The local committed Jazzy image, unless Jazzy is rebuilt from source.

The Jazzy image contains a 16.7 GB committed layer and has no Dockerfile in
this repository. Byte-for-byte recovery requires a `docker save` archive.
Without it, the source rebuild below is functional but not byte-identical.

GitHub does not preserve containers, images, Docker volumes, WSL
distributions, VHDX files, virtual environments, or the robot filesystem.

### Git Sources

```text
Primary repository: https://github.com/Albert-Ludwig/Summer-Research-2026.git
Primary branch: main
GitLab pipeline: https://git.uwaterloo.ca/Johnson_Ji/jackal_peronal.git
Original model: https://github.com/schaiblc/SocialNavDiffusion_Inference.git
Audit baseline: 977e67db5e0b0fc256cb3d4bdaf1156983cfc4b7
```

Use an explicit stable commit or tag when rebuilding:

```powershell
git clone https://github.com/Albert-Ludwig/Summer-Research-2026.git `
  "C:\Users\Administrator\Documents\Summer Research 2026"
cd "C:\Users\Administrator\Documents\Summer Research 2026"
git checkout <stable-commit-or-tag>
git lfs install
git lfs pull
git status --short
```

`git status --short` must be empty before a revision is treated as a complete
recovery point.

Tracked recovery files:

```text
Documentations/run_final_social_nav_test.py
Documentations/final_social_nav_test_steps.md
Documentations/run_jackal_robohub.py
Documentations/humble_launch.md
Documentations/Record.md
Humble_Migration_20260729/pipeline_source/
Humble_Migration_20260729/hunav_humble_patches/
config_files/fastdds_robot_wired.xml
config_files/jackal_robohub_navigation.rviz
config_files/mount-summer-research.sh
config_files/summer-research-mount.service
config_files/wsl.conf
SocialNavDiffusion_Inference/
```

`Humble_Migration_20260729/pipeline_source/dependencies.repos` contains old
TODO placeholders. Do not use it as the recovery authority. Use the revisions
listed in this README.

### Required Model Assets

The validated model assets are stored in Git LFS. Install Git LFS and run
`git lfs pull` after cloning and before creating either Python environment:

| Asset                 | Path under `SocialNavDiffusion_Inference`             |              Size | SHA256                                                             |
| --------------------- | ----------------------------------------------------- | ----------------: | ------------------------------------------------------------------ |
| Stable checkpoint     | `SocialGuidedNavPlanner.pt`                           | 358,855,529 bytes | `e60371f69ea096a0a7ebed512f0dcbbc6d03a7c9c1b72e65261aff0417e5c1e6` |
| Test-mode checkpoint  | `ckpt_step990000_sogudiff_singleaxis_1p5M.pt`         | 359,113,646 bytes | `0b03fdacbc5762f611d3522a5ba999d7dca0c3f1232902b1114f8fc11e125687` |
| Normalization data    | `norm_stats_SOCIAL_NORMS8.npy`                        |         725 bytes | `0eac9b2e7080db7dde83c85577cbe6f105aab9fa54804ac6935f07702b2ed935` |
| YOLO detector         | `yolo11n.pt`                                          |   5,613,764 bytes | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` |

The runtime checkpoint name is a relative symbolic link:

```bash
cd /workspace/SocialNavDiffusion_Inference
ln -sfn SocialGuidedNavPlanner.pt ckpt_step478000_SOCIAL_NORMS8.pt
```

Verify the restored assets inside Linux:

```bash
cd /workspace/SocialNavDiffusion_Inference
sha256sum \
  SocialGuidedNavPlanner.pt \
  ckpt_step990000_sogudiff_singleaxis_1p5M.pt \
  norm_stats_SOCIAL_NORMS8.npy \
  yolo11n.pt
test "$(readlink ckpt_step478000_SOCIAL_NORMS8.pt)" = \
  "SocialGuidedNavPlanner.pt"
```

Do not replace an LFS object with an empty checkpoint placeholder.

Never commit API keys, passwords, SSH private keys, WSL VHDX files, or Docker
volumes. The real launcher prompts for the Jackal password. If
`JACKAL_SSH_PASSWORD` is used, set it only in the current shell.

### Shared Host

```text
Host: Windows 11 with WSL 2 and NVIDIA GPU support
Windows project: C:\Users\Administrator\Documents\Summer Research 2026
Container mount: /workspace
GPU: NVIDIA GeForce RTX 4070 Laptop GPU
PyTorch: 2.12.1+cu130
PyTorch CUDA runtime: 13.0
acados: dab96fc9b8ad486af8166331259834b33e93de37
```

Initial checks:

```powershell
nvidia-smi
docker version
wsl --version
wsl -l -v
```

Docker GPU passthrough must work:

```powershell
docker run --rm --gpus all `
  nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## Humble Real Jackal Environment

### WSL Host Configuration

Humble uses a WSL distribution named exactly `Ubuntu-22.04`. Install it only
on a new machine where it does not already exist:

```powershell
wsl --install -d Ubuntu-22.04
```

Place this in `%UserProfile%\.wslconfig`:

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

Install `config_files/wsl.conf` as `/etc/wsl.conf`, then run:

```powershell
wsl --shutdown
```

Validated native WSL versions:

```text
Ubuntu: 22.04.5 LTS
docker.io: 29.1.3-0ubuntu3~22.04.2
containerd: 2.2.1-0ubuntu1~22.04.2
nvidia-container-toolkit: 1.19.1-1
nvidia-container-toolkit-base: 1.19.1-1
```

After installing Docker and NVIDIA Container Toolkit in WSL:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl enable --now docker
docker run --rm --gpus all \
  nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Install the tracked bind-mount service. This avoids Docker problems with spaces
in the Windows source path:

```bash
sudo install -m 0755 \
  "/mnt/c/Users/Administrator/Documents/Summer Research 2026/config_files/mount-summer-research.sh" \
  /usr/local/sbin/mount-summer-research
sudo install -m 0644 \
  "/mnt/c/Users/Administrator/Documents/Summer Research 2026/config_files/summer-research-mount.service" \
  /etc/systemd/system/summer-research-mount.service
sudo systemctl daemon-reload
sudo systemctl enable --now summer-research-mount.service
mountpoint /srv/summer-research-2026
```

If the Windows username or project path changes, update `SOURCE` in
`config_files/mount-summer-research.sh` before installing it.

### Exact Humble Container

```text
WSL distribution: Ubuntu-22.04
Docker engine: native WSL Docker, not Docker Desktop
Container: jackal_robohub
Image: tiryoh/ros2-desktop-vnc:humble
Image digest: sha256:e56f4276b67f0e040726bf199fb1d36ee0339203c03296723a7426f4b452c2c8
OS: Ubuntu 22.04.5 LTS
ROS: ROS 2 Humble
Python: 3.10.12
Network: host
IPC: host
Privileged: true
Security option: label=disable
GPU: all
Shared memory: 4 GiB
Restart policy: unless-stopped
Working directory: /workspace
Mount: /srv/summer-research-2026 -> /workspace, read/write
Entrypoint: /bin/bash -c /entrypoint.sh
VNC: http://127.0.0.1:6084/vnc.html?autoconnect=1&resize=scale
```

This container does not appear in Docker Desktop. Inspect it with:

```powershell
wsl -d Ubuntu-22.04 -- docker ps -a
```

Create it inside `Ubuntu-22.04`:

```bash
sudo systemctl start docker
sudo systemctl start summer-research-mount.service

docker pull \
  tiryoh/ros2-desktop-vnc@sha256:e56f4276b67f0e040726bf199fb1d36ee0339203c03296723a7426f4b452c2c8

docker run -d \
  --name jackal_robohub \
  --hostname AlbertLudwig \
  --gpus all \
  --network host \
  --ipc host \
  --privileged \
  --security-opt label=disable \
  --shm-size 4g \
  --restart unless-stopped \
  --workdir /workspace \
  --mount type=bind,source=/srv/summer-research-2026,target=/workspace \
  tiryoh/ros2-desktop-vnc@sha256:e56f4276b67f0e040726bf199fb1d36ee0339203c03296723a7426f4b452c2c8
```

Apply the validated noVNC port once, then restart:

```bash
docker exec -u root jackal_robohub sed -i \
  's#websockify --web=/usr/lib/novnc 80 localhost:5901#websockify --web=/usr/lib/novnc 127.0.0.1:6084 localhost:5901#' \
  /entrypoint.sh
docker restart jackal_robohub
```

The launcher changes the live listener to `0.0.0.0:6084` during a test. Do not
expose this port to an untrusted network.

### Humble Source Revisions

| Component        | Repository                                                          | Revision                                                                         |
| ---------------- | ------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| HuNavSim         | `https://github.com/robotics-upo/hunav_sim.git`                     | `d97ac2c96b5de1ef9cd8835f99718504a4a005ae`                                       |
| Fortress wrapper | `https://github.com/robotics-upo/hunav_gazebo_fortress_wrapper.git` | `e6160a9d8a91f2ee4fde39ff5879507acec17cd8`                                       |
| LightSFM         | `https://github.com/robotics-upo/lightsfm.git`                      | `b30327cca189af2fb90443a5d0040cceb46d7195`                                       |
| people_msgs      | `https://github.com/wg-perception/people.git`, branch `ros2`        | `0ae47f6e0208cedd84d19d066743fdc1d05fcafa`                                       |
| acados           | `https://github.com/acados/acados.git`                              | `dab96fc9b8ad486af8166331259834b33e93de37`                                       |
| ROS wrapper      | `Humble_Migration_20260729/pipeline_source`                         | this repository's selected revision                                              |
| Model source     | `SocialNavDiffusion_Inference`                                      | upstream `c785225ee545d79ea63fa06ca8a025b3e5a536ed` plus tracked project changes |

Two Humble compatibility patches are tracked in
`Humble_Migration_20260729/hunav_humble_patches`.

Create the source workspace at the exact revisions:

```bash
mkdir -p /home/ubuntu/hunav_humble_ws/src
cd /home/ubuntu/hunav_humble_ws/src

git clone https://github.com/robotics-upo/hunav_sim.git
git -C hunav_sim checkout d97ac2c96b5de1ef9cd8835f99718504a4a005ae

git clone \
  https://github.com/robotics-upo/hunav_gazebo_fortress_wrapper.git
git -C hunav_gazebo_fortress_wrapper checkout \
  e6160a9d8a91f2ee4fde39ff5879507acec17cd8

git clone https://github.com/robotics-upo/lightsfm.git
git -C lightsfm checkout b30327cca189af2fb90443a5d0040cceb46d7195

git clone --branch ros2 https://github.com/wg-perception/people.git
git -C people checkout 0ae47f6e0208cedd84d19d066743fdc1d05fcafa

cp /workspace/Humble_Migration_20260729/hunav_humble_patches/simulation_fortress.launch.py \
  hunav_gazebo_fortress_wrapper/launch/simulation_fortress.launch.py
cp /workspace/Humble_Migration_20260729/hunav_humble_patches/hunav_agent_manager/bt_node.cpp \
  hunav_sim/hunav_agent_manager/src/bt_node.cpp
```

### Build acados In Humble

Do not copy Jazzy libraries:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake git python3-dev python3-pip python3-venv

git clone --recursive https://github.com/acados/acados.git \
  /home/ubuntu/acados
git -C /home/ubuntu/acados checkout \
  dab96fc9b8ad486af8166331259834b33e93de37
git -C /home/ubuntu/acados submodule update --init --recursive

cmake -S /home/ubuntu/acados -B /home/ubuntu/acados/build \
  -DACADOS_WITH_QPOASES=ON
cmake --build /home/ubuntu/acados/build --target install -j2
```

Expected libraries:

```text
/home/ubuntu/acados/lib/libacados.so
/home/ubuntu/acados/lib/libblasfeo.so.0
/home/ubuntu/acados/lib/libhpipm.so
/home/ubuntu/acados/lib/libqpOASES_e.so
```

### Build Humble Messages And Wrapper

Clone the pinned HuNav and `people_msgs` sources into
`/home/ubuntu/hunav_humble_ws/src`. Apply the tracked Humble patches to their
matching source files. Limit builds to two jobs:

```bash
source /opt/ros/humble/setup.bash
cd /home/ubuntu/hunav_humble_ws
rosdep install --from-paths src --ignore-src -r -y
CMAKE_BUILD_PARALLEL_LEVEL=2 colcon build --symlink-install \
  --packages-select people_msgs hunav_msgs hunav_agent_manager \
  hunav_gazebo_fortress_wrapper \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
```

Restore and build the persistent wrapper:

```bash
mkdir -p /home/ubuntu/waterloo_jackal_pipeline_repo
cp -a /workspace/Humble_Migration_20260729/pipeline_source/. \
  /home/ubuntu/waterloo_jackal_pipeline_repo/

source /opt/ros/humble/setup.bash
source /home/ubuntu/hunav_humble_ws/install/setup.bash
cd /home/ubuntu/waterloo_jackal_pipeline_repo
rosdep install --from-paths . --ignore-src -r -y
colcon build --symlink-install --packages-select social_nav_diffusion_ros
```

### Build The Humble Python Environment

Use Python 3.10 and expose Humble's `rclpy` through system site packages:

```bash
python3 -m venv --system-site-packages \
  /home/ubuntu/social_nav_diffusion_humble_venv
source /home/ubuntu/social_nav_diffusion_humble_venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

python -m pip install \
  --index-url https://download.pytorch.org/whl/cu130 \
  torch==2.12.1 torchvision==0.27.1

python -m pip install -r \
  /workspace/Humble_Migration_20260729/pipeline_source/requirements.txt

export ACADOS_SOURCE_DIR=/home/ubuntu/acados
python -m pip install -e \
  "$ACADOS_SOURCE_DIR/interfaces/acados_template"
python -m pip install -e /workspace/SocialNavDiffusion_Inference
```

Expected PyTorch is `2.12.1+cu130`. Preserve compatible wheels externally if
the CUDA 13.0 index no longer retains them.

### Humble Runtime

```text
HuNav messages: /home/ubuntu/hunav_humble_ws
Pipeline: /home/ubuntu/waterloo_jackal_pipeline_repo
Persistent wrapper: /workspace/Humble_Migration_20260729/pipeline_source
Model: /workspace/SocialNavDiffusion_Inference
Venv: /home/ubuntu/social_nav_diffusion_humble_venv
acados: /home/ubuntu/acados
DDS: /workspace/config_files/fastdds_robot_wired.xml
RViz: /workspace/config_files/jackal_robohub_navigation.rviz
```

Manual environment:

```bash
source /opt/ros/humble/setup.bash
source /home/ubuntu/hunav_humble_ws/install/setup.bash
source /home/ubuntu/waterloo_jackal_pipeline_repo/install/setup.bash
source /home/ubuntu/social_nav_diffusion_humble_venv/bin/activate

export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
unset ROS_AUTOMATIC_DISCOVERY_RANGE
unset FASTDDS_BUILTIN_TRANSPORTS
unset ROS_DISCOVERY_SERVER
unset CYCLONEDDS_URI
export FASTRTPS_DEFAULT_PROFILES_FILE=/workspace/config_files/fastdds_robot_wired.xml
export SOCIAL_NAV_DIFFUSION_VENV=/home/ubuntu/social_nav_diffusion_humble_venv
export SOCIAL_NAV_DIFFUSION_USE_VENV=true
export ACADOS_SOURCE_DIR=/home/ubuntu/acados
export LD_LIBRARY_PATH=/home/ubuntu/acados/lib:$LD_LIBRARY_PATH
export TORCHINDUCTOR_COMPILE_THREADS=1
export MAX_JOBS=1
export OMP_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
```

Validated launcher:

```powershell
python "C:\Users\Administrator\Documents\Summer Research 2026\Documentations\run_jackal_robohub.py"
```

It starts the existing container and real test stack; it does not create the
container. Press `Ctrl+C` once to stop everything it owns.

## Physical Jackal Dependency

```text
Robot IP: 192.168.131.1
Laptop Ethernet: 192.168.131.101/24
SSH: administrator@192.168.131.1
Onboard setup: /etc/clearpath/setup.bash
Onboard workspace: /home/administrator/nahl_ws
Onboard package: jackal_nav
Default map: /home/administrator/nahl_ws/maps/final.yaml
Map topic: /jackal1/map
Namespace: /jackal1
Localization: jackal_nav localisation.launch.py
```

Do not edit or replace the teammate's tuned onboard parameters, package XML,
launch files, or maps. Back them up separately with the teammate's permission;
they cannot be recovered from this repository.

Configure wired Ethernet before a real test:

```powershell
powershell -ExecutionPolicy Bypass -File `
  "C:\Users\Administrator\Documents\Summer Research 2026\config_files\configure_jackal_ethernet.ps1"
ping 192.168.131.1
```

## Jazzy Simulation Environment

### Exact Jazzy Container

```text
Docker engine: Docker Desktop
Container: ros_vnc_jazzy_gpu_full
Image: ros_vnc_jazzy_with_pipeline:gpu-base
Image ID: sha256:0fda676769837768f4eba2881793492e765b21947e955b8f2c20daedeca8b3cc
Base image: tiryoh/ros2-desktop-vnc:jazzy
Base image ID: sha256:3d2fed577544d2641c01bc3c1b97b3918fa94b3021f3114824790da61da9e8f2
OS: Ubuntu 24.04.4 LTS
ROS: ROS 2 Jazzy
Python: 3.12.3
Network: bridge
IPC: private
Privileged: false
GPU: all
Shared memory: Docker default, 64 MiB
Restart policy: no
Working directory: /workspace
Mount: Windows project -> /workspace, read/write
Port: Windows 6082 -> container 80/tcp
Entrypoint: /bin/bash -c /entrypoint.sh
VNC: http://127.0.0.1:6082/vnc.html?autoconnect=1&resize=scale
```

The custom image is local. Preserve it before deleting Docker Desktop data:

```powershell
docker save -o "D:\ContainerBackups\ros_vnc_jazzy_gpu_full.tar" `
  ros_vnc_jazzy_with_pipeline:gpu-base
Get-FileHash -Algorithm SHA256 `
  "D:\ContainerBackups\ros_vnc_jazzy_gpu_full.tar"
```

Do not commit the multi-gigabyte image archive to normal GitHub. Store it and
its SHA256 externally.

Restore the exact image and container:

```powershell
docker load -i "D:\ContainerBackups\ros_vnc_jazzy_gpu_full.tar"

docker run -d `
  --name ros_vnc_jazzy_gpu_full `
  --gpus all `
  -p 6082:80 `
  --workdir /workspace `
  --mount "type=bind,source=C:\Users\Administrator\Documents\Summer Research 2026,target=/workspace" `
  ros_vnc_jazzy_with_pipeline:gpu-base
```

### Jazzy Source Revisions

| Component         | Repository                                                          | Revision                                   |
| ----------------- | ------------------------------------------------------------------- | ------------------------------------------ |
| HuNavSim          | `https://github.com/robotics-upo/hunav_sim.git`                     | `8ecd594d57e921ce735d330782a008b7850672c5` |
| Fortress wrapper  | `https://github.com/robotics-upo/hunav_gazebo_fortress_wrapper.git` | `41f97f764df00303a85227911ec8173f0c25c758` |
| people_msgs       | `https://github.com/wg-perception/people.git`, branch `ros2`        | `0ae47f6e0208cedd84d19d066743fdc1d05fcafa` |
| Pipeline baseline | `https://git.uwaterloo.ca/Johnson_Ji/jackal_peronal.git`            | `9319eb444605be711d2f96f19c48f839c0ab40e9` |
| acados            | `https://github.com/acados/acados.git`                              | `dab96fc9b8ad486af8166331259834b33e93de37` |

The audited Jazzy pipeline has local changes beyond `9319eb4`. The exact
workspace is preserved by the custom image, not by that GitLab commit alone.
Commit reviewed Jazzy changes before retiring the image.

If the image archive is unavailable, create a fresh container from the pinned
base image, then rebuild the source workspaces. This does not reproduce the
uncommitted Jazzy pipeline changes:

```powershell
docker pull tiryoh/ros2-desktop-vnc@sha256:3d2fed577544d2641c01bc3c1b97b3918fa94b3021f3114824790da61da9e8f2

docker run -d `
  --name ros_vnc_jazzy_gpu_full `
  --gpus all `
  -p 6082:80 `
  --workdir /workspace `
  --mount "type=bind,source=C:\Users\Administrator\Documents\Summer Research 2026,target=/workspace" `
  tiryoh/ros2-desktop-vnc@sha256:3d2fed577544d2641c01bc3c1b97b3918fa94b3021f3114824790da61da9e8f2
```

Inside that container, install the simulation ROS dependencies, rebuild
acados using the same procedure and commit recorded above, and create the
Jazzy workspaces:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake git python3-dev python3-pip python3-venv \
  python3-colcon-common-extensions python3-rosdep python3-vcstool \
  ros-jazzy-clearpath-gz ros-jazzy-clearpath-viz \
  ros-jazzy-clearpath-nav2-demos ros-jazzy-slam-toolbox \
  ros-jazzy-ros-gz-bridge

mkdir -p /home/ubuntu/hunav_jazzy_ws/src
cd /home/ubuntu/hunav_jazzy_ws/src
git clone https://github.com/robotics-upo/hunav_sim.git
git -C hunav_sim checkout 8ecd594d57e921ce735d330782a008b7850672c5
git clone \
  https://github.com/robotics-upo/hunav_gazebo_fortress_wrapper.git
git -C hunav_gazebo_fortress_wrapper checkout \
  41f97f764df00303a85227911ec8173f0c25c758
git clone --branch ros2 https://github.com/wg-perception/people.git
git -C people checkout 0ae47f6e0208cedd84d19d066743fdc1d05fcafa

cd /home/ubuntu/hunav_jazzy_ws
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y
CMAKE_BUILD_PARALLEL_LEVEL=2 colcon build --symlink-install

git clone https://git.uwaterloo.ca/Johnson_Ji/jackal_peronal.git \
  /home/ubuntu/waterloo_jackal_pipeline_repo
git -C /home/ubuntu/waterloo_jackal_pipeline_repo checkout \
  9319eb444605be711d2f96f19c48f839c0ab40e9

source /home/ubuntu/hunav_jazzy_ws/install/setup.bash
cd /home/ubuntu/waterloo_jackal_pipeline_repo
rosdep install --from-paths . --ignore-src -r -y
colcon build --symlink-install --packages-select social_nav_diffusion_ros
```

Create the Jazzy Python 3.12 venv after acados is rebuilt:

```bash
cd /workspace/SocialNavDiffusion_Inference
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu130 \
  torch==2.12.1 torchvision==0.27.1
python -m pip install -r \
  /workspace/Humble_Migration_20260729/pipeline_source/requirements.txt
export ACADOS_SOURCE_DIR=/home/ubuntu/acados
python -m pip install -e \
  "$ACADOS_SOURCE_DIR/interfaces/acados_template"
python -m pip install -e /workspace/SocialNavDiffusion_Inference
```

Jazzy runtime paths:

```text
HuNav: /home/ubuntu/hunav_jazzy_ws
Pipeline: /home/ubuntu/waterloo_jackal_pipeline_repo
Model: /workspace/SocialNavDiffusion_Inference
Venv: /workspace/SocialNavDiffusion_Inference/.venv
acados: /home/ubuntu/acados
Scenario: office_2_agents.yaml
```

Dependency records:

```text
SocialNavDiffusion_Inference/requirements_inference.txt
SocialNavDiffusion_Inference/requirements_ros_wrapper_v1_fake_odom_success.txt
Documentations/Config&script.md
Documentations/final_social_nav_test_steps.md
```

Manual environment:

```bash
source /opt/ros/jazzy/setup.bash
source /home/ubuntu/hunav_jazzy_ws/install/setup.bash
source /home/ubuntu/waterloo_jackal_pipeline_repo/install/setup.bash
source /workspace/SocialNavDiffusion_Inference/.venv/bin/activate
export ACADOS_SOURCE_DIR=/home/ubuntu/acados
export LD_LIBRARY_PATH=/home/ubuntu/acados/lib:$LD_LIBRARY_PATH
```

Validated launcher:

```bash
cp /workspace/Documentations/run_final_social_nav_test.py \
  /home/ubuntu/waterloo_jackal_pipeline_repo/run_final_social_nav_test.py
chmod +x /home/ubuntu/waterloo_jackal_pipeline_repo/run_final_social_nav_test.py
cd /home/ubuntu/waterloo_jackal_pipeline_repo
python3 run_final_social_nav_test.py
```

## Rebuild Verification

### GPU And PyTorch

```bash
nvidia-smi
python3 -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

Expected:

```text
torch=2.12.1+cu130
torch CUDA runtime=13.0
torch.cuda.is_available()=True
GPU=NVIDIA GeForce RTX 4070 Laptop GPU
```

### ROS Wrapper

```bash
ros2 pkg executables social_nav_diffusion_ros
```

Humble must include:

```text
social_nav_diffusion_ros jackal_twist_adapter
social_nav_diffusion_ros nav2_goal_to_pose_bridge
social_nav_diffusion_ros policy_cmd_vel_node
social_nav_diffusion_ros rgbd_people_detector
social_nav_diffusion_ros social_nav_diffusion_node
```

Jazzy must include at least:

```text
social_nav_diffusion_ros nav2_goal_to_pose_bridge
social_nav_diffusion_ros policy_cmd_vel_node
social_nav_diffusion_ros social_nav_diffusion_node
```

### acados And Offline Inference

```bash
export ACADOS_SOURCE_DIR=/home/ubuntu/acados
export LD_LIBRARY_PATH=/home/ubuntu/acados/lib:$LD_LIBRARY_PATH
cd /home/ubuntu/waterloo_jackal_pipeline_repo
python3 scripts/run_single_step.py
```

Required result:

```text
device=cuda
checkpoint and EMA weights loaded
conditioning succeeded
diffusion inference succeeded
acados projection status=0
final ActionRot produced
```

The first compiled inference may take about 9.56 seconds. Warmed inference was
measured at about 0.161-0.175 seconds. Complete a no-output warm-up before real
control.

### Read-Only Jackal Check

Before enabling velocity output:

```bash
source /opt/ros/humble/setup.bash
cd /workspace/config_files
source ./ros_ethernet.env
bash ./check_jackal_readonly.sh
```

Confirm odometry, LiDAR, RGB-D camera, TF, emergency stop, map, and people
topics. Do not publish real velocity during environment reconstruction.

## Backup Checklist

Before deleting a container, image, Docker Desktop data, or WSL VHDX:

```powershell
git status --short
git rev-parse HEAD
docker inspect ros_vnc_jazzy_gpu_full
wsl -d Ubuntu-22.04 -- docker inspect jackal_robohub
```

Preserve externally:

```text
Jazzy docker save archive and SHA256
SocialGuidedNavPlanner.pt and SHA256
yolo11n.pt and SHA256
Authorized onboard map/workspace backups
Exact stable Git commit or tag
```

Do not preserve or transfer:

```text
build/
install/
log/
.venv/
generated acados libraries across OS or ROS versions
Docker volumes as a substitute for source control
WSL VHDX files in GitHub
credentials, API keys, passwords, or private keys
```

The launchers start existing containers; they do not recreate them. Recovery is
complete only when model hashes, GPU access, ROS executables, acados projection,
and offline inference all pass.

## Runtime Architecture And Control Reference

This section describes the current ROS 2 Humble real-Jackal implementation. It
documents the code that is active during a test, the important parameters, and
the complete path from sensors and goals to wheel commands.

### Real-Robot Reference

The retained MCAP run is:

[`bags/run_20260820_205934_550984_prox0.00_pass0.00_yield0.00_group0.00`](bags/run_20260820_205934_550984_prox0.00_pass0.00_yield0.00_group0.00)

The corresponding RViz playback video is shown below:

https://github.com/user-attachments/assets/fcc90aa7-beed-4d99-b896-a07c73ec3df7

The video is a real-run rosbag playback reference. It verifies that the bag can
be replayed and that the goal, candidate trajectories, selected trajectory,
projected trajectory, and robot motion trail can be visualized. It is not the
final human-avoidance acceptance evidence by itself. Confirm that `/people`
contains valid detections and pair the playback with synchronized external
camera footage before using a run as the final experiment record.

### Runtime Modes

The durable switch is
`Humble_Migration_20260729/pipeline_source/config/runtime_mode.yaml`.

```yaml
runtime:
  test_mode: false
```

`false` selects the preserved stable mode. `true` selects the teammate
JackalUpdate08-18 experiment mode. Command-line arguments override the file:

```powershell
# Preserved stable mode
python "C:\Users\Administrator\Documents\Summer Research 2026\Documentations\run_jackal_robohub.py" --no-test-mode

# Teammate test mode
python "C:\Users\Administrator\Documents\Summer Research 2026\Documentations\run_jackal_teammate_test.py"
```

`run_jackal_teammate_test.py` always forwards `--test-mode`. It does not alter
the durable YAML switch. Test mode enables the PS4 fixed-goal trigger, style
vector, candidate-trajectory visualization, navigation-output gate, test
checkpoint, and per-run MCAP recording.

### Runtime File Map

| File | Purpose |
|---|---|
| `Documentations/run_jackal_robohub.py` | Canonical one-command launcher. Starts onboard localization over SSH, WSL, the `jackal_robohub` container, detector, policy, command adapter, RViz, and orderly shutdown. |
| `Documentations/run_jackal_teammate_test.py` | Small wrapper that forces teammate test mode while reusing the canonical launcher. |
| `Documentations/humble_launch.md` | Short launch-command and path reference. |
| `Documentations/Record.md` | Chronological test and troubleshooting record. It is not executable configuration. |
| `config_files/ros_ethernet.env` | ROS 2 and wired Fast DDS environment for communication with the Jackal. |
| `config_files/fastdds_robot_wired.xml` | Fast DDS discovery and interface profile used by the offboard ROS processes. |
| `config_files/jackal_robohub_navigation.rviz` | Stable live RViz layout loaded by the canonical launcher. |
| `config_files/social_nav_bag_playback.rviz` | Lightweight offline playback layout. |
| `config_files/check_jackal_readonly.sh` | Read-only preflight for robot topics and connectivity. It does not publish velocity. |
| `launch/jackal_realtime_social_nav_debug.launch.py` | Starts selected combinations of the detector, policy wrapper, goal bridge, PS4 trigger, and optional RViz. |
| `launch/jackal_twist_adapter.launch.py` | Starts the final real-robot command adapter and exposes speed and LiDAR-safety arguments. |
| `social_nav_diffusion_ros/rgbd_people_detector.py` | Runs YOLO on RGB images, uses aligned depth for 3D position, tracks people, optionally associates LiDAR returns, and publishes `/people`. |
| `social_nav_diffusion_ros/ps4_nav_trigger_node.py` | Watches PS4 Options, creates a fixed goal ahead of the robot, toggles autonomous output, publishes the style vector state, and owns one MCAP recorder per accepted run. |
| `social_nav_diffusion_ros/nav2_goal_to_pose_bridge.py` | Implements the `NavigateToPose` action expected by the PS4 trigger and republishes the accepted goal for the policy. It does not replace localization. |
| `social_nav_diffusion_ros/policy_cmd_vel_node.py` | Main integration and control wrapper. Builds model state, calls SocialNavDiffusion, converts actions, executes projected controls between inference results, applies limits, and publishes `/debug_cmd_vel`. |
| `social_nav_diffusion_ros/jackal_twist_adapter.py` | Final hardware boundary. Converts `TwistStamped` to `Twist`, enforces the PS4 gate, e-stop, command watchdog, speed bounds, and swept-footprint LiDAR veto before `/jackal1/cmd_vel`. |
| `social_nav_diffusion_ros/social_nav_diffusion_node.py` | Older standalone debug inference node. It is installed but is not used by the real-Jackal launcher. |
| `social_nav_diffusion_ros/social_nav_diffusion_node_test_mode.py` | Test-mode form of the older standalone debug node. It is not the active real-Jackal controller. |
| `config/runtime_mode.yaml` | Durable stable/test mode selection. |
| `config/test_speed_control.yaml` | Active policy-wrapper control limits, warm-up behavior, goal handling, map handling, and projected-trajectory execution settings. |
| `config/topics_jackal1_live.yaml` | Real Jackal topic names for odometry, map, LiDAR, people, goal, debug command, and visualization outputs. |
| `config/topics_jackal1_live_test_mode.yaml` | PS4, action, navigation gate, and style-vector topics used only by test mode. |
| `config/rgbd_people_jackal1.yaml` | YOLO, depth, tracking, and LiDAR-association parameters. |
| `config/social_nav_trajectories_test_mode.rviz` | Full trajectory layout used for teammate-mode offline playback. |
| `scripts/run_single_step.py` | Non-ROS stable-policy inference and acados smoke test. |
| `scripts/run_single_step_test_mode.py` | Non-ROS teammate-policy test. Supports repeated calls and warm-up-excluded timing. |
| `SocialNavDiffusion_Inference/crowd_nav/configs/policy.config` | Preserved stable model configuration. |
| `SocialNavDiffusion_Inference/crowd_nav/configs/policy_test_mode.config` | Teammate model architecture, DDIM, scoring, projection, map, style, and action-limit configuration. |
| `SocialNavDiffusion_Inference/crowd_nav/policy/diffusion_CondUNetCFG.py` | Preserved stable SocialNavDiffusion policy. |
| `SocialNavDiffusion_Inference/crowd_nav/policy/diffusion_CondUNetCFG_test_mode.py` | Teammate test policy with style conditioning, trajectory sampling, scoring, acados projection, and optional detailed timing. |
| `SocialNavDiffusion_Inference/crowd_nav/policy/projection_solver.py` | acados projection-solver construction and generated-solver interface. |
| `SocialNavDiffusion_Inference/crowd_nav/policy/projection_unicycle_model.py` | Unicycle dynamics and constraints used by the projection layer. |

Paths beginning with `launch/`, `config/`, `scripts/`, or
`social_nav_diffusion_ros/` in this table are relative to
`Humble_Migration_20260729/pipeline_source/`.

### Active Test Parameters

#### Goal, Style, And Trigger

| Parameter | Current value | Meaning |
|---|---:|---|
| `test_mode` | `true` for teammate runs | Selects the teammate policy, checkpoint, trigger, visualizations, and recording path. |
| `goal_distance_m` | `6.0 m` | Goal is computed once, six metres ahead of the current base heading. |
| `trigger_button_index` | `7` | PS4 Options on the verified `/jackal1/joy_teleop/joy` mapping. |
| `style_vector` | `[0, 0, 0, 0]` | Ordered as `[prox, pass, yield, group]`; every value must stay in `[-1, 1]`. |
| `record_bag` | `true` | Starts one timestamped MCAP after the action goal is accepted. |
| `bag_output_dir` | `/workspace/bags` | Host-visible output directory for experiment bags. |
| onboard map | `/home/administrator/nahl_ws/maps/final.yaml` | Saved map used by onboard localization. |
| map topic | `/jackal1/map` | Map received by the offboard pipeline and remapped to the policy's `/map`. |

#### Diffusion And Projection

| Parameter | Current value |
|---|---:|
| test checkpoint | `ckpt_step990000_sogudiff_singleaxis_1p5M.pt` |
| normalization | `norm_stats_sogudiff_allarms_1p5M.npy` |
| kinematics | `unicycle` |
| trajectory horizon | `32` states |
| maximum dynamic agents | `10` |
| DDIM inference steps | `20` |
| sampled trajectories | `5` |
| CFG inference mode | `per_axis` |
| style CFG weight | `5.0` |
| scene CFG weight | `1.0` |
| model safety radius | `0.5 m` |
| collision cost | `10` |
| smoothness cost | `1` |
| goal reward | `5` |
| control-effort cost | `0.5` |
| projection | enabled |
| solver | acados SQP, at most `5` iterations, tolerance `1e-3` |
| static projection points | `20` |
| map conditioning | enabled, `50 x 50` over `10 m` extent |
| mixed precision | CUDA `bfloat16` AMP |

The stable and test checkpoints are both retained. The wrapper overrides the
config's checkpoint and normalization paths with the mode-specific files after
checking their existence and hashes.

#### Wrapper And Command Execution

| Parameter | Current value | Meaning |
|---|---:|---|
| `max_linear_speed` | `1.0 m/s` | Wrapper and adapter linear-speed bound. |
| `max_angular_speed` | `3.14 rad/s` | Wrapper and adapter yaw-rate bound. |
| `max_linear_accel` | `1.5 m/s^2` | Linear slew-rate limit. |
| `max_angular_accel` | `3.14 rad/s^2` | Angular slew-rate limit. |
| `goal_tolerance` | `0.25 m` | Stops and resets the policy at the goal. |
| `goal_timeout_sec` | `150 s` | Maximum active-goal duration. |
| `robot_v_pref` | `1.0 m/s` | Preferred model speed. |
| `robot_radius` | `0.25 m` | Circular robot radius presented to the model. |
| `human_radius` | `0.25 m` | Circular human radius presented to the model. |
| command publish period | `0.10 s` | Publishes or samples the latest projected command at `10 Hz`. |
| inference request period | `0.10 s` | Requests planning at up to `10 Hz`; actual rate is limited by inference time. |
| command hold timeout | `1.0 s` | Rejects an excessively old policy result. |
| projected trajectory sampling | enabled | Executes successive projected controls between diffusion results. |
| latency compensation | disabled | New trajectories are aligned to current odometry instead of blindly skipping by inference duration. |
| warm-up | enabled, no command output | Compiles the CUDA path before autonomous output can be enabled. |
| odometry synchronization | enabled | Synchronizes policy warm start and previous action from measured robot motion. |
| static map updates | change-only | Avoids rebuilding unchanged map conditioning every policy call. |

Near-goal slowdown, heading gate, heading stop, heading-alignment override, and
sign-conflict override are disabled by default. Normal navigation therefore
comes from the diffusion policy and acados projection. The wrapper retains only
goal stopping, physical speed/acceleration bounds, result freshness, and
trajectory execution. The old placeholder `compute_policy_action()` is not used
when `use_diffusion_policy: true`.

#### Perception And Tracking

| Parameter | Current value |
|---|---:|
| detector | Ultralytics YOLO with `yolo11n.pt` |
| YOLO input size | `480` |
| confidence threshold | `0.45` |
| YOLO period | `0.33 s` (about `3 Hz`) |
| `/people` publish period | `0.20 s` (`5 Hz`) |
| maximum tracked people | `10` |
| valid depth range | `0.30-12.0 m` |
| association distance | `1.0 m` |
| camera track timeout | `0.75 s` |
| LiDAR fusion | enabled at `0.10 s` |
| LiDAR association radius | `0.55 m` |
| LiDAR track hold | `1.50 s` |
| velocity smoothing | `0.50` camera, `0.25` LiDAR |

YOLO decides whether an observation is a person. Aligned depth converts the 2D
detection into a 3D position. Tracking estimates velocity. LiDAR association
can update or briefly hold an already identified person, but arbitrary LiDAR
clusters are not promoted to people. Static obstacles are provided separately
through map conditioning and live LiDAR points.

The policy-side live LiDAR stream uses a `0.5 s` timeout, ranges from `0.15` to
`6.0 m`, a `0.25 m` voxel size, at most `64` points, and `0.8 s` obstacle memory.

#### Final Hardware Safety Boundary

| Parameter | Current value |
|---|---:|
| adapter input | `/debug_cmd_vel` (`TwistStamped`) |
| adapter output | `/jackal1/cmd_vel` (`Twist`) |
| e-stop | `/jackal1/platform/emergency_stop` must be clear |
| test-mode gate | `/social_nav_diffusion/nav_enabled` must be `true` |
| command watchdog | `0.5 s` |
| LiDAR watchdog | `0.4 s` |
| LiDAR range | `0.15-6.0 m` |
| LiDAR x offset | `0.12 m` |
| physical footprint | `0.51 x 0.43 m` |
| footprint margin | `0.05 m` |
| collision reaction time | `0.15 s` |
| assumed braking | `1.5 m/s^2`, `3.14 rad/s^2` |
| collision simulation step | `0.05 s` |
| maximum collision horizon | `1.5 s` |

The adapter uses transparent-or-full-veto behavior. A safe command preserves
both model values, `v` and `w`. A predicted swept-footprint collision, stale
LiDAR, stale command, active e-stop, or disabled PS4 gate sends `v=0, w=0`.
The adapter never keeps forward speed while deleting only the planned turn.

### Control-Layer Implementation

The control wrapper intentionally keeps model policy and hardware protection
separate:

1. `policy_cmd_vel_node` transforms odometry, goal, people, map, and LiDAR data
   into the state format expected by `DiffusionConditionalUNet1DCFG`.
2. The model samples five candidate trajectories with 20 DDIM steps.
3. Candidate scoring selects one path using goal, collision, smoothness, and
   control-effort terms.
4. acados projects the selected path onto unicycle dynamics and obstacle
   constraints.
5. The returned `ActionRot(v, r)` is converted with `w = r / policy_dt`.
6. The wrapper clamps `v` and `w` to the configured speed bounds and applies
   linear and angular slew limits.
7. While the next diffusion result is being computed, the wrapper samples the
   projected control trajectory at `10 Hz`. New projected trajectories are
   aligned against current odometry.
8. The wrapper publishes the resulting `TwistStamped` on `/debug_cmd_vel`.
9. `jackal_twist_adapter` performs the final independent hardware checks and
   publishes a `Twist` on `/jackal1/cmd_vel` only when every gate is valid.

This means the diffusion model decides the normal navigation behavior and
human-avoidance intent. acados enforces dynamic feasibility. The wrapper
executes the projected result and enforces physical command limits. The final
adapter is a minimal collision and stale-data veto, not an alternative planner.

### End-To-End Control Chain

```text
Saved map + onboard AMCL localization
    -> /jackal1/map + /jackal1/platform/odom/filtered + /jackal1/tf

RGB image -> YOLO person detection
Aligned depth -> 3D person position
Camera/LiDAR association -> tracked position and velocity -> /people

PS4 Options (button 7)
    -> ps4_nav_trigger_node
    -> fixed goal 6 m ahead + /social_nav_diffusion/nav_enabled=true
    -> nav2_goal_to_pose_bridge
    -> /goal_pose

goal + odom + people + map + live LiDAR
    -> policy_cmd_vel_node
    -> SocialNavDiffusion candidate trajectories
    -> candidate scoring
    -> acados unicycle projection
    -> projected-trajectory execution and slew limits
    -> /debug_cmd_vel

/debug_cmd_vel + e-stop + nav-enabled gate + current LiDAR
    -> jackal_twist_adapter
    -> transparent pass or full zero-command veto
    -> /jackal1/cmd_vel
    -> Clearpath platform controller
    -> wheels
    -> filtered odometry feedback
```

### Visualization And Experiment Data

| Topic | Meaning |
|---|---|
| `/people` | Tracked people used by the policy. |
| `/people_detector/markers` | Person circles and velocity-horizon markers. |
| `/social_nav_diffusion/candidate_trajectories` | All sampled diffusion trajectories, test mode only. |
| `/social_nav_diffusion/predicted_trajectory` | Selected raw diffusion trajectory. |
| `/social_nav_diffusion/projected_trajectory` | Selected trajectory after acados projection. |
| `/social_nav_diffusion/active_goal_marker` | Current autonomous target. |
| `/social_nav_diffusion/policy_debug` | Model inputs, converted command, final command, timing, state, and safety flags. |
| `/social_nav_diffusion/style_vector` | Active `[prox, pass, yield, group]` test condition. |
| `/debug_cmd_vel` | Wrapper output before the final hardware adapter. |
| `/jackal1/cmd_vel` | Command delivered to the Clearpath platform interface. |

One accepted Options start creates a directory under `/workspace/bags` named
with a timestamp and the active style vector. The narrow MCAP records TF, map,
filtered odometry, people, detector status and markers, goal, policy and final
commands, policy debug, candidate/selected/projected trajectories, goal marker,
navigation gate, and style vector. Raw RGB and depth are intentionally excluded
to reduce RAM and disk bandwidth.

Detailed planning diagnostics are disabled during normal control. Set
`SND_DETAILED_TIMING=1` only for measurement because accurate per-stage CUDA
timing adds synchronization. The verified offline steady-state breakdown on the
RTX 4070 Laptop GPU was approximately:

```text
conditioning       1.2 ms
embedding          6.0 ms
DDIM             187.8 ms
scoring            3.0 ms
candidate total  198.1 ms
projection         6.9 ms
backend            0.1 ms
complete planning 205.1 ms
```

DDIM was about `91.6%` of the detailed complete-planning measurement. The first
`torch.compile` call is not representative and must be excluded as warm-up.
