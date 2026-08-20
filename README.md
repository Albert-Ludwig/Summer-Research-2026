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

1. Model weights ignored by Git.
2. The teammate-managed `nahl_ws` and maps on the physical Jackal.
3. The local committed Jazzy image, unless Jazzy is rebuilt from source.

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

`*.pt` is intentionally ignored by Git. Restore these assets before creating
either Python environment:

| Asset                | Path under `SocialNavDiffusion_Inference` |              Size | SHA256                                                             |
| -------------------- | ----------------------------------------- | ----------------: | ------------------------------------------------------------------ |
| Diffusion checkpoint | `SocialGuidedNavPlanner.pt`               | 358,855,529 bytes | `e60371f69ea096a0a7ebed512f0dcbbc6d03a7c9c1b72e65261aff0417e5c1e6` |
| Normalization data   | `norm_stats_SOCIAL_NORMS8.npy`            |         725 bytes | `0eac9b2e7080db7dde83c85577cbe6f105aab9fa54804ac6935f07702b2ed935` |
| YOLO detector        | `yolo11n.pt`                              |   5,613,764 bytes | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` |

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
  norm_stats_SOCIAL_NORMS8.npy \
  yolo11n.pt
test "$(readlink ckpt_step478000_SOCIAL_NORMS8.pt)" = \
  "SocialGuidedNavPlanner.pt"
```

Use Git LFS, a private GitHub Release, or controlled external storage for the
two `.pt` files. Never use an empty checkpoint placeholder.

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
