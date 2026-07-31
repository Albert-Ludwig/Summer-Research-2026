# Humble Launch

## ROS 2 Version

```text
ROS 2 Humble
Container: ros_vnc_humble_gpu_full
```

## File Locations

Windows launcher:

```text
C:\Users\Administrator\Documents\Summer Research 2026\Documentations\run_final_social_nav_test_humble.py
```

Mounted launcher inside the container:

```text
/workspace/Documentations/run_final_social_nav_test_humble.py
```

Runtime launcher:

```text
/home/ubuntu/waterloo_jackal_pipeline_repo/run_final_social_nav_test_humble.py
```

## Required Files And Environments

```text
/opt/ros/humble/setup.bash
/home/ubuntu/hunav_humble_ws/install/setup.bash
/home/ubuntu/waterloo_jackal_pipeline_repo/install/setup.bash
/home/ubuntu/social_nav_diffusion_humble_venv
/home/ubuntu/acados
/home/ubuntu/waterloo_jackal_pipeline_repo/scripts/tf_repair_humble.py
/home/ubuntu/waterloo_jackal_pipeline_repo/config/angular_half_eval.yaml
/home/ubuntu/waterloo_jackal_pipeline_repo/config/topics_sim.yaml
/home/ubuntu/waterloo_jackal_pipeline_repo/config/clearpath_humble/robot.yaml
/home/ubuntu/waterloo_jackal_pipeline_repo/experiment_setup/hunav/scenarios/office_2_agents_humble.yaml
/home/ubuntu/waterloo_jackal_pipeline_repo/experiment_setup/hunav/worlds/office_no_sensors.sdf
/workspace/SocialNavDiffusion_Inference/ckpt_step478000_SOCIAL_NORMS8.pt
```

## Start From Inside The Container

Run inside `ros_vnc_humble_gpu_full`:

```bash
cp /workspace/Documentations/run_final_social_nav_test_humble.py /home/ubuntu/waterloo_jackal_pipeline_repo/run_final_social_nav_test_humble.py

chmod +x /home/ubuntu/waterloo_jackal_pipeline_repo/run_final_social_nav_test_humble.py

cd /home/ubuntu/waterloo_jackal_pipeline_repo

python3 run_final_social_nav_test_humble.py
```
