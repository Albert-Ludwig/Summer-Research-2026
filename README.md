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

## Repository Layout

```text
SocialNavDiffusion_Inference/  # diffusion model and original evaluation code
jackal_pipeline/               # ROS 2 Jazzy wrapper and Jackal integration
```

The ROS wrapper, launch files, HuNav scenario, debug tools, and setup steps are in
[`jackal_pipeline/README.md`](jackal_pipeline/README.md).

Build the wrapper from the repository root with:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install \
  --base-paths jackal_pipeline \
  --packages-select social_nav_diffusion_ros
source install/setup.bash
```
