import math

from social_nav_diffusion_ros.policy_cmd_vel_node import (
    combine_occupancy_points,
    voxelized_laser_points,
)


def test_voxelized_laser_points_filters_and_bounds_scan():
    points = voxelized_laser_points(
        ranges=[float('nan'), 0.05, 1.0, 1.01, float('inf'), 8.0],
        angle_min=-0.5,
        angle_increment=0.0,
        message_range_min=0.1,
        message_range_max=10.0,
        configured_range_min=0.15,
        configured_range_max=6.0,
        voxel_size=0.5,
        max_points=10,
    )

    assert len(points) == 1
    assert math.isclose(math.hypot(*points[0]), 1.0)


def test_voxelized_laser_points_keeps_nearest_bounded_points():
    points = voxelized_laser_points(
        ranges=[3.0, 1.0, 2.0],
        angle_min=-1.0,
        angle_increment=1.0,
        message_range_min=0.1,
        message_range_max=10.0,
        configured_range_min=0.15,
        configured_range_max=6.0,
        voxel_size=0.05,
        max_points=2,
    )

    distances = [math.hypot(x, y) for x, y in points]
    assert distances == [1.0, 2.0]


def test_combine_occupancy_points_supports_lidar_only_and_fusion():
    lidar_only = combine_occupancy_points(None, [(1.0, 2.0)])
    fused = combine_occupancy_points(
        [(0.0, 0.0), (0.5, 0.5)],
        [(1.0, 2.0)],
    )

    assert lidar_only.shape == (1, 2)
    assert fused.shape == (3, 2)
    assert fused[-1].tolist() == [1.0, 2.0]
