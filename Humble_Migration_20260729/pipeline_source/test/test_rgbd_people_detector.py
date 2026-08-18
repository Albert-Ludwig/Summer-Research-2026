from types import SimpleNamespace

import numpy as np

from social_nav_diffusion_ros.rgbd_people_detector import (
    RgbdPeopleDetector,
    Track,
    associate_lidar_points,
    depth_from_box,
    fresh_tracks,
    laser_points_in_target,
    quaternion_rotate,
)


def test_depth_from_box_returns_median_depth_and_center():
    depth = np.full((100, 200), 2500, dtype=np.uint16)

    result = depth_from_box(
        depth,
        (20.0, 10.0, 120.0, 90.0),
        depth_scale=0.001,
        min_depth=0.3,
        max_depth=12.0,
        min_valid_pixels=20,
    )

    assert result is not None
    distance, pixel_x, pixel_y = result
    assert distance == 2.5
    assert 65.0 <= pixel_x <= 75.0
    assert 45.0 <= pixel_y <= 55.0


def test_depth_from_box_rejects_invalid_depth():
    depth = np.zeros((40, 40), dtype=np.uint16)

    assert depth_from_box(
        depth,
        (0.0, 0.0, 40.0, 40.0),
        depth_scale=0.001,
        min_depth=0.3,
        max_depth=12.0,
        min_valid_pixels=5,
    ) is None


def test_quaternion_rotate_applies_yaw_rotation():
    half_angle = np.pi / 4.0
    quaternion = SimpleNamespace(
        x=0.0,
        y=0.0,
        z=float(np.sin(half_angle)),
        w=float(np.cos(half_angle)),
    )

    x, y, z = quaternion_rotate((1.0, 0.0, 0.0), quaternion)

    assert abs(x) < 1e-9
    assert abs(y - 1.0) < 1e-9
    assert abs(z) < 1e-9


def test_fresh_tracks_removes_only_expired_receive_times():
    tracks = {
        1: Track(1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 10.0, 20.0),
        2: Track(2, 1.0, 0.0, 0.0, 0.0, 0.0, 0.8, 10.5, 20.6),
    }

    result = fresh_tracks(tracks, now_sec=21.0, timeout_sec=0.75)

    assert list(result) == [2]


def test_fresh_tracks_limits_lidar_hold_by_camera_confirmation():
    track = Track(
        1,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.9,
        10.0,
        20.9,
        camera_confirmed_receive=19.0,
        last_lidar_receive=20.9,
    )

    result = fresh_tracks(
        {1: track},
        now_sec=21.0,
        timeout_sec=0.75,
        camera_timeout_sec=1.5,
    )

    assert result == {}


def test_laser_points_in_target_filters_and_transforms_ranges():
    transform = SimpleNamespace(
        transform=SimpleNamespace(
            translation=SimpleNamespace(x=1.0, y=2.0, z=0.0),
            rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        )
    )

    points = laser_points_in_target(
        [1.0, float('inf'), 2.0],
        angle_min=0.0,
        angle_increment=np.pi / 2.0,
        range_min=0.1,
        range_max=10.0,
        transform=transform,
    )

    np.testing.assert_allclose(points, [[2.0, 2.0], [-1.0, 2.0]], atol=1e-9)


def test_associate_lidar_points_assigns_each_point_to_nearest_track():
    tracks = [
        Track(1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.9, 10.0, 20.0),
        Track(2, 3.0, 0.0, 0.0, 0.0, 0.0, 0.8, 10.0, 20.0),
    ]
    points = np.asarray([
        [0.9, -0.1],
        [1.1, 0.1],
        [2.9, -0.1],
        [3.1, 0.1],
        [8.0, 8.0],
    ])

    associations = associate_lidar_points(
        points,
        tracks,
        stamp_sec=10.1,
        radius_m=0.5,
        min_points=2,
        max_prediction_sec=0.5,
    )

    assert set(associations) == {1, 2}
    np.testing.assert_allclose(associations[1][:2], [1.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(associations[2][:2], [3.0, 0.0], atol=1e-9)
    assert associations[1][2] == 2
    assert associations[2][2] == 2


def test_yolo_backend_requests_only_people_and_returns_boxes():
    class FakeTensor:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.float32)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    class FakeYolo:
        def __init__(self):
            self.kwargs = None

        def predict(self, **kwargs):
            self.kwargs = kwargs
            boxes = SimpleNamespace(
                xyxy=FakeTensor([[10.0, 20.0, 30.0, 80.0]]),
                conf=FakeTensor([0.91]),
            )
            return [SimpleNamespace(boxes=boxes)]

    model = FakeYolo()
    parameters = {
        'score_threshold': SimpleNamespace(value=0.45),
        'max_people': SimpleNamespace(value=10),
    }
    detector = SimpleNamespace(
        detector_backend='ultralytics_yolo',
        model=model,
        device=SimpleNamespace(type='cpu'),
        yolo_image_size=640,
        get_parameter=lambda name: parameters[name],
    )

    candidates = RgbdPeopleDetector.person_box_candidates(
        detector,
        np.zeros((100, 100, 3), dtype=np.uint8),
    )

    assert len(candidates) == 1
    np.testing.assert_allclose(
        candidates[0][0],
        [10.0, 20.0, 30.0, 80.0],
    )
    assert abs(candidates[0][1] - 0.91) < 1e-6
    assert model.kwargs['classes'] == [0]
    assert model.kwargs['conf'] == 0.45
    assert model.kwargs['imgsz'] == 640
    assert model.kwargs['device'] == 'cpu'
    assert 'half' not in model.kwargs
