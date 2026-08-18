import json
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


VENV_ROOT = os.environ.get(
    'SOCIAL_NAV_DIFFUSION_VENV',
    '/home/ubuntu/social_nav_diffusion_humble_venv',
)
VENV_PYTHON = f'{VENV_ROOT}/bin/python'


def maybe_reexec_into_venv():
    if os.path.abspath(sys.executable) == os.path.abspath(VENV_PYTHON):
        return
    if os.path.abspath(sys.prefix) == os.path.abspath(VENV_ROOT):
        return
    if not os.path.exists(VENV_PYTHON):
        raise RuntimeError(f'Perception venv is missing: {VENV_PYTHON}')

    env = os.environ.copy()
    people_python = (
        '/home/ubuntu/hunav_humble_ws/install/people_msgs/'
        'local/lib/python3.10/dist-packages'
    )
    env['PYTHONPATH'] = ':'.join(
        path for path in (people_python, env.get('PYTHONPATH', '')) if path
    )
    print(f'[rgbd_people_detector] re-exec into {VENV_PYTHON}', flush=True)
    os.execve(VENV_PYTHON, [VENV_PYTHON] + sys.argv, env)


maybe_reexec_into_venv()

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from people_msgs.msg import People, Person
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class Detection:
    x: float
    y: float
    z: float
    score: float


@dataclass
class Track:
    track_id: int
    x: float
    y: float
    z: float
    vx: float
    vy: float
    score: float
    last_seen: float
    last_seen_receive: float
    camera_confirmed_receive: Optional[float] = None
    last_lidar_receive: Optional[float] = None


def stamp_to_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def quaternion_rotate(
    point: Sequence[float],
    quaternion,
) -> Tuple[float, float, float]:
    vector = np.asarray(point, dtype=np.float64)
    axis = np.asarray(
        [quaternion.x, quaternion.y, quaternion.z],
        dtype=np.float64,
    )
    scalar = float(quaternion.w)
    rotated = (
        vector
        + 2.0 * scalar * np.cross(axis, vector)
        + 2.0 * np.cross(axis, np.cross(axis, vector))
    )
    return float(rotated[0]), float(rotated[1]), float(rotated[2])


def transform_point(point: Sequence[float], transform) -> Tuple[float, float, float]:
    rx, ry, rz = quaternion_rotate(point, transform.transform.rotation)
    translation = transform.transform.translation
    return (
        rx + float(translation.x),
        ry + float(translation.y),
        rz + float(translation.z),
    )


def depth_from_box(
    depth_image: np.ndarray,
    box: Sequence[float],
    depth_scale: float,
    min_depth: float,
    max_depth: float,
    min_valid_pixels: int,
) -> Optional[Tuple[float, float, float]]:
    height, width = depth_image.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in box]
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)

    left = max(0, min(width - 1, int(x1 + 0.30 * box_width)))
    right = max(left + 1, min(width, int(x1 + 0.70 * box_width)))
    top = max(0, min(height - 1, int(y1 + 0.25 * box_height)))
    bottom = max(top + 1, min(height, int(y1 + 0.75 * box_height)))
    region = depth_image[top:bottom, left:right].astype(np.float32)
    depths = region * float(depth_scale)
    valid = depths[
        np.isfinite(depths)
        & (depths >= float(min_depth))
        & (depths <= float(max_depth))
    ]
    if valid.size < int(min_valid_pixels):
        return None

    depth = float(np.median(valid))
    pixel_x = 0.5 * (left + right - 1)
    pixel_y = 0.5 * (top + bottom - 1)
    return depth, pixel_x, pixel_y


def fresh_tracks(
    tracks: Dict[int, Track],
    now_sec: float,
    timeout_sec: float,
    camera_timeout_sec: Optional[float] = None,
) -> Dict[int, Track]:
    fresh = {}
    for track_id, track in tracks.items():
        if now_sec - track.last_seen_receive > timeout_sec:
            continue
        camera_receive = (
            track.camera_confirmed_receive
            if track.camera_confirmed_receive is not None
            else track.last_seen_receive
        )
        if (
            camera_timeout_sec is not None
            and now_sec - camera_receive > camera_timeout_sec
        ):
            continue
        fresh[track_id] = track
    return fresh


def laser_points_in_target(
    ranges: Sequence[float],
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
    transform,
) -> np.ndarray:
    values = np.asarray(ranges, dtype=np.float64)
    indices = np.arange(values.size, dtype=np.float64)
    valid = (
        np.isfinite(values)
        & (values >= float(range_min))
        & (values <= float(range_max))
    )
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.float64)

    angles = float(angle_min) + indices[valid] * float(angle_increment)
    distances = values[valid]
    points = np.column_stack((
        distances * np.cos(angles),
        distances * np.sin(angles),
        np.zeros(distances.size, dtype=np.float64),
    ))
    rotation = transform.transform.rotation
    axis = np.asarray(
        [rotation.x, rotation.y, rotation.z],
        dtype=np.float64,
    )
    rotated = (
        points
        + 2.0 * float(rotation.w) * np.cross(axis, points)
        + 2.0 * np.cross(axis, np.cross(axis, points))
    )
    translation = transform.transform.translation
    rotated[:, 0] += float(translation.x)
    rotated[:, 1] += float(translation.y)
    return rotated[:, :2]


def associate_lidar_points(
    points_xy: np.ndarray,
    tracks: Sequence[Track],
    stamp_sec: float,
    radius_m: float,
    min_points: int,
    max_prediction_sec: float,
) -> Dict[int, Tuple[float, float, int]]:
    if points_xy.size == 0 or not tracks:
        return {}

    predictions = []
    for track in tracks:
        prediction_dt = min(
            max(0.0, float(stamp_sec) - track.last_seen),
            max(0.0, float(max_prediction_sec)),
        )
        predictions.append((
            track.x + track.vx * prediction_dt,
            track.y + track.vy * prediction_dt,
        ))
    predicted_xy = np.asarray(predictions, dtype=np.float64)
    offsets = points_xy[:, None, :] - predicted_xy[None, :, :]
    distance_sq = np.sum(offsets * offsets, axis=2)
    nearest_track = np.argmin(distance_sq, axis=1)
    nearest_distance_sq = distance_sq[
        np.arange(points_xy.shape[0]),
        nearest_track,
    ]
    radius_sq = float(radius_m) ** 2

    associations = {}
    for track_index, track in enumerate(tracks):
        assigned = points_xy[
            (nearest_track == track_index)
            & (nearest_distance_sq <= radius_sq)
        ]
        if assigned.shape[0] < int(min_points):
            continue
        median = np.median(assigned, axis=0)
        associations[track.track_id] = (
            float(median[0]),
            float(median[1]),
            int(assigned.shape[0]),
        )
    return associations


class RgbdPeopleDetector(Node):
    def __init__(self):
        super().__init__('rgbd_people_detector')

        self.declare_parameter(
            'color_topic',
            '/jackal1/sensors/camera_0/color/image',
        )
        self.declare_parameter(
            'depth_topic',
            '/jackal1/sensors/camera_0/aligned_depth_to_color/image',
        )
        self.declare_parameter(
            'camera_info_topic',
            '/jackal1/sensors/camera_0/aligned_depth_to_color/camera_info',
        )
        self.declare_parameter('people_topic', '/people')
        self.declare_parameter('marker_topic', '/people_detector/markers')
        self.declare_parameter('status_topic', '/people_detector/status')
        self.declare_parameter('target_frame', 'map')
        self.declare_parameter('detector_backend', 'ultralytics_yolo')
        self.declare_parameter(
            'yolo_model',
            '/workspace/SocialNavDiffusion_Inference/yolo11n.pt',
        )
        self.declare_parameter('yolo_image_size', 640)
        self.declare_parameter('score_threshold', 0.45)
        self.declare_parameter('inference_period_sec', 0.20)
        self.declare_parameter('people_publish_period_sec', 0.20)
        self.declare_parameter('source_timeout_sec', 1.0)
        self.declare_parameter('max_sync_delta_sec', 0.20)
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('min_depth_m', 0.30)
        self.declare_parameter('max_depth_m', 12.0)
        self.declare_parameter('min_valid_depth_pixels', 20)
        self.declare_parameter('max_people', 10)
        self.declare_parameter('association_distance_m', 1.0)
        self.declare_parameter('track_timeout_sec', 0.75)
        self.declare_parameter('velocity_smoothing', 0.50)
        self.declare_parameter('tf_timeout_sec', 0.15)
        self.declare_parameter('enable_lidar_fusion', True)
        self.declare_parameter(
            'lidar_topic',
            '/jackal1/sensors/lidar3d_0/scan',
        )
        self.declare_parameter('lidar_fusion_period_sec', 0.10)
        self.declare_parameter('lidar_association_radius_m', 0.55)
        self.declare_parameter('lidar_min_points', 2)
        self.declare_parameter('lidar_track_hold_sec', 1.50)
        self.declare_parameter('lidar_position_smoothing', 0.25)
        self.declare_parameter('lidar_velocity_smoothing', 0.25)
        self.declare_parameter('lidar_max_prediction_sec', 0.50)

        self.color_topic = str(self.get_parameter('color_topic').value)
        self.depth_topic = str(self.get_parameter('depth_topic').value)
        self.camera_info_topic = str(
            self.get_parameter('camera_info_topic').value
        )
        self.people_topic = str(self.get_parameter('people_topic').value)
        self.marker_topic = str(self.get_parameter('marker_topic').value)
        self.status_topic = str(self.get_parameter('status_topic').value)
        self.target_frame = str(self.get_parameter('target_frame').value)
        self.detector_backend = str(
            self.get_parameter('detector_backend').value
        ).strip().lower()
        self.yolo_model_path = str(
            self.get_parameter('yolo_model').value
        )
        self.enable_lidar_fusion = bool(
            self.get_parameter('enable_lidar_fusion').value
        )
        self.lidar_topic = str(self.get_parameter('lidar_topic').value)
        self.yolo_image_size = int(
            self.get_parameter('yolo_image_size').value
        )

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.latest_color: Optional[Image] = None
        self.latest_depth: Optional[Image] = None
        self.latest_camera_info: Optional[CameraInfo] = None
        self.latest_color_receive: Optional[float] = None
        self.latest_depth_receive: Optional[float] = None
        self.latest_info_receive: Optional[float] = None
        self.last_processed_color_stamp: Optional[Tuple[int, int]] = None
        self.last_successful_inference_receive: Optional[float] = None
        self.last_inference_fields = {
            'ready': False,
            'reason': 'waiting_for_first_inference',
        }
        self.last_lidar_process_receive: Optional[float] = None
        self.last_lidar_receive: Optional[float] = None
        self.last_lidar_point_count = 0
        self.last_lidar_track_updates = 0
        self.last_lidar_error = ''
        self.tracks: Dict[int, Track] = {}
        self.track_lock = threading.Lock()
        self.next_track_id = 1
        self.sensor_callback_group = MutuallyExclusiveCallbackGroup()
        self.inference_callback_group = MutuallyExclusiveCallbackGroup()
        self.publisher_callback_group = MutuallyExclusiveCallbackGroup()

        self.people_pub = self.create_publisher(People, self.people_topic, 10)
        self.marker_pub = self.create_publisher(
            MarkerArray,
            self.marker_topic,
            10,
        )
        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.depth_subscription = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            qos_profile_sensor_data,
            callback_group=self.sensor_callback_group,
        )
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            qos_profile_sensor_data,
            callback_group=self.sensor_callback_group,
        )
        self.color_subscription = self.create_subscription(
            Image,
            self.color_topic,
            self.color_callback,
            qos_profile_sensor_data,
            callback_group=self.sensor_callback_group,
        )
        self.lidar_subscription = None
        if self.enable_lidar_fusion:
            self.lidar_subscription = self.create_subscription(
                LaserScan,
                self.lidar_topic,
                self.lidar_callback,
                qos_profile_sensor_data,
                callback_group=self.sensor_callback_group,
            )
        self.inference_timer = self.create_timer(
            float(self.get_parameter('inference_period_sec').value),
            self.inference_callback,
            callback_group=self.inference_callback_group,
        )
        self.people_timer = self.create_timer(
            float(self.get_parameter('people_publish_period_sec').value),
            self.people_publish_callback,
            callback_group=self.publisher_callback_group,
        )

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        if self.detector_backend == 'ultralytics_yolo':
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    'detector_backend=ultralytics_yolo requires the '
                    'ultralytics package in the perception venv'
                ) from exc
            self.model = YOLO(self.yolo_model_path)
            self.detector_model_name = self.yolo_model_path
        elif self.detector_backend == 'torchvision_ssdlite':
            from torchvision.models.detection import (
                SSDLite320_MobileNet_V3_Large_Weights,
                ssdlite320_mobilenet_v3_large,
            )

            weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
            self.model = ssdlite320_mobilenet_v3_large(weights=weights)
            self.model.to(self.device)
            self.model.eval()
            self.detector_model_name = 'ssdlite320_mobilenet_v3_large'
        else:
            raise ValueError(
                'detector_backend must be ultralytics_yolo or '
                'torchvision_ssdlite'
            )
        self.get_logger().info(
            'RGB-D people detector ready: '
            f'backend={self.detector_backend}, '
            f'model={self.detector_model_name}, device={self.device}, '
            f'color={self.color_topic}, '
            f'depth={self.depth_topic}, output={self.people_topic}, '
            f'target_frame={self.target_frame}, '
            f'lidar_fusion={self.enable_lidar_fusion}, '
            f'lidar={self.lidar_topic}'
        )

    def depth_callback(self, msg: Image):
        self.latest_depth = msg
        self.latest_depth_receive = self.now_sec()

    def camera_info_callback(self, msg: CameraInfo):
        self.latest_camera_info = msg
        self.latest_info_receive = self.now_sec()

    def color_callback(self, msg: Image):
        self.latest_color = msg
        self.latest_color_receive = self.now_sec()

    def lidar_callback(self, msg: LaserScan):
        receive_sec = self.now_sec()
        period = max(
            0.0,
            float(self.get_parameter('lidar_fusion_period_sec').value),
        )
        if (
            self.last_lidar_process_receive is not None
            and receive_sec - self.last_lidar_process_receive < period
        ):
            return
        self.last_lidar_process_receive = receive_sec
        self.last_lidar_receive = receive_sec

        try:
            transform = self.lookup_transform(
                msg.header.frame_id,
                msg.header.stamp,
            )
            points_xy = laser_points_in_target(
                msg.ranges,
                msg.angle_min,
                msg.angle_increment,
                msg.range_min,
                msg.range_max,
                transform,
            )
            stamp_sec = stamp_to_seconds(msg.header.stamp)
            camera_hold = float(
                self.get_parameter('lidar_track_hold_sec').value
            )
            with self.track_lock:
                eligible_tracks = [
                    track
                    for track in self.tracks.values()
                    if receive_sec - (
                        track.camera_confirmed_receive
                        if track.camera_confirmed_receive is not None
                        else track.last_seen_receive
                    ) <= camera_hold
                ]
                associations = associate_lidar_points(
                    points_xy,
                    eligible_tracks,
                    stamp_sec,
                    float(
                        self.get_parameter(
                            'lidar_association_radius_m'
                        ).value
                    ),
                    int(self.get_parameter('lidar_min_points').value),
                    float(
                        self.get_parameter(
                            'lidar_max_prediction_sec'
                        ).value
                    ),
                )
                position_smoothing = min(1.0, max(
                    0.0,
                    float(
                        self.get_parameter(
                            'lidar_position_smoothing'
                        ).value
                    ),
                ))
                velocity_smoothing = min(1.0, max(
                    0.0,
                    float(
                        self.get_parameter(
                            'lidar_velocity_smoothing'
                        ).value
                    ),
                ))
                for track in eligible_tracks:
                    association = associations.get(track.track_id)
                    if association is None:
                        continue
                    measured_x, measured_y, _ = association
                    previous_x = track.x
                    previous_y = track.y
                    delta = stamp_sec - track.last_seen
                    track.x = (
                        position_smoothing * measured_x
                        + (1.0 - position_smoothing) * previous_x
                    )
                    track.y = (
                        position_smoothing * measured_y
                        + (1.0 - position_smoothing) * previous_y
                    )
                    if 0.03 <= delta <= 1.0:
                        measured_vx = (track.x - previous_x) / delta
                        measured_vy = (track.y - previous_y) / delta
                        track.vx = (
                            velocity_smoothing * measured_vx
                            + (1.0 - velocity_smoothing) * track.vx
                        )
                        track.vy = (
                            velocity_smoothing * measured_vy
                            + (1.0 - velocity_smoothing) * track.vy
                        )
                    track.last_seen = stamp_sec
                    track.last_seen_receive = receive_sec
                    track.last_lidar_receive = receive_sec
            self.last_lidar_point_count = int(points_xy.shape[0])
            self.last_lidar_track_updates = len(associations)
            self.last_lidar_error = ''
        except (TransformException, RuntimeError, ValueError) as exc:
            self.last_lidar_point_count = 0
            self.last_lidar_track_updates = 0
            self.last_lidar_error = type(exc).__name__

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def publish_status(self, **fields):
        fields.setdefault('backend', self.detector_backend)
        fields.setdefault('model', self.detector_model_name)
        message = String()
        message.data = json.dumps(fields, sort_keys=True)
        self.status_pub.publish(message)

    def remember_inference_status(self, **fields):
        self.last_inference_fields = dict(fields)

    def lookup_transform(self, source_frame: str, stamp):
        timeout = Duration(
            seconds=float(self.get_parameter('tf_timeout_sec').value)
        )
        try:
            return self.tf_buffer.lookup_transform(
                self.target_frame,
                source_frame,
                Time.from_msg(stamp),
                timeout=timeout,
            )
        except TransformException:
            return self.tf_buffer.lookup_transform(
                self.target_frame,
                source_frame,
                Time(),
                timeout=timeout,
            )

    def person_box_candidates(
        self,
        color_image: np.ndarray,
    ) -> List[Tuple[np.ndarray, float]]:
        threshold = float(self.get_parameter('score_threshold').value)
        max_people = int(self.get_parameter('max_people').value)
        if self.detector_backend == 'ultralytics_yolo':
            results = self.model.predict(
                source=color_image,
                classes=[0],
                conf=threshold,
                imgsz=self.yolo_image_size,
                max_det=max_people,
                device=0 if self.device.type == 'cuda' else 'cpu',
                verbose=False,
            )
            boxes = results[0].boxes
            if boxes is None:
                return []
            xyxy = boxes.xyxy.detach().cpu().numpy()
            scores = boxes.conf.detach().cpu().numpy()
            return [
                (box, float(score))
                for box, score in zip(xyxy, scores)
            ]

        rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        tensor = (
            torch.from_numpy(rgb)
            .permute(2, 0, 1)
            .contiguous()
            .to(self.device, dtype=torch.float32)
            / 255.0
        )
        with torch.inference_mode():
            output = self.model([tensor])[0]

        boxes = output['boxes'].detach().cpu().numpy()
        labels = output['labels'].detach().cpu().numpy()
        scores = output['scores'].detach().cpu().numpy()
        return [
            (box, float(score))
            for box, label, score in zip(boxes, labels, scores)
            if int(label) == 1 and float(score) >= threshold
        ][:max_people]

    def detect_people(
        self,
        color_image: np.ndarray,
        depth_image: np.ndarray,
        camera_info: CameraInfo,
        transform,
    ) -> Tuple[List[Detection], int]:
        candidates = self.person_box_candidates(color_image)
        max_people = int(self.get_parameter('max_people').value)
        depth_failures = 0
        detections: List[Detection] = []

        fx = float(camera_info.k[0])
        fy = float(camera_info.k[4])
        cx = float(camera_info.k[2])
        cy = float(camera_info.k[5])
        for box, score in candidates:
            depth_sample = depth_from_box(
                depth_image,
                box,
                float(self.get_parameter('depth_scale').value),
                float(self.get_parameter('min_depth_m').value),
                float(self.get_parameter('max_depth_m').value),
                int(self.get_parameter('min_valid_depth_pixels').value),
            )
            if depth_sample is None:
                depth_failures += 1
                continue
            depth, pixel_x, pixel_y = depth_sample
            camera_point = (
                (pixel_x - cx) * depth / fx,
                (pixel_y - cy) * depth / fy,
                depth,
            )
            map_x, map_y, map_z = transform_point(
                camera_point,
                transform,
            )
            detections.append(
                Detection(map_x, map_y, map_z, float(score))
            )
            if len(detections) >= max_people:
                break
        return detections, depth_failures

    def update_tracks(
        self,
        detections: List[Detection],
        stamp_sec: float,
        receive_sec: float,
    ) -> List[Track]:
        association_distance = float(
            self.get_parameter('association_distance_m').value
        )
        smoothing = float(self.get_parameter('velocity_smoothing').value)
        unmatched = set(range(len(detections)))

        for track in list(self.tracks.values()):
            candidates = [
                (
                    math.hypot(
                        detections[index].x - track.x,
                        detections[index].y - track.y,
                    ),
                    index,
                )
                for index in unmatched
            ]
            if not candidates:
                continue
            distance, index = min(candidates)
            if distance > association_distance:
                continue
            detection = detections[index]
            unmatched.remove(index)
            delta = stamp_sec - track.last_seen
            if 0.03 <= delta <= 2.0:
                measured_vx = (detection.x - track.x) / delta
                measured_vy = (detection.y - track.y) / delta
                track.vx = smoothing * measured_vx + (1.0 - smoothing) * track.vx
                track.vy = smoothing * measured_vy + (1.0 - smoothing) * track.vy
            track.x = detection.x
            track.y = detection.y
            track.z = detection.z
            track.score = detection.score
            track.last_seen = stamp_sec
            track.last_seen_receive = receive_sec
            track.camera_confirmed_receive = receive_sec

        for index in unmatched:
            detection = detections[index]
            track_id = self.next_track_id
            self.next_track_id += 1
            self.tracks[track_id] = Track(
                track_id=track_id,
                x=detection.x,
                y=detection.y,
                z=detection.z,
                vx=0.0,
                vy=0.0,
                score=detection.score,
                last_seen=stamp_sec,
                last_seen_receive=receive_sec,
                camera_confirmed_receive=receive_sec,
            )

        return sorted(self.tracks.values(), key=lambda track: track.track_id)

    def publish_people(self, tracks: List[Track], stamp):
        people = People()
        people.header.stamp = stamp
        people.header.frame_id = self.target_frame
        for track in tracks:
            person = Person()
            person.name = f'person_{track.track_id}'
            person.position.x = track.x
            person.position.y = track.y
            person.position.z = track.z
            person.velocity.x = track.vx
            person.velocity.y = track.vy
            person.reliability = track.score
            person.tagnames = ['source', 'class']
            person.tags = ['realsense_rgbd', 'person']
            people.people.append(person)
        self.people_pub.publish(people)

        markers = MarkerArray()
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)
        for track in tracks:
            marker = Marker()
            marker.header = people.header
            marker.ns = 'rgbd_people'
            marker.id = track.track_id
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = track.x
            marker.pose.position.y = track.y
            marker.pose.position.z = max(0.9, track.z)
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 1.8
            marker.color.r = 0.1
            marker.color.g = 0.9
            marker.color.b = 0.2
            marker.color.a = 0.75
            marker.lifetime = Duration(seconds=0.5).to_msg()
            markers.markers.append(marker)
        self.marker_pub.publish(markers)

    def people_publish_callback(self):
        now = self.now_sec()
        timeout = float(self.get_parameter('track_timeout_sec').value)
        camera_timeout = (
            float(self.get_parameter('lidar_track_hold_sec').value)
            if self.enable_lidar_fusion
            else None
        )
        with self.track_lock:
            self.tracks = fresh_tracks(
                self.tracks,
                now,
                timeout,
                camera_timeout_sec=camera_timeout,
            )
            tracks = sorted(
                self.tracks.values(),
                key=lambda track: track.track_id,
            )

        self.publish_people(tracks, self.get_clock().now().to_msg())

        source_timeout = float(
            self.get_parameter('source_timeout_sec').value
        )
        color_age = (
            now - self.latest_color_receive
            if self.latest_color_receive is not None
            else float('inf')
        )
        depth_age = (
            now - self.latest_depth_receive
            if self.latest_depth_receive is not None
            else float('inf')
        )
        info_age = (
            now - self.latest_info_receive
            if self.latest_info_receive is not None
            else float('inf')
        )
        inference_age = (
            now - self.last_successful_inference_receive
            if self.last_successful_inference_receive is not None
            else float('inf')
        )
        source_fresh = max(color_age, depth_age) <= source_timeout
        inference_fresh = inference_age <= source_timeout
        lidar_age = (
            now - self.last_lidar_receive
            if self.last_lidar_receive is not None
            else float('inf')
        )
        fields = dict(self.last_inference_fields)
        fields['ready'] = bool(
            fields.get('ready', False)
            and source_fresh
            and inference_fresh
        )
        if not source_fresh:
            fields['reason'] = 'rgbd_source_stale'
        elif not inference_fresh:
            fields['reason'] = 'inference_stale'
        fields.update({
            'heartbeat': True,
            'tracks': len(tracks),
            'color_age_sec': round(color_age, 3),
            'depth_age_sec': round(depth_age, 3),
            'camera_info_age_sec': round(info_age, 3),
            'inference_age_sec': round(inference_age, 3),
            'lidar_fusion_enabled': self.enable_lidar_fusion,
            'lidar_age_sec': round(lidar_age, 3),
            'lidar_points': self.last_lidar_point_count,
            'lidar_tracks_updated': self.last_lidar_track_updates,
            'lidar_error': self.last_lidar_error,
        })
        self.publish_status(**fields)

    def inference_callback(self):
        msg = self.latest_color
        depth_msg = self.latest_depth
        camera_info = self.latest_camera_info
        if msg is None or depth_msg is None or camera_info is None:
            self.remember_inference_status(
                ready=False,
                reason='waiting_for_color_depth_or_info',
            )
            return

        sync_delta = abs(
            stamp_to_seconds(msg.header.stamp)
            - stamp_to_seconds(depth_msg.header.stamp)
        )
        max_delta = float(self.get_parameter('max_sync_delta_sec').value)
        if sync_delta > max_delta:
            self.remember_inference_status(
                ready=False,
                reason='color_depth_out_of_sync',
                sync_delta_sec=sync_delta,
            )
            return

        color_stamp = (
            int(msg.header.stamp.sec),
            int(msg.header.stamp.nanosec),
        )
        if color_stamp == self.last_processed_color_stamp:
            return
        self.last_processed_color_stamp = color_stamp

        start = time.perf_counter()
        try:
            color = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            depth = self.bridge.imgmsg_to_cv2(
                depth_msg,
                desired_encoding='passthrough',
            )
            transform = self.lookup_transform(
                msg.header.frame_id,
                msg.header.stamp,
            )
            detections, depth_failures = self.detect_people(
                color,
                depth,
                camera_info,
                transform,
            )
            receive_sec = self.now_sec()
            with self.track_lock:
                tracks = self.update_tracks(
                    detections,
                    stamp_to_seconds(msg.header.stamp),
                    receive_sec,
                )
            self.last_successful_inference_receive = receive_sec
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.remember_inference_status(
                ready=True,
                device=str(self.device),
                detections=len(detections),
                tracks=len(tracks),
                depth_failures=depth_failures,
                inference_ms=round(elapsed_ms, 2),
                sync_delta_sec=round(sync_delta, 4),
            )
        except (TransformException, RuntimeError, ValueError) as exc:
            self.remember_inference_status(
                ready=False,
                reason=type(exc).__name__,
                detail=str(exc),
            )
            self.get_logger().warn(
                f'People detection frame skipped: {type(exc).__name__}: {exc}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = RgbdPeopleDetector()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
