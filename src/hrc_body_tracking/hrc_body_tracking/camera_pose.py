"""Camera -> world pose shared by sensing_node and human_dynamics_node.

roll/pitch come only from the Kinect IMU, measured by sensing_node at startup.
yaw/t come from server/camera_pose.yaml (override the path with HRC_CAMERA_POSE_FILE).
sensing_node publishes the combined pose once on /hrc/camera_pose (std_msgs/String JSON,
reliable + transient_local); human_dynamics_node computes nothing until it arrives.

Convention:
    cam_R = Rx(roll) @ Ry(pitch) @ Rz(yaw)   (utils.camera_to_world_rotmat)
    p_world = cam_R.T @ p_color + t          (color-camera frame -> world, z up, metres)
"""
import json
import os
import time

import yaml
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from .paths import ws

CAMERA_POSE_TOPIC = "/hrc/camera_pose"
DEFAULT_CAMERA_POSE_FILE = ws("src/hrc_body_tracking/server/camera_pose.yaml")
REQUIRED_KEYS = ("yaw_deg", "t")
REMOVED_KEYS = ("roll_deg", "pitch_deg", "use_imu_gravity", "manual_roll_deg", "last_imu_measurement")


def camera_pose_file():
    return os.environ.get("HRC_CAMERA_POSE_FILE") or DEFAULT_CAMERA_POSE_FILE


def camera_pose_qos():
    """Latched-style QoS: late subscribers still get the single published pose."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def load_camera_pose(path=None):
    """yaw_deg and t from the YAML. roll/pitch are not configurable (IMU only)."""
    path = path or camera_pose_file()
    with open(path, "r") as f:
        pose = yaml.safe_load(f) or {}
    stale = [k for k in REMOVED_KEYS if k in pose]
    if stale:
        raise ValueError(f"{path}: remove {stale}; roll/pitch are measured by the IMU only")
    missing = [k for k in REQUIRED_KEYS if k not in pose]
    if missing:
        raise KeyError(f"{path}: missing keys {missing}")
    t = [float(v) for v in pose["t"]]
    if len(t) != 3:
        raise ValueError(f"{path}: t must have 3 values")
    return {"yaw_deg": float(pose["yaw_deg"]), "t": t}


def write_measurement_log(data, directory=None):
    """Record one IMU camera-pose measurement as debug_log/camera_pose_<time>.yaml."""
    directory = directory or ws("debug_log")
    os.makedirs(directory, exist_ok=True)
    stem = os.path.join(directory, f"camera_pose_{time.strftime('%Y%m%d_%H%M%S')}")
    path, n = stem + ".yaml", 1
    while os.path.exists(path):  # never overwrite an earlier record from the same second
        path, n = f"{stem}_{n}.yaml", n + 1
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return path


def pose_message_data(roll_deg, pitch_deg, yaw_deg, t, source, **extra):
    """JSON payload for /hrc/camera_pose."""
    return json.dumps({"roll_deg": float(roll_deg), "pitch_deg": float(pitch_deg),
                       "yaw_deg": float(yaw_deg), "t": [float(v) for v in t],
                       "source": str(source), **extra})


def parse_pose_message(data):
    d = json.loads(data)
    t = [float(v) for v in d["t"]]
    if len(t) != 3:
        raise ValueError("t must have 3 values")
    return float(d["roll_deg"]), float(d["pitch_deg"]), float(d["yaw_deg"]), t, d.get("source", "?"), d
