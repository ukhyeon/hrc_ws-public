#!/usr/bin/env python3
"""Measure camera roll/pitch from the Azure Kinect IMU (gravity), standalone.

Talks to libk4a (1.3 or 1.4, see imu_k4a.find_libk4a) directly through ctypes (shared code: src/hrc_body_tracking/
hrc_body_tracking/imu_k4a.py, which sensing_node also uses at startup); it does not use
pyk4a and changes nothing in the running system. The sensing node must be stopped first
(only one process can open the device).

What it does
  1. open device 0, start the depth camera with a minimal config (the IMU only
     runs while a camera is running), start the IMU
  2. average the accelerometer for --seconds (default 2 s, first 0.3 s skipped)
  3. stop IMU and cameras, close the device
  4. read accel->depth and accel->color extrinsics from k4a_device_get_calibration
     and express the gravity direction in the accel, depth and color frames
  5. report roll/pitch in two conventions and save everything to YAML

Conventions
  * hrc_ws (human_dynamics_node / sensing_node):
        cam_R = Rx(roll) @ Ry(pitch) @ Rz(yaw)          (utils.camera_to_world_rotmat)
        p_world = cam_R.T @ p_color + t_cam_to_world     (world z up)
    so the unit "down" vector seen in the camera frame is
        g_c = cam_R @ (0, 0, -1) = (-sin p, sin r cos p, -cos r cos p)
    =>  pitch = -asin(g_x),  roll = atan2(g_y, -g_z).  Yaw is not observable from gravity.
    The SMPL results that cam_R is applied to are in the COLOR camera frame
    (Kinect joints are moved depth -> color before inference), so the color-frame
    values are the ones sensing_node measures at startup and publishes on /hrc/camera_pose.
  * make_R (reference, pyk4a.PyK4A.get_imu_rotation_matrix, a port of the lost C++
    make_R / imuToRotaionAngle): applied to the RAW accelerometer vector
        acc_phi = atan2(a_y, a_z), acc_theta = atan2(a_x, hypot(a_y, a_z))
        R = Flip(diag(-1,-1,1)) @ Rz(-acc_phi) @ Rx(-acc_theta)   (4x4)
    Its angles are in accel-sensor axes and are NOT the same numbers as hrc roll/pitch.

Usage
  python3 tools/imu_gravity.py                  # measure and save YAML
  python3 tools/imu_gravity.py --seconds 3 --out /path/result.yaml
  python3 tools/imu_gravity.py --layout         # print ctypes sizeof/offsetof only (no device)
"""
import argparse
import ctypes as C
import datetime as dt
import math
import os
import sys

import numpy as np
import yaml

# Shared ctypes structures, calibration reading and the gravity -> roll/pitch math
# live in hrc_body_tracking/imu_k4a.py (also used by sensing_node at startup).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "hrc_body_tracking"))
from hrc_body_tracking import imu_k4a as K  # noqa: E402
from hrc_body_tracking.imu_k4a import (  # noqa: E402,F401  (re-exported for --layout / tests)
    CAL_ACCEL, CAL_COLOR, CAL_DEPTH, K4A_COLOR_RESOLUTION_OFF, K4A_DEPTH_MODE_NFOV_2X2BINNED,
    K4A_FRAMES_PER_SECOND_30, K4A_IMAGE_FORMAT_COLOR_MJPG, K4A_RESULT_SUCCEEDED,
    K4A_WIRED_SYNC_MODE_STANDALONE, Rx, Ry, Rz, extr, hrc_roll_pitch, k4a_device_configuration_t,
    load_lib, print_layout,
)


def make_R(acc):
    """Exact port of pyk4a get_imu_rotation_matrix (raw accelerometer vector)."""
    g = acc / np.linalg.norm(acc)
    acc_phi = math.atan2(g[1], g[2])
    acc_theta = math.atan2(g[0], math.sqrt(g[1] ** 2 + g[2] ** 2))
    theta_z, theta_x = -acc_phi, -acc_theta
    Rz4, Rx4, Flip = np.eye(4), np.eye(4), np.eye(4)
    Rz4[:3, :3] = Rz(theta_z)
    Rx4[:3, :3] = Rx(theta_x)
    Flip[0, 0] = Flip[1, 1] = -1
    return math.degrees(acc_phi), math.degrees(acc_theta), Flip @ Rz4 @ Rx4


def as_list(a, nd=6):
    return np.round(np.asarray(a, dtype=float), nd).tolist()


# ---------------------------------------------------------------- device
def measure(seconds, skip_s=0.3):
    lib = load_lib()
    n_dev = lib.k4a_device_get_installed_count()
    if n_dev < 1:
        sys.exit("No Azure Kinect device found.")
    dev = C.c_void_p()
    if lib.k4a_device_open(0, C.byref(dev)) != K4A_RESULT_SUCCEEDED:
        sys.exit("k4a_device_open failed (device busy? stop sensing_node / k4aviewer first).")
    cams = False
    try:
        cfg = k4a_device_configuration_t(
            K4A_IMAGE_FORMAT_COLOR_MJPG, K4A_COLOR_RESOLUTION_OFF, K4A_DEPTH_MODE_NFOV_2X2BINNED,
            K4A_FRAMES_PER_SECOND_30, False, 0, K4A_WIRED_SYNC_MODE_STANDALONE, 0, False)
        if lib.k4a_device_start_cameras(dev, C.byref(cfg)) != K4A_RESULT_SUCCEEDED:
            sys.exit("k4a_device_start_cameras failed")
        cams = True
        try:
            cal = K.get_calibration(lib, dev)
            acc, gyro, temp = K.read_imu(lib, dev, seconds=seconds, skip_s=skip_s, start_stop=True)
        except RuntimeError as e:
            sys.exit(str(e))
    finally:
        if cams:
            lib.k4a_device_stop_cameras(dev)
        lib.k4a_device_close(dev)
    return acc, gyro, temp, cal


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--out", default=None,
                    help="YAML path (default ~/hrc_ws/debug_log/imu_gravity_<time>.yaml)")
    ap.add_argument("--layout", action="store_true", help="print ctypes sizeof/offsetof and exit")
    args = ap.parse_args()
    if args.layout:
        print_layout()
        return

    acc, gyro, temp, cal = measure(args.seconds)
    a = acc.mean(0)
    a_norm = float(np.linalg.norm(a))
    a_std = acc.std(0)
    gyro_rms = float(np.sqrt((gyro ** 2).sum(1).mean()))

    R_a2d, t_a2d = extr(cal, CAL_ACCEL, CAL_DEPTH)
    R_a2c, t_a2c = extr(cal, CAL_ACCEL, CAL_COLOR)
    R_d2c, t_d2c = extr(cal, CAL_DEPTH, CAL_COLOR)

    down_a, down_d, down_c, flipped = K.down_vectors(a, cal)

    roll_c, pitch_c, Rc, res_c = hrc_roll_pitch(down_c)
    roll_d, pitch_d, Rd, res_d = hrc_roll_pitch(down_d)
    phi, theta, R_makeR = make_R(a)

    det = {k: float(np.linalg.det(R)) for k, R in (("accel_to_depth", R_a2d), ("accel_to_color", R_a2c),
                                                   ("depth_to_color", R_d2c))}
    ortho = {k: float(np.abs(R @ R.T - np.eye(3)).max()) for k, R in (("accel_to_depth", R_a2d),
                                                                      ("accel_to_color", R_a2c))}
    tilt_c = math.degrees(math.asin(max(-1, min(1, down_c[2]))))  # optical axis below horizontal

    node_ok, node_info, node_reasons = K.quality(acc, gyro)  # same check sensing_node applies
    result = {
        "time": dt.datetime.now().isoformat(timespec="seconds"),
        "samples": int(len(acc)),
        "duration_s": args.seconds,
        "temperature_c": float(temp.mean()),
        "accel_mean_mps2": as_list(a),
        "accel_norm_mps2": round(a_norm, 4),
        "accel_std_mps2": as_list(a_std),
        "gyro_rms_radps": round(gyro_rms, 5),
        "down_sign_flipped": flipped,
        "extrinsics": {
            "accel_to_depth": {"R": as_list(R_a2d), "t_mm": as_list(t_a2d, 3)},
            "accel_to_color": {"R": as_list(R_a2c), "t_mm": as_list(t_a2c, 3)},
            "depth_to_color": {"R": as_list(R_d2c), "t_mm": as_list(t_d2c, 3)},
            "det": {k: round(v, 6) for k, v in det.items()},
            "max_abs_RRt_minus_I": {k: float(f"{v:.2e}") for k, v in ortho.items()},
        },
        "gravity_down_unit": {"accel": as_list(down_a), "depth": as_list(down_d), "color": as_list(down_c)},
        "hrc_convention": {
            "definition": "cam_R = Rx(roll) Ry(pitch) Rz(yaw); p_world = cam_R.T p_cam + t; world z up; "
                          "yaw not observable (kept 0)",
            "color": {"roll_deg": round(roll_c, 3), "pitch_deg": round(pitch_c, 3),
                      "cam_R": as_list(Rc), "fit_residual": float(f"{res_c:.2e}"),
                      "optical_axis_below_horizontal_deg": round(tilt_c, 3)},
            "depth": {"roll_deg": round(roll_d, 3), "pitch_deg": round(pitch_d, 3),
                      "cam_R": as_list(Rd), "fit_residual": float(f"{res_d:.2e}")},
        },
        "sensing_node_check": {"passed": node_ok, "reasons": node_reasons, **node_info},
        "make_R_convention": {
            "definition": "pyk4a get_imu_rotation_matrix on the RAW accel vector: "
                          "R = Flip(diag(-1,-1,1)) Rz(-acc_phi) Rx(-acc_theta)",
            "acc_phi_deg": round(phi, 3), "acc_theta_deg": round(theta, 3),
            "R_total_4x4": as_list(R_makeR),
        },
    }

    print(f"samples {len(acc)} over {args.seconds:.1f} s, temperature {temp.mean():.1f} C")
    print(f"|accel| = {a_norm:.4f} m/s^2 (expect ~9.81)   std {np.round(a_std, 4)}   gyro rms {gyro_rms:.4f} rad/s")
    if abs(a_norm - 9.81) > 0.3:
        print("  WARNING: |accel| far from 9.81 -- camera moving or bad reading")
    if a_std.max() > 0.05 or gyro_rms > 0.02:
        print("  WARNING: camera does not look still")
    print("det(R): " + ", ".join(f"{k} {v:.6f}" for k, v in det.items()))
    print(f"down (unit) accel {np.round(down_a, 4)}  depth {np.round(down_d, 4)}  color {np.round(down_c, 4)}"
          + ("   [sign flipped]" if flipped else ""))
    print(f"hrc convention, COLOR frame: roll {roll_c:.2f} deg, pitch {pitch_c:.2f} deg; "
          f"optical axis {tilt_c:.2f} deg below horizontal")
    print(f"hrc convention, DEPTH frame: roll {roll_d:.2f} deg, pitch {pitch_d:.2f} deg")
    print(f"make_R convention (raw accel): acc_phi {phi:.2f} deg, acc_theta {theta:.2f} deg")
    print(f"sensing_node check: {'PASS' if node_ok else 'FAIL ' + '; '.join(node_reasons)} "
          f"(|g| {node_info['g_norm_mps2']}, accel std {node_info['accel_std_max_mps2']}, "
          f"gyro std {node_info['gyro_std_max_radps']})")
    print("cam_R (color) =\n" + np.array2string(Rc, precision=5, suppress_small=True))

    out = args.out or os.path.expanduser(
        f"~/hrc_ws/debug_log/imu_gravity_{dt.datetime.now():%Y%m%d_%H%M%S}.yaml")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        yaml.safe_dump(result, f, sort_keys=False, allow_unicode=True)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
