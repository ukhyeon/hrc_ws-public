"""Azure Kinect IMU gravity -> camera roll/pitch, via libk4a (1.3 or 1.4) and ctypes.

Shared by tools/imu_gravity.py (opens its own device) and sensing_node.py
(reuses the device handle that pyk4a's k4a_module already opened).
No pyk4a rebuild is needed: k4a_module exports its handle as the global
``device`` (k4a_device_t, 8 bytes). ``load_lib`` opens the libk4a that is already
mapped in the process (the one k4a_module is linked against), so both use the same
library instance; see ``find_libk4a`` for the order.

Structures follow /usr/include/k4a/k4atypes.h (sizeof/offsetof were checked
against gcc; see ``print_layout``).

Camera-pose convention (hrc_ws, utils.camera_to_world_rotmat):
    cam_R = Rx(roll) @ Ry(pitch) @ Rz(yaw);  p_world = cam_R.T @ p_color + t;  world z up
    unit down vector in the camera frame: g = cam_R @ (0,0,-1) = (-sin p, sin r cos p, -cos r cos p)
    => pitch = -asin(g_x), roll = atan2(g_y, -g_z); yaw is not observable from gravity.
cam_R is applied to COLOR-camera-frame points (Kinect joints are moved depth -> color).
"""
import ctypes as C
import math
import time

import numpy as np

# Fallback sonames when no libk4a is loaded yet and ldconfig does not know one.
# 1.3 = SDK on the reference PC (BT 1.0), 1.4 = SDK 1.4.1 (BT 1.1, Ampere and newer GPUs).
LIBK4A_FALLBACKS = ("libk4a.so.1.4", "libk4a.so.1.3")
LIBK4A_ENV = "HRC_LIBK4A"   # optional override: soname or absolute path

# ---------------------------------------------------------------- enums (k4atypes.h)
K4A_RESULT_SUCCEEDED = 0
K4A_WAIT_RESULT_SUCCEEDED, K4A_WAIT_RESULT_FAILED, K4A_WAIT_RESULT_TIMEOUT = 0, 1, 2
K4A_IMAGE_FORMAT_COLOR_MJPG = 0
K4A_COLOR_RESOLUTION_OFF, K4A_COLOR_RESOLUTION_720P = 0, 1
K4A_DEPTH_MODE_OFF, K4A_DEPTH_MODE_NFOV_2X2BINNED = 0, 1
K4A_FRAMES_PER_SECOND_30 = 2
K4A_WIRED_SYNC_MODE_STANDALONE = 0
CAL_DEPTH, CAL_COLOR, CAL_GYRO, CAL_ACCEL, CAL_NUM = 0, 1, 2, 3, 4

# acceptance thresholds for a gravity measurement
G_NOMINAL = 9.81
G_TOL = 0.3            # | |a| - 9.81 | <= 0.3 m/s^2
ACC_STD_MAX = 0.05     # per-axis accel std [m/s^2] while still
GYRO_STD_MAX = 0.02    # per-axis gyro std (bias removed) [rad/s] while still


# ---------------------------------------------------------------- structs (k4atypes.h)
class k4a_device_configuration_t(C.Structure):
    _fields_ = [
        ("color_format", C.c_int),
        ("color_resolution", C.c_int),
        ("depth_mode", C.c_int),
        ("camera_fps", C.c_int),
        ("synchronized_images_only", C.c_bool),
        ("depth_delay_off_color_usec", C.c_int32),
        ("wired_sync_mode", C.c_int),
        ("subordinate_delay_off_master_usec", C.c_uint32),
        ("disable_streaming_indicator", C.c_bool),
    ]


class k4a_float3_t(C.Structure):  # union {struct {x,y,z}; float v[3]} -> same layout
    _fields_ = [("x", C.c_float), ("y", C.c_float), ("z", C.c_float)]


class k4a_imu_sample_t(C.Structure):
    _fields_ = [
        ("temperature", C.c_float),
        ("acc_sample", k4a_float3_t),
        ("acc_timestamp_usec", C.c_uint64),
        ("gyro_sample", k4a_float3_t),
        ("gyro_timestamp_usec", C.c_uint64),
    ]


class k4a_calibration_extrinsics_t(C.Structure):
    _fields_ = [("rotation", C.c_float * 9), ("translation", C.c_float * 3)]


class k4a_calibration_intrinsics_t(C.Structure):
    _fields_ = [
        ("type", C.c_int),
        ("parameter_count", C.c_uint),
        ("parameters", C.c_float * 15),  # union {struct of 15 floats; float v[15]}
    ]


class k4a_calibration_camera_t(C.Structure):
    _fields_ = [
        ("extrinsics", k4a_calibration_extrinsics_t),
        ("intrinsics", k4a_calibration_intrinsics_t),
        ("resolution_width", C.c_int),
        ("resolution_height", C.c_int),
        ("metric_radius", C.c_float),
    ]


class k4a_calibration_t(C.Structure):
    _fields_ = [
        ("depth_camera_calibration", k4a_calibration_camera_t),
        ("color_camera_calibration", k4a_calibration_camera_t),
        ("extrinsics", (k4a_calibration_extrinsics_t * CAL_NUM) * CAL_NUM),
        ("depth_mode", C.c_int),
        ("color_resolution", C.c_int),
    ]


LAYOUT = {
    "k4a_device_configuration_t": (k4a_device_configuration_t,
        ["synchronized_images_only", "depth_delay_off_color_usec", "wired_sync_mode",
         "subordinate_delay_off_master_usec", "disable_streaming_indicator"]),
    "k4a_imu_sample_t": (k4a_imu_sample_t,
        ["acc_sample", "acc_timestamp_usec", "gyro_sample", "gyro_timestamp_usec"]),
    "k4a_calibration_extrinsics_t": (k4a_calibration_extrinsics_t, ["translation"]),
    "k4a_calibration_intrinsics_t": (k4a_calibration_intrinsics_t, ["parameter_count", "parameters"]),
    "k4a_calibration_camera_t": (k4a_calibration_camera_t,
        ["intrinsics", "resolution_width", "resolution_height", "metric_radius"]),
    "k4a_calibration_t": (k4a_calibration_t,
        ["color_camera_calibration", "extrinsics", "depth_mode", "color_resolution"]),
}


def print_layout():
    for name, (cls, fields) in LAYOUT.items():
        print(f"{name} sizeof {C.sizeof(cls)}")
        for f in fields:
            print(f"{name}.{f} offsetof {getattr(cls, f).offset}")


# ---------------------------------------------------------------- library / handle
def find_libk4a():
    """(name_or_path, source) of the libk4a to open, without hard-coding the SDK version.

    1. a libk4a.so* already mapped in this process (e.g. by pyk4a's k4a_module): its path,
       so dlopen returns that same instance (same globals, same device handles)
    2. $HRC_LIBK4A
    3. ctypes.util.find_library("k4a") (ldconfig cache, e.g. libk4a.so.1.4)
    4. LIBK4A_FALLBACKS in order
    """
    import os
    mapped = libk4a_mappings()
    if len(mapped) == 1:
        return mapped[0], "mapped"
    if len(mapped) > 1:
        raise RuntimeError(f"more than one libk4a mapped in this process: {mapped}")
    env = os.environ.get(LIBK4A_ENV)
    if env:
        return env, LIBK4A_ENV
    import ctypes.util
    found = ctypes.util.find_library("k4a")
    if found:
        return found, "find_library"
    for name in LIBK4A_FALLBACKS:
        try:
            C.CDLL(name)
            return name, "fallback"
        except OSError:
            pass
    raise OSError(f"libk4a not found (tried mapped, ${LIBK4A_ENV}, find_library('k4a'), {LIBK4A_FALLBACKS})")


def load_lib():
    """libk4a (see find_libk4a) with argtypes set. If k4a_module is already loaded, this is the same instance."""
    name, _source = find_libk4a()
    lib = C.CDLL(name)
    lib.k4a_device_get_installed_count.restype = C.c_uint32
    lib.k4a_device_open.argtypes = [C.c_uint32, C.POINTER(C.c_void_p)]
    lib.k4a_device_open.restype = C.c_int
    lib.k4a_device_start_cameras.argtypes = [C.c_void_p, C.POINTER(k4a_device_configuration_t)]
    lib.k4a_device_start_cameras.restype = C.c_int
    lib.k4a_device_start_imu.argtypes = [C.c_void_p]
    lib.k4a_device_start_imu.restype = C.c_int
    lib.k4a_device_get_imu_sample.argtypes = [C.c_void_p, C.POINTER(k4a_imu_sample_t), C.c_int32]
    lib.k4a_device_get_imu_sample.restype = C.c_int
    lib.k4a_device_get_calibration.argtypes = [C.c_void_p, C.c_int, C.c_int, C.POINTER(k4a_calibration_t)]
    lib.k4a_device_get_calibration.restype = C.c_int
    for fn in ("k4a_device_stop_imu", "k4a_device_stop_cameras", "k4a_device_close"):
        getattr(lib, fn).argtypes = [C.c_void_p]
        getattr(lib, fn).restype = None
    return lib


def pyk4a_device_handle():
    """The k4a_device_t that pyk4a's k4a_module opened (exported global ``device``).

    Returns a c_void_p bound to the module's global; ``.value`` is None if no device is open.
    """
    import k4a_module  # the extension pyk4a uses; already loaded when pyk4a is imported
    mod = C.CDLL(k4a_module.__file__)  # dlopen of an already loaded object -> same globals
    return C.c_void_p.in_dll(mod, "device")


def libk4a_mappings():
    """Distinct (device, inode) pairs of libk4a.so* mapped in this process (expect exactly 1)."""
    import os
    seen = {}
    with open("/proc/self/maps") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 6 and "libk4a.so" in parts[5] and "libk4abt" not in parts[5]:
                st = os.stat(parts[5])
                seen[(st.st_dev, st.st_ino)] = parts[5]
    return sorted(seen.values())


def get_calibration(lib, handle, depth_mode=K4A_DEPTH_MODE_NFOV_2X2BINNED,
                    color_resolution=K4A_COLOR_RESOLUTION_720P):
    cal = k4a_calibration_t()
    if lib.k4a_device_get_calibration(handle, depth_mode, color_resolution, C.byref(cal)) != K4A_RESULT_SUCCEEDED:
        raise RuntimeError("k4a_device_get_calibration failed")
    return cal


def extr(cal, src, dst):
    """(R, t_mm) of calibration.extrinsics[src][dst]: p_dst = R p_src + t (row-major R)."""
    e = cal.extrinsics[src][dst]
    R = np.array(e.rotation[:], dtype=np.float64).reshape(3, 3)
    t = np.array(e.translation[:], dtype=np.float64)
    return R, t


def read_imu(lib, handle, seconds=2.0, skip_s=0.3, start_stop=True):
    """Collect IMU samples for `seconds` after discarding the first `skip_s`.

    With start_stop=True this starts the IMU and always stops it again (cameras must
    already be running). Returns (acc[N,3], gyro[N,3], temp[N]).
    """
    started = False
    try:
        if start_stop:
            if lib.k4a_device_start_imu(handle) != K4A_RESULT_SUCCEEDED:
                raise RuntimeError("k4a_device_start_imu failed (cameras running? IMU already started?)")
            started = True
        acc, gyro, temp = [], [], []
        s = k4a_imu_sample_t()
        t_skip = time.monotonic() + skip_s
        t_end = t_skip + seconds
        while time.monotonic() < t_end:
            r = lib.k4a_device_get_imu_sample(handle, C.byref(s), 100)
            if r == K4A_WAIT_RESULT_TIMEOUT:
                continue
            if r != K4A_WAIT_RESULT_SUCCEEDED:
                raise RuntimeError("k4a_device_get_imu_sample failed")
            if time.monotonic() < t_skip:
                continue
            acc.append((s.acc_sample.x, s.acc_sample.y, s.acc_sample.z))
            gyro.append((s.gyro_sample.x, s.gyro_sample.y, s.gyro_sample.z))
            temp.append(s.temperature)
    finally:
        if started:
            lib.k4a_device_stop_imu(handle)
    if len(acc) < 10:
        raise RuntimeError(f"too few IMU samples ({len(acc)})")
    return np.array(acc), np.array(gyro), np.array(temp)


# ---------------------------------------------------------------- math
def Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def down_vectors(acc_mean, cal):
    """Unit gravity ("down") direction in the accel, depth and color frames.

    At rest an accelerometer reads the support reaction (+g, pointing up), so down = -acc.
    The camera is mounted upright (image y points down), so down must have a positive
    color-frame y component; if not, the sign is flipped and flipped=True is returned.
    """
    a_norm = float(np.linalg.norm(acc_mean))
    R_a2d, _ = extr(cal, CAL_ACCEL, CAL_DEPTH)
    R_a2c, _ = extr(cal, CAL_ACCEL, CAL_COLOR)
    down_a = -np.asarray(acc_mean, dtype=np.float64) / a_norm
    flipped = False
    if (R_a2c @ down_a)[1] < 0:
        down_a = -down_a
        flipped = True
    return down_a, R_a2d @ down_a, R_a2c @ down_a, flipped


def hrc_roll_pitch(down):
    """roll/pitch (deg) of cam_R = Rx(roll) Ry(pitch) such that cam_R @ (0,0,-1) = down."""
    g = down / np.linalg.norm(down)
    pitch = -math.asin(max(-1.0, min(1.0, g[0])))
    roll = math.atan2(g[1], -g[2])
    R = Rx(roll) @ Ry(pitch)
    resid = float(np.linalg.norm(R @ np.array([0.0, 0.0, -1.0]) - g))
    return math.degrees(roll), math.degrees(pitch), R, resid


def quality(acc, gyro):
    """Acceptance check for a gravity measurement. Returns (ok, info dict, reasons list)."""
    a = acc.mean(0)
    g_norm = float(np.linalg.norm(a))
    acc_std = acc.std(0)
    gyro_std = gyro.std(0)
    reasons = []
    if abs(g_norm - G_NOMINAL) > G_TOL:
        reasons.append(f"|g| {g_norm:.3f} outside {G_NOMINAL}+/-{G_TOL}")
    if acc_std.max() > ACC_STD_MAX:
        reasons.append(f"accel std {acc_std.max():.4f} > {ACC_STD_MAX}")
    if gyro_std.max() > GYRO_STD_MAX:
        reasons.append(f"gyro std {gyro_std.max():.4f} > {GYRO_STD_MAX}")
    info = {"g_norm_mps2": round(g_norm, 4), "accel_std_max_mps2": round(float(acc_std.max()), 5),
            "gyro_std_max_radps": round(float(gyro_std.max()), 5), "samples": int(len(acc))}
    return not reasons, info, reasons


def measure_roll_pitch_with_handle(lib, handle, seconds=2.0, skip_s=0.3):
    """Used by sensing_node: IMU on an already-open, camera-running device handle.

    Returns dict with roll_deg/pitch_deg (color frame, hrc convention), quality info and
    the down vectors. Raises RuntimeError on device errors (caller falls back to YAML).
    """
    cal = get_calibration(lib, handle)
    acc, gyro, temp = read_imu(lib, handle, seconds=seconds, skip_s=skip_s, start_stop=True)
    ok, info, reasons = quality(acc, gyro)
    down_a, down_d, down_c, flipped = down_vectors(acc.mean(0), cal)
    roll, pitch, R, resid = hrc_roll_pitch(down_c)
    return {
        "ok": ok, "reasons": reasons, **info,
        "temperature_c": round(float(temp.mean()), 2),
        "roll_deg": roll, "pitch_deg": pitch, "fit_residual": resid,
        "down_color": down_c.tolist(), "down_sign_flipped": flipped,
    }
