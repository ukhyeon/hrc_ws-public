#!/usr/bin/env python3
"""Fit camera yaw and t (camera_pose.yaml) from a person standing on floor marks.

roll/pitch stay fixed (IMU, /hrc/camera_pose). Only yaw and t are fitted, so the
horizontal part is a 2D rotation + translation (no scale) and t_z is set so the
soles are at world z = 0.

Measured point
  /hrc/human_dynamics_state publishes 19 body-part COM positions in world, not
  joints. The ankle joint is not published, so the point used here is the
  midpoint of the left_foot and right_foot segment COMs (segment = SMPL ankle +
  foot vertices, the same merge human_dynamics_node uses). Its height above the
  sole is computed from the SMPL-H male model with the person's betas
  (from /hrc/human_pose), exactly as human_dynamics_node.mass_inertia does.

Model (human_dynamics_node.aa2euler + rigid_fk_19)
  cam_R = Rx(roll) Ry(pitch) Rz(yaw), A = cam_R^T
  Since 2026-11-18 aa2euler moves the pelvis as a point
  (trans_world = A (trans_cam + J0) + t - J0_rest), so
      x_hd = A x_c + t                   (t = real camera position)
  Before that fix only global_orient/transl were rotated, which gives
      x_hd = A x_c + (I - A) J0 + t      (J0 = SMPL rest-pose pelvis, betas)
  Records carry "hd_pelvis_corrected"; records without it are treated as
  corrected, and --legacy treats them as taken with the old node.
  Records keep the pose that was active; x_c is recovered from it, so records
  taken with any yaw/t can be refitted.

Usage (source install/setup.bash first for record)
  python3 tools/camera_pose_fit.py record --mark M1 -1.40 -0.30 --mark M2 -0.80 0.50 \
        --mark M3 -1.80 0.50 --mark M4 -1.40 1.20 --laps 2
  python3 tools/camera_pose_fit.py fit debug_log/camera_pose_fit/<session>
  python3 tools/camera_pose_fit.py touch --part right_hand --xyz X Y Z --label base_top
  python3 tools/camera_pose_fit.py body --betas-npz debug_npz/<file>.npz
  python3 tools/camera_pose_fit.py selftest
"""
import argparse
import glob
import json
import math
import os
import sys
import time

sys.dont_write_bytecode = True
WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WS, "src", "hrc_body_tracking"))

import numpy as np  # noqa: E402

TAPE_T = (-0.83, -2.10, 1.66)          # tape-measured camera position (world, floor z = 0)
OUT_ROOT = os.path.join(WS, "debug_log", "camera_pose_fit")
FOOT_L, FOOT_R = "left_foot", "right_foot"


# ---------------------------------------------------------------- geometry
def cam_A(roll, pitch, yaw):
    """A = cam_R^T, from utils.camera_to_world_rotmat (the code the nodes use)."""
    from hrc_body_tracking.utils import camera_to_world_rotmat
    return camera_to_world_rotmat(roll, pitch, yaw).numpy().astype(np.float64).T


def Rz2(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


_BODY_CACHE = {}


def body_constants(betas):
    """J0 and foot heights for these betas (SMPL-H male, zero pose, like mass_inertia)."""
    key = tuple(np.round(np.asarray(betas, np.float64), 6))
    if key in _BODY_CACHE:
        return _BODY_CACHE[key]
    import torch
    from hrc_body_tracking.physcis import SMPLH, Physics
    from hrc_body_tracking.utils import batch_roteulerSMPL
    dev = torch.device("cpu")
    if "m" not in _BODY_CACHE:
        _BODY_CACHE["m"] = (SMPLH(gender="male", device=dev), Physics(device=dev))
    smplh, phys = _BODY_CACHE["m"]
    with torch.no_grad():
        b = torch.tensor(np.asarray(betas, np.float32)[None, :10])
        rot0, _ = batch_roteulerSMPL(torch.zeros(1, 72))
        o = smplh.forward(betas=b, rotmat=rot0)
        m24, v24, i24, c24 = phys.compute_massinertia(o.vertices, o.joints_smpl)
        _, _, _, c19 = phys.merge_parts_24_to_19_vertex_avg(m24, v24, i24, c24)
    v = o.vertices[0].numpy().astype(np.float64)
    J = o.joint_smpl_shaped[0].numpy().astype(np.float64)
    c19 = c19[0].numpy().astype(np.float64)
    sole = v[:, 1].min()                                  # SMPL body frame: +y up
    out = dict(J0=J[0], sole_y=sole,
               foot_mid_h=0.5 * (c19[6, 1] + c19[7, 1]) - sole,
               foot_h_L=c19[6, 1] - sole, foot_h_R=c19[7, 1] - sole,
               ankle_mid_h=0.5 * (J[7, 1] + J[8, 1]) - sole,
               stature=v[:, 1].max() - sole)
    _BODY_CACHE[key] = out
    return out


# ---------------------------------------------------------------- fit
def fit(records, roll=None, pitch=None, tape=TAPE_T, legacy=False):
    """records: dicts with mark_xy, foot_mid_mean, pose{roll,pitch,yaw,t}, betas."""
    roll = records[-1]["pose"]["roll_deg"] if roll is None else roll
    pitch = records[-1]["pose"]["pitch_deg"] if pitch is None else pitch
    A00 = cam_A(roll, pitch, 0.0)                        # Ry^T Rx^T  (= A at yaw 0)
    V, Q, Z, H, J0s = [], [], [], [], []
    n_old = 0
    for r in records:
        p = r["pose"]
        A0 = cam_A(p["roll_deg"], p["pitch_deg"], p["yaw_deg"])
        bc = body_constants(r["betas"])
        old = not r.get("hd_pelvis_corrected", not legacy)
        J0 = bc["J0"] if old else np.zeros(3)            # J0 only enters the pre-fix model
        n_old += int(old)
        xc_minus_J0 = A0.T @ (np.asarray(r["foot_mid_mean"]) - J0 - np.asarray(p["t"]))
        v = A00 @ xc_minus_J0                            # x_hd = Rz(-yaw) v + J0 + t
        V.append(v[:2]); Z.append(v[2]); J0s.append(J0)
        Q.append(np.asarray(r["mark_xy"], float) - J0[:2]); H.append(bc["foot_mid_h"])
    V, Q, Z, H, J0s = map(np.asarray, (V, Q, Z, H, J0s))
    vc, qc = V - V.mean(0), Q - Q.mean(0)
    th = math.atan2((vc[:, 0] * qc[:, 1] - vc[:, 1] * qc[:, 0]).sum(), (vc * qc).sum())
    yaw = -math.degrees(th)
    txy = Q.mean(0) - Rz2(th) @ V.mean(0)
    tz = float(np.mean(H - Z - J0s[:, 2]))
    t = np.array([txy[0], txy[1], tz])
    pred_xy = (Rz2(th) @ V.T).T + txy + J0s[:, :2]
    res = pred_xy - np.array([r["mark_xy"] for r in records], float)
    zw = Z + J0s[:, 2] + tz                              # foot-point height in world (fitted pose)
    A = cam_A(roll, pitch, yaw)
    J0m = np.mean([body_constants(r["betas"])["J0"] for r in records], axis=0)
    bias = (np.eye(3) - A) @ J0m
    tape = np.asarray(tape, float)
    expected = tape - bias if n_old == len(records) else tape.copy()   # pre-fix: t absorbs -(I - A) J0
    cam_xy = tape[:2]
    rows = []
    for r, e, z, h in zip(records, res, zw, H):
        rows.append(dict(name=r["name"], lap=r.get("lap"), mark_xy=r["mark_xy"], res_xy_cm=(100 * e).round(2).tolist(),
                         res_cm=round(100 * float(np.hypot(*e)), 2), height_world_cm=round(100 * float(z), 2),
                         height_model_cm=round(100 * float(h), 2),
                         dist_from_tape_cam_m=round(float(np.hypot(*(np.asarray(r["mark_xy"]) - cam_xy))), 3),
                         std_cm=(100 * np.asarray(r.get("foot_mid_std", [np.nan] * 3))).round(2).tolist(),
                         n=r.get("n")))
    return dict(roll_deg=roll, pitch_deg=pitch, yaw_deg=yaw, t=t.tolist(),
                rms_cm=float(100 * np.sqrt((res ** 2).sum(1).mean())), rows=rows,
                tape_t=tape.tolist(), t_minus_tape_cm=(100 * (t - tape)).round(2).tolist(),
                pre_fix_records=n_old, J0_mean=J0m.tolist(), j0_offset_I_minus_A_J0_cm=(100 * bias).round(2).tolist(),
                expected_t=expected.round(4).tolist(), t_minus_expected_cm=(100 * (t - expected)).round(2).tolist())


def print_fit(f):
    print(f"roll {f['roll_deg']:.4f}  pitch {f['pitch_deg']:.4f}  (fixed)")
    print(f"fitted yaw_deg {f['yaw_deg']:.3f}   t [{f['t'][0]:.4f}, {f['t'][1]:.4f}, {f['t'][2]:.4f}]")
    print(f"residual RMS {f['rms_cm']:.2f} cm  over {len(f['rows'])} records")
    print(f"{'mark':5s} {'lap':>3s} {'res dx,dy cm':>16s} {'|res| cm':>8s} {'z world cm':>10s} {'z model':>7s} "
          f"{'dist m':>6s} {'std x,y,z cm':>18s} {'n':>4s}")
    for r in f["rows"]:
        print(f"{r['name']:5s} {str(r['lap']):>3s} {r['res_xy_cm'][0]:7.2f},{r['res_xy_cm'][1]:7.2f} {r['res_cm']:8.2f} "
              f"{r['height_world_cm']:10.2f} {r['height_model_cm']:7.2f} {r['dist_from_tape_cam_m']:6.2f} "
              f"{r['std_cm'][0]:5.2f},{r['std_cm'][1]:5.2f},{r['std_cm'][2]:5.2f} {str(r['n']):>4s}")
    by = {}
    for r in f["rows"]:
        by.setdefault(r["name"], []).append(r["height_world_cm"])
    print("mean foot-point height by mark (cm): " + ", ".join(f"{k} {np.mean(v):.2f}" for k, v in by.items()))
    print(f"t - tape {f['t_minus_tape_cm']} cm (tape {f['tape_t']})")
    kind = "pre-fix node: tape - (I - A) J0" if f["pre_fix_records"] == len(f["rows"]) else "corrected node: tape"
    if 0 < f["pre_fix_records"] < len(f["rows"]):
        kind += f" (mixed: {f['pre_fix_records']} pre-fix records)"
    print(f"(I - A) J0 = {f['j0_offset_I_minus_A_J0_cm']} cm; expected t ({kind}) = {f['expected_t']}, "
          f"t - expected = {f['t_minus_expected_cm']} cm")


# ---------------------------------------------------------------- record (ROS 2)
SOUNDS = {"tick": "/usr/share/sounds/freedesktop/stereo/message.oga",
          "start": "/usr/share/sounds/freedesktop/stereo/bell.oga",
          "end": "/usr/share/sounds/freedesktop/stereo/complete.oga"}


def cue(kind):
    """Best-effort sound so the person on the mark knows when recording starts/ends."""
    import subprocess
    msg = {"start": "  >>> recording, hold still", "end": "  <<< done, you may move"}.get(kind)
    if msg:
        print(msg, flush=True)
    f = SOUNDS.get(kind)
    if f and os.path.exists(f):
        try:
            subprocess.Popen(["paplay", f], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            sys.stdout.write("\a"); sys.stdout.flush()


def countdown(delay):
    """Time to walk from the keyboard to the mark; a tick per second, then the start sound."""
    for k in range(int(round(delay)), 0, -1):
        print(f"  start in {k} s", flush=True)
        cue("tick")
        time.sleep(1.0)
    cue("start")


def record(args):
    import threading
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy
    from std_msgs.msg import String
    from hrc_interfaces.msg import HumanDynamicsState, HumanPose
    from hrc_body_tracking.camera_pose import CAMERA_POSE_TOPIC, camera_pose_qos, parse_pose_message

    be = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                    durability=DurabilityPolicy.VOLATILE)

    class Rec(Node):
        def __init__(self):
            super().__init__("camera_pose_fit")
            self.lock = threading.Lock()
            self.pose, self.betas, self.buf, self.on, self.names = None, None, [], False, []
            self.create_subscription(String, CAMERA_POSE_TOPIC, self.cb_pose, camera_pose_qos())
            self.create_subscription(HumanDynamicsState, "/hrc/human_dynamics_state", self.cb_dyn, be)
            self.create_subscription(HumanPose, "/hrc/human_pose", self.cb_hp, be)

        def cb_pose(self, msg):
            r, p, y, t, src, _ = parse_pose_message(msg.data)
            with self.lock:
                self.pose = dict(roll_deg=r, pitch_deg=p, yaw_deg=y, t=t, source=src)

        def cb_hp(self, msg):
            if len(msg.smpl_betas) >= 10:
                with self.lock:
                    self.betas = list(msg.smpl_betas[:10])

        def cb_dyn(self, msg):
            if not self.on:
                return
            names = list(msg.body_part_names)
            P = np.asarray(msg.body_part_positions, np.float64).reshape(-1, 3)
            iL, iR = names.index(FOOT_L), names.index(FOOT_R)
            st = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
            with self.lock:
                self.names = names
                self.buf.append((st, P[iL], P[iR], None if self.pose is None else dict(self.pose), P))

    try:   # no rclpy signal handlers: Ctrl+C -> KeyboardInterrupt, SIGTERM -> exit, even while waiting in input()
        from rclpy.signals import SignalHandlerOptions
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    except ImportError:
        rclpy.init()
    node = Rec()
    from rclpy.executors import SingleThreadedExecutor
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    th = threading.Thread(target=ex.spin, daemon=True)
    th.start()

    def stop():
        ex.shutdown()
        th.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown()
    node.stop = stop
    session = args.session or time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(args.out_root, session)
    os.makedirs(out, exist_ok=True)
    print(f"records -> {out}")
    t0 = time.time()
    while node.pose is None or node.betas is None:
        if time.time() - t0 > args.wait:
            what = "camera pose" if node.pose is None else "betas (/hrc/human_pose)"
            node.stop()
            sys.exit(f"no {what} within {args.wait} s")
        time.sleep(0.1)
    print(f"camera pose: {node.pose}")
    if args.cmd == "touch":
        return touch_loop(node, args, out)
    marks = [(m[0], float(m[1]), float(m[2])) for m in args.mark]
    try:
        for lap in range(1, args.laps + 1):
            for name, x, y in marks:
                while True:
                    k = input(f"[lap {lap}] stand on {name} ({x:+.2f}, {y:+.2f}), feet together, face the camera. "
                              f"Enter = record {args.seconds:g} s, s = skip, q = quit: ").strip().lower()
                    if k == "q":
                        return
                    if k == "s":
                        break
                    countdown(args.delay)
                    with node.lock:
                        node.buf = []
                    node.on = True
                    time.sleep(args.seconds)
                    node.on = False
                    cue("end")
                    with node.lock:
                        buf, betas = list(node.buf), list(node.betas)
                    poses = {json.dumps(b[3], sort_keys=True) for b in buf}
                    if len(buf) < args.min_samples:
                        print(f"  only {len(buf)} samples (< {args.min_samples}); repeat")
                        continue
                    if len(poses) != 1:
                        print("  camera pose changed during the record; repeat")
                        continue
                    L = np.array([b[1] for b in buf]); R = np.array([b[2] for b in buf]); M = 0.5 * (L + R)
                    rec = dict(name=name, lap=lap, mark_xy=[x, y], time=time.strftime("%Y-%m-%dT%H:%M:%S"),
                               seconds=args.seconds, n=len(buf), rate_hz=round(len(buf) / args.seconds, 2),
                               point="midpoint of left_foot/right_foot segment COM (/hrc/human_dynamics_state)",
                               foot_mid_mean=M.mean(0).tolist(), foot_mid_std=M.std(0).tolist(),
                               left_mean=L.mean(0).tolist(), right_mean=R.mean(0).tolist(),
                               feet_gap_m=float(np.linalg.norm(L - R, axis=1).mean()),
                               pose=buf[0][3], betas=betas, hd_pelvis_corrected=True)
                    path = os.path.join(out, f"lap{lap}_{name}.json")
                    json.dump(rec, open(path, "w"), indent=1)
                    s = 100 * M.std(0)
                    print(f"  saved {os.path.basename(path)}: n {len(buf)}, mean {np.round(M.mean(0), 4).tolist()}, "
                          f"std {s.round(2).tolist()} cm")
                    break
    finally:
        node.stop()
    print(f"done. fit with: python3 tools/camera_pose_fit.py fit {out}")


def touch_loop(node, args, out):
    """Check: hold a body part on a point of known world position, compare the published COM."""
    x = np.array(args.xyz, float)
    try:
        while True:
            k = input(f"put {args.part} on {args.label} {x.tolist()}. Enter = record {args.seconds:g} s, q = quit: ").strip()
            if k.lower() == "q":
                return
            countdown(args.delay)
            with node.lock:
                node.buf = []
            node.on = True
            time.sleep(args.seconds)
            node.on = False
            cue("end")
            with node.lock:
                buf = list(node.buf)
            if len(buf) < args.min_samples:
                print(f"  only {len(buf)} samples; repeat")
                continue
            i = node.names.index(args.part)
            P = np.array([b[4][i] for b in buf])
            d = P.mean(0) - x
            rec = dict(part=args.part, label=args.label, target=x.tolist(), mean=P.mean(0).tolist(), std=P.std(0).tolist(),
                       n=len(buf), diff=d.tolist(), dist=float(np.linalg.norm(d)), dist_xy=float(np.hypot(d[0], d[1])),
                       pose=buf[0][3], time=time.strftime("%Y-%m-%dT%H:%M:%S"),
                       point=f"{args.part} segment COM (/hrc/human_dynamics_state)")
            path = os.path.join(out, f"touch_{args.label}_{time.strftime('%H%M%S')}.json")
            json.dump(rec, open(path, "w"), indent=1)
            print(f"  {args.part} COM mean {np.round(P.mean(0), 4).tolist()} std {np.round(100 * P.std(0), 2).tolist()} cm; "
                  f"diff {np.round(100 * d, 2).tolist()} cm, |diff| {100 * np.linalg.norm(d):.2f} cm "
                  f"(xy {100 * np.hypot(d[0], d[1]):.2f} cm) -> {os.path.basename(path)}")
    finally:
        node.stop()


def load_records(paths):
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, "lap*.json"))) if os.path.isdir(p) else [p]
    return [json.load(open(f)) for f in files]


# ---------------------------------------------------------------- self test (no ROS, no Kinect)
MARKS = [("M1", -1.40, -0.30), ("M2", -0.80, 0.50), ("M3", -1.80, 0.50), ("M4", -1.40, 1.20)]


def selftest(args):
    rng = np.random.default_rng(args.seed)
    betas = np.load(args.betas_npz)["betas"].mean(0) if args.betas_npz else np.zeros(10)
    bc = body_constants(betas)
    roll, pitch = args.roll, args.pitch

    # 1) convention check: SMPL-H forward in camera vs the human_dynamics world conversion
    import torch
    from hrc_body_tracking.utils import batch_rodrigues
    smplh = _BODY_CACHE["m"][0]
    A = cam_A(roll, pitch, args.yaw)
    t = np.array(args.t, float)
    aa = torch.tensor(rng.normal(0, 0.3, (24, 3)), dtype=torch.float32)
    Rs = batch_rodrigues(aa.view(-1, 1, 3)).view(1, 24, 3, 3)
    Tc = np.array([0.1, 0.7, 2.6])
    Rw = Rs.clone(); Rw[:, 0] = torch.tensor(A, dtype=torch.float32) @ Rs[:, 0]
    b = torch.tensor(np.asarray(betas, np.float32)[None])
    with torch.no_grad():
        xc = smplh.forward(betas=b, rotmat=Rs).joints_smpl[0].numpy().astype(np.float64) + Tc
        xw = smplh.forward(betas=b, rotmat=Rw).joints_smpl[0].numpy().astype(np.float64) + A @ Tc + t
    pred = (A @ xc.T).T + (np.eye(3) - A) @ bc["J0"] + t
    print(f"[convention] max |x_hd - (A x_c + (I-A) J0 + t)| = {np.abs(xw - pred).max() * 1000:.3f} mm over 24 joints; "
          f"without the J0 term: {np.abs(xw - ((A @ xc.T).T + t)).max() * 100:.1f} cm")

    # 2) synthetic records: foot point at (mark, h) + noise, observed under the yaml start pose.
    #    corrected node: x_hd = A x_c + t, t = camera position (args.t).
    #    pre-fix node (also checked): x_hd = A x_c + (I - A) J0 + t_hd, with t_hd = t - (I - A) J0.
    yaw0, t0 = 0.0, np.array(TAPE_T)
    A0 = cam_A(roll, pitch, yaw0)

    def make(corrected, t_true):
        J0 = np.zeros(3) if corrected else bc["J0"]
        out = []
        for lap in (1, 2):
            for name, x, y in MARKS:
                p_true = np.array([x, y, bc["foot_mid_h"]]) + rng.normal(0, args.noise, 3)
                xc_m_J0 = A.T @ (p_true - J0 - t_true)          # invert the true pose
                obs = A0 @ xc_m_J0 + J0 + t0                    # what human_dynamics publishes with the yaml start pose
                out.append(dict(name=name, lap=lap, mark_xy=[x, y], foot_mid_mean=obs.tolist(),
                                foot_mid_std=[args.noise] * 3, n=150, betas=list(betas), hd_pelvis_corrected=corrected,
                                pose=dict(roll_deg=roll, pitch_deg=pitch, yaw_deg=yaw0, t=t0.tolist())))
        return out

    t_old = t - (np.eye(3) - A) @ bc["J0"]
    g_old = fit(make(False, t_old), tape=TAPE_T)
    print(f"[synthetic, pre-fix node] yaw err {g_old['yaw_deg'] - args.yaw:+.3f} deg, "
          f"t err {np.round(100 * (np.array(g_old['t']) - t_old), 2).tolist()} cm, RMS {g_old['rms_cm']:.2f} cm")
    recs = make(True, t)
    f = fit(recs, tape=TAPE_T)
    print(f"[synthetic, corrected node] true yaw {args.yaw:.3f}, t {np.round(t, 4).tolist()}, "
          f"noise {100 * args.noise:g} cm/axis, {len(recs)} records")
    print_fit(f)
    print(f"[synthetic] error: yaw {f['yaw_deg'] - args.yaw:+.3f} deg, t {np.round(100 * (np.array(f['t']) - t), 2).tolist()} cm")
    f3 = fit(recs[:3], tape=TAPE_T)
    print(f"[synthetic, 3 records] yaw err {f3['yaw_deg'] - args.yaw:+.3f} deg, "
          f"t err {np.round(100 * (np.array(f3['t']) - t), 2).tolist()} cm, RMS {f3['rms_cm']:.2f} cm")
    # 3) refit stability: records taken with the fitted pose must give the same answer
    recs2 = []
    for r in recs:
        A1 = cam_A(roll, pitch, f["yaw_deg"])
        xc = A0.T @ (np.asarray(r["foot_mid_mean"]) - t0)
        obs = A1 @ xc + np.array(f["t"])
        recs2.append(dict(r, foot_mid_mean=obs.tolist(), pose=dict(r["pose"], yaw_deg=f["yaw_deg"], t=f["t"])))
    g = fit(recs2, tape=TAPE_T)
    print(f"[refit from fitted pose] yaw diff {g['yaw_deg'] - f['yaw_deg']:+.2e} deg, "
          f"t diff {np.abs(np.array(g['t']) - np.array(f['t'])).max():.2e} m")


def body(args):
    betas = np.load(args.betas_npz)["betas"].mean(0) if args.betas_npz else np.zeros(10)
    bc = body_constants(betas)
    A = cam_A(args.roll, args.pitch, args.yaw)
    print(json.dumps({k: (np.round(v, 4).tolist() if isinstance(v, np.ndarray) else round(float(v), 4))
                      for k, v in bc.items()}))
    print(f"(I - A) J0 at roll {args.roll}, pitch {args.pitch}, yaw {args.yaw}: "
          f"{np.round((np.eye(3) - A) @ bc['J0'], 4).tolist()} m")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("record")
    r.add_argument("--delay", type=float, default=0.0, help="seconds between Enter and the start of recording (walk to the mark)")
    r.add_argument("--mark", nargs=3, action="append", required=True, metavar=("NAME", "X", "Y"))
    r.add_argument("--laps", type=int, default=2)
    r.add_argument("--seconds", type=float, default=5.0)
    r.add_argument("--min-samples", type=int, default=30)
    r.add_argument("--wait", type=float, default=30.0)
    r.add_argument("--session", default=None)
    r.add_argument("--out-root", default=OUT_ROOT)
    tc = sp.add_parser("touch", help="check: compare a body-part COM with a known world point")
    tc.add_argument("--delay", type=float, default=0.0, help="seconds between Enter and the start of recording (walk to the mark)")
    tc.add_argument("--part", default="right_hand")
    tc.add_argument("--xyz", nargs=3, type=float, required=True)
    tc.add_argument("--label", default="point")
    tc.add_argument("--seconds", type=float, default=5.0)
    tc.add_argument("--min-samples", type=int, default=30)
    tc.add_argument("--wait", type=float, default=30.0)
    tc.add_argument("--session", default=None)
    tc.add_argument("--out-root", default=OUT_ROOT)
    f = sp.add_parser("fit")
    f.add_argument("paths", nargs="+")
    f.add_argument("--tape", nargs=3, type=float, default=list(TAPE_T))
    f.add_argument("--min-records", type=int, default=3)
    f.add_argument("--legacy", action="store_true", help="records without hd_pelvis_corrected come from the pre-fix node")
    for name in ("selftest", "body"):
        s = sp.add_parser(name)
        s.add_argument("--betas-npz", default=None)
        s.add_argument("--roll", type=float, default=101.02)
        s.add_argument("--pitch", type=float, default=0.59)
        s.add_argument("--yaw", type=float, default=3.0 if name == "selftest" else 0.0)
        if name == "selftest":
            s.add_argument("--t", nargs=3, type=float, default=[-0.80, -2.13, 1.68], help="true camera position")
            s.add_argument("--noise", type=float, default=0.01)
            s.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd in ("record", "touch"):
        record(a)
    elif a.cmd == "fit":
        recs = load_records(a.paths)
        if len(recs) < a.min_records:
            sys.exit(f"{len(recs)} records; need at least {a.min_records}")
        res = fit(recs, tape=a.tape, legacy=a.legacy)
        print_fit(res)
        d = a.paths[0] if os.path.isdir(a.paths[0]) else os.path.dirname(a.paths[0])
        path = os.path.join(d, f"fit_{time.strftime('%Y%m%d_%H%M%S')}.json")
        json.dump(res, open(path, "w"), indent=1)
        print(f"saved {path}")
    elif a.cmd == "selftest":
        selftest(a)
    else:
        body(a)


if __name__ == "__main__":
    main()
