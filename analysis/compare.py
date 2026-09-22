import os
import glob
import numpy as np


OFFLINE_PATH = "/home/robotics/hrc_ws/debug_meff_compare/offline_physics_compare.npz"
RT_GLOB = "/home/robotics/hrc_ws/debug_meff_compare/realtime_physics_compare_q*.npz"


def norm(x):
    return float(np.linalg.norm(x))


def max_abs(x):
    return float(np.max(np.abs(x)))


def mean_abs(x):
    return float(np.mean(np.abs(x)))


def eig_stats(M):
    M_sym = 0.5 * (M + M.T)
    eig = np.linalg.eigvalsh(M_sym)

    return {
        "eig_min": float(eig[0]),
        "eig_max": float(eig[-1]),
        "cond": float(eig[-1] / (abs(eig[0]) + 1e-12)),
    }


def compare_array(name, off_arr, rt_arr, k, rt_idx=0):
    a = off_arr[k]
    b = rt_arr[rt_idx]

    d = a - b

    print(f"\n[{name}]")
    print("shape off/rt :", a.shape, b.shape)
    print("off norm     :", norm(a))
    print("rt  norm     :", norm(b))
    print("diff norm    :", norm(d))
    print("max abs diff :", max_abs(d))
    print("mean abs diff:", mean_abs(d))
    print("argmax       :", np.unravel_index(np.argmax(np.abs(d)), d.shape))


def print_meff_compare(off, rt, k, rt_idx=0):
    print("\n================ MEFF COMPARE ================")

    off_plus = float(off["meff_plus"][k])
    off_minus = float(off["meff_minus"][k])
    rt_plus = float(rt["meff_plus"][rt_idx])
    rt_minus = float(rt["meff_minus"][rt_idx])

    print(f"offline frame : {k}")
    print(f"rt frame      : {rt_idx}")

    print("\n[plus]")
    print(f"offline meff  : {off_plus:.6f}")
    print(f"realtime meff : {rt_plus:.6f}")
    print(f"abs diff      : {abs(off_plus - rt_plus):.6f}")
    print(f"rel diff      : {abs(off_plus - rt_plus) / (abs(off_plus) + 1e-12) * 100:.3f}%")

    print("\n[minus]")
    print(f"offline meff  : {off_minus:.6f}")
    print(f"realtime meff : {rt_minus:.6f}")
    print(f"abs diff      : {abs(off_minus - rt_minus):.6f}")
    print(f"rel diff      : {abs(off_minus - rt_minus) / (abs(off_minus) + 1e-12) * 100:.3f}%")

    print("\n[denom]")
    print(f"offline plus denom  : {float(off['denom_plus'][k]):.8e}")
    print(f"realtime plus denom : {float(rt['denom_plus'][rt_idx]):.8e}")
    print(f"offline minus denom : {float(off['denom_minus'][k]):.8e}")
    print(f"realtime minus denom: {float(rt['denom_minus'][rt_idx]):.8e}")

    print("\n[ROM lock]")
    print("offline plus locked cols :", np.where(off["rom_mask_plus"][k] > 0)[0].tolist())
    print("realtime plus locked cols:", np.where(rt["rom_mask_plus"][rt_idx] > 0)[0].tolist())
    print("offline minus locked cols :", np.where(off["rom_mask_minus"][k] > 0)[0].tolist())
    print("realtime minus locked cols:", np.where(rt["rom_mask_minus"][rt_idx] > 0)[0].tolist())

    print("\n[qddot col17 / col22]")
    print(f"off plus  col17/22: {off['qddot_plus_col17'][k]: .6f}, {off['qddot_plus_col22'][k]: .6f}")
    print(f"rt  plus  col17/22: {rt['qddot_plus_col17'][rt_idx]: .6f}, {rt['qddot_plus_col22'][rt_idx]: .6f}")
    print(f"off minus col17/22: {off['qddot_minus_col17'][k]: .6f}, {off['qddot_minus_col22'][k]: .6f}")
    print(f"rt  minus col17/22: {rt['qddot_minus_col17'][rt_idx]: .6f}, {rt['qddot_minus_col22'][rt_idx]: .6f}")


def print_mq_eig_compare(off, rt, k, rt_idx=0):
    print("\n================ MQ EIG COMPARE ================")

    off_stats = eig_stats(off["Mq_19"][k])
    rt_stats = eig_stats(rt["Mq_19"][rt_idx])

    print("[offline]")
    print(off_stats)

    print("[realtime]")
    print(rt_stats)

    print("\n[eig diff]")
    print("eig_min diff:", off_stats["eig_min"] - rt_stats["eig_min"])
    print("eig_max diff:", off_stats["eig_max"] - rt_stats["eig_max"])
    print("cond diff   :", off_stats["cond"] - rt_stats["cond"])


def print_target_contact_compare(off, rt, k, rt_idx=0, target_idx=1, contacts=(6, 7)):
    print("\n================ TARGET / CONTACT JACOBIAN ================")

    for body_idx in [target_idx, *contacts]:
        Jp_off = off["Jp_19"][k, body_idx]
        Jp_rt = rt["Jp_19"][rt_idx, body_idx]
        Jp_diff = Jp_off - Jp_rt

        Jr_off = off["Jr_19"][k, body_idx]
        Jr_rt = rt["Jr_19"][rt_idx, body_idx]
        Jr_diff = Jr_off - Jr_rt

        print(f"\n[body {body_idx}]")
        print(f"Jp off/rt/diff norm: {norm(Jp_off):.8f} / {norm(Jp_rt):.8f} / {norm(Jp_diff):.8e}")
        print(f"Jp max abs diff    : {max_abs(Jp_diff):.8e}")
        print(f"Jr off/rt/diff norm: {norm(Jr_off):.8f} / {norm(Jr_rt):.8f} / {norm(Jr_diff):.8e}")
        print(f"Jr max abs diff    : {max_abs(Jr_diff):.8e}")


def main():
    off = np.load(OFFLINE_PATH, allow_pickle=True)

    rt_paths = sorted(glob.glob(RT_GLOB))
    if len(rt_paths) == 0:
        raise FileNotFoundError(f"No realtime npz found: {RT_GLOB}")

    print("offline:", OFFLINE_PATH)
    print("num realtime files:", len(rt_paths))

    for rt_path in rt_paths:
        rt = np.load(rt_path, allow_pickle=True)

        # realtime은 보통 1 frame
        rt_pose = rt["pose_19_euler_deg"][0]

        # offline에서 pose가 가장 가까운 frame 찾기
        pose_diff = np.linalg.norm(
            off["pose_19_euler_deg"] - rt_pose[None, :, :],
            axis=(1, 2)
        )
        k = int(np.argmin(pose_diff))

        print("\n\n============================================================")
        print("RT file:", os.path.basename(rt_path))
        print("matched offline frame:", k)
        print("pose diff:", pose_diff[k])
        print("============================================================")

        # 1. 핵심 물리량 비교
        keys = [
            "mass_19",
            "inertia_19_body",
            "world_inertia_19",
            "Jp_19",
            "Jr_19",
            "Mq_linear_19",
            "Mq_angular_19",
            "Mq_19",
            "lambda_lin_plus",
            "lambda_lin_minus",
        ]

        for key in keys:
            if key not in off.files or key not in rt.files:
                print(f"\n[{key}] missing")
                continue

            compare_array(key, off[key], rt[key], k, rt_idx=0)

        # 2. target/contact Jacobian
        print_target_contact_compare(off, rt, k, rt_idx=0, target_idx=1, contacts=(6, 7))

        # 3. Mq eigen
        print_mq_eig_compare(off, rt, k, rt_idx=0)

        # 4. Meff / ROM lock / qddot
        print_meff_compare(off, rt, k, rt_idx=0)

        print("\n\n")


if __name__ == "__main__":
    main()