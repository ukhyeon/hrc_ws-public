import os
import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = "/home/robotics/hrc_ws/analysis/demo_log/demo_metrics_head.csv"
# SAVE_DIR = "/home/robotics/hrc_ws/analysis/demo_log/plots"
SAVE_DIR = "/home/robotics/hrc_ws/analysis/demo_log/plots_frame177_337"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    
def filter_by_frame_range(df: pd.DataFrame, start_frame: int, end_frame: int) -> pd.DataFrame:
    df = df.copy()
    df["frame_idx"] = range(len(df))

    df = df[(df["frame_idx"] >= start_frame) & (df["frame_idx"] <= end_frame)].copy()
    df = df.reset_index(drop=True)
    return df

def filter_by_time_range(df: pd.DataFrame, t_start: float, t_end: float) -> pd.DataFrame:
    df = df[(df["t_demo_sec"] >= t_start) & (df["t_demo_sec"] <= t_end)].copy()
    df = df.reset_index(drop=True)
    return df


def load_demo_metrics(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = [
        "t_demo_sec",
        "selected_human_index",
        "selected_robot_index",
        "selected_link",
        "selected_body",
        "u_x",
        "u_y",
        "u_z",
        "distance",
        "robot_speed_along_u_raw",
        "approach_bias_added",
        "relative_speed_along_u",
        "human_meff_dir",
        "human_meff_iso",
        "robot_meff",
        "reduced_mass_dir",
        "reduced_mass_iso",
        "iso_k",
        "v_allow_dir",
        "v_allow_iso",
        "speed_scale_target",
        "speed_scale_applied",
        "estimated_collision_force_dir",
        "estimated_collision_force_iso",
        "threshold_force",
        "force_ratio_dir",
        "force_ratio_iso",
        "link1_speed",
        "link2_speed",
        "link3_speed",
        "link4_speed",
        "link5_speed",
        "link6_speed",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    df = df.sort_values("t_demo_sec").reset_index(drop=True)
    return df


def plot_speed_scale(df: pd.DataFrame, save_dir: str):
    plt.figure(figsize=(10, 5))
    plt.plot(df["t_demo_sec"], df["speed_scale_target"], label="target speed scale")
    plt.plot(df["t_demo_sec"], df["speed_scale_applied"], label="applied speed scale")
    plt.xlabel("Time [s]")
    plt.ylabel("Speed Scale")
    plt.title("Speed Scale over Time")
    plt.ylim(0.0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "speed_scale_vs_time.png"), dpi=200)
    plt.show()


def plot_force_comparison(df: pd.DataFrame, save_dir: str):
    plt.figure(figsize=(11, 5))

    f_dir = df["estimated_collision_force_dir"]
    f_iso = df["estimated_collision_force_iso"]
    f_th = df["threshold_force"]
    t = df["t_demo_sec"]

    plt.plot(
        t,
        f_dir,
        color="red",
        label="estimated collision force (directional meff)"
    )
    plt.plot(
        t,
        f_iso,
        color="blue",
        label="estimated collision force (ISO fixed mass)"
    )
    plt.plot(
        t,
        f_th,
        color="green",
        label="threshold force"
    )

    # ISO 기준 상대 오차(%)
    eps = 1e-6
    percent_diff = (f_dir - f_iso).abs() / f_iso.clip(lower=eps) * 100.0
    max_idx = percent_diff.idxmax()
    max_diff = percent_diff.loc[max_idx]
    max_t = t.loc[max_idx]

    # 오른쪽 위 텍스트 표시
    info_text = (
        f"Max diff vs ISO: {max_diff:.1f}%\n"
        f"at t = {max_t:.2f} s"
    )

    plt.text(
        0.98, 0.98,
        info_text,
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="black")
    )

    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.title("Directional vs ISO Fixed-Mass Collision Force")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "force_comparison_vs_time.png"), dpi=200)
    plt.show()


def plot_force_ratio_comparison(df: pd.DataFrame, save_dir: str):
    plt.figure(figsize=(11, 5))
    plt.plot(df["t_demo_sec"], df["force_ratio_dir"], label="force ratio (directional)")
    plt.plot(df["t_demo_sec"], df["force_ratio_iso"], label="force ratio (ISO fixed)")
    plt.axhline(1.0, linestyle="--", label="threshold ratio = 1.0")
    plt.xlabel("Time [s]")
    plt.ylabel("Force Ratio")
    plt.title("Force Ratio Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "force_ratio_comparison_vs_time.png"), dpi=200)
    plt.show()


def plot_selected_link(df: pd.DataFrame, save_dir: str):
    link_order = ["link1", "link2", "link3", "link4", "link5", "link6"]
    link_to_num = {name: i + 1 for i, name in enumerate(link_order)}

    y = df["selected_link"].map(link_to_num)

    plt.figure(figsize=(10, 4))
    plt.step(df["t_demo_sec"], y, where="post")
    plt.xlabel("Time [s]")
    plt.ylabel("Selected Link")
    plt.title("Selected Collision Link over Time")
    plt.yticks([1, 2, 3, 4, 5, 6], link_order)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "selected_link_vs_time.png"), dpi=200)
    plt.show()


def plot_selected_body(df: pd.DataFrame, save_dir: str):
    body_order = list(pd.unique(df["selected_body"]))
    body_to_num = {name: i for i, name in enumerate(body_order)}
    y = df["selected_body"].map(body_to_num)

    plt.figure(figsize=(12, 5))
    plt.step(df["t_demo_sec"], y, where="post")
    plt.xlabel("Time [s]")
    plt.ylabel("Selected Body")
    plt.title("Selected Collision Body over Time")
    plt.yticks(range(len(body_order)), body_order)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "selected_body_vs_time.png"), dpi=200)
    plt.show()


def plot_link_speeds(df: pd.DataFrame, save_dir: str):
    plt.figure(figsize=(12, 6))
    for i in range(1, 7):
        col = f"link{i}_speed"
        plt.plot(df["t_demo_sec"], df[col], label=col)

    plt.xlabel("Time [s]")
    plt.ylabel("CoM Speed Magnitude [m/s]")
    plt.title("Link CoM Speed Magnitude over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "link_speeds_vs_time.png"), dpi=200)
    plt.show()


def plot_relative_speed_components(df: pd.DataFrame, save_dir: str):
    plt.figure(figsize=(11, 5))
    plt.plot(df["t_demo_sec"], df["robot_speed_along_u_raw"], label="robot_speed_along_u_raw")
    plt.plot(df["t_demo_sec"], df["approach_bias_added"], label="approach_bias_added")
    plt.plot(df["t_demo_sec"], df["relative_speed_along_u"], label="relative_speed_along_u")
    plt.xlabel("Time [s]")
    plt.ylabel("Speed [m/s]")
    plt.title("Relative Speed Components over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "relative_speed_components_vs_time.png"), dpi=200)
    plt.show()


def plot_u_components(df: pd.DataFrame, save_dir: str):
    plt.figure(figsize=(11, 5))
    plt.plot(df["t_demo_sec"], df["u_x"], label="u_x")
    plt.plot(df["t_demo_sec"], df["u_y"], label="u_y")
    plt.plot(df["t_demo_sec"], df["u_z"], label="u_z")
    plt.xlabel("Time [s]")
    plt.ylabel("Direction Component")
    plt.title("Collision Direction u over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "u_components_vs_time.png"), dpi=200)
    plt.show()


def plot_meff_comparison(df: pd.DataFrame, save_dir: str):
    plt.figure(figsize=(11, 5))
    plt.plot(df["t_demo_sec"], df["human_meff_dir"], label="human_meff_dir")
    plt.plot(df["t_demo_sec"], df["human_meff_iso"], label="human_meff_iso")
    plt.plot(df["t_demo_sec"], df["robot_meff"], label="robot_meff")
    plt.xlabel("Time [s]")
    plt.ylabel("Mass [kg]")
    plt.title("Directional Human Meff vs ISO Fixed Mass vs Robot Meff")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "meff_comparison_vs_time.png"), dpi=200)
    plt.show()


def plot_reduced_mass_comparison(df: pd.DataFrame, save_dir: str):
    plt.figure(figsize=(11, 5))
    plt.plot(df["t_demo_sec"], df["reduced_mass_dir"], label="reduced_mass_dir")
    plt.plot(df["t_demo_sec"], df["reduced_mass_iso"], label="reduced_mass_iso")
    plt.xlabel("Time [s]")
    plt.ylabel("Reduced Mass [kg]")
    plt.title("Reduced Mass Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "reduced_mass_comparison_vs_time.png"), dpi=200)
    plt.show()


def plot_v_allow_comparison(df: pd.DataFrame, save_dir: str):
    plt.figure(figsize=(11, 5))
    plt.plot(df["t_demo_sec"], df["v_allow_dir"], label="v_allow_dir")
    plt.plot(df["t_demo_sec"], df["v_allow_iso"], label="v_allow_iso")
    plt.plot(df["t_demo_sec"], df["relative_speed_along_u"], label="relative_speed_along_u")
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [m/s]")
    plt.title("Allowed Velocity vs Relative Speed")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "v_allow_comparison_vs_time.png"), dpi=200)
    plt.show()


def print_spike_summary(df: pd.DataFrame):
    spike_dir = df[df["force_ratio_dir"] > 1.0]
    spike_iso = df[df["force_ratio_iso"] > 1.0]

    print("\n=== Spike Summary ===")
    print(f"Directional force ratio > 1.0 count : {len(spike_dir)}")
    print(f"ISO fixed force ratio > 1.0 count  : {len(spike_iso)}")

    if len(spike_dir) > 0:
        print("\n[Directional spike rows - top 10]")
        cols = [
            "t_demo_sec",
            "selected_link",
            "selected_body",
            "robot_speed_along_u_raw",
            "approach_bias_added",
            "relative_speed_along_u",
            "human_meff_dir",
            "human_meff_iso",
            "robot_meff",
            "estimated_collision_force_dir",
            "estimated_collision_force_iso",
            "threshold_force",
            "force_ratio_dir",
            "force_ratio_iso",
        ]
        print(spike_dir[cols].head(10))

    link_change_count = (df["selected_link"] != df["selected_link"].shift(1)).sum()
    body_change_count = (df["selected_body"] != df["selected_body"].shift(1)).sum()

    print("\n=== Candidate Switching Summary ===")
    print(f"selected_link changes : {int(link_change_count)}")
    print(f"selected_body changes : {int(body_change_count)}")


def main():
    ensure_dir(SAVE_DIR)
    df = load_demo_metrics(CSV_PATH)
    df = filter_by_frame_range(df, start_frame=80, end_frame=400)

    print("Loaded rows:", len(df))
    print(df.head())

    print_spike_summary(df)

    plot_speed_scale(df, SAVE_DIR)
    plot_force_comparison(df, SAVE_DIR)
    plot_force_ratio_comparison(df, SAVE_DIR)
    plot_selected_link(df, SAVE_DIR)
    plot_selected_body(df, SAVE_DIR)
    plot_link_speeds(df, SAVE_DIR)
    plot_relative_speed_components(df, SAVE_DIR)
    plot_u_components(df, SAVE_DIR)
    plot_meff_comparison(df, SAVE_DIR)
    plot_reduced_mass_comparison(df, SAVE_DIR)
    plot_v_allow_comparison(df, SAVE_DIR)

    print(f"\nSaved plots to: {SAVE_DIR}")


if __name__ == "__main__":
    main()