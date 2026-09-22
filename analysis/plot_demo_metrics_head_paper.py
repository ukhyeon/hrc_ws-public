import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Path settings
# ============================================================


CSV_PATH = "/home/robotics/hrc_ws/analysis/demo_log/demo_metrics_head_paper.csv"
SAVE_DIR = "/home/robotics/hrc_ws/analysis/demo_log/plots_head_paper"


# ============================================================
# Time range settings
# ============================================================

PLOT_T_START = 3.0
PLOT_T_END = 17.0


# ============================================================
# Global plot style for paper/demo figures
# ============================================================

FIG_TITLE_SIZE = 22
AXIS_LABEL_SIZE = 20
TICK_LABEL_SIZE = 18
LEGEND_SIZE = 15
ANNOTATION_SIZE = 15
SMALL_ANNOTATION_SIZE = 12
LINE_WIDTH = 2.5
GRID_ALPHA = 0.35

plt.rcParams.update({
    "font.size": TICK_LABEL_SIZE,
    "axes.titlesize": FIG_TITLE_SIZE,
    "axes.labelsize": AXIS_LABEL_SIZE,
    "xtick.labelsize": TICK_LABEL_SIZE,
    "ytick.labelsize": TICK_LABEL_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "figure.titlesize": FIG_TITLE_SIZE,
})


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def style_axes(ax):
    """
    Apply consistent large-font style to one matplotlib axis.
    """
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)

    ax.xaxis.label.set_size(AXIS_LABEL_SIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_SIZE)
    ax.title.set_size(FIG_TITLE_SIZE)

    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(LEGEND_SIZE)


def load_demo_metrics(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Convert numeric columns safely
    numeric_cols = [
        "t_demo_sec",
        "speed_scale_target",
        "speed_scale_applied",
        "estimated_collision_force_dir",
        "estimated_collision_force_iso",
        "estimated_collision_force",
        "threshold_force",
        "force_ratio_dir",
        "force_ratio_iso",
        "human_meff_dir",
        "human_meff_iso",
        "robot_meff",
        "link1_speed",
        "link2_speed",
        "link3_speed",
        "link4_speed",
        "link5_speed",
        "link6_speed",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def filter_time_range(df: pd.DataFrame, t_start: float, t_end: float) -> pd.DataFrame:
    if "t_demo_sec" not in df.columns:
        raise ValueError("CSV does not contain required column: t_demo_sec")

    df_filtered = df[
        (df["t_demo_sec"] >= t_start) &
        (df["t_demo_sec"] <= t_end)
    ].copy()

    if df_filtered.empty:
        raise RuntimeError(
            f"No data in selected time range: {t_start:.1f}–{t_end:.1f} s"
        )

    return df_filtered


def require_columns(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


# ============================================================
# Plot 1: speed scale
# ============================================================

def plot_speed_scale(df: pd.DataFrame, save_dir: str):
    required_cols = [
        "t_demo_sec",
        "speed_scale_target",
        "speed_scale_applied",
    ]
    require_columns(df, required_cols)

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.plot(
        df["t_demo_sec"],
        df["speed_scale_target"],
        linewidth=LINE_WIDTH,
        label="target speed scale"
    )

    ax.plot(
        df["t_demo_sec"],
        df["speed_scale_applied"],
        linewidth=LINE_WIDTH,
        label="applied speed scale"
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Speed Scale")
    ax.set_title("Speed Scale over Time")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(loc="best", framealpha=0.9)

    style_axes(ax)

    plt.tight_layout()

    save_path = os.path.join(save_dir, "speed_scale_vs_time.png")
    plt.savefig(save_path, dpi=200)
    plt.show()

    print(f"[SAVE] {save_path}")

    
# ============================================================
# Print: max directional force exceedance over threshold
# ============================================================

def print_max_directional_force_exceedance_info(df: pd.DataFrame):
    """
    Print the time point where the directional effective-mass model
    exceeds the threshold force by the largest margin.

    exceedance_N =
        estimated_collision_force_dir - threshold_force

    This is intended for checking:
    - maximum exceedance over threshold [N]
    - directional estimated force [N]
    - ISO fixed-mass estimated force [N] at the same time
    - threshold force [N]
    - related effective masses and selected collision candidate
    """

    required_cols = [
        "t_demo_sec",
        "estimated_collision_force_dir",
        "estimated_collision_force_iso",
        "threshold_force",
    ]
    require_columns(df, required_cols)

    temp = df.copy()

    # Force가 의미 있는 구간만 사용
    valid_mask = (
        temp["estimated_collision_force_dir"].notna()
        & temp["estimated_collision_force_iso"].notna()
        & temp["threshold_force"].notna()
        & (temp["threshold_force"] > 1.0)
    )
    temp = temp[valid_mask].copy()

    if temp.empty:
        print("[WARN] No valid force data for threshold exceedance analysis.")
        return

    temp["dir_exceedance_N"] = (
        temp["estimated_collision_force_dir"] - temp["threshold_force"]
    )

    temp["iso_exceedance_N"] = (
        temp["estimated_collision_force_iso"] - temp["threshold_force"]
    )

    temp["dir_force_ratio"] = (
        temp["estimated_collision_force_dir"] / temp["threshold_force"]
    )

    temp["iso_force_ratio"] = (
        temp["estimated_collision_force_iso"] / temp["threshold_force"]
    )

    temp["force_diff_N"] = (
        temp["estimated_collision_force_dir"]
        - temp["estimated_collision_force_iso"]
    )

    temp["force_diff_percent_from_iso"] = (
        temp["force_diff_N"]
        / (temp["estimated_collision_force_iso"].abs() + 1e-8)
        * 100.0
    )

    # 핵심: directional model이 threshold를 가장 크게 초과한 시점
    idx = temp["dir_exceedance_N"].idxmax()
    row = temp.loc[idx]

    print("\n" + "=" * 78)
    print("[MAX DIRECTIONAL FORCE EXCEEDANCE OVER THRESHOLD]")
    print("=" * 78)

    print(f"Time                               : {row['t_demo_sec']:.3f} s")
    print(f"Threshold force                    : {row['threshold_force']:.3f} N")
    print(f"Directional estimated force         : {row['estimated_collision_force_dir']:.3f} N")
    print(f"ISO fixed-mass estimated force      : {row['estimated_collision_force_iso']:.3f} N")
    print("-" * 78)
    print(f"Directional exceedance              : {row['dir_exceedance_N']:.3f} N")
    print(f"ISO exceedance                      : {row['iso_exceedance_N']:.3f} N")
    print(f"Directional force ratio             : {row['dir_force_ratio']:.3f}")
    print(f"ISO force ratio                     : {row['iso_force_ratio']:.3f}")
    print("-" * 78)
    print(f"Force difference Dir - ISO          : {row['force_diff_N']:.3f} N")
    print(f"Relative difference from ISO        : {row['force_diff_percent_from_iso']:.2f} %")

    optional_cols = [
        "human_meff_dir",
        "human_meff_iso",
        "robot_meff",
        "selected_link",
        "selected_body",
        "speed_scale_target",
        "speed_scale_applied",
        "force_ratio_dir",
        "force_ratio_iso",
    ]

    print("-" * 78)
    print("[RELATED VALUES AT MAX DIRECTIONAL EXCEEDANCE]")

    for col in optional_cols:
        if col in row.index:
            value = row[col]
            if isinstance(value, float):
                print(f"{col:35s}: {value:.3f}")
            else:
                print(f"{col:35s}: {value}")

    print("=" * 78 + "\n")


# ============================================================
# Plot 2: force comparison
# ============================================================

def plot_force_comparison(df: pd.DataFrame, save_dir: str):
    """
    Save force_comparison_vs_time.png.

    This plot shows:
    - estimated_collision_force_dir
    - estimated_collision_force_iso
    - threshold_force

    The force difference percentage is not plotted as a dashed line.
    It is summarized in a text box.

    force_diff_percent =
        (F_dir - F_iso) / |F_iso| * 100
    """

    required_cols = [
        "t_demo_sec",
        "estimated_collision_force_dir",
        "estimated_collision_force_iso",
        "threshold_force",
    ]
    require_columns(df, required_cols)

    eps = 1e-8

    F_dir = df["estimated_collision_force_dir"]
    F_iso = df["estimated_collision_force_iso"]

    force_diff_percent = (F_dir - F_iso) / (F_iso.abs() + eps) * 100.0

    mean_abs_diff = force_diff_percent.abs().mean()
    max_abs_diff = force_diff_percent.abs().max()

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.plot(
        df["t_demo_sec"],
        F_dir,
        linewidth=LINE_WIDTH,
        label="Directional effective-mass model"
    )

    ax.plot(
        df["t_demo_sec"],
        F_iso,
        linewidth=LINE_WIDTH,
        label="ISO fixed-mass model"
    )

    # threshold force는 4초 이후부터만 표시
    threshold_df = df[df["t_demo_sec"] >= 4.0].copy()
    ax.plot(
        threshold_df["t_demo_sec"],
        threshold_df["threshold_force"],
        linewidth=LINE_WIDTH,
        label="threshold force"
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Force [N]")
    ax.set_title("Estimated Collision Force: Directional vs ISO Fixed-Mass Model")
    ax.grid(True, alpha=GRID_ALPHA)

    # 필요하면 y축 고정
    # ax.set_ylim(0, 150)

    # Force line legend box: bottom-right
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(0.98, 0.05),
        framealpha=0.9,
        fontsize=LEGEND_SIZE
    )

    # Summary box: upper-right
    summary_text = (
        "Relative force difference from ISO\n"
        f"Mean |ΔF|: {mean_abs_diff:.1f}%\n"
        f"Max |ΔF|: {max_abs_diff:.1f}%"
    )

    ax.text(
        0.98,
        0.95,
        summary_text,
        transform=ax.transAxes,
        fontsize=ANNOTATION_SIZE,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.88
        )
    )

    style_axes(ax)

    plt.tight_layout()

    save_path = os.path.join(save_dir, "force_comparison_vs_time.png")
    plt.savefig(save_path, dpi=200)
    plt.show()

    print(f"[SAVE] {save_path}")

# ============================================================
# Plot 3: force ratio comparison
# ============================================================

def plot_force_ratio_comparison(df: pd.DataFrame, save_dir: str):
    required_cols = [
        "t_demo_sec",
        "force_ratio_dir",
        "force_ratio_iso",
    ]
    require_columns(df, required_cols)

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.plot(
        df["t_demo_sec"],
        df["force_ratio_dir"],
        linewidth=LINE_WIDTH,
        label="force ratio (directional)"
    )

    ax.plot(
        df["t_demo_sec"],
        df["force_ratio_iso"],
        linewidth=LINE_WIDTH,
        label="force ratio (ISO fixed)"
    )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=LINE_WIDTH,
        label="threshold ratio = 1.0"
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Force Ratio")
    ax.set_title("Force Ratio Comparison")
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(loc="best", framealpha=0.9)

    style_axes(ax)

    plt.tight_layout()

    save_path = os.path.join(save_dir, "force_ratio_comparison_vs_time.png")
    plt.savefig(save_path, dpi=200)
    plt.show()

    print(f"[SAVE] {save_path}")


# ============================================================
# Plot 4: effective mass comparison
# ============================================================

def plot_meff_comparison(df: pd.DataFrame, save_dir: str):
    """
    Plot directional human effective mass, ISO fixed human mass, and robot effective mass.

    Values before 4.0 s are hidden, but an invisible anchor line is added
    so that the x-axis autoscale remains identical to other figures.
    """

    required_cols = [
        "t_demo_sec",
        "human_meff_dir",
        "human_meff_iso",
        "robot_meff",
    ]
    require_columns(df, required_cols)

    meff_df = df.copy()

    # 4초 이전 값은 실제 plot에서 보이지 않도록 NaN 처리
    mask_before_4s = meff_df["t_demo_sec"] < 4.0

    meff_df.loc[mask_before_4s, "human_meff_dir"] = float("nan")
    meff_df.loc[mask_before_4s, "human_meff_iso"] = float("nan")
    meff_df.loc[mask_before_4s, "robot_meff"] = float("nan")

    valid_meff = meff_df[
        ["human_meff_dir", "human_meff_iso", "robot_meff"]
    ].dropna(how="all")

    if valid_meff.empty:
        raise RuntimeError("No effective mass data available after 4.0 s.")

    fig, ax = plt.subplots(figsize=(13, 6))

    # 핵심:
    # NaN 처리된 선만 그리면 matplotlib이 4초 이전 x값을 autoscale에서 제외할 수 있음.
    # 따라서 보이지 않는 anchor line을 전체 시간 범위에 추가해서 x축 scale을 다른 Fig와 맞춤.
    anchor_y = valid_meff.min().min()

    ax.plot(
        df["t_demo_sec"],
        [anchor_y] * len(df),
        color="none",
        alpha=0.0,
        linewidth=0.0,
        label="_nolegend_"
    )

    ax.plot(
        meff_df["t_demo_sec"],
        meff_df["human_meff_dir"],
        linewidth=LINE_WIDTH,
        label="Human directional effective mass"
    )

    ax.plot(
        meff_df["t_demo_sec"],
        meff_df["human_meff_iso"],
        linewidth=LINE_WIDTH,
        label="ISO fixed human mass"
    )

    ax.plot(
        meff_df["t_demo_sec"],
        meff_df["robot_meff"],
        linewidth=LINE_WIDTH,
        label="Robot effective mass"
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Effective Mass [kg]")
    ax.set_title("Human and Robot Effective Masses over Time")
    ax.grid(True, alpha=GRID_ALPHA)

    # legend 우측 상단 고정
    ax.legend(
        loc="upper right",
        framealpha=0.9,
        fontsize=LEGEND_SIZE
    )

    style_axes(ax)

    plt.tight_layout()

    save_path = os.path.join(save_dir, "meff_comparison_vs_time.png")
    plt.savefig(save_path, dpi=200)
    plt.show()

    print(f"[SAVE] {save_path}")

# ============================================================
# Plot 5: selected collision link
# ============================================================

def plot_selected_link(df: pd.DataFrame, save_dir: str):
    required_cols = [
        "t_demo_sec",
        "selected_link",
    ]
    require_columns(df, required_cols)

    link_order = ["link1", "link2", "link3", "link4", "link5", "link6"]
    link_to_num = {name: i + 1 for i, name in enumerate(link_order)}

    y = df["selected_link"].map(link_to_num)

    fig, ax = plt.subplots(figsize=(13, 5.5))

    ax.step(
        df["t_demo_sec"],
        y,
        where="post",
        linewidth=LINE_WIDTH
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Selected Link")
    ax.set_title("Selected Collision Link over Time")
    ax.set_yticks([1, 2, 3, 4, 5, 6])
    ax.set_yticklabels(link_order)
    ax.grid(True, alpha=GRID_ALPHA)

    style_axes(ax)

    plt.tight_layout()

    save_path = os.path.join(save_dir, "selected_link_vs_time.png")
    plt.savefig(save_path, dpi=200)
    plt.show()

    print(f"[SAVE] {save_path}")


# ============================================================
# Plot 6: selected body part
# ============================================================

def plot_selected_body(df: pd.DataFrame, save_dir: str):
    required_cols = [
        "t_demo_sec",
        "selected_body",
    ]
    require_columns(df, required_cols)

    body_names = list(dict.fromkeys(df["selected_body"].astype(str).tolist()))
    body_to_num = {name: i + 1 for i, name in enumerate(body_names)}

    y = df["selected_body"].astype(str).map(body_to_num)

    fig, ax = plt.subplots(figsize=(13, 5.5))

    ax.step(
        df["t_demo_sec"],
        y,
        where="post",
        linewidth=LINE_WIDTH
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Selected Body Part")
    ax.set_title("Selected Human Body Part over Time")
    ax.set_yticks(list(body_to_num.values()))
    ax.set_yticklabels(list(body_to_num.keys()))
    ax.grid(True, alpha=GRID_ALPHA)

    style_axes(ax)

    plt.tight_layout()

    save_path = os.path.join(save_dir, "selected_body_vs_time.png")
    plt.savefig(save_path, dpi=200)
    plt.show()

    print(f"[SAVE] {save_path}")


# ============================================================
# Plot 7: robot link speeds
# ============================================================

def plot_link_speeds(df: pd.DataFrame, save_dir: str):
    link_speed_cols = [
        "link1_speed",
        "link2_speed",
        "link3_speed",
        "link4_speed",
        "link5_speed",
        "link6_speed",
    ]

    available_cols = [c for c in link_speed_cols if c in df.columns]

    if not available_cols:
        print("[SKIP] No link speed columns found.")
        return

    required_cols = ["t_demo_sec"] + available_cols
    require_columns(df, required_cols)

    fig, ax = plt.subplots(figsize=(13, 6))

    for col in available_cols:
        ax.plot(
            df["t_demo_sec"],
            df[col],
            linewidth=LINE_WIDTH,
            label=col.replace("_speed", " speed")
        )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Link CoM Speed [m/s]")
    ax.set_title("Robot Link CoM Speeds over Time")
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(loc="best", framealpha=0.9)

    style_axes(ax)

    plt.tight_layout()

    save_path = os.path.join(save_dir, "link_speeds_vs_time.png")
    plt.savefig(save_path, dpi=200)
    plt.show()

    print(f"[SAVE] {save_path}")


# ============================================================
# Plot 8: selected link speed only
# ============================================================

def plot_selected_link_speed(df: pd.DataFrame, save_dir: str):
    required_cols = [
        "t_demo_sec",
        "selected_link",
    ]
    require_columns(df, required_cols)

    link_speed_cols = {
        "link1": "link1_speed",
        "link2": "link2_speed",
        "link3": "link3_speed",
        "link4": "link4_speed",
        "link5": "link5_speed",
        "link6": "link6_speed",
    }

    for link_name, col in link_speed_cols.items():
        if col not in df.columns:
            print(f"[SKIP] Missing column for selected link speed: {col}")
            return

    selected_speed = []

    for _, row in df.iterrows():
        selected_link = str(row["selected_link"])
        speed_col = link_speed_cols.get(selected_link, None)

        if speed_col is None:
            selected_speed.append(float("nan"))
        else:
            selected_speed.append(row[speed_col])

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.plot(
        df["t_demo_sec"],
        selected_speed,
        linewidth=LINE_WIDTH,
        label="selected link CoM speed"
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Speed [m/s]")
    ax.set_title("Selected Collision Link Speed over Time")
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(loc="best", framealpha=0.9)

    style_axes(ax)

    plt.tight_layout()

    save_path = os.path.join(save_dir, "selected_link_speed_vs_time.png")
    plt.savefig(save_path, dpi=200)
    plt.show()

    print(f"[SAVE] {save_path}")


# ============================================================
# Main
# ============================================================

def main():
    ensure_dir(SAVE_DIR)

    df = load_demo_metrics(CSV_PATH)

    # Apply common time range to all plots
    df = filter_time_range(
        df,
        t_start=PLOT_T_START,
        t_end=PLOT_T_END
    )

    print(f"[INFO] Loaded CSV: {CSV_PATH}")
    print(f"[INFO] Plot time range: {PLOT_T_START:.1f}–{PLOT_T_END:.1f} s")
    print(f"[INFO] Number of rows used: {len(df)}")
    print(f"[INFO] Save directory: {SAVE_DIR}")

    # Generate plots
    plot_speed_scale(df, SAVE_DIR)
    plot_force_comparison(df, SAVE_DIR)
    print_max_directional_force_exceedance_info(df)

    # Optional plots depending on available CSV columns
    if "force_ratio_dir" in df.columns and "force_ratio_iso" in df.columns:
        plot_force_ratio_comparison(df, SAVE_DIR)
    else:
        print("[SKIP] force_ratio_comparison_vs_time.png: force_ratio columns not found.")

    if (
        "human_meff_dir" in df.columns and
        "human_meff_iso" in df.columns and
        "robot_meff" in df.columns
    ):
        plot_meff_comparison(df, SAVE_DIR)
    else:
        print("[SKIP] effective_mass_comparison_vs_time.png: meff columns not found.")

    if "selected_link" in df.columns:
        plot_selected_link(df, SAVE_DIR)
    else:
        print("[SKIP] selected_link_vs_time.png: selected_link column not found.")

    if "selected_body" in df.columns:
        plot_selected_body(df, SAVE_DIR)
    else:
        print("[SKIP] selected_body_vs_time.png: selected_body column not found.")

    plot_link_speeds(df, SAVE_DIR)

    if "selected_link" in df.columns:
        plot_selected_link_speed(df, SAVE_DIR)
    else:
        print("[SKIP] selected_link_speed_vs_time.png: selected_link column not found.")


if __name__ == "__main__":
    main()