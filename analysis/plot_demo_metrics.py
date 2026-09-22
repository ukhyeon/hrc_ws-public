import os
import math
import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = "/home/robotics/hrc_ws/analysis/demo_log/demo_metrics.csv"
SAVE_DIR = "/home/robotics/hrc_ws/analysis/demo_log/plots"
FORCE_GUARD_KAPPA = 8.0

PLOT_T_START = 3.0
PLOT_T_END = 17.0
ONE_CYCLE_T_START = 9.25
ONE_CYCLE_T_END = 15.50

FIG_TITLE_SIZE = 22
AXIS_LABEL_SIZE = 20
TICK_LABEL_SIZE = 18
LEGEND_SIZE = 15
ANNOTATION_SIZE = 15
SMALL_ANNOTATION_SIZE = 12
LINE_WIDTH = 2.5
GRID_ALPHA = 0.35
SAVE_DPI = 600
LABEL_PAD = 10
TITLE_PAD = 16
TICK_PAD = 5
LAYOUT_PAD = 0.8

plt.rcParams.update({
    "font.size": TICK_LABEL_SIZE,
    "axes.titlesize": FIG_TITLE_SIZE,
    "axes.labelsize": AXIS_LABEL_SIZE,
    "xtick.labelsize": TICK_LABEL_SIZE,
    "ytick.labelsize": TICK_LABEL_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "figure.titlesize": FIG_TITLE_SIZE,
    "axes.labelpad": LABEL_PAD,
    "axes.titlepad": TITLE_PAD,
})


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def style_axes(ax):
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE, pad=TICK_PAD)
    ax.xaxis.label.set_size(AXIS_LABEL_SIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_SIZE)
    ax.xaxis.labelpad = LABEL_PAD
    ax.yaxis.labelpad = LABEL_PAD
    ax.title.set_size(FIG_TITLE_SIZE)

    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(LEGEND_SIZE)


def require_columns(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def filter_time_range(df: pd.DataFrame, t_start: float, t_end: float) -> pd.DataFrame:
    require_columns(df, ["t_demo_sec"])

    filtered = df[
        (df["t_demo_sec"] >= t_start) &
        (df["t_demo_sec"] <= t_end)
    ].copy()

    if filtered.empty:
        raise RuntimeError(
            f"No data in selected time range: {t_start:.1f}-{t_end:.1f} s"
        )

    return filtered


def save_figure(fig, save_path: str, tight: bool = True):
    if tight:
        fig.tight_layout(pad=LAYOUT_PAD)
    fig.savefig(save_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {save_path}")


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
        "estimated_collision_force_dir",
        "estimated_collision_force_iso",
        "threshold_force",
        "force_ratio_dir",
        "force_ratio_iso",
        "speed_scale_applied",
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

    numeric_cols = [
        "t_demo_sec",
        "selected_human_index",
        "selected_robot_index",
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
        "estimated_collision_force_dir",
        "estimated_collision_force_iso",
        "threshold_force",
        "force_ratio_dir",
        "force_ratio_iso",
        "speed_scale_applied",
        "link1_speed",
        "link2_speed",
        "link3_speed",
        "link4_speed",
        "link5_speed",
        "link6_speed",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("t_demo_sec").reset_index(drop=True)
    if "command_target_speed_scale" not in df.columns:
        force_ratio = df["force_ratio_dir"].clip(lower=0.0)
        guard_cap = pd.Series(1.0, index=df.index)
        over_limit = force_ratio > 1.0
        guard_cap.loc[over_limit] = (
            (-FORCE_GUARD_KAPPA * (force_ratio.loc[over_limit] - 1.0))
            .apply(lambda x: min(1.0, max(0.0, math.exp(x))))
        )
        df["command_target_speed_scale"] = df[["speed_scale_target"]].join(
            guard_cap.rename("guard_cap")
        ).min(axis=1)
    return df


def plot_speed_scale(df: pd.DataFrame, save_dir: str):
    require_columns(
        df,
        ["t_demo_sec", "command_target_speed_scale", "speed_scale_applied"],
    )

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(
        df["t_demo_sec"],
        df["command_target_speed_scale"],
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
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        framealpha=0.95,
    )
    style_axes(ax)

    save_figure(fig, os.path.join(save_dir, "speed_scale_vs_time.png"))


def plot_force_comparison(df: pd.DataFrame, save_dir: str):
    """
    Plot directional-meff-based force, ISO-fixed-mass force,
    and threshold force.
    """

    require_columns(
        df,
        [
            "t_demo_sec",
            "relative_speed_along_u",
            "estimated_collision_force_dir",
            "estimated_collision_force_iso",
            "threshold_force",
        ],
    )

    F_dir = df["estimated_collision_force_dir"]
    F_iso = df["estimated_collision_force_iso"]
    eps = 1e-8
    approach = (
        (df["relative_speed_along_u"] > 0.0) &
        (F_iso.abs() > eps)
    )
    force_diff_percent = (F_dir.loc[approach] - F_iso.loc[approach]) / (
        F_iso.loc[approach].abs()
    ) * 100.0
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

    threshold_df = df[df["t_demo_sec"] >= 4.0].copy()
    ax.plot(
        threshold_df["t_demo_sec"],
        threshold_df["threshold_force"],
        linewidth=LINE_WIDTH,
        label="permissible force limit"
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Force [N]")
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        framealpha=0.9,
        fontsize=LEGEND_SIZE,
    )
    summary_text = (
        r"$\Delta F_{\mathrm{rel}}$"
        "\n"
        f"Mean: {mean_abs_diff:.2f}%\n"
        f"Max: {max_abs_diff:.2f}%"
    )
    ax.text(
        0.84,
        0.95,
        summary_text,
        transform=ax.transAxes,
        fontsize=ANNOTATION_SIZE,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.88),
        zorder=30,
    )
    style_axes(ax)

    save_figure(fig, os.path.join(save_dir, "force_comparison_vs_time.png"))


def plot_effective_mass_comparison(df: pd.DataFrame, save_dir: str):
    """
    Save effective_mass_comparison_vs_time.png.

    This plot shows:
    1) human_meff_dir
    2) human_meff_iso
    3) robot_meff
    """

    required_cols = [
        "t_demo_sec",
        "human_meff_dir",
        "human_meff_iso",
        "robot_meff",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for effective mass plot: {missing}")

    t = df["t_demo_sec"]
    human_meff_dir = df["human_meff_dir"]
    human_meff_iso = df["human_meff_iso"]
    robot_meff = df["robot_meff"]

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.plot(
        t,
        human_meff_dir,
        linewidth=LINE_WIDTH,
        label="Human directional effective mass"
    )

    ax.plot(
        t,
        human_meff_iso,
        linewidth=LINE_WIDTH,
        linestyle="--",
        label="ISO fixed human mass"
    )

    ax.plot(
        t,
        robot_meff,
        linewidth=LINE_WIDTH,
        label="Robot effective mass"
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Effective Mass [kg]")
    ax.set_title("Human and Robot Effective Masses over Time")
    ax.grid(True, alpha=GRID_ALPHA)

    human_dir_mean = human_meff_dir.mean()
    human_dir_min = human_meff_dir.min()
    human_dir_max = human_meff_dir.max()

    robot_mean = robot_meff.mean()
    robot_min = robot_meff.min()
    robot_max = robot_meff.max()

    summary_text = (
        f"Human dir.: mean={human_dir_mean:.2f} kg, "
        f"min={human_dir_min:.2f}, max={human_dir_max:.2f}\n"
        f"Robot: mean={robot_mean:.2f} kg, "
        f"min={robot_min:.2f}, max={robot_max:.2f}"
    )
    ax.text(
        0.02,
        0.95,
        summary_text,
        transform=ax.transAxes,
        fontsize=SMALL_ANNOTATION_SIZE,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.88),
    )

    # selected link/body 변경 지점 표시
    if "selected_link" in df.columns and "selected_body" in df.columns:
        prev_label = None
        y_top = max(human_meff_dir.max(), human_meff_iso.max(), robot_meff.max())
        y_text = y_top * 1.03 if y_top > 0 else 1.0

        for i in range(len(df)):
            link = str(df["selected_link"].iloc[i])
            body = str(df["selected_body"].iloc[i])
            label = f"{link}-{body}"

            if label != prev_label:
                x = df["t_demo_sec"].iloc[i]
                ax.axvline(x=x, linestyle=":", linewidth=1.0, alpha=0.5)
                ax.text(
                    x,
                    y_text,
                    label,
                    fontsize=SMALL_ANNOTATION_SIZE,
                    rotation=90,
                    verticalalignment="bottom"
                )
                prev_label = label

    ax.legend(loc="upper right", framealpha=0.9, fontsize=LEGEND_SIZE)
    style_axes(ax)
    save_figure(fig, os.path.join(save_dir, "effective_mass_comparison_vs_time.png"))

def plot_force_ratio_comparison(df: pd.DataFrame, save_dir: str):
    require_columns(df, ["t_demo_sec", "force_ratio_dir", "force_ratio_iso"])

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(
        df["t_demo_sec"],
        df["force_ratio_dir"],
        linewidth=LINE_WIDTH,
        label="force ratio (directional)",
    )
    ax.plot(
        df["t_demo_sec"],
        df["force_ratio_iso"],
        linewidth=LINE_WIDTH,
        label="force ratio (ISO fixed)",
    )
    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=LINE_WIDTH,
        label="threshold ratio = 1.0",
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Force Ratio")
    ax.set_title("Force Ratio Comparison")
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(loc="best", framealpha=0.9)
    style_axes(ax)

    save_figure(fig, os.path.join(save_dir, "force_ratio_comparison_vs_time.png"))


def plot_selected_link(df: pd.DataFrame, save_dir: str):
    link_order = ["link1", "link2", "link3", "link4", "link5", "link6"]
    link_to_num = {name: i + 1 for i, name in enumerate(link_order)}

    y = df["selected_link"].map(link_to_num)

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.step(df["t_demo_sec"], y, where="post", linewidth=LINE_WIDTH)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Selected Link")
    ax.set_title("Selected Collision Link over Time")
    ax.set_yticks([1, 2, 3, 4, 5, 6])
    ax.set_yticklabels(link_order)
    ax.grid(True, alpha=GRID_ALPHA)
    style_axes(ax)

    save_figure(fig, os.path.join(save_dir, "selected_link_vs_time.png"))


def plot_directional_meff_and_selected_link(df: pd.DataFrame, save_dir: str):
    """
    Save a combined figure:
    - left y-axis: directional human and robot effective mass
    - right y-axis: selected collision link index
    """
    required_cols = [
        "t_demo_sec",
        "human_meff_dir",
        "human_meff_iso",
        "robot_meff",
        "selected_link",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns for directional Meff/link plot: {missing}"
        )

    link_order = ["link1", "link2", "link3", "link4", "link5", "link6"]
    link_to_num = {name: i + 1 for i, name in enumerate(link_order)}
    selected_link_num = df["selected_link"].map(link_to_num)

    if selected_link_num.isna().any():
        unknown = sorted(
            str(v) for v in df.loc[selected_link_num.isna(), "selected_link"].unique()
        )
        raise ValueError(f"Unknown selected_link values: {unknown}")

    t = df["t_demo_sec"]
    fig, ax_meff = plt.subplots(figsize=(13, 6))
    ax_link = ax_meff.twinx()
    ax_link.set_zorder(1)
    ax_meff.set_zorder(2)
    ax_meff.patch.set_visible(False)

    human_meff_line, = ax_meff.plot(
        t,
        df["human_meff_dir"],
        color="tab:red",
        linewidth=LINE_WIDTH,
        label="Directional human effective mass",
        zorder=5,
    )
    iso_meff_line, = ax_meff.plot(
        t,
        df["human_meff_iso"],
        color="tab:orange",
        linewidth=LINE_WIDTH,
        label="ISO fixed head mass",
        zorder=5,
    )
    robot_meff_line, = ax_meff.plot(
        t,
        df["robot_meff"],
        color="tab:green",
        linewidth=LINE_WIDTH,
        label="Robot effective mass",
        zorder=5,
    )

    link_line, = ax_link.step(
        t,
        selected_link_num,
        where="post",
        color="tab:blue",
        linewidth=LINE_WIDTH,
        linestyle="--",
        alpha=0.9,
        label="Selected collision link",
        zorder=4,
    )

    ax_meff.set_xlabel("Time [s]", labelpad=10)
    ax_meff.set_ylabel("Effective mass [kg]", labelpad=12)
    ax_link.set_ylabel("Selected collision link", labelpad=22)
    ax_link.set_yticks([1, 2, 3, 4, 5, 6])
    ax_link.set_yticklabels(link_order)
    ax_link.tick_params(axis="y", pad=8)

    ax_meff.grid(True, alpha=GRID_ALPHA)
    ax_link.patch.set_visible(False)

    lines = [human_meff_line, iso_meff_line, robot_meff_line, link_line]
    labels = [line.get_label() for line in lines]
    legend = ax_meff.legend(
        lines,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.8",
        fontsize=LEGEND_SIZE,
    )
    legend.set_zorder(30)
    ax_meff.add_artist(legend)
    style_axes(ax_meff)
    style_axes(ax_link)

    save_figure(
        fig,
        os.path.join(save_dir, "directional_meff_and_selected_link_vs_time.png"),
    )


def plot_selected_body(df: pd.DataFrame, save_dir: str):
    body_order = list(pd.unique(df["selected_body"]))
    body_to_num = {name: i for i, name in enumerate(body_order)}
    y = df["selected_body"].map(body_to_num)

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.step(df["t_demo_sec"], y, where="post", linewidth=LINE_WIDTH)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Selected Body")
    ax.set_title("Selected Collision Body over Time")
    ax.set_yticks(range(len(body_order)))
    ax.set_yticklabels(body_order)
    ax.grid(True, alpha=GRID_ALPHA)
    style_axes(ax)

    save_figure(fig, os.path.join(save_dir, "selected_body_vs_time.png"))


def plot_link_speeds(df: pd.DataFrame, save_dir: str):
    fig, ax = plt.subplots(figsize=(13, 6))
    for i in range(1, 7):
        col = f"link{i}_speed"
        ax.plot(
            df["t_demo_sec"],
            df[col],
            linewidth=LINE_WIDTH,
            label=f"link {i}",
        )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("CoM Speed Magnitude [m/s]")
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=6,
        framealpha=0.95,
    )
    style_axes(ax)

    save_figure(fig, os.path.join(save_dir, "link_speeds_vs_time.png"))


def plot_link_speeds_one_cycle(df: pd.DataFrame, save_dir: str):
    """
    Save only the representative one-cycle interval.

    The current demo contains roughly two repeated cycles in the paper window.
    The attached cycle corresponds to the stop-to-stop interval around
    9.27--15.46 s; a small margin is used for readability.
    """
    cycle_df = filter_time_range(df, ONE_CYCLE_T_START, ONE_CYCLE_T_END)

    fig, ax = plt.subplots(figsize=(13, 6))
    for i in range(1, 7):
        col = f"link{i}_speed"
        ax.plot(
            cycle_df["t_demo_sec"],
            cycle_df[col],
            linewidth=LINE_WIDTH,
            label=f"link {i}",
        )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("CoM Speed Magnitude [m/s]")
    ax.set_title("Link CoM Speed Magnitude over Time")
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(loc="best", framealpha=0.9)
    style_axes(ax)

    save_figure(
        fig,
        os.path.join(save_dir, "link_speeds_one_cycle_vs_time.png"),
    )

    print(
        "[INFO] One-cycle link-speed window: "
        f"{ONE_CYCLE_T_START:.2f}-{ONE_CYCLE_T_END:.2f} s"
    )


def plot_speed_scale_and_link_speeds(df: pd.DataFrame, save_dir: str):
    """
    Save a journal-friendly combined figure:
    - left y-axis: command speed scale
    - right y-axis: robot link CoM speed magnitudes
    """
    link_cols = [f"link{i}_speed" for i in range(1, 7)]
    missing = [
        c for c in (
            ["t_demo_sec", "speed_scale_applied"]
            + link_cols
        )
        if c not in df.columns
    ]
    if missing:
        raise ValueError(f"Missing columns for combined speed plot: {missing}")

    t = df["t_demo_sec"]
    fig, ax_scale = plt.subplots(figsize=(13, 6))
    ax_speed = ax_scale.twinx()

    ax_scale.plot(
        t,
        df["speed_scale_applied"],
        color="black",
        linewidth=4.5,
        alpha=0.75,
        zorder=5,
    )
    speed_scale_line, = ax_scale.plot(
        t,
        df["speed_scale_applied"],
        color="tab:blue",
        linewidth=3.0,
        label="Command speed scale",
        zorder=6,
    )
    ax_scale.set_xlabel("Time [s]")
    ax_scale.set_ylabel("Command speed scale")
    ax_scale.set_ylim(0.0, 1.05)
    ax_scale.set_title("Command Speed Scale and Link CoM Speeds over Time")
    ax_scale.grid(True, alpha=GRID_ALPHA)

    link_lines = []
    for i, col in enumerate(link_cols, start=1):
        line, = ax_speed.plot(
            t,
            df[col],
            linewidth=2.0,
            alpha=0.85,
            label=f"link {i}",
        )
        link_lines.append(line)

    ax_speed.set_ylabel("CoM speed magnitude [m/s]")

    lines = [speed_scale_line] + link_lines
    labels = [line.get_label() for line in lines]
    ax_scale.legend(lines, labels, loc="upper right", ncol=1, framealpha=0.9)
    style_axes(ax_scale)
    style_axes(ax_speed)

    save_figure(
        fig,
        os.path.join(save_dir, "speed_scale_and_link_com_speed_vs_time.png"),
    )


def plot_relative_speed_components(df: pd.DataFrame, save_dir: str):
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(
        df["t_demo_sec"],
        df["robot_speed_along_u_raw"],
        linewidth=LINE_WIDTH,
        label="robot speed along u",
    )
    ax.plot(
        df["t_demo_sec"],
        df["approach_bias_added"],
        linewidth=LINE_WIDTH,
        label="approach bias added",
    )
    ax.plot(
        df["t_demo_sec"],
        df["relative_speed_along_u"],
        linewidth=LINE_WIDTH,
        label="relative speed along u",
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Speed [m/s]")
    ax.set_title("Relative Speed Components over Time")
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(loc="best", framealpha=0.9)
    style_axes(ax)

    save_figure(fig, os.path.join(save_dir, "relative_speed_components_vs_time.png"))


def plot_u_components(df: pd.DataFrame, save_dir: str):
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(df["t_demo_sec"], df["u_x"], linewidth=LINE_WIDTH, label="u_x")
    ax.plot(df["t_demo_sec"], df["u_y"], linewidth=LINE_WIDTH, label="u_y")
    ax.plot(df["t_demo_sec"], df["u_z"], linewidth=LINE_WIDTH, label="u_z")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Direction Component")
    ax.set_title("Collision Direction u over Time")
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(loc="best", framealpha=0.9)
    style_axes(ax)

    save_figure(fig, os.path.join(save_dir, "u_components_vs_time.png"))


def plot_meff_comparison(df: pd.DataFrame, save_dir: str):
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(
        df["t_demo_sec"],
        df["human_meff_dir"],
        linewidth=LINE_WIDTH,
        label="Human directional effective mass",
    )
    ax.plot(
        df["t_demo_sec"],
        df["human_meff_iso"],
        linewidth=LINE_WIDTH,
        label="ISO fixed human mass",
    )
    ax.plot(
        df["t_demo_sec"],
        df["robot_meff"],
        linewidth=LINE_WIDTH,
        label="Robot effective mass",
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Effective Mass [kg]")
    ax.set_title("Human and Robot Effective Masses over Time")
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(loc="upper right", framealpha=0.9)
    style_axes(ax)

    save_figure(fig, os.path.join(save_dir, "meff_comparison_vs_time.png"))


def plot_reduced_mass_comparison(df: pd.DataFrame, save_dir: str):
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(
        df["t_demo_sec"],
        df["reduced_mass_dir"],
        linewidth=LINE_WIDTH,
        label="reduced mass (directional)",
    )
    ax.plot(
        df["t_demo_sec"],
        df["reduced_mass_iso"],
        linewidth=LINE_WIDTH,
        label="reduced mass (ISO fixed)",
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Reduced Mass [kg]")
    ax.set_title("Reduced Mass Comparison")
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(loc="best", framealpha=0.9)
    style_axes(ax)

    save_figure(fig, os.path.join(save_dir, "reduced_mass_comparison_vs_time.png"))


def plot_v_allow_comparison(df: pd.DataFrame, save_dir: str):
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(
        df["t_demo_sec"],
        df["v_allow_dir"],
        linewidth=LINE_WIDTH,
        label="allowable velocity (directional)",
    )
    ax.plot(
        df["t_demo_sec"],
        df["v_allow_iso"],
        linewidth=LINE_WIDTH,
        label="allowable velocity (ISO fixed)",
    )
    ax.plot(
        df["t_demo_sec"],
        df["relative_speed_along_u"],
        linewidth=LINE_WIDTH,
        label="relative speed along u",
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Velocity [m/s]")
    ax.set_title("Allowed Velocity vs Relative Speed")
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(loc="best", framealpha=0.9)
    style_axes(ax)

    save_figure(fig, os.path.join(save_dir, "v_allow_comparison_vs_time.png"))


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


def print_journal_cycle_summary(df: pd.DataFrame):
    require_columns(
        df,
        [
            "t_demo_sec",
            "relative_speed_along_u",
            "estimated_collision_force_dir",
            "estimated_collision_force_iso",
            "speed_scale_applied",
            "selected_link",
        ],
    )

    eps = 1e-8
    approach = (
        (df["relative_speed_along_u"] > 0.0) &
        (df["estimated_collision_force_iso"].abs() > eps)
    )
    force_diff_percent = (
        (
            df.loc[approach, "estimated_collision_force_dir"] -
            df.loc[approach, "estimated_collision_force_iso"]
        ) /
        df.loc[approach, "estimated_collision_force_iso"].abs()
    ) * 100.0

    link_changes = df["selected_link"] != df["selected_link"].shift(1)
    transition_rows = df.loc[link_changes & df.index.to_series().ne(df.index[0])]

    print("\n=== Journal Cycle Summary ===")
    print(
        "time window [s] : "
        f"{df['t_demo_sec'].min():.3f}-{df['t_demo_sec'].max():.3f}"
    )
    print(f"rows           : {len(df)}")
    print(
        "Mean |Delta F| [%] (v_app > 0): "
        f"{force_diff_percent.abs().mean():.2f}"
    )
    print(
        "Max |Delta F| [%] (v_app > 0) : "
        f"{force_diff_percent.abs().max():.2f}"
    )
    print(
        "Min applied speed scale       : "
        f"{df['speed_scale_applied'].min():.3f}"
    )

    print("Selected-link transitions:")
    if transition_rows.empty:
        print("  none")
    else:
        for idx, row in transition_rows.iterrows():
            prev_link = df.loc[df.index[df.index.get_loc(idx) - 1], "selected_link"]
            print(
                f"  t={row['t_demo_sec']:.3f} s: "
                f"{prev_link} -> {row['selected_link']}"
            )


def main():
    ensure_dir(SAVE_DIR)
    df = load_demo_metrics(CSV_PATH)
    df = filter_time_range(df, PLOT_T_START, PLOT_T_END)

    print("Loaded rows:", len(df))
    print(df.head())

    print_spike_summary(df)

    plot_speed_scale(df, SAVE_DIR)
    plot_force_comparison(df, SAVE_DIR)
    plot_force_ratio_comparison(df, SAVE_DIR)
    plot_selected_link(df, SAVE_DIR)
    plot_directional_meff_and_selected_link(df, SAVE_DIR)
    plot_selected_body(df, SAVE_DIR)
    plot_link_speeds(df, SAVE_DIR)
    plot_link_speeds_one_cycle(df, SAVE_DIR)
    plot_speed_scale_and_link_speeds(df, SAVE_DIR)
    plot_relative_speed_components(df, SAVE_DIR)
    plot_u_components(df, SAVE_DIR)
    plot_meff_comparison(df, SAVE_DIR)
    plot_reduced_mass_comparison(df, SAVE_DIR)
    plot_v_allow_comparison(df, SAVE_DIR)
    plot_effective_mass_comparison(df, SAVE_DIR)

    print(f"\nSaved plots to: {SAVE_DIR}")


if __name__ == "__main__":
    main()
