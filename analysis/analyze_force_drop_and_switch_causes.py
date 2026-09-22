#!/usr/bin/env python3
"""
Standalone root-cause diagnostic for two collision-force discontinuities.

Target events
-------------
A. Sudden force DROP immediately after the local maximum around 6.31 s.
   This is not called a peak. It is a sharp drop, dip, or negative step.

B. Sudden force SPIKE around 7.22 s, typically coincident with a
   selected collision-link transition such as link4 -> link6.

The script is intentionally independent from the paper plotting script.
It reads the original demo CSV directly and writes a separate diagnostic
report, CSV tables, and zoomed plots.

Model decomposition
-------------------
For a transient collision-force model of the form

    F_est = v_app * sqrt(k * m_red)

    m_red = (m_human * m_robot) / (m_human + m_robot)

an exact event-to-before force ratio can be decomposed into

    F_event / F_before
      = (v_app_event / v_app_before)
        * sqrt(k_event / k_before)
        * sqrt(m_red_event / m_red_before)

The code computes each factor when the corresponding columns are logged.
If direct v_app or stiffness columns are absent, it does NOT claim an exact
low-level cause. Instead, it reports the combined unexplained non-mass factor
and lists the missing log columns required for confirmation.

Recommended CSV columns for definitive diagnosis
-------------------------------------------------
Required baseline columns:
    t_demo_sec
    estimated_collision_force_dir
    human_meff_dir
    robot_meff
    selected_link
    selected_body
    link1_speed ... link6_speed

Strongly recommended direct model columns:
    v_app
    selected_body_stiffness
    u_x, u_y, u_z

Recommended vector decomposition columns:
    selected_link_vx, selected_link_vy, selected_link_vz
    selected_body_vx, selected_body_vy, selected_body_vz
    robot_speed_along_u
    human_speed_along_u

Recommended asynchronous-service diagnostics:
    selected_pair_id
    human_state_age_ms
    robot_state_age_ms
    human_meff_query_age_ms
    robot_meff_query_age_ms
    human_query_id
    robot_query_id

Usage
-----
    python analyze_force_drop_and_switch_causes.py

    python analyze_force_drop_and_switch_causes.py --no-show

    python analyze_force_drop_and_switch_causes.py \
        --csv /home/robotics/hrc_ws/analysis/demo_log/demo_metrics_head_paper.csv \
        --save-dir /home/robotics/hrc_ws/analysis/demo_log/force_drop_switch_diagnostics \
        --no-show
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Default paths and event-search windows
# ============================================================

DEFAULT_CSV_PATH = (
    "/home/robotics/hrc_ws/analysis/demo_log/"
    "demo_metrics_head_paper.csv"
)

DEFAULT_SAVE_DIR = (
    "/home/robotics/hrc_ws/analysis/demo_log/"
    "force_drop_switch_diagnostics_head_paper"
)

# Wide plot window
DEFAULT_PLOT_T_START = 5.75
DEFAULT_PLOT_T_END = 7.60

# Event A: sudden negative step immediately after the ~6.31 s local maximum
DEFAULT_DROP_SEARCH_T_START = 6.25
DEFAULT_DROP_SEARCH_T_END = 6.55

# Event B: ~7.22 s spike / selected-link transition
DEFAULT_SWITCH_SEARCH_T_START = 7.10
DEFAULT_SWITCH_SEARCH_T_END = 7.35

# Comparison windows for robust local medians
DEFAULT_BASELINE_WINDOW_SEC = 0.12
DEFAULT_AFTER_WINDOW_SEC = 0.12

# Numerical tolerances
EPS = 1e-12
ANGLE_EPS = 1e-12

# Plot style
FIG_TITLE_SIZE = 16
AXIS_LABEL_SIZE = 12
TICK_LABEL_SIZE = 10
LEGEND_SIZE = 9
LINE_WIDTH = 2.0
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


# ============================================================
# Column aliases
# ============================================================

V_APP_CANDIDATES = [
    "v_app",
    "v_app_ij",
    "selected_v_app",
    "approach_speed",
    "relative_approach_speed",
    "projected_relative_speed",
    "collision_approach_speed",
]

STIFFNESS_CANDIDATES = [
    "selected_body_stiffness",
    "body_stiffness",
    "stiffness",
    "spring_constant",
    "k_body",
]

ROBOT_SPEED_ALONG_U_CANDIDATES = [
    "robot_speed_along_u",
    "selected_robot_speed_along_u",
    "robot_velocity_along_u",
    "robot_v_along_u",
]

HUMAN_SPEED_ALONG_U_CANDIDATES = [
    "human_speed_along_u",
    "selected_human_speed_along_u",
    "human_velocity_along_u",
    "human_v_along_u",
]

U_X_CANDIDATES = ["u_x", "selected_u_x", "collision_u_x", "direction_u_x"]
U_Y_CANDIDATES = ["u_y", "selected_u_y", "collision_u_y", "direction_u_y"]
U_Z_CANDIDATES = ["u_z", "selected_u_z", "collision_u_z", "direction_u_z"]

ROBOT_VX_CANDIDATES = ["selected_link_vx", "robot_vx", "selected_robot_vx"]
ROBOT_VY_CANDIDATES = ["selected_link_vy", "robot_vy", "selected_robot_vy"]
ROBOT_VZ_CANDIDATES = ["selected_link_vz", "robot_vz", "selected_robot_vz"]

HUMAN_VX_CANDIDATES = ["selected_body_vx", "human_vx", "selected_human_vx"]
HUMAN_VY_CANDIDATES = ["selected_body_vy", "human_vy", "selected_human_vy"]
HUMAN_VZ_CANDIDATES = ["selected_body_vz", "human_vz", "selected_human_vz"]

PAIR_ID_CANDIDATES = ["selected_pair_id", "collision_pair_id", "pair_id"]

AGE_CANDIDATES = [
    "human_state_age_ms",
    "robot_state_age_ms",
    "human_meff_query_age_ms",
    "robot_meff_query_age_ms",
    "human_query_latency_ms",
    "robot_query_latency_ms",
]

QUERY_ID_CANDIDATES = [
    "human_query_id",
    "robot_query_id",
    "query_id",
    "request_id",
]

BASE_NUMERIC_COLUMNS = [
    "t_demo_sec",
    "estimated_collision_force_dir",
    "estimated_collision_force_iso",
    "estimated_collision_force",
    "threshold_force",
    "speed_scale_target",
    "speed_scale_applied",
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


# ============================================================
# Data structures
# ============================================================

@dataclass(frozen=True)
class DetectedEvent:
    event_name: str
    event_kind: str
    event_index: int
    previous_index: int
    next_index: int
    event_time_sec: float
    previous_time_sec: float
    next_time_sec: float


# ============================================================
# Generic helpers
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def first_available_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def safe_float(value) -> float:
    try:
        if pd.isna(value):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def safe_ratio(numerator, denominator) -> float:
    n = safe_float(numerator)
    d = safe_float(denominator)
    if not np.isfinite(n) or not np.isfinite(d) or abs(d) < EPS:
        return float("nan")
    return n / d


def fmt(value, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (float, np.floating, int, np.integer)):
        if not np.isfinite(float(value)):
            return "NaN"
        return f"{float(value):.{digits}f}"
    return str(value)


def nearest_index(df: pd.DataFrame, target_time: float) -> int:
    return int((df["t_demo_sec"] - target_time).abs().idxmin())


def previous_index(df: pd.DataFrame, idx: int) -> int:
    loc = df.index.get_loc(idx)
    if isinstance(loc, slice):
        loc = loc.start
    if loc <= 0:
        return int(df.index[0])
    return int(df.index[loc - 1])


def next_index(df: pd.DataFrame, idx: int) -> int:
    loc = df.index.get_loc(idx)
    if isinstance(loc, slice):
        loc = loc.start
    if loc >= len(df.index) - 1:
        return int(df.index[-1])
    return int(df.index[loc + 1])


def local_window(
    df: pd.DataFrame,
    center_time: float,
    start_offset: float,
    end_offset: float,
) -> pd.DataFrame:
    return df[
        (df["t_demo_sec"] >= center_time + start_offset)
        & (df["t_demo_sec"] <= center_time + end_offset)
    ].copy()


def median_or_nan(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return float("nan")
    return float(vals.median())


def vector_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)):
        return float("nan")
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < ANGLE_EPS or nb < ANGLE_EPS:
        return float("nan")
    cos_value = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return math.degrees(math.acos(cos_value))


def normalize_link_name(value) -> str:
    raw = str(value).strip().lower().replace(" ", "")
    raw = raw.replace("_", "")
    if raw.startswith("link"):
        suffix = raw[4:]
        if suffix.isdigit():
            return f"link{int(suffix)}"
    return str(value)


# ============================================================
# Loading and feature engineering
# ============================================================

def load_metrics(csv_path: str) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    require_columns(
        df,
        [
            "t_demo_sec",
            "estimated_collision_force_dir",
            "human_meff_dir",
            "robot_meff",
        ],
    )

    optional_numeric = set(BASE_NUMERIC_COLUMNS)
    optional_numeric.update(V_APP_CANDIDATES)
    optional_numeric.update(STIFFNESS_CANDIDATES)
    optional_numeric.update(ROBOT_SPEED_ALONG_U_CANDIDATES)
    optional_numeric.update(HUMAN_SPEED_ALONG_U_CANDIDATES)
    optional_numeric.update(U_X_CANDIDATES)
    optional_numeric.update(U_Y_CANDIDATES)
    optional_numeric.update(U_Z_CANDIDATES)
    optional_numeric.update(ROBOT_VX_CANDIDATES)
    optional_numeric.update(ROBOT_VY_CANDIDATES)
    optional_numeric.update(ROBOT_VZ_CANDIDATES)
    optional_numeric.update(HUMAN_VX_CANDIDATES)
    optional_numeric.update(HUMAN_VY_CANDIDATES)
    optional_numeric.update(HUMAN_VZ_CANDIDATES)
    optional_numeric.update(AGE_CANDIDATES)

    for col in optional_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("t_demo_sec").reset_index(drop=True)

    if "selected_link" in df.columns:
        df["selected_link_normalized"] = df["selected_link"].map(normalize_link_name)
    else:
        df["selected_link_normalized"] = "not_logged"

    if "selected_body" not in df.columns:
        df["selected_body"] = "not_logged"

    return df


def selected_link_speed_series(df: pd.DataFrame) -> pd.Series:
    result = pd.Series(np.nan, index=df.index, dtype=float)
    if "selected_link_normalized" not in df.columns:
        return result

    for link_idx in range(1, 7):
        link_name = f"link{link_idx}"
        speed_col = f"link{link_idx}_speed"
        if speed_col not in df.columns:
            continue
        mask = df["selected_link_normalized"] == link_name
        result.loc[mask] = pd.to_numeric(df.loc[mask, speed_col], errors="coerce")

    return result


def add_derived_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Optional[str]]]:
    out = df.copy()

    aliases = {
        "v_app": first_available_column(out, V_APP_CANDIDATES),
        "stiffness": first_available_column(out, STIFFNESS_CANDIDATES),
        "robot_speed_along_u": first_available_column(out, ROBOT_SPEED_ALONG_U_CANDIDATES),
        "human_speed_along_u": first_available_column(out, HUMAN_SPEED_ALONG_U_CANDIDATES),
        "u_x": first_available_column(out, U_X_CANDIDATES),
        "u_y": first_available_column(out, U_Y_CANDIDATES),
        "u_z": first_available_column(out, U_Z_CANDIDATES),
        "robot_vx": first_available_column(out, ROBOT_VX_CANDIDATES),
        "robot_vy": first_available_column(out, ROBOT_VY_CANDIDATES),
        "robot_vz": first_available_column(out, ROBOT_VZ_CANDIDATES),
        "human_vx": first_available_column(out, HUMAN_VX_CANDIDATES),
        "human_vy": first_available_column(out, HUMAN_VY_CANDIDATES),
        "human_vz": first_available_column(out, HUMAN_VZ_CANDIDATES),
        "pair_id": first_available_column(out, PAIR_ID_CANDIDATES),
    }

    # Core reduced mass
    valid_mass = (
        out["human_meff_dir"].notna()
        & out["robot_meff"].notna()
        & (out["human_meff_dir"] > 0.0)
        & (out["robot_meff"] > 0.0)
    )
    out["reduced_mass_dir"] = np.nan
    out.loc[valid_mass, "reduced_mass_dir"] = (
        out.loc[valid_mass, "human_meff_dir"]
        * out.loc[valid_mass, "robot_meff"]
        / (
            out.loc[valid_mass, "human_meff_dir"]
            + out.loc[valid_mass, "robot_meff"]
        )
    )
    out["sqrt_reduced_mass_dir"] = np.sqrt(out["reduced_mass_dir"])

    # This is proportional to v_app * sqrt(k).
    out["non_mass_proxy"] = (
        out["estimated_collision_force_dir"]
        / out["sqrt_reduced_mass_dir"]
    )

    out["selected_link_speed"] = selected_link_speed_series(out)

    # Copy aliases into normalized internal columns.
    for normalized_name in [
        "v_app",
        "stiffness",
        "robot_speed_along_u",
        "human_speed_along_u",
        "u_x",
        "u_y",
        "u_z",
        "robot_vx",
        "robot_vy",
        "robot_vz",
        "human_vx",
        "human_vy",
        "human_vz",
    ]:
        src = aliases.get(normalized_name)
        if src is not None:
            out[f"logged_{normalized_name}"] = pd.to_numeric(out[src], errors="coerce")

    if aliases["pair_id"] is not None:
        out["logged_pair_id"] = out[aliases["pair_id"]].astype(str)

    # Recompute directional projections if vector logs are present.
    vector_requirements = [
        "logged_u_x",
        "logged_u_y",
        "logged_u_z",
        "logged_robot_vx",
        "logged_robot_vy",
        "logged_robot_vz",
        "logged_human_vx",
        "logged_human_vy",
        "logged_human_vz",
    ]
    if all(col in out.columns for col in vector_requirements):
        u = out[["logged_u_x", "logged_u_y", "logged_u_z"]].to_numpy(dtype=float)
        vr = out[["logged_robot_vx", "logged_robot_vy", "logged_robot_vz"]].to_numpy(dtype=float)
        vh = out[["logged_human_vx", "logged_human_vy", "logged_human_vz"]].to_numpy(dtype=float)

        out["recomputed_robot_speed_along_u"] = np.einsum("ij,ij->i", vr, u)
        out["recomputed_human_speed_along_u"] = np.einsum("ij,ij->i", vh, u)
        out["recomputed_relative_speed_dot_u"] = np.einsum("ij,ij->i", vr - vh, u)

        # The sign convention depends on how u is defined in the controller.
        # Export both forms instead of silently assuming one.
        out["recomputed_v_app_positive_dot"] = np.maximum(
            0.0,
            out["recomputed_relative_speed_dot_u"],
        )
        out["recomputed_v_app_negative_dot"] = np.maximum(
            0.0,
            -out["recomputed_relative_speed_dot_u"],
        )

    # Reconstructed force when v_app and stiffness are logged.
    if "logged_v_app" in out.columns and "logged_stiffness" in out.columns:
        out["reconstructed_force_from_logged_terms"] = (
            out["logged_v_app"]
            * np.sqrt(out["logged_stiffness"] * out["reduced_mass_dir"])
        )
        out["force_reconstruction_error_N"] = (
            out["estimated_collision_force_dir"]
            - out["reconstructed_force_from_logged_terms"]
        )

    # One-step differences: useful for abrupt-step detection.
    diff_cols = [
        "estimated_collision_force_dir",
        "robot_meff",
        "human_meff_dir",
        "reduced_mass_dir",
        "sqrt_reduced_mass_dir",
        "non_mass_proxy",
        "selected_link_speed",
        "speed_scale_target",
        "speed_scale_applied",
        "logged_v_app",
        "logged_stiffness",
        "logged_robot_speed_along_u",
        "logged_human_speed_along_u",
    ]
    for col in diff_cols:
        if col in out.columns:
            out[f"delta_{col}"] = out[col].diff()

    out["selected_link_changed"] = (
        out["selected_link_normalized"]
        != out["selected_link_normalized"].shift(1)
    )
    out["selected_body_changed"] = (
        out["selected_body"].astype(str)
        != out["selected_body"].astype(str).shift(1)
    )

    return out, aliases


# ============================================================
# Event detection
# ============================================================

def detect_largest_negative_force_step(
    df: pd.DataFrame,
    t_start: float,
    t_end: float,
) -> DetectedEvent:
    window = df[
        (df["t_demo_sec"] >= t_start)
        & (df["t_demo_sec"] <= t_end)
    ].copy()

    if window.empty:
        raise RuntimeError(f"No data in drop-search window {t_start:.3f}--{t_end:.3f} s")

    valid = window["delta_estimated_collision_force_dir"].dropna()
    if valid.empty:
        raise RuntimeError("No finite force differences found in drop-search window")

    event_idx = int(valid.idxmin())
    prev_idx = previous_index(df, event_idx)
    nxt_idx = next_index(df, event_idx)

    return DetectedEvent(
        event_name="event_A_post_6p31_force_drop",
        event_kind="largest_negative_force_step",
        event_index=event_idx,
        previous_index=prev_idx,
        next_index=nxt_idx,
        event_time_sec=float(df.loc[event_idx, "t_demo_sec"]),
        previous_time_sec=float(df.loc[prev_idx, "t_demo_sec"]),
        next_time_sec=float(df.loc[nxt_idx, "t_demo_sec"]),
    )


def detect_link_switch_or_largest_positive_step(
    df: pd.DataFrame,
    t_start: float,
    t_end: float,
) -> DetectedEvent:
    window = df[
        (df["t_demo_sec"] >= t_start)
        & (df["t_demo_sec"] <= t_end)
    ].copy()

    if window.empty:
        raise RuntimeError(f"No data in switch-search window {t_start:.3f}--{t_end:.3f} s")

    switches = window[window["selected_link_changed"]].copy()
    switches = switches[switches.index > 0]

    if not switches.empty:
        # Prefer the switch with the largest positive force jump.
        candidates = switches["delta_estimated_collision_force_dir"].dropna()
        if not candidates.empty:
            event_idx = int(candidates.idxmax())
        else:
            event_idx = int(switches.index[0])
        event_kind = "selected_link_switch"
    else:
        valid = window["delta_estimated_collision_force_dir"].dropna()
        if valid.empty:
            raise RuntimeError("No finite force differences found in switch-search window")
        event_idx = int(valid.idxmax())
        event_kind = "largest_positive_force_step_no_link_switch_found"

    prev_idx = previous_index(df, event_idx)
    nxt_idx = next_index(df, event_idx)

    return DetectedEvent(
        event_name="event_B_7p22_force_spike_or_link_switch",
        event_kind=event_kind,
        event_index=event_idx,
        previous_index=prev_idx,
        next_index=nxt_idx,
        event_time_sec=float(df.loc[event_idx, "t_demo_sec"]),
        previous_time_sec=float(df.loc[prev_idx, "t_demo_sec"]),
        next_time_sec=float(df.loc[nxt_idx, "t_demo_sec"]),
    )


# ============================================================
# Event summarization and model-factor decomposition
# ============================================================

def get_row_value(df: pd.DataFrame, idx: int, col: str):
    if col not in df.columns:
        return np.nan
    return df.loc[idx, col]


def summarize_exact_step(df: pd.DataFrame, event: DetectedEvent) -> dict:
    before_idx = event.previous_index
    event_idx = event.event_index
    after_idx = event.next_index

    summary: dict[str, object] = {
        **asdict(event),
    }

    scalar_cols = [
        "estimated_collision_force_dir",
        "estimated_collision_force_iso",
        "threshold_force",
        "human_meff_dir",
        "robot_meff",
        "reduced_mass_dir",
        "sqrt_reduced_mass_dir",
        "selected_link_speed",
        "non_mass_proxy",
        "speed_scale_target",
        "speed_scale_applied",
        "logged_v_app",
        "logged_stiffness",
        "logged_robot_speed_along_u",
        "logged_human_speed_along_u",
        "logged_u_x",
        "logged_u_y",
        "logged_u_z",
        "recomputed_robot_speed_along_u",
        "recomputed_human_speed_along_u",
        "recomputed_relative_speed_dot_u",
        "recomputed_v_app_positive_dot",
        "recomputed_v_app_negative_dot",
        "reconstructed_force_from_logged_terms",
        "force_reconstruction_error_N",
    ]

    for col in scalar_cols:
        if col not in df.columns:
            continue
        before_val = get_row_value(df, before_idx, col)
        event_val = get_row_value(df, event_idx, col)
        after_val = get_row_value(df, after_idx, col)
        summary[f"before_{col}"] = before_val
        summary[f"event_{col}"] = event_val
        summary[f"after_{col}"] = after_val
        summary[f"ratio_event_to_before_{col}"] = safe_ratio(event_val, before_val)

    categorical_cols = [
        "selected_link_normalized",
        "selected_body",
        "logged_pair_id",
    ]
    for col in categorical_cols:
        if col not in df.columns:
            continue
        summary[f"before_{col}"] = get_row_value(df, before_idx, col)
        summary[f"event_{col}"] = get_row_value(df, event_idx, col)
        summary[f"after_{col}"] = get_row_value(df, after_idx, col)

    # Direction-vector angle jump.
    if all(c in df.columns for c in ["logged_u_x", "logged_u_y", "logged_u_z"]):
        u_before = df.loc[before_idx, ["logged_u_x", "logged_u_y", "logged_u_z"]].to_numpy(dtype=float)
        u_event = df.loc[event_idx, ["logged_u_x", "logged_u_y", "logged_u_z"]].to_numpy(dtype=float)
        summary["u_angle_change_before_to_event_deg"] = vector_angle_deg(u_before, u_event)

    # Core model-factor decomposition.
    force_ratio = safe_ratio(
        get_row_value(df, event_idx, "estimated_collision_force_dir"),
        get_row_value(df, before_idx, "estimated_collision_force_dir"),
    )
    mass_factor = safe_ratio(
        get_row_value(df, event_idx, "sqrt_reduced_mass_dir"),
        get_row_value(df, before_idx, "sqrt_reduced_mass_dir"),
    )
    combined_non_mass_factor = safe_ratio(force_ratio, mass_factor)

    summary["observed_force_ratio"] = force_ratio
    summary["mass_factor_sqrt_mred_ratio"] = mass_factor
    summary["combined_non_mass_factor_ratio"] = combined_non_mass_factor

    v_app_ratio = float("nan")
    stiffness_sqrt_factor = float("nan")
    fully_reconstructed_factor = float("nan")
    residual_ratio = float("nan")

    if "logged_v_app" in df.columns:
        v_app_ratio = safe_ratio(
            get_row_value(df, event_idx, "logged_v_app"),
            get_row_value(df, before_idx, "logged_v_app"),
        )
        summary["v_app_ratio"] = v_app_ratio

    if "logged_stiffness" in df.columns:
        stiffness_ratio = safe_ratio(
            get_row_value(df, event_idx, "logged_stiffness"),
            get_row_value(df, before_idx, "logged_stiffness"),
        )
        if np.isfinite(stiffness_ratio) and stiffness_ratio >= 0.0:
            stiffness_sqrt_factor = math.sqrt(stiffness_ratio)
        summary["stiffness_ratio"] = stiffness_ratio
        summary["stiffness_sqrt_factor"] = stiffness_sqrt_factor

    if np.isfinite(v_app_ratio) and np.isfinite(stiffness_sqrt_factor) and np.isfinite(mass_factor):
        fully_reconstructed_factor = v_app_ratio * stiffness_sqrt_factor * mass_factor
        residual_ratio = safe_ratio(force_ratio, fully_reconstructed_factor)
        summary["fully_reconstructed_force_ratio"] = fully_reconstructed_factor
        summary["force_ratio_residual_after_logged_terms"] = residual_ratio

    # Robust local median comparison around the exact one-step event.
    center_t = event.event_time_sec
    before_window = local_window(
        df,
        center_time=center_t,
        start_offset=-DEFAULT_BASELINE_WINDOW_SEC,
        end_offset=-EPS,
    )
    after_window = local_window(
        df,
        center_time=center_t,
        start_offset=0.0,
        end_offset=DEFAULT_AFTER_WINDOW_SEC,
    )

    robust_cols = [
        "estimated_collision_force_dir",
        "robot_meff",
        "human_meff_dir",
        "reduced_mass_dir",
        "sqrt_reduced_mass_dir",
        "selected_link_speed",
        "non_mass_proxy",
        "logged_v_app",
        "logged_stiffness",
    ]
    for col in robust_cols:
        if col not in df.columns:
            continue
        before_median = median_or_nan(before_window[col])
        after_median = median_or_nan(after_window[col])
        summary[f"robust_before_median_{col}"] = before_median
        summary[f"robust_after_median_{col}"] = after_median
        summary[f"robust_after_to_before_ratio_{col}"] = safe_ratio(after_median, before_median)

    # Change flags.
    summary["selected_link_changed_at_event"] = bool(df.loc[event_idx, "selected_link_changed"])
    summary["selected_body_changed_at_event"] = bool(df.loc[event_idx, "selected_body_changed"])
    summary["robot_meff_decreased_at_event"] = (
        safe_float(df.loc[event_idx, "robot_meff"])
        < safe_float(df.loc[before_idx, "robot_meff"])
    )
    summary["force_decreased_at_event"] = (
        safe_float(df.loc[event_idx, "estimated_collision_force_dir"])
        < safe_float(df.loc[before_idx, "estimated_collision_force_dir"])
    )

    # State-age and query-ID diagnostics.
    for col in AGE_CANDIDATES:
        if col in df.columns:
            summary[f"before_{col}"] = get_row_value(df, before_idx, col)
            summary[f"event_{col}"] = get_row_value(df, event_idx, col)
            summary[f"after_{col}"] = get_row_value(df, after_idx, col)
            summary[f"delta_event_minus_before_{col}"] = (
                safe_float(get_row_value(df, event_idx, col))
                - safe_float(get_row_value(df, before_idx, col))
            )

    for col in QUERY_ID_CANDIDATES:
        if col in df.columns:
            summary[f"before_{col}"] = str(get_row_value(df, before_idx, col))
            summary[f"event_{col}"] = str(get_row_value(df, event_idx, col))
            summary[f"after_{col}"] = str(get_row_value(df, after_idx, col))

    summary["diagnostic_verdict"] = build_verdict(summary)
    return summary


def build_verdict(summary: dict) -> str:
    event_name = str(summary.get("event_name", "event"))
    force_ratio = safe_float(summary.get("observed_force_ratio", np.nan))
    mass_factor = safe_float(summary.get("mass_factor_sqrt_mred_ratio", np.nan))
    non_mass_factor = safe_float(summary.get("combined_non_mass_factor_ratio", np.nan))
    link_changed = bool(summary.get("selected_link_changed_at_event", False))
    body_changed = bool(summary.get("selected_body_changed_at_event", False))
    u_angle = safe_float(summary.get("u_angle_change_before_to_event_deg", np.nan))
    v_ratio = safe_float(summary.get("v_app_ratio", np.nan))
    k_factor = safe_float(summary.get("stiffness_sqrt_factor", np.nan))
    residual = safe_float(summary.get("force_ratio_residual_after_logged_terms", np.nan))

    parts = [
        f"{event_name}: observed force ratio={fmt(force_ratio, 3)}x",
        f"mass factor sqrt(m_red) ratio={fmt(mass_factor, 3)}x",
        f"combined non-mass factor={fmt(non_mass_factor, 3)}x",
    ]

    if link_changed:
        parts.append("selected-link transition occurred at the exact step")
    else:
        parts.append("selected link did not change at the exact step")

    if body_changed:
        parts.append("selected body changed")
    else:
        parts.append("selected body did not change")

    if np.isfinite(u_angle):
        parts.append(f"collision-direction jump={fmt(u_angle, 2)} deg")

    if np.isfinite(v_ratio):
        parts.append(f"direct logged v_app ratio={fmt(v_ratio, 3)}x")
    else:
        parts.append("direct v_app log is absent: exact velocity contribution is not confirmable")

    if np.isfinite(k_factor):
        parts.append(f"sqrt(stiffness ratio)={fmt(k_factor, 3)}x")
    else:
        parts.append("direct stiffness log is absent: exact stiffness contribution is not confirmable")

    if np.isfinite(residual):
        parts.append(f"remaining ratio residual after logged terms={fmt(residual, 3)}x")
        if abs(residual - 1.0) <= 0.03:
            parts.append("logged terms reconstruct the force change within 3%")
        else:
            parts.append("logged terms do not fully reconstruct the force change; inspect sign convention, stale states, or implementation")

    return "; ".join(parts)


# ============================================================
# Reporting
# ============================================================

def columns_required_for_definitive_diagnosis(df: pd.DataFrame) -> list[dict[str, str]]:
    requirements = [
        ("Direct projected approach speed", V_APP_CANDIDATES),
        ("Body stiffness used by force model", STIFFNESS_CANDIDATES),
        ("Collision direction x", U_X_CANDIDATES),
        ("Collision direction y", U_Y_CANDIDATES),
        ("Collision direction z", U_Z_CANDIDATES),
        ("Selected robot-link velocity x", ROBOT_VX_CANDIDATES),
        ("Selected robot-link velocity y", ROBOT_VY_CANDIDATES),
        ("Selected robot-link velocity z", ROBOT_VZ_CANDIDATES),
        ("Selected human-body velocity x", HUMAN_VX_CANDIDATES),
        ("Selected human-body velocity y", HUMAN_VY_CANDIDATES),
        ("Selected human-body velocity z", HUMAN_VZ_CANDIDATES),
        ("Robot velocity projected along u", ROBOT_SPEED_ALONG_U_CANDIDATES),
        ("Human velocity projected along u", HUMAN_SPEED_ALONG_U_CANDIDATES),
        ("Selected candidate pair identifier", PAIR_ID_CANDIDATES),
    ]

    rows: list[dict[str, str]] = []
    for purpose, candidates in requirements:
        found = first_available_column(df, candidates)
        rows.append({
            "purpose": purpose,
            "status": "FOUND" if found is not None else "MISSING",
            "resolved_column": found or "",
            "accepted_aliases": ", ".join(candidates),
        })

    for col in AGE_CANDIDATES + QUERY_ID_CANDIDATES:
        rows.append({
            "purpose": f"Optional async diagnostic: {col}",
            "status": "FOUND" if col in df.columns else "MISSING",
            "resolved_column": col if col in df.columns else "",
            "accepted_aliases": col,
        })

    return rows


def print_event_summary(summary: dict) -> None:
    print("\n" + "=" * 118)
    print(f"{summary['event_name']}  [{summary['event_kind']}]")
    print("=" * 118)
    print(
        f"Exact step: {summary['previous_time_sec']:.5f} s"
        f" -> {summary['event_time_sec']:.5f} s"
        f" -> {summary['next_time_sec']:.5f} s"
    )

    keys = [
        "estimated_collision_force_dir",
        "human_meff_dir",
        "robot_meff",
        "reduced_mass_dir",
        "sqrt_reduced_mass_dir",
        "selected_link_speed",
        "non_mass_proxy",
        "speed_scale_target",
        "speed_scale_applied",
        "logged_v_app",
        "logged_stiffness",
        "logged_robot_speed_along_u",
        "logged_human_speed_along_u",
    ]

    print("\n[Exact adjacent samples]")
    for key in keys:
        before_key = f"before_{key}"
        event_key = f"event_{key}"
        after_key = f"after_{key}"
        ratio_key = f"ratio_event_to_before_{key}"
        if before_key not in summary:
            continue
        print(
            f"  {key:36s}: "
            f"before={fmt(summary[before_key], 6):>12s}  "
            f"event={fmt(summary[event_key], 6):>12s}  "
            f"after={fmt(summary[after_key], 6):>12s}  "
            f"ratio={fmt(summary.get(ratio_key, np.nan), 4):>9s}x"
        )

    print("\n[Candidate identity]")
    for key in ["selected_link_normalized", "selected_body", "logged_pair_id"]:
        before_key = f"before_{key}"
        if before_key not in summary:
            continue
        print(
            f"  {key:36s}: "
            f"{summary[before_key]} -> {summary[f'event_{key}']} -> {summary[f'after_{key}']}"
        )

    print("\n[Model-factor decomposition]")
    print(f"  observed force ratio                         : {fmt(summary.get('observed_force_ratio'), 6)} x")
    print(f"  mass factor sqrt(m_red_event / m_red_before) : {fmt(summary.get('mass_factor_sqrt_mred_ratio'), 6)} x")
    print(f"  combined non-mass factor                     : {fmt(summary.get('combined_non_mass_factor_ratio'), 6)} x")
    print(f"  direct logged v_app ratio                    : {fmt(summary.get('v_app_ratio'), 6)} x")
    print(f"  sqrt(stiffness ratio)                        : {fmt(summary.get('stiffness_sqrt_factor'), 6)} x")
    print(f"  reconstructed total factor                   : {fmt(summary.get('fully_reconstructed_force_ratio'), 6)} x")
    print(f"  residual after logged terms                  : {fmt(summary.get('force_ratio_residual_after_logged_terms'), 6)} x")
    print(f"  collision-direction angle jump               : {fmt(summary.get('u_angle_change_before_to_event_deg'), 3)} deg")

    print("\n[Verdict]")
    print("  " + str(summary["diagnostic_verdict"]))


def write_text_report(
    save_path: str,
    csv_path: str,
    summaries: list[dict],
    audit_df: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("Collision-force drop and switch diagnostic report")
    lines.append("=" * 72)
    lines.append(f"Input CSV: {csv_path}")
    lines.append("")
    lines.append("Terminology")
    lines.append("-----------")
    lines.append("The abrupt decrease immediately after the ~6.31 s maximum is a force drop, dip, or negative step, not a peak.")
    lines.append("The ~7.22 s upward discontinuity is a force spike or positive step.")
    lines.append("")

    for summary in summaries:
        lines.append(summary["event_name"])
        lines.append("-" * len(summary["event_name"]))
        lines.append(f"Event kind: {summary['event_kind']}")
        lines.append(
            f"Exact step: {summary['previous_time_sec']:.6f} s -> "
            f"{summary['event_time_sec']:.6f} s -> {summary['next_time_sec']:.6f} s"
        )
        lines.append(f"Observed force ratio: {fmt(summary.get('observed_force_ratio'), 6)} x")
        lines.append(f"Mass factor: {fmt(summary.get('mass_factor_sqrt_mred_ratio'), 6)} x")
        lines.append(f"Combined non-mass factor: {fmt(summary.get('combined_non_mass_factor_ratio'), 6)} x")
        lines.append(f"Direct v_app ratio: {fmt(summary.get('v_app_ratio'), 6)} x")
        lines.append(f"sqrt(stiffness ratio): {fmt(summary.get('stiffness_sqrt_factor'), 6)} x")
        lines.append(f"Direction angle jump: {fmt(summary.get('u_angle_change_before_to_event_deg'), 3)} deg")
        lines.append(f"Verdict: {summary['diagnostic_verdict']}")
        lines.append("")

    lines.append("Logging sufficiency audit")
    lines.append("-------------------------")
    for _, row in audit_df.iterrows():
        resolved = row["resolved_column"] if row["resolved_column"] else "-"
        lines.append(f"[{row['status']}] {row['purpose']}: {resolved}")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ============================================================
# Plotting
# ============================================================

def add_event_lines(axes, events: list[DetectedEvent]) -> None:
    line_styles = ["--", ":"]
    for idx, event in enumerate(events):
        for ax in axes:
            ax.axvline(
                event.event_time_sec,
                linestyle=line_styles[idx % len(line_styles)],
                linewidth=1.7,
                alpha=0.9,
                label=None,
            )


def create_overview_plot(
    df: pd.DataFrame,
    events: list[DetectedEvent],
    save_path: str,
    show: bool,
) -> None:
    fig, axes = plt.subplots(nrows=8, ncols=1, figsize=(15, 23), sharex=True)
    ax_force, ax_mass, ax_link_speed, ax_proxy, ax_scale, ax_link, ax_vapp, ax_async = axes

    # 1. Force
    ax_force.plot(df["t_demo_sec"], df["estimated_collision_force_dir"], linewidth=LINE_WIDTH, label="directional estimated force")
    if "estimated_collision_force_iso" in df.columns:
        ax_force.plot(df["t_demo_sec"], df["estimated_collision_force_iso"], linewidth=LINE_WIDTH, label="ISO fixed-mass estimated force")
    if "threshold_force" in df.columns:
        ax_force.plot(df["t_demo_sec"], df["threshold_force"], linewidth=LINE_WIDTH, label="threshold force")
    ax_force.set_ylabel("Force [N]")
    ax_force.set_title("Directional Collision Force: Sudden Drop and Link-Switch Spike")
    ax_force.legend(loc="best")

    # 2. Effective masses
    ax_mass.plot(df["t_demo_sec"], df["human_meff_dir"], linewidth=LINE_WIDTH, label="human directional M_eff")
    ax_mass.plot(df["t_demo_sec"], df["robot_meff"], linewidth=LINE_WIDTH, label="robot M_eff")
    ax_mass.plot(df["t_demo_sec"], df["reduced_mass_dir"], linewidth=LINE_WIDTH, label="directional reduced mass")
    ax_mass.set_ylabel("Mass [kg]")
    ax_mass.set_title("Effective-Mass Terms")
    ax_mass.legend(loc="best")

    # 3. Selected-link CoM speed magnitude
    ax_link_speed.plot(df["t_demo_sec"], df["selected_link_speed"], linewidth=LINE_WIDTH, label="selected-link CoM speed magnitude")
    ax_link_speed.set_ylabel("Speed [m/s]")
    ax_link_speed.set_title("Selected-Link CoM Speed Magnitude (Not Direction-Projected)")
    ax_link_speed.legend(loc="best")

    # 4. Non-mass proxy
    ax_proxy.plot(df["t_demo_sec"], df["non_mass_proxy"], linewidth=LINE_WIDTH, label="F_dir / sqrt(m_red)")
    ax_proxy.set_ylabel("Proxy")
    ax_proxy.set_title("Combined Non-Mass Proxy: v_app * sqrt(k)")
    ax_proxy.legend(loc="best")

    # 5. Speed scale
    if "speed_scale_target" in df.columns:
        ax_scale.plot(df["t_demo_sec"], df["speed_scale_target"], linewidth=LINE_WIDTH, label="target speed scale")
    if "speed_scale_applied" in df.columns:
        ax_scale.plot(df["t_demo_sec"], df["speed_scale_applied"], linewidth=LINE_WIDTH, label="applied speed scale")
    ax_scale.set_ylabel("Scale")
    ax_scale.set_title("Speed-Scale Response")
    ax_scale.legend(loc="best")

    # 6. Selected link
    link_to_y = {f"link{i}": i for i in range(1, 7)}
    selected_link_y = df["selected_link_normalized"].map(link_to_y)
    ax_link.step(df["t_demo_sec"], selected_link_y, where="post", linewidth=LINE_WIDTH, label="selected collision link")
    ax_link.set_yticks(list(link_to_y.values()), list(link_to_y.keys()))
    ax_link.set_ylabel("Link")
    ax_link.set_title("Selected Collision Link")
    ax_link.legend(loc="best")

    # 7. Direct v_app or explicit missing-log note
    if "logged_v_app" in df.columns:
        ax_vapp.plot(df["t_demo_sec"], df["logged_v_app"], linewidth=LINE_WIDTH, label="direct logged v_app")
        if "recomputed_v_app_positive_dot" in df.columns:
            ax_vapp.plot(df["t_demo_sec"], df["recomputed_v_app_positive_dot"], linewidth=1.4, label="recomputed max(0, (v_r-v_h)^T u)")
            ax_vapp.plot(df["t_demo_sec"], df["recomputed_v_app_negative_dot"], linewidth=1.4, label="recomputed max(0, -(v_r-v_h)^T u)")
        ax_vapp.set_title("Projected Approach Speed v_app")
    else:
        ax_vapp.plot(df["t_demo_sec"], df["non_mass_proxy"], linewidth=LINE_WIDTH, label="proxy repeated: direct v_app not logged")
        ax_vapp.set_title("Direct v_app Log Missing: Exact Velocity Cause Cannot Be Confirmed")
    ax_vapp.set_ylabel("Speed / proxy")
    ax_vapp.legend(loc="best")

    # 8. Async state ages or explanatory placeholder
    age_cols = [col for col in AGE_CANDIDATES if col in df.columns]
    if age_cols:
        for col in age_cols:
            ax_async.plot(df["t_demo_sec"], df[col], linewidth=1.5, label=col)
        ax_async.set_title("Asynchronous State and Query Ages")
        ax_async.set_ylabel("Age [ms]")
        ax_async.legend(loc="best")
    else:
        ax_async.plot(df["t_demo_sec"], np.zeros(len(df)), linewidth=1.5, label="state/query-age logs not found")
        ax_async.set_title("State/Query Age Logs Missing: Stale-State Hypothesis Cannot Be Tested")
        ax_async.set_ylabel("N/A")
        ax_async.legend(loc="best")

    add_event_lines(axes, events)

    # Event annotations on force panel.
    for event in events:
        y = float(df.loc[event.event_index, "estimated_collision_force_dir"])
        ax_force.scatter([event.event_time_sec], [y], s=55, zorder=6)
        short = "A: force drop" if event.event_name.startswith("event_A") else "B: link-switch spike"
        ax_force.annotate(
            f"{short}\n{event.event_time_sec:.3f} s",
            xy=(event.event_time_sec, y),
            xytext=(8, 12),
            textcoords="offset points",
            fontsize=9,
        )

    for ax in axes:
        ax.grid(True, alpha=GRID_ALPHA)

    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    if show:
        plt.show()
    plt.close(fig)


def create_event_zoom_plot(
    df: pd.DataFrame,
    event: DetectedEvent,
    save_path: str,
    show: bool,
    half_window_sec: float = 0.22,
) -> None:
    w = df[
        (df["t_demo_sec"] >= event.event_time_sec - half_window_sec)
        & (df["t_demo_sec"] <= event.event_time_sec + half_window_sec)
    ].copy()

    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(13, 14), sharex=True)
    ax_force, ax_mass, ax_speed, ax_proxy, ax_detail = axes

    ax_force.plot(w["t_demo_sec"], w["estimated_collision_force_dir"], marker="o", linewidth=1.8, label="directional force")
    if "estimated_collision_force_iso" in w.columns:
        ax_force.plot(w["t_demo_sec"], w["estimated_collision_force_iso"], marker="o", linewidth=1.4, label="ISO force")
    ax_force.set_ylabel("Force [N]")
    ax_force.set_title(f"{event.event_name}: exact adjacent-sample inspection")
    ax_force.legend(loc="best")

    ax_mass.plot(w["t_demo_sec"], w["human_meff_dir"], marker="o", linewidth=1.5, label="human M_eff")
    ax_mass.plot(w["t_demo_sec"], w["robot_meff"], marker="o", linewidth=1.5, label="robot M_eff")
    ax_mass.plot(w["t_demo_sec"], w["reduced_mass_dir"], marker="o", linewidth=1.5, label="reduced mass")
    ax_mass.set_ylabel("Mass [kg]")
    ax_mass.legend(loc="best")

    ax_speed.plot(w["t_demo_sec"], w["selected_link_speed"], marker="o", linewidth=1.5, label="selected-link CoM speed magnitude")
    ax_speed.set_ylabel("Speed [m/s]")
    ax_speed.legend(loc="best")

    ax_proxy.plot(w["t_demo_sec"], w["non_mass_proxy"], marker="o", linewidth=1.5, label="F_dir / sqrt(m_red)")
    ax_proxy.set_ylabel("Proxy")
    ax_proxy.legend(loc="best")

    if "logged_v_app" in w.columns:
        ax_detail.plot(w["t_demo_sec"], w["logged_v_app"], marker="o", linewidth=1.5, label="direct logged v_app")
        if "logged_stiffness" in w.columns:
            ax_detail.plot(w["t_demo_sec"], w["logged_stiffness"], marker="o", linewidth=1.5, label="logged stiffness")
        ax_detail.set_ylabel("Logged term")
        ax_detail.set_title("Direct Model Inputs")
    else:
        link_to_y = {f"link{i}": i for i in range(1, 7)}
        selected_y = w["selected_link_normalized"].map(link_to_y)
        ax_detail.step(w["t_demo_sec"], selected_y, where="post", linewidth=1.8, label="selected link")
        ax_detail.set_yticks(list(link_to_y.values()), list(link_to_y.keys()))
        ax_detail.set_ylabel("Link")
        ax_detail.set_title("Selected Link; direct v_app log missing")
    ax_detail.legend(loc="best")

    for ax in axes:
        ax.axvline(event.event_time_sec, linestyle="--", linewidth=1.5)
        ax.grid(True, alpha=GRID_ALPHA)

    ax_detail.set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    if show:
        plt.show()
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the post-6.31 s force drop and the 7.22 s link-switch force spike."
    )
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH, help="Input demo metrics CSV path")
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR, help="Directory for diagnostic outputs")
    parser.add_argument("--plot-t-start", type=float, default=DEFAULT_PLOT_T_START)
    parser.add_argument("--plot-t-end", type=float, default=DEFAULT_PLOT_T_END)
    parser.add_argument("--drop-search-t-start", type=float, default=DEFAULT_DROP_SEARCH_T_START)
    parser.add_argument("--drop-search-t-end", type=float, default=DEFAULT_DROP_SEARCH_T_END)
    parser.add_argument("--switch-search-t-start", type=float, default=DEFAULT_SWITCH_SEARCH_T_START)
    parser.add_argument("--switch-search-t-end", type=float, default=DEFAULT_SWITCH_SEARCH_T_END)
    parser.add_argument("--strict", action="store_true", help="Fail if direct v_app or stiffness logs are absent")
    parser.add_argument("--no-show", action="store_true", help="Save plots without opening GUI windows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.save_dir)

    raw_df = load_metrics(args.csv)
    df, aliases = add_derived_columns(raw_df)

    plot_df = df[
        (df["t_demo_sec"] >= args.plot_t_start)
        & (df["t_demo_sec"] <= args.plot_t_end)
    ].copy()
    if plot_df.empty:
        raise RuntimeError("No rows in selected overview plot window")

    event_a = detect_largest_negative_force_step(
        df,
        t_start=args.drop_search_t_start,
        t_end=args.drop_search_t_end,
    )
    event_b = detect_link_switch_or_largest_positive_step(
        df,
        t_start=args.switch_search_t_start,
        t_end=args.switch_search_t_end,
    )
    events = [event_a, event_b]

    summaries = [summarize_exact_step(df, event) for event in events]
    summary_df = pd.DataFrame(summaries)

    audit_rows = columns_required_for_definitive_diagnosis(df)
    audit_df = pd.DataFrame(audit_rows)

    # Optional strict mode for exact factor attribution.
    missing_strict = []
    if "logged_v_app" not in df.columns:
        missing_strict.append("v_app")
    if "logged_stiffness" not in df.columns:
        missing_strict.append("selected_body_stiffness")
    if args.strict and missing_strict:
        raise RuntimeError(
            "Strict diagnosis requested, but direct model inputs are missing: "
            + ", ".join(missing_strict)
            + ". Add these logs to the runtime CSV and rerun."
        )

    # Export diagnostics.
    plot_df.to_csv(
        os.path.join(args.save_dir, "force_drop_switch_diagnostic_timeseries.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    summary_df.to_csv(
        os.path.join(args.save_dir, "force_drop_switch_event_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    audit_df.to_csv(
        os.path.join(args.save_dir, "logging_sufficiency_audit.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    exact_indices = sorted({
        event.previous_index for event in events
    } | {
        event.event_index for event in events
    } | {
        event.next_index for event in events
    })
    df.loc[exact_indices].to_csv(
        os.path.join(args.save_dir, "force_drop_switch_exact_samples.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    with open(os.path.join(args.save_dir, "resolved_column_aliases.json"), "w", encoding="utf-8") as f:
        json.dump(aliases, f, ensure_ascii=False, indent=2)

    write_text_report(
        save_path=os.path.join(args.save_dir, "force_drop_switch_report.txt"),
        csv_path=args.csv,
        summaries=summaries,
        audit_df=audit_df,
    )

    create_overview_plot(
        plot_df,
        events=events,
        save_path=os.path.join(args.save_dir, "force_drop_switch_overview.png"),
        show=not args.no_show,
    )
    create_event_zoom_plot(
        df,
        event=event_a,
        save_path=os.path.join(args.save_dir, "event_A_post_6p31_force_drop_zoom.png"),
        show=not args.no_show,
    )
    create_event_zoom_plot(
        df,
        event=event_b,
        save_path=os.path.join(args.save_dir, "event_B_7p22_link_switch_spike_zoom.png"),
        show=not args.no_show,
    )

    # Terminal output.
    print("\n" + "#" * 118)
    print("[COLLISION-FORCE ROOT-CAUSE DIAGNOSTIC]")
    print("#" * 118)
    print("Terminology: the abrupt decrease after ~6.31 s is a force drop/dip/negative step, not a peak.")
    print("The ~7.22 s upward discontinuity is a force spike/positive step.")
    print(f"Input CSV : {args.csv}")
    print(f"Save dir  : {args.save_dir}")

    for summary in summaries:
        print_event_summary(summary)

    print("\n" + "=" * 118)
    print("[LOGGING SUFFICIENCY AUDIT]")
    print("=" * 118)
    for _, row in audit_df.iterrows():
        resolved = row["resolved_column"] if row["resolved_column"] else "-"
        print(f"  [{row['status']:7s}] {row['purpose']}: {resolved}")

    if "logged_v_app" not in df.columns or "logged_stiffness" not in df.columns:
        print("\n[IMPORTANT]")
        print("  Current CSV is insufficient for a mathematically final attribution of the low-level cause.")
        print("  The script has quantified the mass contribution and the combined non-mass contribution.")
        print("  To prove whether the remaining factor is v_app, stiffness, or stale async state,")
        print("  log direct v_app, selected_body_stiffness, vector velocity components, and state/query ages.")
        print("  After adding direct v_app and stiffness logs, rerun with --strict.")

    print("\n[SAVED FILES]")
    for filename in [
        "force_drop_switch_diagnostic_timeseries.csv",
        "force_drop_switch_event_summary.csv",
        "force_drop_switch_exact_samples.csv",
        "logging_sufficiency_audit.csv",
        "resolved_column_aliases.json",
        "force_drop_switch_report.txt",
        "force_drop_switch_overview.png",
        "event_A_post_6p31_force_drop_zoom.png",
        "event_B_7p22_link_switch_spike_zoom.png",
    ]:
        print("  " + os.path.join(args.save_dir, filename))


if __name__ == "__main__":
    main()
