#!/usr/bin/env python3
"""
Standalone diagnostic script for transient collision-force peaks.

Purpose
-------
Analyze why estimated collision-force spikes occur around a selected time window,
with particular attention to:
1. force peaks that occur while the selected collision link remains unchanged,
2. force peaks that occur near a selected-link transition such as link4 -> link6,
3. cases where robot effective mass decreases but the estimated force increases.

The script is intentionally independent from the existing paper-plot script.
It reads the same demo CSV directly and generates:
- a detailed timeseries CSV for the diagnostic window,
- an event-summary CSV,
- an event-samples CSV containing before / event / after values,
- a multi-panel diagnostic figure,
- terminal reports with model-based interpretations.

Important interpretation note
-----------------------------
For a transient-contact model of the form

    F_est = v_app * sqrt(k * m_red)

where

    m_red = (m_human * m_robot) / (m_human + m_robot),

F_est / sqrt(m_red) is proportional to v_app * sqrt(k).
Therefore, when stiffness k is unchanged, a rise in this proxy indicates that
the force rise cannot be explained by effective-mass variation alone.

The current CSV may not include the actual projected approach velocity v_app,
contact-direction vector u, stiffness, or query timestamps. When such optional
columns exist, this script automatically includes them in reports and plots.
Without those columns, the proxy analysis identifies the likely cause category
but cannot prove the exact low-level source of the spike.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Default settings
# ============================================================

DEFAULT_CSV_PATH = (
    "/home/robotics/hrc_ws/analysis/demo_log/"
    "demo_metrics_head_paper.csv"
)

DEFAULT_SAVE_DIR = (
    "/home/robotics/hrc_ws/analysis/demo_log/"
    "force_peak_diagnostics_head_paper"
)

DEFAULT_T_START = 5.8
DEFAULT_T_END = 7.6
DEFAULT_MIN_PEAK_DISTANCE_SEC = 0.20
DEFAULT_BASELINE_WINDOW_SEC = 0.20
DEFAULT_EVENT_HALF_WINDOW_SEC = 0.035
DEFAULT_AFTER_WINDOW_SEC = 0.20
DEFAULT_TOP_N_PEAKS = 6

# The target peaks seen in the current plots are around 6--7 s.
# Set this to a positive value to suppress small local undulations.
DEFAULT_MIN_PEAK_PROMINENCE_N = 1.0

FIG_TITLE_SIZE = 18
AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 12
LEGEND_SIZE = 10
ANNOTATION_SIZE = 10
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
# Column candidates
# ============================================================

# Actual collision-direction projected approach speed.
V_APP_CANDIDATES = [
    "v_app",
    "v_app_ij",
    "selected_v_app",
    "approach_speed",
    "relative_approach_speed",
    "projected_relative_speed",
    "collision_approach_speed",
]

# Robot and human velocity components projected along the collision direction.
ROBOT_SPEED_ALONG_U_CANDIDATES = [
    "robot_speed_along_u",
    "robot_velocity_along_u",
    "robot_v_along_u",
    "selected_robot_speed_along_u",
]

HUMAN_SPEED_ALONG_U_CANDIDATES = [
    "human_speed_along_u",
    "human_velocity_along_u",
    "human_v_along_u",
    "selected_human_speed_along_u",
]

# Selected collision-direction components.
U_X_CANDIDATES = ["u_x", "collision_u_x", "selected_u_x", "direction_u_x"]
U_Y_CANDIDATES = ["u_y", "collision_u_y", "selected_u_y", "direction_u_y"]
U_Z_CANDIDATES = ["u_z", "collision_u_z", "selected_u_z", "direction_u_z"]

# Human-body stiffness used by the collision model.
STIFFNESS_CANDIDATES = [
    "selected_body_stiffness",
    "body_stiffness",
    "stiffness",
    "spring_constant",
    "k_body",
]

# Pair identity and asynchronous state-age diagnostics.
PAIR_ID_CANDIDATES = [
    "selected_pair_id",
    "collision_pair_id",
    "pair_id",
]

STATE_AGE_CANDIDATES = [
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


# ============================================================
# Data structures
# ============================================================

@dataclass(frozen=True)
class Event:
    event_type: str
    event_label: str
    event_index: int
    event_time: float


# ============================================================
# Generic helpers
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def first_available_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def existing_columns(
    df: pd.DataFrame,
    candidates: Iterable[str],
) -> list[str]:
    return [col for col in candidates if col in df.columns]


def safe_float(value) -> float:
    try:
        if pd.isna(value):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def safe_ratio(numerator, denominator, eps: float = 1e-12) -> float:
    num = safe_float(numerator)
    den = safe_float(denominator)

    if not np.isfinite(num) or not np.isfinite(den):
        return float("nan")

    if abs(den) < eps:
        return float("nan")

    return num / den


def format_value(value, digits: int = 4) -> str:
    if value is None:
        return "NaN"

    if isinstance(value, str):
        return value

    try:
        if pd.isna(value):
            return "NaN"
    except TypeError:
        pass

    if isinstance(value, (float, np.floating, int, np.integer)):
        return f"{float(value):.{digits}f}"

    return str(value)


def normalize_link_name(value) -> str:
    """
    Normalize selected-link values to link1 ... link6 where possible.

    Examples
    --------
    "link4" -> "link4"
    "Link 4" -> "link4"
    4        -> "link4"
    "4.0"    -> "link4"
    ""       -> "unknown"
    """
    if value is None:
        return "unknown"

    try:
        if pd.isna(value):
            return "unknown"
    except TypeError:
        pass

    text = str(value).strip().lower()

    if not text:
        return "unknown"

    match = re.search(r"(?:link\s*)?(\d+)", text)
    if match:
        return f"link{int(match.group(1))}"

    return text.replace(" ", "")


def median_or_nan(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.median())


def max_or_nan(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.max())


def mode_or_unknown(series: pd.Series) -> str:
    values = series.dropna().astype(str)
    if values.empty:
        return "unknown"
    modes = values.mode()
    if modes.empty:
        return str(values.iloc[-1])
    return str(modes.iloc[0])


def nearest_row_index(df: pd.DataFrame, target_time: float) -> int:
    time_error = (df["t_demo_sec"] - target_time).abs()
    return int(time_error.idxmin())


def style_axes(ax) -> None:
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.xaxis.label.set_size(AXIS_LABEL_SIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_SIZE)
    ax.title.set_size(FIG_TITLE_SIZE)


# ============================================================
# Load and preprocess
# ============================================================

def load_demo_metrics(csv_path: str) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    optional_numeric_columns = (
        V_APP_CANDIDATES
        + ROBOT_SPEED_ALONG_U_CANDIDATES
        + HUMAN_SPEED_ALONG_U_CANDIDATES
        + U_X_CANDIDATES
        + U_Y_CANDIDATES
        + U_Z_CANDIDATES
        + STIFFNESS_CANDIDATES
        + STATE_AGE_CANDIDATES
    )

    numeric_columns = list(dict.fromkeys(
        BASE_NUMERIC_COLUMNS + optional_numeric_columns
    ))

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    require_columns(
        df,
        [
            "t_demo_sec",
            "estimated_collision_force_dir",
            "human_meff_dir",
            "robot_meff",
        ],
    )

    df = df.dropna(subset=["t_demo_sec"]).copy()
    df = df.sort_values("t_demo_sec").reset_index(drop=True)

    if df.empty:
        raise RuntimeError("CSV contains no valid rows after preprocessing.")

    return df


def add_selected_link_speed(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["selected_link_normalized"] = (
        result.get("selected_link", pd.Series("unknown", index=result.index))
        .apply(normalize_link_name)
    )

    link_speed_columns = {
        "link1": "link1_speed",
        "link2": "link2_speed",
        "link3": "link3_speed",
        "link4": "link4_speed",
        "link5": "link5_speed",
        "link6": "link6_speed",
    }

    selected_speed = pd.Series(np.nan, index=result.index, dtype=float)

    for link_name, speed_col in link_speed_columns.items():
        if speed_col not in result.columns:
            continue

        mask = result["selected_link_normalized"] == link_name
        selected_speed.loc[mask] = pd.to_numeric(
            result.loc[mask, speed_col],
            errors="coerce",
        )

    result["selected_link_speed"] = selected_speed
    return result


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    result = add_selected_link_speed(df)

    valid_mass_mask = (
        result["human_meff_dir"].notna()
        & result["robot_meff"].notna()
        & (result["human_meff_dir"] > 0.0)
        & (result["robot_meff"] > 0.0)
    )

    result["reduced_mass_dir"] = np.nan
    result.loc[valid_mass_mask, "reduced_mass_dir"] = (
        result.loc[valid_mass_mask, "human_meff_dir"]
        * result.loc[valid_mass_mask, "robot_meff"]
        / (
            result.loc[valid_mass_mask, "human_meff_dir"]
            + result.loc[valid_mass_mask, "robot_meff"]
        )
    )

    result["sqrt_reduced_mass_dir"] = np.sqrt(result["reduced_mass_dir"])

    result["force_per_sqrt_reduced_mass"] = (
        result["estimated_collision_force_dir"]
        / result["sqrt_reduced_mass_dir"]
    )

    # When a stiffness log is available, remove sqrt(k) as well.
    stiffness_col = first_available_column(result, STIFFNESS_CANDIDATES)

    result["velocity_like_proxy"] = result["force_per_sqrt_reduced_mass"]

    if stiffness_col is not None:
        valid_stiffness_mask = result[stiffness_col] > 0.0
        result.loc[valid_stiffness_mask, "velocity_like_proxy"] = (
            result.loc[valid_stiffness_mask, "estimated_collision_force_dir"]
            / np.sqrt(
                result.loc[valid_stiffness_mask, "reduced_mass_dir"]
                * result.loc[valid_stiffness_mask, stiffness_col]
            )
        )

    # Direct optional logs.
    v_app_col = first_available_column(result, V_APP_CANDIDATES)
    robot_u_col = first_available_column(result, ROBOT_SPEED_ALONG_U_CANDIDATES)
    human_u_col = first_available_column(result, HUMAN_SPEED_ALONG_U_CANDIDATES)

    if v_app_col is not None:
        result["detected_v_app"] = result[v_app_col]

    if robot_u_col is not None:
        result["detected_robot_speed_along_u"] = result[robot_u_col]

    if human_u_col is not None:
        result["detected_human_speed_along_u"] = result[human_u_col]

    # Detect selected-link changes.
    result["selected_link_changed"] = (
        result["selected_link_normalized"]
        != result["selected_link_normalized"].shift(1)
    )

    # First row is not a meaningful transition.
    if not result.empty:
        result.loc[result.index[0], "selected_link_changed"] = False

    # First differences for inspection.
    diff_columns = [
        "estimated_collision_force_dir",
        "human_meff_dir",
        "robot_meff",
        "reduced_mass_dir",
        "selected_link_speed",
        "force_per_sqrt_reduced_mass",
        "velocity_like_proxy",
        "speed_scale_target",
        "speed_scale_applied",
        "detected_v_app",
        "detected_robot_speed_along_u",
        "detected_human_speed_along_u",
    ]

    for col in diff_columns:
        if col in result.columns:
            result[f"delta_{col}"] = result[col].diff()

    return result


# ============================================================
# Peak and event detection
# ============================================================

def estimate_median_dt(df: pd.DataFrame) -> float:
    dt = df["t_demo_sec"].diff().dropna()
    dt = dt[dt > 0.0]

    if dt.empty:
        return float("nan")

    return float(dt.median())


def detect_local_force_peaks(
    df: pd.DataFrame,
    t_start: float,
    t_end: float,
    min_distance_sec: float,
    min_prominence_n: float,
    top_n: int,
) -> list[int]:
    """
    Detect local maxima without scipy.

    A lightweight prominence approximation is used:
    the candidate value must exceed the larger of its immediate neighbors
    by at least min_prominence_n. Candidates closer than min_distance_sec
    are deduplicated by retaining the larger force value.
    """
    window = df[
        (df["t_demo_sec"] >= t_start)
        & (df["t_demo_sec"] <= t_end)
    ].copy()

    window = window.dropna(subset=["estimated_collision_force_dir"])

    if len(window) < 3:
        return []

    indices = window.index.to_numpy()
    values = window["estimated_collision_force_dir"].to_numpy(dtype=float)

    candidates: list[int] = []

    for i in range(1, len(window) - 1):
        prev_value = values[i - 1]
        value = values[i]
        next_value = values[i + 1]

        is_local_max = value >= prev_value and value > next_value
        prominence_approx = value - max(prev_value, next_value)

        if is_local_max and prominence_approx >= min_prominence_n:
            candidates.append(int(indices[i]))

    # If no abrupt spike passes the local-neighbor prominence threshold,
    # still retain ordinary local maxima for inspection.
    if not candidates:
        for i in range(1, len(window) - 1):
            prev_value = values[i - 1]
            value = values[i]
            next_value = values[i + 1]

            if value >= prev_value and value > next_value:
                candidates.append(int(indices[i]))

    candidates = sorted(
        candidates,
        key=lambda idx: float(df.loc[idx, "estimated_collision_force_dir"]),
        reverse=True,
    )

    kept: list[int] = []

    for idx in candidates:
        candidate_time = float(df.loc[idx, "t_demo_sec"])

        if all(
            abs(candidate_time - float(df.loc[kept_idx, "t_demo_sec"]))
            >= min_distance_sec
            for kept_idx in kept
        ):
            kept.append(idx)

        if len(kept) >= top_n:
            break

    return sorted(kept, key=lambda idx: float(df.loc[idx, "t_demo_sec"]))


def detect_link_switch_events(
    df: pd.DataFrame,
    t_start: float,
    t_end: float,
) -> list[int]:
    switch_rows = df[
        (df["t_demo_sec"] >= t_start)
        & (df["t_demo_sec"] <= t_end)
        & df["selected_link_changed"]
    ]

    return [int(idx) for idx in switch_rows.index]


def build_events(
    df: pd.DataFrame,
    t_start: float,
    t_end: float,
    min_peak_distance_sec: float,
    min_peak_prominence_n: float,
    top_n_peaks: int,
) -> list[Event]:
    peak_indices = detect_local_force_peaks(
        df=df,
        t_start=t_start,
        t_end=t_end,
        min_distance_sec=min_peak_distance_sec,
        min_prominence_n=min_peak_prominence_n,
        top_n=top_n_peaks,
    )

    switch_indices = detect_link_switch_events(
        df=df,
        t_start=t_start,
        t_end=t_end,
    )

    events: list[Event] = []

    for number, idx in enumerate(peak_indices, start=1):
        events.append(
            Event(
                event_type="force_peak",
                event_label=f"force_peak_{number}",
                event_index=idx,
                event_time=float(df.loc[idx, "t_demo_sec"]),
            )
        )

    for number, idx in enumerate(switch_indices, start=1):
        previous_idx = max(int(idx) - 1, 0)
        previous_link = str(df.loc[previous_idx, "selected_link_normalized"])
        current_link = str(df.loc[idx, "selected_link_normalized"])

        events.append(
            Event(
                event_type="link_switch",
                event_label=(
                    f"link_switch_{number}_{previous_link}_to_{current_link}"
                ),
                event_index=idx,
                event_time=float(df.loc[idx, "t_demo_sec"]),
            )
        )

    return sorted(events, key=lambda event: (event.event_time, event.event_type))


# ============================================================
# Event aggregation and interpretation
# ============================================================

def time_slice(
    df: pd.DataFrame,
    t_min: float,
    t_max: float,
) -> pd.DataFrame:
    return df[
        (df["t_demo_sec"] >= t_min)
        & (df["t_demo_sec"] <= t_max)
    ].copy()


def summarize_numeric_window(
    window: pd.DataFrame,
    col: str,
    aggregation: str,
) -> float:
    if col not in window.columns:
        return float("nan")

    if aggregation == "median":
        return median_or_nan(window[col])

    if aggregation == "max":
        return max_or_nan(window[col])

    raise ValueError(f"Unsupported aggregation: {aggregation}")


def summarize_text_window(window: pd.DataFrame, col: str) -> str:
    if col not in window.columns:
        return "unknown"

    return mode_or_unknown(window[col])


def select_event_columns(df: pd.DataFrame) -> list[str]:
    preferred_columns = [
        "t_demo_sec",
        "selected_link",
        "selected_link_normalized",
        "selected_body",
        "estimated_collision_force_dir",
        "estimated_collision_force_iso",
        "threshold_force",
        "human_meff_dir",
        "human_meff_iso",
        "robot_meff",
        "reduced_mass_dir",
        "sqrt_reduced_mass_dir",
        "selected_link_speed",
        "force_per_sqrt_reduced_mass",
        "velocity_like_proxy",
        "speed_scale_target",
        "speed_scale_applied",
        "detected_v_app",
        "detected_robot_speed_along_u",
        "detected_human_speed_along_u",
    ]

    optional_columns = (
        existing_columns(df, STIFFNESS_CANDIDATES)
        + existing_columns(df, U_X_CANDIDATES)
        + existing_columns(df, U_Y_CANDIDATES)
        + existing_columns(df, U_Z_CANDIDATES)
        + existing_columns(df, PAIR_ID_CANDIDATES)
        + existing_columns(df, STATE_AGE_CANDIDATES)
        + existing_columns(df, QUERY_ID_CANDIDATES)
    )

    ordered = []

    for col in preferred_columns + optional_columns:
        if col in df.columns and col not in ordered:
            ordered.append(col)

    return ordered


def build_event_summary(
    df: pd.DataFrame,
    events: list[Event],
    baseline_window_sec: float,
    event_half_window_sec: float,
    after_window_sec: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_columns = select_event_columns(df)

    numeric_focus_columns = [
        "estimated_collision_force_dir",
        "estimated_collision_force_iso",
        "threshold_force",
        "human_meff_dir",
        "human_meff_iso",
        "robot_meff",
        "reduced_mass_dir",
        "sqrt_reduced_mass_dir",
        "selected_link_speed",
        "force_per_sqrt_reduced_mass",
        "velocity_like_proxy",
        "speed_scale_target",
        "speed_scale_applied",
        "detected_v_app",
        "detected_robot_speed_along_u",
        "detected_human_speed_along_u",
    ]

    numeric_focus_columns += existing_columns(df, STIFFNESS_CANDIDATES)
    numeric_focus_columns += existing_columns(df, U_X_CANDIDATES)
    numeric_focus_columns += existing_columns(df, U_Y_CANDIDATES)
    numeric_focus_columns += existing_columns(df, U_Z_CANDIDATES)
    numeric_focus_columns += existing_columns(df, STATE_AGE_CANDIDATES)

    numeric_focus_columns = list(dict.fromkeys(
        col for col in numeric_focus_columns if col in df.columns
    ))

    text_focus_columns = [
        col for col in [
            "selected_link",
            "selected_link_normalized",
            "selected_body",
        ] + existing_columns(df, PAIR_ID_CANDIDATES) + existing_columns(df, QUERY_ID_CANDIDATES)
        if col in df.columns
    ]

    summary_rows: list[dict] = []
    sample_rows: list[dict] = []

    for event in events:
        event_time = event.event_time

        baseline_window = time_slice(
            df,
            event_time - baseline_window_sec,
            event_time - event_half_window_sec,
        )

        event_window = time_slice(
            df,
            event_time - event_half_window_sec,
            event_time + event_half_window_sec,
        )

        after_window = time_slice(
            df,
            event_time + event_half_window_sec,
            event_time + after_window_sec,
        )

        # Fallbacks for sparse logs.
        if baseline_window.empty:
            baseline_idx = nearest_row_index(df, event_time - baseline_window_sec)
            baseline_window = df.loc[[baseline_idx]].copy()

        if event_window.empty:
            event_window = df.loc[[event.event_index]].copy()

        if after_window.empty:
            after_idx = nearest_row_index(df, event_time + after_window_sec)
            after_window = df.loc[[after_idx]].copy()

        row: dict = {
            "event_type": event.event_type,
            "event_label": event.event_label,
            "event_time_sec": event_time,
            "event_row_index": event.event_index,
        }

        for col in numeric_focus_columns:
            row[f"before_{col}"] = summarize_numeric_window(
                baseline_window,
                col,
                aggregation="median",
            )

            # Use the maximum within the small event window for transient spikes.
            # For mass and speed values, this captures the state associated with
            # the force-spike interval rather than only one arbitrary CSV row.
            row[f"event_{col}"] = summarize_numeric_window(
                event_window,
                col,
                aggregation="max",
            )

            row[f"after_{col}"] = summarize_numeric_window(
                after_window,
                col,
                aggregation="median",
            )

            row[f"ratio_event_to_before_{col}"] = safe_ratio(
                row[f"event_{col}"],
                row[f"before_{col}"],
            )

        for col in text_focus_columns:
            row[f"before_{col}"] = summarize_text_window(baseline_window, col)
            row[f"event_{col}"] = summarize_text_window(event_window, col)
            row[f"after_{col}"] = summarize_text_window(after_window, col)

        observed_force_ratio = safe_ratio(
            row.get("event_estimated_collision_force_dir"),
            row.get("before_estimated_collision_force_dir"),
        )

        mass_only_force_ratio = safe_ratio(
            row.get("event_sqrt_reduced_mass_dir"),
            row.get("before_sqrt_reduced_mass_dir"),
        )

        required_non_mass_ratio = safe_ratio(
            observed_force_ratio,
            mass_only_force_ratio,
        )

        row["observed_force_ratio"] = observed_force_ratio
        row["mass_only_expected_force_ratio"] = mass_only_force_ratio
        row["required_non_mass_factor_ratio"] = required_non_mass_ratio

        # Determine whether a switch occurred inside the event window.
        link_switch_in_event_window = bool(
            event_window["selected_link_changed"].fillna(False).any()
        )
        row["link_switch_in_event_window"] = link_switch_in_event_window

        before_robot_meff = row.get("before_robot_meff", float("nan"))
        event_robot_meff = row.get("event_robot_meff", float("nan"))
        robot_meff_drop = (
            np.isfinite(before_robot_meff)
            and np.isfinite(event_robot_meff)
            and event_robot_meff < before_robot_meff
        )
        row["robot_meff_decreased"] = robot_meff_drop

        row["automatic_interpretation"] = classify_event(row)
        summary_rows.append(row)

        # Store raw before / event / after samples for manual inspection.
        for phase_name, phase_df in [
            ("before", baseline_window),
            ("event", event_window),
            ("after", after_window),
        ]:
            sample = phase_df[event_columns].copy()
            sample.insert(0, "phase", phase_name)
            sample.insert(0, "event_time_sec", event_time)
            sample.insert(0, "event_label", event.event_label)
            sample_rows.extend(sample.to_dict(orient="records"))

    summary_df = pd.DataFrame(summary_rows)
    samples_df = pd.DataFrame(sample_rows)

    return summary_df, samples_df


def classify_event(row: dict) -> str:
    observed_force_ratio = safe_float(row.get("observed_force_ratio"))
    mass_only_ratio = safe_float(row.get("mass_only_expected_force_ratio"))
    required_non_mass_ratio = safe_float(row.get("required_non_mass_factor_ratio"))
    selected_link_speed_ratio = safe_float(
        row.get("ratio_event_to_before_selected_link_speed")
    )

    robot_meff_decreased = bool(row.get("robot_meff_decreased", False))
    link_switch = bool(row.get("link_switch_in_event_window", False))

    direct_v_app_ratio = safe_float(
        row.get("ratio_event_to_before_detected_v_app")
    )

    parts: list[str] = []

    if np.isfinite(observed_force_ratio):
        parts.append(f"observed force ratio={observed_force_ratio:.3f}x")

    if np.isfinite(mass_only_ratio):
        parts.append(f"mass-only expected ratio={mass_only_ratio:.3f}x")

    if np.isfinite(required_non_mass_ratio):
        parts.append(f"required non-mass factor ratio={required_non_mass_ratio:.3f}x")

    if robot_meff_decreased and np.isfinite(observed_force_ratio) and observed_force_ratio > 1.0:
        parts.append(
            "robot M_eff decreased while force increased; mass variation alone cannot explain the spike"
        )

    if link_switch:
        parts.append(
            "selected-link transition occurred inside the event window; inspect collision direction and projected relative speed"
        )

    if np.isfinite(direct_v_app_ratio):
        if direct_v_app_ratio > 1.10:
            parts.append(
                f"logged v_app increased materially ({direct_v_app_ratio:.3f}x)"
            )
        else:
            parts.append(
                f"logged v_app did not increase materially ({direct_v_app_ratio:.3f}x)"
            )

    if np.isfinite(selected_link_speed_ratio):
        if selected_link_speed_ratio > 1.10:
            parts.append(
                f"selected-link CoM speed magnitude increased ({selected_link_speed_ratio:.3f}x)"
            )
        else:
            parts.append(
                f"selected-link CoM speed magnitude changed only modestly ({selected_link_speed_ratio:.3f}x); projected speed may still differ"
            )

    if np.isfinite(required_non_mass_ratio) and required_non_mass_ratio > 1.10:
        parts.append(
            "check v_app, direction vector u, body stiffness, selected body, state ages, and query IDs"
        )
    elif np.isfinite(required_non_mass_ratio):
        parts.append("mass term explains most of the force variation")

    return "; ".join(parts)


# ============================================================
# Terminal report
# ============================================================

def print_available_optional_logs(df: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print("[OPTIONAL DIAGNOSTIC LOG AVAILABILITY]")
    print("=" * 100)

    groups = {
        "v_app": V_APP_CANDIDATES,
        "robot speed along u": ROBOT_SPEED_ALONG_U_CANDIDATES,
        "human speed along u": HUMAN_SPEED_ALONG_U_CANDIDATES,
        "direction u_x": U_X_CANDIDATES,
        "direction u_y": U_Y_CANDIDATES,
        "direction u_z": U_Z_CANDIDATES,
        "stiffness": STIFFNESS_CANDIDATES,
        "pair ID": PAIR_ID_CANDIDATES,
        "state/query age": STATE_AGE_CANDIDATES,
        "query ID": QUERY_ID_CANDIDATES,
    }

    for label, candidates in groups.items():
        found = existing_columns(df, candidates)
        found_text = ", ".join(found) if found else "not logged"
        print(f"{label:24s}: {found_text}")


def print_event_report(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        print("\n[WARN] No diagnostic events were detected.")
        return

    report_columns = [
        "event_type",
        "event_label",
        "event_time_sec",
        "before_selected_link_normalized",
        "event_selected_link_normalized",
        "before_selected_body",
        "event_selected_body",
        "before_estimated_collision_force_dir",
        "event_estimated_collision_force_dir",
        "before_human_meff_dir",
        "event_human_meff_dir",
        "before_robot_meff",
        "event_robot_meff",
        "before_reduced_mass_dir",
        "event_reduced_mass_dir",
        "before_selected_link_speed",
        "event_selected_link_speed",
        "before_velocity_like_proxy",
        "event_velocity_like_proxy",
        "observed_force_ratio",
        "mass_only_expected_force_ratio",
        "required_non_mass_factor_ratio",
        "link_switch_in_event_window",
        "robot_meff_decreased",
        "automatic_interpretation",
    ]

    optional_report_columns = [
        "before_detected_v_app",
        "event_detected_v_app",
        "ratio_event_to_before_detected_v_app",
        "before_detected_robot_speed_along_u",
        "event_detected_robot_speed_along_u",
        "before_detected_human_speed_along_u",
        "event_detected_human_speed_along_u",
    ]

    report_columns += [
        col for col in optional_report_columns if col in summary_df.columns
    ]

    for _, row in summary_df.iterrows():
        print("\n" + "=" * 100)
        print(f"[{row['event_label']}] t = {row['event_time_sec']:.4f} s")
        print("=" * 100)

        for col in report_columns:
            if col not in row.index:
                continue

            label = col.replace("_", " ")
            print(f"{label:46s}: {format_value(row[col])}")


def print_key_conclusion(summary_df: pd.DataFrame) -> None:
    print("\n" + "#" * 100)
    print("[KEY CONCLUSION GUIDE]")
    print("#" * 100)

    if summary_df.empty:
        print("No events were available for interpretation.")
        return

    critical_rows = summary_df[
        summary_df["robot_meff_decreased"].fillna(False)
        & (summary_df["observed_force_ratio"] > 1.0)
    ]

    if critical_rows.empty:
        print(
            "No detected event simultaneously showed a robot M_eff decrease and a directional-force increase."
        )
    else:
        print(
            "At least one detected event showed a robot M_eff decrease while the directional estimated force increased."
        )
        print(
            "For that event, robot effective mass is not the cause of the upward spike. It acts in the opposite direction."
        )
        print(
            "Inspect the required_non_mass_factor_ratio column. Under the transient-contact model assumption,"
        )
        print(
            "this is the multiplicative increase that must come from projected approach speed, stiffness, or a logging/state mismatch."
        )

    if "ratio_event_to_before_detected_v_app" not in summary_df.columns:
        print(
            "\nThe CSV does not contain a directly detected v_app column. The script therefore uses a proxy analysis."
        )
        print(
            "To prove the exact source, add v_app, u_x/u_y/u_z, projected robot/human velocities, state ages, and query IDs to the logger."
        )


# ============================================================
# Plotting
# ============================================================

def add_event_lines(
    axes: Iterable,
    events: list[Event],
) -> None:
    for event in events:
        linestyle = "--" if event.event_type == "link_switch" else ":"
        alpha = 0.85 if event.event_type == "link_switch" else 0.65

        for ax in axes:
            ax.axvline(
                event.event_time,
                linestyle=linestyle,
                linewidth=1.4,
                alpha=alpha,
            )


def plot_diagnostics(
    df: pd.DataFrame,
    events: list[Event],
    t_start: float,
    t_end: float,
    save_dir: str,
    show_plot: bool,
) -> str:
    window = time_slice(df, t_start, t_end)

    if window.empty:
        raise RuntimeError(
            f"No rows found in diagnostic window: {t_start:.3f}--{t_end:.3f} s"
        )

    fig, axes = plt.subplots(
        nrows=7,
        ncols=1,
        figsize=(15, 23),
        sharex=True,
    )

    ax_force, ax_mass, ax_speed, ax_proxy, ax_scale, ax_link, ax_optional = axes

    # --------------------------------------------------------
    # Panel 1: collision force
    # --------------------------------------------------------
    ax_force.plot(
        window["t_demo_sec"],
        window["estimated_collision_force_dir"],
        linewidth=LINE_WIDTH,
        label="directional estimated force",
    )

    if "estimated_collision_force_iso" in window.columns:
        ax_force.plot(
            window["t_demo_sec"],
            window["estimated_collision_force_iso"],
            linewidth=LINE_WIDTH,
            label="ISO fixed-mass estimated force",
        )

    if "threshold_force" in window.columns:
        ax_force.plot(
            window["t_demo_sec"],
            window["threshold_force"],
            linewidth=LINE_WIDTH,
            label="threshold force",
        )

    for event in events:
        if event.event_type != "force_peak":
            continue

        force = safe_float(df.loc[event.event_index, "estimated_collision_force_dir"])

        if np.isfinite(force):
            ax_force.scatter(
                [event.event_time],
                [force],
                s=55,
                zorder=6,
            )

            ax_force.annotate(
                f"{event.event_time:.2f} s",
                xy=(event.event_time, force),
                xytext=(5, 7),
                textcoords="offset points",
                fontsize=ANNOTATION_SIZE,
            )

    ax_force.set_ylabel("Force [N]")
    ax_force.set_title("Directional Collision-Force Peaks")
    ax_force.grid(True, alpha=GRID_ALPHA)
    ax_force.legend(loc="best", framealpha=0.9)

    # --------------------------------------------------------
    # Panel 2: effective mass terms
    # --------------------------------------------------------
    ax_mass.plot(
        window["t_demo_sec"],
        window["human_meff_dir"],
        linewidth=LINE_WIDTH,
        label="human directional M_eff",
    )

    ax_mass.plot(
        window["t_demo_sec"],
        window["robot_meff"],
        linewidth=LINE_WIDTH,
        label="robot M_eff",
    )

    ax_mass.plot(
        window["t_demo_sec"],
        window["reduced_mass_dir"],
        linewidth=LINE_WIDTH,
        label="directional reduced mass",
    )

    ax_mass.set_ylabel("Mass [kg]")
    ax_mass.set_title("Effective-Mass Terms")
    ax_mass.grid(True, alpha=GRID_ALPHA)
    ax_mass.legend(loc="best", framealpha=0.9)

    # --------------------------------------------------------
    # Panel 3: selected-link speed magnitude
    # --------------------------------------------------------
    ax_speed.plot(
        window["t_demo_sec"],
        window["selected_link_speed"],
        linewidth=LINE_WIDTH,
        label="selected-link CoM speed magnitude",
    )

    ax_speed.set_ylabel("Speed [m/s]")
    ax_speed.set_title(
        "Selected-Link CoM Speed Magnitude "
        "(not collision-direction projected speed)"
    )
    ax_speed.grid(True, alpha=GRID_ALPHA)
    ax_speed.legend(loc="best", framealpha=0.9)

    # --------------------------------------------------------
    # Panel 4: mass-normalized force proxy
    # --------------------------------------------------------
    ax_proxy.plot(
        window["t_demo_sec"],
        window["force_per_sqrt_reduced_mass"],
        linewidth=LINE_WIDTH,
        label="F_dir / sqrt(m_red)",
    )

    if "velocity_like_proxy" in window.columns:
        stiffness_col = first_available_column(window, STIFFNESS_CANDIDATES)

        if stiffness_col is not None:
            ax_proxy.plot(
                window["t_demo_sec"],
                window["velocity_like_proxy"],
                linewidth=LINE_WIDTH,
                label="F_dir / sqrt(k * m_red)",
            )

    ax_proxy.set_ylabel("Proxy value")
    ax_proxy.set_title("Non-Mass Contribution Proxy")
    ax_proxy.grid(True, alpha=GRID_ALPHA)
    ax_proxy.legend(loc="best", framealpha=0.9)

    # --------------------------------------------------------
    # Panel 5: target and applied speed scales
    # --------------------------------------------------------
    if "speed_scale_target" in window.columns:
        ax_scale.plot(
            window["t_demo_sec"],
            window["speed_scale_target"],
            linewidth=LINE_WIDTH,
            label="target speed scale",
        )

    if "speed_scale_applied" in window.columns:
        ax_scale.plot(
            window["t_demo_sec"],
            window["speed_scale_applied"],
            linewidth=LINE_WIDTH,
            label="applied speed scale",
        )

    ax_scale.set_ylabel("Speed scale")
    ax_scale.set_title("Speed-Scale Response")
    ax_scale.grid(True, alpha=GRID_ALPHA)
    ax_scale.legend(loc="best", framealpha=0.9)

    # --------------------------------------------------------
    # Panel 6: selected collision link
    # --------------------------------------------------------
    link_order = ["link1", "link2", "link3", "link4", "link5", "link6"]
    link_to_num = {name: i + 1 for i, name in enumerate(link_order)}
    selected_link_y = window["selected_link_normalized"].map(link_to_num)

    ax_link.step(
        window["t_demo_sec"],
        selected_link_y,
        where="post",
        linewidth=LINE_WIDTH,
        label="selected collision link",
    )

    ax_link.set_ylabel("Selected link")
    ax_link.set_title("Selected Collision Link")
    ax_link.set_yticks(list(link_to_num.values()))
    ax_link.set_yticklabels(list(link_to_num.keys()))
    ax_link.grid(True, alpha=GRID_ALPHA)
    ax_link.legend(loc="best", framealpha=0.9)

    # --------------------------------------------------------
    # Panel 7: direct optional diagnostics
    # --------------------------------------------------------
    plotted_optional = False

    optional_plot_columns = [
        ("detected_v_app", "logged v_app"),
        ("detected_robot_speed_along_u", "robot speed along u"),
        ("detected_human_speed_along_u", "human speed along u"),
    ]

    for col, label in optional_plot_columns:
        if col in window.columns:
            ax_optional.plot(
                window["t_demo_sec"],
                window[col],
                linewidth=LINE_WIDTH,
                label=label,
            )
            plotted_optional = True

    if plotted_optional:
        ax_optional.set_ylabel("Projected speed")
        ax_optional.set_title("Direct Projected-Velocity Logs")
        ax_optional.legend(loc="best", framealpha=0.9)
    else:
        ax_optional.plot(
            window["t_demo_sec"],
            window["velocity_like_proxy"],
            linewidth=LINE_WIDTH,
            label="velocity-like proxy",
        )
        ax_optional.set_ylabel("Proxy value")
        ax_optional.set_title(
            "Direct v_app Log Not Found: Velocity-Like Proxy Repeated for Inspection"
        )
        ax_optional.legend(loc="best", framealpha=0.9)

    ax_optional.set_xlabel("Time [s]")
    ax_optional.grid(True, alpha=GRID_ALPHA)

    add_event_lines(axes, events)

    for ax in axes:
        style_axes(ax)

    plt.tight_layout()

    save_path = os.path.join(save_dir, "force_peak_diagnostic_vs_time.png")
    plt.savefig(save_path, dpi=220)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return save_path


# ============================================================
# Save CSV outputs
# ============================================================

def save_csv_outputs(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    samples_df: pd.DataFrame,
    t_start: float,
    t_end: float,
    save_dir: str,
) -> tuple[str, str, str]:
    window = time_slice(df, t_start, t_end)

    timeseries_path = os.path.join(
        save_dir,
        "force_peak_diagnostic_timeseries.csv",
    )

    summary_path = os.path.join(
        save_dir,
        "force_peak_event_summary.csv",
    )

    samples_path = os.path.join(
        save_dir,
        "force_peak_event_samples.csv",
    )

    window.to_csv(timeseries_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    samples_df.to_csv(samples_path, index=False, encoding="utf-8-sig")

    return timeseries_path, summary_path, samples_path


# ============================================================
# Argument parser and main
# ============================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose transient directional collision-force peaks using "
            "demo_metrics CSV logs."
        )
    )

    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV_PATH,
        help=f"Input CSV path. Default: {DEFAULT_CSV_PATH}",
    )

    parser.add_argument(
        "--save-dir",
        default=DEFAULT_SAVE_DIR,
        help=f"Output directory. Default: {DEFAULT_SAVE_DIR}",
    )

    parser.add_argument(
        "--t-start",
        type=float,
        default=DEFAULT_T_START,
        help=f"Diagnostic-window start time [s]. Default: {DEFAULT_T_START}",
    )

    parser.add_argument(
        "--t-end",
        type=float,
        default=DEFAULT_T_END,
        help=f"Diagnostic-window end time [s]. Default: {DEFAULT_T_END}",
    )

    parser.add_argument(
        "--min-peak-distance",
        type=float,
        default=DEFAULT_MIN_PEAK_DISTANCE_SEC,
        help=(
            "Minimum temporal separation between retained local force peaks [s]. "
            f"Default: {DEFAULT_MIN_PEAK_DISTANCE_SEC}"
        ),
    )

    parser.add_argument(
        "--min-peak-prominence",
        type=float,
        default=DEFAULT_MIN_PEAK_PROMINENCE_N,
        help=(
            "Approximate immediate-neighbor force prominence threshold [N]. "
            f"Default: {DEFAULT_MIN_PEAK_PROMINENCE_N}"
        ),
    )

    parser.add_argument(
        "--top-n-peaks",
        type=int,
        default=DEFAULT_TOP_N_PEAKS,
        help=f"Maximum number of force peaks to retain. Default: {DEFAULT_TOP_N_PEAKS}",
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the plot without opening an interactive matplotlib window.",
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.t_end <= args.t_start:
        raise ValueError("--t-end must be larger than --t-start.")

    if args.min_peak_distance < 0.0:
        raise ValueError("--min-peak-distance must be non-negative.")

    if args.min_peak_prominence < 0.0:
        raise ValueError("--min-peak-prominence must be non-negative.")

    if args.top_n_peaks <= 0:
        raise ValueError("--top-n-peaks must be positive.")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(args)

    ensure_dir(args.save_dir)

    df = load_demo_metrics(args.csv)
    df = add_derived_metrics(df)

    dt_median = estimate_median_dt(df)

    print("=" * 100)
    print("[FORCE PEAK DIAGNOSTIC ANALYSIS]")
    print("=" * 100)
    print(f"Input CSV                     : {args.csv}")
    print(f"Output directory              : {args.save_dir}")
    print(f"Total rows                    : {len(df)}")
    print(f"Full time range               : {df['t_demo_sec'].min():.4f}--{df['t_demo_sec'].max():.4f} s")
    print(f"Diagnostic time range         : {args.t_start:.4f}--{args.t_end:.4f} s")
    print(f"Median logging interval       : {format_value(dt_median, digits=6)} s")
    print(f"Approx. logging rate          : {format_value(safe_ratio(1.0, dt_median), digits=3)} Hz")

    print_available_optional_logs(df)

    events = build_events(
        df=df,
        t_start=args.t_start,
        t_end=args.t_end,
        min_peak_distance_sec=args.min_peak_distance,
        min_peak_prominence_n=args.min_peak_prominence,
        top_n_peaks=args.top_n_peaks,
    )

    summary_df, samples_df = build_event_summary(
        df=df,
        events=events,
        baseline_window_sec=DEFAULT_BASELINE_WINDOW_SEC,
        event_half_window_sec=DEFAULT_EVENT_HALF_WINDOW_SEC,
        after_window_sec=DEFAULT_AFTER_WINDOW_SEC,
    )

    print_event_report(summary_df)
    print_key_conclusion(summary_df)

    timeseries_path, summary_path, samples_path = save_csv_outputs(
        df=df,
        summary_df=summary_df,
        samples_df=samples_df,
        t_start=args.t_start,
        t_end=args.t_end,
        save_dir=args.save_dir,
    )

    figure_path = plot_diagnostics(
        df=df,
        events=events,
        t_start=args.t_start,
        t_end=args.t_end,
        save_dir=args.save_dir,
        show_plot=not args.no_show,
    )

    print("\n" + "=" * 100)
    print("[SAVED OUTPUTS]")
    print("=" * 100)
    print(f"Diagnostic timeseries CSV     : {timeseries_path}")
    print(f"Event summary CSV             : {summary_path}")
    print(f"Event samples CSV             : {samples_path}")
    print(f"Diagnostic figure             : {figure_path}")


if __name__ == "__main__":
    main()
