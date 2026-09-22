import os
import pandas as pd
import numpy as np


BASE_DIR = "/home/robotics/hrc_ws/analysis/realtime_perf"
OUT_DIR = "/home/robotics/hrc_ws/analysis/realtime_tables"
os.makedirs(OUT_DIR, exist_ok=True)

WARMUP_SEC = 10.0


def load_and_filter_csv(file_name):
    """
    Load a performance CSV file and remove the initial warm-up period.

    The warm-up period is excluded to avoid camera initialization,
    model loading, and GPU warm-up outliers.
    """
    path = os.path.join(BASE_DIR, file_name)

    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)

    if len(df) == 0:
        return df

    if "t_wall_sec" in df.columns:
        t_wall = pd.to_numeric(df["t_wall_sec"], errors="coerce")
        t0 = t_wall.min()
        df = df[t_wall >= t0 + WARMUP_SEC]

    return df


def empty_runtime_summary(node, operation):
    return {
        "Node": node,
        "Operation": operation,
        "Mean [ms]": np.nan,
        "Std. [ms]": np.nan,
        "P95 [ms]": np.nan,
        "Max [ms]": np.nan,
        "Mean Processing Rate [Hz]": np.nan,
        "N": 0,
    }


def summarize_csv(
    file_name,
    node,
    operation,
    value_col,
    filter_condition=None,
):
    """
    Summarize one runtime latency column.

    Mean Processing Rate [Hz] is computed as:
        1000 / Mean latency [ms]

    For timer-driven nodes, this value indicates processing capacity
    based on execution time, not the actual trigger frequency.
    """
    df = load_and_filter_csv(file_name)

    if df is None or len(df) == 0 or value_col not in df.columns:
        return empty_runtime_summary(node, operation)

    if filter_condition is not None:
        df = filter_condition(df)

    x = pd.to_numeric(df[value_col], errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).dropna()

    # Negative latency means invalid or unavailable.
    x = x[x >= 0]

    if len(x) == 0:
        return empty_runtime_summary(node, operation)

    mean_ms = float(x.mean())
    mean_rate_hz = 1000.0 / mean_ms if mean_ms > 0 else np.nan

    return {
        "Node": node,
        "Operation": operation,
        "Mean [ms]": mean_ms,
        "Std. [ms]": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        "P95 [ms]": float(x.quantile(0.95)),
        "Max [ms]": float(x.max()),
        "Mean Processing Rate [Hz]": mean_rate_hz,
        "N": int(len(x)),
    }


def make_runtime_table():
    """
    Table X:
    Runtime latency and mean processing rate for the five ROS 2 nodes.

    One representative process is reported for each node.
    """
    rows = []

    rows.append(summarize_csv(
        file_name="sensing_perf.csv",
        node="Human sensing node",
        operation="Human pose estimation",
        value_col="total_loop_ms",
    ))

    rows.append(summarize_csv(
        file_name="human_dynamics_update_perf.csv",
        node="Human dynamics node",
        operation="Human dynamics state update",
        value_col="total_update_ms",
    ))

    rows.append(summarize_csv(
        file_name="robot_dynamics_update_perf.csv",
        node="Robot dynamics node",
        operation="Robot dynamics state update",
        value_col="total_update_ms",
    ))

    rows.append(summarize_csv(
        file_name="collision_assessment_perf.csv",
        node="Safety assessment node",
        operation="Collision assessment and speed scale computation",
        value_col="total_timer_ms",
    ))

    rows.append(summarize_csv(
        file_name="robot_speed_command_perf.csv",
        node="Robot speed command node",
        operation="Speed ratio command call",
        value_col="set_speed_ratio_ms",
    ))

    df = pd.DataFrame(rows)

    numeric_cols = [
        "Mean [ms]",
        "Std. [ms]",
        "P95 [ms]",
        "Max [ms]",
        "Mean Processing Rate [Hz]",
    ]

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").round(2)

    df.to_csv(
        os.path.join(OUT_DIR, "table_runtime_latency_processing_rate.csv"),
        index=False,
    )
    df.to_csv(
        os.path.join(OUT_DIR, "table_runtime_latency_processing_rate.tsv"),
        index=False,
        sep="\t",
    )

    return df


def summarize_latency_only(
    file_name,
    metric,
    value_col,
    filter_condition=None,
):
    """
    Summarize a latency metric for Table Y.
    """
    df = load_and_filter_csv(file_name)

    if df is None or len(df) == 0 or value_col not in df.columns:
        return {
            "Metric": metric,
            "Mean [ms]": np.nan,
            "P95 [ms]": np.nan,
            "Max [ms]": np.nan,
            "Rate [Hz]": np.nan,
            "N": 0,
        }

    if filter_condition is not None:
        df = filter_condition(df)

    x = pd.to_numeric(df[value_col], errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    x = x[x >= 0]

    if len(x) == 0:
        return {
            "Metric": metric,
            "Mean [ms]": np.nan,
            "P95 [ms]": np.nan,
            "Max [ms]": np.nan,
            "Rate [Hz]": np.nan,
            "N": 0,
        }

    mean_ms = float(x.mean())
    rate_hz = 1000.0 / mean_ms if mean_ms > 0 else np.nan

    return {
        "Metric": metric,
        "Mean [ms]": mean_ms,
        "P95 [ms]": float(x.quantile(0.95)),
        "Max [ms]": float(x.max()),
        "Rate [Hz]": rate_hz,
        "N": int(len(x)),
    }


def safe_int(value):
    if pd.isna(value):
        return 0
    return int(value)


def make_safety_command_table():
    """
    Table Y:
    Safety command update latency.

    This table is not a node-by-node runtime table.
    It summarizes the latency from human–robot effective mass result completion
    to the speed ratio command call.

    Safety decision to command latency is computed as:
        Human–robot effective mass result completion
        + Robot speed command transmission

    The P95 and Max values are conservative approximations obtained by summing
    the corresponding statistics of the two components.
    """

    def successful_meff_response_only(df):
        if "query_timeout" in df.columns:
            return df[pd.to_numeric(df["query_timeout"], errors="coerce") == 0]
        return df

    meff_row = summarize_latency_only(
        file_name="collision_assessment_perf.csv",
        metric="Human–robot effective mass result completion",
        value_col="batch_query_latency_ms",
        filter_condition=successful_meff_response_only,
    )

    speed_row = summarize_latency_only(
        file_name="robot_speed_command_perf.csv",
        metric="Robot speed command transmission",
        value_col="set_speed_ratio_ms",
    )

    decision_mean = meff_row["Mean [ms]"] + speed_row["Mean [ms]"]
    decision_p95 = meff_row["P95 [ms]"] + speed_row["P95 [ms]"]
    decision_max = meff_row["Max [ms]"] + speed_row["Max [ms]"]
    decision_rate = 1000.0 / decision_mean if decision_mean > 0 else np.nan
    decision_n = min(safe_int(meff_row["N"]), safe_int(speed_row["N"]))

    decision_row = {
        "Metric": "Safety decision to command latency",
        "Mean [ms]": decision_mean,
        "P95 [ms]": decision_p95,
        "Max [ms]": decision_max,
        "Rate [Hz]": decision_rate,
        "N": decision_n,
    }

    df = pd.DataFrame([
        meff_row,
        speed_row,
        decision_row,
    ])

    numeric_cols = [
        "Mean [ms]",
        "P95 [ms]",
        "Max [ms]",
        "Rate [Hz]",
    ]

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").round(2)

    df.to_csv(
        os.path.join(OUT_DIR, "table_safety_command_update_latency.csv"),
        index=False,
    )
    df.to_csv(
        os.path.join(OUT_DIR, "table_safety_command_update_latency.tsv"),
        index=False,
        sep="\t",
    )

    return df


if __name__ == "__main__":
    runtime_table = make_runtime_table()
    safety_command_table = make_safety_command_table()

    print("\n[Table X] Runtime latency and mean processing rate after warm-up exclusion")
    print(runtime_table.to_string(index=False))

    print("\n[Table Y] Safety command update latency")
    print(safety_command_table.to_string(index=False))

    print(f"\nSaved to: {OUT_DIR}")