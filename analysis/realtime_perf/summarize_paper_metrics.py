#!/usr/bin/env python3
import argparse
import csv
import json
import math
import statistics
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_WARMUP_SEC = 10.0
DEFAULT_TAIL_SEC = 5.0
DEFAULT_SESSION_GAP_SEC = 30.0
DEFAULT_MIN_WINDOW_SEC = 5.0


def read_timestamps(path):
    timestamps = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            try:
                timestamp = float(row["t_ros_sec"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(timestamp):
                timestamps.append(timestamp)
    if not timestamps:
        raise ValueError(
            f"No metric rows with valid t_ros_sec in {path}. "
            "The CSV may contain only its header; verify that the node's "
            "PerfCsvLogger.write() call is enabled and run a new session."
        )
    return timestamps


def split_sessions(timestamps, session_gap_sec):
    sessions = []
    start = timestamps[0]
    previous = timestamps[0]
    count = 1
    for timestamp in timestamps[1:]:
        if timestamp < previous or timestamp - previous >= session_gap_sec:
            sessions.append((start, previous, count))
            start = timestamp
            count = 1
        else:
            count += 1
        previous = timestamp
    sessions.append((start, previous, count))
    return sessions


def latest_session_bounds(path, session_gap_sec):
    sessions = split_sessions(read_timestamps(path), session_gap_sec)
    return sessions[-1], sessions


def common_measurement_window(
    paths,
    warmup_sec,
    tail_sec,
    session_gap_sec,
    min_window_sec,
    since=None,
):
    latest_sessions = {}
    all_sessions = {}
    for name, path in paths.items():
        latest, sessions = latest_session_bounds(path, session_gap_sec)
        latest_sessions[name] = latest
        all_sessions[name] = sessions

    common_start = max(start for start, _, _ in latest_sessions.values())
    common_end = min(end for _, end, _ in latest_sessions.values())
    analysis_start = common_start + warmup_sec
    analysis_end = common_end - tail_sec
    if since is not None:
        analysis_start = max(analysis_start, since)

    duration_sec = analysis_end - analysis_start
    if duration_sec < min_window_sec:
        details = "\n".join(
            f"  {name}: [{start:.6f}, {end:.6f}], rows={count}"
            for name, (start, end, count) in latest_sessions.items()
        )
        raise ValueError(
            "The common measurement window is too short after exclusions.\n"
            f"warm-up={warmup_sec:.3f}s, tail={tail_sec:.3f}s\n"
            f"remaining={max(0.0, duration_sec):.3f}s, "
            f"required={min_window_sec:.3f}s\n"
            f"{details}\n"
            "Record a longer session or reduce the exclusion durations."
        )
    return analysis_start, analysis_end, latest_sessions, all_sessions


def read_values(path, column, start_time, end_time, predicate=None):
    values = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            timestamp = float(row["t_ros_sec"])
            if timestamp < start_time or timestamp > end_time:
                continue
            if predicate is not None and not predicate(row):
                continue
            value = float(row[column])
            if math.isfinite(value) and value >= 0.0:
                values.append(value)
    return values


def percentile(values, probability):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(values, include_rate=False):
    if not values:
        return None
    mean = statistics.fmean(values)
    result = {
        "mean_ms": mean,
        "std_ms": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "p95_ms": percentile(values, 0.95),
        "max_ms": max(values),
        "n": len(values),
    }
    if include_rate:
        result["equivalent_processing_rate_hz"] = 1000.0 / mean
    return result


def fmt(value):
    return f"{value:.2f}"


def main():
    parser = argparse.ArgumentParser(
        description="Summarize the paper runtime and online latency metrics."
    )
    parser.add_argument("--since", type=float, default=None)
    parser.add_argument(
        "--warmup-sec", type=float, default=DEFAULT_WARMUP_SEC
    )
    parser.add_argument("--tail-sec", type=float, default=DEFAULT_TAIL_SEC)
    parser.add_argument(
        "--session-gap-sec",
        type=float,
        default=DEFAULT_SESSION_GAP_SEC,
        help="Start a new run when adjacent timestamps differ by this amount.",
    )
    parser.add_argument(
        "--min-window-sec",
        type=float,
        default=DEFAULT_MIN_WINDOW_SEC,
        help="Minimum common duration required after exclusions.",
    )
    parser.add_argument(
        "--output-prefix",
        default=str(BASE_DIR / "paper_metrics"),
    )
    args = parser.parse_args()
    if min(
        args.warmup_sec,
        args.tail_sec,
        args.session_gap_sec,
        args.min_window_sec,
    ) < 0.0:
        raise SystemExit("Time exclusion and session-gap values must be non-negative.")

    files = {
        "sensing": BASE_DIR / "paper_human_sensing_perf.csv",
        "human_dynamics": BASE_DIR / "paper_human_dynamics_perf.csv",
        "robot_dynamics": BASE_DIR / "paper_robot_dynamics_perf.csv",
        "assessment": BASE_DIR / "paper_collision_assessment_perf.csv",
        "speed": BASE_DIR / "paper_robot_speed_command_perf.csv",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise SystemExit("Missing metric files:\n" + "\n".join(missing))

    try:
        window_start, window_end, latest_sessions, all_sessions = (
            common_measurement_window(
                files,
                args.warmup_sec,
                args.tail_sec,
                args.session_gap_sec,
                args.min_window_sec,
                args.since,
            )
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    completed = lambda row: int(row["assessment_id"]) > 0
    commanded = lambda row: int(row["assessment_id"]) > 0

    def values(name, column, predicate=None):
        return read_values(
            files[name], column, window_start, window_end, predicate
        )

    table1 = {
        "Human sensing": summarize(
            values("sensing", "total_operation_ms"), include_rate=True
        ),
        "Human dynamics": summarize(
            values("human_dynamics", "total_operation_ms"), include_rate=True
        ),
        "Robot dynamics": summarize(
            values("robot_dynamics", "total_update_ms"), include_rate=True
        ),
        "Safety assessment": summarize(
            values("assessment", "total_timer_ms", completed), include_rate=True
        ),
        "Robot speed command": summarize(
            values("speed", "set_speed_ratio_ms", commanded), include_rate=True
        ),
    }
    table2 = {
        "Human-M_eff query latency": summarize(
            values("assessment", "human_meff_latency_ms", completed)
        ),
        "Robot-M_eff query latency": summarize(
            values("assessment", "robot_meff_latency_ms", completed)
        ),
        "Candidate-pair assessment latency": summarize(
            values("assessment", "candidate_assessment_latency_ms", completed)
        ),
        "Speed-command call latency": summarize(
            values("speed", "set_speed_ratio_ms", commanded)
        ),
        "End-to-end assessment-to-command latency": summarize(
            values(
                "speed",
                "end_to_end_assessment_to_command_ms",
                commanded,
            )
        ),
    }

    lines = [
        "# Table 1. ROS 2 node operation runtime",
        "",
        "| Node | Operation | Mean [ms] | Std. [ms] | P95 [ms] | Max [ms] | Equivalent Processing Rate [Hz] | N |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    operations = {
        "Human sensing": "Human pose estimation",
        "Human dynamics": "Human dynamics state update",
        "Robot dynamics": "Robot dynamics state update",
        "Safety assessment": "Collision assessment",
        "Robot speed command": "Speed-ratio command call",
    }
    for name, stats in table1.items():
        if stats is not None:
            lines.append(
                f"| {name} | {operations[name]} | {fmt(stats['mean_ms'])} | "
                f"{fmt(stats['std_ms'])} | {fmt(stats['p95_ms'])} | "
                f"{fmt(stats['max_ms'])} | "
                f"{fmt(stats['equivalent_processing_rate_hz'])} | "
                f"{stats['n']} |"
            )

    lines.extend([
        "",
        "# Table 2. Online candidate-pair and command latency",
        "",
        "| Metric | Mean [ms] | P95 [ms] | Max [ms] | N |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for name, stats in table2.items():
        if stats is not None:
            lines.append(
                f"| {name} | {fmt(stats['mean_ms'])} | "
                f"{fmt(stats['p95_ms'])} | {fmt(stats['max_ms'])} | "
                f"{stats['n']} |"
            )

    session_info = {
        name: {
            "selected_start_t_ros_sec": start,
            "selected_end_t_ros_sec": end,
            "selected_rows": count,
            "detected_session_count": len(all_sessions[name]),
        }
        for name, (start, end, count) in latest_sessions.items()
    }
    output = {
        "measurement_window": {
            "start_t_ros_sec": window_start,
            "end_t_ros_sec": window_end,
            "duration_sec": window_end - window_start,
            "warmup_excluded_sec": args.warmup_sec,
            "tail_excluded_sec": args.tail_sec,
            "session_gap_sec": args.session_gap_sec,
            "minimum_window_sec": args.min_window_sec,
            "files": session_info,
        },
        "table1": table1,
        "table2": table2,
    }

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n")
    prefix.with_suffix(".json").write_text(
        json.dumps(output, indent=2) + "\n"
    )

    print("Selected latest contiguous sessions:")
    for name, info in session_info.items():
        print(
            f"  {name}: sessions={info['detected_session_count']}, "
            f"[{info['selected_start_t_ros_sec']:.6f}, "
            f"{info['selected_end_t_ros_sec']:.6f}]"
        )
    print(
        "Common measurement window: "
        f"[{window_start:.6f}, {window_end:.6f}] "
        f"({window_end - window_start:.3f}s)"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
