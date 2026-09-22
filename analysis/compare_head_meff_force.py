#!/usr/bin/env python3
"""
Compare ISO fixed head mass and proposed head effective mass using the
same estimated collision-force equation as collision_assessment_node.

Formula:
    F_est = v_app * sqrt(m_reduced * k)

where:
    k = ISO stiffness [kN/m] * 1000 [N/m]
    m_reduced = (m_human * m_robot) / (m_human + m_robot)

If --robot-meff is not provided, the script uses m_reduced = m_human. This is
equivalent to assuming a very large robot-side effective mass.

The approach speed is not user-provided. It is computed as the speed at which
the ISO fixed head mass reaches the ISO head force limit:

    v_app = F_max / sqrt(m_reduced_iso * k)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable


DEFAULT_ISO_JSON = Path(
    "/home/robotics/indy_ws/src/indy_control_cpp/ISO15066/iso_15066.json"
)


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def load_iso_head_limits(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "head" not in data:
        raise KeyError(f"'head' entry not found in ISO JSON: {path}")

    head = data["head"]
    for key in ("m_h_iso", "k", "F_max"):
        if key not in head:
            raise KeyError(f"'{key}' missing in ISO head entry: {path}")

    return head


def reduced_mass(human_meff: float, robot_meff: float | None) -> float:
    if robot_meff is None or math.isinf(robot_meff):
        return human_meff
    eps = 1e-9
    human_meff = max(human_meff, eps)
    robot_meff = max(robot_meff, eps)
    return 1.0 / (1.0 / human_meff + 1.0 / robot_meff)


def estimate_collision_force(
    v_app: float,
    human_meff: float,
    robot_meff: float | None,
    k_kn_per_m: float,
) -> float:
    if v_app <= 0.0:
        return 0.0
    mu = reduced_mass(human_meff, robot_meff)
    k_n_per_m = k_kn_per_m * 1e3
    return v_app * math.sqrt(max(mu * k_n_per_m, 1e-9))


def compute_v_app_for_force_limit(
    force_limit: float,
    human_meff: float,
    robot_meff: float | None,
    k_kn_per_m: float,
) -> float:
    mu = reduced_mass(human_meff, robot_meff)
    k_n_per_m = k_kn_per_m * 1e3
    denom = math.sqrt(max(mu * k_n_per_m, 1e-9))
    return force_limit / denom


def format_markdown_table(headers: list[str], rows: Iterable[list[str]]) -> str:
    align = ["---"] + ["---:" for _ in headers[1:]]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(align) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_rows(args: argparse.Namespace, iso_head: dict) -> list[list[str]]:
    iso_head_mass = float(iso_head["m_h_iso"])
    k_kn_per_m = float(iso_head["k"])
    force_limit = float(iso_head["F_max"])
    proposed_head_mass = float(args.head_meff)

    v_app = compute_v_app_for_force_limit(
        force_limit=force_limit,
        human_meff=iso_head_mass,
        robot_meff=args.robot_meff,
        k_kn_per_m=k_kn_per_m,
    )
    f_iso = estimate_collision_force(
        v_app=v_app,
        human_meff=iso_head_mass,
        robot_meff=args.robot_meff,
        k_kn_per_m=k_kn_per_m,
    )
    f_proposed = estimate_collision_force(
        v_app=v_app,
        human_meff=proposed_head_mass,
        robot_meff=args.robot_meff,
        k_kn_per_m=k_kn_per_m,
    )
    force_increase = (f_proposed - f_iso) / max(abs(f_iso), 1e-9) * 100.0

    return [[
        args.scenario,
        f"{v_app:.3f}",
        f"{iso_head_mass:.3f}",
        f"{proposed_head_mass:.3f}",
        f"{f_iso:.3f}",
        f"{f_proposed:.3f}",
        f"{force_increase:.2f}",
    ]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print ISO fixed head mass vs proposed head effective-mass "
            "force comparison using the current ISO15066 head stiffness."
        )
    )
    parser.add_argument(
        "head_meff",
        type=positive_float,
        help="Proposed head effective mass [kg].",
    )
    parser.add_argument(
        "--robot-meff",
        type=positive_float,
        default=None,
        help=(
            "Robot effective mass [kg]. If omitted, uses m_reduced = "
            "human_meff, matching a very large robot effective mass assumption."
        ),
    )
    parser.add_argument(
        "--scenario",
        default="Head",
        help="Scenario label printed in the table. Default: Head",
    )
    parser.add_argument(
        "--iso-json",
        type=Path,
        default=DEFAULT_ISO_JSON,
        help=f"ISO JSON path. Default: {DEFAULT_ISO_JSON}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    iso_head = load_iso_head_limits(args.iso_json)

    headers = [
        "Scenario",
        "v_app [m/s]",
        "ISO fixed head mass [kg]",
        "Proposed head effective mass [kg]",
        "F_est using ISO mass [N]",
        "F_est using proposed mass [N]",
        "Force increase [%]",
    ]
    rows = build_rows(args, iso_head)

    print(format_markdown_table(headers, rows))
    print()
    print(
        "Formula: F_est = v_app * sqrt(m_reduced * k), "
        f"k = {float(iso_head['k']):.3f} kN/m for ISO head."
    )
    print(
        "v_app is computed from the ISO fixed head mass so that "
        f"F_est using ISO mass equals ISO head F_max = {float(iso_head['F_max']):.3f} N."
    )
    if args.robot_meff is None:
        print("Reduced mass: m_reduced = head mass (robot effective mass omitted).")
    else:
        print(f"Reduced mass: m_reduced = m_head * {args.robot_meff:.3f} / (m_head + {args.robot_meff:.3f}).")


if __name__ == "__main__":
    main()
