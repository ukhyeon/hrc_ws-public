# Real-Time Human–Robot Collaboration: Human-Side Pipeline

This ROS 2 workspace implements the human-side perception and dynamics pipeline
for a real-time human–robot collision-risk assessment system. It estimates human
pose from camera input, derives dynamics-related quantities, and exchanges the
results with the robot-side controller through custom ROS 2 interfaces.

The robot dynamics, collision assessment, and Indy7 speed-control components are
maintained in [`indy_ws`](https://github.com/ukhyeon/indy_ws).

## System overview

```text
Camera input
    │
    ▼
Human sensing and 3D pose estimation
    │
    ▼
Human dynamics / directional effective mass
    │
    ├──► Collision visualization
    │
    └──► Robot-side collision assessment and speed control (indy_ws)
```

The two workspaces communicate using the messages defined in
`hrc_interfaces`. The intended end-to-end flow is:

1. Detect and track the human body from sensor images.
2. Estimate the human pose and dynamics state.
3. Respond to directional effective-mass queries.
4. Visualize collision candidates received from the robot-side pipeline.

## Packages

### `hrc_interfaces`

Custom message definitions shared by the human and robot pipelines, including:

- human and robot dynamics states;
- directional effective-mass queries and results;
- collision candidates and collision state;
- robot speed-scale commands;
- demo state messages.

### `hrc_body_tracking`

Python ROS 2 package containing:

- camera-image processing and human sensing;
- CLiFF-based 3D human pose inference;
- human dynamics and effective-mass estimation;
- collision-candidate and activation-region visualization;
- performance logging and validation utilities.

The main console entry points are:

| Command | Role |
| --- | --- |
| `sensing_start` | Runs the sensing and pose-estimation pipeline |
| `human_dynamics` | Runs the human dynamics node |
| `human_dynamics_debug` | Runs the validation/debug variant |

## Repository layout

```text
hrc_ws/
├── src/
│   ├── hrc_interfaces/       # Shared ROS 2 messages
│   ├── hrc_body_tracking/    # Perception and human dynamics nodes
│   └── body_models/          # Local body-model assets (not all are redistributable)
├── analysis/                 # Performance and experiment analysis scripts
└── README.md
```

Generated `build/`, `install/`, and `log/` directories are local ROS 2 build
artifacts and are not part of the source distribution.

## Prerequisites

- ROS 2 with `colcon`
- Python 3
- A ROS-compatible image source and `cv_bridge`
- PyTorch and the dependencies required by the included pose-estimation code
- The body-model files and trained weights expected by the local configuration

Some model assets, including SMPL-family data, may be subject to separate
licenses. Obtain and use them according to their original license terms.

## Build

From the workspace root:

```bash
source /opt/ros/<ros-distro>/setup.bash
colcon build --symlink-install
source install/setup.bash
```

Replace `<ros-distro>` with the ROS 2 distribution installed on the system.

## Run

After building and sourcing the workspace, start the required nodes in separate
terminals:

```bash
ros2 run hrc_body_tracking sensing_start
```

```bash
ros2 run hrc_body_tracking human_dynamics
```

When using the complete system, build and source the robot-side
[`indy_ws`](https://github.com/ukhyeon/indy_ws) as well. The robot workspace
consumes `hrc_interfaces`, so this workspace must be built and sourced first.

## Analysis

The `analysis/` directory contains scripts and selected tabular results for:

- real-time latency and processing-rate analysis;
- force and effective-mass comparisons;
- demo-metric visualization;
- validation of collision-risk calculations.

Large raw outputs and generated visualizations are intentionally excluded from
version control.

## Current limitations

- Several runtime paths are currently configured for the original development
  machine and must be adjusted before running on another system.
- Model weights and licensed body-model assets are not fully provisioned by an
  automated installer.
- The full demonstration requires the sensor setup and the robot-side workspace.

## Related repository

- [`indy_ws`](https://github.com/ukhyeon/indy_ws) — Indy7 dynamics, collision
  assessment, and risk-aware speed control

## License and third-party software

Original code in this repository is licensed under the Apache License 2.0.
Portions derived from PhysPT, CLIFF, HRNet, and related projects retain their
original licenses and copyright notices. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for details.

Body-model data and weights are not covered by the repository license. In
particular, SMPL-family software and assets must be obtained separately under
their official terms. The repository imports the separately installed
`patched_smplx` package but does not distribute that dependency or its model
data.
