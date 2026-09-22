# Paper Runtime Metrics

Start a clean measurement session:

```bash
python3 analysis/realtime_perf/reset_paper_metrics.py
```

Run the ROS 2 pipeline, then generate both paper tables:

```bash
python3 analysis/realtime_perf/summarize_paper_metrics.py
```

Definitions:

- Node runtime: wall-clock time spent inside the corresponding operation.
- Human/robot effective-mass latency: query publication to result callback receipt.
  The reported value is the slowest candidate response in each completed top-k batch.
- Candidate-pair latency: batch query publication to completion of final assessment.
- Speed-command latency: duration of the `SetSpeedRatio` RPC.
- End-to-end latency: batch query publication to completion of that RPC.
- P95: linearly interpolated 0.95 quantile.

Only `paper_*_perf.csv` files are reset. Other experiment and debug logs are untouched.
