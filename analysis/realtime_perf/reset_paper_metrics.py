#!/usr/bin/env python3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
for path in BASE_DIR.glob("paper_*_perf.csv"):
    path.unlink()
    print(f"removed {path}")

for path in (BASE_DIR / "paper_metrics.md", BASE_DIR / "paper_metrics.json"):
    if path.exists():
        path.unlink()
        print(f"removed {path}")