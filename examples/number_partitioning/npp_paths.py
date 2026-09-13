from pathlib import Path

import pandas as pd

IMAGES_DIR = "images/number_partitioning"

# Raw sampling traces are written next to the figures they reproduce.
LONS_DATA_DIR = "images/number_partitioning/lons_data"
SWEEP_DATA_DIR = "images/number_partitioning/sweep_data"


_TRACE_README = """\
# Raw sampling traces

Each `*.csv` is the raw Iterated Local Search (ILS) sampling trace for one NPP
instance (identified by the hardness parameter `k`), with columns
`[run, fit1, node1, fit2, node2]`. The trace is the reproducible source for every
figure and metric: rebuild the LON/CMLON and reproduce the results with a few lines.

```python
import pandas as pd
from lonkit import LON, LONVisualizer

trace = pd.read_csv("NPP_k0.300.csv")   # pick any file in this folder
lon = LON.from_trace_data(trace)
cmlon = lon.to_cmlon()
print(cmlon.compute_metrics())          # matches metrics.csv
LONVisualizer().plot_2d(cmlon)
```

`metrics.csv` (when present) lists the computed metrics per `k`.
"""


def save_traces(traces: dict[str, pd.DataFrame], data_dir: str | Path) -> None:
    """Write one CSV per raw sampling trace, plus a README explaining reuse."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for stem, trace_df in traces.items():
        path = data_dir / f"{stem}.csv"
        trace_df.to_csv(path, index=False)
        print(f"Saved {path}")
    (data_dir / "README.md").write_text(_TRACE_README, encoding="utf-8")


def save_metrics_csv(rows: list[dict], output_path: str | Path) -> None:
    """Write a metrics.csv summary with one row per k."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved {output_path}")
