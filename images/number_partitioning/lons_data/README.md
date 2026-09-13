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
