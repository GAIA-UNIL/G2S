import numpy as np
from g2s import g2s


rng = np.random.default_rng(4)
y, x = np.mgrid[0:96, 0:128].astype(np.float32)
ti1 = (np.sin(x / 8.0) + np.cos(y / 11.0)).astype(np.float32)
ti2 = (np.cos((x + y) / 13.0) + np.sin(y / 7.0)).astype(np.float32)
conditioning = np.full(ti1.shape, np.nan, dtype=np.float32)
known = rng.random(ti1.shape) < 0.05
conditioning[known] = ti1[known]

result = g2s(
    "-a", "as",
    "-ti", ti1, ti2,
    "-di", conditioning,
    "-dt", [0],
    "-k", 2,
    "-n", 8,
    "-s", 100,
)

simulation = result["simulation"]
selected_ti = result["indexmap"]
print("job", result["job_id"], "simulation", simulation.shape, "selected TI", selected_ti.shape)
