import numpy as np
from g2s import g2s


dim = 24
x, y, z = np.indices((dim, dim, dim))
ti = (((x // 4) + (y // 4) + (z // 4)) % 2).astype(np.float32)
di = np.full_like(ti, np.nan, dtype=np.float32)

angle = np.pi / 4.0
quaternion = np.array(
    [0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)],
    dtype=np.float32,
)
rotation_map = np.zeros(di.shape + (4,), dtype=np.float32)
rotation_map[..., :] = quaternion

_schema_result = g2s(
    "-a", "qs",
    "-ti", ti,
    "-di", di,
    "-dt", [1],
    "-k", 1.2,
    "-n", 24,
    "-s", 123,
    "-rmi", rotation_map)
simulation = _schema_result["simulation"]
index = _schema_result["indexmap"]
time = _schema_result.get("time")

print("Simulation shape:", simulation.shape)
print("Index shape:", index.shape)
print("Elapsed:", time)
