"""Check that a run log matches the simulation G2S returned.

    python check.py qs_strebelle [ds_stone_50 ...]

For every logged node: its path index and its value must match the simulation, and every neighbour
of its data event must have been simulated before it.
"""
import os
import pickle
import sys

import numpy

SP = os.path.dirname(os.path.abspath(__file__))


def check(name):
    rec = os.path.join(SP, "records", name)
    with open(f"{rec}.pkl", "rb") as f:
        d = pickle.load(f)
    sim = numpy.load(f"{rec}_sim.npy")
    path = numpy.load(f"{rec}_path.npy")
    R = d["records"]
    wrong = sum(1 for i, r in R.items() if path[tuple(r["cell"])] != i or sim[tuple(r["cell"])] != r["value"])
    later = sum(1 for i, r in R.items() for o in r["offsets"] if path[tuple(numpy.array(r["cell"]) + o)] >= i)
    ok = len(R) == sim.size and not numpy.isnan(sim).any() and wrong == 0 and later == 0
    print(f"{name}: {len(R)} nodes logged, {wrong} wrong path index or value, "
          f"{later} neighbours simulated after their node -> {'OK' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    results = [check(name) for name in sys.argv[1:]]
    sys.exit(0 if all(results) else 1)
