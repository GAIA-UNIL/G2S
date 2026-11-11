"""Sequential-simulation algorithms (QS, DS, SNESIM) for pyDev that record every step.

G2S does the path and the neighbour search; each class predicts one node and stores
what it did (data event, TI search, chosen value) so the run can be animated.
The run configuration (output file, parameters) is read from records/next_run.json, written by run.py.
No shebang here: run.py --python adds one to a run copy to select the pyDev environment.
"""
import json
import os
import pickle
import threading

import numpy

HERE = os.path.dirname(os.path.abspath(__file__))
RECORDS = os.path.join(os.path.dirname(HERE), "records")


class _Recorder:
    def __init__(self, TIs, kernel, di, dt, path):
        with open(os.path.join(RECORDS, "next_run.json")) as f:
            self.cfg = json.load(f)
        self.ti = numpy.asarray(TIs[0], dtype=numpy.float64)
        if self.ti.ndim == 3:
            self.ti = self.ti[..., 0]
        self.categorical = bool(numpy.asarray(dt).ravel()[0] == 1)
        self.shape = numpy.asarray(di).shape[:2]
        p = numpy.asarray(path, dtype=numpy.float64).reshape(self.shape)
        self.order = numpy.argsort(p.ravel(), kind="stable")   # rank -> flat cell
        self.records = {}
        self.lock = threading.Lock()
        H, W = self.ti.shape
        self.grid = numpy.stack(numpy.meshgrid(numpy.arange(H), numpy.arange(W), indexing="ij"), -1)

    def cell(self, index):
        return numpy.unravel_index(int(self.order[index]), self.shape)

    def gather(self, offsets):
        """TI values at every centre + offset, NaN outside the TI: (K, H, W)."""
        H, W = self.ti.shape
        out = numpy.full((len(offsets), H, W), numpy.nan)
        for i, (dy, dx) in enumerate(offsets):
            y0, y1 = max(0, -dy), min(H, H - dy)
            x0, x1 = max(0, -dx), min(W, W - dx)
            if y0 < y1 and x0 < x1:
                out[i, y0:y1, x0:x1] = self.ti[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        return out

    def mismatch(self, offsets, values, power=2):
        """Mean mismatch at every TI centre (inf where a neighbour falls outside the TI).
        Categorical: fraction of differing neighbours. Continuous: mean |diff|**power."""
        if len(offsets) == 0:
            return numpy.zeros(self.ti.shape)
        g = self.gather(offsets)
        v = numpy.asarray(values)[:, None, None]
        if self.categorical:
            d = (g != v).astype(float)
        else:
            d = numpy.abs(g - v) ** power
        out = d.mean(axis=0)
        out[numpy.isnan(g).any(axis=0)] = numpy.inf
        return out

    def store(self, index, rec):
        with self.lock:
            self.records[index] = rec

    def finalize(self):
        out = {"cfg": self.cfg, "shape": self.shape, "order": self.order,
               "ti": self.ti, "categorical": self.categorical, "records": self.records}
        with open(os.path.join(RECORDS, self.cfg["output"]), "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def neighbours(coordinates, values):
        c = numpy.asarray(coordinates)[1:]
        v = numpy.asarray(values)[1:, 0]
        known = ~numpy.isnan(v)
        return c[known].astype(int), v[known]

    @staticmethod
    def rng(randUniform, index):
        return numpy.random.default_rng([int(randUniform * 2**52), int(index)])


def keep_map(index):
    """The full mismatch maps are only kept for the nodes the animation explains in detail."""
    return index < 40 or index % 50 == 0


class QSRec(_Recorder):
    """Quick Sampling: mismatch against every TI position, pick at random among the k best."""

    def sample(self, coordinates, values, randUniform, indexInThePath, threadId,
               fullStationary, variableOfInterest, idTI4Sampling):
        off, val = self.neighbours(coordinates, values)
        rng = self.rng(randUniform, indexInThePath)
        err = self.mismatch(off, val)
        k = float(self.cfg.get("k", 1.2))
        kc = int(numpy.ceil(k))
        noisy = err.ravel() + rng.random(err.size) * 1e-9          # random tie-breaking
        best = numpy.argpartition(noisy, kc)[:kc]
        best = best[numpy.argsort(noisy[best])]
        pick = int(best[min(int(numpy.floor(randUniform * k)), kc - 1)])
        pos = numpy.unravel_index(pick, err.shape)
        value = float(self.ti[pos])
        self.store(int(indexInThePath), {
            "cell": self.cell(indexInThePath), "offsets": off, "values": val,
            "best": [numpy.unravel_index(int(b), err.shape) for b in best],
            "best_err": [float(err.ravel()[b]) for b in best],
            "pos": pos, "err": float(err[pos]), "value": value,
            "errmap": err.astype(numpy.float16) if keep_map(indexInThePath) else None})
        return value


class DSRec(_Recorder):
    """Direct Sampling: scan the TI in random order, accept the first candidate under the threshold."""

    def sample(self, coordinates, values, randUniform, indexInThePath, threadId,
               fullStationary, variableOfInterest, idTI4Sampling):
        off, val = self.neighbours(coordinates, values)
        rng = self.rng(randUniform, indexInThePath)
        err = self.mismatch(off, val).ravel()
        if not self.categorical:
            err = numpy.sqrt(err)                                   # RMSE, as -cnorm 2 in G2S DS
        n_scan = max(1, int(self.cfg.get("f", 0.3) * err.size))
        scan = rng.permutation(err.size)[:n_scan]
        th = float(self.cfg.get("th", 0.05))
        ok = numpy.nonzero(err[scan] <= th)[0]
        if len(ok):
            stop, accepted = int(ok[0]), True
        else:
            stop, accepted = int(numpy.argmin(err[scan])), False
        scanned = scan[:(stop + 1) if accepted else n_scan]
        chosen = int(scan[stop])
        pos = numpy.unravel_index(chosen, self.ti.shape)
        value = float(self.ti[pos])
        keep = scanned[:4000]
        self.store(int(indexInThePath), {
            "cell": self.cell(indexInThePath), "offsets": off, "values": val,
            "scanned": numpy.stack(numpy.unravel_index(keep, self.ti.shape), -1).astype(numpy.int16),
            "scanned_err": err[keep].astype(numpy.float16), "n_scanned": int(len(scanned)),
            "accepted": accepted, "pos": pos, "err": float(err[chosen]), "value": value})
        return value


def snesim_template(radius, level):
    """Template offsets of a grid level, ordered as G2S does (snesim.cpp, buildPathPositionArrayForLevel):
    Manhattan distance, then lexicographic on the G2S offset vector (x first)."""
    step = 2 ** level
    g = numpy.arange(-radius, radius + 1) * step
    offs = [(int(dy), int(dx)) for dy in g for dx in g if (dy, dx) != (0, 0)]
    offs.sort(key=lambda o: (abs(o[0]) + abs(o[1]), o[1], o[0]))
    return numpy.array(offs, dtype=int)


class SearchTree:
    """SNESIM search tree of one grid level (snesimTree.cpp, buildTreeForLevel): every TI position where
    the whole template fits adds its centre category to the counts of each node along its pattern."""

    def __init__(self, ti, categories, template):
        H, W = ti.shape
        cls = numpy.searchsorted(categories, ti).astype(numpy.int64)
        reach = numpy.abs(template).max(axis=0)
        ys = numpy.arange(reach[0], H - reach[0])
        xs = numpy.arange(reach[1], W - reach[1])
        cy, cx = numpy.meshgrid(ys, xs, indexing="ij")
        cy, cx = cy.ravel(), cx.ravel()
        seq = cls[cy[:, None] + template[:, 0], cx[:, None] + template[:, 1]]       # (P, K)
        centre = cls[cy, cx]
        C = len(categories)
        self.n_patterns = len(centre)
        children = [numpy.full((1, C), -1, numpy.int64)]
        counts = [numpy.bincount(centre, minlength=C)[None, :]]
        node = numpy.zeros(len(centre), numpy.int64)
        next_id = 1
        for d in range(seq.shape[1]):
            key = node * C + seq[:, d]
            uniq, inv = numpy.unique(key, return_inverse=True)
            ids = next_id + numpy.arange(len(uniq))
            parents, branch = uniq // C, uniq % C
            children.append(numpy.full((len(uniq), C), -1, numpy.int64))
            all_children = numpy.concatenate(children)
            all_children[parents, branch] = ids
            children = [all_children]
            c = numpy.zeros((len(uniq), C), numpy.int64)
            numpy.add.at(c, (inv, centre), 1)
            counts.append(c)
            node = ids[inv]
            next_id += len(uniq)
        self.children = children[0]
        self.counts = numpy.concatenate(counts)
        self.n_nodes = len(self.counts)

    def descend(self, template_values):
        """template_values: (K,) class index per template node, -1 when not simulated yet.
        Unknown nodes branch into every child; the counts are read at the deepest depth reached
        (snesimCPUThreadDevice.cpp, depthSearchTreeRecursive)."""
        current = numpy.array([0])
        steps = []
        stop = None
        for d, v in enumerate(template_values):
            if v >= 0:
                nxt = self.children[current, v]
            else:
                nxt = self.children[current].ravel()
            nxt = nxt[nxt >= 0]
            if len(nxt) == 0:
                stop = d
                break
            current = nxt
            if v >= 0:
                steps.append((d, int(v), self.counts[current].sum(axis=0)))
        return self.counts[current].sum(axis=0), steps, stop


class SNESIMRec(_Recorder):
    """SNESIM with one search tree per grid level, built once from the TI. Each node descends the tree
    of its level with the data event and draws its category from the counts reached."""

    def __init__(self, TIs, kernel, di, dt, path):
        super().__init__(TIs, kernel, di, dt, path)
        self.radius = int(self.cfg.get("radius", 3))
        self.levels = int(self.cfg.get("levels", 4))
        self.categories = numpy.unique(self.ti)
        self.templates, self.trees = {}, {}
        for L in range(self.levels, -1, -1):
            self.templates[L] = snesim_template(self.radius, L)
            self.trees[L] = SearchTree(self.ti, self.categories, self.templates[L])
        self.tree_stats = {L: (t.n_patterns, t.n_nodes) for L, t in self.trees.items()}
        self.template_index = {L: {tuple(o): k for k, o in enumerate(t)} for L, t in self.templates.items()}

    def level(self, cell):
        y, x = cell
        for L in range(self.levels, 0, -1):
            if y % 2**L == 0 and x % 2**L == 0:
                return L
        return 0

    def sample(self, coordinates, values, randUniform, indexInThePath, threadId,
               fullStationary, variableOfInterest, idTI4Sampling):
        off, val = self.neighbours(coordinates, values)
        cell = self.cell(indexInThePath)
        L = self.level(cell)
        index = self.template_index[L]
        tv = numpy.full(len(self.templates[L]), -1)
        for o, v in zip(off, val):
            k = index.get((int(o[0]), int(o[1])))
            if k is not None:
                tv[k] = int(numpy.searchsorted(self.categories, v))
        counts, steps, stop = self.trees[L].descend(tv)
        prob = counts / counts.sum()
        u = randUniform
        cat = min(int(numpy.searchsorted(numpy.cumsum(prob), u, side="right")), len(prob) - 1)
        value = float(self.categories[cat])
        informed = numpy.nonzero(tv >= 0)[0]
        tmpl = self.templates[L]
        self.store(int(indexInThePath), {
            "cell": cell, "level": L,
            "offsets": tmpl[informed], "values": self.categories[tv[informed]],
            "steps": [(int(d), v, c.copy()) for d, v, c in steps],   # one per data-event node followed
            "stop": stop,                                             # template index with no branch
            "used": len(steps), "root": self.trees[L].counts[0].copy(),
            "counts": counts, "prob": prob, "u": float(u), "value": value})
        return value

    def finalize(self):
        self.cfg["tree_stats"] = self.tree_stats
        super().finalize()
