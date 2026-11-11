"""Explanatory animation of a recorded sequential simulation (QS, DS or SNESIM run with pyDev).

    python render.py qs_strebelle [--theme dark] [--preview FRAME ...]

Everything drawn comes from the run log (records/<name>.pkl): the path, each node's data event,
the TI search and the value pasted. Output: out/<name>.mp4 (1920x1080, 30 fps, H.264), or
out/<name>_dark.mp4 with --theme dark.
"""
import argparse
import os
import pickle
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse, Rectangle

SP = os.path.dirname(os.path.abspath(__file__))
W, H, FPS = 1920, 1080, 30

ACCENT = "#1683d8"
DARK = "#1f2933"
MUTED = "#7b8594"
FAINT = "#b9c1cc"
GRIDC = "#e9ecf0"
NBEDGE = "#4b5563"
LINEC = "#8f99a6"
THC = "#e8590c"
OKC = "#2f9e44"
BG = "#ffffff"
BAD = "#eef0f3"                                   # mismatch map where the pattern does not fit
CMAP = ListedColormap(cm.YlGnBu(numpy.linspace(0.30, 0.95, 256)))
MISMATCH_CMAP = cm.magma_r.copy()

# Dark background: light text and lines; the colour map is flipped (low values medium blue, high
# values pale yellow) so that both ends stay visible on the dark background.
THEMES = {
    "light": dict(ACCENT="#1683d8", DARK="#1f2933", MUTED="#7b8594", FAINT="#b9c1cc", GRIDC="#e9ecf0",
                  NBEDGE="#4b5563", LINEC="#8f99a6", THC="#e8590c", OKC="#2f9e44", BG="#ffffff", BAD="#eef0f3",
                  cmap=(0.30, 0.95)),
    "dark": dict(ACCENT="#4dabf7", DARK="#e6e9ee", MUTED="#98a2b3", FAINT="#475162", GRIDC="#1c232d",
                 NBEDGE="#d0d7e1", LINEC="#8793a3", THC="#ff922b", OKC="#51cf66", BG="#0e1217", BAD="#232a35",
                 cmap=(0.85, 0.06)),
}


def set_theme(name):
    """Set the colours used by every drawing function (module globals, read at draw time)."""
    g = globals()
    t = THEMES[name]
    for k, v in t.items():
        if k != "cmap":
            g[k] = v
    g["CMAP"] = ListedColormap(cm.YlGnBu(numpy.linspace(*t["cmap"], 256)))
    g["MISMATCH_CMAP"] = cm.magma_r.copy()
    g["MISMATCH_CMAP"].set_bad(t["BAD"])
    plt.rcParams.update({"font.family": ["Avenir Next", "Helvetica Neue", "DejaVu Sans"],
                         "font.size": 16, "text.color": t["DARK"], "axes.edgecolor": t["FAINT"],
                         "axes.labelcolor": t["MUTED"], "xtick.color": t["MUTED"], "ytick.color": t["MUTED"],
                         "axes.facecolor": t["BG"], "figure.facecolor": t["BG"], "savefig.facecolor": t["BG"]})


set_theme("light")


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def fmt_int(n):
    return f"{n:,}".replace(",", " ")


def pt2(diameter_px):
    """scatter size (points^2) for a marker of the given diameter in pixels at dpi 100."""
    return (diameter_px * 0.72) ** 2


class Box:
    """A rectangle in pixel coordinates (origin top-left)."""

    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h

    def axes(self, fig, **kw):
        return fig.add_axes([self.x / W, 1 - (self.y + self.h) / H, self.w / W, self.h / H], **kw)


TI_BOX = Box(60, 190, 420, 420)
SEARCH_BOX = Box(540, 190, 420, 420)
SIM_BOX = Box(1030, 190, 850, 850)


# ----------------------------------------------------------------------------------------------
class Animation:
    steps = []
    title = ""
    step_map = {}
    show_ti_pattern = True

    def __init__(self, name):
        with open(os.path.join(SP, "records", f"{name}.pkl"), "rb") as f:
            d = pickle.load(f)
        self.name = name
        self.cfg = d["cfg"]
        self.ti = d["ti"]
        self.categorical = d["categorical"]
        self.shape = tuple(d["shape"])
        self.R = d["records"]
        self.N = len(self.R)
        self.cells = numpy.array([self.R[i]["cell"] for i in range(self.N)])      # path order
        self.values = numpy.array([self.R[i]["value"] for i in range(self.N)])
        self.rank = numpy.empty(self.shape, int)
        self.rank[tuple(self.cells.T)] = numpy.arange(self.N)
        self.norm_values = self.values if not self.categorical else self.values  # 0/1 or [0,1]
        self.node_colors = CMAP(self.norm_values)
        self.ti_name = "Strebelle" if "strebelle" in name else "Stone"
        self.build_figure()

    # ---- text helpers ------------------------------------------------------------------------
    def value_label(self, v):
        if self.categorical:
            return "channel" if v >= 0.5 else "background"
        return f"{v:.2f}"

    # ---- figure ------------------------------------------------------------------------------
    def build_figure(self):
        self.fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
        fig = self.fig
        T = lambda x, y, s, **kw: fig.text(x / W, 1 - y / H, s, **kw)

        T(60, 62, self.title, fontsize=34, fontweight="bold", va="center")
        T(60, 108, self.subtitle(), fontsize=17, color=MUTED, va="center")

        # training image
        T(TI_BOX.x, TI_BOX.y - 22, f"Training image ({self.ti.shape[0]} × {self.ti.shape[1]})",
          fontsize=17, fontweight="bold", va="center")
        self.ax_ti = TI_BOX.axes(fig)
        self.ax_ti.imshow(self.ti, cmap=CMAP, vmin=0, vmax=1, interpolation="nearest")
        self.ax_ti.set_axis_off()
        self.ti_lines = LineCollection([], colors=DARK, linewidths=0.8, alpha=0.75, zorder=3)
        self.ax_ti.add_collection(self.ti_lines)
        self.ti_extra = self.ax_ti.scatter([], [], s=pt2(4), c=ACCENT, zorder=3, linewidths=0)
        self.ti_nb = self.ax_ti.scatter([], [], s=pt2(6), zorder=4, edgecolors=DARK, linewidths=0.8)
        self.ti_center = self.ax_ti.scatter([], [], s=pt2(16), facecolors="none", edgecolors=ACCENT,
                                            linewidths=2.4, zorder=5)
        self.ti_marks = self.ax_ti.scatter([], [], s=pt2(13), facecolors="none", edgecolors=THC,
                                           linewidths=1.8, zorder=5)

        # search panel (algorithm specific)
        self.search_label = T(SEARCH_BOX.x, SEARCH_BOX.y - 22, "", fontsize=17, fontweight="bold",
                              va="center")
        self.build_search(fig)

        # simulation grid
        T(SIM_BOX.x, SIM_BOX.y - 22, f"Simulation ({self.shape[0]} × {self.shape[1]})",
          fontsize=17, fontweight="bold", va="center")
        self.counter = T(SIM_BOX.x + SIM_BOX.w, SIM_BOX.y - 22, "", fontsize=17, color=MUTED,
                         ha="right", va="center")
        self.ax_sim = SIM_BOX.axes(fig)
        ny, nx = self.shape
        self.ax_sim.set_xlim(-0.6, nx - 0.4)
        self.ax_sim.set_ylim(ny - 0.4, -0.6)
        self.ax_sim.set_axis_off()
        grid = [[(x, 0), (x, ny - 1)] for x in range(nx)] + [[(0, y), (nx - 1, y)] for y in range(ny)]
        self.ax_sim.add_collection(LineCollection(grid, colors=GRIDC, linewidths=0.5, zorder=0))
        yy, xx = numpy.indices(self.shape)
        self.grid_xy = numpy.column_stack([xx.ravel(), yy.ravel()])
        self.flat_rank = self.rank.ravel()
        self.flat_colors = self.node_colors[self.flat_rank]
        spacing = SIM_BOX.w / nx
        self.s_full, self.s_empty = pt2(spacing * 0.86), pt2(spacing * 0.24)
        self.sim_dots = self.ax_sim.scatter(self.grid_xy[:, 0], self.grid_xy[:, 1], s=self.s_empty,
                                            c=[FAINT], linewidths=0, zorder=1)
        self.sim_template = self.ax_sim.scatter([], [], s=pt2(spacing * 0.55), marker="s",
                                                facecolors="none", edgecolors=ACCENT, linewidths=0.8,
                                                alpha=0.55, zorder=2)
        self.sim_lines = LineCollection([], colors=LINEC, linewidths=1.1, zorder=3)
        self.ax_sim.add_collection(self.sim_lines)
        self.sim_nb = self.ax_sim.scatter([], [], s=pt2(spacing * 1.25), facecolors="none",
                                          edgecolors=NBEDGE, linewidths=1.3, zorder=4)
        self.sim_ring = self.ax_sim.scatter([], [], s=pt2(spacing * 1.9), facecolors="none",
                                            edgecolors=ACCENT, linewidths=2.6, zorder=6)
        self.sim_box = Rectangle((0, 0), 0, 0, fill=False, edgecolor=ACCENT, linewidth=1.2,
                                 linestyle=(0, (4, 3)), alpha=0.0, zorder=2)
        self.ax_sim.add_patch(self.sim_box)

        # steps list and caption
        self.step_texts, self.step_badges = [], []
        y0 = 700
        for i, s in enumerate(self.steps):
            y = y0 + i * 52
            badge = Ellipse(((TI_BOX.x + 16) / W, 1 - y / H), 30 / W, 30 / H, transform=fig.transFigure,
                            facecolor=FAINT, edgecolor="none")
            fig.patches.append(badge)
            self.step_badges.append((badge, T(TI_BOX.x + 16, y + 1, str(i + 1), fontsize=15,
                                              color="white", ha="center", va="center",
                                              fontweight="bold")))
            self.step_texts.append(T(TI_BOX.x + 46, y, s, fontsize=19, va="center", color=FAINT))
        self.caption = T(TI_BOX.x, y0 + len(self.steps) * 52 + 30, "", fontsize=20, color=ACCENT,
                         va="center", style="italic")

        # overlay in pixel coordinates (flying value)
        self.ax_over = fig.add_axes([0, 0, 1, 1], zorder=10)
        self.ax_over.set_xlim(0, W)
        self.ax_over.set_ylim(H, 0)
        self.ax_over.set_axis_off()
        self.ax_over.patch.set_alpha(0)
        self.fly_line, = self.ax_over.plot([], [], color=ACCENT, lw=1.2, ls=(0, (3, 3)), alpha=0.6)
        self.fly_dot = self.ax_over.scatter([], [], s=pt2(12), edgecolors=DARK, linewidths=1, zorder=11)

    # ---- coordinates -------------------------------------------------------------------------
    def ti_px(self, pos):
        h, w = self.ti.shape
        return TI_BOX.x + (pos[1] + 0.5) / w * TI_BOX.w, TI_BOX.y + (pos[0] + 0.5) / h * TI_BOX.h

    def sim_px(self, cell):
        ny, nx = self.shape
        return (SIM_BOX.x + (cell[1] + 0.6) / (nx + 0.2) * SIM_BOX.w,
                SIM_BOX.y + (cell[0] + 0.6) / (ny + 0.2) * SIM_BOX.h)

    # ---- per-node data used by the drawing ---------------------------------------------------
    def data_event(self, i):
        r = self.R[i]
        return numpy.asarray(r["offsets"]).reshape(-1, 2), numpy.asarray(r["values"])

    def chosen_pos(self, i):
        return self.R[i]["pos"]

    # ---- rendering of one frame state --------------------------------------------------------
    def render(self, st):
        filled = st["filled"]
        done = self.flat_rank < filled
        self.sim_dots.set_sizes(numpy.where(done, self.s_full, self.s_empty))
        colors = numpy.tile(numpy.array(matplotlib.colors.to_rgba(FAINT)), (len(done), 1))
        colors[done] = self.flat_colors[done]
        hl = st.get("level_highlight")
        if hl is not None:
            colors[hl & ~done] = matplotlib.colors.to_rgba(MUTED)
        self.sim_dots.set_facecolors(colors)

        i = st.get("cur")
        empty2 = numpy.zeros((0, 2))
        if i is None or i >= self.N:
            for a in (self.sim_ring, self.sim_nb, self.ti_nb, self.ti_center, self.sim_template):
                a.set_offsets(empty2)
            self.sim_lines.set_segments([])
            self.ti_lines.set_segments([])
            self.sim_box.set_alpha(0)
        else:
            cell = numpy.array(self.R[i]["cell"])
            node_xy = cell[::-1].astype(float)
            ring = st.get("ring", 1.0)
            self.sim_ring.set_offsets([node_xy] if ring > 0 else empty2)
            self.sim_ring.set_alpha(ring)
            off, val = self.data_event(i)
            grow = st.get("nb", 0.0)
            if grow > 0 and len(off):
                nb_xy = (cell + off)[:, ::-1].astype(float)
                ends = node_xy + (nb_xy - node_xy) * ease(grow)
                self.sim_lines.set_segments([[node_xy, e] for e in ends])
                self.sim_lines.set_alpha(st.get("nb_alpha", 0.9))
                self.sim_lines.set_linestyles(self.line_styles(i))
                self.sim_lines.set_colors(self.line_colors(i))
                self.sim_nb.set_offsets(nb_xy if grow >= 1 else empty2)
                self.sim_nb.set_edgecolors(self.nb_edge_colors(i))
            else:
                self.sim_lines.set_segments([])
                self.sim_nb.set_offsets(empty2)
            self.sim_template_draw(st, i, cell)

            # pattern in the TI
            pat = st.get("ti_pattern", 0.0)
            pos = st.get("ti_pos", None)
            if pos is None:
                pos = self.chosen_pos(i)
            if pat > 0 and pos is not None and self.show_ti_pattern:
                p = numpy.array(pos)
                q = (p + off)
                h, w = self.ti.shape
                inside = numpy.all((q >= 0) & (q < [h, w]), axis=1) if len(off) else numpy.zeros(0, bool)
                q = q[inside]
                pxy = p[::-1].astype(float)
                self.ti_lines.set_segments([[pxy, e] for e in q[:, ::-1].astype(float)])
                self.ti_lines.set_alpha(0.7 * pat)
                self.ti_nb.set_offsets(q[:, ::-1] if len(q) else empty2)
                self.ti_nb.set_facecolors(CMAP(self.ti[tuple(q.T)]) if len(q) else [])
                self.ti_nb.set_alpha(pat)
                self.ti_center.set_offsets([pxy])
                self.ti_center.set_alpha(pat)
            else:
                self.ti_lines.set_segments([])
                self.ti_nb.set_offsets(empty2)
                self.ti_center.set_offsets(empty2)
        self.render_search(st)

        # flying value
        fly = st.get("fly")
        if fly is not None and i is not None:
            a = numpy.array(self.fly_source(i))
            b = numpy.array(self.sim_px(self.R[i]["cell"]))
            t = ease(fly)
            mid = (a + b) / 2 + numpy.array([0, -140])
            pt = (1 - t) ** 2 * a + 2 * (1 - t) * t * mid + t ** 2 * b
            ts = numpy.linspace(0, 1, 40)
            curve = ((1 - ts) ** 2)[:, None] * a + (2 * (1 - ts) * ts)[:, None] * mid + (ts ** 2)[:, None] * b
            self.fly_line.set_data(curve[:, 0], curve[:, 1])
            self.fly_dot.set_offsets([pt])
            self.fly_dot.set_facecolors([self.node_colors[i]])
            self.fly_dot.set_sizes([pt2(10 + 6 * numpy.sin(numpy.pi * t))])
        else:
            self.fly_line.set_data([], [])
            self.fly_dot.set_offsets(empty2)

        # texts
        active = st["step_abs"] if "step_abs" in st else self.step_map.get(st.get("step", 0), st.get("step", 0))
        for k, (txt, (badge, num)) in enumerate(zip(self.step_texts, self.step_badges)):
            on = (k + 1) == active
            txt.set_color(DARK if on else FAINT)
            txt.set_fontweight("bold" if on else "normal")
            badge.set_facecolor(ACCENT if on else FAINT)
        self.caption.set_text(st.get("caption", ""))
        self.counter.set_text(st.get("counter") or self.counter_text(filled, i))

    def counter_text(self, filled, i):
        n = filled if i is None else min(i + 1, self.N)
        return f"node {fmt_int(n)} / {fmt_int(self.N)}"

    def line_styles(self, i):
        return "solid"

    def line_colors(self, i):
        return LINEC

    def fly_source(self, i):
        return self.ti_px(self.chosen_pos(i))

    def nb_edge_colors(self, i):
        return NBEDGE

    def sim_template_draw(self, st, i, cell):
        self.sim_template.set_offsets(numpy.zeros((0, 2)))
        self.sim_box.set_alpha(0)

    # ---- to be provided by subclasses ---------------------------------------------------------
    def subtitle(self):
        return ""

    def build_search(self, fig):
        pass

    def render_search(self, st):
        pass

    def search_frames(self, i, speed):
        return []

    # ---- timeline ----------------------------------------------------------------------------
    def node_frames(self, i, speed=1.0, detail=True):
        """Frames explaining node i; speed < 1 shortens every stage."""
        n = lambda f: max(1, int(round(f * speed)))
        base = dict(filled=i, cur=i)
        off, _ = self.data_event(i)
        frames = []
        for f in range(n(16)):
            frames.append(dict(base, step=1, ring=ease((f + 1) / n(16)),
                               caption=self.cap_visit(i) if detail else ""))
        cap2 = self.cap_event(i) if detail else ""
        for f in range(n(26)):
            frames.append(dict(base, step=2, nb=(f + 1) / n(26), caption=cap2, search_phase="event"))
        frames += [dict(base, step=2, nb=1.0, caption=cap2, search_phase="event") for _ in range(n(16))]
        frames += self.search_frames(i, speed, detail)
        cap5 = f"Paste: {self.value_label(self.values[i])}" if detail else ""
        for f in range(n(22)):
            frames.append(dict(base, step=5, nb=1.0, ti_pattern=1.0, fly=(f + 1) / n(22), caption=cap5,
                               search_phase="done"))
        frames += [dict(base, filled=i + 1, step=5, nb=1.0, ti_pattern=1.0, caption=cap5,
                        search_phase="done") for _ in range(n(14))]
        return frames

    def cap_visit(self, i):
        return f"Node {fmt_int(i + 1)} of the path"

    def cap_event(self, i):
        off, _ = self.data_event(i)
        if len(off) == 0:
            return "No simulated node yet: the data event is empty"
        return f"Data event: {len(off)} simulated neighbours"

    def fast_frames(self, start, end, seconds, slow_in=True, slow_out=True, caption=""):
        """Fill nodes start..end-1 with a log-spaced speed-up (and slow-down)."""
        F = int(seconds * FPS)
        frames = []
        a, b = numpy.log(start + 1), numpy.log(end + 1)
        for f in range(1, F + 1):
            t = f / F
            if slow_in and slow_out:
                u = ease(t)
            elif slow_in:
                u = t * t
            elif slow_out:
                u = 1 - (1 - t) ** 2
            else:
                u = t
            filled = int(round(numpy.exp(a + (b - a) * u) - 1))
            filled = min(max(filled, start + 1), end)
            frames.append(dict(filled=filled, cur=filled - 1, ring=1.0, nb=1.0, nb_alpha=0.7,
                               ti_pattern=0.9, step=0, caption=caption, search_phase="fast"))
        return frames

    def intro_frames(self):
        return [dict(filled=0, cur=None, step=0,
                     caption="The grid is filled one node at a time, following a random path")
                for _ in range(int(2.5 * FPS))]

    def outro_frames(self):
        return [dict(filled=self.N, cur=None, step=0,
                     caption=f"Done: {fmt_int(self.N)} nodes simulated, each one from the training image",
                     search_phase="end")
                for _ in range(int(3.5 * FPS))]

    def pick_mid(self, target):
        return target

    def timeline(self):
        fr = self.intro_frames()
        fr += self.node_frames(0)
        fr += self.node_frames(1)
        early = list(range(2, 12))
        for j, i in enumerate(early):
            fr += self.node_frames(i, speed=0.45 - 0.3 * j / (len(early) - 1), detail=False)
        mid = self.pick_mid(int(round(0.234 * self.N)))
        fr += self.fast_frames(12, mid, 6.0, caption="… and so on for every node, faster")
        fr += self.node_frames(mid)
        fr += self.fast_frames(mid + 1, self.N, 7.0, slow_out=False,
                               caption="… until the whole grid is simulated")
        fr += self.outro_frames()
        return fr


# ----------------------------------------------------------------------------------------------
class QSAnimation(Animation):
    title = "QuickSampling (QS)"

    @property
    def steps(self):
        k, n = self.cfg["k"], self.cfg["n"]
        return ["Visit the next node of the random path",
                f"Take its data event: the n = {n} closest simulated nodes",
                "Compute the mismatch at every position of the training image",
                f"Draw one of the k = {k:g} best positions",
                "Paste the value of that position into the simulation"]

    def subtitle(self):
        return (f"Training image: {self.ti_name}  ·  n = {self.cfg['n']}  ·  k = {self.cfg['k']:g}"
                f"  ·  simulated with G2S pyDev")

    def errmap(self, i):
        r = self.R[i]
        if r.get("errmap") is not None:
            return r["errmap"].astype(float)
        # same computation as during the run (seqsim.QSRec / _Recorder.mismatch)
        off, val = self.data_event(i)
        if len(off) == 0:
            return numpy.zeros(self.ti.shape)
        h, w = self.ti.shape
        g = numpy.full((len(off), h, w), numpy.nan)
        for j, (dy, dx) in enumerate(off):
            y0, y1, x0, x1 = max(0, -dy), min(h, h - dy), max(0, -dx), min(w, w - dx)
            if y0 < y1 and x0 < x1:
                g[j, y0:y1, x0:x1] = self.ti[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        v = val[:, None, None]
        d = (g != v).astype(float) if self.categorical else (g - v) ** 2
        out = d.mean(axis=0)
        out[numpy.isnan(g).any(axis=0)] = numpy.inf
        return out

    def build_search(self, fig):
        self.search_label.set_text("Mismatch with the data event")
        self.ax_s = SEARCH_BOX.axes(fig)
        self.ax_s.set_axis_off()
        self.err_im = self.ax_s.imshow(numpy.full(self.ti.shape, numpy.nan), cmap=MISMATCH_CMAP,
                                       vmin=0, vmax=1, interpolation="nearest")
        self.best_marks = self.ax_s.scatter([], [], s=pt2(18), facecolors="none", edgecolors=ACCENT,
                                            linewidths=2.2, zorder=4)
        self.pick_mark = self.ax_s.scatter([], [], s=pt2(18), facecolors="none", edgecolors=THC,
                                           linewidths=3, zorder=5)
        self.err_note = fig.text(SEARCH_BOX.x / W, 1 - (SEARCH_BOX.y + SEARCH_BOX.h + 20) / H, "",
                                 ha="left", va="center", fontsize=14, color=MUTED)
        self._err_cache = (None, None)

    def render_search(self, st):
        i, ph = st.get("cur"), st.get("search_phase")
        e2 = numpy.zeros((0, 2))
        if i is None or ph in (None, "event", "end"):
            self.err_im.set_data(numpy.full(self.ti.shape, numpy.nan))
            self.best_marks.set_offsets(e2)
            self.pick_mark.set_offsets(e2)
            self.err_note.set_text("" if ph != "end" else "")
            return
        if self._err_cache[0] != i:
            m = self.errmap(i)
            finite = m[numpy.isfinite(m)]
            vmax = max(float(numpy.quantile(finite, 0.98)), 1e-6) if len(finite) and finite.max() > 0 else 1
            self._err_cache = (i, numpy.ma.masked_invalid(m), vmax)
        _, m, vmax = self._err_cache
        self.err_im.set_data(m)
        self.err_im.set_clim(0, vmax)
        self.err_im.set_alpha(st.get("err_alpha", 1.0))
        r = self.R[i]
        best = numpy.array(r["best"])
        nb = st.get("best_shown", 0 if ph == "map" else len(best))
        self.best_marks.set_offsets(best[:nb, ::-1] if nb else e2)
        show_pick = ph in ("pick", "done", "fast")
        self.pick_mark.set_offsets([numpy.array(r["pos"])[::-1]] if show_pick else e2)
        self.err_note.set_text("bright = good match  \u00b7  grey = pattern does not fit in the TI")

    def search_frames(self, i, speed, detail=True):
        n = lambda f: max(1, int(round(f * speed)))
        r = self.R[i]
        base = dict(filled=i, cur=i, nb=1.0)
        off, _ = self.data_event(i)
        k = self.cfg["k"]
        kc = int(numpy.ceil(k))
        fr = []
        if len(off) == 0:
            cap3 = "Empty data event: every position fits equally well"
        elif self.categorical:
            cap3 = f"Mismatch = share of the {len(off)} neighbours that differ"
        else:
            cap3 = f"Mismatch = mean squared difference over the {len(off)} neighbours"
        for f in range(n(20)):
            fr.append(dict(base, step=3, search_phase="map", err_alpha=ease((f + 1) / n(20)),
                           caption=cap3 if detail else ""))
        fr += [dict(base, step=3, search_phase="map", caption=cap3 if detail else "") for _ in range(n(26))]
        rank = r["best"].index(r["pos"]) if r["pos"] in r["best"] else 0
        probs = [1 / k] * (kc - 1) + [(k - (kc - 1)) / k]
        cap4 = (f"{kc} best candidates (mismatch "
                + ", ".join(f"{e:.2f}" for e in r["best_err"]) + f") → drew #{rank + 1}"
                f" (p = {probs[rank]:.2f})") if len(off) else f"{kc} candidates drawn at random → drew #{rank + 1}"
        for f in range(n(24)):
            fr.append(dict(base, step=4, search_phase="best", best_shown=1 + int(f / n(24) * kc),
                           caption=cap4 if detail else ""))
        for f in range(n(26)):
            fr.append(dict(base, step=4, search_phase="pick", ti_pattern=ease((f + 1) / n(14)),
                           caption=cap4 if detail else ""))
        return fr

    def pick_mid(self, target):
        return target - target % 50          # nodes whose mismatch map was stored during the run


# ----------------------------------------------------------------------------------------------
class DSAnimation(Animation):
    title = "Direct Sampling (DS)"

    @property
    def steps(self):
        c = self.cfg
        return ["Visit the next node of the random path",
                f"Take its data event: the n = {c['n']} closest simulated nodes",
                "Scan the training image at random positions, compute the distance",
                f"Stop at the first distance ≤ th = {c['th']:g} (scan at most f = {c['f']:.0%})",
                "Paste the value of that position into the simulation"]

    def subtitle(self):
        c = self.cfg
        return (f"Training image: {self.ti_name}  ·  n = {c['n']}  ·  th = {c['th']:g}  ·  "
                f"f = {c['f']:g}  ·  simulated with G2S pyDev")

    def build_search(self, fig):
        self.search_label.set_text("Distance of each scanned candidate")
        self.ax_s = Box(SEARCH_BOX.x + 62, SEARCH_BOX.y + 10, SEARCH_BOX.w - 62, SEARCH_BOX.h - 60).axes(fig)
        ax = self.ax_s
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlabel("scan order", fontsize=14)
        ax.set_ylabel("distance to the data event", fontsize=14)
        ax.tick_params(labelsize=12)
        self.th_line = ax.axhline(self.cfg["th"], color=THC, lw=1.6, ls="--", zorder=2)
        self.th_text = ax.text(1.0, self.cfg["th"], f" th = {self.cfg['th']:g}", color=THC, fontsize=13,
                               va="bottom", ha="right", transform=ax.get_yaxis_transform())
        self.scan_pts = ax.scatter([], [], s=pt2(5), c=FAINT, linewidths=0, zorder=3)
        self.scan_hit = ax.scatter([], [], s=pt2(14), facecolors="none", edgecolors=ACCENT, linewidths=2.4,
                                   zorder=4)
        self.scan_note = ax.text(0.98, 0.97, "", transform=ax.transAxes, ha="right", va="top", fontsize=15,
                                 color=DARK)
        if self.categorical:
            self.ymax = 1.0
        else:
            allerr = numpy.concatenate([numpy.asarray(r["scanned_err"], float) for r in self.R.values()])
            self.ymax = float(numpy.quantile(allerr[numpy.isfinite(allerr)], 0.995)) * 1.05
        ax.set_ylim(0, self.ymax)
        self._scan_cache = (None,)

    def render_search(self, st):
        i, ph = st.get("cur"), st.get("search_phase")
        e2 = numpy.zeros((0, 2))
        if i is None or ph in (None, "event", "end"):
            self.scan_pts.set_offsets(e2)
            self.scan_hit.set_offsets(e2)
            self.scan_note.set_text("")
            self.ti_extra.set_offsets(e2)
            self.ti_marks.set_offsets(e2)
            self.ax_s.set_xlim(0, 10)
            return
        r = self.R[i]
        total = r["n_scanned"]
        shown_all = len(r["scanned"])
        cnt = st.get("scan_count", total)
        cnt_draw = min(cnt, shown_all)
        err = numpy.asarray(r["scanned_err"][:cnt_draw], float)
        xs = numpy.arange(1, cnt_draw + 1)
        ok = numpy.isfinite(err)                     # candidates where the pattern leaves the TI have no distance
        self.ax_s.set_xlim(0, max(10, total * 1.04))
        self.scan_pts.set_offsets(numpy.column_stack([xs[ok], err[ok]]) if ok.any() else e2)
        finished = cnt >= total
        if finished:
            if r["accepted"]:
                hit = (total, r["err"])
            else:
                hit = (int(numpy.argmin(numpy.asarray(r["scanned_err"], float))) + 1, r["err"])
            self.scan_hit.set_offsets([hit])
            self.scan_hit.set_edgecolors(ACCENT)
        else:
            self.scan_hit.set_offsets(e2)
        self.scan_note.set_text(f"{fmt_int(cnt)} scanned" + ("" if not finished else
                                ("  ✓ accepted" if r["accepted"] else "  → keep the best")))
        # scanned positions on the TI
        sc = numpy.asarray(r["scanned"][:cnt_draw], float)
        self.ti_extra.set_offsets(sc[:, ::-1] if cnt_draw else e2)
        self.ti_extra.set_alpha(0.55)
        self.ti_extra.set_color(THC)
        self.ti_marks.set_offsets(e2)

    def search_frames(self, i, speed, detail=True):
        n = lambda f: max(1, int(round(f * speed)))
        r = self.R[i]
        base = dict(filled=i, cur=i, nb=1.0)
        total = r["n_scanned"]
        F = n(70) if total > 3 else n(20)
        fr = []
        th = self.cfg["th"]
        for f in range(F):
            t = (f + 1) / F
            cnt = max(1, int(round(total ** t)))            # log-spaced: first candidates one by one
            k = min(cnt, len(r["scanned"])) - 1
            pos = tuple(int(v) for v in r["scanned"][k]) if k >= 0 else r["pos"]
            e = float(r["scanned_err"][k]) if k >= 0 else float("inf")
            if detail:
                cap = (f"Candidate {fmt_int(cnt)}: distance "
                       + ("not defined (outside the TI)" if not numpy.isfinite(e) else f"{e:.2f}")
                       + (" > th" if not numpy.isfinite(e) or e > th else " ≤ th"))
            else:
                cap = ""
            fr.append(dict(base, step=3, search_phase="scan", scan_count=cnt, ti_pattern=0.8, ti_pos=pos,
                           caption=cap))
        if r["accepted"]:
            cap4 = f"Candidate {fmt_int(total)}: distance {r['err']:.2f} ≤ th = {th:g} → accepted"
        else:
            cap4 = (f"No distance ≤ th after {self.cfg['f']:.0%} of the TI → keep the best "
                    f"({r['err']:.2f})")
        fr += [dict(base, step=4, search_phase="pick", scan_count=total, ti_pattern=1.0,
                    caption=cap4 if detail else "") for _ in range(n(34))]
        return fr

    def pick_mid(self, target):
        # an accepted node with a visible scan, close to the target
        for i in sorted(range(self.N), key=lambda j: abs(j - target)):
            r = self.R[i]
            if r["accepted"] and 15 <= r["n_scanned"] <= 600:
                return i
        return target


# ----------------------------------------------------------------------------------------------
class SNESIMAnimation(Animation):
    """SNESIM: the TI is scanned once per grid level into a search tree; each node only descends the tree."""
    title = "SNESIM"
    step_map = {1: 2, 2: 3, 3: 4, 4: 5, 5: 5}      # step 1 is the tree construction
    show_ti_pattern = False
    BUILD_DEPTH = 5                                 # depth of the tree sketch drawn while building

    @property
    def steps(self):
        r = self.cfg["radius"]
        return ["Before simulating: store every TI pattern in a search tree (one per grid level)",
                "Visit the next node of the multigrid path",
                f"Data event: simulated nodes in the {2 * r + 1} × {2 * r + 1} template, scaled to the grid",
                "Descend the tree of that level with the data event",
                "Draw the value from the counts reached, paste it"]

    def subtitle(self):
        c = self.cfg
        return (f"Training image: {self.ti_name}  ·  template {2 * c['radius'] + 1} × "
                f"{2 * c['radius'] + 1}  ·  {c['levels'] + 1} grid levels  ·  simulated with G2S pyDev")

    def __init__(self, name):
        sys.path.insert(0, os.path.join(SP, "algos"))
        import seqsim
        self.seqsim = seqsim
        super().__init__(name)
        self.levels = numpy.array([self.R[i]["level"] for i in range(self.N)])
        yy, xx = numpy.indices(self.shape)
        self.level_masks = {L: ((yy % 2 ** L == 0) & (xx % 2 ** L == 0)).ravel()
                            for L in range(self.cfg["levels"] + 1)}
        self.tree_stats = {int(k): v for k, v in self.cfg["tree_stats"].items()}
        # patterns of the coarsest level, in scan order, for the construction sketch
        L = self.cfg["levels"]
        self.build_tmpl = seqsim.snesim_template(self.cfg["radius"], L)
        reach = numpy.abs(self.build_tmpl).max(axis=0)
        h, w = self.ti.shape
        cy, cx = numpy.meshgrid(numpy.arange(reach[0], h - reach[0]), numpy.arange(reach[1], w - reach[1]),
                                indexing="ij")
        self.build_centres = numpy.column_stack([cy.ravel(), cx.ravel()])
        self.build_seq = self.ti[self.build_centres[:, 0:1] + self.build_tmpl[:, 0],
                                 self.build_centres[:, 1:2] + self.build_tmpl[:, 1]].astype(int)
        self.build_centre_cat = self.ti[self.build_centres[:, 0], self.build_centres[:, 1]].astype(int)

    # ---- figure ------------------------------------------------------------------------------
    def build_figure(self):
        super().build_figure()
        spacing = SIM_BOX.w / self.shape[1]
        self.sim_focus = self.ax_sim.scatter([], [], s=pt2(spacing * 2.0), facecolors="none", edgecolors=THC,
                                             linewidths=2.4, zorder=7)
        self.ti_tmpl = self.ax_ti.scatter([], [], s=pt2(8), marker="s", edgecolors=ACCENT,
                                          linewidths=1.4, zorder=4)
        self.ti_tbox = Rectangle((0, 0), 0, 0, fill=False, edgecolor=ACCENT, linewidth=1.4, zorder=4)
        self.ax_ti.add_patch(self.ti_tbox)
        self.ti_note = self.fig.text(TI_BOX.x / W, 1 - (TI_BOX.y + TI_BOX.h + 20) / H, "", ha="left",
                                     va="center", fontsize=14, color=MUTED)

    def build_search(self, fig):
        self.search_label.set_text("Search tree")
        self.ax_s = SEARCH_BOX.axes(fig)
        ax = self.ax_s
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()
        self.cat_colors = [CMAP(0.0), CMAP(1.0)]
        # construction sketch
        self.sk_edges = LineCollection([], colors=FAINT, linewidths=1.0, zorder=1)
        ax.add_collection(self.sk_edges)
        self.sk_nodes = ax.scatter([], [], zorder=2, edgecolors=DARK, linewidths=0.6)
        self.sk_text = ax.text(0.5, 0.0, "", ha="center", va="bottom", fontsize=14, color=DARK, linespacing=1.5)
        self.list_texts = [ax.text(0.04, 0.86 - k * 0.12, "", ha="left", va="center", fontsize=17, color=DARK)
                           for k in range(6)]
        # descent view
        self.row_line = LineCollection([], colors=FAINT, linewidths=1.6, zorder=1)
        ax.add_collection(self.row_line)
        self.row_dots = ax.scatter([], [], s=pt2(16), edgecolors=DARK, linewidths=0.8, zorder=3)
        self.row_bars = []
        for k in range(12):
            a = Rectangle((0, 0), 0, 0.05, facecolor=self.cat_colors[0], edgecolor="none", zorder=2)
            b = Rectangle((0, 0), 0, 0.05, facecolor=self.cat_colors[1], edgecolor="none", zorder=2)
            ax.add_patch(a)
            ax.add_patch(b)
            self.row_bars.append((a, b))
        self.row_texts = [ax.text(0, 0, "", ha="left", va="center", fontsize=14, color=DARK) for _ in range(12)]
        self.stop_text = ax.text(0.17, 0, "", ha="left", va="center", fontsize=14, color=THC)
        self.more_text = ax.text(0.08, 0, "", ha="center", va="center", fontsize=16, color=MUTED)
        # draw bar
        self.cum = [Rectangle((0.05, 0.06), 0, 0.08, facecolor=self.cat_colors[j], edgecolor=DARK, linewidth=0.8)
                    for j in range(2)]
        for c in self.cum:
            ax.add_patch(c)
        self.cum_txt = [ax.text(0, 0.165, "", ha="center", va="bottom", fontsize=13, color=DARK) for _ in range(2)]
        self.u_mark, = ax.plot([], [], marker="v", color=THC, markersize=13, ls="none")
        self.u_txt = ax.text(0, 0.0, "", ha="center", va="bottom", fontsize=13, color=THC)

    # ---- tree construction sketch -----------------------------------------------------------
    def sketch(self, m):
        """Nodes (depth <= BUILD_DEPTH) of the coarsest tree built from the first m patterns."""
        D = self.BUILD_DEPTH
        seq = self.build_seq[:m, :D]
        cat = self.build_centre_cat[:m]
        nodes, edges = {(): numpy.bincount(cat, minlength=2)}, []
        for d in range(1, D + 1):
            if m == 0:
                break
            keys = seq[:, :d]
            uniq, inv = numpy.unique(keys, axis=0, return_inverse=True)
            for u_i, key in enumerate(uniq):
                nodes[tuple(key)] = numpy.bincount(cat[inv.ravel() == u_i], minlength=2)
        xy = {}
        for key in nodes:
            d = len(key)
            x = sum(b * 0.5 ** (k + 1) for k, b in enumerate(key)) + 0.5 ** (d + 1)
            xy[key] = (0.04 + 0.92 * x, 0.90 - 0.13 * d)
        for key in nodes:
            if key:
                edges.append([xy[key[:-1]], xy[key]])
        return nodes, xy, edges

    def render_build(self, st):
        e2 = numpy.zeros((0, 2))
        m = st.get("build_m")
        if m is not None:
            nodes, xy, edges = self.sketch(m)
            keys = list(nodes)
            tot = numpy.array([nodes[k].sum() for k in keys], float)
            frac = numpy.array([nodes[k][1] / max(nodes[k].sum(), 1) for k in keys])
            self.sk_nodes.set_offsets(numpy.array([xy[k] for k in keys]))
            self.sk_nodes.set_sizes(pt2(6 + 26 * numpy.sqrt(tot / max(len(self.build_seq), 1))))
            self.sk_nodes.set_facecolors(CMAP(frac))
            self.sk_edges.set_segments(edges)
            self.sk_text.set_text(f"grid level {self.cfg['levels']}: {fmt_int(m)} / {fmt_int(len(self.build_seq))} "
                                  f"patterns stored\n(only the first {self.BUILD_DEPTH} levels of the tree are drawn)")
            if 0 < m <= len(self.build_seq):
                c = self.build_centres[m - 1]
                pts = c + self.build_tmpl
                self.ti_tmpl.set_offsets(pts[:, ::-1])
                self.ti_tmpl.set_facecolors(CMAP(self.ti[pts[:, 0], pts[:, 1]]))
                self.ti_center.set_offsets([c[::-1]])
                self.ti_center.set_alpha(1)
                rr = numpy.abs(self.build_tmpl).max(axis=0)
                self.ti_tbox.set_bounds(c[1] - rr[1] - 0.5, c[0] - rr[0] - 0.5, 2 * rr[1] + 1, 2 * rr[0] + 1)
                self.ti_tbox.set_visible(True)
            else:
                self.ti_tmpl.set_offsets(e2)
                self.ti_tbox.set_visible(False)
        else:
            self.sk_nodes.set_offsets(e2)
            self.sk_edges.set_segments([])
            self.sk_text.set_text("")
            self.ti_tmpl.set_offsets(e2)
            self.ti_tbox.set_visible(False)
        lines = st.get("build_list", 0)
        for k, t in enumerate(self.list_texts):
            if k < lines:
                L = self.cfg["levels"] - k
                pat, nod = self.tree_stats[L]
                t.set_text(f"✓  level {L}:  {fmt_int(pat)} patterns, {fmt_int(nod)} nodes")
            elif k == lines and lines and lines == len(self.tree_stats):
                t.set_text("")
            else:
                t.set_text("")

    # ---- descent view -----------------------------------------------------------------------
    def render_search(self, st):
        i, ph = st.get("cur"), st.get("search_phase")
        e2 = numpy.zeros((0, 2))
        self.render_build(st)
        if ph == "build" or ph is None and st.get("build_list"):
            self.search_label.set_text("Building the search trees")
        elif i is not None and i < self.N:
            self.search_label.set_text(f"Search tree of grid level {self.levels[i]}")
        else:
            self.search_label.set_text("Search tree")
        show = i is not None and i < self.N and ph in ("descent", "stop", "draw", "pick", "done", "fast")
        r = self.R[i] if show else None
        steps = r["steps"] if show else []
        nrows = len(steps) if ph not in ("descent",) else min(st.get("tree_rows", 0), len(steps))
        stop_shown = show and r["stop"] is not None and ph != "descent"
        # rows: root + steps (last few), each with a stacked bar of the counts
        MAXR = 9
        first = max(0, nrows - (MAXR - 1))
        rows = []
        if show:
            rows.append(("root", None, numpy.asarray(r["root"])))
            for k in range(first, nrows):
                d, v, c = steps[k]
                rows.append((k, v, numpy.asarray(c)))
        y_top, dy = 0.93, 0.072
        ys = [y_top - j * dy - (0.03 if (j >= 1 and first > 0) else 0) for j in range(len(rows))]
        root_tot = float(numpy.asarray(r["root"]).sum()) if show else 1.0
        dots, cols = [], []
        for j, ((k, v, c), y) in enumerate(zip(rows, ys)):
            dots.append((0.08, y))
            cols.append(FAINT if v is None else self.cat_colors[int(v)])
            tot = float(c.sum())
            wmax = 0.5 * numpy.log10(tot + 1) / numpy.log10(root_tot + 1)
            a, b = self.row_bars[j]
            a.set_bounds(0.17, y - 0.022, wmax * c[0] / max(tot, 1), 0.044)
            b.set_bounds(0.17 + wmax * c[0] / max(tot, 1), y - 0.022, wmax * c[1] / max(tot, 1), 0.044)
            a.set_visible(True)
            b.set_visible(True)
            label = (f"root: {fmt_int(int(tot))} patterns" if k == "root"
                     else f"{fmt_int(int(tot))}  ({fmt_int(int(c[0]))} / {fmt_int(int(c[1]))})")
            self.row_texts[j].set_position((0.19 + wmax, y))
            self.row_texts[j].set_text(label)
        for j in range(len(rows), 12):
            for patch in self.row_bars[j]:
                patch.set_visible(False)
            self.row_texts[j].set_text("")
        self.more_text.set_text("⋮" if first > 0 else "")
        self.more_text.set_position((0.025, y_top - dy - 0.01))
        if stop_shown:
            ys_stop = (ys[-1] if ys else y_top) - dy
            dots.append((0.08, ys_stop))
            cols.append(BG)
            d = r["stop"]
            ignored = len(r["offsets"]) - r["used"]
            self.stop_text.set_position((0.13, ys_stop))
            self.stop_text.set_text(f"✗ no branch: stop ({ignored} node{'s' if ignored > 1 else ''} ignored)")
        else:
            self.stop_text.set_text("")
        self.row_dots.set_offsets(numpy.array(dots) if dots else e2)
        self.row_dots.set_facecolors(cols if cols else [])
        self.row_dots.set_edgecolors([THC if (stop_shown and j == len(dots) - 1) else DARK for j in range(len(dots))])
        self.row_line.set_segments([[dots[j], dots[j + 1]] for j in range(len(dots) - 1)])
        # draw from the counts
        draw = show and ph in ("draw", "pick", "done", "fast")
        if draw:
            p0 = r["prob"][0]
            self.cum[0].set_width(0.9 * p0)
            self.cum[1].set_x(0.05 + 0.9 * p0)
            self.cum[1].set_width(0.9 * (1 - p0))
            for j, c in enumerate(self.cum):
                c.set_visible(c.get_width() > 0)
            self.cum_txt[0].set_position((0.05 + 0.45 * p0, 0.15))
            self.cum_txt[0].set_text(f"background {p0:.2f}" if p0 > 0.18 else "")
            self.cum_txt[1].set_position((0.05 + 0.9 * p0 + 0.45 * (1 - p0), 0.15))
            self.cum_txt[1].set_text(f"channel {1 - p0:.2f}" if 1 - p0 > 0.18 else "")
            u = st.get("u_shown", r["u"])
            self.u_mark.set_data([0.05 + 0.9 * u], [0.035])
            self.u_txt.set_position((0.05 + 0.9 * u, -0.045))
            self.u_txt.set_text(f"u = {u:.2f}")
        else:
            for c in self.cum:
                c.set_visible(False)
            for t in self.cum_txt:
                t.set_text("")
            self.u_mark.set_data([], [])
            self.u_txt.set_text("")
        # neighbour of the current row, highlighted in the simulation
        foc = st.get("focus")
        if show and foc is not None and foc < len(r["offsets"]):
            self.sim_focus.set_offsets([(numpy.array(r["cell"]) + r["offsets"][foc])[::-1]])
        else:
            self.sim_focus.set_offsets(e2)

    def fly_source(self, i):
        u = self.R[i]["u"]
        return (SEARCH_BOX.x + (0.05 + 0.9 * u) * SEARCH_BOX.w, SEARCH_BOX.y + (1 - 0.1) * SEARCH_BOX.h)

    def chosen_pos(self, i):
        return None

    def line_styles(self, i):
        r = self.R[i]
        n, used = len(r["offsets"]), r["used"]
        return ["solid"] * used + [(0, (2, 2))] * (n - used)

    def line_colors(self, i):
        r = self.R[i]
        n, used = len(r["offsets"]), r["used"]
        return [LINEC] * used + ([THC] if r["stop"] is not None else []) + [FAINT] * (n - used - 1)

    def nb_edge_colors(self, i):
        r = self.R[i]
        n, used = len(r["offsets"]), r["used"]
        return [NBEDGE] * used + ([THC] if r["stop"] is not None else []) + [FAINT] * (n - used - 1)

    def sim_template_draw(self, st, i, cell):
        L = self.levels[i]
        step, rad = 2 ** L, self.cfg["radius"]
        if st.get("nb", 0) > 0 and st.get("search_phase") != "fast":
            g = numpy.arange(-rad, rad + 1) * step
            ty, tx = numpy.meshgrid(g, g, indexing="ij")
            pts = numpy.column_stack([ty.ravel(), tx.ravel()]) + cell
            ok = numpy.all((pts >= 0) & (pts < self.shape), axis=1) & ~numpy.all(pts == cell, axis=1)
            self.sim_template.set_offsets(pts[ok][:, ::-1])
            self.sim_box.set_bounds(cell[1] - rad * step - 0.5, cell[0] - rad * step - 0.5,
                                    2 * rad * step + 1, 2 * rad * step + 1)
            self.sim_box.set_alpha(0.8)
        else:
            self.sim_template.set_offsets(numpy.zeros((0, 2)))
            self.sim_box.set_alpha(0)

    def counter_text(self, filled, i):
        n = filled if i is None else min(i + 1, self.N)
        L = self.levels[min(i if i is not None else max(filled - 1, 0), self.N - 1)]
        lv = f"grid level {L} (every {2 ** L}th node)" if L else "grid level 0 (all nodes)"
        return f"{lv}  ·  node {fmt_int(n)} / {fmt_int(self.N)}"

    def render(self, st):
        i = st.get("cur")
        if i is not None and i < self.N:
            st = dict(st, level_highlight=self.level_masks[self.levels[i]] if self.levels[i] > 0 else None)
        super().render(st)
        building = st.get("search_phase") == "build"
        self.ti_note.set_text("" if building or (i is None and st["filled"] == 0) else
                              "not scanned any more: only the search trees are used")
        if st.get("build_m") is not None or building:
            self.counter.set_text("")

    def cap_visit(self, i):
        L = self.levels[i]
        return (f"Node {fmt_int(i + 1)}, grid level {L}: template spacing {2 ** L}" if L
                else f"Node {fmt_int(i + 1)}, finest grid level")

    def cap_event(self, i):
        tot = len(self.R[i]["offsets"])
        if tot == 0:
            return "No simulated node in the template yet: the data event is empty"
        return f"Data event: {tot} simulated node{'s' if tot > 1 else ''} inside the template"

    def search_frames(self, i, speed, detail=True):
        n = lambda f: max(1, int(round(f * speed)))
        r = self.R[i]
        base = dict(filled=i, cur=i, nb=1.0)
        used, tot = r["used"], len(r["offsets"])
        fr = []
        last = r["steps"][-1][2] if used else r["root"]
        cap3 = (f"Follow the data event down the tree: {used} node{'s' if used != 1 else ''} → "
                f"{fmt_int(int(numpy.sum(last)))} pattern{'s' if numpy.sum(last) != 1 else ''} match") if used else \
            "Empty data event: the counts of the root (all patterns) are used"
        F = n(max(24, min(60, 6 * used)))
        for f in range(F):
            k = int(round(ease((f + 1) / F) * used))
            fr.append(dict(base, step=3, search_phase="descent", tree_rows=k, focus=(k - 1) if k else None,
                           caption=cap3 if detail else ""))
        fr += [dict(base, step=3, search_phase="descent", tree_rows=used, caption=cap3 if detail else "")
               for _ in range(n(16))]
        if r["stop"] is not None:
            ignored = tot - used
            cap_stop = (f"Next node ({self.value_label(r['values'][used])}) has no branch: stop here, "
                        f"{ignored} node{'s' if ignored > 1 else ''} ignored")
            fr += [dict(base, step=3, search_phase="stop", focus=used, caption=cap_stop if detail else "")
                   for _ in range(n(40))]
        cap4 = (f"P(channel) = {r['prob'][1]:.2f}: random u = {r['u']:.2f} → "
                f"{self.value_label(self.values[i])}")
        for f in range(n(26)):
            fr.append(dict(base, step=4, search_phase="draw", u_shown=r["u"] * ease((f + 1) / n(26)),
                           caption=cap4 if detail else ""))
        fr += [dict(base, step=4, search_phase="draw", caption=cap4 if detail else "") for _ in range(n(12))]
        return fr

    def build_frames(self):
        P = len(self.build_seq)
        fr = [dict(filled=0, cur=None, step_abs=1, search_phase="build", build_m=0,
                   caption="SNESIM first scans the training image with the template") for _ in range(int(1.2 * FPS))]
        F = int(5.0 * FPS)
        for f in range(F):
            m = int(round(P ** ((f + 1) / F)))           # log-spaced: the first patterns one by one
            fr.append(dict(filled=0, cur=None, step_abs=1, search_phase="build", build_m=m,
                           caption="Each pattern is stored with the category found at its centre"))
        fr += [dict(filled=0, cur=None, step_abs=1, search_phase="build", build_m=P,
                    caption="Each pattern is stored with the category found at its centre")
               for _ in range(int(1.0 * FPS))]
        nlev = len(self.tree_stats)
        for k in range(1, nlev + 1):
            fr += [dict(filled=0, cur=None, step_abs=1, search_phase="build", build_list=k,
                        caption="One tree per grid level, with the template scaled to the level")
                   for _ in range(int(0.6 * FPS))]
        fr += [dict(filled=0, cur=None, step_abs=1, search_phase="build", build_list=nlev,
                    caption="Trees ready: the simulation never scans the training image again")
               for _ in range(int(2.0 * FPS))]
        return fr

    def pick_detail(self, start, end, want_stop):
        for i in range(start, end):
            r = self.R[i]
            if (r["stop"] is not None) == want_stop and r["used"] >= 5:
                return i
        return start

    def timeline(self):
        counts = [int((self.levels == L).sum()) for L in range(self.cfg["levels"], -1, -1)]
        starts = numpy.cumsum([0] + counts)            # first path index of each level (coarse first)
        fr = self.build_frames()
        fr += [dict(filled=0, cur=None, step=0,
                    caption="Multigrid path: coarse nodes first, then finer and finer grids")
               for _ in range(int(2.0 * FPS))]
        fr += self.node_frames(0)
        fr += self.node_frames(1)
        early = list(range(2, 10))
        for j, i in enumerate(early):
            fr += self.node_frames(i, speed=0.45 - 0.3 * j / (len(early) - 1), detail=False)
        lvl3 = self.pick_detail(int(starts[1]), int(starts[2]), want_stop=False)
        fr += self.fast_frames(10, lvl3, 1.5, caption="Coarsest grid: every 16th node")
        fr += self.node_frames(lvl3, speed=0.8)
        mid = self.pick_detail(int(starts[4]) + int(round(0.0625 * self.N)), self.N, want_stop=True)
        fr += self.fast_frames(lvl3 + 1, mid, 6.0,
                               caption="… then the finer grids, conditioned by the coarse nodes")
        fr += self.node_frames(mid)
        fr += self.fast_frames(mid + 1, self.N, 6.5, slow_out=False,
                               caption="… until the finest grid is complete")
        fr += self.outro_frames()
        return fr

CLASSES = {"qs": QSAnimation, "ds": DSAnimation, "snesim": SNESIMAnimation}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--preview", type=int, nargs="*", help="write PNGs of these frame numbers instead")
    ap.add_argument("--theme", choices=sorted(THEMES), default="light",
                    help="light or dark background (dark files get a _dark suffix)")
    args = ap.parse_args()
    set_theme(args.theme)
    suffix = "" if args.theme == "light" else f"_{args.theme}"
    anim = CLASSES[args.name.split("_")[0]](args.name)
    frames = anim.timeline()
    print(f"{args.name}: {len(frames)} frames ({len(frames) / FPS:.1f} s)", flush=True)
    if args.preview:
        os.makedirs(os.path.join(SP, "out"), exist_ok=True)
        for f in args.preview:
            anim.render(frames[f])
            anim.fig.savefig(os.path.join(SP, "out", f"{args.name}{suffix}_f{f}.png"), dpi=100, facecolor=BG)
        return
    os.makedirs(os.path.join(SP, "out"), exist_ok=True)
    out = os.path.join(SP, "out", f"{args.name}{suffix}.mp4")
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
                           "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out],
                          stdin=subprocess.PIPE)
    canvas = anim.fig.canvas
    for k, st in enumerate(frames):
        anim.render(st)
        canvas.draw()
        ff.stdin.write(canvas.buffer_rgba())
        if k % 300 == 0:
            print(f"  {k}/{len(frames)}", flush=True)
    ff.stdin.close()
    ff.wait()
    print("wrote", out)


if __name__ == "__main__":
    main()
