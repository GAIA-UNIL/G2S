"""Animated SVG (SMIL) version of a recorded sequential simulation.

    python svg.py qs_strebelle_50 [dark]

Uses the same run log and the same timeline as render.py. Every element carries discrete or linear
<animate> tracks over the whole duration and repeats indefinitely, so the animation loops in sync.
To keep the file small, neighbour lines, TI patterns and search panels are only drawn for the nodes
explained step by step; during the fast phases only the ring and the filling are animated.
"""
import base64
import io
import os
import sys
from html import escape

import numpy
from matplotlib.colors import to_hex, to_rgba
from PIL import Image

import render
from render import TI_BOX, SEARCH_BOX, SIM_BOX, W, H, FPS

THEME_NAMES = ["ACCENT", "DARK", "MUTED", "FAINT", "GRIDC", "NBEDGE", "LINEC", "THC", "BG", "BAD", "CMAP",
               "MISMATCH_CMAP"]


def use_theme(name):
    """Take the colours of a render.py theme (module globals, read at build time)."""
    render.set_theme(name)
    globals().update({k: getattr(render, k) for k in THEME_NAMES})


use_theme("light")

FONT = "'Avenir Next', 'Helvetica Neue', Helvetica, Arial, sans-serif"


def px(pt):
    return pt * 100 / 72


def f3(x):
    return f"{x:.1f}".rstrip("0").rstrip(".") if abs(x - round(x)) > 1e-9 else str(int(round(x)))


class SVG:
    def __init__(self, anim):
        self.a = anim
        self.F = anim.timeline()
        self.nf = len(self.F)
        self.T = self.nf / FPS
        self.out = []

    # ---- animation helpers -------------------------------------------------------------------
    def kt(self, f):
        return f"{f / self.nf:.5f}"

    def discrete(self, attr, vals, static_ok=True):
        """<animate> for a per-frame list of values (compressed to the changes)."""
        changes = [(0, vals[0])]
        for k in range(1, len(vals)):
            if vals[k] != changes[-1][1]:
                changes.append((k, vals[k]))
        if len(changes) == 1 and static_ok:
            return None
        return (f'<animate attributeName="{attr}" dur="{self.T:.3f}s" repeatCount="indefinite" '
                f'calcMode="discrete" values="{";".join(str(v) for _, v in changes)}" '
                f'keyTimes="{";".join(self.kt(k) for k, _ in changes)}"/>')

    def visible(self, mask):
        """show/hide track from a per-frame boolean list."""
        # display (not visibility): a hidden group then hides its children whatever their own tracks say
        return self.discrete("display", ["inline" if m else "none" for m in mask], static_ok=False)

    def window(self, f0, f1):
        mask = [False] * self.nf
        for k in range(max(0, f0), min(self.nf, f1)):
            mask[k] = True
        return mask

    def linear(self, attr, a, b, f0, f1):
        """attr goes from a to b between frames f0 and f1 (held before and after)."""
        k0, k1 = max(f0, 1) / self.nf, min(f1, self.nf - 1) / self.nf
        return (f'<animate attributeName="{attr}" dur="{self.T:.3f}s" repeatCount="indefinite" '
                f'calcMode="linear" values="{a};{a};{b};{b}" keyTimes="0;{k0:.5f};{k1:.5f};1"/>')

    def text(self, x, y, s, size, color=None, weight="normal", anchor="start", italic=False, extra=""):
        color = color or DARK
        st = ' font-style="italic"' if italic else ""
        return (f'<text x="{f3(x)}" y="{f3(y)}" font-size="{px(size):.1f}" fill="{color}" '
                f'font-weight="{weight}" text-anchor="{anchor}" dominant-baseline="central"{st}{extra}>'
                f'{escape(s)}')

    def text_track(self, x, y, strings, size, **kw):
        """One <text> per distinct string, shown when the per-frame string equals it."""
        out = []
        for s in sorted({v for v in strings if v}):
            mask = [v == s for v in strings]
            out.append(self.text(x, y, s, size, extra=' display="none"', **kw) + self.visible(mask) + "</text>")
        return "\n".join(out)

    @staticmethod
    def png(rgba):
        im = Image.fromarray((numpy.asarray(rgba) * 255).astype(numpy.uint8))
        buf = io.BytesIO()
        im.save(buf, "PNG", optimize=True)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    # ---- coordinates ---------------------------------------------------------------------------
    def sim_xy(self, cell):
        return self.a.sim_px(cell)

    def ti_xy(self, pos):
        return self.a.ti_px(pos)

    # ---- per-frame state ---------------------------------------------------------------------
    def frame_info(self):
        a = self.a
        self.cur = [st.get("cur") for st in self.F]
        self.filled = [st["filled"] for st in self.F]
        self.phase = [st.get("search_phase") for st in self.F]
        self.detail_nodes = sorted({st["cur"] for st in self.F
                                    if st.get("cur") is not None and st.get("search_phase") != "fast"
                                    and st.get("nb", 0) > 0})
        self.node_frames = {i: [k for k, st in enumerate(self.F)
                                if st.get("cur") == i and st.get("search_phase") != "fast"]
                            for i in self.detail_nodes}

    # ---- static layout -------------------------------------------------------------------------
    def build_static(self):
        a, o = self.a, self.out
        o.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
        o.append(self.text(60, 62, a.title, 34, weight="bold") + "</text>")
        o.append(self.text(60, 108, a.subtitle(), 17, color=MUTED) + "</text>")
        o.append(self.text(TI_BOX.x, TI_BOX.y - 22, f"Training image ({a.ti.shape[0]} × {a.ti.shape[1]})",
                           17, weight="bold") + "</text>")
        o.append(f'<image x="{TI_BOX.x}" y="{TI_BOX.y}" width="{TI_BOX.w}" height="{TI_BOX.h}" '
                 f'preserveAspectRatio="none" style="image-rendering:pixelated" href="{self.png(CMAP(a.ti))}"/>')
        o.append(self.text(SIM_BOX.x, SIM_BOX.y - 22, f"Simulation ({a.shape[0]} × {a.shape[1]})", 17,
                           weight="bold") + "</text>")
        ny, nx = a.shape
        x0, y0 = self.sim_xy((0, 0))
        x1, y1 = self.sim_xy((ny - 1, nx - 1))
        d = "".join(f"M{f3(self.sim_xy((0, c))[0])} {f3(y0)}V{f3(y1)}" for c in range(nx))
        d += "".join(f"M{f3(x0)} {f3(self.sim_xy((r, 0))[1])}H{f3(x1)}" for r in range(ny))
        o.append(f'<path d="{d}" stroke="{GRIDC}" stroke-width="0.7" fill="none"/>')

    def build_steps(self):
        a = self.a
        y0 = 700
        active = [st["step_abs"] if "step_abs" in st else a.step_map.get(st.get("step", 0), st.get("step", 0))
                  for st in self.F]
        for k, s in enumerate(a.steps):
            y = y0 + k * 52
            on = [v == k + 1 for v in active]
            self.out.append(f'<circle cx="{TI_BOX.x + 16}" cy="{y}" r="15" fill="{FAINT}">'
                            + (self.discrete("fill", [ACCENT if m else FAINT for m in on]) or "") + "</circle>")
            self.out.append(self.text(TI_BOX.x + 16, y + 1, str(k + 1), 15, color="white", weight="bold",
                                      anchor="middle") + "</text>")
            self.out.append(self.text(TI_BOX.x + 46, y, s, 19, color=FAINT)
                            + (self.discrete("fill", [DARK if m else FAINT for m in on]) or "")
                            + (self.discrete("font-weight", ["bold" if m else "normal" for m in on]) or "")
                            + "</text>")
        caps = [st.get("caption", "") for st in self.F]
        self.out.append(self.text_track(TI_BOX.x, y0 + len(a.steps) * 52 + 30, caps, 20, color=ACCENT,
                                        italic=True))
        counters = []
        for st in self.F:
            c = st.get("counter")
            if c is None:
                i = st.get("cur")
                c = a.counter_text(st["filled"], i)
            if isinstance(a, render.SNESIMAnimation) and st.get("search_phase") == "build":
                c = ""
            counters.append(c)
        self.out.append(self.text_track(SIM_BOX.x + SIM_BOX.w, SIM_BOX.y - 22, counters, 17, color=MUTED,
                                        anchor="end"))

    # ---- simulation grid -----------------------------------------------------------------------
    def build_dots(self):
        a = self.a
        spacing = SIM_BOX.w / a.shape[1]
        r_full, r_empty = spacing * 0.43, spacing * 0.12
        filled = numpy.array(self.filled)
        # frame at which each path index is filled
        fill_frame = numpy.searchsorted(filled, numpy.arange(a.N), side="right")
        levels = getattr(a, "levels", None)
        cur_level = None
        if levels is not None:
            cur_level = numpy.array([levels[c] if c is not None and c < a.N else -1 for c in self.cur])
        g = ['<g>']
        for idx in range(a.N):
            cell = a.cells[idx]
            x, y = self.sim_xy(cell)
            ff = int(fill_frame[idx])
            color = to_hex(a.node_colors[idx])
            fills = [(0, FAINT)]
            if levels is not None and levels[idx] > 0:
                L = levels[idx]
                hl = numpy.nonzero(cur_level[:ff] == L)[0]
                if len(hl):
                    fills.append((int(hl[0]), MUTED))
            if ff < self.nf:
                fills.append((ff, color))
            el = f'<circle cx="{f3(x)}" cy="{f3(y)}" r="{f3(r_empty)}" fill="{FAINT}">'
            if len(fills) > 1:
                el += (f'<animate attributeName="fill" dur="{self.T:.3f}s" repeatCount="indefinite" '
                       f'calcMode="discrete" values="{";".join(c for _, c in fills)}" '
                       f'keyTimes="{";".join(self.kt(k) for k, _ in fills)}"/>')
            if ff < self.nf:
                el += (f'<animate attributeName="r" dur="{self.T:.3f}s" repeatCount="indefinite" '
                       f'calcMode="discrete" values="{f3(r_empty)};{f3(r_full)}" keyTimes="0;{self.kt(ff)}"/>')
            g.append(el + "</circle>")
        g.append("</g>")
        self.out.append("\n".join(g))

    def build_ring(self):
        a = self.a
        spacing = SIM_BOX.w / a.shape[1]
        xs, ys, vis = [], [], []
        for st in self.F:
            i = st.get("cur")
            if i is None or i >= a.N or st.get("ring", 1.0) <= 0:
                xs.append(xs[-1] if xs else 0)
                ys.append(ys[-1] if ys else 0)
                vis.append(False)
            else:
                x, y = self.sim_xy(a.cells[i])
                xs.append(f3(x))
                ys.append(f3(y))
                vis.append(True)
        self.ring_track = (xs, ys, vis)
        self.out.append(f'<circle r="{f3(spacing * 0.95)}" fill="none" stroke="{ACCENT}" stroke-width="2.6" '
                        f'display="none">' + (self.discrete("cx", xs) or "") + (self.discrete("cy", ys) or "")
                        + self.visible(vis) + "</circle>")

    # ---- data event of the detailed nodes ------------------------------------------------------
    def build_events(self):
        a = self.a
        spacing = SIM_BOX.w / a.shape[1]
        for i in self.detail_nodes:
            fr = self.node_frames[i]
            off, _ = a.data_event(i)
            cell = numpy.array(a.cells[i])
            nx0, ny0 = self.sim_xy(cell)
            grow = [k for k in fr if 0 < self.F[k].get("nb", 0) < 1]
            shown = [k for k in fr if self.F[k].get("nb", 0) > 0]
            if not shown:
                continue
            g0, g1 = (grow[0], grow[-1] + 1) if grow else (shown[0], shown[0] + 1)
            parts = [f'<g display="none">' + self.visible(self.window(shown[0], shown[-1] + 1))]
            # template of SNESIM
            if isinstance(a, render.SNESIMAnimation):
                L = a.levels[i]
                step, rad = 2 ** L, a.cfg["radius"]
                gg = numpy.arange(-rad, rad + 1) * step
                for dy in gg:
                    for dx in gg:
                        p = cell + [dy, dx]
                        if (dy or dx) and numpy.all((p >= 0) & (p < a.shape)):
                            x, y = self.sim_xy(p)
                            s = spacing * 0.4
                            parts.append(f'<rect x="{f3(x - s / 2)}" y="{f3(y - s / 2)}" width="{f3(s)}" '
                                         f'height="{f3(s)}" fill="none" stroke="{ACCENT}" stroke-width="0.8" '
                                         f'opacity="0.55"/>')
                bx0, by0 = self.sim_xy(numpy.maximum(cell - rad * step, 0))
                bx1, by1 = self.sim_xy(numpy.minimum(cell + rad * step, numpy.array(a.shape) - 1))
                parts.append(f'<rect x="{f3(bx0 - spacing / 2)}" y="{f3(by0 - spacing / 2)}" '
                             f'width="{f3(bx1 - bx0 + spacing)}" height="{f3(by1 - by0 + spacing)}" fill="none" '
                             f'stroke="{ACCENT}" stroke-width="1.2" stroke-dasharray="4 3" opacity="0.8"/>')
            colors = a.line_colors(i)
            styles = a.line_styles(i)
            edges = a.nb_edge_colors(i)
            n = len(off)
            colors = colors if isinstance(colors, list) else [colors] * n
            styles = styles if isinstance(styles, list) else [styles] * n
            edges = edges if isinstance(edges, list) else [edges] * n
            for k, o in enumerate(off):
                x, y = self.sim_xy(cell + o)
                dash = ' stroke-dasharray="3 3"' if styles[k] != "solid" else ""
                parts.append(f'<line x1="{f3(nx0)}" y1="{f3(ny0)}" x2="{f3(nx0)}" y2="{f3(ny0)}" '
                             f'stroke="{to_hex(colors[k])}" stroke-width="1.3"{dash}>'
                             + self.linear("x2", f3(nx0), f3(x), g0, g1) + self.linear("y2", f3(ny0), f3(y), g0, g1)
                             + "</line>")
            ring = [f'<circle cx="{f3(self.sim_xy(cell + o)[0])}" cy="{f3(self.sim_xy(cell + o)[1])}" '
                    f'r="{f3(spacing * 0.62)}" fill="none" stroke="{to_hex(edges[k])}" stroke-width="1.4"/>'
                    for k, o in enumerate(off)]
            parts.append('<g display="none">' + self.visible(self.window(g1, shown[-1] + 1)) + "".join(ring)
                         + "</g>")
            parts.append("</g>")
            self.out.append("\n".join(parts))

    # ---- TI pattern and flying value -------------------------------------------------------------
    def pattern_group(self, i, pos, colored=True):
        a = self.a
        off, _ = a.data_event(i)
        p = numpy.array(pos)
        h, w = a.ti.shape
        cx, cy = self.ti_xy(p)
        cell_px = TI_BOX.w / w
        parts = []
        for o in off:
            q = p + o
            if not numpy.all((q >= 0) & (q < [h, w])):
                continue
            x, y = self.ti_xy(q)
            parts.append(f'<line x1="{f3(cx)}" y1="{f3(cy)}" x2="{f3(x)}" y2="{f3(y)}" stroke="{DARK}" '
                         f'stroke-width="0.8" opacity="0.7"/>')
        for o in off:
            q = p + o
            if not numpy.all((q >= 0) & (q < [h, w])):
                continue
            x, y = self.ti_xy(q)
            fill = to_hex(CMAP(a.ti[tuple(q)])) if colored else BG
            parts.append(f'<circle cx="{f3(x)}" cy="{f3(y)}" r="{f3(max(3, cell_px * 0.8))}" fill="{fill}" '
                         f'stroke="{DARK}" stroke-width="0.8"/>')
        parts.append(f'<circle cx="{f3(cx)}" cy="{f3(cy)}" r="8" fill="none" stroke="{ACCENT}" stroke-width="2.4"/>')
        return "".join(parts)

    def build_patterns(self):
        a = self.a
        if not a.show_ti_pattern:
            return
        for i in self.detail_nodes:
            fr = self.node_frames[i]
            mask = [False] * self.nf
            for k in fr:
                st = self.F[k]
                if st.get("ti_pattern", 0) > 0 and st.get("ti_pos") is None:
                    mask[k] = True
            if any(mask):
                self.out.append('<g display="none">' + self.visible(mask)
                                + self.pattern_group(i, a.chosen_pos(i)) + "</g>")
        # fast phases: the matched position only
        xs, ys, vis = [], [], []
        for st in self.F:
            i = st.get("cur")
            if st.get("search_phase") == "fast" and i is not None and i < a.N:
                x, y = self.ti_xy(a.chosen_pos(i))
                xs.append(f3(x)), ys.append(f3(y)), vis.append(True)
            else:
                xs.append(xs[-1] if xs else 0), ys.append(ys[-1] if ys else 0), vis.append(False)
        self.out.append(f'<circle r="8" fill="none" stroke="{ACCENT}" stroke-width="2.4" display="none">'
                        + (self.discrete("cx", xs) or "") + (self.discrete("cy", ys) or "") + self.visible(vis)
                        + "</circle>")

    def build_fly(self):
        a = self.a
        for i in self.detail_nodes:
            fr = [k for k in self.node_frames[i] if self.F[k].get("fly") is not None]
            if not fr:
                continue
            A = numpy.array(a.fly_source(i))
            B = numpy.array(self.sim_xy(a.cells[i]))
            M = (A + B) / 2 + numpy.array([0, -140])
            d = f"M{f3(A[0])} {f3(A[1])}Q{f3(M[0])} {f3(M[1])} {f3(B[0])} {f3(B[1])}"
            f0, f1 = fr[0], fr[-1] + 1
            k0, k1 = f0 / self.nf, min(f1, self.nf - 1) / self.nf
            color = to_hex(a.node_colors[i])
            self.out.append(f'<g display="none">' + self.visible(self.window(f0, f1))
                            + f'<path d="{d}" fill="none" stroke="{ACCENT}" stroke-width="1.2" '
                              f'stroke-dasharray="3 3" opacity="0.6"/>'
                            + f'<circle r="7" fill="{color}" stroke="{DARK}" stroke-width="1">'
                              f'<animateMotion dur="{self.T:.3f}s" repeatCount="indefinite" path="{d}" '
                              f'keyPoints="0;0;1;1" keyTimes="0;{k0:.5f};{k1:.5f};1" calcMode="linear"/>'
                              f'</circle></g>')

    # ---- search panels -------------------------------------------------------------------------
    def build_search_qs(self):
        a = self.a
        self.out.append(self.text(SEARCH_BOX.x, SEARCH_BOX.y - 22, "Mismatch with the data event", 17,
                                  weight="bold") + "</text>")
        h, w = a.ti.shape
        for i in self.detail_nodes:
            fr = self.node_frames[i]
            ph = {k: self.F[k].get("search_phase") for k in fr}
            shown = [k for k in fr if ph[k] in ("map", "best", "pick", "done")]
            if not shown:
                continue
            m = a.errmap(i)
            finite = m[numpy.isfinite(m)]
            vmax = max(float(numpy.quantile(finite, 0.98)), 1e-6) if len(finite) and finite.max() > 0 else 1
            rgba = MISMATCH_CMAP(numpy.ma.masked_invalid(m) / vmax)
            rgba[~numpy.isfinite(m)] = to_rgba(BAD)
            parts = ['<g display="none">' + self.visible(self.window(shown[0], shown[-1] + 1)),
                     f'<image x="{SEARCH_BOX.x}" y="{SEARCH_BOX.y}" width="{SEARCH_BOX.w}" height="{SEARCH_BOX.h}" '
                     f'preserveAspectRatio="none" style="image-rendering:pixelated" href="{self.png(rgba)}"/>',
                     self.text(SEARCH_BOX.x, SEARCH_BOX.y + SEARCH_BOX.h + 20,
                               "bright = good match  ·  grey = pattern does not fit in the TI", 14,
                               color=MUTED) + "</text>"]
            r = a.R[i]
            best_from = [k for k in fr if ph[k] in ("best", "pick", "done")]
            pick_from = [k for k in fr if ph[k] in ("pick", "done")]
            sx, sy = SEARCH_BOX.w / w, SEARCH_BOX.h / h
            if best_from:
                bm = "".join(f'<circle cx="{f3(SEARCH_BOX.x + (b[1] + 0.5) * sx)}" '
                             f'cy="{f3(SEARCH_BOX.y + (b[0] + 0.5) * sy)}" r="9" fill="none" stroke="{ACCENT}" '
                             f'stroke-width="2.2"/>' for b in r["best"])
                parts.append('<g display="none">' + self.visible(self.window(best_from[0], shown[-1] + 1))
                             + bm + "</g>")
            if pick_from:
                p = r["pos"]
                parts.append(f'<circle cx="{f3(SEARCH_BOX.x + (p[1] + 0.5) * sx)}" '
                             f'cy="{f3(SEARCH_BOX.y + (p[0] + 0.5) * sy)}" r="9" fill="none" stroke="{THC}" '
                             f'stroke-width="3" display="none">'
                             + self.visible(self.window(pick_from[0], shown[-1] + 1)) + "</circle>")
            parts.append("</g>")
            self.out.append("\n".join(parts))

    def build_search_ds(self):
        a = self.a
        th = a.cfg["th"]
        self.out.append(self.text(SEARCH_BOX.x, SEARCH_BOX.y - 22, "Distance of each scanned candidate", 17,
                                  weight="bold") + "</text>")
        ax0, ay0 = SEARCH_BOX.x + 70, SEARCH_BOX.y + 10
        aw, ah = SEARCH_BOX.w - 70, SEARCH_BOX.h - 70
        ymax = a.ymax
        Y = lambda v: ay0 + ah * (1 - v / ymax)
        o = self.out
        o.append(f'<path d="M{ax0} {ay0}V{ay0 + ah}H{ax0 + aw}" stroke="{FAINT}" fill="none" stroke-width="1"/>')
        for t in numpy.linspace(0, ymax, 5):
            o.append(self.text(ax0 - 8, Y(t), f"{t:.2f}", 12, color=MUTED, anchor="end") + "</text>")
        o.append(f'<text x="{ax0 - 52}" y="{ay0 + ah / 2}" font-size="{px(14):.1f}" fill="{MUTED}" '
                 f'text-anchor="middle" dominant-baseline="central" '
                 f'transform="rotate(-90 {ax0 - 52} {ay0 + ah / 2})">distance to the data event</text>')
        o.append(self.text(ax0 + aw / 2, ay0 + ah + 42, "scan order", 14, color=MUTED, anchor="middle") + "</text>")
        o.append(f'<line x1="{ax0}" x2="{ax0 + aw}" y1="{f3(Y(th))}" y2="{f3(Y(th))}" stroke="{THC}" '
                 f'stroke-width="1.6" stroke-dasharray="6 4"/>')
        o.append(self.text(ax0 + aw, Y(th) - 10, f"th = {th:g}", 13, color=THC, anchor="end") + "</text>")
        notes = [""] * self.nf
        h, w = a.ti.shape
        for i in self.detail_nodes:
            fr = self.node_frames[i]
            scan_fr = [k for k in fr if self.F[k].get("search_phase") in ("scan", "pick", "done")]
            if not scan_fr:
                continue
            r = a.R[i]
            total = r["n_scanned"]
            xmax = max(10, total * 1.04)
            X = lambda c: ax0 + aw * c / xmax
            err = numpy.asarray(r["scanned_err"], float)
            ok = numpy.isfinite(err)
            idx = numpy.arange(1, len(err) + 1)
            d = "".join(f"M{f3(X(c))} {f3(Y(min(e, ymax)))}h0" for c, e in zip(idx[ok], err[ok]))
            clip_id = f"clip{i}"
            widths = []
            for k in range(self.nf):
                st = self.F[k]
                if k in scan_fr:
                    cnt = st.get("scan_count", total)
                    widths.append(f3(X(cnt + 0.5) - ax0 + 2))
                    fin = cnt >= total
                    notes[k] = f"{render.fmt_int(cnt)} scanned" + (
                        "" if not fin else ("  ✓ accepted" if r["accepted"] else "  → keep the best"))
                else:
                    widths.append(widths[-1] if widths else "0")
            parts = [f'<clipPath id="{clip_id}"><rect x="{ax0 - 2}" y="{ay0 - 5}" width="0" height="{ah + 10}">'
                     + self.discrete("width", widths) + "</rect></clipPath>",
                     '<g display="none">' + self.visible(self.window(scan_fr[0], scan_fr[-1] + 1)),
                     f'<path d="{d}" stroke="{FAINT}" stroke-width="5" stroke-linecap="round" '
                     f'clip-path="url(#{clip_id})"/>']
            for c in (0, total):
                parts.append(self.text(X(c), ay0 + ah + 16, render.fmt_int(c), 12, color=MUTED, anchor="middle")
                             + "</text>")
            hit_frames = [k for k in scan_fr if self.F[k].get("scan_count", total) >= total]
            if hit_frames:
                hx = total if r["accepted"] else int(numpy.argmin(numpy.where(ok, err, numpy.inf))) + 1
                parts.append(f'<circle cx="{f3(X(hx))}" cy="{f3(Y(r["err"]))}" r="9" fill="none" '
                             f'stroke="{ACCENT}" stroke-width="2.4" display="none">'
                             + self.visible(self.window(hit_frames[0], scan_fr[-1] + 1)) + "</circle>")
            # scanned positions on the TI, in groups revealed frame by frame
            sc = numpy.asarray(r["scanned"], int)
            prev = 0
            for k in scan_fr:
                cnt = min(self.F[k].get("scan_count", total), len(sc))
                if cnt > prev:
                    pts = "".join(f"M{f3(self.ti_xy(p)[0])} {f3(self.ti_xy(p)[1])}h0" for p in sc[prev:cnt])
                    parts.append(f'<path d="{pts}" stroke="{THC}" stroke-width="4" stroke-linecap="round" '
                                 f'opacity="0.55" display="none">'
                                 + self.visible(self.window(k, scan_fr[-1] + 1)) + "</path>")
                    prev = cnt
            # moving data event while scanning (neutral colours), accepted one in colour
            moving = [k for k in scan_fr if self.F[k].get("ti_pos") is not None]
            if moving:
                p0 = numpy.array(a.chosen_pos(i))
                tx, ty = [], []
                for k in range(self.nf):
                    if k in moving:
                        q = numpy.array(self.F[k]["ti_pos"])
                        dx, dy = numpy.array(self.ti_xy(q)) - numpy.array(self.ti_xy(p0))
                        tx.append(f"{f3(dx)},{f3(dy)}")
                    else:
                        tx.append(tx[-1] if tx else "0,0")
                parts.append('<g display="none">' + self.visible([k in moving for k in range(self.nf)])
                             + (self.discrete_transform(tx) or "")
                             + self.pattern_group(i, p0, colored=False) + "</g>")
            parts.append("</g>")
            o.append("\n".join(parts))
        o.append(self.text_track(ax0 + aw - 4, ay0 + 14, notes, 15, anchor="end"))

    def discrete_transform(self, vals):
        changes = [(0, vals[0])]
        for k in range(1, len(vals)):
            if vals[k] != changes[-1][1]:
                changes.append((k, vals[k]))
        return (f'<animateTransform attributeName="transform" type="translate" dur="{self.T:.3f}s" '
                f'repeatCount="indefinite" calcMode="discrete" values="{";".join(v for _, v in changes)}" '
                f'keyTimes="{";".join(self.kt(k) for k, _ in changes)}"/>')

    def build_search_snesim(self):
        a = self.a
        o = self.out
        labels = []
        for st in self.F:
            i, ph = st.get("cur"), st.get("search_phase")
            if ph == "build":
                labels.append("Building the search trees")
            elif i is not None and i < a.N:
                labels.append(f"Search tree of grid level {a.levels[i]}")
            else:
                labels.append("Search tree")
        o.append(self.text_track(SEARCH_BOX.x, SEARCH_BOX.y - 22, labels, 17, weight="bold"))
        notes = ["" if (st.get("search_phase") == "build" or (st.get("cur") is None and st["filled"] == 0))
                 else "not scanned any more: only the search trees are used" for st in self.F]
        o.append(self.text_track(TI_BOX.x, TI_BOX.y + TI_BOX.h + 20, notes, 14, color=MUTED))
        SX = lambda x: SEARCH_BOX.x + x * SEARCH_BOX.w
        SY = lambda y: SEARCH_BOX.y + (1 - y) * SEARCH_BOX.h
        # --- construction: template stamp on the TI and tree sketch
        bf = [k for k, st in enumerate(self.F) if st.get("build_m") is not None]
        if bf:
            P = len(a.build_seq)
            reach = numpy.abs(a.build_tmpl).max(axis=0)
            c0 = a.build_centres[0]
            tx = []
            for k in range(self.nf):
                m = self.F[k].get("build_m")
                if m:
                    c = a.build_centres[min(m, P) - 1]
                    dx, dy = numpy.array(self.ti_xy(c)) - numpy.array(self.ti_xy(c0))
                    tx.append(f"{f3(dx)},{f3(dy)}")
                else:
                    tx.append(tx[-1] if tx else "0,0")
            cell_px = TI_BOX.w / a.ti.shape[1]
            stamp = [f'<g display="none">' + self.visible([bool(self.F[k].get("build_m")) for k in range(self.nf)])
                     + self.discrete_transform(tx)]
            bx, by = self.ti_xy(c0 - reach)
            stamp.append(f'<rect x="{f3(bx - cell_px / 2)}" y="{f3(by - cell_px / 2)}" '
                         f'width="{f3((2 * reach[1] + 1) * cell_px)}" height="{f3((2 * reach[0] + 1) * cell_px)}" '
                         f'fill="none" stroke="{ACCENT}" stroke-width="1.4"/>')
            for t in a.build_tmpl:
                x, y = self.ti_xy(c0 + t)
                stamp.append(f'<rect x="{f3(x - 4)}" y="{f3(y - 4)}" width="8" height="8" fill="{BG}" '
                             f'fill-opacity="0.6" stroke="{ACCENT}" stroke-width="1.4"/>')
            x, y = self.ti_xy(c0)
            stamp.append(f'<circle cx="{f3(x)}" cy="{f3(y)}" r="8" fill="none" stroke="{ACCENT}" stroke-width="2.4"/>')
            o.append("".join(stamp) + "</g>")
            # sketch: nodes appear when their first pattern is stored, radius follows the counts
            final_nodes, xy, _ = a.sketch(P)
            ms = sorted({self.F[k]["build_m"] for k in bf})
            first_frame = {}
            for k in bf:
                first_frame.setdefault(self.F[k]["build_m"], k)
            sketches = {m: a.sketch(m)[0] for m in ms}
            vis_build = [self.F[k].get("search_phase") == "build" and self.F[k].get("build_m") is not None
                         for k in range(self.nf)]
            g = ['<g display="none">' + self.visible(vis_build)]
            for key in final_nodes:
                x, y = xy[key]
                appear = next((first_frame[m] for m in ms if key in sketches[m]), None)
                if appear is None:
                    continue
                if key:
                    px0, py0 = xy[key[:-1]]
                    g.append(f'<line x1="{f3(SX(px0))}" y1="{f3(SY(py0))}" x2="{f3(SX(x))}" y2="{f3(SY(y))}" '
                             f'stroke="{FAINT}" stroke-width="1" display="none">'
                             + self.visible(self.window(appear, self.nf)) + "</line>")
                rs, cs = [], []
                for k in range(self.nf):
                    m = self.F[k].get("build_m")
                    c = sketches[m].get(key) if m is not None else None
                    if c is None:
                        rs.append(rs[-1] if rs else "0")
                        cs.append(cs[-1] if cs else FAINT)
                    else:
                        tot = c.sum()
                        rs.append(f3((6 + 26 * numpy.sqrt(tot / P)) / 2))
                        cs.append(to_hex(CMAP(c[1] / max(tot, 1))))
                g.append(f'<circle cx="{f3(SX(x))}" cy="{f3(SY(y))}" r="0" stroke="{DARK}" stroke-width="0.6" '
                         f'display="none">' + (self.discrete("r", rs) or "") + (self.discrete("fill", cs) or "")
                         + self.visible(self.window(appear, self.nf)) + "</circle>")
            g.append("</g>")
            o.append("\n".join(g))
            sk = [""] * self.nf
            for k in bf:
                m = self.F[k]["build_m"]
                sk[k] = f"grid level {a.cfg['levels']}: {render.fmt_int(m)} / {render.fmt_int(P)} patterns stored"
            o.append(self.text_track(SX(0.5), SY(0.07), sk, 14, anchor="middle"))
            o.append(self.text(SX(0.5), SY(0.0), f"(only the first {a.BUILD_DEPTH} levels of the tree are drawn)",
                               14, color=MUTED, anchor="middle", extra=' display="none"')
                     + self.visible(vis_build) + "</text>")
        for kk in range(len(a.tree_stats)):
            L = a.cfg["levels"] - kk
            pat, nod = a.tree_stats[L]
            mask = [self.F[k].get("search_phase") == "build" and self.F[k].get("build_list", 0) > kk
                    for k in range(self.nf)]
            o.append(self.text(SX(0.04), SY(0.86 - kk * 0.12),
                               f"✓  level {L}:  {render.fmt_int(pat)} patterns, {render.fmt_int(nod)} nodes", 17,
                               extra=' display="none"') + self.visible(mask) + "</text>")
        # --- descent of the detailed nodes
        for i in self.detail_nodes:
            fr = self.node_frames[i]
            ph = {k: self.F[k].get("search_phase") for k in fr}
            shown = [k for k in fr if ph[k] in ("descent", "stop", "draw", "pick", "done")]
            if not shown:
                continue
            r = a.R[i]
            steps = r["steps"]
            MAXR = 9
            first = max(0, len(steps) - (MAXR - 1))
            rows = [("root", None, numpy.asarray(r["root"]))] + [(k, steps[k][1], numpy.asarray(steps[k][2]))
                                                                 for k in range(first, len(steps))]
            root_tot = float(numpy.asarray(r["root"]).sum())
            y_top, dy = 0.93, 0.072
            ys = [y_top - j * dy - (0.03 if (j >= 1 and first > 0) else 0) for j in range(len(rows))]
            parts = ['<g display="none">' + self.visible(self.window(shown[0], shown[-1] + 1))]
            dots = []
            end = shown[-1] + 1
            for j, ((k, v, c), y) in enumerate(zip(rows, ys)):
                if k == "root":
                    appear = shown[0]
                else:
                    appear = next((f for f in fr if ph[f] != "descent" or self.F[f].get("tree_rows", 0) > k), end)
                tot = float(c.sum())
                wmax = 0.5 * numpy.log10(tot + 1) / numpy.log10(root_tot + 1)
                w0 = wmax * c[0] / max(tot, 1)
                w1 = wmax * c[1] / max(tot, 1)
                col = FAINT if v is None else to_hex(CMAP(float(v)))
                label = (f"root: {render.fmt_int(int(tot))} patterns" if k == "root"
                         else f"{render.fmt_int(int(tot))}  ({render.fmt_int(int(c[0]))} / {render.fmt_int(int(c[1]))})")
                row = (f'<g display="none">' + self.visible(self.window(appear, end))
                       + f'<rect x="{f3(SX(0.17))}" y="{f3(SY(y + 0.022))}" width="{f3(w0 * SEARCH_BOX.w)}" '
                         f'height="{f3(0.044 * SEARCH_BOX.h)}" fill="{to_hex(CMAP(0.0))}"/>'
                       + f'<rect x="{f3(SX(0.17 + w0))}" y="{f3(SY(y + 0.022))}" width="{f3(w1 * SEARCH_BOX.w)}" '
                         f'height="{f3(0.044 * SEARCH_BOX.h)}" fill="{to_hex(CMAP(1.0))}"/>'
                       + self.text(SX(0.19 + wmax), SY(y), label, 14) + "</text>"
                       + (f'<line x1="{f3(SX(0.08))}" y1="{f3(SY(ys[j - 1]))}" x2="{f3(SX(0.08))}" '
                          f'y2="{f3(SY(y))}" stroke="{FAINT}" stroke-width="1.6"/>' if j else "")
                       + f'<circle cx="{f3(SX(0.08))}" cy="{f3(SY(y))}" r="8" fill="{col}" stroke="{DARK}" '
                         f'stroke-width="0.8"/></g>')
                parts.append(row)
            if first > 0:
                parts.append(self.text(SX(0.025), SY(y_top - dy - 0.01), "⋮", 16, color=MUTED, anchor="middle")
                             + "</text>")
            stop_fr = [k for k in fr if ph[k] in ("stop", "draw", "pick", "done")] if r["stop"] is not None else []
            if stop_fr:
                ys_stop = ys[-1] - dy
                ignored = len(r["offsets"]) - r["used"]
                parts.append('<g display="none">' + self.visible(self.window(stop_fr[0], end))
                             + f'<line x1="{f3(SX(0.08))}" y1="{f3(SY(ys[-1]))}" x2="{f3(SX(0.08))}" '
                               f'y2="{f3(SY(ys_stop))}" stroke="{FAINT}" stroke-width="1.6"/>'
                             + f'<circle cx="{f3(SX(0.08))}" cy="{f3(SY(ys_stop))}" r="8" fill="{BG}" '
                               f'stroke="{THC}" stroke-width="1.2"/>'
                             + self.text(SX(0.13), SY(ys_stop),
                                         f"✗ no branch: stop ({ignored} node{'s' if ignored > 1 else ''} ignored)",
                                         14, color=THC) + "</text></g>")
            draw_fr = [k for k in fr if ph[k] in ("draw", "pick", "done")]
            if draw_fr:
                p0 = r["prob"][0]
                u = r["u"]
                d0 = draw_fr[0]
                grow = [k for k in draw_fr if "u_shown" in self.F[k]]
                g1 = grow[-1] + 1 if grow else d0 + 1
                parts.append('<g display="none">' + self.visible(self.window(d0, end))
                             + f'<rect x="{f3(SX(0.05))}" y="{f3(SY(0.14))}" width="{f3(0.9 * p0 * SEARCH_BOX.w)}" '
                               f'height="{f3(0.08 * SEARCH_BOX.h)}" fill="{to_hex(CMAP(0.0))}" stroke="{DARK}" '
                               f'stroke-width="0.8"/>'
                             + f'<rect x="{f3(SX(0.05 + 0.9 * p0))}" y="{f3(SY(0.14))}" '
                               f'width="{f3(0.9 * (1 - p0) * SEARCH_BOX.w)}" height="{f3(0.08 * SEARCH_BOX.h)}" '
                               f'fill="{to_hex(CMAP(1.0))}" stroke="{DARK}" stroke-width="0.8"/>'
                             + (self.text(SX(0.05 + 0.45 * p0), SY(0.18), f"background {p0:.2f}", 13, anchor="middle")
                                + "</text>" if p0 > 0.18 else "")
                             + (self.text(SX(0.05 + 0.9 * p0 + 0.45 * (1 - p0)), SY(0.18), f"channel {1 - p0:.2f}",
                                          13, anchor="middle") + "</text>" if 1 - p0 > 0.18 else "")
                             + f'<g><path d="M-8 -12H8L0 0Z" fill="{THC}"/>'
                             + self.text(0, 26, f"u = {u:.2f}", 13, color=THC, anchor="middle") + "</text>"
                             + f'<animateTransform attributeName="transform" type="translate" dur="{self.T:.3f}s" '
                               f'repeatCount="indefinite" calcMode="linear" '
                               f'values="{f3(SX(0.05))},{f3(SY(0.035))};{f3(SX(0.05))},{f3(SY(0.035))};'
                               f'{f3(SX(0.05 + 0.9 * u))},{f3(SY(0.035))};{f3(SX(0.05 + 0.9 * u))},{f3(SY(0.035))}" '
                               f'keyTimes="0;{max(d0, 1) / self.nf:.5f};{min(g1, self.nf - 1) / self.nf:.5f};1"/></g>'
                             + "</g>")
            parts.append("</g>")
            o.append("\n".join(parts))
        # neighbour followed in the tree, highlighted in the simulation
        spacing = SIM_BOX.w / a.shape[1]
        xs, ys_, vis = [], [], []
        for st in self.F:
            i, foc = st.get("cur"), st.get("focus")
            if i is not None and foc is not None and i < a.N and foc < len(a.R[i]["offsets"]):
                x, y = self.sim_xy(numpy.array(a.cells[i]) + a.R[i]["offsets"][foc])
                xs.append(f3(x)), ys_.append(f3(y)), vis.append(True)
            else:
                xs.append(xs[-1] if xs else 0), ys_.append(ys_[-1] if ys_ else 0), vis.append(False)
        o.append(f'<circle r="{f3(spacing)}" fill="none" stroke="{THC}" stroke-width="2.4" display="none">'
                 + (self.discrete("cx", xs) or "") + (self.discrete("cy", ys_) or "") + self.visible(vis)
                 + "</circle>")

    # ---- document ------------------------------------------------------------------------------
    def build(self):
        self.frame_info()
        self.build_static()
        if isinstance(self.a, render.QSAnimation):
            self.build_search_qs()
        elif isinstance(self.a, render.DSAnimation):
            self.build_search_ds()
        else:
            self.build_search_snesim()
        self.build_dots()
        self.build_events()
        self.build_ring()
        self.build_patterns()
        self.build_steps()
        self.build_fly()
        body = "\n".join(self.out)
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
                f'font-family="{FONT}">\n<title>{escape(self.a.title)} ({escape(self.a.ti_name)})</title>\n'
                f'{body}\n</svg>\n')


def main():
    name = sys.argv[1]
    theme = sys.argv[2] if len(sys.argv) > 2 else "light"
    use_theme(theme)
    suffix = "" if theme == "light" else f"_{theme}"
    anim = render.CLASSES[name.split("_")[0]](name)
    doc = SVG(anim).build()
    os.makedirs(os.path.join(render.SP, "out"), exist_ok=True)
    out = os.path.join(render.SP, "out", f"{name}{suffix}.svg")
    with open(out, "w") as f:
        f.write(doc)
    print(f"{name}: {len(doc) / 1e6:.2f} MB -> {out}")


if __name__ == "__main__":
    main()
