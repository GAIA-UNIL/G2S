#!/usr/bin/env python3
"""QS performance benchmark.

Runs the standalone `qs` binary on a fixed set of scenarios and reports the
best (minimum) wall and CPU time over --repeat runs of each, together with a hash of the simulation and index
outputs, so that an optimisation can be checked to leave results unchanged.

    python3 benchmark/qs_benchmark.py --qs build/c++-build/qs
    python3 benchmark/qs_benchmark.py --qs build/c++-build/qs --ab /path/to/baseline/qs

With --ab the two binaries are run alternately and the best wall and CPU time
of each is reported, which keeps comparisons meaningful on a loaded machine.

Inputs are written to /tmp/G2S/data (where qs reads them) and removed after.
"""

import argparse
import hashlib
import json
import os
import resource
import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = Path("/tmp/G2S/data")

CONTINUOUS, CATEGORICAL = 0, 1


# --------------------------------------------------------------------------- IO

def read_png_gray(path):
    """Minimal PNG decoder for 8-bit grayscale or palette images."""
    raw = Path(path).read_bytes()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n", path
    pos, idat, palette = 8, b"", None
    while pos < len(raw):
        length, ctype = struct.unpack(">I4s", raw[pos:pos + 8])
        chunk = raw[pos + 8:pos + 8 + length]
        pos += 12 + length
        if ctype == b"IHDR":
            width, height, depth, color = struct.unpack(">IIBB", chunk[:10])
            assert depth == 8 and color in (0, 3), (path, depth, color)
        elif ctype == b"PLTE":
            palette = np.frombuffer(chunk, np.uint8).reshape(-1, 3)
        elif ctype == b"IDAT":
            idat += chunk
    data = np.frombuffer(zlib.decompress(idat), np.uint8)
    stride = width + 1
    out = np.zeros((height, width), np.int32)
    prev = np.zeros(width, np.int32)
    for y in range(height):
        f = data[y * stride]
        line = data[y * stride + 1:(y + 1) * stride].astype(np.int32)
        cur = np.zeros(width, np.int32)
        if f == 0:
            cur = line
        elif f == 2:
            cur = (line + prev) & 255
        else:
            for x in range(width):
                a = cur[x - 1] if x else 0
                b = prev[x]
                c = prev[x - 1] if x else 0
                if f == 1:
                    p = a
                elif f == 3:
                    p = (a + b) >> 1
                else:  # Paeth
                    pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                    p = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                cur[x] = (line[x] + p) & 255
        out[y] = cur
        prev = cur
    if palette is not None:
        rgb = palette[out]
        return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
    return out.astype(np.float32)


def write_bgrid(name, array, var_type):
    """Write a single-variable DataImage. `array` is indexed [z][y][x]."""
    dims = list(array.shape[::-1])  # G2S stores x first
    header = struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}I", *dims)
    header += struct.pack("<II", 1, var_type) + struct.pack("<I", 0)  # 1 var, Float encoding
    payload = np.ascontiguousarray(array, dtype=np.float32).tobytes()
    full = 8 + len(header) + len(payload)
    (DATA_DIR / f"{name}.bgrid").write_bytes(struct.pack("<Q", full) + header + payload)


def read_bgrid_payload(name):
    for suffix, opener in ((".bgrid", lambda p: p.read_bytes()),
                           (".bgrid.gz", lambda p: zlib.decompress(p.read_bytes(), 31))):
        path = DATA_DIR / f"{name}{suffix}"
        if path.exists():
            return opener(path)
    return None


# ------------------------------------------------------------------- scenarios

def smooth_field_3d(shape, sigma, seed):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(shape)
    freqs = np.meshgrid(*[np.fft.fftfreq(n) for n in shape], indexing="ij")
    k2 = sum(f * f for f in freqs)
    field = np.real(np.fft.ifftn(np.fft.fftn(noise) * np.exp(-2 * (np.pi * sigma) ** 2 * k2)))
    return ((field - field.mean()) / field.std()).astype(np.float32)


def prepare_inputs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    stone = read_png_gray(REPO / "browser/deploy/assets/Stone.png")
    strebelle = read_png_gray(REPO / "browser/deploy/assets/Strebelle.png")
    gobi = read_png_gray(REPO / "build/TrainingImages/gobi_dune.png")
    write_bgrid("bench_stone", stone, CONTINUOUS)
    write_bgrid("bench_strebelle", (strebelle > 127).astype(np.float32), CATEGORICAL)
    write_bgrid("bench_gobi", gobi, CONTINUOUS)
    write_bgrid("bench_3d", smooth_field_3d((32, 64, 64), 3.0, 1), CONTINUOUS)
    write_bgrid("bench_di_200", np.full((200, 200), np.nan, np.float32), CONTINUOUS)
    write_bgrid("bench_di_250c", np.full((250, 250), np.nan, np.float32), CATEGORICAL)
    write_bgrid("bench_di_120", np.full((120, 120), np.nan, np.float32), CONTINUOUS)
    write_bgrid("bench_di_3d", np.full((16, 32, 32), np.nan, np.float32), CONTINUOUS)


# name, ti, di, extra args. All use a fixed seed so single-thread runs are reproducible.
SCENARIOS = [
    ("2d_cont_stone_j1", "bench_stone", "bench_di_200", ["-k", "1.2", "-n", "50", "-j", "1"]),
    ("2d_cat_strebelle_j1", "bench_strebelle", "bench_di_250c", ["-k", "1.2", "-n", "50", "-j", "1"]),
    ("2d_cont_gobi_j1", "bench_gobi", "bench_di_120", ["-k", "1.2", "-n", "50", "-j", "1"]),
    ("3d_cont_j1", "bench_3d", "bench_di_3d", ["-k", "1.2", "-n", "50", "-j", "1"]),
    ("2d_cont_gobi_j8", "bench_gobi", "bench_di_120", ["-k", "1.2", "-n", "50", "-j", "8"]),
    ("3d_cont_j8", "bench_3d", "bench_di_3d", ["-k", "1.2", "-n", "50", "-j", "8"]),
]


def run_scenario(qs, name, ti, di, extra, run_id):
    """Run qs once; return (wall seconds, child CPU seconds, output hash)."""
    qs = Path(qs).resolve()
    cmd = [str(qs), "-ti", ti, "-di", di, "-s", "100", "-id", str(run_id), *extra]
    cpu0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=qs.parent, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    cpu1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (cpu1.ru_utime - cpu0.ru_utime) + (cpu1.ru_stime - cpu0.ru_stime)
    if proc.returncode != 0:
        sys.exit(f"{name}: qs failed ({proc.returncode})\n{proc.stderr[-2000:]}")
    digest = hashlib.sha256()
    for prefix in ("im_1_", "im_2_"):
        payload = read_bgrid_payload(f"{prefix}{run_id}")
        if payload is None:
            sys.exit(f"{name}: missing output {prefix}{run_id}\n{proc.stderr[-2000:]}")
        digest.update(payload)
    for f in DATA_DIR.glob(f"im_*_{run_id}.*"):
        f.unlink()
    return elapsed, cpu, digest.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qs", default=str(REPO / "build/c++-build/qs"))
    ap.add_argument("--ab", help="baseline qs binary: alternate runs with --qs and report both")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--only", help="comma-separated scenario names")
    ap.add_argument("--json", help="write results to this file")
    args = ap.parse_args()

    prepare_inputs()
    only = set(args.only.split(",")) if args.only else None
    binaries = [("new", args.qs)] + ([("base", args.ab)] if args.ab else [])
    results = {}
    run_id = 900000
    try:
        for name, ti, di, extra in SCENARIOS:
            if only and name not in only:
                continue
            stats = {label: {"wall": [], "cpu": [], "hashes": set()} for label, _ in binaries}
            for _ in range(args.repeat):
                for label, qs in binaries:  # interleave so load changes hit both equally
                    run_id += 1
                    wall, cpu, digest = run_scenario(qs, name, ti, di, extra, run_id)
                    stats[label]["wall"].append(wall)
                    stats[label]["cpu"].append(cpu)
                    stats[label]["hashes"].add(digest)
            results[name] = {label: {"min_wall_s": min(v["wall"]), "min_cpu_s": min(v["cpu"]),
                                     "wall_s": v["wall"], "cpu_s": v["cpu"], "hashes": sorted(v["hashes"])}
                             for label, v in stats.items()}
            new = results[name]["new"]
            line = (f"{name:22s} wall {new['min_wall_s']:7.2f}s  cpu {new['min_cpu_s']:7.2f}s"
                    f"  hash {','.join(new['hashes'])}")
            if args.ab:
                base = results[name]["base"]
                same = new["hashes"] == base["hashes"]
                line += (f"  | base wall {base['min_wall_s']:7.2f}s cpu {base['min_cpu_s']:7.2f}s"
                         f"  speedup wall x{base['min_wall_s'] / new['min_wall_s']:.2f}"
                         f" cpu x{base['min_cpu_s'] / new['min_cpu_s']:.2f}"
                         f"  {'same output' if same else 'OUTPUT DIFFERS'}")
            print(line, flush=True)
    finally:
        for f in DATA_DIR.glob("bench_*.bgrid"):
            f.unlink()
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
