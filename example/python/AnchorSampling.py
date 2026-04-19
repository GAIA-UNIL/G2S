from argparse import ArgumentParser
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


def load_g2s():
    try:
        from g2s import g2s

        return g2s
    except ImportError as first_error:
        repo_root = Path(__file__).resolve().parents[2]
        python_build = repo_root / "build" / "python-build"
        if python_build.exists():
            sys.path.insert(0, str(python_build))
            try:
                from g2s import g2s

                return g2s
            except ImportError as second_error:
                raise ImportError(
                    "Unable to import the Python g2s client. "
                    "Build/install the Python interface and ensure pyzmq is available."
                ) from second_error
        raise ImportError(
            "Unable to import the Python g2s client. "
            "Build/install the Python interface first."
        ) from first_error


def build_synthetic_case(height=100, width=120, num_ti=7, keep_ratio=0.05, seed=4):
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    x_norm = (x - 0.5 * (width - 1)) / (0.5 * width)
    y_norm = (y - 0.5 * (height - 1)) / (0.5 * height)

    tis = []
    for ti_id in range(num_ti):
        stretch_x = 0.9 + 0.10 * ti_id
        stretch_y = 1.15 - 0.05 * ti_id
        radius = np.sqrt((x_norm / stretch_x) ** 2 + (y_norm / stretch_y) ** 2)
        disk = 1.0 / (1.0 + np.exp((radius - (0.24 + 0.055 * ti_id)) / 0.035))
        stripes = 0.5 + 0.5 * np.sin(6.0 * x_norm + 3.5 * y_norm + 0.55 * ti_id)
        plume = 0.5 + 0.5 * np.cos(4.0 * radius - 0.35 * ti_id + 0.8 * x_norm)
        tis.append((0.60 * disk + 0.25 * stripes + 0.15 * plume).astype(np.float32))
    tis = np.stack(tis, axis=0)

    x01 = x / max(width - 1, 1)
    y01 = y / max(height - 1, 1)
    truth_index = np.clip(
        np.rint(
            (num_ti - 1)
            * (0.10 + 0.78 * x01 + 0.08 * np.sin(2.0 * np.pi * y01) - 0.04 * np.cos(2.5 * np.pi * x01 * y01))
        ),
        0,
        num_ti - 1,
    ).astype(np.int32)

    row = np.arange(height)[:, None]
    col = np.arange(width)[None, :]
    truth = tis[truth_index, row, col].astype(np.float32)

    rng = np.random.default_rng(seed)
    conditioning = np.full_like(truth, np.nan, dtype=np.float32)
    known = rng.random(truth.shape) < keep_ratio
    known[:4, :] = True
    known[-4:, :] = True
    known[:, :4] = True
    known[:, -4:] = True
    conditioning[known] = truth[known]

    mask = np.full((height, width, num_ti), np.nan, dtype=np.float32)
    rr, cc = np.indices(truth_index.shape)
    mask[rr, cc, truth_index] = 1.0

    return tis, truth, truth_index, conditioning, mask


def run_as(g2s, tis, conditioning, mask, k, n, seed, jobs):
    args = ["-a", "as"]
    for ti in tis:
        args.extend(["-ti", ti.astype(np.float32, copy=False)])
    args.extend(
        [
            "-di",
            conditioning.astype(np.float32, copy=False),
            "-mi",
            mask.astype(np.float32, copy=False),
            "-dt",
            np.array([0.0], dtype=np.float32),
            "-k",
            int(k),
            "-n",
            int(n),
            "-s",
            int(seed),
            "-j",
            int(jobs),
            "-silent",
        ]
    )
    simulation, selected_ti, *_ = g2s(*args)
    return simulation, selected_ti.astype(np.int64, copy=False)


def rmse(reference, estimate, valid_mask):
    if not np.any(valid_mask):
        return float("nan")
    delta = estimate[valid_mask] - reference[valid_mask]
    return float(np.sqrt(np.mean(delta * delta)))


def show_minimal_figure(truth, conditioning, simulation, selected_ti):
    conditioning_display = np.nan_to_num(conditioning, nan=-0.2)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharex=True, sharey=True)
    fig.suptitle("Anchor Sampling: minimal synthetic demo")

    for ax, image, title in zip(
        axes,
        [truth, conditioning_display, simulation, selected_ti],
        ["Ground truth", "Conditioning", "AS result", "Selected TI id"],
    ):
        ax.imshow(image, cmap="viridis")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    parser = ArgumentParser(description="Minimal Anchor Sampling synthetic example.")
    parser.add_argument("--height", type=int, default=100)
    parser.add_argument("--width", type=int, default=120)
    parser.add_argument("--num-ti", type=int, default=7)
    parser.add_argument("--keep-ratio", type=float, default=0.05)
    parser.add_argument("--k", type=int, default=None, help="defaults to num-ti")
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()

    g2s = load_g2s()
    tis, truth, truth_index, conditioning, mask = build_synthetic_case(
        height=args.height,
        width=args.width,
        num_ti=args.num_ti,
        keep_ratio=args.keep_ratio,
        seed=args.seed,
    )
    simulation, selected_ti = run_as(
        g2s,
        tis,
        conditioning,
        mask,
        k=args.num_ti if args.k is None else args.k,
        n=args.n,
        seed=args.seed,
        jobs=args.jobs,
    )

    unknown = np.isnan(conditioning)
    agreement = float(np.mean(selected_ti[unknown] == truth_index[unknown]))
    print(f"Unknown-cell RMSE: {rmse(truth, simulation, unknown):.4f}")
    print(f"Unknown-cell TI-index agreement: {agreement:.4f}")

    show_minimal_figure(truth, conditioning, simulation, selected_ti)


if __name__ == "__main__":
    main()
