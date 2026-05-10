from argparse import ArgumentParser
from pathlib import Path
import sys

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
                    "Build/install the Python interface and ensure its runtime "
                    "dependencies, including pyzmq, are available."
                ) from second_error
        raise ImportError(
            "Unable to import the Python g2s client. "
            "Build/install the Python interface first."
        ) from first_error


def make_ti_stack(height=100, width=120, num_ti=7):
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    x = (x - 0.5 * (width - 1)) / (0.5 * width)
    y = (y - 0.5 * (height - 1)) / (0.5 * height)

    tis = []
    for ti_id in range(num_ti):
        stretch_x = 0.9 + 0.10 * ti_id
        stretch_y = 1.15 - 0.05 * ti_id
        radius = np.sqrt((x / stretch_x) ** 2 + (y / stretch_y) ** 2)

        disk_radius = 0.24 + 0.055 * ti_id
        phase = 0.55 * ti_id
        disk = 1.0 / (1.0 + np.exp((radius - disk_radius) / 0.035))
        stripes = 0.5 + 0.5 * np.sin(6.0 * x + 3.5 * y + phase)
        plume = 0.5 + 0.5 * np.cos(4.0 * radius - 0.35 * ti_id + 0.8 * x)

        ti = 0.60 * disk + 0.25 * stripes + 0.15 * plume
        tis.append(ti.astype(np.float32))
    return np.stack(tis, axis=0)


def make_truth_index(height, width, num_ti):
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    x = x / max(width - 1, 1)
    y = y / max(height - 1, 1)

    smooth_index = (num_ti - 1) * (
        0.10
        + 0.78 * x
        + 0.08 * np.sin(2.0 * np.pi * y)
        - 0.04 * np.cos(2.5 * np.pi * x * y)
    )
    return np.clip(np.rint(smooth_index), 0, num_ti - 1).astype(np.int32)


def make_truth_image(tis, truth_index):
    height, width = truth_index.shape
    row = np.arange(height)[:, None]
    col = np.arange(width)[None, :]
    return tis[truth_index, row, col].astype(np.float32)


def make_conditioning(truth, keep_ratio=0.05, seed=4):
    rng = np.random.default_rng(seed)
    conditioning = np.full_like(truth, np.nan, dtype=np.float32)
    known = rng.random(truth.shape) < keep_ratio

    known[:4, :] = True
    known[-4:, :] = True
    known[:, :4] = True
    known[:, -4:] = True

    conditioning[known] = truth[known]
    return conditioning


def make_mask(truth_index, num_ti):
    weights = np.full(truth_index.shape + (num_ti,), np.nan, dtype=np.float32)
    rows, cols = np.indices(truth_index.shape)
    weights[rows, cols, truth_index] = 1.0
    return weights


def build_synthetic_case(height=100, width=120, num_ti=7):
    tis = make_ti_stack(height=height, width=width, num_ti=num_ti)
    truth_index = make_truth_index(height=height, width=width, num_ti=num_ti)
    truth = make_truth_image(tis, truth_index)
    conditioning = make_conditioning(truth)
    mask = make_mask(truth_index, num_ti=num_ti)
    return {
        "tis": tis,
        "truth_index": truth_index,
        "truth": truth,
        "conditioning": conditioning,
        "mask": mask,
        "dt": np.array([0.0], dtype=np.float32),
        "suggested_k": 3,
        "suggested_masked_k": num_ti,
        "suggested_n": 24,
        "seed": 17,
    }


def decode_index_image(raw_index, num_ti):
    raw_index = raw_index.astype(np.int64, copy=False)
    if np.any((raw_index < 0) | (raw_index >= num_ti)):
        raise ValueError("AS index image should contain TI ids in [0, n_ti).")
    return raw_index


def infer_ti_from_simulation(dataset, simulation):
    tis = dataset["tis"]
    residual = np.abs(tis - simulation[None, :, :])
    inferred_ti = np.argmin(residual, axis=0).astype(np.int64)
    inferred_residual = np.min(residual, axis=0).astype(np.float32)
    return inferred_ti, inferred_residual


def run_anchor_sampling(g2s, dataset, server_address, jobs, seed, use_mask, candidate_k):
    tis = [ti.astype(np.float32, copy=False) for ti in dataset["tis"]]
    arguments = [
        "-sa",
        server_address,
        "-a",
        "as",
        "-ti",
        tis,
        "-di",
        dataset["conditioning"].astype(np.float32, copy=False),
        "-dt",
        dataset["dt"],
        "-k",
        int(candidate_k),
        "-n",
        int(dataset["suggested_n"]),
        "-s",
        int(seed),
        "-j",
        int(jobs),
        "-silent",
    ]
    if use_mask:
        arguments.extend(["-mi", dataset["mask"].astype(np.float32, copy=False)])
    simulation, raw_index, *extra = g2s(*arguments)
    return simulation, raw_index, extra


def rmse(reference, estimate, valid_mask):
    if not np.any(valid_mask):
        return float("nan")
    delta = estimate[valid_mask] - reference[valid_mask]
    return float(np.sqrt(np.mean(delta * delta)))


def summarize_counts(values):
    unique_values, counts = np.unique(values, return_counts=True)
    return ", ".join(
        f"{int(value)}:{int(count)}" for value, count in zip(unique_values, counts)
    )


def print_index_debug(name, run, unknown, debug=False):
    raw_unknown = run["selected_ti"][unknown]
    inferred_unknown = run["inferred_ti"][unknown]
    residual_unknown = run["inferred_residual"][unknown]

    print(f"{name} raw TI ids on unknown cells: [{summarize_counts(raw_unknown)}]")
    print(
        f"{name} inferred TI ids from simulation on unknown cells: "
        f"[{summarize_counts(inferred_unknown)}]"
    )
    if debug:
        print(f"{name} first 12x12 raw TI ids:\n{run['selected_ti'][:12, :12]}")
        print(f"{name} first 12x12 inferred TI ids:\n{run['inferred_ti'][:12, :12]}")
    print(
        f"{name} inference residual on unknown cells: "
        f"min={float(np.min(residual_unknown)):.6g}, "
        f"mean={float(np.mean(residual_unknown)):.6g}, "
        f"max={float(np.max(residual_unknown)):.6g}"
    )


def summarize_run(name, dataset, run):
    truth = dataset["truth"]
    truth_index = dataset["truth_index"].astype(np.int64)
    conditioning = dataset["conditioning"]
    unknown = np.isnan(conditioning)

    print(
        f"{name}: "
        f"RMSE on unknown cells={rmse(truth, run['simulation'], unknown):.4f}, "
        f"TI-index agreement on unknown cells={float(np.mean(run['selected_ti'][unknown] == truth_index[unknown])):.4f}, "
        f"selected-TI histogram on unknown cells=[{summarize_counts(run['selected_ti'][unknown])}], "
        f"raw-vs-inferred mismatch on unknown cells={float(np.mean(run['selected_ti'][unknown] != run['inferred_ti'][unknown])):.4f}, "
        f"max inference residual on unknown cells={float(np.max(run['inferred_residual'][unknown])):.6g}"
    )


def build_center_path(shape):
    path = np.full(shape, 1000.0, dtype=np.float32)
    center = (shape[0] // 2, shape[1] // 2)
    path[center] = 0.0
    return path


def run_single_pixel_case(g2s, server_address, jobs, seed, tis, conditioning, mask, k, n):
    simulation, raw_index, *_ = g2s(
        "-sa",
        server_address,
        "-a",
        "as",
        "-ti",
        [ti.astype(np.float32, copy=False) for ti in tis],
        "-di",
        conditioning.astype(np.float32, copy=False),
        "-sp",
        build_center_path(conditioning.shape),
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
    )
    center = (conditioning.shape[0] // 2, conditioning.shape[1] // 2)
    return int(raw_index[center]), float(simulation[center])


def run_mi_edge_cases(g2s, server_address, jobs, seed, num_ti):
    size_y, size_x = 11, 13
    center = (size_y // 2, size_x // 2)
    neighbor_pattern = np.zeros((size_y, size_x), dtype=np.float32)
    neighbor_pattern[center[0] - 1, center[1]] = 1.0
    neighbor_pattern[center[0] + 1, center[1]] = 2.0
    neighbor_pattern[center[0], center[1] - 1] = 3.0
    neighbor_pattern[center[0], center[1] + 1] = 4.0

    conditioning = neighbor_pattern.copy()
    conditioning[center] = np.nan

    print("\nMI edge cases:")

    # Case 1: all TI neighborhoods are identical, only the center value differs.
    # The one-hot mask should deterministically select the requested TI.
    tie_tis = np.repeat(neighbor_pattern[None, :, :], num_ti, axis=0)
    for ti in range(num_ti):
        tie_tis[ti, center[0], center[1]] = 10.0 + ti
    tie_target = max(num_ti - 2, 0)
    tie_mask = np.zeros((size_y, size_x, num_ti), dtype=np.float32)
    tie_mask[center[0], center[1], tie_target] = 1.0
    selected_ti, selected_value = run_single_pixel_case(
        g2s, server_address, jobs, seed, tie_tis, conditioning, tie_mask, k=num_ti, n=4
    )
    print(
        f"  tie case: expected TI {tie_target}, got TI {selected_ti}, "
        f"center value {selected_value:.3f}"
    )

    # Case 2: top-k blocking. TI 0 matches the neighborhood best, but the prior
    # requests the last TI. With k=1 the mask must not win; with k=n_ti it can.
    blocking_tis = np.repeat(neighbor_pattern[None, :, :], num_ti, axis=0)
    for ti in range(num_ti):
        blocking_tis[ti, center[0], center[1]] = 20.0 + ti
        blocking_tis[ti, center[0] - 1, center[1]] += 0.35 * ti
        blocking_tis[ti, center[0], center[1] + 1] += 0.20 * ti
    blocking_mask = np.zeros((size_y, size_x, num_ti), dtype=np.float32)
    blocking_mask[center[0], center[1], num_ti - 1] = 1.0
    selected_k1, _ = run_single_pixel_case(
        g2s, server_address, jobs, seed, blocking_tis, conditioning, blocking_mask, k=1, n=4
    )
    selected_kall, _ = run_single_pixel_case(
        g2s, server_address, jobs, seed, blocking_tis, conditioning, blocking_mask, k=num_ti, n=4
    )
    print(
        f"  top-k blocking case: with k=1 got TI {selected_k1}, "
        f"with k={num_ti} got TI {selected_kall}"
    )

    # Case 3: NaN exclusion. The preferred TI is invalidated by NaN in the mask,
    # so the next positive-weight TI should be chosen.
    nan_mask = np.zeros((size_y, size_x, num_ti), dtype=np.float32)
    nan_mask[center[0], center[1], 1] = np.nan
    nan_mask[center[0], center[1], 2] = 1.0
    selected_nan, _ = run_single_pixel_case(
        g2s, server_address, jobs, seed, tie_tis, conditioning, nan_mask, k=num_ti, n=4
    )
    print(f"  NaN exclusion case: preferred TI 1 invalidated, got TI {selected_nan}")


def show_figure(dataset, unmasked, masked):
    try:
        import matplotlib.pyplot as plt
        from matplotlib import colors
        from matplotlib import colormaps
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to display the demo figure") from exc

    truth = dataset["truth"]
    truth_index = dataset["truth_index"].astype(np.int64)
    mask = dataset["mask"]
    conditioning = dataset["conditioning"]
    unknown = np.isnan(conditioning)
    num_ti = int(dataset["tis"].shape[0])

    conditioning_display = np.nan_to_num(conditioning, nan=-0.2)
    valid_mask = ~np.isnan(mask)
    any_valid_mask = np.any(valid_mask, axis=-1)
    safe_mask = np.where(valid_mask, mask, -np.inf)
    mask_argmax = np.full(mask.shape[:2], np.nan, dtype=np.float32)
    mask_peak = np.full(mask.shape[:2], np.nan, dtype=np.float32)
    if np.any(any_valid_mask):
        mask_argmax[any_valid_mask] = np.argmax(safe_mask, axis=-1)[any_valid_mask]
        mask_peak[any_valid_mask] = np.max(safe_mask, axis=-1)[any_valid_mask]

    discrete_cmap = colormaps.get_cmap("tab10").resampled(max(num_ti, 1)).copy()
    discrete_cmap.set_bad(color="#d9d9d9")
    discrete_norm = colors.BoundaryNorm(
        np.arange(-0.5, num_ti + 0.5, 1.0), discrete_cmap.N
    )

    mismatch_cmap = colormaps.get_cmap("gray_r").resampled(2).copy()
    mismatch_cmap.set_bad(color="#d9d9d9")

    fig, axes = plt.subplots(4, 4, figsize=(16, 16), sharex=True, sharey=True)
    axes = axes.ravel()

    panels = [
        (dataset["tis"][0], "TI 0", "continuous"),
        (
            dataset["tis"][len(dataset["tis"]) // 2],
            f"TI {len(dataset['tis']) // 2}",
            "continuous",
        ),
        (dataset["tis"][-1], f"TI {len(dataset['tis']) - 1}", "continuous"),
        (truth, "Ground truth", "continuous"),
        (conditioning_display, "Conditioning", "continuous"),
        (truth_index, "Ground-truth TI index", "discrete"),
        (mask_argmax, "Argmax of -mi", "discrete"),
        (mask_peak, "Peak weight of -mi", "continuous_bar"),
        (unmasked["simulation"], "AS without -mi", "continuous"),
        (
            np.where(unknown, unmasked["selected_ti"], np.nan),
            "Raw TI id without -mi",
            "discrete",
        ),
        (
            np.where(unknown, unmasked["inferred_ti"], np.nan),
            "Inferred TI without -mi",
            "discrete",
        ),
        (
            np.where(
                unknown,
                unmasked["selected_ti"] != unmasked["inferred_ti"],
                np.nan,
            ),
            "Mismatch without -mi",
            "mismatch",
        ),
        (masked["simulation"], "AS with -mi", "continuous"),
        (
            np.where(unknown, masked["selected_ti"], np.nan),
            "Raw TI id with -mi",
            "discrete",
        ),
        (
            np.where(unknown, masked["inferred_ti"], np.nan),
            "Inferred TI with -mi",
            "discrete",
        ),
        (
            np.where(
                unknown,
                masked["selected_ti"] != masked["inferred_ti"],
                np.nan,
            ),
            "Mismatch with -mi",
            "mismatch",
        ),
    ]

    for ax, (image, title, kind) in zip(axes, panels):
        if kind == "discrete":
            handle = ax.imshow(image, cmap=discrete_cmap, norm=discrete_norm)
            fig.colorbar(
                handle, ax=ax, fraction=0.046, pad=0.04, ticks=np.arange(num_ti)
            )
        elif kind == "mismatch":
            handle = ax.imshow(image, cmap=mismatch_cmap, vmin=0, vmax=1)
            fig.colorbar(handle, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1])
        else:
            handle = ax.imshow(image, cmap="viridis")
            if kind == "continuous_bar":
                fig.colorbar(handle, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(
        "Anchor Sampling diagnostics on a 100x120 grid\n"
        f"Unknown-cell RMSE without -mi: {rmse(truth, unmasked['simulation'], unknown):.4f} | "
        f"with -mi: {rmse(truth, masked['simulation'], unknown):.4f}",
        fontsize=12,
    )
    fig.tight_layout()
    plt.show()


def main():
    parser = ArgumentParser(
        description=(
            "Generate a rectangular synthetic Anchor Sampling case, run AS with and "
            "without -mi, print diagnostics, and keep the result window open."
        )
    )
    parser.add_argument("--server-address", default="localhost")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--height", type=int, default=100)
    parser.add_argument("--width", type=int, default=120)
    parser.add_argument("--num-ti", type=int, default=7)
    parser.add_argument("--unmasked-k", type=int, default=3)
    parser.add_argument("--masked-k", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--skip-edge-cases", action="store_true")
    args = parser.parse_args()

    dataset = build_synthetic_case(
        height=args.height,
        width=args.width,
        num_ti=args.num_ti,
    )
    g2s = load_g2s()
    unknown = np.isnan(dataset["conditioning"])
    masked_k = dataset["suggested_masked_k"] if args.masked_k is None else args.masked_k

    simulation_unmasked, raw_index_unmasked, _ = run_anchor_sampling(
        g2s,
        dataset,
        args.server_address,
        args.jobs,
        args.seed,
        use_mask=False,
        candidate_k=args.unmasked_k,
    )
    selected_ti_unmasked = decode_index_image(raw_index_unmasked, args.num_ti)
    inferred_ti_unmasked, inferred_residual_unmasked = infer_ti_from_simulation(
        dataset, simulation_unmasked
    )
    unmasked = {
        "simulation": simulation_unmasked,
        "selected_ti": selected_ti_unmasked,
        "inferred_ti": inferred_ti_unmasked,
        "inferred_residual": inferred_residual_unmasked,
    }

    simulation_masked, raw_index_masked, _ = run_anchor_sampling(
        g2s,
        dataset,
        args.server_address,
        args.jobs,
        args.seed,
        use_mask=True,
        candidate_k=masked_k,
    )
    selected_ti_masked = decode_index_image(raw_index_masked, args.num_ti)
    inferred_ti_masked, inferred_residual_masked = infer_ti_from_simulation(
        dataset, simulation_masked
    )
    masked = {
        "simulation": simulation_masked,
        "selected_ti": selected_ti_masked,
        "inferred_ti": inferred_ti_masked,
        "inferred_residual": inferred_residual_masked,
    }

    print("Synthetic case:")
    print(f"  grid shape = {args.height} x {args.width}")
    print(f"  number of TIs = {args.num_ti}")
    print(f"  unknown-cell fraction = {float(np.mean(unknown)):.4f}")
    print(
        f"Run parameters: without -mi uses k={args.unmasked_k}, "
        f"with -mi uses k={masked_k}"
    )
    print(
        "Mask semantics in this demo: hard exclusion mask, 1 for the preferred TI "
        "and NaN for every other TI."
    )
    print(
        "Interpretation note: with a hard exclusion mask, the masked run needs a "
        "large enough retained set. If the preferred TI is not inside the retained "
        "top-k candidates, AS falls back after ranking and the result can resemble "
        "the unmasked run."
    )
    print_index_debug("AS without -mi", unmasked, unknown, debug=args.debug)
    print_index_debug("AS with -mi", masked, unknown, debug=args.debug)
    summarize_run("AS without -mi", dataset, unmasked)
    summarize_run("AS with -mi", dataset, masked)
    if not args.skip_edge_cases:
        run_mi_edge_cases(
            g2s,
            server_address=args.server_address,
            jobs=args.jobs,
            seed=args.seed,
            num_ti=args.num_ti,
        )
    show_figure(dataset, unmasked, masked)


if __name__ == "__main__":
    main()
