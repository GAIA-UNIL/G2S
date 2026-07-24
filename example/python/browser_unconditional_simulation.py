#!/usr/bin/env python3
"""Run a larger unconditional QS simulation inside the browser.

Before running this file:

1. Start the browser file server from the repository root:

   python3 browser/serve.py

2. Open http://localhost:8000/ in Chrome/Chromium or Firefox and leave it open.
3. Rebuild/reinstall the G2S Python wheel from this checkout if necessary.
4. Run this script with a Python environment containing G2S, NumPy, Pillow,
   and Matplotlib.

The Python process only hosts the temporary localhost transport. QS and FFTW
execute in the browser's Web Worker.
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
from typing import Optional
from urllib.request import urlopen

import numpy as np
from PIL import Image


TRAINING_IMAGE_URL = (
    "https://raw.githubusercontent.com/GAIA-UNIL/"
    "TrainingImagesTIFF/master/stone.tiff"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the continuous stone QS example in the browser."
    )
    parser.add_argument(
        "--size",
        type=int,
        default=200,
        help="Width and height of the simulated grid (default: 200).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="QS random seed (default: 12345).",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=30_000,
        help="Browser connection/heartbeat timeout in milliseconds.",
    )
    parser.add_argument(
        "--threads",
        type=float,
        default=4,
        help=(
            "Requested QS threads using normal -j semantics (default: 4). "
            "The browser page may impose a lower maximum."
        ),
    )
    parser.add_argument(
        "--browser-origin",
        help=(
            "Optional exact origin restriction for the open browser page. "
            "Omit it to accept the active hosted preview."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the generated comparison figure.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the interactive Matplotlib window.",
    )
    arguments = parser.parse_args()
    if arguments.size <= 0:
        parser.error("--size must be positive")
    if arguments.timeout_ms <= 0:
        parser.error("--timeout-ms must be positive")
    if arguments.browser_origin and not arguments.browser_origin.startswith(
        ("http://", "https://")
    ):
        parser.error("--browser-origin must be an http:// or https:// origin")
    return arguments


def download_training_image() -> np.ndarray:
    """Download the public stone training image as a 2-D float32 array."""
    print(f"Downloading training image:\n  {TRAINING_IMAGE_URL}")
    with urlopen(TRAINING_IMAGE_URL, timeout=30) as response:
        encoded_image = response.read()

    with Image.open(BytesIO(encoded_image)) as image:
        training_image = np.asarray(image.convert("F"), dtype=np.float32)

    if training_image.ndim != 2 or training_image.size == 0:
        raise RuntimeError(
            f"Expected a non-empty 2-D training image, got {training_image.shape}"
        )
    print(
        "Training image:",
        f"shape={training_image.shape},",
        f"range=[{training_image.min():.3f}, {training_image.max():.3f}]",
    )
    return np.ascontiguousarray(training_image)


def run_browser_qs(
    training_image: np.ndarray,
    size: int,
    seed: int,
    timeout_ms: int,
    threads: float,
    browser_origin: Optional[str],
) -> tuple[np.ndarray, np.ndarray, float, dict[str, str]]:
    """Send one job to the preloaded browser page and return its results."""
    try:
        import g2s as g2s_package
        from g2s import g2s
    except ImportError as error:
        raise RuntimeError(
            "The G2S Python interface is not installed in this environment."
        ) from error
    package_location = getattr(g2s_package, "__file__", "unknown location")

    destination = np.full((size, size), np.nan, dtype=np.float32)

    print(
        f"Starting {size}x{size} browser QS simulation "
        f"(requested threads: {threads:g})..."
    )
    browser_options: list[object] = []
    if browser_origin:
        browser_options.extend(("-browserOrigin", browser_origin))
    try:
        simulation, index_map, duration, metadata, *_ = g2s(
            "-a",
            "qs",
            "-sa",
            "browser",
            *browser_options,
            "-ti",
            training_image,
            "-di",
            destination,
            "-dt",
            [0],  # One continuous variable.
            "-k",
            1.2,
            "-n",
            50,
            "-s",
            seed,
            "-TO",
            timeout_ms,
            "-j",
            threads,
            "-returnMeta",
        )
    except Exception as error:
        page_description = browser_origin or "the open browser preview"
        raise RuntimeError(
            f"Browser QS failed. Confirm that {page_description} is open, "
            "the page says it is waiting for a local command, and port 8129 "
            "is available. If the original error refers to the ordinary G2S "
            "server, rebuild and reinstall the Python wheel from this checkout. "
            f"Imported G2S: {package_location}. Original error: {error}"
        ) from error

    simulation = np.asarray(simulation)
    index_map = np.asarray(index_map)
    duration = float(np.asarray(duration).squeeze())
    metadata = {str(key): str(value) for key, value in dict(metadata).items()}
    print(
        "Simulation complete:",
        f"shape={simulation.shape},",
        f"duration={duration:.3f} s,",
        f"effective threads={metadata.get('effective_threads', 'unknown')}",
    )
    return simulation, index_map, duration, metadata


def display_results(
    training_image: np.ndarray,
    simulation: np.ndarray,
    index_map: np.ndarray,
    duration: float,
    output: Optional[Path],
    show: bool,
) -> None:
    """Display the training image, simulation, and returned source-index map."""
    if not show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    value_min = float(np.nanmin(training_image))
    value_max = float(np.nanmax(training_image))
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    figure.suptitle(
        f"Browser QS unconditional simulation ({duration:.3f} s)",
        fontsize="x-large",
    )

    training_plot = axes[0].imshow(
        training_image,
        cmap="gray",
        vmin=value_min,
        vmax=value_max,
    )
    axes[0].set_title("Training image: stone")
    axes[0].axis("off")

    axes[1].imshow(
        simulation,
        cmap="gray",
        vmin=value_min,
        vmax=value_max,
    )
    axes[1].set_title("Browser/Wasm simulation")
    axes[1].axis("off")

    axes[2].imshow(index_map, cmap="viridis")
    axes[2].set_title("Returned index map")
    axes[2].axis("off")
    figure.colorbar(training_plot, ax=axes[:2], shrink=0.78, label="Value")

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=160)
        print(f"Saved figure to {output.resolve()}")
    if show:
        plt.show()
    else:
        plt.close(figure)


def main() -> None:
    arguments = parse_arguments()
    training_image = download_training_image()
    simulation, index_map, duration, metadata = run_browser_qs(
        training_image,
        arguments.size,
        arguments.seed,
        arguments.timeout_ms,
        arguments.threads,
        arguments.browser_origin,
    )
    if "thread_warning" in metadata:
        print(f"Thread warning: {metadata['thread_warning']}")
    display_results(
        training_image,
        simulation,
        index_map,
        duration,
        arguments.output,
        not arguments.no_show,
    )


if __name__ == "__main__":
    main()
