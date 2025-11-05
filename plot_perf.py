"""
plot_results.py

Read one or more Reporter CSV files (from Reporter.Reporter in the TSP EA)
and plot the recorded mean and best objective values over time or over iterations.

Usage examples:
    # Plot a single CSV (shows interactive window)
    python plot_results.py run1.csv

    # Plot multiple CSVs, save to file
    python plot_results.py run1.csv run2.csv -o results.png

    # Compute the average curve across the provided runs (interpolates onto a common time grid)
    python plot_results.py run1.csv run2.csv run3.csv --avg -o avg_results.png

    # Plot based on iteration instead of elapsed time
    python plot_results.py run1.csv --xmode iter

Notes about CSV format produced by Reporter:
- Reporter writes two comment header lines that start with '#'. The data lines are:
    Iteration, Elapsed time, Mean value, Best value, <tour entries...>
  We only need columns 0..3.

Dependencies:
    numpy, pandas, matplotlib

"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys


def read_report_csv(path: str):
    """
    Read Reporter CSV produced by Reporter.Reporter.
    Returns a dict with keys: 'iteration', 'time', 'mean', 'best'.
    Raises ValueError if file has no data rows.
    """
    # read with pandas, skip comment lines starting with '#'
    df = pd.read_csv(path, comment="#", header=None)
    if df.shape[0] == 0:
        raise ValueError(f"No data rows found in {path}")
    if df.shape[1] < 4:
        raise ValueError(f"Expected at least 4 columns in {path}, got {df.shape[1]}")
    # columns: 0 iteration, 1 elapsed time, 2 mean, 3 best
    iteration = df.iloc[:, 0].to_numpy(dtype=float)
    time = df.iloc[:, 1].to_numpy(dtype=float)
    mean = df.iloc[:, 2].to_numpy(dtype=float)
    best = df.iloc[:, 3].to_numpy(dtype=float)
    # Ensure strictly increasing x for interpolation (monotonic)
    # If not strictly increasing, make it non-decreasing by tiny eps jitter
    # (np.interp accepts non-decreasing x)
    return {"iteration": iteration, "time": time, "mean": mean, "best": best}


def make_time_grid(all_times, mode="min", npoints=1000):
    """
    Create a common time grid for interpolation.
    all_times: list of 1D arrays with run times
    mode: 'min' -> use min(max_time) across runs (so all runs cover the grid)
          'max' -> use max(max_time) across runs (runs shorter will be padded by last value)
    """
    maxes = [t.max() for t in all_times if len(t) > 0]
    if len(maxes) == 0:
        return np.linspace(0.0, 1.0, npoints)
    if mode == "min":
        end = min(maxes)
    else:
        end = max(maxes)
    return np.linspace(0.0, end, npoints)


def interp_to_grid(x, y, grid, extrapolate_last=True):
    """
    Interpolate y(x) onto grid.
    If grid contains values past x.max(), np.interp will use y[-1] for those points (this corresponds
    to 'extrapolate_last' behavior). This is usually fine for averaging runs of different lengths.
    """
    # Make sure x is non-decreasing for np.interp
    # If there are duplicate x values, np.interp tolerates non-decreasing x.
    return np.interp(grid, x, y)


def plot_runs(
    paths,
    xmode="time",
    avg=False,
    overlay=False,
    out=None,
    grid_points=1000,
    resample_mode="min",
):
    """
    Main plotting function.
    - paths: list of CSV paths
    - xmode: 'time' or 'iter'
    - avg: boolean: compute average curve across runs (interpolates)
    - overlay: boolean: also draw individual runs (faded)
    - out: path to save figure (if None, show interactively)
    """
    runs = []
    labels = []
    for p in paths:
        try:
            runs.append(read_report_csv(p))
            labels.append(Path(p).stem)
        except Exception as e:
            print(f"Error reading {p}: {e}", file=sys.stderr)
            raise

    # Choose x arrays
    xs = [r["time"] if xmode == "time" else r["iteration"] for r in runs]
    means = [r["mean"] for r in runs]
    bests = [r["best"] for r in runs]

    # If averaging requested: build a common grid
    if avg:
        grid = make_time_grid(xs, mode=resample_mode, npoints=grid_points)
        mean_interp = np.vstack(
            [interp_to_grid(xs[i], means[i], grid) for i in range(len(runs))]
        )
        best_interp = np.vstack(
            [interp_to_grid(xs[i], bests[i], grid) for i in range(len(runs))]
        )

        mean_avg = mean_interp.mean(axis=0)
        mean_std = mean_interp.std(axis=0)
        best_avg = best_interp.mean(axis=0)
        best_std = best_interp.std(axis=0)

        fig, ax = plt.subplots(figsize=(9, 6))
        # plot averaged mean and best
        ax.plot(grid, mean_avg, label="Mean (avg of runs)", color="tab:blue", lw=2)
        ax.fill_between(
            grid, mean_avg - mean_std, mean_avg + mean_std, color="tab:blue", alpha=0.2
        )

        ax.plot(grid, best_avg, label="Best (avg of runs)", color="tab:orange", lw=2)
        ax.fill_between(
            grid,
            best_avg - best_std,
            best_avg + best_std,
            color="tab:orange",
            alpha=0.18,
        )

        if overlay:
            # overlay individual runs faintly
            for i in range(len(runs)):
                ax.plot(grid, mean_interp[i], color="tab:blue", alpha=0.12, lw=1)
                ax.plot(grid, best_interp[i], color="tab:orange", alpha=0.12, lw=1)

        ax.set_xlabel("Elapsed time (s)" if xmode == "time" else "Iteration")
        ax.set_ylabel("Objective (lower is better)")
        ax.set_title("Averaged mean and best objective across runs")
        ax.grid(True)
        ax.legend()
    else:
        # Plot each run individually
        fig, ax = plt.subplots(figsize=(9, 6))
        for i, r in enumerate(runs):
            x = xs[i]
            ax.plot(x, means[i], label=f"{labels[i]} mean", lw=1.2, alpha=0.9)
            ax.plot(
                x,
                bests[i],
                label=f"{labels[i]} best",
                lw=1.2,
                linestyle="--",
                alpha=0.9,
            )
        ax.set_xlabel("Elapsed time (s)" if xmode == "time" else "Iteration")
        ax.set_ylabel("Objective (lower is better)")
        ax.set_title("Mean and best objective per run")
        ax.grid(True)
        ax.legend(fontsize="small", ncol=2)

    plt.tight_layout()
    if out:
        plt.savefig(out, dpi=200)
        print(f"Saved plot to {out}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot mean and best objective values from Reporter CSV files."
    )
    parser.add_argument("csv", nargs="+", help="Paths to Reporter CSV files")
    parser.add_argument(
        "-o",
        "--out",
        help="Save plot to file (e.g. results.png). If omitted, show interactively.",
    )
    parser.add_argument(
        "--xmode",
        choices=["time", "iter"],
        default="time",
        help="Use elapsed time or iteration for x-axis (default: time).",
    )
    parser.add_argument(
        "--avg",
        action="store_true",
        help="Interpolate runs to a common grid and plot the average (with std shading).",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="When --avg is used, overlay individual runs faintly.",
    )
    parser.add_argument(
        "--grid-points",
        type=int,
        default=1000,
        help="Number of points in the common interpolation grid (default: 1000).",
    )
    parser.add_argument(
        "--resample-mode",
        choices=["min", "max"],
        default="min",
        help="When averaging, use 'min' to average only up to the shortest run, or 'max' to extend to the longest run by holding last value (default: min).",
    )
    args = parser.parse_args()

    plot_runs(
        args.csv,
        xmode=args.xmode,
        avg=args.avg,
        overlay=args.overlay,
        out=args.out,
        grid_points=args.grid_points,
        resample_mode=args.resample_mode,
    )


if __name__ == "__main__":
    main()
