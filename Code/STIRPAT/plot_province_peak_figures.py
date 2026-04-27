from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent


SCENARIO_LABELS = {
    "low_carbon": "Low-carbon",
    "baseline": "Baseline",
    "green_growth": "Green growth",
    "extensive": "Extensive",
}

SCENARIO_ORDER = ["low_carbon", "baseline", "green_growth", "extensive"]


def parse_args() -> argparse.Namespace:
    base = SCRIPT_DIR / "output" / "scenario_forecast"
    parser = argparse.ArgumentParser(description="Plot province-level peak year figures for scenario forecasts.")
    parser.add_argument(
        "--province-peak-csv",
        type=Path,
        default=base / "organized" / "02_province_peak_summary.csv",
        help="Province peak summary CSV path.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=base / "province_peak_figures",
        help="Directory for exported figures and summary tables.",
    )
    return parser.parse_args()


def _read_peak_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Province peak summary not found: {path}")
    df = pd.read_csv(path)
    required = {"scenario", "province", "peak_year", "peak_co2", "peaked_within_horizon"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df["peak_year"] = pd.to_numeric(df["peak_year"], errors="coerce")
    df["peak_co2"] = pd.to_numeric(df["peak_co2"], errors="coerce")
    df["peaked_within_horizon"] = df["peaked_within_horizon"].astype(str).str.lower().isin(["true", "1", "yes"])
    return df.dropna(subset=["peak_year", "peak_co2"]).copy()


def plot_baseline_peak_ranking(df: pd.DataFrame, out_file: Path) -> None:
    baseline = df[df["scenario"] == "baseline"].copy()
    if baseline.empty:
        raise ValueError("No baseline rows found.")
    baseline = baseline.sort_values(["peak_year", "peak_co2", "province"], ascending=[True, True, True])

    colors = np.where(baseline["peaked_within_horizon"], "#2ca02c", "#d62728")
    fig_h = max(7.2, 0.28 * len(baseline) + 1.5)
    fig, ax = plt.subplots(figsize=(10.8, fig_h))
    ax.barh(baseline["province"], baseline["peak_year"], color=colors, alpha=0.82)
    ax.axvline(2030, color="#333333", linestyle="--", linewidth=1.2, label="2030 target")
    ax.set_xlim(2023.5, 2035.8)
    ax.set_xlabel("Peak year")
    ax.set_ylabel("Province")
    ax.set_title("Baseline Scenario Province-level CO2 Peak Years")
    ax.grid(axis="x", alpha=0.22, linestyle="--")
    ax.invert_yaxis()

    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor="#2ca02c", alpha=0.82, label="Peaked within horizon"),
        Patch(facecolor="#d62728", alpha=0.82, label="Not peaked by 2035"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_file, dpi=300)
    plt.close(fig)


def plot_peak_year_heatmap(df: pd.DataFrame, out_file: Path) -> None:
    pivot = df.pivot_table(index="province", columns="scenario", values="peak_year", aggfunc="first")
    existing_order = [s for s in SCENARIO_ORDER if s in pivot.columns]
    pivot = pivot.reindex(columns=existing_order)
    pivot = pivot.sort_values(["baseline", "low_carbon"], kind="stable")

    data = pivot.to_numpy(dtype=float)
    fig_h = max(7.2, 0.28 * len(pivot) + 1.8)
    fig, ax = plt.subplots(figsize=(8.8, fig_h))
    im = ax.imshow(data, aspect="auto", cmap="YlGnBu", vmin=2024, vmax=2035)

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([SCENARIO_LABELS.get(c, c) for c in pivot.columns], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Province-level CO2 Peak Years under Different Scenarios")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            ax.text(j, i, f"{int(val)}", ha="center", va="center", fontsize=8, color="#111111")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Peak year")
    fig.tight_layout()
    fig.savefig(out_file, dpi=300)
    plt.close(fig)


def write_summary_tables(df: pd.DataFrame, out_dir: Path) -> None:
    summary = (
        df.assign(scenario_label=df["scenario"].map(SCENARIO_LABELS).fillna(df["scenario"]))
        .groupby(["scenario", "peak_year"], as_index=False)
        .agg(province_count=("province", "count"))
        .sort_values(["scenario", "peak_year"], kind="stable")
    )
    summary.to_csv(out_dir / "province_peak_year_counts.csv", index=False, encoding="utf-8-sig")

    late = (
        df.sort_values(["scenario", "peaked_within_horizon", "peak_year", "peak_co2"], ascending=[True, True, False, False])
        .groupby("scenario", as_index=False)
        .head(8)
        [["scenario", "province", "peak_year", "peak_co2", "peaked_within_horizon"]]
    )
    late.to_csv(out_dir / "province_late_peak_examples.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = _read_peak_table(args.province_peak_csv)

    plot_baseline_peak_ranking(df, args.out_dir / "baseline_province_peak_year_ranking.png")
    plot_peak_year_heatmap(df, args.out_dir / "province_peak_year_heatmap_by_scenario.png")
    write_summary_tables(df, args.out_dir)

    print(f"Saved province peak figures to: {args.out_dir}")


if __name__ == "__main__":
    main()
