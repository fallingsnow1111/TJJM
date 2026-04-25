from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize policy scenario forecast outputs.")
    base = Path(__file__).resolve().parent / "output" / "scenario_forecast"
    parser.add_argument(
        "--national-csv",
        type=Path,
        default=base / "scenario_forecast_national.csv",
        help="National yearly forecast CSV path",
    )
    parser.add_argument(
        "--peak-csv",
        type=Path,
        default=base / "scenario_peak_summary.csv",
        help="Peak summary CSV path",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=base,
        help="Directory for output figures",
    )
    return parser


def _pretty_name(s: str) -> str:
    mapping = {
        "baseline": "Baseline",
        "low_carbon": "Low-carbon",
        "extensive": "Extensive",
        "green_growth": "Green Growth",
        "deep_decarb": "Deep Decarb",
    }
    return mapping.get(s, s)


def plot_trend_with_peak(national_df: pd.DataFrame, peak_df: pd.DataFrame, out_file: Path) -> None:
    style = {
        "baseline": {"color": "#1f77b4", "ls": "-", "marker": "o"},
        "low_carbon": {"color": "#2ca02c", "ls": "--", "marker": "s"},
        "extensive": {"color": "#d62728", "ls": "-.", "marker": "^"},
        "green_growth": {"color": "#9467bd", "ls": "-", "marker": "D"},
        "deep_decarb": {"color": "#8c564b", "ls": "--", "marker": "x"},
    }

    fig, ax = plt.subplots(figsize=(12, 6.5))

    for scenario, grp in national_df.groupby("scenario", sort=False):
        g = grp.sort_values("year", kind="stable")
        st = style.get(scenario, {"color": "#555555", "ls": "-", "marker": "o"})
        ax.plot(
            g["year"],
            g["co2_pred"],
            color=st["color"],
            linestyle=st["ls"],
            linewidth=2.2,
            marker=st["marker"],
            markersize=4.5,
            label=_pretty_name(scenario),
        )

        one_peak = peak_df.loc[peak_df["scenario"] == scenario]
        if not one_peak.empty:
            py = int(one_peak.iloc[0]["peak_year"])
            pv = float(one_peak.iloc[0]["peak_co2"])
            ax.scatter([py], [pv], color=st["color"], s=75, zorder=4)
            note = "未达峰" if py == 2035 else f"{py}: {pv:.1f}"
            ax.annotate(
                f"{_pretty_name(scenario)} peak\n{note}",
                xy=(py, pv),
                xytext=(8, 10),
                textcoords="offset points",
                fontsize=9,
                color=st["color"],
            )

    ax.set_title("National CO2 Forecast by Scenario (2024-2035)", fontsize=14)
    ax.set_xlabel("Year")
    ax.set_ylabel("Predicted CO2")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_file, dpi=300)
    plt.close(fig)


def plot_key_years(national_df: pd.DataFrame, out_file: Path) -> None:
    years = [2025, 2030, 2035]
    df = national_df[national_df["year"].isin(years)].copy()
    if df.empty:
        return

    # 定义所有情景及其展示顺序
    scenario_order = ["deep_decarb", "low_carbon", "green_growth", "baseline", "extensive"]
    colors = {
        "deep_decarb": "#8c564b",
        "low_carbon": "#2ca02c",
        "green_growth": "#9467bd",
        "baseline": "#1f77b4",
        "extensive": "#d62728",
    }

    year_labels = [str(y) for y in years]
    x = np.arange(len(years))
    n_scenarios = len(scenario_order)
    width = 0.8 / n_scenarios   # 自动计算柱宽，避免重叠

    fig, ax = plt.subplots(figsize=(12, 6.5))

    for i, scenario in enumerate(scenario_order):
        vals = []
        for y in years:
            row = df[(df["scenario"] == scenario) & (df["year"] == y)]
            vals.append(float(row.iloc[0]["co2_pred"]) if not row.empty else np.nan)
        bars = ax.bar(
            x + (i - (n_scenarios - 1) / 2) * width,
            vals,
            width=width,
            color=colors[scenario],
            label=_pretty_name(scenario),
        )
        for b in bars:
            h = b.get_height()
            if np.isfinite(h):
                ax.text(b.get_x() + b.get_width() / 2, h, f"{h:.0f}", ha="center", va="bottom", fontsize=7)

    ax.set_title("Scenario Comparison at Key Years", fontsize=14)
    ax.set_xlabel("Year")
    ax.set_ylabel("Predicted CO2")
    ax.set_xticks(x)
    ax.set_xticklabels(year_labels)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_file, dpi=300)
    plt.close(fig)

def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.national_csv.exists():
        raise FileNotFoundError(f"National CSV not found: {args.national_csv}")
    if not args.peak_csv.exists():
        raise FileNotFoundError(f"Peak summary CSV not found: {args.peak_csv}")

    national_df = pd.read_csv(args.national_csv)
    peak_df = pd.read_csv(args.peak_csv)

    required_national = {"scenario", "year", "co2_pred"}
    required_peak = {"scenario", "peak_year", "peak_co2"}
    if not required_national.issubset(set(national_df.columns)):
        raise ValueError(f"National CSV missing columns: {sorted(required_national - set(national_df.columns))}")
    if not required_peak.issubset(set(peak_df.columns)):
        raise ValueError(f"Peak CSV missing columns: {sorted(required_peak - set(peak_df.columns))}")

    trend_png = args.out_dir / "scenario_forecast_national_trend.png"
    key_png = args.out_dir / "scenario_forecast_key_years.png"

    plot_trend_with_peak(national_df=national_df, peak_df=peak_df, out_file=trend_png)
    plot_key_years(national_df=national_df, out_file=key_png)

    print(f"Saved figure: {trend_png}")
    print(f"Saved figure: {key_png}")


if __name__ == "__main__":
    main()
