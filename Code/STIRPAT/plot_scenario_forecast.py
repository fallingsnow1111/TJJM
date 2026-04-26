from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXCLUDED_PROVINCES = {"Tibet"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize policy scenario forecast outputs.")
    base = Path(__file__).resolve().parent / "output" / "scenario_forecast"
    parser.add_argument(
        "--detail-csv",
        type=Path,
        default=base / "scenario_forecast_detail.csv",
        help="Province-year detailed forecast CSV path",
    )
    parser.add_argument(
        "--history-csv",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "Preprocess" / "output" / "panel_master.csv",
        help="Historical province-year panel CSV with CO2 (e.g., 1990-2023)",
    )
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
    parser.add_argument(
        "--provinces-per-figure",
        type=int,
        default=9,
        help="How many provinces to show per grouped province figure",
    )
    return parser


def _pretty_name(s: str) -> str:
    mapping = {
        "baseline": "Baseline",
        "low_carbon": "Low-carbon",
        "extensive": "Extensive",
        "green_growth": "Green Growth",
    }
    return mapping.get(s, s)


def _scenario_style(scenario: str) -> dict:
    style = {
        "baseline": {"color": "#1f77b4", "ls": "-", "marker": "o"},
        "low_carbon": {"color": "#2ca02c", "ls": "--", "marker": "s"},
        "extensive": {"color": "#d62728", "ls": "-.", "marker": "^"},
        "green_growth": {"color": "#9467bd", "ls": "-", "marker": "D"},
    }
    return style.get(scenario, {"color": "#555555", "ls": "-", "marker": "o"})


def _exclude_provinces(df: pd.DataFrame, province_col: str = "province") -> pd.DataFrame:
    if province_col not in df.columns:
        return df
    return df.loc[~df[province_col].astype(str).isin(EXCLUDED_PROVINCES)].copy()


def _infer_forecast_co2_scale(
    history_national_df: Optional[pd.DataFrame],
    national_df: pd.DataFrame,
) -> float:
    if history_national_df is None or history_national_df.empty or national_df.empty:
        return 1.0

    h = history_national_df.sort_values("year", kind="stable")
    hist_ref = pd.to_numeric(h["co2_actual"], errors="coerce").dropna()
    if hist_ref.empty:
        return 1.0

    baseline = national_df.copy()
    if "scenario" in baseline.columns and baseline["scenario"].notna().any():
        preferred = baseline[baseline["scenario"].astype(str) == "baseline"].copy()
        if not preferred.empty:
            baseline = preferred
    forecast_ref = pd.to_numeric(
        baseline.sort_values("year", kind="stable")["co2_pred"],
        errors="coerce",
    ).dropna()
    if forecast_ref.empty:
        return 1.0

    hist_last = float(hist_ref.iloc[-1])
    forecast_first = float(forecast_ref.iloc[0])
    if not np.isfinite(hist_last) or not np.isfinite(forecast_first):
        return 1.0
    if abs(hist_last) < 1e-8 or abs(forecast_first) < 1e-8:
        return 1.0

    ratio = abs(forecast_first) / abs(hist_last)
    if 0.05 <= ratio <= 20.0:
        return 1.0

    exponent = int(round(np.log10(ratio)))
    scale = 10.0 ** (-exponent)
    scaled_ratio = ratio * scale
    return scale if 0.2 <= scaled_ratio <= 5.0 else 1.0


def _add_transition_segment(
    ax: plt.Axes,
    history_year: int,
    history_value: float,
    forecast_year: int,
    forecast_value: float,
    color: str,
    linestyle: str,
) -> None:
    ax.plot(
        [history_year, forecast_year],
        [history_value, forecast_value],
        color=color,
        linestyle=linestyle,
        linewidth=1.4,
        alpha=0.7,
        zorder=2,
    )


def _province_alignment_factor(
    history_df: Optional[pd.DataFrame],
    province: str,
    forecast_df: pd.DataFrame,
) -> float:
    if history_df is None or history_df.empty or forecast_df.empty:
        return 1.0

    hp = history_df[history_df["province"].astype(str) == str(province)].sort_values("year", kind="stable")
    if hp.empty:
        return 1.0

    hist_last = pd.to_numeric(hp["CO2"], errors="coerce").dropna()
    forecast_first = pd.to_numeric(
        forecast_df.sort_values("year", kind="stable")["co2_pred"],
        errors="coerce",
    ).dropna()
    if hist_last.empty or forecast_first.empty:
        return 1.0

    hist_value = float(hist_last.iloc[-1])
    forecast_value = float(forecast_first.iloc[0])
    if not np.isfinite(hist_value) or not np.isfinite(forecast_value):
        return 1.0
    if abs(forecast_value) < 1e-8:
        return 1.0
    return hist_value / forecast_value


def _national_alignment_factor(
    history_national_df: Optional[pd.DataFrame],
    forecast_df: pd.DataFrame,
) -> float:
    if history_national_df is None or history_national_df.empty or forecast_df.empty:
        return 1.0

    hist_ref = pd.to_numeric(
        history_national_df.sort_values("year", kind="stable")["co2_actual"],
        errors="coerce",
    ).dropna()
    forecast_ref = pd.to_numeric(
        forecast_df.sort_values("year", kind="stable")["co2_pred"],
        errors="coerce",
    ).dropna()
    if hist_ref.empty or forecast_ref.empty:
        return 1.0

    hist_value = float(hist_ref.iloc[-1])
    forecast_value = float(forecast_ref.iloc[0])
    if not np.isfinite(hist_value) or not np.isfinite(forecast_value):
        return 1.0
    if abs(forecast_value) < 1e-8:
        return 1.0
    return hist_value / forecast_value


def plot_trend_with_peak(
    national_df: pd.DataFrame,
    peak_df: pd.DataFrame,
    history_national_df: Optional[pd.DataFrame],
    out_file: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.5))

    if history_national_df is not None and not history_national_df.empty:
        h = history_national_df.sort_values("year", kind="stable")
        ax.plot(
            h["year"],
            h["co2_actual"],
            color="#111111",
            linestyle="-",
            linewidth=2.4,
            marker="o",
            markersize=3.0,
            label="Historical (1990-2023)",
        )

    for scenario, grp in national_df.groupby("scenario", sort=False):
        g = grp.sort_values("year", kind="stable")
        st = _scenario_style(scenario)
        plot_g = g.copy()
        align_factor = _national_alignment_factor(
            history_national_df=history_national_df,
            forecast_df=g,
        )
        plot_g["co2_pred"] = pd.to_numeric(plot_g["co2_pred"], errors="coerce") * align_factor

        if history_national_df is not None and not history_national_df.empty and not plot_g.empty:
            hist_last = history_national_df.sort_values("year", kind="stable").iloc[-1]
            _add_transition_segment(
                ax=ax,
                history_year=int(hist_last["year"]),
                history_value=float(hist_last["co2_actual"]),
                forecast_year=int(plot_g.iloc[0]["year"]),
                forecast_value=float(plot_g.iloc[0]["co2_pred"]),
                color=st["color"],
                linestyle=st["ls"],
            )

        ax.plot(
            plot_g["year"],
            plot_g["co2_pred"],
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
            pv = float(one_peak.iloc[0]["peak_co2"]) * align_factor
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

    ax.set_title("National CO2 History and Scenario Forecast (1990-2035)", fontsize=14)
    ax.set_xlabel("Year")
    ax.set_ylabel("CO2 Emissions")
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

    # Use available scenarios and keep a preferred display order.
    preferred = ["low_carbon", "green_growth", "baseline", "extensive"]
    existing = df["scenario"].astype(str).unique().tolist()
    scenario_order = [s for s in preferred if s in existing] + [s for s in existing if s not in preferred]
    colors = {
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
            color=colors.get(scenario, "#555555"),
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


def plot_province_trends(detail_df: pd.DataFrame, out_file: Path) -> None:
    scenarios = detail_df["scenario"].astype(str).unique().tolist()
    if not scenarios:
        return

    n = len(scenarios)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4.8 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)

    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]
        s = detail_df[detail_df["scenario"] == scenario].copy()
        st = _scenario_style(scenario)

        # Province trajectories in light color.
        for _, g in s.groupby("province", sort=False):
            gg = g.sort_values("year", kind="stable")
            ax.plot(gg["year"], gg["co2_pred"], color=st["color"], alpha=0.18, linewidth=0.8)

        # Overlay national aggregation for readability.
        nat = s.groupby("year", as_index=False)["co2_pred"].sum().sort_values("year", kind="stable")
        ax.plot(nat["year"], nat["co2_pred"], color=st["color"], linewidth=2.3, label=f"{_pretty_name(scenario)} national")
        ax.set_title(f"{_pretty_name(scenario)}: Province CO2 Paths")
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend(loc="best", fontsize=8)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.supxlabel("Year")
    fig.supylabel("Predicted CO2")
    fig.tight_layout()
    fig.savefig(out_file, dpi=300)
    plt.close(fig)


def plot_province_heatmap(detail_df: pd.DataFrame, out_file: Path, year: int = 2030) -> None:
    df = detail_df[detail_df["year"] == year].copy()
    if df.empty:
        return

    pivot = df.pivot_table(index="province", columns="scenario", values="co2_pred", aggfunc="sum")
    if pivot.empty:
        return

    # Sort provinces by total across scenarios for a cleaner heatmap.
    pivot = pivot.assign(_total=pivot.sum(axis=1)).sort_values("_total", ascending=False).drop(columns=["_total"])

    fig_h = max(6.0, 0.3 * len(pivot.index) + 1.5)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="YlOrRd")

    ax.set_title(f"Province CO2 Heatmap ({year})")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Province")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([_pretty_name(str(c)) for c in pivot.columns], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Predicted CO2")

    fig.tight_layout()
    fig.savefig(out_file, dpi=300)
    plt.close(fig)


def plot_province_panels_grouped(
    detail_df: pd.DataFrame,
    history_df: Optional[pd.DataFrame],
    out_dir: Path,
    provinces_per_figure: int = 9,
    ncols: int = 3,
) -> List[Path]:
    provinces = sorted(detail_df["province"].astype(str).unique().tolist())
    if not provinces:
        return []

    scenarios = detail_df["scenario"].astype(str).unique().tolist()
    if not scenarios:
        return []

    chunk = max(int(provinces_per_figure), 1)
    ncols = max(int(ncols), 1)
    files: List[Path] = []

    for i in range(0, len(provinces), chunk):
        group = provinces[i : i + chunk]
        n = len(group)
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.6 * ncols, 3.4 * nrows), sharex=True)
        axes = np.array(axes).reshape(-1)

        for k, province in enumerate(group):
            ax = axes[k]
            p = detail_df[detail_df["province"] == province].copy()

            if history_df is not None and not history_df.empty:
                hp = history_df[history_df["province"] == province].sort_values("year", kind="stable")
                if not hp.empty:
                    ax.plot(
                        hp["year"],
                        hp["CO2"],
                        color="#111111",
                        linestyle="-",
                        linewidth=1.6,
                        marker="o",
                        markersize=2.0,
                        label="Historical",
                    )

            for scenario in scenarios:
                s = p[p["scenario"] == scenario].sort_values("year", kind="stable")
                if s.empty:
                    continue
                st = _scenario_style(scenario)
                plot_s = s.copy()
                align_factor = _province_alignment_factor(
                    history_df=history_df,
                    province=province,
                    forecast_df=s,
                )
                plot_s["co2_pred"] = pd.to_numeric(plot_s["co2_pred"], errors="coerce") * align_factor
                if history_df is not None and not history_df.empty:
                    hp = history_df[history_df["province"] == province].sort_values("year", kind="stable")
                    if not hp.empty:
                        hist_last = hp.iloc[-1]
                        _add_transition_segment(
                            ax=ax,
                            history_year=int(hist_last["year"]),
                            history_value=float(hist_last["CO2"]),
                            forecast_year=int(plot_s.iloc[0]["year"]),
                            forecast_value=float(plot_s.iloc[0]["co2_pred"]),
                            color=st["color"],
                            linestyle=st["ls"],
                        )
                ax.plot(
                    plot_s["year"],
                    plot_s["co2_pred"],
                    color=st["color"],
                    linestyle=st["ls"],
                    linewidth=1.4,
                    marker=st["marker"],
                    markersize=2.8,
                    label=_pretty_name(scenario),
                )

            ax.set_title(str(province), fontsize=9)
            ax.grid(alpha=0.22, linestyle="--")

        for z in range(n, len(axes)):
            axes[z].axis("off")

        # Put one legend on first visible axis to reduce clutter.
        if n > 0:
            axes[0].legend(loc="best", fontsize=7, frameon=False)

        fig.suptitle(f"Province CO2 Trends (Group {i // chunk + 1})", fontsize=12)
        fig.supxlabel("Year")
        fig.supylabel("Predicted CO2")
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        out_file = out_dir / f"scenario_forecast_province_panels_group_{i // chunk + 1:02d}.png"
        fig.savefig(out_file, dpi=300)
        plt.close(fig)
        files.append(out_file)

    return files

def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.history_csv.exists():
        raise FileNotFoundError(f"History CSV not found: {args.history_csv}")
    if not args.detail_csv.exists():
        raise FileNotFoundError(f"Detail CSV not found: {args.detail_csv}")
    if not args.national_csv.exists():
        raise FileNotFoundError(f"National CSV not found: {args.national_csv}")
    if not args.peak_csv.exists():
        raise FileNotFoundError(f"Peak summary CSV not found: {args.peak_csv}")

    history_df = _exclude_provinces(pd.read_csv(args.history_csv))
    detail_df = _exclude_provinces(pd.read_csv(args.detail_csv))
    national_df = pd.read_csv(args.national_csv)
    peak_df = pd.read_csv(args.peak_csv)

    required_history = {"province", "year", "CO2"}
    required_detail = {"scenario", "province", "year", "co2_pred"}
    required_national = {"scenario", "year", "co2_pred"}
    required_peak = {"scenario", "peak_year", "peak_co2"}
    if not required_history.issubset(set(history_df.columns)):
        raise ValueError(f"History CSV missing columns: {sorted(required_history - set(history_df.columns))}")
    if not required_detail.issubset(set(detail_df.columns)):
        raise ValueError(f"Detail CSV missing columns: {sorted(required_detail - set(detail_df.columns))}")
    if not required_national.issubset(set(national_df.columns)):
        raise ValueError(f"National CSV missing columns: {sorted(required_national - set(national_df.columns))}")
    if not required_peak.issubset(set(peak_df.columns)):
        raise ValueError(f"Peak CSV missing columns: {sorted(required_peak - set(peak_df.columns))}")

    history_df = history_df.copy()
    history_df["year"] = pd.to_numeric(history_df["year"], errors="coerce")
    history_df["CO2"] = pd.to_numeric(history_df["CO2"], errors="coerce")
    history_df = history_df.dropna(subset=["year", "CO2"])  # type: ignore[arg-type]
    history_df["year"] = history_df["year"].astype(int)
    history_df = history_df[(history_df["year"] >= 1990) & (history_df["year"] <= 2023)].copy()

    history_national_df = (
        history_df.groupby("year", as_index=False)["CO2"]
        .sum()
        .rename(columns={"CO2": "co2_actual"})
        .sort_values("year", kind="stable")
    )

    co2_scale = _infer_forecast_co2_scale(
        history_national_df=history_national_df,
        national_df=national_df,
    )
    if co2_scale != 1.0:
        detail_df = detail_df.copy()
        national_df = national_df.copy()
        peak_df = peak_df.copy()
        detail_df["co2_pred"] = pd.to_numeric(detail_df["co2_pred"], errors="coerce") * co2_scale
        national_df["co2_pred"] = pd.to_numeric(national_df["co2_pred"], errors="coerce") * co2_scale
        peak_df["peak_co2"] = pd.to_numeric(peak_df["peak_co2"], errors="coerce") * co2_scale

    trend_png = args.out_dir / "scenario_forecast_national_trend.png"
    key_png = args.out_dir / "scenario_forecast_key_years.png"
    province_trend_png = args.out_dir / "scenario_forecast_province_trends.png"
    province_heatmap_png = args.out_dir / "scenario_forecast_province_heatmap_2030.png"

    plot_trend_with_peak(
        national_df=national_df,
        peak_df=peak_df,
        history_national_df=history_national_df,
        out_file=trend_png,
    )
    plot_key_years(national_df=national_df, out_file=key_png)
    plot_province_trends(detail_df=detail_df, out_file=province_trend_png)
    plot_province_heatmap(detail_df=detail_df, out_file=province_heatmap_png, year=2030)
    panel_files = plot_province_panels_grouped(
        detail_df=detail_df,
        history_df=history_df,
        out_dir=args.out_dir,
        provinces_per_figure=args.provinces_per_figure,
    )

    print(f"Saved figure: {trend_png}")
    print(f"Saved figure: {key_png}")
    print(f"Saved figure: {province_trend_png}")
    print(f"Saved figure: {province_heatmap_png}")
    for f in panel_files:
        print(f"Saved figure: {f}")


if __name__ == "__main__":
    main()
