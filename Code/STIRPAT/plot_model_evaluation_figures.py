from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot model evaluation figures for STIRPAT and STIRPAT-EE-GRU.")
    base = SCRIPT_DIR / "output"
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=base / "historical_backtest" / "historical_backtest_metrics.csv",
        help="Historical backtest metrics CSV path.",
    )
    parser.add_argument(
        "--detail-csv",
        type=Path,
        default=base / "historical_backtest" / "historical_backtest_detail.csv",
        help="Historical backtest detail CSV path.",
    )
    parser.add_argument(
        "--national-csv",
        type=Path,
        default=base / "historical_backtest" / "historical_backtest_national.csv",
        help="Historical backtest national yearly CSV path.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=base / "historical_backtest" / "paper_figures",
        help="Directory for exported figures.",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    return pd.read_csv(path)


def plot_metrics_comparison(metrics_df: pd.DataFrame, out_file: Path) -> None:
    df = metrics_df.copy()
    df["scope_label"] = df["scope"].map(
        {
            "province_pooled": "Province pooled",
            "national_yearly": "National yearly",
        }
    )
    df["model_label"] = df["model"].map(
        {
            "stirpat_only": "STIRPAT",
            "hybrid_reconstruct": "STIRPAT-EE-GRU",
        }
    )
    df = df[df["target"].astype(str).str.lower() == "co2"].copy()
    if df.empty:
        raise ValueError("No CO2 rows found in metrics CSV.")

    metrics = [
        ("mae", "MAE"),
        ("rmse", "RMSE"),
        ("mape", "MAPE"),
        ("r2", "R²"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2))
    axes = axes.ravel()
    x = np.arange(df["scope_label"].nunique())
    width = 0.36
    order_scope = ["Province pooled", "National yearly"]
    order_model = ["STIRPAT", "STIRPAT-EE-GRU"]
    colors = {"STIRPAT": "#1f77b4", "STIRPAT-EE-GRU": "#2ca02c"}

    for ax, (col, title) in zip(axes, metrics):
        pivot = (
            df.pivot_table(index="scope_label", columns="model_label", values=col, aggfunc="first")
            .reindex(order_scope)
            .reindex(columns=order_model)
        )
        for i, model_name in enumerate(order_model):
            vals = pivot[model_name].to_numpy(dtype=float)
            if col == "mape":
                vals = vals * 100.0
            ax.bar(x + (i - 0.5) * width, vals, width=width, color=colors[model_name], label=model_name)

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(order_scope)
        ax.grid(axis="y", alpha=0.22, linestyle="--")
        if col == "mape":
            ax.set_ylabel("%")
        elif col == "r2":
            ax.set_ylim(min(0.90, float(np.nanmin(pivot.to_numpy(dtype=float))) - 0.01), 1.0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("CO2 Reconstruction Metrics Comparison", y=0.96, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_file, dpi=300)
    plt.close(fig)


def plot_national_residuals(national_df: pd.DataFrame, out_file: Path) -> None:
    df = national_df.copy()
    for col in ["true_co2", "stirpat_co2_pred", "hybrid_co2_pred"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["stirpat_residual"] = df["true_co2"] - df["stirpat_co2_pred"]
    df["hybrid_residual"] = df["true_co2"] - df["hybrid_co2_pred"]

    fig, ax = plt.subplots(figsize=(11.8, 5.8))
    ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle=":")
    ax.plot(df["year"], df["stirpat_residual"], color="#1f77b4", linestyle="--", linewidth=2.0, label="STIRPAT residual")
    hybrid = df.dropna(subset=["hybrid_residual"])
    ax.plot(hybrid["year"], hybrid["hybrid_residual"], color="#2ca02c", linewidth=2.0, label="STIRPAT-EE-GRU residual")
    ax.set_title("National Yearly CO2 Residual Comparison")
    ax.set_xlabel("Year")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.grid(alpha=0.22, linestyle="--")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_file, dpi=300)
    plt.close(fig)


def plot_province_error_boxplot(detail_df: pd.DataFrame, out_file: Path) -> None:
    df = detail_df.copy()
    for col in ["true_co2", "stirpat_co2_pred", "hybrid_co2_pred"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    base = df.dropna(subset=["true_co2", "stirpat_co2_pred"]).copy()
    hybrid = df.dropna(subset=["true_co2", "hybrid_co2_pred"]).copy()

    stirpat_ape = (np.abs(base["stirpat_co2_pred"] - base["true_co2"]) / np.clip(np.abs(base["true_co2"]), 1e-8, None)) * 100.0
    hybrid_ape = (np.abs(hybrid["hybrid_co2_pred"] - hybrid["true_co2"]) / np.clip(np.abs(hybrid["true_co2"]), 1e-8, None)) * 100.0

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    bp = ax.boxplot(
        [stirpat_ape.to_numpy(dtype=float), hybrid_ape.to_numpy(dtype=float)],
        tick_labels=["STIRPAT", "STIRPAT-EE-GRU"],
        patch_artist=True,
        showfliers=False,
    )
    colors = ["#1f77b4", "#2ca02c"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)

    ax.set_title("Province-level CO2 Absolute Percentage Error Distribution")
    ax.set_ylabel("Absolute percentage error (%)")
    ax.grid(axis="y", alpha=0.22, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_file, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = _read_csv(args.metrics_csv)
    detail_df = _read_csv(args.detail_csv)
    national_df = _read_csv(args.national_csv)

    plot_metrics_comparison(metrics_df, args.out_dir / "co2_metrics_comparison.png")
    plot_national_residuals(national_df, args.out_dir / "national_co2_residual_comparison.png")
    plot_province_error_boxplot(detail_df, args.out_dir / "province_co2_error_boxplot.png")

    print(f"Saved figures to: {args.out_dir}")


if __name__ == "__main__":
    main()
