from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stirpat_ee_gru import PanelRidgeSTIRPAT

EXCLUDED_PROVINCES = {"Tibet"}


@dataclass
class DirectCO2Config:
    input_csv: Path
    output_dir: Path
    train_end_year: int = 2020
    valid_start_year: int = 2021
    valid_end_year: int = 2023
    ridge_alpha: float = 1.0


MODEL_SPECS: Dict[str, List[str]] = {
    # More suitable for explanation: no direct Energy total, but keeps energy intensity.
    "core_drivers": [
        "log_Population",
        "log_pGDP",
        "Industry",
        "Urbanization",
        "EnergyIntensity",
    ],
    # Adds energy structure, useful for policy interpretation.
    "structure_drivers": [
        "log_Population",
        "log_pGDP",
        "Industry",
        "Urbanization",
        "EnergyIntensity",
        "CoalShare",
        "OilShare",
        "GasShare",
        "NonFossilShare",
    ],
    # Benchmark with direct energy use. Strong predictive benchmark, weaker as a causal driver model.
    "with_energy": [
        "log_Population",
        "log_pGDP",
        "log_Energy",
        "Industry",
        "Urbanization",
        "CoalShare",
        "OilShare",
        "GasShare",
    ],
}


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    den = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_pred - y_true) / den)))


def _smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    den = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / den))


def _metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "mape": np.nan, "smape": np.nan, "r2": np.nan}
    yt = df["y_true"].to_numpy(dtype=float)
    yp = df["y_pred"].to_numpy(dtype=float)
    err = yp - yt
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    return {
        "n": int(len(df)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape": _safe_mape(yt, yp),
        "smape": _smape(yt, yp),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan,
    }


def prepare_panel(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    required = {
        "province",
        "year",
        "CO2",
        "GDP",
        "Population",
        "Energy",
        "CoalShare",
        "OilShare",
        "GasShare",
        "NonFossilShare",
        "Industry",
        "Urbanization",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["province"] = df["province"].astype(str)
    df = df.loc[~df["province"].isin(EXCLUDED_PROVINCES)].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)

    numeric_cols = sorted(required - {"province"})
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["pGDP"] = df["GDP"] / df["Population"]
    df["EnergyIntensity"] = df["Energy"] / df["GDP"]
    df["CarbonIntensity"] = df["CO2"] / df["GDP"]
    for col in ["CO2", "GDP", "Population", "Energy", "pGDP", "CarbonIntensity"]:
        df[f"log_{col}"] = np.log(df[col].clip(lower=1e-8))

    return df.sort_values(["province", "year"], kind="stable").reset_index(drop=True)


def fit_predict_one(df: pd.DataFrame, cfg: DirectCO2Config, model_name: str, feature_cols: List[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(f"{model_name} missing feature columns: {sorted(missing)}")

    train = df[df["year"] <= cfg.train_end_year].copy()
    model = PanelRidgeSTIRPAT(alpha=cfg.ridge_alpha)
    model.fit(train, feature_cols=feature_cols, target_col="log_CO2", province_col="province")

    pred = df[["province", "year", "CO2"] + feature_cols].copy()
    pred["model"] = model_name
    pred["split"] = np.select(
        [
            pred["year"] <= cfg.train_end_year,
            (pred["year"] >= cfg.valid_start_year) & (pred["year"] <= cfg.valid_end_year),
        ],
        ["train", "valid"],
        default="other",
    )
    pred["log_co2_pred"] = model.predict(df, province_col="province")
    pred["co2_pred"] = np.exp(pred["log_co2_pred"].to_numpy(dtype=float))
    pred = pred.rename(columns={"CO2": "co2_actual"})
    pred["abs_pct_err"] = np.abs(
        (pred["co2_pred"] - pred["co2_actual"]) / np.maximum(np.abs(pred["co2_actual"]), 1e-8)
    )

    beta = model.result.beta if model.result is not None else np.array([])
    coef_rows = [
        {
            "model": model_name,
            "feature": feature,
            "coef": float(beta[i]),
            "note": "direct log_CO2 ridge coefficient; province fixed effects omitted",
        }
        for i, feature in enumerate(feature_cols)
    ]
    return pred, pd.DataFrame(coef_rows)


def summarize(pred_all: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: List[Dict[str, Any]] = []
    national_rows: List[pd.DataFrame] = []

    for model_name, mdf in pred_all.groupby("model", sort=False):
        nat = (
            mdf.groupby(["model", "year"], as_index=False)[["co2_actual", "co2_pred"]]
            .sum()
            .sort_values(["model", "year"], kind="stable")
        )
        nat["split"] = np.select(
            [
                nat["year"] <= 2020,
                (nat["year"] >= 2021) & (nat["year"] <= 2023),
            ],
            ["train", "valid"],
            default="other",
        )
        national_rows.append(nat)

        for split in ["train", "valid", "all"]:
            part = mdf if split == "all" else mdf[mdf["split"] == split]
            metric_rows.append(
                {
                    "scope": "province_pooled",
                    "model": model_name,
                    "split": split,
                    **_metrics(part["co2_actual"], part["co2_pred"]),
                }
            )

            nat_part = nat if split == "all" else nat[nat["split"] == split]
            metric_rows.append(
                {
                    "scope": "national_yearly",
                    "model": model_name,
                    "split": split,
                    **_metrics(nat_part["co2_actual"], nat_part["co2_pred"]),
                }
            )

    return pd.DataFrame(metric_rows), pd.concat(national_rows, axis=0, ignore_index=True)


def plot_national(national: pd.DataFrame, out_file: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.2))
    actual = (
        national[["year", "co2_actual"]]
        .drop_duplicates("year")
        .sort_values("year", kind="stable")
    )
    ax.plot(actual["year"], actual["co2_actual"], color="#111111", linewidth=2.5, label="Actual CO2")
    styles = {
        "core_drivers": {"color": "#1f77b4", "ls": "--"},
        "structure_drivers": {"color": "#2ca02c", "ls": "-"},
        "with_energy": {"color": "#d62728", "ls": "-."},
    }
    for model_name, grp in national.groupby("model", sort=False):
        st = styles.get(model_name, {"color": "#555555", "ls": ":"})
        g = grp.sort_values("year", kind="stable")
        ax.plot(g["year"], g["co2_pred"], color=st["color"], linestyle=st["ls"], linewidth=2.0, label=model_name)
    ax.axvspan(2021, 2023, color="#f2c94c", alpha=0.15, label="Validation period")
    ax.set_title("Direct CO2 Model Backtest (1990-2023)", fontsize=14)
    ax.set_xlabel("Year")
    ax.set_ylabel("CO2")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_file, dpi=300)
    plt.close(fig)


def parse_args() -> DirectCO2Config:
    parser = argparse.ArgumentParser(description="Directly model historical CO2 against driving factors.")
    base = SCRIPT_DIR
    parser.add_argument("--input-csv", type=Path, default=base.parent / "Preprocess" / "output" / "panel_master.csv")
    parser.add_argument("--output-dir", type=Path, default=base / "output" / "direct_co2_backtest")
    parser.add_argument("--train-end-year", type=int, default=2020)
    parser.add_argument("--valid-start-year", type=int, default=2021)
    parser.add_argument("--valid-end-year", type=int, default=2023)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    args = parser.parse_args()
    return DirectCO2Config(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        train_end_year=args.train_end_year,
        valid_start_year=args.valid_start_year,
        valid_end_year=args.valid_end_year,
        ridge_alpha=args.ridge_alpha,
    )


def main() -> None:
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    df = prepare_panel(cfg.input_csv)

    pred_parts: List[pd.DataFrame] = []
    coef_parts: List[pd.DataFrame] = []
    for model_name, feature_cols in MODEL_SPECS.items():
        pred, coef = fit_predict_one(df, cfg, model_name, feature_cols)
        pred_parts.append(pred)
        coef_parts.append(coef)

    pred_all = pd.concat(pred_parts, axis=0, ignore_index=True)
    coef_all = pd.concat(coef_parts, axis=0, ignore_index=True)
    metrics, national = summarize(pred_all)

    detail_out = cfg.output_dir / "direct_co2_backtest_detail.csv"
    national_out = cfg.output_dir / "direct_co2_backtest_national.csv"
    metrics_out = cfg.output_dir / "direct_co2_backtest_metrics.csv"
    coef_out = cfg.output_dir / "direct_co2_coefficients.csv"
    json_out = cfg.output_dir / "direct_co2_backtest_summary.json"
    fig_out = cfg.output_dir / "direct_co2_backtest_national.png"

    pred_all.to_csv(detail_out, index=False, encoding="utf-8")
    national.to_csv(national_out, index=False, encoding="utf-8")
    metrics.to_csv(metrics_out, index=False, encoding="utf-8")
    coef_all.to_csv(coef_out, index=False, encoding="utf-8")
    json_out.write_text(
        json.dumps(
            {
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
                "model_specs": MODEL_SPECS,
                "metrics": metrics.to_dict(orient="records"),
                "note": "All direct CO2 models are trained through train_end_year and evaluated on 2021-2023 validation years. Province fixed effects are included by PanelRidgeSTIRPAT.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_national(national, fig_out)

    print("=== Direct CO2 Backtest Metrics ===")
    print(metrics.to_string(index=False))
    print(f"Saved detail csv: {detail_out}")
    print(f"Saved national csv: {national_out}")
    print(f"Saved metrics csv: {metrics_out}")
    print(f"Saved coefficients csv: {coef_out}")
    print(f"Saved figure: {fig_out}")


if __name__ == "__main__":
    main()
