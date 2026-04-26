from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stirpat_ee_gru import EntityEmbeddingGRU

EXCLUDED_PROVINCES = {"Tibet"}


@dataclass
class BacktestConfig:
    panel_with_residual_csv: Path
    dataset_npz: Path
    model_ckpt: Path
    output_dir: Path
    start_year: int = 1990
    end_year: int = 2023
    window: int = 3
    batch_size: int = 256


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


def _load_model(model_ckpt: Path, device: torch.device) -> EntityEmbeddingGRU:
    ckpt = torch.load(model_ckpt, map_location=device, weights_only=False)
    shape_info = ckpt["shape_info"]
    train_cfg = ckpt.get("train_config", {})
    model = EntityEmbeddingGRU(
        num_provinces=int(shape_info["num_provinces"]),
        num_dynamic_features=int(shape_info["num_dynamic_features"]),
        embed_dim=int(train_cfg.get("embed_dim", 8)),
        hidden_dim=int(train_cfg.get("hidden_dim", 32)),
        dropout=float(train_cfg.get("dropout", 0.2)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _predict_residual(
    model: EntityEmbeddingGRU,
    dynamic_x: np.ndarray,
    lag_residual_x: np.ndarray,
    province_idx: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for s in range(0, dynamic_x.shape[0], batch_size):
            e = min(s + batch_size, dynamic_x.shape[0])
            x_dyn = torch.tensor(dynamic_x[s:e], dtype=torch.float32, device=device)
            x_lag = torch.tensor(lag_residual_x[s:e], dtype=torch.float32, device=device)
            x_pid = torch.tensor(province_idx[s:e], dtype=torch.long, device=device)
            pred = model(x_dyn, x_lag, x_pid).detach().cpu().numpy()
            preds.append(pred)
    return np.concatenate(preds, axis=0)


def _prepare_panel(cfg: BacktestConfig, npz_data: Any) -> pd.DataFrame:
    panel = pd.read_csv(cfg.panel_with_residual_csv)
    panel["province"] = panel["province"].astype(str)
    panel = panel.loc[~panel["province"].isin(EXCLUDED_PROVINCES)].copy()
    panel["year"] = pd.to_numeric(panel["year"], errors="coerce").astype("Int64")
    panel = panel.dropna(subset=["year"]).copy()
    panel["year"] = panel["year"].astype(int)
    panel = panel[(panel["year"] >= cfg.start_year) & (panel["year"] <= cfg.end_year)].copy()

    required = {
        "province",
        "year",
        "CO2",
        "Energy",
        "stirpat_log_pred",
        "residual",
    }
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"Missing required columns in panel_with_residual: {sorted(missing)}")

    for col in ["CO2", "Energy", "stirpat_log_pred", "residual"]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")

    if "province_id" not in panel.columns:
        province_levels = sorted(panel["province"].unique().tolist())
        panel["province_id"] = panel["province"].map({p: i for i, p in enumerate(province_levels)}).astype(int)
    else:
        panel["province_id"] = pd.to_numeric(panel["province_id"], errors="coerce").astype(int)

    gru_cols = [str(x) for x in npz_data["gru_feature_names"].tolist()]
    mean = npz_data["standardizer_mean"].astype(np.float32)
    std = npz_data["standardizer_std"].astype(np.float32)
    missing_gru = set(gru_cols) - set(panel.columns)
    if missing_gru:
        raise ValueError(f"Missing GRU feature columns: {sorted(missing_gru)}")

    panel = panel.sort_values(["province", "year"], kind="stable").reset_index(drop=True)
    panel["true_energy"] = panel["Energy"]
    panel["true_co2"] = panel["CO2"]
    panel["stirpat_energy_pred"] = np.exp(panel["stirpat_log_pred"].to_numpy(dtype=float))
    panel["observed_co2_per_energy"] = panel["true_co2"] / np.clip(panel["true_energy"], 1e-8, None)
    panel["stirpat_co2_pred"] = panel["stirpat_energy_pred"] * panel["observed_co2_per_energy"]

    standardized = (panel[gru_cols].to_numpy(dtype=np.float32) - mean) / std
    panel_std = panel.copy()
    std_cols = [f"__std_{col}" for col in gru_cols]
    panel_std[std_cols] = standardized
    panel_std.attrs["gru_cols"] = std_cols
    return panel_std


def build_hybrid_reconstruction(
    panel_std: pd.DataFrame,
    cfg: BacktestConfig,
    model: EntityEmbeddingGRU,
    device: torch.device,
) -> pd.DataFrame:
    gru_cols: List[str] = panel_std.attrs["gru_cols"]
    dynamic_list: List[np.ndarray] = []
    lag_res_list: List[np.ndarray] = []
    province_list: List[int] = []
    row_index_list: List[int] = []

    for _, grp in panel_std.groupby("province", sort=False):
        g = grp.sort_values("year", kind="stable")
        years = g["year"].to_numpy(dtype=int)
        idx = g.index.to_numpy()
        for end_pos in range(cfg.window, len(g)):
            start_pos = end_pos - cfg.window
            if np.any(np.diff(years[start_pos : end_pos + 1]) != 1):
                continue
            dynamic_list.append(g.iloc[start_pos:end_pos][gru_cols].to_numpy(dtype=np.float32))
            lag_res_list.append(g.iloc[start_pos:end_pos][["residual"]].to_numpy(dtype=np.float32))
            province_list.append(int(g.iloc[end_pos]["province_id"]))
            row_index_list.append(int(idx[end_pos]))

    out = panel_std.copy()
    out["hybrid_residual_pred"] = np.nan
    out["hybrid_energy_pred"] = np.nan
    out["hybrid_co2_pred"] = np.nan

    if dynamic_list:
        pred_residual = _predict_residual(
            model=model,
            dynamic_x=np.stack(dynamic_list, axis=0),
            lag_residual_x=np.stack(lag_res_list, axis=0),
            province_idx=np.asarray(province_list, dtype=np.int64),
            device=device,
            batch_size=cfg.batch_size,
        )
        hybrid_log_energy = out.loc[row_index_list, "stirpat_log_pred"].to_numpy(dtype=float) + pred_residual
        hybrid_energy = np.exp(hybrid_log_energy)
        out.loc[row_index_list, "hybrid_residual_pred"] = pred_residual
        out.loc[row_index_list, "hybrid_energy_pred"] = hybrid_energy
        out.loc[row_index_list, "hybrid_co2_pred"] = (
            hybrid_energy * out.loc[row_index_list, "observed_co2_per_energy"].to_numpy(dtype=float)
        )

    return out


def summarize_metrics(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    for target in ["energy", "co2"]:
        true_col = f"true_{target}"
        for model_name, pred_col in [
            ("stirpat_only", f"stirpat_{target}_pred"),
            ("hybrid_reconstruct", f"hybrid_{target}_pred"),
        ]:
            m = _metrics(detail[true_col], detail[pred_col])
            rows.append({"scope": "province_pooled", "target": target, "model": model_name, **m})

    national = (
        detail.groupby("year", as_index=False)[
            [
                "true_energy",
                "stirpat_energy_pred",
                "hybrid_energy_pred",
                "true_co2",
                "stirpat_co2_pred",
                "hybrid_co2_pred",
            ]
        ]
        .sum(min_count=1)
        .sort_values("year", kind="stable")
    )

    for target in ["energy", "co2"]:
        true_col = f"true_{target}"
        for model_name, pred_col in [
            ("stirpat_only", f"stirpat_{target}_pred"),
            ("hybrid_reconstruct", f"hybrid_{target}_pred"),
        ]:
            m = _metrics(national[true_col], national[pred_col])
            rows.append({"scope": "national_yearly", "target": target, "model": model_name, **m})

    return pd.DataFrame(rows), national


def plot_national_backtest(national: pd.DataFrame, out_file: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.plot(national["year"], national["true_co2"], color="#111111", linewidth=2.4, label="Actual CO2")
    ax.plot(
        national["year"],
        national["stirpat_co2_pred"],
        color="#1f77b4",
        linestyle="--",
        linewidth=2.0,
        label="STIRPAT reconstructed CO2",
    )
    hybrid = national.dropna(subset=["hybrid_co2_pred"])
    ax.plot(
        hybrid["year"],
        hybrid["hybrid_co2_pred"],
        color="#2ca02c",
        linestyle="-",
        linewidth=2.0,
        label="STIRPAT + EE-GRU reconstructed CO2",
    )
    ax.set_title("Historical CO2 Backtest (1990-2023)", fontsize=14)
    ax.set_xlabel("Year")
    ax.set_ylabel("CO2")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_file, dpi=300)
    plt.close(fig)


def parse_args() -> BacktestConfig:
    parser = argparse.ArgumentParser(description="Backtest historical CO2 reconstruction for 1990-2023.")
    base = SCRIPT_DIR
    parser.add_argument(
        "--panel-with-residual-csv",
        type=Path,
        default=base / "output" / "dataset" / "panel_with_residual.csv",
    )
    parser.add_argument(
        "--dataset-npz",
        type=Path,
        default=base / "output" / "dataset" / "stirpat_ee_gru_dataset.npz",
    )
    parser.add_argument("--model-ckpt", type=Path, default=base / "output" / "model" / "best_ee_gru.pt")
    parser.add_argument("--output-dir", type=Path, default=base / "output" / "historical_backtest")
    parser.add_argument("--start-year", type=int, default=1990)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    return BacktestConfig(
        panel_with_residual_csv=args.panel_with_residual_csv,
        dataset_npz=args.dataset_npz,
        model_ckpt=args.model_ckpt,
        output_dir=args.output_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        window=args.window,
        batch_size=args.batch_size,
    )


def main() -> None:
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    npz_data = np.load(cfg.dataset_npz, allow_pickle=True)
    panel_std = _prepare_panel(cfg, npz_data)
    model = _load_model(cfg.model_ckpt, device=device)
    detail = build_hybrid_reconstruction(panel_std, cfg, model, device)

    metrics, national = summarize_metrics(detail)

    detail_out = cfg.output_dir / "historical_backtest_detail.csv"
    national_out = cfg.output_dir / "historical_backtest_national.csv"
    metrics_out = cfg.output_dir / "historical_backtest_metrics.csv"
    json_out = cfg.output_dir / "historical_backtest_metrics.json"
    fig_out = cfg.output_dir / "historical_backtest_national_co2.png"

    detail_export = detail.drop(columns=[c for c in detail.columns if c.startswith("__std_")])
    detail_export.to_csv(detail_out, index=False, encoding="utf-8")
    national.to_csv(national_out, index=False, encoding="utf-8")
    metrics.to_csv(metrics_out, index=False, encoding="utf-8")
    json_out.write_text(
        json.dumps(
            {
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
                "metrics": metrics.to_dict(orient="records"),
                "note": "Hybrid reconstruction starts after the lag window. CO2 predictions are converted from predicted energy using observed historical CO2/Energy intensity.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_national_backtest(national, fig_out)

    print("=== Historical Backtest Metrics (1990-2023) ===")
    print(metrics.to_string(index=False))
    print(f"Saved detail csv: {detail_out}")
    print(f"Saved national csv: {national_out}")
    print(f"Saved metrics csv: {metrics_out}")
    print(f"Saved metrics json: {json_out}")
    print(f"Saved figure: {fig_out}")


if __name__ == "__main__":
    main()
