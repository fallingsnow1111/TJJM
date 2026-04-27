from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stirpat_ee_gru import EntityEmbeddingGRU, PanelRidgeSTIRPAT

EXCLUDED_PROVINCES = {"Tibet"}


@dataclass
class ValidateConfig:
    input_csv: Path
    output_dir: Path
    train_end_year: int = 2020
    valid_start_year: int = 2021
    valid_end_year: int = 2023
    window: int = 3
    ridge_alpha: float = 1.0
    batch_size: int = 64
    epochs: int = 120
    lr: float = 1e-3
    weight_decay: float = 1e-4
    embed_dim: int = 8
    hidden_dim: int = 32
    dropout: float = 0.2
    patience: int = 15
    max_grad_norm: float = 1.0
    seed: int = 42


class ResidualDataset(Dataset):
    def __init__(
        self,
        dynamic_x: np.ndarray,
        lag_residual_x: np.ndarray,
        province_idx: np.ndarray,
        target_residual: np.ndarray,
    ):
        self.dynamic_x = torch.tensor(dynamic_x, dtype=torch.float32)
        self.lag_residual_x = torch.tensor(lag_residual_x, dtype=torch.float32)
        self.province_idx = torch.tensor(province_idx, dtype=torch.long)
        self.target_residual = torch.tensor(target_residual, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.dynamic_x.shape[0])

    def __getitem__(self, idx: int):
        return (
            self.dynamic_x[idx],
            self.lag_residual_x[idx],
            self.province_idx[idx],
            self.target_residual[idx],
        )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_log(series: pd.Series, eps: float = 1e-8) -> pd.Series:
    logged = np.log(pd.to_numeric(series, errors="coerce").clip(lower=eps))
    return pd.Series(logged, index=series.index)


def reg_metrics(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "mape": np.nan, "smape": np.nan, "r2": np.nan}

    err = y_pred - y_true
    den = np.maximum(np.abs(y_true), eps)
    smape_den = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "n": int(y_true.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape": float(np.mean(np.abs(err) / den)),
        "smape": float(np.mean(2.0 * np.abs(err) / smape_den)),
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

    for col in ["CO2", "GDP", "Population", "Energy", "Industry", "Urbanization"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["pGDP"] = df["GDP"] / df["Population"]
    df["EnergyIntensity"] = df["Energy"] / df["GDP"]
    df["log_GDP"] = safe_log(df["GDP"])
    df["log_Population"] = safe_log(df["Population"])
    df["log_Energy"] = safe_log(df["Energy"])
    df["log_pGDP"] = safe_log(df["pGDP"])
    df["log_EnergyIntensity"] = safe_log(df["EnergyIntensity"])
    df["log_Industry"] = safe_log(df["Industry"])
    df["log_Urbanization"] = safe_log(df["Urbanization"])

    province_levels = sorted(df["province"].unique().tolist())
    province_mapping = {p: i for i, p in enumerate(province_levels)}
    df["province_id"] = df["province"].map(province_mapping).astype(int)

    return df.sort_values(["province", "year"], kind="stable").reset_index(drop=True)


def build_samples(
    df: pd.DataFrame,
    cfg: ValidateConfig,
    stirpat_feature_cols: List[str],
) -> Dict[str, Any]:
    train_mask = df["year"] <= cfg.train_end_year
    stirpat = PanelRidgeSTIRPAT(alpha=cfg.ridge_alpha)
    stirpat.fit(
        df=df.loc[train_mask].copy(),
        feature_cols=stirpat_feature_cols,
        target_col="log_Energy",
        province_col="province",
    )

    panel = df.copy()
    panel["stirpat_log_pred"] = stirpat.predict(panel, province_col="province")
    panel["residual"] = panel["log_Energy"] - panel["stirpat_log_pred"]

    gru_cols = [
        "log_GDP",
        "log_pGDP",
        "log_Population",
        "Energy",
        "EnergyIntensity",
        "Industry",
        "Urbanization",
    ]
    standardize_ref = panel.loc[train_mask, gru_cols].to_numpy(dtype=np.float32)
    mean = standardize_ref.mean(axis=0)
    std = standardize_ref.std(axis=0)
    std[std < 1e-8] = 1.0
    panel_std = panel.copy()
    panel_std[gru_cols] = (panel_std[gru_cols].to_numpy(dtype=np.float32) - mean) / std

    records: List[Dict[str, Any]] = []
    for _, grp in panel_std.groupby("province", sort=False):
        g = grp.sort_values("year", kind="stable").reset_index(drop=False)
        years = g["year"].to_numpy(dtype=int)

        for end_idx in range(cfg.window, len(g)):
            start_idx = end_idx - cfg.window
            if np.any(np.diff(years[start_idx : end_idx + 1]) != 1):
                continue

            row = g.iloc[end_idx]
            records.append(
                {
                    "row_index": int(row["index"]),
                    "province": str(row["province"]),
                    "province_id": int(row["province_id"]),
                    "year": int(row["year"]),
                    "dynamic_x": g.loc[start_idx : end_idx - 1, gru_cols].to_numpy(dtype=np.float32),
                    "lag_residual_x": g.loc[start_idx : end_idx - 1, ["residual"]].to_numpy(dtype=np.float32),
                    "target_residual": float(row["residual"]),
                    "target_log_energy": float(row["log_Energy"]),
                    "target_stirpat_log": float(row["stirpat_log_pred"]),
                    "true_energy": float(np.exp(row["log_Energy"])),
                    "true_co2": float(pd.to_numeric(df.loc[int(row["index"]), "CO2"], errors="coerce")),
                }
            )

    if not records:
        raise RuntimeError("No sliding-window samples generated.")

    sample_df = pd.DataFrame(records)
    train_s = sample_df[sample_df["year"] <= cfg.train_end_year].copy()
    valid_s = sample_df[
        (sample_df["year"] >= cfg.valid_start_year) & (sample_df["year"] <= cfg.valid_end_year)
    ].copy()

    if train_s.empty or valid_s.empty:
        raise RuntimeError("Train or valid split is empty under current year settings.")

    def _pack_split(part: pd.DataFrame) -> Dict[str, np.ndarray]:
        return {
            "dynamic_x": np.stack(part["dynamic_x"].to_list(), axis=0).astype(np.float32),
            "lag_residual_x": np.stack(part["lag_residual_x"].to_list(), axis=0).astype(np.float32),
            "province_idx": part["province_id"].to_numpy(dtype=np.int64),
            "target_residual": part["target_residual"].to_numpy(dtype=np.float32),
            "target_log_energy": part["target_log_energy"].to_numpy(dtype=np.float32),
            "target_stirpat_log": part["target_stirpat_log"].to_numpy(dtype=np.float32),
            "true_energy": part["true_energy"].to_numpy(dtype=np.float64),
            "true_co2": part["true_co2"].to_numpy(dtype=np.float64),
            "year": part["year"].to_numpy(dtype=np.int32),
            "province": part["province"].to_numpy(dtype=object),
        }

    return {
        "train": _pack_split(train_s),
        "valid": _pack_split(valid_s),
        "num_provinces": int(df["province_id"].nunique()),
        "num_dynamic_features": 7,
        "sample_df": sample_df,
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: float,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for dynamic_x, lag_residual_x, province_idx, target in loader:
        dynamic_x = dynamic_x.to(device)
        lag_residual_x = lag_residual_x.to(device)
        province_idx = province_idx.to(device)
        target = target.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(dynamic_x, lag_residual_x, province_idx)
        loss = criterion(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        bs = target.size(0)
        total_loss += loss.item() * bs
        n += bs
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    for dynamic_x, lag_residual_x, province_idx, target in loader:
        dynamic_x = dynamic_x.to(device)
        lag_residual_x = lag_residual_x.to(device)
        province_idx = province_idx.to(device)
        target = target.to(device)
        pred = model(dynamic_x, lag_residual_x, province_idx)
        loss = criterion(pred, target)
        bs = target.size(0)
        total_loss += loss.item() * bs
        n += bs
    return total_loss / max(n, 1)


@torch.no_grad()
def predict_residuals(
    model: nn.Module,
    dynamic_x: np.ndarray,
    lag_residual_x: np.ndarray,
    province_idx: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    preds: List[np.ndarray] = []
    for s in range(0, dynamic_x.shape[0], batch_size):
        e = min(s + batch_size, dynamic_x.shape[0])
        x_dyn = torch.tensor(dynamic_x[s:e], dtype=torch.float32, device=device)
        x_lag = torch.tensor(lag_residual_x[s:e], dtype=torch.float32, device=device)
        x_pid = torch.tensor(province_idx[s:e], dtype=torch.long, device=device)
        out = model(x_dyn, x_lag, x_pid).detach().cpu().numpy()
        preds.append(out)
    return np.concatenate(preds, axis=0)


def train_and_evaluate_variant(
    variant_name: str,
    stirpat_feature_cols: List[str],
    panel_df: pd.DataFrame,
    cfg: ValidateConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    sample_pack = build_samples(panel_df, cfg, stirpat_feature_cols)
    train_np = sample_pack["train"]
    valid_np = sample_pack["valid"]

    train_ds = ResidualDataset(
        dynamic_x=train_np["dynamic_x"],
        lag_residual_x=train_np["lag_residual_x"],
        province_idx=train_np["province_idx"],
        target_residual=train_np["target_residual"],
    )
    valid_ds = ResidualDataset(
        dynamic_x=valid_np["dynamic_x"],
        lag_residual_x=valid_np["lag_residual_x"],
        province_idx=valid_np["province_idx"],
        target_residual=valid_np["target_residual"],
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EntityEmbeddingGRU(
        num_provinces=int(sample_pack["num_provinces"]),
        num_dynamic_features=int(sample_pack["num_dynamic_features"]),
        embed_dim=cfg.embed_dim,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
    ).to(device)

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_valid_loss = float("inf")
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    patience_count = 0
    train_trace: List[Dict[str, float]] = []

    for epoch in range(1, cfg.epochs + 1):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            max_grad_norm=cfg.max_grad_norm,
        )
        valid_loss = evaluate_loss(model=model, loader=valid_loader, criterion=criterion, device=device)
        train_trace.append({"epoch": float(epoch), "train_loss": float(train_loss), "valid_loss": float(valid_loss)})

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= cfg.patience:
            break

    model.load_state_dict(best_state)

    pred_valid_res = predict_residuals(
        model=model,
        dynamic_x=valid_np["dynamic_x"],
        lag_residual_x=valid_np["lag_residual_x"],
        province_idx=valid_np["province_idx"],
        device=device,
        batch_size=cfg.batch_size,
    )

    true_energy = valid_np["true_energy"].astype(float)
    true_co2 = valid_np["true_co2"].astype(float)
    stirpat_log = valid_np["target_stirpat_log"].astype(float)
    hybrid_log = stirpat_log + pred_valid_res.astype(float)

    stirpat_energy_pred = np.exp(stirpat_log)
    hybrid_energy_pred = np.exp(hybrid_log)

    co2_per_energy = true_co2 / np.clip(true_energy, 1e-8, None)
    stirpat_co2_pred = stirpat_energy_pred * co2_per_energy
    hybrid_co2_pred = hybrid_energy_pred * co2_per_energy

    metrics_rows = []
    for target_name, y_true, y_s, y_h in [
        ("energy", true_energy, stirpat_energy_pred, hybrid_energy_pred),
        ("co2", true_co2, stirpat_co2_pred, hybrid_co2_pred),
    ]:
        m_s = reg_metrics(y_true, y_s)
        m_h = reg_metrics(y_true, y_h)
        metrics_rows.append({"variant": variant_name, "target": target_name, "model": "stirpat_only", **m_s})
        metrics_rows.append({"variant": variant_name, "target": target_name, "model": "hybrid", **m_h})

    detail_df = pd.DataFrame(
        {
            "variant": variant_name,
            "year": valid_np["year"].astype(int),
            "province": valid_np["province"].astype(str),
            "true_energy": true_energy,
            "stirpat_energy_pred": stirpat_energy_pred,
            "hybrid_energy_pred": hybrid_energy_pred,
            "true_co2": true_co2,
            "stirpat_co2_pred": stirpat_co2_pred,
            "hybrid_co2_pred": hybrid_co2_pred,
            "pred_residual": pred_valid_res,
        }
    ).sort_values(["province", "year"], kind="stable")

    meta = {
        "variant": variant_name,
        "stirpat_feature_cols": stirpat_feature_cols,
        "best_epoch": int(best_epoch),
        "best_valid_loss": float(best_valid_loss),
        "train_samples": int(train_np["dynamic_x"].shape[0]),
        "valid_samples": int(valid_np["dynamic_x"].shape[0]),
        "train_trace": train_trace,
    }
    return pd.DataFrame(metrics_rows), detail_df, meta


def parse_args() -> ValidateConfig:
    parser = argparse.ArgumentParser(
        description="Validate whether full-log STIRPAT features matter in Energy->CO2 hybrid pipeline."
    )
    default_input = SCRIPT_DIR.parent / "Preprocess" / "output" / "panel_master.csv"
    default_output = SCRIPT_DIR / "output" / "robustness" / "log_transform_validation"

    parser.add_argument("--input-csv", type=Path, default=default_input)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--train-end-year", type=int, default=2020)
    parser.add_argument("--valid-start-year", type=int, default=2021)
    parser.add_argument("--valid-end-year", type=int, default=2023)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embed-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return ValidateConfig(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        train_end_year=args.train_end_year,
        valid_start_year=args.valid_start_year,
        valid_end_year=args.valid_end_year,
        window=args.window,
        ridge_alpha=args.ridge_alpha,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        patience=args.patience,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
    )


def main() -> None:
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)

    panel = prepare_panel(cfg.input_csv)

    variants = {
        # Current implementation style in project.
        "partial_log": ["log_Population", "log_pGDP", "Industry", "Urbanization", "EnergyIntensity"],
        # Strict STIRPAT-style full log transform for all explanatory terms.
        "full_log": ["log_Population", "log_pGDP", "log_Industry", "log_Urbanization", "log_EnergyIntensity"],
    }

    all_metrics: List[pd.DataFrame] = []
    all_meta: Dict[str, Any] = {}
    all_detail_paths: Dict[str, str] = {}

    for variant_name, feature_cols in variants.items():
        print(f"[RUN] variant={variant_name} features={feature_cols}")
        metrics_df, detail_df, meta = train_and_evaluate_variant(
            variant_name=variant_name,
            stirpat_feature_cols=feature_cols,
            panel_df=panel,
            cfg=cfg,
        )
        all_metrics.append(metrics_df)
        all_meta[variant_name] = meta

        detail_path = cfg.output_dir / f"{variant_name}_valid_detail.csv"
        detail_df.to_csv(detail_path, index=False, encoding="utf-8")
        all_detail_paths[variant_name] = str(detail_path)

    metrics_out = pd.concat(all_metrics, axis=0, ignore_index=True)
    metrics_path = cfg.output_dir / "log_transform_metrics.csv"
    metrics_out.to_csv(metrics_path, index=False, encoding="utf-8")

    pivot = metrics_out.pivot_table(
        index=["target", "model"],
        columns="variant",
        values=["rmse", "mape", "r2", "mae"],
        aggfunc="first",
    )
    pivot_path = cfg.output_dir / "log_transform_metrics_pivot.csv"
    pivot.to_csv(pivot_path, encoding="utf-8")

    compare_rows = []
    for target in ["energy", "co2"]:
        for model in ["stirpat_only", "hybrid"]:
            base = metrics_out[
                (metrics_out["variant"] == "partial_log")
                & (metrics_out["target"] == target)
                & (metrics_out["model"] == model)
            ].iloc[0]
            full = metrics_out[
                (metrics_out["variant"] == "full_log")
                & (metrics_out["target"] == target)
                & (metrics_out["model"] == model)
            ].iloc[0]
            compare_rows.append(
                {
                    "target": target,
                    "model": model,
                    "partial_rmse": float(base["rmse"]),
                    "full_rmse": float(full["rmse"]),
                    "rmse_change_full_minus_partial": float(full["rmse"] - base["rmse"]),
                    "partial_mape": float(base["mape"]),
                    "full_mape": float(full["mape"]),
                    "mape_change_full_minus_partial": float(full["mape"] - base["mape"]),
                    "partial_r2": float(base["r2"]),
                    "full_r2": float(full["r2"]),
                    "r2_change_full_minus_partial": float(full["r2"] - base["r2"]),
                }
            )

    compare_df = pd.DataFrame(compare_rows)
    compare_path = cfg.output_dir / "log_transform_comparison.csv"
    compare_df.to_csv(compare_path, index=False, encoding="utf-8")

    summary_path = cfg.output_dir / "log_transform_summary.json"
    payload = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
        "variants": variants,
        "detail_files": all_detail_paths,
        "meta": all_meta,
        "comparison": compare_rows,
        "note": (
            "This validation keeps the same Energy->CO2 pipeline and only changes STIRPAT feature log-transform strategy. "
            "CO2 prediction is derived from predicted energy using observed CO2/Energy ratio in validation years."
        ),
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Log-transform validation finished ===")
    print(metrics_out.to_string(index=False))
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved comparison: {compare_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
