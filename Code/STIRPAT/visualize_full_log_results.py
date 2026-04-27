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

import matplotlib.pyplot as plt
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
class Config:
    input_csv: Path
    out_dir: Path
    train_end_year: int = 2020
    valid_start_year: int = 2021
    valid_end_year: int = 2023
    future_start_year: int = 2024
    future_end_year: int = 2035
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
    recent_growth_years: int = 5
    provinces_per_figure: int = 9


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
    arr = np.log(pd.to_numeric(series, errors="coerce").clip(lower=eps))
    return pd.Series(arr, index=series.index)


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
    df["CO2_per_Energy"] = df["CO2"] / np.clip(df["Energy"], 1e-8, None)

    df["log_GDP"] = safe_log(df["GDP"])
    df["log_Population"] = safe_log(df["Population"])
    df["log_Energy"] = safe_log(df["Energy"])
    df["log_pGDP"] = safe_log(df["pGDP"])
    df["log_EnergyIntensity"] = safe_log(df["EnergyIntensity"])
    df["log_Industry"] = safe_log(df["Industry"])
    df["log_Urbanization"] = safe_log(df["Urbanization"])

    levels = sorted(df["province"].unique().tolist())
    mapper = {p: i for i, p in enumerate(levels)}
    df["province_id"] = df["province"].map(mapper).astype(int)

    return df.sort_values(["province", "year"], kind="stable").reset_index(drop=True)


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


def build_samples(
    panel_df: pd.DataFrame,
    cfg: Config,
) -> Dict[str, Any]:
    stirpat_cols = [
        "log_Population",
        "log_pGDP",
        "log_Industry",
        "log_Urbanization",
        "log_EnergyIntensity",
    ]

    train_mask = panel_df["year"] <= cfg.train_end_year
    ridge = PanelRidgeSTIRPAT(alpha=cfg.ridge_alpha)
    ridge.fit(
        df=panel_df.loc[train_mask].copy(),
        feature_cols=stirpat_cols,
        target_col="log_Energy",
        province_col="province",
    )

    df = panel_df.copy()
    df["stirpat_log_pred"] = ridge.predict(df, province_col="province")
    df["residual"] = df["log_Energy"] - df["stirpat_log_pred"]

    gru_cols = [
        "log_GDP",
        "log_pGDP",
        "log_Population",
        "Energy",
        "EnergyIntensity",
        "Industry",
        "Urbanization",
    ]
    ref = df.loc[train_mask, gru_cols].to_numpy(dtype=np.float32)
    mean = ref.mean(axis=0)
    std = ref.std(axis=0)
    std[std < 1e-8] = 1.0

    df_std = df.copy()
    df_std[gru_cols] = (df_std[gru_cols].to_numpy(dtype=np.float32) - mean) / std

    records: List[Dict[str, Any]] = []
    for _, grp in df_std.groupby("province", sort=False):
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
                    "true_co2": float(np.asarray(df.loc[int(row["index"]), "CO2"], dtype=np.float64)),
                    "co2_per_energy": float(np.asarray(df.loc[int(row["index"]), "CO2_per_Energy"], dtype=np.float64)),
                }
            )

    if not records:
        raise RuntimeError("No sliding-window samples generated.")

    sample_df = pd.DataFrame(records)

    def _pack(part: pd.DataFrame) -> Dict[str, np.ndarray]:
        return {
            "dynamic_x": np.stack(part["dynamic_x"].to_list(), axis=0).astype(np.float32),
            "lag_residual_x": np.stack(part["lag_residual_x"].to_list(), axis=0).astype(np.float32),
            "province_idx": part["province_id"].to_numpy(dtype=np.int64),
            "target_residual": part["target_residual"].to_numpy(dtype=np.float32),
        }

    train_part = sample_df[sample_df["year"] <= cfg.train_end_year].copy()
    valid_part = sample_df[
        (sample_df["year"] >= cfg.valid_start_year) & (sample_df["year"] <= cfg.valid_end_year)
    ].copy()
    if train_part.empty or valid_part.empty:
        raise RuntimeError("Train/valid split is empty under current year settings.")

    return {
        "ridge": ridge,
        "df_with_residual": df,
        "sample_df": sample_df,
        "train_np": _pack(train_part),
        "valid_np": _pack(valid_part),
        "train_rows": train_part,
        "valid_rows": valid_part,
        "num_provinces": int(df["province_id"].nunique()),
        "num_dynamic_features": len(gru_cols),
        "gru_cols": gru_cols,
        "standardizer_mean": mean,
        "standardizer_std": std,
    }


def train_full_log_model(pack: Dict[str, Any], cfg: Config) -> Tuple[EntityEmbeddingGRU, Dict[str, Any]]:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_np = pack["train_np"]
    valid_np = pack["valid_np"]

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

    model = EntityEmbeddingGRU(
        num_provinces=int(pack["num_provinces"]),
        num_dynamic_features=int(pack["num_dynamic_features"]),
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
    history: List[Dict[str, float]] = []

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
        history.append({"epoch": float(epoch), "train_loss": float(train_loss), "valid_loss": float(valid_loss)})

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
    return model, {
        "best_epoch": int(best_epoch),
        "best_valid_loss": float(best_valid_loss),
        "history": history,
    }


def build_historical_prediction_detail(
    model: EntityEmbeddingGRU,
    pack: Dict[str, Any],
    cfg: Config,
) -> pd.DataFrame:
    device = next(model.parameters()).device
    sample_df = pack["sample_df"].copy().reset_index(drop=True)

    dynamic_x = np.stack(sample_df["dynamic_x"].to_list(), axis=0).astype(np.float32)
    lag_residual_x = np.stack(sample_df["lag_residual_x"].to_list(), axis=0).astype(np.float32)
    province_idx = sample_df["province_id"].to_numpy(dtype=np.int64)
    pred_res = predict_residuals(
        model=model,
        dynamic_x=dynamic_x,
        lag_residual_x=lag_residual_x,
        province_idx=province_idx,
        device=device,
        batch_size=cfg.batch_size,
    )

    out = sample_df[["province", "province_id", "year", "true_energy", "true_co2", "target_stirpat_log", "co2_per_energy"]].copy()
    out["pred_residual"] = pred_res
    out["stirpat_energy_pred"] = np.exp(out["target_stirpat_log"].to_numpy(dtype=float))
    out["hybrid_energy_pred"] = np.exp(out["target_stirpat_log"].to_numpy(dtype=float) + pred_res)
    out["stirpat_co2_pred"] = out["stirpat_energy_pred"] * out["co2_per_energy"]
    out["hybrid_co2_pred"] = out["hybrid_energy_pred"] * out["co2_per_energy"]

    def _split(y: int) -> str:
        if y <= cfg.train_end_year:
            return "train"
        if cfg.valid_start_year <= y <= cfg.valid_end_year:
            return "valid"
        return "other"

    out["split"] = out["year"].astype(int).map(_split)
    return out.sort_values(["province", "year"], kind="stable").reset_index(drop=True)


def _avg_growth(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2:
        return 0.0
    growth = s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if growth.empty:
        return 0.0
    return float(np.clip(growth.mean(), -0.05, 0.08))


def _avg_delta(series: pd.Series, lower: float, upper: float) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2:
        return 0.0
    delta = s.diff().dropna()
    if delta.empty:
        return 0.0
    return float(np.clip(delta.mean(), lower, upper))


def forecast_future(
    model: EntityEmbeddingGRU,
    pack: Dict[str, Any],
    cfg: Config,
) -> pd.DataFrame:
    device = next(model.parameters()).device
    ridge: PanelRidgeSTIRPAT = pack["ridge"]
    base_df = pack["df_with_residual"].copy()
    mean = pack["standardizer_mean"]
    std = pack["standardizer_std"]
    gru_cols: List[str] = pack["gru_cols"]

    future_records: List[Dict[str, Any]] = []
    residual_lower = float(np.log(0.8))
    residual_upper = float(np.log(1.2))
    history_end = cfg.future_start_year - 1

    for province, grp in base_df.groupby("province", sort=False):
        g = grp.sort_values("year", kind="stable").copy()
        g = g[g["year"] <= history_end].copy().reset_index(drop=True)
        if g.empty:
            continue
        if len(g) < max(cfg.window + 1, cfg.recent_growth_years):
            continue

        pid = int(g["province_id"].iloc[-1])
        recent = g.tail(max(int(cfg.recent_growth_years), 2)).copy()

        pop_g = _avg_growth(recent["Population"])
        pgdp_g = _avg_growth(recent["pGDP"])
        ei_g = _avg_growth(recent["EnergyIntensity"])
        ind_d = _avg_delta(recent["Industry"], -1.5, 1.5)
        urb_d = _avg_delta(recent["Urbanization"], -0.5, 1.2)

        co2_intensity = float(np.clip(recent["CO2_per_Energy"].mean(), 1e-8, None))

        for year in range(cfg.future_start_year, cfg.future_end_year + 1):
            prev = g.iloc[-1]

            population = float(prev["Population"]) * (1.0 + pop_g)
            pgdp = float(prev["pGDP"]) * (1.0 + pgdp_g)
            gdp = population * pgdp
            energy_intensity = max(float(prev["EnergyIntensity"]) * (1.0 + ei_g), 1e-8)
            industry = float(np.clip(float(prev["Industry"]) + ind_d, 1.0, 99.0))
            urbanization = float(np.clip(float(prev["Urbanization"]) + urb_d, 1.0, 99.5))

            stirpat_input = pd.DataFrame(
                {
                    "province": [province],
                    "log_Population": [np.log(max(population, 1e-8))],
                    "log_pGDP": [np.log(max(pgdp, 1e-8))],
                    "log_Industry": [np.log(max(industry, 1e-8))],
                    "log_Urbanization": [np.log(max(urbanization, 1e-8))],
                    "log_EnergyIntensity": [np.log(max(energy_intensity, 1e-8))],
                }
            )
            stirpat_log_pred = float(ridge.predict(stirpat_input, province_col="province")[0])

            win = g.tail(cfg.window).copy()
            x_dyn = win[gru_cols].to_numpy(dtype=np.float32)
            x_dyn = (x_dyn - mean) / std
            x_lag = win[["residual"]].to_numpy(dtype=np.float32)

            x_dyn_t = torch.tensor(x_dyn[None, :, :], dtype=torch.float32, device=device)
            x_lag_t = torch.tensor(x_lag[None, :, :], dtype=torch.float32, device=device)
            x_pid_t = torch.tensor([pid], dtype=torch.long, device=device)
            with torch.no_grad():
                pred_res = float(model(x_dyn_t, x_lag_t, x_pid_t).detach().cpu().numpy()[0])
            pred_res = float(np.clip(pred_res, residual_lower, residual_upper))

            hybrid_log = stirpat_log_pred + pred_res
            energy_pred = float(np.exp(hybrid_log))
            co2_pred = float(energy_pred * co2_intensity)

            row = {
                "province": province,
                "province_id": pid,
                "year": int(year),
                "Population": population,
                "pGDP": pgdp,
                "GDP": gdp,
                "EnergyIntensity": energy_intensity,
                "Industry": industry,
                "Urbanization": urbanization,
                "Energy": energy_pred,
                "stirpat_log_pred": stirpat_log_pred,
                "pred_residual": pred_res,
                "hybrid_log_energy_pred": hybrid_log,
                "energy_pred": energy_pred,
                "co2_pred": co2_pred,
                "co2_per_energy_assumed": co2_intensity,
                "log_GDP": np.log(max(gdp, 1e-8)),
                "log_pGDP": np.log(max(pgdp, 1e-8)),
                "log_Population": np.log(max(population, 1e-8)),
                "residual": pred_res,
            }
            future_records.append(row)

            g = pd.concat([g, pd.DataFrame([row])], ignore_index=True)

    if not future_records:
        return pd.DataFrame(columns=["province", "province_id", "year", "energy_pred", "co2_pred"])
    return pd.DataFrame(future_records).sort_values(["province", "year"], kind="stable").reset_index(drop=True)


def plot_national_historical_compare(detail: pd.DataFrame, out_file: Path) -> None:
    nat = (
        detail.groupby("year", as_index=False)[["true_co2", "stirpat_co2_pred", "hybrid_co2_pred"]]
        .sum()
        .sort_values("year", kind="stable")
    )
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.plot(nat["year"], nat["true_co2"], color="#111111", linewidth=2.6, label="Actual CO2")
    ax.plot(nat["year"], nat["stirpat_co2_pred"], color="#1f77b4", linestyle="--", linewidth=2.0, label="STIRPAT (full-log)")
    ax.plot(nat["year"], nat["hybrid_co2_pred"], color="#2ca02c", linewidth=2.0, label="Full-log STIRPAT + GRU")
    ax.axvspan(2021, 2023, color="#f2c94c", alpha=0.18, label="Validation")
    ax.set_title("National CO2: Actual vs Predicted (Historical)")
    ax.set_xlabel("Year")
    ax.set_ylabel("CO2")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(str(out_file), dpi=300)
    plt.close(fig)


def plot_province_historical_compare(detail: pd.DataFrame, out_dir: Path, provinces_per_figure: int) -> List[Path]:
    provinces = sorted(detail["province"].astype(str).unique().tolist())
    files: List[Path] = []
    if not provinces:
        return files

    chunk = max(int(provinces_per_figure), 1)
    for i in range(0, len(provinces), chunk):
        group = provinces[i : i + chunk]
        n = len(group)
        ncols = 3
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.8 * ncols, 3.4 * nrows), sharex=True)
        axes = np.array(axes).reshape(-1)

        for k, p in enumerate(group):
            ax = axes[k]
            g = detail[detail["province"] == p].sort_values("year", kind="stable")
            ax.plot(g["year"], g["true_co2"], color="#111111", linewidth=1.9, label="Actual")
            ax.plot(g["year"], g["hybrid_co2_pred"], color="#2ca02c", linewidth=1.7, label="Hybrid")
            ax.plot(g["year"], g["stirpat_co2_pred"], color="#1f77b4", linestyle="--", linewidth=1.5, label="STIRPAT")
            ax.set_title(p)
            ax.grid(alpha=0.2, linestyle="--")

        for j in range(n, len(axes)):
            axes[j].axis("off")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
        fig.supxlabel("Year")
        fig.supylabel("CO2")
        fig.tight_layout(rect=(0, 0, 1, 0.95))

        out_file = out_dir / f"province_historical_compare_{i // chunk + 1:02d}.png"
        fig.savefig(str(out_file), dpi=300)
        plt.close(fig)
        files.append(out_file)
    return files


def plot_future_national(history_panel: pd.DataFrame, future_df: pd.DataFrame, out_file: Path) -> None:
    hist_nat = history_panel.groupby("year", as_index=False).agg(CO2=("CO2", "sum"))
    hist_nat = hist_nat.sort_values(by="year")
    fut_nat = future_df.groupby("year", as_index=False).agg(co2_pred=("co2_pred", "sum"))
    fut_nat = fut_nat.sort_values(by="year")

    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.plot(hist_nat["year"], hist_nat["CO2"], color="#111111", linewidth=2.4, label="Historical actual")
    if not fut_nat.empty:
        ax.plot(fut_nat["year"], fut_nat["co2_pred"], color="#d62728", linewidth=2.2, marker="o", markersize=4, label="Future forecast (full-log hybrid)")
        if not hist_nat.empty:
            ax.plot([hist_nat["year"].iloc[-1], fut_nat["year"].iloc[0]], [hist_nat["CO2"].iloc[-1], fut_nat["co2_pred"].iloc[0]], color="#d62728", linestyle="--", linewidth=1.4, alpha=0.8)
    ax.set_title("National CO2: Historical and Future Forecast")
    ax.set_xlabel("Year")
    ax.set_ylabel("CO2")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(str(out_file), dpi=300)
    plt.close(fig)


def plot_future_province_top(future_df: pd.DataFrame, out_file: Path, top_n: int = 12) -> None:
    if future_df.empty:
        return
    total = future_df.groupby("province", as_index=False).agg(co2_pred=("co2_pred", "sum"))
    total = total.sort_values(by="co2_pred", ascending=False)
    top = total.head(top_n)["province"].tolist()
    plot_df = future_df[future_df["province"].isin(top)].copy()

    n = len(top)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.8 * ncols, 3.4 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1)

    for i, p in enumerate(top):
        ax = axes[i]
        g = plot_df[plot_df["province"] == p].sort_values("year", kind="stable")
        ax.plot(g["year"], g["co2_pred"], color="#d62728", linewidth=2.0)
        ax.set_title(p)
        ax.grid(alpha=0.2, linestyle="--")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Province Future CO2 Forecast (Top Emitters)", y=0.995)
    fig.supxlabel("Year")
    fig.supylabel("Predicted CO2")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(str(out_file), dpi=300)
    plt.close(fig)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Visualize full-log STIRPAT + GRU training/backtest and future forecast.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=SCRIPT_DIR.parent / "Preprocess" / "output" / "panel_master.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=SCRIPT_DIR / "output" / "robustness" / "full_log_visualization",
    )
    parser.add_argument("--train-end-year", type=int, default=2020)
    parser.add_argument("--valid-start-year", type=int, default=2021)
    parser.add_argument("--valid-end-year", type=int, default=2023)
    parser.add_argument("--future-start-year", type=int, default=2024)
    parser.add_argument("--future-end-year", type=int, default=2035)
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
    parser.add_argument("--recent-growth-years", type=int, default=5)
    parser.add_argument("--provinces-per-figure", type=int, default=9)
    args = parser.parse_args()

    return Config(
        input_csv=args.input_csv,
        out_dir=args.out_dir,
        train_end_year=args.train_end_year,
        valid_start_year=args.valid_start_year,
        valid_end_year=args.valid_end_year,
        future_start_year=args.future_start_year,
        future_end_year=args.future_end_year,
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
        recent_growth_years=args.recent_growth_years,
        provinces_per_figure=args.provinces_per_figure,
    )


def main() -> None:
    cfg = parse_args()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    panel = prepare_panel(cfg.input_csv)
    pack = build_samples(panel, cfg)
    model, train_meta = train_full_log_model(pack, cfg)
    hist_detail = build_historical_prediction_detail(model, pack, cfg)
    future_df = forecast_future(model, pack, cfg)

    hist_detail_path = cfg.out_dir / "full_log_historical_detail.csv"
    hist_detail.to_csv(hist_detail_path, index=False, encoding="utf-8")

    hist_national = (
        hist_detail.groupby(["year", "split"], as_index=False)[["true_co2", "stirpat_co2_pred", "hybrid_co2_pred"]]
        .sum()
        .sort_values("year", kind="stable")
    )
    hist_national_path = cfg.out_dir / "full_log_historical_national.csv"
    hist_national.to_csv(hist_national_path, index=False, encoding="utf-8")

    future_path = cfg.out_dir / "full_log_future_detail.csv"
    future_df.to_csv(future_path, index=False, encoding="utf-8")

    future_national = (
        future_df.groupby("year", as_index=False)[["energy_pred", "co2_pred"]]
        .sum()
        .sort_values("year", kind="stable")
    )
    future_national_path = cfg.out_dir / "full_log_future_national.csv"
    future_national.to_csv(future_national_path, index=False, encoding="utf-8")

    fig_hist_nat = cfg.out_dir / "fig_full_log_national_historical_compare.png"
    fig_future_nat = cfg.out_dir / "fig_full_log_national_future_forecast.png"
    fig_future_prov = cfg.out_dir / "fig_full_log_province_future_top12.png"

    plot_national_historical_compare(hist_detail, fig_hist_nat)
    province_figs = plot_province_historical_compare(hist_detail, cfg.out_dir, cfg.provinces_per_figure)
    plot_future_national(panel, future_df, fig_future_nat)
    plot_future_province_top(future_df, fig_future_prov, top_n=12)

    summary = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
        "train_meta": train_meta,
        "outputs": {
            "historical_detail": str(hist_detail_path),
            "historical_national": str(hist_national_path),
            "future_detail": str(future_path),
            "future_national": str(future_national_path),
            "fig_national_historical": str(fig_hist_nat),
            "fig_national_future": str(fig_future_nat),
            "fig_province_future_top12": str(fig_future_prov),
            "fig_province_historical_grouped": [str(x) for x in province_figs],
        },
        "note": (
            "Historical CO2 predictions use observed CO2/Energy ratio at each year. "
            "Future CO2 predictions use province-specific recent average CO2/Energy ratio and recursively forecasted Energy."
        ),
    }
    summary_path = cfg.out_dir / "full_log_visualization_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Full-log visualization package generated ===")
    print(f"Historical detail: {hist_detail_path}")
    print(f"Future detail: {future_path}")
    print(f"National historical fig: {fig_hist_nat}")
    print(f"National future fig: {fig_future_nat}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
