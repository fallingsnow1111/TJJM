from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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
from policy_scenario_forecast import (
    EXCLUDED_PROVINCES,
    ScenarioPeakSpec,
    PathGenerator,
    ConstraintChecker,
    Calibrator,
    _build_row_for_next_year,
    _co2_from_ipcc,
    build_scenario_policy,
    estimate_energy_calibration_factor,
    apply_energy_calibration,
    summarize_peak,
    build_organized_outputs,
    extract_final_policy_paths,
)


@dataclass
class FullLogScenarioConfig:
    panel_csv: Path
    output_dir: Path
    start_year: int = 2024
    end_year: int = 2035
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
    national_validation_csv: Path = SCRIPT_DIR.parent / "Preprocess" / "output" / "national_energy_validation.csv"
    calibration_year_window: int = 5


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


def prepare_panel(panel_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(panel_csv)
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
        "EnergyIntensity",
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

    for col in [
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
        "EnergyIntensity",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["pGDP"] = df["GDP"] / df["Population"]
    df["CarbonIntensity"] = df["CO2"] / df["GDP"]

    df["log_CO2"] = safe_log(df["CO2"])
    df["log_GDP"] = safe_log(df["GDP"])
    df["log_Population"] = safe_log(df["Population"])
    df["log_Energy"] = safe_log(df["Energy"])
    df["log_pGDP"] = safe_log(df["pGDP"])
    df["log_CarbonIntensity"] = safe_log(df["CarbonIntensity"])
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


def train_full_log_models(panel: pd.DataFrame, cfg: FullLogScenarioConfig) -> Dict[str, Any]:
    set_seed(cfg.seed)

    stirpat_cols = [
        "log_Population",
        "log_pGDP",
        "log_Industry",
        "log_Urbanization",
        "log_EnergyIntensity",
    ]
    ridge = PanelRidgeSTIRPAT(alpha=cfg.ridge_alpha)
    ridge.fit(
        df=panel[panel["year"] <= cfg.train_end_year].copy(),
        feature_cols=stirpat_cols,
        target_col="log_Energy",
        province_col="province",
    )

    df = panel.copy()
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
    ref = df[df["year"] <= cfg.train_end_year][gru_cols].to_numpy(dtype=np.float32)
    mean = ref.mean(axis=0)
    std = ref.std(axis=0)
    std[std < 1e-8] = 1.0

    df_std = df.copy()
    df_std[gru_cols] = (df_std[gru_cols].to_numpy(dtype=np.float32) - mean) / std

    rows: List[Dict[str, Any]] = []
    for _, grp in df_std.groupby("province", sort=False):
        g = grp.sort_values("year", kind="stable").reset_index(drop=False)
        years = g["year"].to_numpy(dtype=int)
        for end_idx in range(cfg.window, len(g)):
            start_idx = end_idx - cfg.window
            if np.any(np.diff(years[start_idx : end_idx + 1]) != 1):
                continue
            row = g.iloc[end_idx]
            rows.append(
                {
                    "province": str(row["province"]),
                    "province_id": int(row["province_id"]),
                    "year": int(row["year"]),
                    "dynamic_x": g.loc[start_idx : end_idx - 1, gru_cols].to_numpy(dtype=np.float32),
                    "lag_residual_x": g.loc[start_idx : end_idx - 1, ["residual"]].to_numpy(dtype=np.float32),
                    "target_residual": float(row["residual"]),
                }
            )

    sample_df = pd.DataFrame(rows)
    train_part = sample_df[sample_df["year"] <= cfg.train_end_year].copy()
    valid_part = sample_df[
        (sample_df["year"] >= cfg.valid_start_year) & (sample_df["year"] <= cfg.valid_end_year)
    ].copy()

    if train_part.empty or valid_part.empty:
        raise RuntimeError("Empty train/valid samples for full-log GRU training.")

    def _stack(part: pd.DataFrame) -> Dict[str, np.ndarray]:
        return {
            "dynamic_x": np.stack(part["dynamic_x"].to_list(), axis=0).astype(np.float32),
            "lag_residual_x": np.stack(part["lag_residual_x"].to_list(), axis=0).astype(np.float32),
            "province_idx": part["province_id"].to_numpy(dtype=np.int64),
            "target_residual": part["target_residual"].to_numpy(dtype=np.float32),
        }

    train_np = _stack(train_part)
    valid_np = _stack(valid_part)

    train_ds = ResidualDataset(
        train_np["dynamic_x"],
        train_np["lag_residual_x"],
        train_np["province_idx"],
        train_np["target_residual"],
    )
    valid_ds = ResidualDataset(
        valid_np["dynamic_x"],
        valid_np["lag_residual_x"],
        valid_np["province_idx"],
        valid_np["target_residual"],
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EntityEmbeddingGRU(
        num_provinces=int(df["province_id"].nunique()),
        num_dynamic_features=len(gru_cols),
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

    for epoch in range(1, cfg.epochs + 1):
        _ = run_epoch(model, train_loader, optimizer, criterion, device, cfg.max_grad_norm)
        valid_loss = evaluate_loss(model, valid_loader, criterion, device)
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

    return {
        "ridge": ridge,
        "gru_model": model,
        "panel_with_residual": df,
        "gru_cols": gru_cols,
        "standardizer_mean": mean,
        "standardizer_std": std,
        "device": device,
        "train_meta": {
            "best_epoch": int(best_epoch),
            "best_valid_loss": float(best_valid_loss),
            "train_samples": int(train_np["dynamic_x"].shape[0]),
            "valid_samples": int(valid_np["dynamic_x"].shape[0]),
        },
    }


def predict_residual_one_step(
    model: EntityEmbeddingGRU,
    window_df: pd.DataFrame,
    province_idx: int,
    gru_cols: List[str],
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> float:
    x_dyn = window_df[gru_cols].to_numpy(dtype=np.float32)
    x_dyn = (x_dyn - mean) / std
    x_lag = window_df[["residual"]].to_numpy(dtype=np.float32)

    x_dyn_t = torch.tensor(x_dyn[None, :, :], dtype=torch.float32, device=device)
    x_lag_t = torch.tensor(x_lag[None, :, :], dtype=torch.float32, device=device)
    x_pid_t = torch.tensor([province_idx], dtype=torch.long, device=device)

    with torch.no_grad():
        pred = model(x_dyn_t, x_lag_t, x_pid_t).detach().cpu().numpy()[0]
    return float(pred)


def forecast_scenario_full_log(
    scenario_name: str,
    policy: Dict[str, Dict[str, float]],
    annual_path: Optional[Dict[str, Dict[int, float]]],
    model_pack: Dict[str, Any],
    cfg: FullLogScenarioConfig,
    scenario_rule: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    ridge: PanelRidgeSTIRPAT = model_pack["ridge"]
    model: EntityEmbeddingGRU = model_pack["gru_model"]
    panel_res = model_pack["panel_with_residual"]
    gru_cols: List[str] = model_pack["gru_cols"]
    mean = model_pack["standardizer_mean"]
    std = model_pack["standardizer_std"]
    device: torch.device = model_pack["device"]

    records: List[Dict[str, Any]] = []
    residual_lower = float(np.log(0.8))
    residual_upper = float(np.log(1.2))
    energy_inertia_alpha = 0.7
    scenario_rule = scenario_rule or {}
    require_peak = bool(scenario_rule.get("require_peak", False))
    target_peak_year = int(scenario_rule.get("target_peak_year", cfg.end_year))

    hist = panel_res[panel_res["year"] <= (cfg.start_year - 1)].copy()
    keep_cols = [
        "province",
        "year",
        "province_id",
        "Population",
        "pGDP",
        "GDP",
        "Energy",
        "CoalShare",
        "OilShare",
        "GasShare",
        "NonFossilShare",
        "EnergyIntensity",
        "Industry",
        "Urbanization",
        "CarbonIntensity",
        "log_Population",
        "log_pGDP",
        "log_GDP",
        "log_Energy",
        "log_CarbonIntensity",
        "residual",
    ]
    hist = hist[keep_cols].copy()

    for province, grp in hist.groupby("province", sort=False):
        g = grp.sort_values("year", kind="stable").copy()
        pid = int(g["province_id"].iloc[-1])

        for year in range(cfg.start_year, cfg.end_year + 1):
            prev = g.iloc[-1]
            row = _build_row_for_next_year(
                prev_row=prev,
                policy=policy,
                annual_path=annual_path,
                year=year,
                start_year=cfg.start_year,
            )

            stirpat_input = pd.DataFrame(
                {
                    "province": [province],
                    "log_Population": [row["log_Population"]],
                    "log_pGDP": [row["log_pGDP"]],
                    "log_Industry": [np.log(max(float(row["Industry"]), 1e-8))],
                    "log_Urbanization": [np.log(max(float(row["Urbanization"]), 1e-8))],
                    "log_EnergyIntensity": [np.log(max(float(row["EnergyIntensity"]), 1e-8))],
                }
            )
            stirpat_log_energy = float(ridge.predict(stirpat_input, province_col="province")[0])

            win = g[g["year"].between(year - cfg.window, year - 1)].copy()
            if len(win) != cfg.window:
                raise RuntimeError(f"Province {province} year {year} has insufficient window rows: {len(win)}")

            pred_residual = predict_residual_one_step(
                model=model,
                window_df=win,
                province_idx=pid,
                gru_cols=gru_cols,
                mean=mean,
                std=std,
                device=device,
            )
            pred_residual = float(np.clip(pred_residual, residual_lower, residual_upper))

            hybrid_log_energy = stirpat_log_energy + pred_residual
            energy_model_pred = float(np.exp(hybrid_log_energy))
            prev_energy = float(prev["Energy"])
            energy_pred = float(energy_inertia_alpha * prev_energy + (1.0 - energy_inertia_alpha) * energy_model_pred)

            if require_peak and year <= target_peak_year:
                policy_energy = float(row["Energy"])
                w_policy = 0.88
                energy_pred = float((1.0 - w_policy) * energy_pred + w_policy * policy_energy)

            hybrid_log_energy = float(np.log(max(energy_pred, 1e-8)))
            co2_pred = _co2_from_ipcc(
                energy=energy_pred,
                coal_share=float(row["CoalShare"]),
                oil_share=float(row["OilShare"]),
                gas_share=float(row["GasShare"]),
            )

            out_row: Dict[str, Any] = {
                "scenario": scenario_name,
                "province": province,
                "province_id": pid,
                "year": int(year),
                "stirpat_log_energy_pred": stirpat_log_energy,
                "residual_pred": pred_residual,
                "hybrid_log_energy_pred": hybrid_log_energy,
                "energy_pred": energy_pred,
                "co2_pred": co2_pred,
            }
            out_row.update(row)
            records.append(out_row)

            row_for_hist = dict(row)
            row_for_hist["Energy"] = energy_pred
            row_for_hist["log_Energy"] = float(np.log(max(energy_pred, 1e-8)))
            row_for_hist["province"] = province
            row_for_hist["province_id"] = pid
            row_for_hist["residual"] = pred_residual
            row_for_hist["log_GDP"] = float(np.log(max(float(row_for_hist["GDP"]), 1e-8)))
            row_for_hist["log_pGDP"] = float(np.log(max(float(row_for_hist["pGDP"]), 1e-8)))
            row_for_hist["log_Population"] = float(np.log(max(float(row_for_hist["Population"]), 1e-8)))
            g = pd.concat([g, pd.DataFrame([row_for_hist])], ignore_index=True)

    out = pd.DataFrame(records)
    return out.sort_values(["scenario", "province", "year"], kind="stable").reset_index(drop=True)


def parse_args() -> FullLogScenarioConfig:
    parser = argparse.ArgumentParser(description="Run full-log STIRPAT + GRU scenario forecast with policy constraints.")
    parser.add_argument(
        "--panel-csv",
        type=Path,
        default=SCRIPT_DIR.parent / "Preprocess" / "output" / "panel_master.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "output" / "scenario_forecast_full_log",
    )
    parser.add_argument("--start-year", type=int, default=2024)
    parser.add_argument("--end-year", type=int, default=2035)
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
    parser.add_argument(
        "--national-validation-csv",
        type=Path,
        default=SCRIPT_DIR.parent / "Preprocess" / "output" / "national_energy_validation.csv",
    )
    parser.add_argument("--calibration-year-window", type=int, default=5)
    args = parser.parse_args()

    return FullLogScenarioConfig(
        panel_csv=args.panel_csv,
        output_dir=args.output_dir,
        start_year=args.start_year,
        end_year=args.end_year,
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
        national_validation_csv=args.national_validation_csv,
        calibration_year_window=args.calibration_year_window,
    )


def main() -> None:
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    panel = prepare_panel(cfg.panel_csv)
    model_pack = train_full_log_models(panel, cfg)

    policies = build_scenario_policy()
    path_generator = PathGenerator(start_year=cfg.start_year, end_year=cfg.end_year)
    checker = ConstraintChecker(start_year=cfg.start_year, end_year=cfg.end_year)
    calibrator = Calibrator()

    scenario_specs: Dict[str, ScenarioPeakSpec] = {
        "baseline": ScenarioPeakSpec(
            target_peak_year=2030,
            require_peak=True,
            t_start=2029,
            energy=(3.6, 3.0, 1.2),
            carbon_intensity=(-0.9, -2.4, -2.8),
            coal_share=(-0.4, -1.4, -1.4),
            industry=(-1.2, -0.8),
        ),
        "low_carbon": ScenarioPeakSpec(
            target_peak_year=2029,
            require_peak=True,
            t_start=2028,
            energy=(2.8, 2.3, 0.9),
            carbon_intensity=(-1.4, -3.2, -3.0),
            coal_share=(-0.7, -2.0, -1.7),
            industry=(-1.5, -1.1),
        ),
        "green_growth": ScenarioPeakSpec(
            target_peak_year=2030,
            require_peak=True,
            t_start=2027,
            energy=(3.9, 3.2, 1.4),
            carbon_intensity=(-0.9, -2.5, -2.8),
            coal_share=(-0.4, -1.5, -1.4),
            industry=(-1.3, -1.0),
        ),
        "extensive": ScenarioPeakSpec(
            target_peak_year=2034,
            require_peak=False,
            t_start=2029,
            energy=(4.4, 4.0, 3.4),
            carbon_intensity=(-0.2, -0.1, 0.0),
            coal_share=(0.0, 0.0, 0.0),
            industry=(-0.2, 0.0),
        ),
    }

    all_forecasts: List[pd.DataFrame] = []
    constraint_report: Dict[str, List[str]] = {}
    calibration_meta_by_scenario: Dict[str, Any] = {}

    for scenario_name, policy in policies.items():
        spec = scenario_specs[scenario_name]
        rule = {
            "require_peak": bool(spec.require_peak),
            "target_peak_year": int(spec.target_peak_year),
            "t_start": int(spec.t_start),
        }

        def _simulate_once(paths: Dict[str, Dict[int, float]], t_start_for_run: int) -> pd.DataFrame:
            run_rule = dict(rule)
            run_rule["t_start"] = int(t_start_for_run)
            return forecast_scenario_full_log(
                scenario_name=scenario_name,
                policy=policy,
                annual_path=paths,
                model_pack=model_pack,
                cfg=cfg,
                scenario_rule=run_rule,
            )

        if bool(rule["require_peak"]):
            def _regenerate_paths(
                t_start: int,
                ci_mid_adjust: float,
                coal_mid_adjust: float,
                energy_early_adjust: float,
                energy_mid_adjust: float,
            ) -> Dict[str, Dict[int, float]]:
                return path_generator.generate_paths(
                    policy=policy,
                    spec=spec,
                    t_start_override=t_start,
                    ci_mid_adjust=ci_mid_adjust,
                    coal_mid_adjust=coal_mid_adjust,
                    energy_early_adjust=energy_early_adjust,
                    energy_mid_adjust=energy_mid_adjust,
                )

            calibrated_path, tuned_t_start, calib_meta = calibrator.calibrate(
                scenario_name=scenario_name,
                initial_t_start=int(rule["t_start"]),
                target_peak_year=int(rule["target_peak_year"]),
                regenerate_paths=_regenerate_paths,
                simulate_once=_simulate_once,
                checker=checker,
            )
            calibration_meta_by_scenario[scenario_name] = calib_meta
            calibration_meta_by_scenario[scenario_name]["t_start_final"] = tuned_t_start
            calibration_meta_by_scenario[scenario_name]["final_policy_path"] = extract_final_policy_paths(calibrated_path)
            fdf = _simulate_once(calibrated_path, tuned_t_start)
        else:
            annual_path = path_generator.generate_paths(
                policy=policy,
                spec=spec,
                t_start_override=int(rule["t_start"]),
            )
            calibration_meta_by_scenario[scenario_name] = {
                "iterations": 0,
                "peak_year": None,
                "target_peak_year": int(rule["target_peak_year"]),
                "t_start": int(rule["t_start"]),
                "t_start_final": int(rule["t_start"]),
                "ci_mid_adjust": 0.0,
                "coal_mid_adjust": 0.0,
                "energy_early_adjust": 0.0,
                "energy_mid_adjust": 0.0,
                "violations": [],
                "history": [],
                "final_policy_path": extract_final_policy_paths(annual_path),
            }
            fdf = _simulate_once(annual_path, int(rule["t_start"]))

        national_one = (
            fdf.groupby("year", as_index=False)[["co2_pred", "energy_pred"]]
            .sum()
            .sort_values("year", kind="stable")
        )
        constraint_report[scenario_name] = checker.check_constraints(
            scenario_df=fdf,
            national_df=national_one,
            scenario_name=scenario_name,
            require_peak=bool(rule["require_peak"]),
            peak_year_min=int(rule["target_peak_year"]),
            peak_year_max=int(rule["target_peak_year"]),
        )
        all_forecasts.append(fdf)

    forecast_df = pd.concat(all_forecasts, axis=0, ignore_index=True)
    forecast_df = forecast_df.sort_values(["scenario", "province", "year"], kind="stable")

    calib_factor, calib_meta = estimate_energy_calibration_factor(
        validation_csv=cfg.national_validation_csv,
        calibration_year_window=cfg.calibration_year_window,
    )
    forecast_df = apply_energy_calibration(forecast_df, factor=calib_factor)

    peak_summary_df, national_df = summarize_peak(forecast_df, end_year=cfg.end_year)

    detail_path = cfg.output_dir / "scenario_forecast_detail.csv"
    national_path = cfg.output_dir / "scenario_forecast_national.csv"
    peak_path = cfg.output_dir / "scenario_peak_summary.csv"
    json_path = cfg.output_dir / "scenario_peak_summary.json"
    final_path_table_path = cfg.output_dir / "scenario_final_policy_path_summary.csv"

    forecast_df.to_csv(detail_path, index=False, encoding="utf-8")
    national_df.to_csv(national_path, index=False, encoding="utf-8")
    peak_summary_df.to_csv(peak_path, index=False, encoding="utf-8")

    path_rows: List[Dict[str, Any]] = []
    for scenario_name, meta in calibration_meta_by_scenario.items():
        final_path = meta.get("final_policy_path", {})
        for var, phase_values in final_path.items():
            path_rows.append(
                {
                    "scenario": scenario_name,
                    "variable": var,
                    "2024_2027": phase_values.get("2024_2027"),
                    "2028_2030": phase_values.get("2028_2030"),
                    "2031_2035": phase_values.get("2031_2035"),
                }
            )
    if path_rows:
        pd.DataFrame(path_rows).sort_values(["scenario", "variable"], kind="stable").to_csv(
            final_path_table_path,
            index=False,
            encoding="utf-8",
        )

    payload = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
        "train_meta": model_pack["train_meta"],
        "energy_calibration": calib_meta,
        "constraint_report": constraint_report,
        "calibration_meta": calibration_meta_by_scenario,
        "scenarios": peak_summary_df.to_dict(orient="records"),
        "note": "This run uses full-log STIRPAT features for first-stage Energy modeling.",
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    organized = build_organized_outputs(
        forecast_df=forecast_df,
        national_df=national_df,
        national_peak_df=peak_summary_df,
        cfg=type("TmpCfg", (), {"output_dir": cfg.output_dir, "start_year": cfg.start_year, "end_year": cfg.end_year})(),
    )

    print("=== Full-log scenario forecast finished ===")
    print(f"Saved scenario detail: {detail_path}")
    print(f"Saved national summary: {national_path}")
    print(f"Saved peak summary csv: {peak_path}")
    print(f"Saved peak summary json: {json_path}")
    print(f"Saved final policy path summary: {final_path_table_path}")
    print(f"Saved organized package dir: {organized['organized_dir']}")
    print(f"Applied energy calibration factor: {calib_factor:.6f}")
    print("\nPeak results:")
    print(peak_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

