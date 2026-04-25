from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stirpat_ee_gru import EntityEmbeddingGRU, PanelRidgeSTIRPAT


@dataclass
class ForecastConfig:
    panel_csv: Path
    panel_with_residual_csv: Path
    dataset_npz: Path
    model_ckpt: Path
    output_dir: Path
    start_year: int = 2024
    end_year: int = 2035
    window: int = 3
    train_end_year: int = 2020
    national_validation_csv: Path = SCRIPT_DIR.parent / "Preprocess" / "output" / "national_energy_validation.csv"
    calibration_year_window: int = 5


IPCC_EF_TCO2_PER_TCE = {
    "coal": 2.66,
    "oil": 2.02,
    "gas": 1.62,
}


def _phase_for_year(year: int) -> str:
    if 2024 <= year <= 2025:
        return "2024_2025"
    if 2026 <= year <= 2030:
        return "2026_2030"
    if 2031 <= year <= 2035:
        return "2031_2035"
    raise ValueError(f"Year out of policy range: {year}")


def build_scenario_policy() -> Dict[str, Dict[str, Dict[str, float]]]:
    baseline = {
        "Population": {"2024_2025": 0.0005, "2026_2030": 0.0, "2031_2035": -0.0005},
        "pGDP": {"2024_2025": 0.05, "2026_2030": 0.045, "2031_2035": 0.04},
        "Energy": {"2024_2025": 0.025, "2026_2030": 0.018, "2031_2035": 0.01},
        "CoalShare": {"2024_2025": -1.5, "2026_2030": -1.0, "2031_2035": -0.8},
        "OilShare": {"2024_2025": 0.05, "2026_2030": 0.03, "2031_2035": 0.0},
        "GasShare": {"2024_2025": 0.10, "2026_2030": 0.08, "2031_2035": 0.05},
        "Industry": {"2024_2025": -1.2, "2026_2030": -0.8, "2031_2035": -0.5},
        "Urbanization": {"2024_2025": 0.75, "2026_2030": 0.65, "2031_2035": 0.5},
        "CarbonIntensity": {"2024_2025": -0.03, "2026_2030": -0.025, "2031_2035": -0.02},
        "PrivateCars": {"2024_2025": 0.05, "2026_2030": 0.035, "2031_2035": 0.02},
    }

    def with_adjustments(
        energy_adj: float,
        coal_pp_adj: float,
        oil_pp_adj: float,
        gas_pp_adj: float,
        industry_pp_adj: float,
        ci_adj: float,
    ) -> Dict[str, Dict[str, float]]:
        out = {k: dict(v) for k, v in baseline.items()}
        for phase in ["2024_2025", "2026_2030", "2031_2035"]:
            out["Energy"][phase] = out["Energy"][phase] + energy_adj
            out["CoalShare"][phase] = out["CoalShare"][phase] + coal_pp_adj
            out["OilShare"][phase] = out["OilShare"][phase] + oil_pp_adj
            out["GasShare"][phase] = out["GasShare"][phase] + gas_pp_adj
            out["Industry"][phase] = out["Industry"][phase] + industry_pp_adj
            out["CarbonIntensity"][phase] = out["CarbonIntensity"][phase] + ci_adj
        return out

    return {
        "baseline": baseline,
        "low_carbon": with_adjustments(
            energy_adj=-0.003,
            coal_pp_adj=-0.5,
            oil_pp_adj=-0.1,
            gas_pp_adj=0.1,
            industry_pp_adj=-0.3,
            ci_adj=-0.005,
        ),
        "extensive": with_adjustments(
            energy_adj=0.003,
            coal_pp_adj=0.3,
            oil_pp_adj=0.08,
            gas_pp_adj=-0.05,
            industry_pp_adj=0.2,
            ci_adj=0.005,
        ),
    }


def _safe_log(x: float) -> float:
    return float(np.log(max(float(x), 1e-8)))


def _clip_pct(x: float) -> float:
    return float(np.clip(x, 0.0, 100.0))


def _normalize_energy_shares(coal: float, oil: float, gas: float) -> Tuple[float, float, float, float]:
    coal = _clip_pct(coal)
    oil = _clip_pct(oil)
    gas = _clip_pct(gas)
    fossil_sum = coal + oil + gas
    if fossil_sum > 100.0:
        scale = 100.0 / max(fossil_sum, 1e-8)
        coal *= scale
        oil *= scale
        gas *= scale
        nonfossil = 0.0
    else:
        nonfossil = 100.0 - fossil_sum
    return float(coal), float(oil), float(gas), float(nonfossil)


def _co2_from_ipcc(energy: float, coal_share: float, oil_share: float, gas_share: float) -> float:
    intensity = (
        coal_share * IPCC_EF_TCO2_PER_TCE["coal"]
        + oil_share * IPCC_EF_TCO2_PER_TCE["oil"]
        + gas_share * IPCC_EF_TCO2_PER_TCE["gas"]
    ) / 100.0
    return float(max(energy, 0.0) * intensity)


def load_model(model_ckpt: Path, device: torch.device) -> Tuple[EntityEmbeddingGRU, Dict[str, int]]:
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
    return model, shape_info


def estimate_highway_growth(panel_df: pd.DataFrame) -> Dict[str, float]:
    rates: Dict[str, float] = {}
    for province, grp in panel_df.groupby("province", sort=False):
        g = grp.sort_values("year", kind="stable")
        recent = g[g["year"].between(2018, 2023)].copy()
        if len(recent) < 2:
            rates[province] = 0.0
            continue

        vals = recent["HighwayMileage"].to_numpy(dtype=float)
        yoy: List[float] = []
        for i in range(1, len(vals)):
            prev = vals[i - 1]
            curr = vals[i]
            if prev > 1e-8:
                yoy.append(curr / prev - 1.0)

        if not yoy:
            rates[province] = 0.0
            continue

        median_rate = float(np.median(yoy))
        rates[province] = float(np.clip(median_rate, -0.05, 0.10))
    return rates


def prepare_base_data(cfg: ForecastConfig) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, List[str], Dict[str, int]]:
    panel = pd.read_csv(cfg.panel_csv)
    panel["province"] = panel["province"].astype(str)
    panel["year"] = panel["year"].astype(int)
    panel = panel.sort_values(["province", "year"], kind="stable").reset_index(drop=True)

    panel_res = pd.read_csv(cfg.panel_with_residual_csv)
    panel_res["province"] = panel_res["province"].astype(str)
    panel_res["year"] = panel_res["year"].astype(int)
    panel_res = panel_res.sort_values(["province", "year"], kind="stable").reset_index(drop=True)

    npz = np.load(cfg.dataset_npz, allow_pickle=True)
    mean = npz["standardizer_mean"].astype(np.float32)
    std = npz["standardizer_std"].astype(np.float32)
    gru_cols = [str(x) for x in npz["gru_feature_names"].tolist()]

    provinces = sorted(panel["province"].unique().tolist())
    province_to_id = {p: i for i, p in enumerate(provinces)}

    return panel, panel_res, mean, std, gru_cols, province_to_id


def fit_stirpat(panel_df: pd.DataFrame, train_end_year: int) -> PanelRidgeSTIRPAT:
    stirpat_cols = [
        "log_Population",
        "log_pGDP",
        "Industry",
        "Urbanization",
        "CoalShare",
        "log_CarbonIntensity",
        "log_PrivateCars",
    ]

    df = panel_df.copy()
    df["pGDP"] = df["GDP"] / df["Population"]
    df["CarbonIntensity"] = df["CO2"] / df["GDP"]
    for col in ["CO2", "GDP", "Population", "Energy", "PrivateCars", "pGDP", "CarbonIntensity"]:
        df[f"log_{col}"] = np.log(df[col].clip(lower=1e-8))

    ridge = PanelRidgeSTIRPAT(alpha=1.0)
    ridge.fit(
        df=df[df["year"] <= train_end_year],
        feature_cols=stirpat_cols,
        target_col="log_Energy",
        province_col="province",
    )
    return ridge


def _build_row_for_next_year(prev_row: pd.Series, policy: Dict[str, Dict[str, float]], year: int, highway_rate: float) -> Dict[str, float]:
    phase = _phase_for_year(year)

    population = float(prev_row["Population"]) * (1.0 + policy["Population"][phase])
    pgdp = float(prev_row["pGDP"]) * (1.0 + policy["pGDP"][phase])
    gdp = population * pgdp
    energy = float(prev_row["Energy"]) * (1.0 + policy["Energy"][phase])
    carbon_intensity = float(prev_row["CarbonIntensity"]) * (1.0 + policy["CarbonIntensity"][phase])
    private_cars = float(prev_row["PrivateCars"]) * (1.0 + policy["PrivateCars"][phase])

    coal_share = _clip_pct(float(prev_row["CoalShare"]) + policy["CoalShare"][phase])
    oil_share = _clip_pct(float(prev_row["OilShare"]) + policy["OilShare"][phase])
    gas_share = _clip_pct(float(prev_row["GasShare"]) + policy["GasShare"][phase])
    coal_share, oil_share, gas_share, nonfossil_share = _normalize_energy_shares(coal_share, oil_share, gas_share)
    industry = _clip_pct(float(prev_row["Industry"]) + policy["Industry"][phase])
    urbanization = _clip_pct(float(prev_row["Urbanization"]) + policy["Urbanization"][phase])
    highway = max(float(prev_row["HighwayMileage"]) * (1.0 + highway_rate), 1e-8)

    return {
        "year": int(year),
        "Population": population,
        "pGDP": pgdp,
        "GDP": gdp,
        "Energy": energy,
        "CoalShare": coal_share,
        "OilShare": oil_share,
        "GasShare": gas_share,
        "NonFossilShare": nonfossil_share,
        "Industry": industry,
        "Urbanization": urbanization,
        "CarbonIntensity": carbon_intensity,
        "PrivateCars": private_cars,
        "HighwayMileage": highway,
        "log_Population": _safe_log(population),
        "log_pGDP": _safe_log(pgdp),
        "log_GDP": _safe_log(gdp),
        "log_Energy": _safe_log(energy),
        "log_CarbonIntensity": _safe_log(carbon_intensity),
        "log_PrivateCars": _safe_log(private_cars),
    }


def _predict_residual_one_step(
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


def forecast_scenario(
    scenario_name: str,
    policy: Dict[str, Dict[str, float]],
    panel_df: pd.DataFrame,
    panel_res_df: pd.DataFrame,
    ridge: PanelRidgeSTIRPAT,
    model: EntityEmbeddingGRU,
    province_to_id: Dict[str, int],
    highway_growth: Dict[str, float],
    mean: np.ndarray,
    std: np.ndarray,
    gru_cols: List[str],
    cfg: ForecastConfig,
    device: torch.device,
) -> pd.DataFrame:
    records: List[Dict[str, float]] = []

    # Seed residual recursion with historical true residuals through 2023.
    hist = panel_res_df[panel_res_df["year"] <= (cfg.start_year - 1)].copy()
    hist = hist[[
        "province",
        "year",
        "Population",
        "pGDP",
        "GDP",
        "Energy",
        "CoalShare",
        "OilShare",
        "GasShare",
        "NonFossilShare",
        "Industry",
        "Urbanization",
        "CarbonIntensity",
        "PrivateCars",
        "HighwayMileage",
        "log_Population",
        "log_pGDP",
        "log_GDP",
        "log_Energy",
        "log_CarbonIntensity",
        "log_PrivateCars",
        "residual",
    ]].copy()

    for province, grp in hist.groupby("province", sort=False):
        g = grp.sort_values("year", kind="stable").copy()
        pid = province_to_id[province]
        hwy_rate = highway_growth.get(province, 0.0)

        for year in range(cfg.start_year, cfg.end_year + 1):
            prev = g.iloc[-1]
            row = _build_row_for_next_year(prev, policy, year, hwy_rate)

            stirpat_input = pd.DataFrame(
                {
                    "province": [province],
                    "log_Population": [row["log_Population"]],
                    "log_pGDP": [row["log_pGDP"]],
                    "Industry": [row["Industry"]],
                    "Urbanization": [row["Urbanization"]],
                    "CoalShare": [row["CoalShare"]],
                    "log_CarbonIntensity": [row["log_CarbonIntensity"]],
                    "log_PrivateCars": [row["log_PrivateCars"]],
                }
            )
            stirpat_log_energy = float(ridge.predict(stirpat_input, province_col="province")[0])

            win = g[g["year"].between(year - cfg.window, year - 1)].copy()
            if len(win) != cfg.window:
                raise RuntimeError(
                    f"Province {province} year {year} has insufficient window rows: {len(win)}"
                )
            pred_residual = _predict_residual_one_step(
                model=model,
                window_df=win,
                province_idx=pid,
                gru_cols=gru_cols,
                mean=mean,
                std=std,
                device=device,
            )

            hybrid_log_energy = stirpat_log_energy + pred_residual
            energy_pred = float(np.exp(hybrid_log_energy))
            co2_pred = _co2_from_ipcc(
                energy=energy_pred,
                coal_share=float(row["CoalShare"]),
                oil_share=float(row["OilShare"]),
                gas_share=float(row["GasShare"]),
            )

            out_row: Dict[str, float] = {
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
            row_for_hist["log_Energy"] = _safe_log(energy_pred)
            row_for_hist["province"] = province
            row_for_hist["residual"] = pred_residual
            g = pd.concat([g, pd.DataFrame([row_for_hist])], ignore_index=True)

    out_df = pd.DataFrame(records)
    out_df = out_df.sort_values(["scenario", "province", "year"], kind="stable").reset_index(drop=True)
    return out_df


def summarize_peak(forecast_df: pd.DataFrame, end_year: int) -> pd.DataFrame:
    national = (
        forecast_df.groupby(["scenario", "year"], as_index=False)[["co2_pred", "energy_pred"]]
        .sum()
        .sort_values(["scenario", "year"], kind="stable")
    )

    summary_rows: List[Dict[str, float]] = []
    for scenario, grp in national.groupby("scenario", sort=False):
        g = grp.sort_values("year", kind="stable").reset_index(drop=True)
        peak_idx = int(g["co2_pred"].idxmax())
        peak_row = g.loc[peak_idx]
        peak_year = int(peak_row["year"])
        summary_rows.append(
            {
                "scenario": scenario,
                "peak_year": peak_year,
                "peak_co2": float(peak_row["co2_pred"]),
                "peaked_within_horizon": bool(peak_year < end_year),
                "note": "未在预测期内达峰" if peak_year >= end_year else "已达峰",
            }
        )

    return pd.DataFrame(summary_rows), national


def estimate_energy_calibration_factor(validation_csv: Path, calibration_year_window: int) -> Tuple[float, Dict[str, float]]:
    if not validation_csv.exists():
        return 1.0, {"enabled": False, "reason": "validation file not found"}

    df = pd.read_csv(validation_csv)
    required = {"year", "ratio_actual_over_province_sum"}
    if not required.issubset(set(df.columns)):
        return 1.0, {"enabled": False, "reason": "missing ratio/year columns"}

    ratio = df[["year", "ratio_actual_over_province_sum"]].copy()
    ratio["year"] = pd.to_numeric(ratio["year"], errors="coerce")
    ratio["ratio_actual_over_province_sum"] = pd.to_numeric(ratio["ratio_actual_over_province_sum"], errors="coerce")
    ratio = ratio.dropna(subset=["year", "ratio_actual_over_province_sum"])
    ratio = ratio[(ratio["ratio_actual_over_province_sum"] > 0.0) & np.isfinite(ratio["ratio_actual_over_province_sum"])]
    if ratio.empty:
        return 1.0, {"enabled": False, "reason": "no valid overlap ratio"}

    ratio = ratio.sort_values("year", kind="stable")
    recent = ratio.tail(max(int(calibration_year_window), 1))
    factor = float(recent["ratio_actual_over_province_sum"].mean())
    return factor, {
        "enabled": True,
        "window": int(calibration_year_window),
        "factor": factor,
        "year_min": int(recent["year"].min()),
        "year_max": int(recent["year"].max()),
        "sample_count": int(len(recent)),
    }


def apply_energy_calibration(forecast_df: pd.DataFrame, factor: float) -> pd.DataFrame:
    out = forecast_df.copy()
    out["energy_pred_raw"] = out["energy_pred"]
    out["energy_pred"] = out["energy_pred_raw"] * float(factor)
    out["hybrid_log_energy_pred"] = np.log(np.clip(out["energy_pred"].to_numpy(dtype=float), 1e-8, None))
    out["co2_pred"] = (
        out["energy_pred"]
        * (
            out["CoalShare"] * IPCC_EF_TCO2_PER_TCE["coal"]
            + out["OilShare"] * IPCC_EF_TCO2_PER_TCE["oil"]
            + out["GasShare"] * IPCC_EF_TCO2_PER_TCE["gas"]
        )
        / 100.0
    )
    return out


def summarize_province_peak(forecast_df: pd.DataFrame, end_year: int) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    grouped = forecast_df.groupby(["scenario", "province"], as_index=False)
    for _, grp in grouped:
        g = grp.sort_values("year", kind="stable").reset_index(drop=True)
        peak_local_idx = int(np.argmax(g["co2_pred"].to_numpy(dtype=float)))
        peak_row = g.iloc[peak_local_idx]
        peak_year = int(peak_row["year"])
        rows.append(
            {
                "scenario": str(peak_row["scenario"]),
                "province": str(peak_row["province"]),
                "peak_year": peak_year,
                "peak_co2": float(peak_row["co2_pred"]),
                "peaked_within_horizon": bool(peak_year < end_year),
                "note": "未在预测期内达峰" if peak_year >= end_year else "已达峰",
            }
        )

    return pd.DataFrame(rows).sort_values(["scenario", "province"], kind="stable").reset_index(drop=True)


def build_organized_outputs(
    forecast_df: pd.DataFrame,
    national_df: pd.DataFrame,
    national_peak_df: pd.DataFrame,
    cfg: ForecastConfig,
) -> Dict[str, Path]:
    organized_dir = cfg.output_dir / "organized"
    organized_dir.mkdir(parents=True, exist_ok=True)

    province_detail_df = forecast_df.sort_values(["scenario", "province", "year"], kind="stable").reset_index(drop=True)
    province_peak_df = summarize_province_peak(province_detail_df, end_year=cfg.end_year)

    # Combined table to keep province and national trajectories in one file.
    national_long_df = national_df.copy()
    national_long_df["province"] = "ALL"
    national_long_df["province_id"] = -1
    national_long_df["level"] = "national"
    national_long_df = national_long_df[["level", "scenario", "province", "province_id", "year", "energy_pred", "co2_pred"]]

    province_long_df = province_detail_df[["scenario", "province", "province_id", "year", "energy_pred", "co2_pred"]].copy()
    province_long_df["level"] = "province"
    province_long_df = province_long_df[["level", "scenario", "province", "province_id", "year", "energy_pred", "co2_pred"]]

    combined_long_df = pd.concat([province_long_df, national_long_df], axis=0, ignore_index=True)
    combined_long_df = combined_long_df.sort_values(["level", "scenario", "province", "year"], kind="stable")

    province_detail_path = organized_dir / "01_province_yearly_detail.csv"
    province_peak_path = organized_dir / "02_province_peak_summary.csv"
    national_yearly_path = organized_dir / "03_national_yearly.csv"
    national_peak_path = organized_dir / "04_national_peak_summary.csv"
    combined_path = organized_dir / "00_combined_long.csv"
    manifest_path = organized_dir / "manifest.json"

    province_detail_df.to_csv(province_detail_path, index=False, encoding="utf-8")
    province_peak_df.to_csv(province_peak_path, index=False, encoding="utf-8")
    national_df.to_csv(national_yearly_path, index=False, encoding="utf-8")
    national_peak_df.to_csv(national_peak_path, index=False, encoding="utf-8")
    combined_long_df.to_csv(combined_path, index=False, encoding="utf-8")

    manifest = {
        "description": "Organized scenario forecast package",
        "year_range": [cfg.start_year, cfg.end_year],
        "files": {
            "combined_long": str(combined_path),
            "province_yearly_detail": str(province_detail_path),
            "province_peak_summary": str(province_peak_path),
            "national_yearly": str(national_yearly_path),
            "national_peak_summary": str(national_peak_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "organized_dir": organized_dir,
        "combined_long": combined_path,
        "province_yearly_detail": province_detail_path,
        "province_peak_summary": province_peak_path,
        "national_yearly": national_yearly_path,
        "national_peak_summary": national_peak_path,
        "manifest": manifest_path,
    }


def parse_args() -> ForecastConfig:
    parser = argparse.ArgumentParser(description="Forecast 2024-2035 CO2 under policy-constrained scenarios.")
    base = SCRIPT_DIR
    parser.add_argument(
        "--panel-csv",
        type=Path,
        default=base.parent / "Preprocess" / "output" / "panel_master.csv",
    )
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
    parser.add_argument(
        "--model-ckpt",
        type=Path,
        default=base / "output" / "model" / "best_ee_gru.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base / "output" / "scenario_forecast",
    )
    parser.add_argument("--start-year", type=int, default=2024)
    parser.add_argument("--end-year", type=int, default=2035)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--train-end-year", type=int, default=2020)
    parser.add_argument(
        "--national-validation-csv",
        type=Path,
        default=base.parent / "Preprocess" / "output" / "national_energy_validation.csv",
    )
    parser.add_argument("--calibration-year-window", type=int, default=5)

    args = parser.parse_args()
    return ForecastConfig(
        panel_csv=args.panel_csv,
        panel_with_residual_csv=args.panel_with_residual_csv,
        dataset_npz=args.dataset_npz,
        model_ckpt=args.model_ckpt,
        output_dir=args.output_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        window=args.window,
        train_end_year=args.train_end_year,
        national_validation_csv=args.national_validation_csv,
        calibration_year_window=args.calibration_year_window,
    )


def main() -> None:
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    panel_df, panel_res_df, mean, std, gru_cols, province_to_id = prepare_base_data(cfg)
    ridge = fit_stirpat(panel_df=panel_df, train_end_year=cfg.train_end_year)
    highway_growth = estimate_highway_growth(panel_df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, shape_info = load_model(cfg.model_ckpt, device)

    if shape_info["num_provinces"] != len(province_to_id):
        raise RuntimeError(
            f"Province count mismatch: model={shape_info['num_provinces']} current={len(province_to_id)}"
        )

    policies = build_scenario_policy()
    all_forecasts: List[pd.DataFrame] = []

    for scenario_name, policy in policies.items():
        fdf = forecast_scenario(
            scenario_name=scenario_name,
            policy=policy,
            panel_df=panel_df,
            panel_res_df=panel_res_df,
            ridge=ridge,
            model=model,
            province_to_id=province_to_id,
            highway_growth=highway_growth,
            mean=mean,
            std=std,
            gru_cols=gru_cols,
            cfg=cfg,
            device=device,
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

    forecast_df.to_csv(detail_path, index=False, encoding="utf-8")
    national_df.to_csv(national_path, index=False, encoding="utf-8")
    peak_summary_df.to_csv(peak_path, index=False, encoding="utf-8")

    payload = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
        "energy_calibration": calib_meta,
        "scenarios": peak_summary_df.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    organized = build_organized_outputs(
        forecast_df=forecast_df,
        national_df=national_df,
        national_peak_df=peak_summary_df,
        cfg=cfg,
    )

    print(f"Saved scenario detail: {detail_path}")
    print(f"Saved national summary: {national_path}")
    print(f"Saved peak summary csv: {peak_path}")
    print(f"Saved peak summary json: {json_path}")
    print(f"Saved organized package dir: {organized['organized_dir']}")
    print(f"Saved organized combined table: {organized['combined_long']}")
    print(f"Applied energy calibration factor: {calib_factor:.6f}")
    print("\nPeak results:")
    print(peak_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
