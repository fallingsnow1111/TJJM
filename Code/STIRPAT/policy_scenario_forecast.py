from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _blend(a: float, b: float, w: float) -> float:
    return float((1.0 - w) * a + w * b)


def _interp(year: int, y0: int, y1: int, v0: float, v1: float) -> float:
    if y1 <= y0:
        return float(v1)
    t = np.clip((year - y0) / float(y1 - y0), 0.0, 1.0)
    return float(v0 + (v1 - v0) * t)


@dataclass
class ScenarioPeakSpec:
    target_peak_year: int
    require_peak: bool
    t_start: int
    energy: Tuple[float, float, float]  # percentages: (2024-2027, 2028-2030, 2031-2035)
    carbon_intensity: Tuple[float, float, float]  # percentages: early/mid/late
    coal_share: Tuple[float, float, float]  # pp per year: early/mid/late
    industry: Tuple[float, float]  # pp per year: 2024-2027 / 2028+


@dataclass
class PathGenerator:
    start_year: int
    end_year: int

    def _expand_phase_values(self, policy: Dict[str, Dict[str, float]]) -> Dict[str, Dict[int, float]]:
        annual: Dict[str, Dict[int, float]] = {}
        for var, phase_values in policy.items():
            annual[var] = {}
            for year in range(self.start_year, self.end_year + 1):
                phase = _phase_for_year(year)
                annual[var][year] = float(phase_values[phase])
        return annual

    @staticmethod
    def _piecewise_value(year: int, v1: float, v2: float, v3: float) -> float:
        if 2024 <= year <= 2027:
            return float(v1)
        if 2028 <= year <= 2030:
            return float(v2)
        return float(v3)

    def _apply_target_peak_paths(
        self,
        annual: Dict[str, Dict[int, float]],
        spec: ScenarioPeakSpec,
        t_start: int,
        ci_mid_adjust: float = 0.0,
        coal_mid_adjust: float = 0.0,
        energy_mid_adjust: float = 0.0,
    ) -> None:
        t_peak = int(spec.target_peak_year)
        ci_early, ci_mid, ci_late = spec.carbon_intensity
        coal_early, coal_mid, coal_late = spec.coal_share

        for year in range(self.start_year, self.end_year + 1):
            energy_rate = self._piecewise_value(year, *spec.energy) / 100.0
            annual["Industry"][year] = float(spec.industry[0] if year <= 2027 else spec.industry[1])

            if year < t_start:
                ci = ci_early / 100.0
                coal = coal_early
            elif year < t_peak:
                ci = (ci_mid / 100.0) + ci_mid_adjust
                coal = coal_mid + coal_mid_adjust
                energy_rate = energy_rate + energy_mid_adjust
            else:
                ci = ci_late / 100.0
                coal = coal_late

            annual["Energy"][year] = float(np.clip(energy_rate, -0.03, 0.06))
            annual["CarbonIntensity"][year] = float(np.clip(ci, -0.06, 0.0))
            annual["EnergyIntensity"][year] = float(np.clip(ci * 0.8, -0.06, 0.0))
            annual["CoalShare"][year] = float(np.clip(coal, -3.0, 0.0))

    def generate_paths(
        self,
        policy: Dict[str, Dict[str, float]],
        spec: ScenarioPeakSpec,
        t_start_override: Optional[int] = None,
        ci_mid_adjust: float = 0.0,
        coal_mid_adjust: float = 0.0,
        energy_mid_adjust: float = 0.0,
    ) -> Dict[str, Dict[int, float]]:
        annual = self._expand_phase_values(policy)
        t_start = int(spec.t_start if t_start_override is None else t_start_override)
        self._apply_target_peak_paths(
            annual=annual,
            spec=spec,
            t_start=t_start,
            ci_mid_adjust=ci_mid_adjust,
            coal_mid_adjust=coal_mid_adjust,
            energy_mid_adjust=energy_mid_adjust,
        )
        return annual


@dataclass
class ConstraintChecker:
    start_year: int
    end_year: int
    peak_year_min: int = 2026
    peak_year_max: int = 2030

    def _nonfossil_target(self, year: int) -> float:
        if year <= 2025:
            return float(19.0 + (year - 2024) * 1.0)
        if year <= 2030:
            return float(20.0 + (year - 2025) * 1.0)
        return float(25.0 + (year - 2030) * 1.0)

    def check_constraints(
        self,
        scenario_df: pd.DataFrame,
        national_df: pd.DataFrame,
        scenario_name: str,
        require_peak: bool = True,
        peak_year_min: Optional[int] = None,
        peak_year_max: Optional[int] = None,
    ) -> List[str]:
        violations: List[str] = []

        if scenario_df.empty or national_df.empty:
            return [f"{scenario_name}: empty scenario output"]

        nat = national_df.sort_values("year", kind="stable").reset_index(drop=True)
        peak_idx = int(np.argmax(nat["co2_pred"].to_numpy(dtype=float)))
        peak_year = int(nat.loc[peak_idx, "year"])
        min_peak = self.peak_year_min if peak_year_min is None else int(peak_year_min)
        max_peak = self.peak_year_max if peak_year_max is None else int(peak_year_max)
        if require_peak and not (min_peak <= peak_year <= max_peak):
            violations.append(
                f"{scenario_name}: peak year {peak_year} out of [{min_peak}, {max_peak}]"
            )

        nat = nat.copy()
        nat["ei"] = nat["co2_pred"] / np.clip(nat["energy_pred"].to_numpy(dtype=float), 1e-8, None)
        nat["g_co2"] = nat["co2_pred"].pct_change()
        nat["g_energy"] = nat["energy_pred"].pct_change()
        nat["g_ei"] = nat["ei"].pct_change()
        if require_peak:
            if peak_year <= self.start_year:
                violations.append(f"{scenario_name}: peak year must be after {self.start_year}")

            pre_peak = nat[nat["year"] == (peak_year - 1)]
            if not pre_peak.empty:
                g_pre = float(pre_peak["g_co2"].iloc[0])
                if not (0.0 <= g_pre <= 0.01):
                    violations.append(f"{scenario_name}: g_CO2 at peak-1 not in [0, 1%]")

            post_peak = nat[nat["year"] == (peak_year + 1)]
            if not post_peak.empty:
                g_post = float(post_peak["g_co2"].iloc[0])
                if g_post >= -1e-8:
                    violations.append(f"{scenario_name}: g_CO2 at peak+1 must be negative")

        s = scenario_df.copy()
        share_sum = s["CoalShare"] + s["OilShare"] + s["GasShare"] + s["NonFossilShare"]
        if np.max(np.abs(share_sum.to_numpy(dtype=float) - 100.0)) > 1e-4:
            violations.append(f"{scenario_name}: energy shares do not sum to 100")

        for year, grp in s.groupby("year", sort=False):
            target = self._nonfossil_target(int(year))
            if float(grp["NonFossilShare"].min()) + 1e-6 < target:
                violations.append(
                    f"{scenario_name}: NonFossilShare below target in {int(year)} (target={target:.2f})"
                )

        s = s.sort_values(["province", "year"], kind="stable")
        s["d_coal"] = s.groupby("province", sort=False)["CoalShare"].diff()
        s["d_industry"] = s.groupby("province", sort=False)["Industry"].diff()
        if float(s["d_coal"].abs().max(skipna=True)) > 3.0 + 1e-8:
            violations.append(f"{scenario_name}: |ΔCoalShare| exceeds 3 pp/year")
        if float(s["d_industry"].abs().max(skipna=True)) > 2.0 + 1e-8:
            violations.append(f"{scenario_name}: |ΔIndustry| exceeds 2 pp/year")

        return violations


@dataclass
class Calibrator:
    max_iter: int = 20

    def calibrate(
        self,
        scenario_name: str,
        initial_t_start: int,
        target_peak_year: int,
        regenerate_paths: Callable[[int, float, float, float], Dict[str, Dict[int, float]]],
        simulate_once: Callable[[Dict[str, Dict[int, float]], int], pd.DataFrame],
        checker: ConstraintChecker,
    ) -> Tuple[Dict[str, Dict[int, float]], int, Dict[str, Any]]:
        t_start = int(initial_t_start)
        ci_mid_adjust = 0.0
        coal_mid_adjust = 0.0
        energy_mid_adjust = 0.0
        paths = regenerate_paths(t_start, ci_mid_adjust, coal_mid_adjust, energy_mid_adjust)
        history: List[Dict[str, Any]] = []
        best: Optional[Dict[str, Any]] = None
        peak_year = target_peak_year
        violations: List[str] = []

        for i in range(self.max_iter):
            paths = regenerate_paths(t_start, ci_mid_adjust, coal_mid_adjust, energy_mid_adjust)
            scenario_df = simulate_once(paths, t_start)
            national_df = (
                scenario_df.groupby("year", as_index=False)[["co2_pred", "energy_pred"]]
                .sum()
                .sort_values("year", kind="stable")
            )
            peak_year = int(national_df.loc[int(np.argmax(national_df["co2_pred"].to_numpy(dtype=float))), "year"])
            objective = float((peak_year - target_peak_year) ** 2)
            violations = checker.check_constraints(
                scenario_df=scenario_df,
                national_df=national_df,
                scenario_name=scenario_name,
                require_peak=True,
                peak_year_min=target_peak_year,
                peak_year_max=target_peak_year,
            )
            history.append(
                {
                    "iter": i + 1,
                    "peak_year": peak_year,
                    "target_peak": target_peak_year,
                    "objective": objective,
                    "t_start": t_start,
                    "ci_mid_adjust": ci_mid_adjust,
                    "coal_mid_adjust": coal_mid_adjust,
                    "energy_mid_adjust": energy_mid_adjust,
                    "violations": len(violations),
                }
            )

            if best is None or objective < float(best["objective"]):
                best = {
                    "objective": objective,
                    "paths": paths,
                    "t_start": t_start,
                    "peak_year": peak_year,
                    "violations": violations,
                    "ci_mid_adjust": ci_mid_adjust,
                    "coal_mid_adjust": coal_mid_adjust,
                    "energy_mid_adjust": energy_mid_adjust,
                }

            if peak_year == target_peak_year and not violations:
                return paths, t_start, {
                    "iterations": i + 1,
                    "peak_year": peak_year,
                    "target_peak_year": target_peak_year,
                    "t_start": t_start,
                    "ci_mid_adjust": ci_mid_adjust,
                    "coal_mid_adjust": coal_mid_adjust,
                    "energy_mid_adjust": energy_mid_adjust,
                    "violations": violations,
                    "history": history,
                }

            if peak_year < target_peak_year:
                t_start = min(t_start + 1, 2033)
                ci_mid_adjust = min(ci_mid_adjust + 0.003, 0.02)
                coal_mid_adjust = min(coal_mid_adjust + 0.10, 1.0)
                energy_mid_adjust = max(energy_mid_adjust - 0.002, -0.02)
            elif peak_year > target_peak_year:
                t_start = max(t_start - 1, 2024)
                ci_mid_adjust = max(ci_mid_adjust - 0.003, -0.02)
                coal_mid_adjust = max(coal_mid_adjust - 0.10, -1.0)
                energy_mid_adjust = min(energy_mid_adjust + 0.002, 0.02)

        if best is None:
            best = {
                "paths": paths,
                "t_start": t_start,
                "peak_year": peak_year,
                "violations": violations,
                "objective": float((peak_year - target_peak_year) ** 2),
                "ci_mid_adjust": ci_mid_adjust,
                "coal_mid_adjust": coal_mid_adjust,
                "energy_mid_adjust": energy_mid_adjust,
            }

        final_paths = best["paths"]
        final_t_start = int(best["t_start"])
        final_df = simulate_once(final_paths, final_t_start)
        final_national_df = (
            final_df.groupby("year", as_index=False)[["co2_pred", "energy_pred"]]
            .sum()
            .sort_values("year", kind="stable")
        )
        final_peak = int(final_national_df.loc[int(np.argmax(final_national_df["co2_pred"].to_numpy(dtype=float))), "year"])
        final_violations = checker.check_constraints(
            scenario_df=final_df,
            national_df=final_national_df,
            scenario_name=scenario_name,
            require_peak=True,
            peak_year_min=target_peak_year,
            peak_year_max=target_peak_year,
        )
        return final_paths, final_t_start, {
            "iterations": self.max_iter,
            "peak_year": final_peak,
            "target_peak_year": target_peak_year,
            "objective": float(best["objective"]),
            "t_start": final_t_start,
            "ci_mid_adjust": float(best["ci_mid_adjust"]),
            "coal_mid_adjust": float(best["coal_mid_adjust"]),
            "energy_mid_adjust": float(best["energy_mid_adjust"]),
            "violations": final_violations,
            "history": history,
        }

def build_scenario_policy() -> Dict[str, Dict[str, Dict[str, float]]]:
    # Base values shared across all scenarios for unchanged variables
    base_shared = {
        "Population": {"2024_2025": 0.0005, "2026_2030": 0.0, "2031_2035": -0.0005},
        "Urbanization": {"2024_2025": 0.75, "2026_2030": 0.65, "2031_2035": 0.5},
        "PrivateCars": {"2024_2025": 0.05, "2026_2030": 0.035, "2031_2035": 0.02},
    }

    baseline = {
        "Population": dict(base_shared["Population"]),
        "pGDP": {"2024_2025": 0.05, "2026_2030": 0.045, "2031_2035": 0.04},
        "OilShare": {"2024_2025": -0.15, "2026_2030": -0.10, "2031_2035": -0.10},
        "GasShare": {"2024_2025": 0.10, "2026_2030": 0.10, "2031_2035": 0.05},
        "Urbanization": dict(base_shared["Urbanization"]),
        "PrivateCars": dict(base_shared["PrivateCars"]),
    }

    low_carbon = {
        "Population": dict(base_shared["Population"]),
        "pGDP": {"2024_2025": 0.05, "2026_2030": 0.045, "2031_2035": 0.04},
        "OilShare": {"2024_2025": -0.25, "2026_2030": -0.20, "2031_2035": -0.20},
        "GasShare": {"2024_2025": 0.15, "2026_2030": 0.15, "2031_2035": 0.10},
        "Urbanization": dict(base_shared["Urbanization"]),
        "PrivateCars": dict(base_shared["PrivateCars"]),
    }

    extensive = {
        "Population": dict(base_shared["Population"]),
        "pGDP": {"2024_2025": 0.05, "2026_2030": 0.045, "2031_2035": 0.04},
        "OilShare": {"2024_2025": -0.05, "2026_2030": -0.02, "2031_2035": 0.0},
        "GasShare": {"2024_2025": 0.05, "2026_2030": 0.05, "2031_2035": 0.03},
        "Urbanization": dict(base_shared["Urbanization"]),
        "PrivateCars": dict(base_shared["PrivateCars"]),
    }

    green_growth = {
        "Population": dict(base_shared["Population"]),
        "pGDP": {"2024_2025": 0.052, "2026_2030": 0.047, "2031_2035": 0.042},
        "OilShare": {"2024_2025": -0.25, "2026_2030": -0.20, "2031_2035": -0.20},
        "GasShare": {"2024_2025": 0.15, "2026_2030": 0.15, "2031_2035": 0.10},
        "Urbanization": dict(base_shared["Urbanization"]),
        "PrivateCars": dict(base_shared["PrivateCars"]),
    }

    deep_decarb = {
        "Population": dict(base_shared["Population"]),
        "pGDP": {"2024_2025": 0.05, "2026_2030": 0.045, "2031_2035": 0.04},
        "OilShare": {"2024_2025": -0.35, "2026_2030": -0.30, "2031_2035": -0.30},
        "GasShare": {"2024_2025": 0.20, "2026_2030": 0.20, "2031_2035": 0.15},
        "Urbanization": dict(base_shared["Urbanization"]),
        "PrivateCars": dict(base_shared["PrivateCars"]),
    }

    return {
        "baseline": baseline,
        "low_carbon": low_carbon,
        "extensive": extensive,
        "green_growth": green_growth,
        "deep_decarb": deep_decarb,
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


def _co2_intensity_from_shares(coal_share: Any, oil_share: Any, gas_share: Any) -> Any:
    return (
        coal_share * IPCC_EF_TCO2_PER_TCE["coal"]
        + oil_share * IPCC_EF_TCO2_PER_TCE["oil"]
        + gas_share * IPCC_EF_TCO2_PER_TCE["gas"]
    ) / 100.0


def _co2_from_ipcc(energy: float, coal_share: float, oil_share: float, gas_share: float) -> float:
    intensity = _co2_intensity_from_shares(coal_share, oil_share, gas_share)
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
    "EnergyIntensity",    # 新：替代 CoalShare
    "log_PrivateCars",
    ]
    # 注意：删除了 CoalShare 和 log_CarbonIntensity

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


def _build_row_for_next_year(
    prev_row: pd.Series,
    policy: Dict[str, Dict[str, float]],
    annual_path: Optional[Dict[str, Dict[int, float]]],
    year: int,
    highway_rate: float,
    start_year: int = 2024,      # 新增参数
) -> Dict[str, float]:
    phase = _phase_for_year(year)

    def _get_policy_value(var: str) -> float:
        if annual_path is not None and var in annual_path and year in annual_path[var]:
            return float(annual_path[var][year])
        if var not in policy or phase not in policy[var]:
            raise KeyError(f"Missing policy value for {var} in {phase} at year={year}")
        return float(policy[var][phase])

    energy_growth = _get_policy_value("Energy")
    coal_share_delta = _get_policy_value("CoalShare")
    carbon_intensity_rate = _get_policy_value("CarbonIntensity")
    energy_intensity_rate = _get_policy_value("EnergyIntensity")
    industry_delta = _get_policy_value("Industry")
    urbanization_delta = _get_policy_value("Urbanization")
    oil_share_delta = _get_policy_value("OilShare")
    gas_share_delta = _get_policy_value("GasShare")

    population = float(prev_row["Population"]) * (1.0 + policy["Population"][phase])
    pgdp = float(prev_row["pGDP"]) * (1.0 + policy["pGDP"][phase])
    gdp = population * pgdp
    energy = float(prev_row["Energy"]) * (1.0 + energy_growth)
    carbon_intensity = float(prev_row["CarbonIntensity"]) * (1.0 + carbon_intensity_rate)
    private_cars = float(prev_row["PrivateCars"]) * (1.0 + policy["PrivateCars"][phase])

    # ---- 过渡逻辑：2024 年沿用上一年能源结构，避免突变 ----
    if year == start_year:
        coal_share, oil_share, gas_share, nonfossil_share = _normalize_energy_shares(
            float(prev_row["CoalShare"]),
            float(prev_row["OilShare"]),
            float(prev_row["GasShare"]),
        )
    else:
        coal_share = _clip_pct(float(prev_row["CoalShare"]) + coal_share_delta)
        oil_share = _clip_pct(float(prev_row["OilShare"]) + oil_share_delta)
        gas_share = _clip_pct(float(prev_row["GasShare"]) + gas_share_delta)
        coal_share, oil_share, gas_share, nonfossil_share = _normalize_energy_shares(
            coal_share, oil_share, gas_share
        )
    # ---------------------------------------------------------

    industry = _clip_pct(float(prev_row["Industry"]) + industry_delta)
    urbanization = _clip_pct(float(prev_row["Urbanization"]) + urbanization_delta)
    highway = max(float(prev_row["HighwayMileage"]) * (1.0 + highway_rate), 1e-8)

    energy_intensity = max(float(prev_row["EnergyIntensity"]) * (1.0 + energy_intensity_rate), 1e-8)

    return {
        "year": int(year),
        "Population": population,
        "pGDP": pgdp,
        "GDP": gdp,
        "Energy": energy,
        "EnergyIntensity": energy_intensity,    # 新增
        "CoalShare": coal_share,
        "OilShare": oil_share,
        "GasShare": gas_share,
        "NonFossilShare": nonfossil_share,
        "Industry": industry,
        "Urbanization": urbanization,
        "CarbonIntensity": carbon_intensity,    # 保留，但不作为 STIRPAT 输入
        "PrivateCars": private_cars,
        "HighwayMileage": highway,
        "log_Population": _safe_log(population),
        "log_pGDP": _safe_log(pgdp),
        "log_GDP": _safe_log(gdp),
        "log_Energy": _safe_log(energy),
        "log_CarbonIntensity": _safe_log(carbon_intensity),  # 保留
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
    annual_path: Optional[Dict[str, Dict[int, float]]],
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
    scenario_rule: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    residual_lower = float(np.log(0.8))
    residual_upper = float(np.log(1.2))
    energy_inertia_alpha = 0.7
    scenario_rule = scenario_rule or {}
    require_peak = bool(scenario_rule.get("require_peak", False))
    t_start = int(scenario_rule.get("t_start", cfg.start_year))
    target_peak_year = int(scenario_rule.get("target_peak_year", cfg.end_year))

    # Seed residual recursion with historical true residuals through 2023.
    hist = panel_res_df[panel_res_df["year"] <= (cfg.start_year - 1)].copy()
    hist = hist[[
        "province",
        "year",
        "Population",
        "pGDP",
        "GDP",
        "Energy",
        "EnergyIntensity",     # 新增
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
            row = _build_row_for_next_year(
                prev_row=prev,
                policy=policy,
                annual_path=annual_path,
                year=year,
                highway_rate=hwy_rate,
                start_year=cfg.start_year,
            )

            stirpat_input = pd.DataFrame(
                {
                    "province": [province],
                    "log_Population": [row["log_Population"]],
                    "log_pGDP": [row["log_pGDP"]],
                    "Industry": [row["Industry"]],
                    "Urbanization": [row["Urbanization"]],
                    "EnergyIntensity": [row["EnergyIntensity"]],    # 替换原来的 "CoalShare"
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
            pred_residual = float(np.clip(pred_residual, residual_lower, residual_upper))

            hybrid_log_energy = stirpat_log_energy + pred_residual
            energy_model_pred = float(np.exp(hybrid_log_energy))
            prev_energy = float(prev["Energy"])
            energy_pred = float(energy_inertia_alpha * prev_energy + (1.0 - energy_inertia_alpha) * energy_model_pred)

            hybrid_log_energy = _safe_log(energy_pred)
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


def summarize_peak(forecast_df: pd.DataFrame, end_year: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
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


def estimate_energy_calibration_factor(validation_csv: Path, calibration_year_window: int) -> Tuple[float, Dict[str, Any]]:
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
    co2_intensity = _co2_intensity_from_shares(
        out["CoalShare"].to_numpy(dtype=float),
        out["OilShare"].to_numpy(dtype=float),
        out["GasShare"].to_numpy(dtype=float),
    )
    out["co2_pred"] = out["energy_pred"] * co2_intensity
    return out


def extract_final_policy_paths(paths: Dict[str, Dict[int, float]]) -> Dict[str, Dict[str, float]]:
    phase_ranges: Dict[str, Tuple[int, int]] = {
        "2024_2027": (2024, 2027),
        "2028_2030": (2028, 2030),
        "2031_2035": (2031, 2035),
    }
    summary: Dict[str, Dict[str, float]] = {}

    for var, yearly in paths.items():
        phase_means: Dict[str, float] = {}
        for phase_name, (y0, y1) in phase_ranges.items():
            values = [float(v) for y, v in yearly.items() if y0 <= int(y) <= y1]
            phase_means[phase_name] = float(np.mean(values)) if values else float("nan")
        summary[var] = phase_means

    return summary


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
    path_generator = PathGenerator(start_year=cfg.start_year, end_year=cfg.end_year)
    checker = ConstraintChecker(start_year=cfg.start_year, end_year=cfg.end_year)
    calibrator = Calibrator()
    scenario_specs: Dict[str, ScenarioPeakSpec] = {
        "baseline": ScenarioPeakSpec(
            target_peak_year=2029,
            require_peak=True,
            t_start=2026,
            energy=(2.5, 1.8, 1.0),
            carbon_intensity=(-2.0, -2.8, -2.2),
            coal_share=(-1.2, -1.5, -1.0),
            industry=(-1.2, -0.8),
        ),
        "low_carbon": ScenarioPeakSpec(
            target_peak_year=2027,
            require_peak=True,
            t_start=2025,
            energy=(2.3, 1.6, 0.8),
            carbon_intensity=(-2.5, -3.5, -2.8),
            coal_share=(-1.5, -2.0, -1.5),
            industry=(-1.5, -1.1),
        ),
        "green_growth": ScenarioPeakSpec(
            target_peak_year=2030,
            require_peak=True,
            t_start=2027,
            energy=(2.7, 2.0, 1.2),
            carbon_intensity=(-2.0, -3.0, -2.5),
            coal_share=(-1.2, -1.8, -1.2),
            industry=(-1.3, -1.0),
        ),
        "deep_decarb": ScenarioPeakSpec(
            target_peak_year=2028,
            require_peak=False,
            t_start=2027,
            energy=(2.4, 1.5, 0.6),
            carbon_intensity=(-2.0, -4.0, -3.5),
            coal_share=(-1.0, -3.0, -2.5),
            industry=(-1.5, -1.2),
        ),
        "extensive": ScenarioPeakSpec(
            target_peak_year=2034,
            require_peak=False,
            t_start=2029,
            energy=(2.8, 2.2, 1.5),
            carbon_intensity=(-1.5, -2.0, -1.8),
            coal_share=(-0.8, -1.0, -0.8),
            industry=(-1.0, -0.6),
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
            return forecast_scenario(
                scenario_name=scenario_name,
                policy=policy,
                annual_path=paths,
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
                scenario_rule=run_rule,
            )

        if bool(rule["require_peak"]):
            def _regenerate_paths(
                t_start: int,
                ci_mid_adjust: float,
                coal_mid_adjust: float,
                energy_mid_adjust: float,
            ) -> Dict[str, Dict[int, float]]:
                return path_generator.generate_paths(
                    policy=policy,
                    spec=spec,
                    t_start_override=t_start,
                    ci_mid_adjust=ci_mid_adjust,
                    coal_mid_adjust=coal_mid_adjust,
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
        "energy_calibration": calib_meta,
        "constraint_report": constraint_report,
        "calibration_meta": calibration_meta_by_scenario,
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
    print(f"Saved final policy path summary: {final_path_table_path}")
    print(f"Saved organized package dir: {organized['organized_dir']}")
    print(f"Saved organized combined table: {organized['combined_long']}")
    print(f"Applied energy calibration factor: {calib_factor:.6f}")
    print("\nPeak results:")
    print(peak_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
