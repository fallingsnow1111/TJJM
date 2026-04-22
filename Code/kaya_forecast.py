from __future__ import annotations

"""Policy-constrained Kaya forecast.

The module uses the final historical panel as a baseline and applies the policy
constraints recorded in forecast_policy_15th_fyp.json. The executable forecast
uses a data-generated total energy-intensity path EI = Energy / GDP that is
estimated from historical log(A) and log(S) plus a recent residual-trend
extrapolation. S remains an explanatory structural series, and the forecast
identity is:

	CO2 = P * A * EI * C

where EI is model-implied rather than hand-set by a bridge path.
"""

from pathlib import Path
import json
from typing import Dict, List, cast

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "Code"
OUTPUT_DIR = CODE_DIR / "output"
FIGURE_DIR = OUTPUT_DIR / "figures"
PANEL_PATH = OUTPUT_DIR / "panel_master.csv"
CONFIG_PATH = CODE_DIR / "forecast_policy_15th_fyp.json"
OUT_CSV = OUTPUT_DIR / "kaya_forecast_paths.csv"
OUT_SUMMARY = OUTPUT_DIR / "kaya_forecast_summary.md"
OUT_FIGURE = FIGURE_DIR / "kaya_forecast_paths.png"


def load_config() -> Dict:
	return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def load_panel() -> pd.DataFrame:
	return pd.read_csv(PANEL_PATH)


def year_band(year: int) -> str:
	if 2023 <= year <= 2025:
		return "2023-2025"
	if 2026 <= year <= 2030:
		return "2026-2030"
	return "2031-2035"


def build_annual_series(panel: pd.DataFrame) -> pd.DataFrame:
	panel = panel.copy()
	panel["Industry"] = pd.to_numeric(panel["Industry"], errors="coerce")
	annual = panel.groupby("year")[["CO2", "GDP", "Population", "Energy"]].sum().reset_index()
	structure = (panel["Industry"] * panel["GDP"]).groupby(panel["year"]).sum() / panel.groupby("year")["GDP"].sum() / 100.0
	annual["S"] = structure.to_numpy()
	annual["A"] = annual["GDP"] / annual["Population"]
	annual["EI"] = annual["Energy"] / annual["GDP"]
	annual["C"] = annual["CO2"] / annual["Energy"]
	annual["log_A"] = np.log(annual["A"])
	annual["log_S"] = np.log(annual["S"])
	annual["log_EI"] = np.log(annual["EI"])
	return annual


def fit_ei_model(annual: pd.DataFrame, ei_cfg: Dict) -> Dict[str, float]:
	fit_start_year = int(ei_cfg.get("fit_start_year", 1990))
	residual_window_years = max(2, int(ei_cfg.get("residual_window_years", 4)))
	fit = annual[annual["year"] >= fit_start_year].copy()
	if fit.empty:
		raise ValueError(f"No annual rows available from {fit_start_year} onward")
	design = np.column_stack([np.ones(len(fit)), fit["log_A"], fit["log_S"]])
	coef, *_ = np.linalg.lstsq(design, fit["log_EI"].to_numpy(), rcond=None)
	fitted = design @ coef
	residuals = fit["log_EI"].to_numpy() - fitted
	recent = fit.tail(residual_window_years)
	if len(recent) < 2:
		raise ValueError("Not enough residual observations to fit EI trend")
	anchor_year = int(fit["year"].iloc[-1])
	recent_offsets = recent["year"].to_numpy(dtype=float) - float(anchor_year)
	trend_design = np.column_stack([np.ones(len(recent)), recent_offsets])
	trend_coef, *_ = np.linalg.lstsq(trend_design, residuals[-len(recent):], rcond=None)
	ss_res = float(((fit["log_EI"].to_numpy() - fitted) ** 2).sum())
	ss_tot = float(((fit["log_EI"].to_numpy() - float(fit["log_EI"].mean())) ** 2).sum())
	fit_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
	return {
		"alpha": float(coef[0]),
		"beta_A": float(coef[1]),
		"beta_S": float(coef[2]),
		"trend_intercept": float(trend_coef[0]),
		"trend_slope": float(trend_coef[1]),
		"anchor_year": float(anchor_year),
		"fit_start_year": float(fit_start_year),
		"residual_window_years": float(residual_window_years),
		"fit_r2": float(fit_r2),
	}


def project_log_ei(ei_model: Dict[str, float], year: int, a_value: float, s_value: float) -> float:
	anchor_year = float(ei_model["anchor_year"])
	residual_adjustment = ei_model["trend_intercept"] + ei_model["trend_slope"] * (float(year) - anchor_year)
	return (
		ei_model["alpha"]
		+ ei_model["beta_A"] * float(np.log(a_value))
		+ ei_model["beta_S"] * float(np.log(s_value))
		+ residual_adjustment
	)


def project_scenario(base_state: Dict[str, float], config: Dict, ei_model: Dict[str, float], scenario: str, theta: Dict[str, float]) -> pd.DataFrame:
	rows: List[Dict[str, float]] = []
	state = base_state.copy()
	base_row = {"scenario": scenario, **state}
	base_row["year"] = int(state["year"])
	rows.append(base_row)
	period_rates = config["base_growth_rates"]
	for year in range(2023, 2036):
		band = year_band(year)
		rates = {
			"P": float(period_rates["P"][band]),
			"A": float(period_rates["A"][band]),
			"S": float(period_rates["S"][band]),
			"C": float(period_rates["C"][band]),
		}
		for factor in ["P", "A", "S", "C"]:
			state[factor] = state[factor] * np.exp(float(rates[factor]) * float(theta[factor]))
		state["S"] = min(max(state["S"], 0.01), 0.99)
		state["C"] = max(state["C"], 1e-8)
		state["EI"] = max(np.exp(project_log_ei(ei_model, year, state["A"], state["S"])), 1e-8)
		state["B"] = state["EI"] / state["S"] if state["S"] > 0 else float("nan")
		state["CO2"] = state["P"] * state["A"] * state["EI"] * state["C"]
		row = {"scenario": scenario, **state}
		row["year"] = int(year)
		rows.append(row)
	return pd.DataFrame(rows)


def build_base_state(panel: pd.DataFrame) -> Dict[str, float]:
	base = panel[panel["year"] == 2022].copy()
	if base.empty:
		raise ValueError("No 2022 baseline found in panel_master.csv")
	gdp = float(base["GDP"].sum())
	population = float(base["Population"].sum())
	energy = float(base["Energy"].sum())
	co2 = float(base["CO2"].sum())
	secondary_share = float((pd.to_numeric(base["Industry"], errors="coerce") * pd.to_numeric(base["GDP"], errors="coerce")).sum() / gdp / 100.0)
	A = gdp / population
	S = secondary_share
	EI = energy / gdp
	B = EI / S if S > 0 else float("nan")
	C = co2 / energy
	return {
		"year": 2022,
		"P": population,
		"A": A,
		"S": S,
		"EI": EI,
		"B": B,
		"C": C,
		"CO2": co2,
	}


def forecast_paths(base_state: Dict[str, float], config: Dict, annual: pd.DataFrame, residual_window_years: int | None = None) -> pd.DataFrame:
	ei_cfg = dict(config.get("ei_model", {}))
	if residual_window_years is not None:
		ei_cfg["residual_window_years"] = int(residual_window_years)
	ei_model = fit_ei_model(annual, ei_cfg)
	rows: List[Dict[str, object]] = []
	for scenario, theta in config["scenario_multipliers"].items():
		scenario_frame = project_scenario(base_state, config, ei_model, scenario, theta)
		for record in scenario_frame.to_dict(orient="records"):
			rows.append(cast(Dict[str, object], record))
	return pd.DataFrame(rows)


def peak_year_for_scenario(df: pd.DataFrame, scenario: str) -> int:
	g = df[df["scenario"] == scenario]
	peak = g.loc[g["CO2"].idxmax()]
	return int(cast(float, peak["year"]))


def make_summary(df: pd.DataFrame, base_state: Dict[str, float], config: Dict, annual: pd.DataFrame) -> str:
	lines: List[str] = []
	ei_cfg = config.get("ei_model", {})
	main_model = fit_ei_model(annual, ei_cfg)
	lines.append("# Kaya Forecast Summary")
	lines.append("")
	lines.append(f"- Base year: {int(base_state['year'])}")
	lines.append(f"- Base CO2: {base_state['CO2']:.3f}")
	lines.append(f"- Forecast horizon: {config['forecast_horizon']['start_year']}-{config['forecast_horizon']['end_year']}")
	lines.append(f"- EI model: log(EI) = alpha + beta_A * log(A) + beta_S * log(S) + recent residual trend")
	lines.append(f"- EI fit start year: {int(main_model['fit_start_year'])}")
	lines.append(f"- EI residual window: {int(main_model['residual_window_years'])} years")
	lines.append(f"- EI fit R^2: {main_model['fit_r2']:.4f}")
	lines.append("")
	for scenario, g in df.groupby("scenario"):
		peak = g.loc[g["CO2"].idxmax()]
		peak_year = int(cast(float, peak["year"]))
		peak_co2 = float(cast(float, peak["CO2"]))
		year_lookup = g.set_index("year")["CO2"]
		co2_2030 = float(year_lookup.loc[2030]) if 2030 in year_lookup.index else float("nan")
		co2_2035 = float(year_lookup.loc[2035]) if 2035 in year_lookup.index else float("nan")
		lines.append(f"## {scenario}")
		lines.append(f"- Peak year: {peak_year}")
		lines.append(f"- Peak CO2: {peak_co2:.3f}")
		lines.append(f"- 2030 CO2: {co2_2030:.3f}")
		lines.append(f"- 2035 CO2: {co2_2035:.3f}")
		lines.append(f"- 2022-2035 cumulative change: {g['CO2'].iloc[-1] - g['CO2'].iloc[0]:.3f}")
		lines.append("")
	lines.append("## Sensitivity")
	for window in ei_cfg.get("sensitivity_windows", [3, 4, 5]):
		sensitivity_df = forecast_paths(base_state, config, annual, residual_window_years=int(window))
		baseline_peak_year = peak_year_for_scenario(sensitivity_df, "baseline")
		lines.append(f"- Residual window {int(window)} years: baseline peak year {baseline_peak_year}")
	lines.append("")
	lines.append("## Interpretation")
	lines.append("This forecast is a deterministic Kaya-path projection under policy constraints. EI is not hand-set by a bridge; it is generated from historical structure plus a residual trend estimated from the most recent observations. If the implied peak year is still outside the target narrative, the first diagnostic should be the EI residual-window sensitivity rather than editing the historical panel.")
	return "\n".join(lines)


def make_figure(df: pd.DataFrame) -> None:
	fig, ax = plt.subplots(figsize=(12, 6.5))
	for scenario, g in df.groupby("scenario"):
		g = g.sort_values("year")
		ax.plot(g["year"], g["CO2"], marker="o", ms=3.5, lw=2.0, label=scenario)
	ax.axvline(2022, color="#666666", ls="--", lw=1.0)
	ax.set_title("Regression-generated Kaya forecast, 2022-2035")
	ax.set_xlabel("Year")
	ax.set_ylabel("CO2")
	ax.grid(alpha=0.2)
	ax.legend(frameon=False)
	ax.spines[["top", "right"]].set_visible(False)
	fig.tight_layout()
	fig.savefig(OUT_FIGURE, dpi=300, bbox_inches="tight")
	plt.close(fig)


def main() -> None:
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	FIGURE_DIR.mkdir(parents=True, exist_ok=True)
	config = load_config()
	panel = load_panel()
	annual = build_annual_series(panel)
	base_state = build_base_state(panel)
	forecast = forecast_paths(base_state, config, annual)
	forecast.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
	OUT_SUMMARY.write_text(make_summary(forecast, base_state, config, annual), encoding="utf-8")
	make_figure(forecast)
	print(f"WROTE {OUT_CSV}")
	print(f"WROTE {OUT_SUMMARY}")
	print(f"WROTE {OUT_FIGURE}")
	for scenario, g in forecast.groupby("scenario"):
		peak = g.loc[g["CO2"].idxmax()]
		peak_year = int(cast(float, peak["year"]))
		peak_co2 = float(cast(float, peak["CO2"]))
		print(scenario, "peak_year", peak_year, "peak_CO2", round(peak_co2, 3))


if __name__ == "__main__":
	main()
