from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Ensure imports work when running as: python Code/FutureScenario/future_projection.py
CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
	sys.path.insert(0, str(CODE_ROOT))

from Model.stirpat_ee_gru import EntityEmbeddingGRU, PanelRidgeSTIRPAT


STIRPAT_FEATURES = [
	"log_Population",
	"log_pGDP",
	"Industry",
	"Urbanization",
	"CoalShare",
	"log_CarbonIntensity",
	"log_Energy",
	"log_PrivateCars",
]

DEFAULT_DYNAMIC_FEATURES = [
	"log_GDP",
	"log_pGDP",
	"log_Population",
	"Energy",
	"CoalShare",
	"Industry",
	"Urbanization",
	"HighwayMileage",
	"log_PrivateCars",
	"log_CarbonIntensity",
]

SCENARIO_RATES: Dict[str, Dict[str, Dict[str, float]]] = {
	"baseline": {
		"2024-2025": {
			"Population": 0.0005,
			"pGDP": 0.050,
			"Energy": 0.025,
			"CoalShare": -1.5,
			"Industry": -1.2,
			"Urbanization": 0.75,
			"CarbonIntensity": -0.030,
			"PrivateCars": 0.050,
		},
		"2026-2030": {
			"Population": 0.0000,
			"pGDP": 0.045,
			"Energy": 0.018,
			"CoalShare": -1.0,
			"Industry": -0.8,
			"Urbanization": 0.65,
			"CarbonIntensity": -0.025,
			"PrivateCars": 0.035,
		},
		"2031-2035": {
			"Population": -0.0005,
			"pGDP": 0.040,
			"Energy": 0.010,
			"CoalShare": -0.8,
			"Industry": -0.5,
			"Urbanization": 0.50,
			"CarbonIntensity": -0.020,
			"PrivateCars": 0.020,
		},
	},
	"low_carbon": {
		"2024-2025": {
			"Population": 0.0005,
			"pGDP": 0.050,
			"Energy": 0.022,
			"CoalShare": -2.0,
			"Industry": -1.5,
			"Urbanization": 0.75,
			"CarbonIntensity": -0.035,
			"PrivateCars": 0.050,
		},
		"2026-2030": {
			"Population": 0.0000,
			"pGDP": 0.045,
			"Energy": 0.015,
			"CoalShare": -1.5,
			"Industry": -1.1,
			"Urbanization": 0.65,
			"CarbonIntensity": -0.030,
			"PrivateCars": 0.035,
		},
		"2031-2035": {
			"Population": -0.0005,
			"pGDP": 0.040,
			"Energy": 0.007,
			"CoalShare": -1.3,
			"Industry": -0.8,
			"Urbanization": 0.50,
			"CarbonIntensity": -0.025,
			"PrivateCars": 0.020,
		},
	},
	"extensive": {
		"2024-2025": {
			"Population": 0.0005,
			"pGDP": 0.050,
			"Energy": 0.028,
			"CoalShare": -1.2,
			"Industry": -1.0,
			"Urbanization": 0.75,
			"CarbonIntensity": -0.025,
			"PrivateCars": 0.050,
		},
		"2026-2030": {
			"Population": 0.0000,
			"pGDP": 0.045,
			"Energy": 0.021,
			"CoalShare": -0.7,
			"Industry": -0.6,
			"Urbanization": 0.65,
			"CarbonIntensity": -0.020,
			"PrivateCars": 0.035,
		},
		"2031-2035": {
			"Population": -0.0005,
			"pGDP": 0.040,
			"Energy": 0.013,
			"CoalShare": -0.5,
			"Industry": -0.3,
			"Urbanization": 0.50,
			"CarbonIntensity": -0.015,
			"PrivateCars": 0.020,
		},
	},
}


@dataclass
class ProjectionConfig:
	input_csv: Path
	dataset_npz: Path
	model_ckpt: Path
	output_dir: Path
	start_year: int = 2024
	end_year: int = 2035
	train_end_year: int = 2020
	ridge_alpha: float = 1.0
	scenario: str = "all"
	device: str = "auto"


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Project provincial future carbon emissions and aggregate to national totals.")
	base_dir = Path(__file__).resolve().parents[1]
	parser.add_argument(
		"--input-csv",
		type=Path,
		default=base_dir / "Dataset" / "output" / "panel_with_residual.csv",
		help="Historical provincial panel with residuals",
	)
	parser.add_argument(
		"--dataset-npz",
		type=Path,
		default=base_dir / "Dataset" / "output" / "stirpat_ee_gru_dataset.npz",
		help="Training NPZ with dynamic feature order and standardizer stats",
	)
	parser.add_argument(
		"--model-ckpt",
		type=Path,
		default=base_dir / "Model" / "output" / "best_ee_gru.pt",
		help="Trained EE-GRU residual model checkpoint",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path(__file__).resolve().parent / "output",
		help="Directory for future projection outputs",
	)
	parser.add_argument("--start-year", type=int, default=2024)
	parser.add_argument("--end-year", type=int, default=2035)
	parser.add_argument("--train-end-year", type=int, default=2020)
	parser.add_argument("--ridge-alpha", type=float, default=1.0)
	parser.add_argument(
		"--scenario",
		choices=["all", "baseline", "low_carbon", "extensive"],
		default="all",
		help="Scenario to run; all emits every scenario and a comparison plot",
	)
	parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
	return parser


def choose_device(device_arg: str) -> torch.device:
	if device_arg == "cpu":
		return torch.device("cpu")
	if device_arg == "cuda":
		if not torch.cuda.is_available():
			raise RuntimeError("CUDA requested but not available.")
		return torch.device("cuda")
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_npz_metadata(npz_path: Path) -> tuple[list[str], np.ndarray, np.ndarray, int]:
	npz = np.load(npz_path, allow_pickle=True)
	feature_names = [str(x) for x in npz["gru_feature_names"].tolist()]
	mean = np.asarray(npz["standardizer_mean"], dtype=np.float32)
	std = np.asarray(npz["standardizer_std"], dtype=np.float32)
	std = np.where(std < 1e-8, 1.0, std)
	window = int(npz["train_dynamic_x"].shape[1])
	return feature_names, mean, std, window


def load_residual_model(model_ckpt: Path, device: torch.device) -> EntityEmbeddingGRU:
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


def load_historical_panel(input_csv: Path) -> pd.DataFrame:
	if not input_csv.exists():
		raise FileNotFoundError(f"Historical panel not found: {input_csv}")
	df = pd.read_csv(input_csv)
	required = {
		"province",
		"province_id",
		"year",
		"CO2",
		"GDP",
		"Population",
		"Energy",
		"CoalShare",
		"Industry",
		"Urbanization",
		"HighwayMileage",
		"PrivateCars",
		"CarbonIntensity",
		"log_Population",
		"log_pGDP",
		"log_Energy",
		"log_PrivateCars",
		"log_CarbonIntensity",
		"log_CO2",
		"residual",
	}
	missing = required - set(df.columns)
	if missing:
		raise ValueError(f"Missing required columns in historical panel: {sorted(missing)}")
	df = df.copy()
	df["province"] = df["province"].astype(str)
	df["year"] = df["year"].astype(int)
	df = df.sort_values(["province", "year"], kind="stable").reset_index(drop=True)
	return df


def fit_stirpat_model(df: pd.DataFrame, train_end_year: int, ridge_alpha: float) -> PanelRidgeSTIRPAT:
	model = PanelRidgeSTIRPAT(alpha=ridge_alpha)
	train_df = df[df["year"] <= train_end_year].copy()
	if train_df.empty:
		raise RuntimeError("Training window for STIRPAT baseline is empty.")
	model.fit(train_df, feature_cols=STIRPAT_FEATURES, target_col="log_CO2", province_col="province")
	return model


def stage_for_year(year: int) -> str:
	if year <= 2025:
		return "2024-2025"
	if year <= 2030:
		return "2026-2030"
	return "2031-2035"


def clamp_percentage(value: float) -> float:
	return float(np.clip(value, 0.0, 100.0))


def estimate_highway_growth(df: pd.DataFrame) -> Dict[str, float]:
	"""用 2020-2023 的历史增速外推高速公路里程，作为残差模型的辅助动态变量。"""
	window_df = df[df["year"].between(2020, 2023)].sort_values(["province", "year"], kind="stable")
	province_growth: Dict[str, float] = {}
	all_growth: List[float] = []
	for province, group in window_df.groupby("province", sort=False):
		series = group.sort_values("year", kind="stable")["HighwayMileage"].to_numpy(dtype=float)
		if len(series) >= 2 and np.all(series > 0):
			years = int(group["year"].iloc[-1] - group["year"].iloc[0])
			if years > 0:
				growth = float((series[-1] / series[0]) ** (1.0 / years) - 1.0)
				growth = float(np.clip(growth, 0.0, 0.08))
				province_growth[province] = growth
				all_growth.append(growth)
	median_growth = float(np.median(all_growth)) if all_growth else 0.03
	for province in df["province"].astype(str).unique().tolist():
		province_growth.setdefault(province, median_growth)
	return province_growth


def project_one_province(
	base_row: pd.Series,
	scenario_name: str,
	years: Iterable[int],
	highway_growth: float,
) -> pd.DataFrame:
	rates_by_stage = SCENARIO_RATES[scenario_name]
	state = {
		"Population": float(base_row["Population"]),
		"pGDP": float(base_row["GDP"] / base_row["Population"]),
		"Energy": float(base_row["Energy"]),
		"CoalShare": float(base_row["CoalShare"]),
		"Industry": float(base_row["Industry"]),
		"Urbanization": float(base_row["Urbanization"]),
		"CarbonIntensity": float(base_row["CarbonIntensity"]),
		"PrivateCars": float(base_row["PrivateCars"]),
		"HighwayMileage": float(base_row["HighwayMileage"]),
	}

	rows: List[Dict[str, Any]] = []
	for year in years:
		stage = stage_for_year(int(year))
		rates = rates_by_stage[stage]

		state["Population"] *= 1.0 + rates["Population"]
		state["pGDP"] *= 1.0 + rates["pGDP"]
		state["Energy"] *= 1.0 + rates["Energy"]
		state["CarbonIntensity"] *= 1.0 + rates["CarbonIntensity"]
		state["PrivateCars"] *= 1.0 + rates["PrivateCars"]
		state["CoalShare"] = clamp_percentage(state["CoalShare"] + rates["CoalShare"])
		state["Industry"] = clamp_percentage(state["Industry"] + rates["Industry"])
		state["Urbanization"] = clamp_percentage(state["Urbanization"] + rates["Urbanization"])
		state["HighwayMileage"] *= 1.0 + highway_growth

		gdp = state["Population"] * state["pGDP"]
		row = {
			"province": str(base_row["province"]),
			"province_id": int(base_row["province_id"]),
			"scenario": scenario_name,
			"year": int(year),
			"GDP": float(gdp),
			"Population": float(state["Population"]),
			"Energy": float(state["Energy"]),
			"CoalShare": float(state["CoalShare"]),
			"Industry": float(state["Industry"]),
			"Urbanization": float(state["Urbanization"]),
			"HighwayMileage": float(state["HighwayMileage"]),
			"PrivateCars": float(state["PrivateCars"]),
			"CarbonIntensity": float(state["CarbonIntensity"]),
		}
		row["pGDP"] = float(state["pGDP"])
		row["log_GDP"] = float(np.log(max(row["GDP"], 1e-8)))
		row["log_Population"] = float(np.log(max(row["Population"], 1e-8)))
		row["log_pGDP"] = float(np.log(max(row["pGDP"], 1e-8)))
		row["log_Energy"] = float(np.log(max(row["Energy"], 1e-8)))
		row["log_PrivateCars"] = float(np.log(max(row["PrivateCars"], 1e-8)))
		row["log_CarbonIntensity"] = float(np.log(max(row["CarbonIntensity"], 1e-8)))
		rows.append(row)

	return pd.DataFrame(rows)


def build_future_panel(historical_df: pd.DataFrame, scenario_name: str, start_year: int, end_year: int) -> pd.DataFrame:
	province_growth = estimate_highway_growth(historical_df)
	latest = historical_df[historical_df["year"] <= 2023].sort_values(["province", "year"], kind="stable")
	latest = latest.groupby("province", as_index=False, sort=False).tail(1)
	future_years = list(range(start_year, end_year + 1))
	future_parts: List[pd.DataFrame] = []
	for _, base_row in latest.sort_values(["province_id", "province"], kind="stable").iterrows():
		province = str(base_row["province"])
		future_parts.append(
			project_one_province(
				base_row=base_row,
				scenario_name=scenario_name,
				years=future_years,
				highway_growth=province_growth[province],
			)
		)
	future_df = pd.concat(future_parts, ignore_index=True)
	future_df = future_df.sort_values(["province", "year"], kind="stable").reset_index(drop=True)
	return future_df


def standardize_dynamic_features(df: pd.DataFrame, feature_names: list[str], mean: np.ndarray, std: np.ndarray) -> np.ndarray:
	arr = df[feature_names].to_numpy(dtype=np.float32)
	return (arr - mean) / std


def recursive_future_residuals(
	historical_df: pd.DataFrame,
	future_df: pd.DataFrame,
	feature_names: list[str],
	mean: np.ndarray,
	std: np.ndarray,
	window: int,
	model: EntityEmbeddingGRU,
	device: torch.device,
) -> pd.DataFrame:
	combined = pd.concat([historical_df.copy(), future_df.copy()], ignore_index=True)
	combined = combined.sort_values(["province", "year"], kind="stable").reset_index(drop=True)
	combined["pred_residual"] = np.nan
	combined["pred_log_co2"] = np.nan
	combined["pred_co2"] = np.nan

	future_years = sorted(int(x) for x in future_df["year"].unique().tolist())

	with torch.no_grad():
		for province, group in combined.groupby("province", sort=False):
			group = group.sort_values("year", kind="stable").reset_index(drop=True)
			province_idx_values = group["province_id"].astype(int).unique().tolist()
			if len(province_idx_values) != 1:
				raise ValueError(f"Province has inconsistent province_id values: {province_idx_values}")
			province_idx = int(province_idx_values[0])

			residual_map: Dict[int, float] = {
				int(row.year): float(row.residual)
				for row in group.itertuples(index=False)
				if int(row.year) < future_years[0] and pd.notna(row.residual)
			}

			for year in future_years:
				seq_years = list(range(year - window, year))
				seq = group[group["year"].isin(seq_years)].sort_values("year", kind="stable")
				if len(seq) != window:
					raise RuntimeError(
						f"Province {province} lacks a full {window}-year sequence for year {year}. Check historical continuity."
					)

				dyn_std = standardize_dynamic_features(seq, feature_names, mean, std)
				lag_res = np.asarray([[residual_map[int(y)]] for y in seq_years], dtype=np.float32)

				x_dyn = torch.tensor(dyn_std[None, :, :], dtype=torch.float32, device=device)
				x_lag = torch.tensor(lag_res[None, :, :], dtype=torch.float32, device=device)
				x_pid = torch.tensor([province_idx], dtype=torch.long, device=device)

				res_hat = float(model(x_dyn, x_lag, x_pid).cpu().item())
				residual_map[year] = res_hat

				mask = (combined["province"] == province) & (combined["year"] == year)
				stirpat_log = float(combined.loc[mask, "stirpat_log_pred"].iloc[0])
				combined.loc[mask, "pred_residual"] = res_hat
				combined.loc[mask, "pred_log_co2"] = stirpat_log + res_hat
				combined.loc[mask, "pred_co2"] = float(np.exp(stirpat_log + res_hat))

	return combined


def predict_future_panel(
	historical_df: pd.DataFrame,
	future_df: pd.DataFrame,
	feature_names: list[str],
	mean: np.ndarray,
	std: np.ndarray,
	window: int,
	residual_model: EntityEmbeddingGRU,
	stirpat_model: PanelRidgeSTIRPAT,
	device: torch.device,
) -> pd.DataFrame:
	future_df = future_df.copy()
	future_df["stirpat_log_pred"] = stirpat_model.predict(future_df, province_col="province")
	future_df["stirpat_co2_pred"] = np.exp(future_df["stirpat_log_pred"].to_numpy(dtype=float))
	combined = recursive_future_residuals(
		historical_df=historical_df,
		future_df=future_df,
		feature_names=feature_names,
		mean=mean,
		std=std,
		window=window,
		model=residual_model,
		device=device,
	)

	future_only = combined[combined["year"] >= future_df["year"].min()].copy()
	future_only["pred_log_co2"] = future_only["pred_log_co2"].astype(float)
	future_only["pred_co2"] = future_only["pred_co2"].astype(float)
	return future_only.sort_values(["province", "year"], kind="stable").reset_index(drop=True)


def summarize_national(future_df: pd.DataFrame, historical_df: pd.DataFrame, scenario_name: str) -> pd.DataFrame:
	actual_2023 = historical_df[historical_df["year"] == 2023]["CO2"].sum()
	rows = [
		{
			"scenario": scenario_name,
			"year": 2023,
			"national_co2": float(actual_2023),
			"is_actual": True,
		},
	]
	for year, group in future_df.groupby("year", sort=True):
		rows.append(
			{
				"scenario": scenario_name,
				"year": int(year),
				"national_co2": float(group["pred_co2"].sum()),
				"is_actual": False,
			}
		)
	return pd.DataFrame(rows)


def peak_info(summary_df: pd.DataFrame) -> Dict[str, Any]:
	future_only = summary_df[summary_df["year"] >= 2024].copy()
	if future_only.empty:
		return {"peak_year": None, "peak_value": None, "peaked_within_forecast": False}
	peak_idx = future_only["national_co2"].idxmax()
	peak_year = int(summary_df.loc[peak_idx, "year"])
	peak_value = float(summary_df.loc[peak_idx, "national_co2"])
	return {
		"peak_year": peak_year,
		"peak_value": peak_value,
		"peaked_within_forecast": peak_year < int(future_only["year"].max()),
	}


def plot_national_comparison(summary_map: Dict[str, pd.DataFrame], output_path: Path) -> None:
	fig, ax = plt.subplots(figsize=(12.5, 6.8))
	colors = {
		"baseline": "#1f77b4",
		"low_carbon": "#2ca02c",
		"extensive": "#d62728",
	}
	labels = {
		"baseline": "Baseline",
		"low_carbon": "Low-carbon",
		"extensive": "Extensive",
	}
	for scenario_name, summary_df in summary_map.items():
		plot_df = summary_df.copy()
		ax.plot(
			plot_df["year"],
			plot_df["national_co2"],
			color=colors.get(scenario_name, "#444444"),
			linewidth=2.4,
			marker="o",
			markersize=4,
			label=labels.get(scenario_name, scenario_name),
		)
		info = peak_info(plot_df)
		if info["peak_year"] is not None:
			peak_row = plot_df[plot_df["year"] == info["peak_year"]].iloc[0]
			ax.scatter(
				[peak_row["year"]],
				[peak_row["national_co2"]],
				color=colors.get(scenario_name, "#444444"),
				s=70,
				zorder=5,
			)

	ax.axvline(2023, color="#666666", linestyle="--", linewidth=1.0, alpha=0.75)
	ax.text(2023 + 0.1, ax.get_ylim()[0], "2023 base", fontsize=9, color="#555555")
	ax.set_title("National CO2 Projection by Scenario (2024-2035)", fontsize=14)
	ax.set_xlabel("Year")
	ax.set_ylabel("National CO2")
	ax.grid(True, alpha=0.25, linestyle="--")
	ax.legend(loc="best")
	fig.tight_layout()
	fig.savefig(output_path, dpi=300)
	plt.close(fig)


def run_scenario(
	historical_df: pd.DataFrame,
	residual_model: EntityEmbeddingGRU,
	stirpat_model: PanelRidgeSTIRPAT,
	feature_names: list[str],
	mean: np.ndarray,
	std: np.ndarray,
	window: int,
	device: torch.device,
	scenario_name: str,
	start_year: int,
	end_year: int,
	output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	future_panel = build_future_panel(historical_df, scenario_name=scenario_name, start_year=start_year, end_year=end_year)
	predicted = predict_future_panel(
		historical_df=historical_df,
		future_df=future_panel,
		feature_names=feature_names,
		mean=mean,
		std=std,
		window=window,
		residual_model=residual_model,
		stirpat_model=stirpat_model,
		device=device,
	)

	summary = summarize_national(predicted, historical_df=historical_df, scenario_name=scenario_name)
	peak = peak_info(summary)
	peak["scenario"] = scenario_name
	peak["national_co2_at_peak"] = peak.pop("peak_value")
	peak["forecast_start_year"] = start_year
	peak["forecast_end_year"] = end_year
	peak["province_count"] = int(predicted["province"].nunique())

	output_dir.mkdir(parents=True, exist_ok=True)
	predicted.to_csv(output_dir / f"future_provincial_projection_{scenario_name}.csv", index=False, encoding="utf-8")
	summary.to_csv(output_dir / f"national_projection_{scenario_name}.csv", index=False, encoding="utf-8")
	(output_dir / f"peak_summary_{scenario_name}.json").write_text(json.dumps(peak, ensure_ascii=False, indent=2), encoding="utf-8")
	return predicted, summary


def main() -> None:
	args = build_parser().parse_args()
	cfg = ProjectionConfig(
		input_csv=args.input_csv,
		dataset_npz=args.dataset_npz,
		model_ckpt=args.model_ckpt,
		output_dir=args.output_dir,
		start_year=args.start_year,
		end_year=args.end_year,
		train_end_year=args.train_end_year,
		ridge_alpha=args.ridge_alpha,
		scenario=args.scenario,
		device=args.device,
	)

	if not cfg.dataset_npz.exists():
		raise FileNotFoundError(f"Dataset NPZ not found: {cfg.dataset_npz}")
	if not cfg.model_ckpt.exists():
		raise FileNotFoundError(f"Model checkpoint not found: {cfg.model_ckpt}")

	historical_df = load_historical_panel(cfg.input_csv)
	feature_names, mean, std, window = load_npz_metadata(cfg.dataset_npz)
	device = choose_device(cfg.device)
	residual_model = load_residual_model(cfg.model_ckpt, device=device)
	stirpat_model = fit_stirpat_model(historical_df, train_end_year=cfg.train_end_year, ridge_alpha=cfg.ridge_alpha)

	scenarios = [cfg.scenario] if cfg.scenario != "all" else ["baseline", "low_carbon", "extensive"]
	all_summary: Dict[str, pd.DataFrame] = {}
	peak_payload: Dict[str, Dict[str, Any]] = {}
	combined_future_frames: List[pd.DataFrame] = []

	for scenario_name in scenarios:
		predicted, summary = run_scenario(
			historical_df=historical_df,
			residual_model=residual_model,
			stirpat_model=stirpat_model,
			feature_names=feature_names,
			mean=mean,
			std=std,
			window=window,
			device=device,
			scenario_name=scenario_name,
			start_year=cfg.start_year,
			end_year=cfg.end_year,
			output_dir=cfg.output_dir,
		)
		all_summary[scenario_name] = summary
		combined_future_frames.append(predicted)
		peak_payload[scenario_name] = json.loads((cfg.output_dir / f"peak_summary_{scenario_name}.json").read_text(encoding="utf-8"))

	combined_future = pd.concat(combined_future_frames, ignore_index=True) if combined_future_frames else pd.DataFrame()
	combined_future.to_csv(cfg.output_dir / "future_provincial_projection_all_scenarios.csv", index=False, encoding="utf-8")
	combined_national = pd.concat(all_summary.values(), ignore_index=True) if all_summary else pd.DataFrame()
	combined_national.to_csv(cfg.output_dir / "national_projection_all_scenarios.csv", index=False, encoding="utf-8")
	(cfg.output_dir / "scenario_peak_summary.json").write_text(json.dumps(peak_payload, ensure_ascii=False, indent=2), encoding="utf-8")

	if cfg.scenario == "all":
		plot_national_comparison(all_summary, cfg.output_dir / "national_projection_comparison.png")
	else:
		plot_national_comparison({cfg.scenario: all_summary[cfg.scenario]}, cfg.output_dir / f"national_projection_{cfg.scenario}.png")

	print("Saved provincial future projections and national summaries to:", cfg.output_dir)
	for scenario_name, summary in all_summary.items():
		info = peak_info(summary)
		peak_value = info["peak_value"]
		peak_value_text = f"{peak_value:.3f}" if peak_value is not None else "None"
		print(
			f"[{scenario_name}] peak_year={info['peak_year']}, "
			f"peak_value={peak_value_text}"
		)


if __name__ == "__main__":
	main()