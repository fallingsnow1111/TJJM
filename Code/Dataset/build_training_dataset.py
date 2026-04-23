from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Ensure imports work when running as: python Code/Dataset/build_training_dataset.py
CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
	sys.path.insert(0, str(CODE_ROOT))

from Model.stirpat_ee_gru import PanelRidgeSTIRPAT


@dataclass
class BuildConfig:
	"""数据集构建参数配置。"""

	input_csv: Path
	output_npz: Path
	output_panel_csv: Path
	output_meta_json: Path
	window: int = 3
	train_end_year: int = 2020
	valid_start_year: int = 2021
	valid_end_year: int = 2023
	ridge_alpha: float = 1.0


class Standardizer:
	"""简易标准化器：保存均值和标准差并执行变换。"""

	def __init__(self) -> None:
		"""初始化标准化器状态。"""
		self.mean_: np.ndarray | None = None
		self.std_: np.ndarray | None = None

	def fit(self, x: np.ndarray) -> None:
		"""在训练样本上拟合均值和标准差。"""
		self.mean_ = x.mean(axis=0)
		std = x.std(axis=0)
		std[std < 1e-8] = 1.0
		self.std_ = std

	def transform(self, x: np.ndarray) -> np.ndarray:
		"""使用已拟合统计量对输入数组进行标准化。"""
		if self.mean_ is None or self.std_ is None:
			raise RuntimeError("Standardizer must be fit before transform.")
		return (x - self.mean_) / self.std_


def prepare_dataframe(input_csv: Path) -> pd.DataFrame:
	"""读取面板 CSV 并完成基础校验与特征工程。"""
	df = pd.read_csv(input_csv)

	required_cols = {
		"province",
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
	}
	missing = required_cols - set(df.columns)
	if missing:
		raise ValueError(f"Missing required columns: {sorted(missing)}")

	df = df.copy()
	df["province"] = df["province"].astype(str)
	df["year"] = df["year"].astype(int)
	df = df.sort_values(["province", "year"], kind="stable").reset_index(drop=True)

	# Feature engineering aligned to STIRPAT guidance.
	df["pGDP"] = df["GDP"] / df["Population"]
	df["CarbonIntensity"] = df["CO2"] / df["GDP"]

	for col in ["CO2", "GDP", "Population", "Energy", "PrivateCars", "pGDP", "CarbonIntensity"]:
		df[f"log_{col}"] = np.log(df[col].clip(lower=1e-8))

	province_levels = sorted(df["province"].unique().tolist())
	mapping = {p: i for i, p in enumerate(province_levels)}
	df["province_id"] = df["province"].map(mapping).astype(int)

	return df


def build_residual_target(
	df: pd.DataFrame,
	ridge_alpha: float,
	train_end_year: int,
) -> tuple[pd.DataFrame, List[str]]:
	"""拟合 STIRPAT 面板岭回归并生成残差目标。"""
	stirpat_cols = [
		"log_Population",
		"log_pGDP",
		"Industry",
		"Urbanization",
		"CoalShare",
		"log_CarbonIntensity",
		"log_Energy",
		"log_PrivateCars",
	]

	model = PanelRidgeSTIRPAT(alpha=ridge_alpha)
	train_mask = df["year"] <= train_end_year
	model.fit(df.loc[train_mask], feature_cols=stirpat_cols, target_col="log_CO2", province_col="province")

	df = df.copy()
	df["stirpat_log_pred"] = model.predict(df, province_col="province")
	df["residual"] = df["log_CO2"] - df["stirpat_log_pred"]
	return df, stirpat_cols


def make_windows(df: pd.DataFrame, cfg: BuildConfig) -> Dict[str, np.ndarray]:
	"""将面板数据重构为滑动窗口三维张量。"""
	gru_cols = [
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

	standardizer = Standardizer()
	train_rows = df[df["year"] <= cfg.train_end_year]
	standardizer.fit(train_rows[gru_cols].to_numpy(dtype=np.float32))
	df_std = df.copy()
	df_std[gru_cols] = standardizer.transform(df_std[gru_cols].to_numpy(dtype=np.float32))

	dynamic_list: List[np.ndarray] = []
	lag_res_list: List[np.ndarray] = []
	province_list: List[int] = []
	target_res_list: List[float] = []
	target_log_co2_list: List[float] = []
	target_stirpat_log_list: List[float] = []
	target_year_list: List[int] = []

	for _, grp in df_std.groupby("province", sort=False):
		g = grp.sort_values("year", kind="stable").reset_index(drop=True)
		years = g["year"].to_numpy(dtype=int)

		for end_idx in range(cfg.window, len(g)):
			start_idx = end_idx - cfg.window
			time_slice = years[start_idx : end_idx + 1]
			if np.any(np.diff(time_slice) != 1):
				continue

			dynamic_seq = g.loc[start_idx : end_idx - 1, gru_cols].to_numpy(dtype=np.float32)
			lag_res_seq = g.loc[start_idx : end_idx - 1, ["residual"]].to_numpy(dtype=np.float32)

			target_row = g.iloc[end_idx]
			dynamic_list.append(dynamic_seq)
			lag_res_list.append(lag_res_seq)
			province_list.append(int(target_row["province_id"]))
			target_res_list.append(float(target_row["residual"]))
			target_log_co2_list.append(float(target_row["log_CO2"]))
			target_stirpat_log_list.append(float(target_row["stirpat_log_pred"]))
			target_year_list.append(int(target_row["year"]))

	if not dynamic_list:
		raise RuntimeError("No sliding-window samples generated. Check year continuity and window size.")
	if standardizer.mean_ is None or standardizer.std_ is None:
		raise RuntimeError("Standardizer state is missing after fit.")

	pack = {
		"dynamic_x": np.stack(dynamic_list, axis=0),
		"lag_residual_x": np.stack(lag_res_list, axis=0),
		"province_idx": np.asarray(province_list, dtype=np.int64),
		"target_residual": np.asarray(target_res_list, dtype=np.float32),
		"target_log_co2": np.asarray(target_log_co2_list, dtype=np.float32),
		"target_stirpat_log": np.asarray(target_stirpat_log_list, dtype=np.float32),
		"target_year": np.asarray(target_year_list, dtype=np.int32),
		"gru_feature_names": np.asarray(gru_cols, dtype=object),
		"standardizer_mean": standardizer.mean_.astype(np.float32),
		"standardizer_std": standardizer.std_.astype(np.float32),
	}
	return pack


def split_and_save(pack: Dict[str, np.ndarray], cfg: BuildConfig) -> None:
	"""按时间切分训练/验证集并写入压缩 NPZ。"""
	years = pack["target_year"]
	train_mask = years <= cfg.train_end_year
	valid_mask = (years >= cfg.valid_start_year) & (years <= cfg.valid_end_year)

	save_payload: Dict[str, np.ndarray] = {}
	for split, mask in [("train", train_mask), ("valid", valid_mask)]:
		for key in [
			"dynamic_x",
			"lag_residual_x",
			"province_idx",
			"target_residual",
			"target_log_co2",
			"target_stirpat_log",
			"target_year",
		]:
			save_payload[f"{split}_{key}"] = pack[key][mask]

	# Keep metadata arrays in the NPZ package.
	save_payload["gru_feature_names"] = pack["gru_feature_names"]
	save_payload["standardizer_mean"] = pack["standardizer_mean"]
	save_payload["standardizer_std"] = pack["standardizer_std"]

	cfg.output_npz.parent.mkdir(parents=True, exist_ok=True)
	np.savez_compressed(cfg.output_npz, **save_payload)


def parse_args() -> BuildConfig:
	"""解析命令行参数并构造构建配置。"""
	parser = argparse.ArgumentParser(description="Build sliding-window residual dataset for STIRPAT-EE-GRU.")
	default_input = Path(__file__).resolve().parents[1] / "Preprocess" / "output" / "panel_master.csv"
	default_out_dir = Path(__file__).resolve().parent / "output"

	parser.add_argument("--input-csv", type=Path, default=default_input)
	parser.add_argument("--output-npz", type=Path, default=default_out_dir / "stirpat_ee_gru_dataset.npz")
	parser.add_argument("--output-panel-csv", type=Path, default=default_out_dir / "panel_with_residual.csv")
	parser.add_argument("--output-meta-json", type=Path, default=default_out_dir / "dataset_meta.json")
	parser.add_argument("--window", type=int, default=5)
	parser.add_argument("--train-end-year", type=int, default=2020)
	parser.add_argument("--valid-start-year", type=int, default=2021)
	parser.add_argument("--valid-end-year", type=int, default=2023)
	parser.add_argument("--ridge-alpha", type=float, default=1.0)

	args = parser.parse_args()
	return BuildConfig(
		input_csv=args.input_csv,
		output_npz=args.output_npz,
		output_panel_csv=args.output_panel_csv,
		output_meta_json=args.output_meta_json,
		window=args.window,
		train_end_year=args.train_end_year,
		valid_start_year=args.valid_start_year,
		valid_end_year=args.valid_end_year,
		ridge_alpha=args.ridge_alpha,
	)


def main() -> None:
	"""数据集构建主入口：读取、加工、切片、保存。"""
	cfg = parse_args()
	df = prepare_dataframe(cfg.input_csv)
	df, stirpat_cols = build_residual_target(
		df,
		ridge_alpha=cfg.ridge_alpha,
		train_end_year=cfg.train_end_year,
	)
	pack = make_windows(df, cfg)
	split_and_save(pack, cfg)

	cfg.output_panel_csv.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(cfg.output_panel_csv, index=False, encoding="utf-8")

	config_dict = asdict(cfg)
	config_dict = {k: str(v) if isinstance(v, Path) else v for k, v in config_dict.items()}

	meta = {
		"config": config_dict,
		"stirpat_feature_names": stirpat_cols,
		"num_provinces": int(df["province_id"].nunique()),
		"year_range": [int(df["year"].min()), int(df["year"].max())],
		"splits": {
			"train": f"<= {cfg.train_end_year}",
			"valid": f"{cfg.valid_start_year}-{cfg.valid_end_year}",
		},
	}
	cfg.output_meta_json.parent.mkdir(parents=True, exist_ok=True)
	cfg.output_meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

	print(f"Saved dataset NPZ: {cfg.output_npz}")
	print(f"Saved residual panel CSV: {cfg.output_panel_csv}")
	print(f"Saved dataset metadata: {cfg.output_meta_json}")


if __name__ == "__main__":
	main()
