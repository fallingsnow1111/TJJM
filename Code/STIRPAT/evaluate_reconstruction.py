from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
	sys.path.insert(0, str(SCRIPT_DIR))

from stirpat_ee_gru import EntityEmbeddingGRU


@dataclass
class EvalConfig:
	"""重构评估参数配置。"""

	dataset_npz: Path
	model_ckpt: Path
	output_json: Path
	output_detail_dir: Path
	batch_size: int = 3


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
	"""计算带下限保护的 MAPE，避免分母接近 0 导致发散。"""
	den = np.maximum(np.abs(y_true), eps)
	return float(np.mean(np.abs((y_pred - y_true) / den)))


def _smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
	"""计算 sMAPE，相比 MAPE 对小目标值更稳定。"""
	den = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
	return float(np.mean(2.0 * np.abs(y_pred - y_true) / den))


def _reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
	"""统一计算回归误差指标。"""
	err = y_pred - y_true
	mae = float(np.mean(np.abs(err)))
	rmse = float(np.sqrt(np.mean(err**2)))
	mape = _safe_mape(y_true, y_pred)
	smape = _smape(y_true, y_pred)

	ss_res = float(np.sum((y_true - y_pred) ** 2))
	ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
	r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

	return {
		"mae": mae,
		"rmse": rmse,
		"mape": mape,
		"smape": smape,
		"r2": r2,
	}


def _predict_residual(
	model: EntityEmbeddingGRU,
	dynamic_x: np.ndarray,
	lag_residual_x: np.ndarray,
	province_idx: np.ndarray,
	device: torch.device,
	batch_size: int,
) -> np.ndarray:
	"""分批预测残差，避免一次性推理占用过高显存。"""
	model.eval()
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

def evaluate_split(
    split: str,
    npz_data: Any,
    model: EntityEmbeddingGRU,
    device: torch.device,
    batch_size: int,
    output_detail_dir: Path,
) -> Dict[str, Dict[str, float]]:
    """评估单个切分：比较 STIRPAT 基线与融合重构的能源消费结果。"""
    required_keys = [
        f"{split}_dynamic_x",
        f"{split}_lag_residual_x",
        f"{split}_province_idx",
        f"{split}_target_log_energy",          # 改
        f"{split}_target_stirpat_log",
        f"{split}_target_year",
    ]
    for k in required_keys:
        if k not in npz_data:
            raise KeyError(f"Missing key in dataset npz: {k}")

    dynamic_x = npz_data[f"{split}_dynamic_x"]
    lag_residual_x = npz_data[f"{split}_lag_residual_x"]
    province_idx = npz_data[f"{split}_province_idx"]
    true_log_energy = npz_data[f"{split}_target_log_energy"]   # 改
    stirpat_log_energy = npz_data[f"{split}_target_stirpat_log"]  # 实际是 log_Energy 的 STIRPAT 预测
    years = npz_data[f"{split}_target_year"]

    pred_residual = _predict_residual(
        model=model,
        dynamic_x=dynamic_x,
        lag_residual_x=lag_residual_x,
        province_idx=province_idx,
        device=device,
        batch_size=batch_size,
    )

    hybrid_log_energy = stirpat_log_energy + pred_residual

    # 转换到能源消费总量（亿吨标准煤等原始单位）
    true_energy = np.exp(true_log_energy)
    stirpat_energy = np.exp(stirpat_log_energy)
    hybrid_energy = np.exp(hybrid_log_energy)

    metrics = {
        "stirpat_only": _reg_metrics(true_energy, stirpat_energy),
        "hybrid_reconstruct": _reg_metrics(true_energy, hybrid_energy),
    }

    output_detail_dir.mkdir(parents=True, exist_ok=True)
    detail_df = pd.DataFrame(
        {
            "split": split,
            "year": years.astype(int),
            "province_idx": province_idx.astype(int),
            "true_log_energy": true_log_energy,
            "stirpat_log_energy_pred": stirpat_log_energy,
            "residual_pred": pred_residual,
            "hybrid_log_energy_pred": hybrid_log_energy,
            "true_energy": true_energy,
            "stirpat_energy_pred": stirpat_energy,
            "hybrid_energy_pred": hybrid_energy,
        }
    )
    detail_df["stirpat_abs_pct_err"] = np.abs(
        (detail_df["stirpat_energy_pred"] - detail_df["true_energy"])
        / np.maximum(np.abs(detail_df["true_energy"]), 1e-6)
    )
    detail_df["hybrid_abs_pct_err"] = np.abs(
        (detail_df["hybrid_energy_pred"] - detail_df["true_energy"])
        / np.maximum(np.abs(detail_df["true_energy"]), 1e-6)
    )
    detail_path = output_detail_dir / f"{split}_energy_reconstruction_detail.csv"
    detail_df.to_csv(detail_path, index=False, encoding="utf-8")

    return metrics

def load_model(cfg: EvalConfig, device: torch.device) -> EntityEmbeddingGRU:
	"""从训练输出的 checkpoint 还原模型结构和参数。"""
	ckpt = torch.load(cfg.model_ckpt, map_location=device, weights_only=False)
	shape_info = ckpt["shape_info"]
	train_cfg = ckpt.get("train_config", {})

	embed_dim = int(train_cfg.get("embed_dim", 8))
	hidden_dim = int(train_cfg.get("hidden_dim", 32))
	dropout = float(train_cfg.get("dropout", 0.2))

	model = EntityEmbeddingGRU(
		num_provinces=int(shape_info["num_provinces"]),
		num_dynamic_features=int(shape_info["num_dynamic_features"]),
		embed_dim=embed_dim,
		hidden_dim=hidden_dim,
		dropout=dropout,
	).to(device)
	model.load_state_dict(ckpt["model_state_dict"])
	return model


def parse_args() -> EvalConfig:
	"""解析命令行参数并构造评估配置。"""
	parser = argparse.ArgumentParser(description="Evaluate final CO2 reconstruction: STIRPAT + residual model.")
	base_dir = SCRIPT_DIR
	parser.add_argument(
		"--dataset-npz",
		type=Path,
		default=base_dir / "output" / "dataset" / "stirpat_ee_gru_dataset.npz",
	)
	parser.add_argument(
		"--model-ckpt",
		type=Path,
		default=base_dir / "output" / "model" / "best_ee_gru.pt",
	)
	parser.add_argument(
		"--output-json",
		type=Path,
		default=base_dir / "output" / "model" / "reconstruction_metrics.json",
	)
	parser.add_argument(
		"--output-detail-dir",
		type=Path,
		default=base_dir / "output" / "model",
	)
	parser.add_argument("--batch-size", type=int, default=256)

	args = parser.parse_args()
	return EvalConfig(
		dataset_npz=args.dataset_npz,
		model_ckpt=args.model_ckpt,
		output_json=args.output_json,
		output_detail_dir=args.output_detail_dir,
		batch_size=args.batch_size,
	)


def main() -> None:
	"""评估主入口：重构 CO2 并输出误差对比结果。"""
	cfg = parse_args()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	npz_data = np.load(cfg.dataset_npz, allow_pickle=True)
	model = load_model(cfg, device=device)

	all_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
	for split in ["train", "valid"]:
		all_metrics[split] = evaluate_split(
			split=split,
			npz_data=npz_data,
			model=model,
			device=device,
			batch_size=cfg.batch_size,
			output_detail_dir=cfg.output_detail_dir,
		)

	payload = {
		"config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
		"metrics": all_metrics,
	}
	cfg.output_json.parent.mkdir(parents=True, exist_ok=True)
	cfg.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

	print("=== Reconstruction Metrics (Energy scale) ===")
	for split, split_metric in all_metrics.items():
		base = split_metric["stirpat_only"]
		hybrid = split_metric["hybrid_reconstruct"]
		print(
			f"[{split}] "
			f"STIRPAT rmse={base['rmse']:.6f}, mape={base['mape']:.6f} | "
			f"HYBRID rmse={hybrid['rmse']:.6f}, mape={hybrid['mape']:.6f}"
		)

	print(f"Saved metrics json: {cfg.output_json}")
	print(f"Saved detail csvs in: {cfg.output_detail_dir}")


if __name__ == "__main__":
	main()
