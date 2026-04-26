from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
	sys.path.insert(0, str(SCRIPT_DIR))

from stirpat_ee_gru import EntityEmbeddingGRU


@dataclass
class TrainConfig:
	"""训练阶段超参数与输出路径配置。"""

	dataset_npz: Path
	model_out: Path
	metrics_out: Path
	batch_size: int = 8
	epochs: int = 100
	lr: float = 1e-3
	weight_decay: float = 1e-4
	embed_dim: int = 8
	hidden_dim: int = 32
	dropout: float = 0.2
	patience: int = 15
	max_grad_norm: float = 1.0
	seed: int = 42


class ResidualDataset(Dataset):
	"""从 NPZ 读取残差学习数据并提供 PyTorch 索引访问。"""

	def __init__(self, npz_data: Any, split: str):
		"""按数据切分名加载对应张量。"""
		self.dynamic_x = torch.tensor(npz_data[f"{split}_dynamic_x"], dtype=torch.float32)
		self.lag_residual_x = torch.tensor(npz_data[f"{split}_lag_residual_x"], dtype=torch.float32)
		self.province_idx = torch.tensor(npz_data[f"{split}_province_idx"], dtype=torch.long)
		self.target_residual = torch.tensor(npz_data[f"{split}_target_residual"], dtype=torch.float32)

	def __len__(self) -> int:
		"""返回样本数量。"""
		return self.dynamic_x.size(0)

	def __getitem__(self, idx: int):
		"""按索引返回单条训练样本。"""
		return (
			self.dynamic_x[idx],
			self.lag_residual_x[idx],
			self.province_idx[idx],
			self.target_residual[idx],
		)


def set_seed(seed: int) -> None:
	"""设置随机种子以提升训练可复现性。"""
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def train(
	model: nn.Module,
	loader: DataLoader,
	optimizer: torch.optim.Optimizer,
	criterion: nn.Module,
	device: torch.device,
	max_grad_norm: float = 1.0,
) -> Dict[str, float]:
	"""执行一个训练 epoch，并返回平均损失与 MAPE。"""
	model.train()
	loss_sum = 0.0
	abs_pct_sum = 0.0
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
		loss_sum += loss.item() * bs
		abs_pct_sum += torch.mean(torch.abs((pred - target) / torch.clamp(torch.abs(target), min=1e-6))).item() * bs
		n += bs

	return {
		"loss": loss_sum / max(n, 1),
		"mape": abs_pct_sum / max(n, 1),
	}


@torch.no_grad()
def valid(
	model: nn.Module,
	loader: DataLoader,
	criterion: nn.Module,
	device: torch.device,
) -> Dict[str, float]:
	"""执行一个验证 epoch，并返回平均损失与 MAPE。"""
	model.eval()
	loss_sum = 0.0
	abs_pct_sum = 0.0
	n = 0

	for dynamic_x, lag_residual_x, province_idx, target in loader:
		dynamic_x = dynamic_x.to(device)
		lag_residual_x = lag_residual_x.to(device)
		province_idx = province_idx.to(device)
		target = target.to(device)

		pred = model(dynamic_x, lag_residual_x, province_idx)
		loss = criterion(pred, target)

		bs = target.size(0)
		loss_sum += loss.item() * bs
		abs_pct_sum += torch.mean(torch.abs((pred - target) / torch.clamp(torch.abs(target), min=1e-6))).item() * bs
		n += bs

	return {
		"loss": loss_sum / max(n, 1),
		"mape": abs_pct_sum / max(n, 1),
	}


def build_dataloaders(npz_path: Path, batch_size: int) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
	"""构建训练/验证 DataLoader，并返回模型形状信息。"""
	npz_data = np.load(npz_path, allow_pickle=True)
	train_ds = ResidualDataset(npz_data, split="train")
	valid_ds = ResidualDataset(npz_data, split="valid")

	if len(train_ds) == 0:
		raise RuntimeError("Train split is empty. Please rebuild dataset with wider train years.")
	if len(valid_ds) == 0:
		raise RuntimeError("Valid split is empty. Please rebuild dataset with valid years in data range.")

	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
	valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, drop_last=False)

	num_dynamic_features = int(train_ds.dynamic_x.shape[-1])
	num_provinces = int(max(train_ds.province_idx.max().item(), valid_ds.province_idx.max().item()) + 1)
	shape_info = {
		"num_dynamic_features": num_dynamic_features,
		"num_provinces": num_provinces,
		"train_samples": len(train_ds),
		"valid_samples": len(valid_ds),
	}
	return train_loader, valid_loader, shape_info


def run_training(cfg: TrainConfig) -> None:
	"""完整训练流程：加载数据、训练迭代、早停、保存最优模型。"""
	set_seed(cfg.seed)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	train_loader, valid_loader, shape_info = build_dataloaders(cfg.dataset_npz, cfg.batch_size)
	model = EntityEmbeddingGRU(
		num_provinces=shape_info["num_provinces"],
		num_dynamic_features=shape_info["num_dynamic_features"],
		embed_dim=cfg.embed_dim,
		hidden_dim=cfg.hidden_dim,
		dropout=cfg.dropout,
	).to(device)

	criterion = nn.HuberLoss(delta=1.0)
	optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
	scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)

	best_valid = float("inf")
	best_epoch = -1
	patience_count = 0
	history = []

	for epoch in range(1, cfg.epochs + 1):
		train_metrics = train(
			model=model,
			loader=train_loader,
			optimizer=optimizer,
			criterion=criterion,
			device=device,
			max_grad_norm=cfg.max_grad_norm,
		)
		valid_metrics = valid(model=model, loader=valid_loader, criterion=criterion, device=device)
		scheduler.step()

		record = {
			"epoch": epoch,
			"train_loss": train_metrics["loss"],
			"train_mape": train_metrics["mape"],
			"valid_loss": valid_metrics["loss"],
			"valid_mape": valid_metrics["mape"],
		}
		history.append(record)

		if valid_metrics["loss"] < best_valid:
			best_valid = valid_metrics["loss"]
			best_epoch = epoch
			patience_count = 0
			cfg.model_out.parent.mkdir(parents=True, exist_ok=True)
			torch.save(
				{
					"model_state_dict": model.state_dict(),
					"shape_info": shape_info,
					"train_config": asdict(cfg),
				},
				cfg.model_out,
			)
		else:
			patience_count += 1

		print(
			f"Epoch {epoch:03d} | "
			f"train_loss={train_metrics['loss']:.6f} valid_loss={valid_metrics['loss']:.6f} "
			f"train_mape={train_metrics['mape']:.6f} valid_mape={valid_metrics['mape']:.6f}"
		)

		if patience_count >= cfg.patience:
			print(f"Early stopping at epoch {epoch}, best epoch = {best_epoch}.")
			break

	metrics = {
		"best_valid_loss": best_valid,
		"best_epoch": best_epoch,
		"shape_info": shape_info,
		"history": history,
		"train_config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
	}
	cfg.metrics_out.parent.mkdir(parents=True, exist_ok=True)
	cfg.metrics_out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
	print(f"Saved best model: {cfg.model_out}")
	print(f"Saved training metrics: {cfg.metrics_out}")


def parse_args() -> TrainConfig:
	"""解析命令行参数并返回训练配置对象。"""
	parser = argparse.ArgumentParser(description="Train EE-GRU residual model with train/valid loops.")
	base_dir = SCRIPT_DIR
	default_npz = base_dir / "output" / "dataset" / "stirpat_ee_gru_dataset.npz"
	default_model = base_dir / "output" / "model" / "best_ee_gru.pt"
	default_metrics = base_dir / "output" / "model" / "train_metrics.json"

	parser.add_argument("--dataset-npz", type=Path, default=default_npz)
	parser.add_argument("--model-out", type=Path, default=default_model)
	parser.add_argument("--metrics-out", type=Path, default=default_metrics)
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
	return TrainConfig(
		dataset_npz=args.dataset_npz,
		model_out=args.model_out,
		metrics_out=args.metrics_out,
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


if __name__ == "__main__":
	run_training(parse_args())
