from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch
from torch import nn


@dataclass
class PanelRidgeResult:
	"""保存面板岭回归结果：系数、特征列顺序、省份类别顺序。"""

	beta: np.ndarray
	feature_cols: List[str]
	province_levels: List[str]


class PanelRidgeSTIRPAT:
	"""在对数 STIRPAT 特征空间上执行带省份固定效应的岭回归。"""

	def __init__(self, alpha: float = 1.0):
		"""初始化回归器。

		Args:
			alpha: 岭惩罚系数，仅作用于数值特征部分。
		"""
		self.alpha = alpha
		self.result: PanelRidgeResult | None = None

	def _build_design(
		self,
		df: pd.DataFrame,
		feature_cols: Iterable[str],
		province_col: str,
		province_levels: List[str] | None = None,
	) -> tuple[np.ndarray, List[str], List[str], int]:
		"""构建设计矩阵：数值特征 + 省份虚拟变量。

		Returns:
			x: 合并后的设计矩阵。
			feature_cols: 数值特征名列表。
			province_levels: 省份类别顺序。
			n_num: 数值特征列数。
		"""
		feature_cols = list(feature_cols)
		x_num = df[feature_cols].to_numpy(dtype=np.float64)

		if province_levels is None:
			province_levels = sorted(df[province_col].astype(str).unique().tolist())

		cat = pd.Categorical(df[province_col].astype(str), categories=province_levels)
		x_prov = pd.get_dummies(cat, drop_first=False).to_numpy(dtype=np.float64)
		x = np.concatenate([x_num, x_prov], axis=1)
		return x, feature_cols, province_levels, x_num.shape[1]

	def fit(
		self,
		df: pd.DataFrame,
		feature_cols: Iterable[str],
		target_col: str = "log_CO2",
		province_col: str = "province",
	) -> None:
		"""拟合面板岭回归参数。"""
		x, feat, province_levels, n_num = self._build_design(df, feature_cols, province_col)
		y = df[target_col].to_numpy(dtype=np.float64)

		xtx = x.T @ x
		reg = np.eye(xtx.shape[0], dtype=np.float64)
		reg[n_num:, n_num:] = 0.0
		beta = np.linalg.solve(xtx + self.alpha * reg, x.T @ y)

		self.result = PanelRidgeResult(beta=beta, feature_cols=feat, province_levels=province_levels)

	def predict(self, df: pd.DataFrame, province_col: str = "province") -> np.ndarray:
		"""使用已拟合参数对输入样本进行预测。"""
		if self.result is None:
			raise RuntimeError("PanelRidgeSTIRPAT must be fit before predict.")
		x, _, _, _ = self._build_design(
			df=df,
			feature_cols=self.result.feature_cols,
			province_col=province_col,
			province_levels=self.result.province_levels,
		)
		return x @ self.result.beta


class EntityEmbeddingGRU(nn.Module):
	"""实体嵌入 + GRU 的残差预测网络。"""

	def __init__(
		self,
		num_provinces: int,
		num_dynamic_features: int,
		embed_dim: int = 8,
		hidden_dim: int = 32,
		dropout: float = 0.2,
	):
		"""初始化网络结构。

		Args:
			num_provinces: 省份类别数量。
			num_dynamic_features: 时序动态特征维度。
			embed_dim: 省份嵌入向量维度。
			hidden_dim: GRU 隐状态维度。
			dropout: 输出层前的失活比例。
		"""
		super().__init__()
		self.embedding = nn.Embedding(num_embeddings=num_provinces, embedding_dim=embed_dim)
		self.gru = nn.GRU(
			input_size=num_dynamic_features + 1 + embed_dim,
			hidden_size=hidden_dim,
			num_layers=1,
			batch_first=True,
		)
		self.dropout = nn.Dropout(p=dropout)
		self.head = nn.Linear(hidden_dim, 1)

	def forward(
		self,
		dynamic_x: torch.Tensor,
		lag_residual_x: torch.Tensor,
		province_idx: torch.Tensor,
	) -> torch.Tensor:
		"""前向传播并输出每条样本在目标时点的残差预测值。"""
		# dynamic_x: [B, T, F], lag_residual_x: [B, T, 1], province_idx: [B]
		emb = self.embedding(province_idx)
		emb_t = emb.unsqueeze(1).expand(-1, dynamic_x.size(1), -1)
		x = torch.cat([dynamic_x, lag_residual_x, emb_t], dim=-1)

		out, _ = self.gru(x)
		last = out[:, -1, :]
		pred = self.head(self.dropout(last)).squeeze(-1)
		return pred