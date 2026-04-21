from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from openpyxl import load_workbook


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "Dataset"
SUMMARY_JSON = PROJECT_ROOT / "Code" / "dataset_index_summary.json"
OUT_DIR = PROJECT_ROOT / "Code" / "output"


# Canonical province names aligned to inventory/MEIC naming.
PROVINCE_ALIAS = {
	"beijing": "Beijing",
	"tianjin": "Tianjin",
	"hebei": "Hebei",
	"shanxi": "Shanxi",
	"innermongolia": "InnerMongolia",
	"liaoning": "Liaoning",
	"jilin": "Jilin",
	"heilongjiang": "Heilongjiang",
	"shanghai": "Shanghai",
	"jiangsu": "Jiangsu",
	"zhejiang": "Zhejiang",
	"anhui": "Anhui",
	"fujian": "Fujian",
	"jiangxi": "Jiangxi",
	"shandong": "Shandong",
	"henan": "Henan",
	"hubei": "Hubei",
	"hunan": "Hunan",
	"guangdong": "Guangdong",
	"guangxi": "Guangxi",
	"hainan": "Hainan",
	"chongqing": "Chongqing",
	"sichuan": "Sichuan",
	"guizhou": "Guizhou",
	"yunnan": "Yunnan",
	"shaanxi": "Shaanxi",
	"gansu": "Gansu",
	"qinghai": "Qinghai",
	"ningxia": "Ningxia",
	"xinjiang": "Xinjiang",
	# Common variants
	"innermongoliaautonomousregion": "InnerMongolia",
	"innermongoliaregion": "InnerMongolia",
}

ZH_PROVINCE_ALIAS = {
	"北京": "Beijing",
	"北京市": "Beijing",
	"天津": "Tianjin",
	"天津市": "Tianjin",
	"河北": "Hebei",
	"河北省": "Hebei",
	"山西": "Shanxi",
	"山西省": "Shanxi",
	"内蒙古": "InnerMongolia",
	"内蒙古自治区": "InnerMongolia",
	"辽宁": "Liaoning",
	"辽宁省": "Liaoning",
	"吉林": "Jilin",
	"吉林省": "Jilin",
	"黑龙江": "Heilongjiang",
	"黑龙江省": "Heilongjiang",
	"上海": "Shanghai",
	"上海市": "Shanghai",
	"江苏": "Jiangsu",
	"江苏省": "Jiangsu",
	"浙江": "Zhejiang",
	"浙江省": "Zhejiang",
	"安徽": "Anhui",
	"安徽省": "Anhui",
	"福建": "Fujian",
	"福建省": "Fujian",
	"江西": "Jiangxi",
	"江西省": "Jiangxi",
	"山东": "Shandong",
	"山东省": "Shandong",
	"河南": "Henan",
	"河南省": "Henan",
	"湖北": "Hubei",
	"湖北省": "Hubei",
	"湖南": "Hunan",
	"湖南省": "Hunan",
	"广东": "Guangdong",
	"广东省": "Guangdong",
	"广西": "Guangxi",
	"广西壮族自治区": "Guangxi",
	"海南": "Hainan",
	"海南省": "Hainan",
	"重庆": "Chongqing",
	"重庆市": "Chongqing",
	"四川": "Sichuan",
	"四川省": "Sichuan",
	"贵州": "Guizhou",
	"贵州省": "Guizhou",
	"云南": "Yunnan",
	"云南省": "Yunnan",
	"陕西": "Shaanxi",
	"陕西省": "Shaanxi",
	"甘肃": "Gansu",
	"甘肃省": "Gansu",
	"青海": "Qinghai",
	"青海省": "Qinghai",
	"宁夏": "Ningxia",
	"宁夏回族自治区": "Ningxia",
	"新疆": "Xinjiang",
	"新疆维吾尔自治区": "Xinjiang",
}


@dataclass
class VariableSeries:
	"""保存单个变量的时间序列数据及其来源信息。"""

	name: str
	source: str
	data: pd.DataFrame


def load_dataset_index() -> List[dict]:
	"""读取数据索引文件并返回条目列表。"""

	payload = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
	return payload["items"]


def ensure_output_dir() -> None:
	"""确保输出目录存在。"""

	OUT_DIR.mkdir(parents=True, exist_ok=True)


def log_progress(message: str) -> None:
	"""打印带时间戳的进度信息，便于长任务观察运行状态。"""

	stamp = datetime.now().strftime("%H:%M:%S")
	print(f"[{stamp}] {message}", flush=True)


def normalize_province_name(name: str) -> Optional[str]:
	"""将中英文省份名称标准化为统一英文名称。"""

	if not isinstance(name, str):
		return None
	name = name.strip()
	if name in ZH_PROVINCE_ALIAS:
		return ZH_PROVINCE_ALIAS[name]
	compact = re.sub(r"[^A-Za-z]", "", name).lower()
	return PROVINCE_ALIAS.get(compact)


def extract_year_from_text(text: str) -> Optional[int]:
	"""从字符串中提取四位年份。"""

	if not isinstance(text, str):
		return None
	m = re.search(r"(19\d{2}|20\d{2})", text)
	return int(m.group(1)) if m else None


def parse_year_cell(v) -> Optional[int]:
	"""解析单元格中的年份值。"""

	if isinstance(v, (int, float)):
		iv = int(v)
		if 1900 <= iv <= 2100:
			return iv
	elif isinstance(v, str):
		return extract_year_from_text(v)
	return None


def find_header_row(ws) -> int:
	"""在工作表中识别最可能的表头行。"""

	max_row = ws.max_row or 0
	max_col = ws.max_column or 0
	scan_rows = min(max_row, 15)

	best_row = 1
	best_year_hits = -1

	# Primary strategy: row with the largest count of year-like columns is the header row.
	for ridx in range(1, scan_rows + 1):
		hits = 0
		for c in range(2, max_col + 1):
			if parse_year_cell(ws.cell(row=ridx, column=c).value) is not None:
				hits += 1
		if hits > best_year_hits:
			best_year_hits = hits
			best_row = ridx

	if best_year_hits >= 5:
		return best_row

	# Fallback strategy for non-standard templates.
	for ridx in range(1, min(max_row, 25) + 1):
		val = ws.cell(row=ridx, column=1).value
		if isinstance(val, str) and val.strip() == "指标":
			return ridx

	return 1


def parse_year_headers(ws, header_row: int) -> Dict[int, int]:
	"""解析表头行，建立 年份->列号 的映射。"""

	year_to_col: Dict[int, int] = {}
	max_col = ws.max_column or 0
	for col in range(2, max_col + 1):
		v = ws.cell(row=header_row, column=col).value
		year = None
		year = parse_year_cell(v)
		if year is not None:
			year_to_col[year] = col
	return year_to_col


def to_float(v) -> Optional[float]:
	"""将常见数值格式安全转换为浮点数。"""

	if v is None:
		return None
	if isinstance(v, (int, float, np.number)):
		return float(v)
	if isinstance(v, str):
		s = v.strip().replace(",", "")
		if not s:
			return None
		try:
			return float(s)
		except ValueError:
			return None
	return None


def sort_panel_by_province_year(df: pd.DataFrame) -> pd.DataFrame:
	"""按省份和年份稳定排序，保证省内时间顺序。"""

	if "province" not in df.columns or "year" not in df.columns:
		return df
	return cast(
		pd.DataFrame,
		df.sort_values(["province", "year"], kind="mergesort").reset_index(drop=True),
	)


def ensure_metric_columns(df: pd.DataFrame, metric: str, include_imputed: bool = False) -> pd.DataFrame:
	"""确保变量及其来源标记列存在，减少重复样板代码。"""

	if metric not in df.columns:
		df[metric] = np.nan
	source_col = f"{metric}_source"
	proxy_col = f"{metric}_is_national_proxy"
	if source_col not in df.columns:
		df[source_col] = np.nan
	if proxy_col not in df.columns:
		df[proxy_col] = np.nan
	if include_imputed:
		imputed_col = f"{metric}_is_imputed"
		if imputed_col not in df.columns:
			df[imputed_col] = 0
	return df


def get_control_series(controls: List[VariableSeries], name: str) -> Optional[VariableSeries]:
	"""按名称获取国家兜底序列。"""

	return next((s for s in controls if s.name == name), None)


def read_meic_co2(items: List[dict]) -> pd.DataFrame:
	"""读取MEIC总排放并提取1990-2023省份CO2。"""

	meic_path = next((x["path"] for x in items if "MEIC" in x["path"]), None)
	if not meic_path:
		return pd.DataFrame(columns=["province", "year", "CO2", "CO2_source", "CO2_source_priority"])

	raw = pd.read_excel(meic_path, sheet_name="MEIC-China-CO2 total emissions", header=8)
	raw = raw.rename(columns={"Province": "province_raw", "Sector": "sector"})
	raw = raw[raw["sector"].astype(str).str.lower() == "total"].copy()

	year_cols = [c for c in raw.columns if isinstance(c, (int, float)) and 1900 <= int(c) <= 2100]
	if not year_cols:
		return pd.DataFrame(columns=["province", "year", "CO2", "CO2_source", "CO2_source_priority"])

	long_df = raw.melt(
		id_vars=["province_raw"],
		value_vars=year_cols,
		var_name="year",
		value_name="CO2",
	)
	long_df["year"] = long_df["year"].astype(int)
	long_df = long_df[(long_df["year"] >= 1990) & (long_df["year"] <= 2023)]
	long_df["province"] = long_df["province_raw"].map(normalize_province_name)
	long_df["CO2"] = pd.to_numeric(long_df["CO2"], errors="coerce")
	long_df = long_df.dropna(subset=["province", "CO2"])

	out = long_df[["province", "year", "CO2"]].copy()
	out["CO2_source"] = "MEIC_1990_2023"
	out["CO2_source_priority"] = 2
	return out


def build_co2_panel(items: List[dict]) -> pd.DataFrame:
	"""使用MEIC省级总排放构建CO2主面板。"""

	meic = read_meic_co2(items)

	combo = meic.sort_values(["province", "year", "CO2_source_priority"]).drop_duplicates(
		subset=["province", "year"],
		keep="first",
	)
	combo = combo[(combo["year"] >= 1990) & (combo["year"] <= 2023)].copy()

	return combo.sort_values(["province", "year"]).reset_index(drop=True)


def read_provincial_energy_inventory(items: List[dict]) -> pd.DataFrame:
	"""读取省级能源清单并汇总总终端消费作为Energy。"""

	records: List[dict] = []

	energy_files = sorted(
		[
			x["path"]
			for x in items
			if Path(x["path"]).name.startswith("省级能源清单_") and Path(x["path"]).suffix.lower() == ".xlsx"
		]
	)

	total_files = len(energy_files)
	for fidx, path_str in enumerate(energy_files, start=1):
		log_progress(f"读取省级能源清单 {fidx}/{total_files}: {Path(path_str).name}")
		year = extract_year_from_text(Path(path_str).name)
		if year is None or year < 1990 or year > 2024:
			continue

		wb = load_workbook(path_str, read_only=False, data_only=True)
		try:
			for sheet_name in wb.sheetnames:
				if str(sheet_name).upper() == "NOTE":
					continue

				province = normalize_province_name(sheet_name)
				if not province:
					continue

				ws = wb[sheet_name]
				max_row = int(ws.max_row or 0)
				max_col = int(ws.max_column or 0)
				if max_row < 3 or max_col < 2:
					continue

				total_row = None
				for ridx in range(1, min(max_row, 120) + 1):
					v = ws.cell(row=ridx, column=1).value
					if isinstance(v, str) and v.strip().lower() == "total final consumption":
						total_row = ridx
						break

				if total_row is None:
					continue

				vals = [to_float(ws.cell(row=total_row, column=c).value) for c in range(2, max_col + 1)]
				nums = [float(v) for v in vals if v is not None and np.isfinite(v)]
				if not nums:
					continue

				records.append(
					{
						"province": province,
						"year": int(year),
						"Energy": float(np.nansum(nums)),
						"Energy_source": "Provincial_energy_inventory_1997_2022_total_final_consumption_sum",
						"Energy_is_national_proxy": 0,
					}
				)
		finally:
			wb.close()

	df = pd.DataFrame(records)
	if df.empty:
		return pd.DataFrame(columns=["province", "year", "Energy", "Energy_source", "Energy_is_national_proxy"])

	df = df.sort_values(["province", "year"]).drop_duplicates(subset=["province", "year"], keep="last")
	return df.reset_index(drop=True)


def read_provincial_macro_series(
	items: List[dict],
	path_keywords: Iterable[str],
	value_name: str,
	source_name: str,
	year_min: int = 1990,
	year_max: int = 2023,
) -> pd.DataFrame:
	"""读取省级宏观变量（第一列指标、第二列省份、后续列年份）的通用格式。"""

	path_str = select_single_sheet_path_by_keywords(items, path_keywords)
	if not path_str:
		return pd.DataFrame(columns=["province", "year", value_name, f"{value_name}_source", f"{value_name}_is_national_proxy"])

	wb = load_workbook(path_str, read_only=False, data_only=True)
	try:
		ws = wb[wb.sheetnames[0]]
		year_cols: Dict[int, int] = {}
		for col in range(3, int(ws.max_column or 0) + 1):
			year = parse_year_cell(ws.cell(row=1, column=col).value)
			if year is not None and year_min <= int(year) <= year_max:
				year_cols[int(year)] = col

		if not year_cols:
			return pd.DataFrame(columns=["province", "year", value_name, f"{value_name}_source", f"{value_name}_is_national_proxy"])

		records: List[dict] = []
		for ridx in range(2, int(ws.max_row or 0) + 1):
			province_raw = ws.cell(row=ridx, column=2).value
			if not isinstance(province_raw, str):
				continue
			province = normalize_province_name(province_raw)
			if not province:
				continue

			for year, col in sorted(year_cols.items()):
				v = to_float(ws.cell(row=ridx, column=col).value)
				if v is None:
					continue
				records.append(
					{
						"province": province,
						"year": int(year),
						value_name: float(v),
						f"{value_name}_source": source_name,
						f"{value_name}_is_national_proxy": 0,
					}
				)
	finally:
		wb.close()

	df = pd.DataFrame(records)
	if df.empty:
		return pd.DataFrame(columns=["province", "year", value_name, f"{value_name}_source", f"{value_name}_is_national_proxy"])

	df = df.sort_values(["province", "year"]).drop_duplicates(subset=["province", "year"], keep="last")
	return df.reset_index(drop=True)


def read_provincial_gdp(items: List[dict]) -> pd.DataFrame:
	"""读取省级GDP观测值。"""

	return read_provincial_macro_series(
		items,
		path_keywords=["省份维度", "经济相关", "1949-2024省级GDP"],
		value_name="GDP",
		source_name="Provincial_GDP_1949_2024_observed",
		year_min=1990,
		year_max=2023,
	)


def read_provincial_population(items: List[dict]) -> pd.DataFrame:
	"""读取省级人口观测值。"""

	return read_provincial_macro_series(
		items,
		path_keywords=["省份维度", "人口相关", "1949-2024省级人口"],
		value_name="Population",
		source_name="Provincial_population_1949_2024_observed",
		year_min=1990,
		year_max=2023,
	)


def log_observed_coverage(name: str, df: pd.DataFrame, value_col: str) -> None:
	"""输出变量覆盖范围，作为是否需要补全的诊断依据。"""

	if df.empty or value_col not in df.columns:
		log_progress(f"{name} 省级观测为空")
		return
	obs = df[["province", "year", value_col]].copy()
	obs[value_col] = pd.to_numeric(obs[value_col], errors="coerce")
	obs = obs.dropna(subset=[value_col])
	if obs.empty:
		log_progress(f"{name} 省级观测为空")
		return
	log_progress(
		f"{name} 省级覆盖: 省份{obs['province'].nunique()}个, 年份{int(obs['year'].min())}-{int(obs['year'].max())}, 观测{len(obs)}条"
	)


def fill_positive_with_provincial_first(
	panel: pd.DataFrame,
	value_col: str,
	national_proxy_series: Optional[VariableSeries],
	interpolation_source: str,
	carry_source: str,
	national_fallback_source: str,
) -> pd.DataFrame:
	"""正值变量补全：省级插值+时序延展优先，国家序列仅在省级全缺时兜底。"""

	df = panel.copy()
	df = ensure_metric_columns(df, value_col, include_imputed=True)
	source_col = f"{value_col}_source"
	proxy_col = f"{value_col}_is_national_proxy"
	imputed_col = f"{value_col}_is_imputed"
	ref_col = f"{value_col}_national_ref"

	if national_proxy_series is not None and not national_proxy_series.data.empty:
		proxy = national_proxy_series.data.rename(columns={value_col: ref_col})
		df = df.merge(proxy, on="year", how="left")
	else:
		df[ref_col] = np.nan

	for _, idx in df.groupby("province").groups.items():
		sub = cast(pd.DataFrame, df.loc[list(idx), :].sort_values("year").copy())
		sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
		sub.loc[sub[value_col] <= 0, value_col] = np.nan

		# 1) 省内内部缺口插值
		log_series = cast(pd.Series, np.log(sub[value_col].where(sub[value_col] > 0)))
		log_interp = log_series.interpolate(method="linear", limit_area="inside")
		inside_mask = sub[value_col].isna() & log_interp.notna()
		if inside_mask.any():
			sub.loc[inside_mask, value_col] = np.exp(log_interp.loc[inside_mask])
			sub.loc[inside_mask, source_col] = interpolation_source
			sub.loc[inside_mask, proxy_col] = 0
			sub.loc[inside_mask, imputed_col] = 1

		# 2) 省内边界缺口优先时序延展
		still_missing = sub[value_col].isna()
		if still_missing.any():
			carried = cast(pd.Series, sub[value_col].ffill().bfill())
			carry_mask = still_missing & carried.notna()
			if carry_mask.any():
				sub.loc[carry_mask, value_col] = carried.loc[carry_mask]
				sub.loc[carry_mask, source_col] = carry_source
				sub.loc[carry_mask, proxy_col] = 0
				sub.loc[carry_mask, imputed_col] = 1

		# 3) 仅在省级全缺时，使用国家序列兜底
		still_missing = sub[value_col].isna()
		if still_missing.any():
			ref_ok = sub[ref_col].notna() & (pd.to_numeric(sub[ref_col], errors="coerce") > 0)
			overlap_mask = sub[value_col].notna() & ref_ok
			ratio = 1.0
			if overlap_mask.any():
				ratios = cast(pd.Series, sub.loc[overlap_mask, value_col] / sub.loc[overlap_mask, ref_col])
				ratios = ratios.replace([np.inf, -np.inf], np.nan).dropna()
				if not ratios.empty:
					ratio = float(np.clip(ratios.median(), 0.05, 20.0))

			fill_with_ref = still_missing & ref_ok
			if fill_with_ref.any():
				sub.loc[fill_with_ref, value_col] = pd.to_numeric(sub.loc[fill_with_ref, ref_col], errors="coerce") * ratio
				sub.loc[fill_with_ref, source_col] = national_fallback_source
				sub.loc[fill_with_ref, proxy_col] = 1
				sub.loc[fill_with_ref, imputed_col] = 1

		df.loc[sub.index, [value_col, source_col, proxy_col, imputed_col]] = sub[[
			value_col,
			source_col,
			proxy_col,
			imputed_col,
		]]

	df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
	df[proxy_col] = pd.to_numeric(df[proxy_col], errors="coerce").fillna(0)
	df[imputed_col] = pd.to_numeric(df[imputed_col], errors="coerce").fillna(0)
	df = df.drop(columns=[ref_col], errors="ignore")
	return sort_panel_by_province_year(df)


def select_single_sheet_path(items: List[dict], keyword: str) -> Optional[str]:
	"""按关键词选择单表文件路径。"""

	for it in items:
		if it.get("sheet_count") == 1 and keyword in it["path"]:
			return it["path"]
	return None


def select_single_sheet_path_by_keywords(items: List[dict], keywords: Iterable[str]) -> Optional[str]:
	"""按多个关键词同时匹配选择单表文件路径。"""

	for it in items:
		if it.get("sheet_count") != 1:
			continue
		path = str(it.get("path", ""))
		if all(k in path for k in keywords):
			return path
	return None


def read_national_series(
	path_str: str,
	variable_name: str,
	source_name: str,
	indicator_keywords: Iterable[str],
) -> VariableSeries:
	"""读取国家层指标序列并抽取目标变量。"""

	wb = load_workbook(path_str, read_only=False, data_only=True)
	try:
		ws = wb[wb.sheetnames[0]]
		header_row = find_header_row(ws)
		year_cols = parse_year_headers(ws, header_row)

		target_row = None
		max_row = ws.max_row or 0
		candidates: List[tuple] = []
		for ridx in range(header_row + 1, max_row + 1):
			label = ws.cell(row=ridx, column=1).value
			if not isinstance(label, str):
				continue
			label = label.strip()
			if not label:
				continue

			vals = [to_float(ws.cell(row=ridx, column=c).value) for c in year_cols.values()]
			non_null = sum(v is not None for v in vals)
			numeric_vals = [v for v in vals if v is not None]
			std = float(np.std(numeric_vals)) if len(numeric_vals) >= 2 else 0.0

			if any(k in label for k in indicator_keywords):
				# Rank by coverage first, then by variability to avoid selecting all-100 total rows.
				candidates.append((non_null, std, ridx))

		if candidates:
			candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
			target_row = candidates[0][2]

		if target_row is None:
			# Fallback: first non-empty numeric row after header.
			for ridx in range(header_row + 1, max_row + 1):
				label = ws.cell(row=ridx, column=1).value
				if not isinstance(label, str) or not label.strip():
					continue
				vals = [to_float(ws.cell(row=ridx, column=c).value) for c in year_cols.values()]
				if any(v is not None for v in vals):
					target_row = ridx
					break

		rows: List[dict] = []
		if target_row is not None:
			for year, col in sorted(year_cols.items()):
				rows.append({"year": int(year), variable_name: to_float(ws.cell(row=target_row, column=col).value)})

		df = pd.DataFrame(rows, columns=["year", variable_name])
		if not df.empty:
			df = cast(pd.DataFrame, df.loc[(df["year"] >= 1990) & (df["year"] <= 2022), :].copy())
		return VariableSeries(name=variable_name, source=source_name, data=cast(pd.DataFrame, df))
	finally:
		wb.close()


def build_national_controls(items: List[dict]) -> List[VariableSeries]:
	"""构建国家层控制变量序列（用于补全和锚定）。"""

	# National proxies remain needed for fallback and anchoring.
	gdp_path = select_single_sheet_path_by_keywords(items, ["国家维度", "经济相关", "人均GDP"])
	population_path = select_single_sheet_path_by_keywords(items, ["国家维度", "人口相关", "年末总人口"])
	urban_path = select_single_sheet_path_by_keywords(items, ["国家维度", "人口相关", "城镇化率"])
	industry_path = select_single_sheet_path_by_keywords(items, ["国家维度", "经济相关", "三次产业构成"])
	energy_path = select_single_sheet_path_by_keywords(items, ["国家维度", "能源相关", "能源消费总量"])

	if gdp_path is None:
		gdp_path = select_single_sheet_path(items, "GDP")

	series_list: List[VariableSeries] = []
	if gdp_path:
		series_list.append(
			read_national_series(
				gdp_path,
				variable_name="GDP",
				source_name="National_macro_GDP",
				indicator_keywords=["国民总收入", "国内生产总值", "GDP"],
			)
		)
	if population_path:
		series_list.append(
			read_national_series(
				population_path,
				variable_name="Population",
				source_name="National_macro_Population",
				indicator_keywords=["年末总人口"],
			)
		)
		# Prefer computing urbanization from population components when available.
		series_list.append(
			read_national_series(
				population_path,
				variable_name="UrbanPopulation",
				source_name="National_macro_UrbanPopulation",
				indicator_keywords=["城镇人口"],
			)
		)
	if energy_path:
		series_list.append(
			read_national_series(
				energy_path,
				variable_name="Energy",
				source_name="National_macro_Energy",
				indicator_keywords=["能源消费总量", "总量"],
			)
		)
	if industry_path:
		series_list.append(
			read_national_series(
				industry_path,
				variable_name="Industry",
				source_name="National_macro_IndustryShare",
				indicator_keywords=["第二产业"],
			)
		)
	if urban_path:
		series_list.append(
			read_national_series(
				urban_path,
				variable_name="Urbanization",
				source_name="National_macro_Urbanization",
				indicator_keywords=["城镇化率", "城镇"],
			)
		)

	# If direct urbanization series is missing, derive it from urban/total population.
	name_to_series = {s.name: s for s in series_list}
	urban_series = name_to_series.get("Urbanization")
	pop_series = name_to_series.get("Population")
	urban_pop_series = name_to_series.get("UrbanPopulation")
	if (
		(urban_series is None or urban_series.data.empty)
		and pop_series is not None
		and urban_pop_series is not None
		and not pop_series.data.empty
		and not urban_pop_series.data.empty
	):
		merged = pop_series.data.merge(urban_pop_series.data, on="year", how="inner")
		if not merged.empty:
			merged["Urbanization"] = (
			100.0 * pd.to_numeric(merged["UrbanPopulation"], errors="coerce")
			/ pd.to_numeric(merged["Population"], errors="coerce")
			)
			merged["Urbanization"] = merged["Urbanization"].replace([np.inf, -np.inf], np.nan)
			urb_df = cast(pd.DataFrame, merged.loc[:, ["year", "Urbanization"]].copy())
			replacement = VariableSeries(
				name="Urbanization",
				source="National_macro_Urbanization_from_population",
				data=urb_df,
			)
			if urban_series is None:
				series_list.append(replacement)
			else:
				series_list = [replacement if s.name == "Urbanization" else s for s in series_list]

	return series_list


def read_provincial_industry_share(items: List[dict]) -> pd.DataFrame:
	"""读取省级第二产业占比观测值。"""

	path_str = select_single_sheet_path_by_keywords(items, ["省份维度", "经济相关", "省份第二产业占比"])
	if not path_str:
		return pd.DataFrame(columns=["province", "year", "Industry", "Industry_source", "Industry_is_national_proxy"])

	wb = load_workbook(path_str, read_only=False, data_only=True)
	try:
		ws = wb[wb.sheetnames[0]]

		year_cols: Dict[int, int] = {}
		for col in range(3, int(ws.max_column or 0) + 1):
			year = parse_year_cell(ws.cell(row=1, column=col).value)
			if year is not None and 1990 <= int(year) <= 2022:
				year_cols[int(year)] = col

		if not year_cols:
			return pd.DataFrame(columns=["province", "year", "Industry", "Industry_source", "Industry_is_national_proxy"])

		records: List[dict] = []
		last_indicator: Optional[str] = None
		for ridx in range(2, int(ws.max_row or 0) + 1):
			indicator_cell = ws.cell(row=ridx, column=1).value
			if isinstance(indicator_cell, str) and indicator_cell.strip():
				last_indicator = indicator_cell.strip()
			indicator = last_indicator
			province_raw = ws.cell(row=ridx, column=2).value
			if not isinstance(indicator, str):
				continue
			if not isinstance(province_raw, str):
				continue
			if "第二产业" not in indicator or "比重" not in indicator:
				continue

			province = normalize_province_name(province_raw)
			if not province:
				continue

			for year, col in sorted(year_cols.items()):
				v = to_float(ws.cell(row=ridx, column=col).value)
				if v is None:
					continue
				records.append(
					{
						"province": province,
						"year": int(year),
						"Industry": float(v),
						"Industry_source": "Provincial_industry_share_1996_2024_observed",
						"Industry_is_national_proxy": 0,
					}
				)
	finally:
		wb.close()

	df = pd.DataFrame(records)
	if df.empty:
		return pd.DataFrame(columns=["province", "year", "Industry", "Industry_source", "Industry_is_national_proxy"])

	df = df.sort_values(["province", "year"]).drop_duplicates(subset=["province", "year"], keep="last")
	return df.reset_index(drop=True)


def fill_share_with_provincial_first(
	panel: pd.DataFrame,
	metric: str,
	national_proxy_series: Optional[VariableSeries],
	interpolation_source: str,
	carry_source: str,
	national_fallback_source: str,
	ratio_clip: Tuple[float, float],
	observed_col: Optional[str] = None,
	observed_source: Optional[str] = None,
) -> pd.DataFrame:
	"""占比类变量补全：省级优先，国家序列仅用于最后兜底。"""

	df = panel.copy()
	df = ensure_metric_columns(df, metric, include_imputed=True)

	source_col = f"{metric}_source"
	proxy_col = f"{metric}_is_national_proxy"
	imputed_col = f"{metric}_is_imputed"
	nat_ref_col = f"{metric}_national_ref"

	df[source_col] = df[source_col].astype(object)

	if observed_col and observed_col in df.columns:
		obs_mask = df[observed_col].notna()
		df.loc[obs_mask, metric] = pd.to_numeric(df.loc[obs_mask, observed_col], errors="coerce")
		if observed_source:
			df.loc[obs_mask, source_col] = observed_source
		df.loc[obs_mask, proxy_col] = 0
		df.loc[obs_mask, imputed_col] = 0
		df = df.drop(columns=[observed_col], errors="ignore")

	if national_proxy_series is not None and not national_proxy_series.data.empty:
		proxy = national_proxy_series.data.rename(columns={metric: nat_ref_col})
		df = df.merge(proxy, on="year", how="left")
	else:
		df[nat_ref_col] = np.nan

	def clip_share(series: pd.Series) -> pd.Series:
		return series.clip(lower=0.01, upper=99.99)

	for _, idx in df.groupby("province").groups.items():
		sub = cast(pd.DataFrame, df.loc[list(idx), :].sort_values("year").copy())
		sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
		sub.loc[(sub[metric] <= 0) | (sub[metric] >= 100), metric] = np.nan

		# 1) 省内内部缺口插值。
		base = clip_share(sub[metric])
		logit = cast(pd.Series, np.log(base / (100.0 - base)))
		logit_interp = logit.interpolate(method="linear", limit_area="inside")
		inside_mask = sub[metric].isna() & logit_interp.notna()
		if inside_mask.any():
			sub.loc[inside_mask, metric] = 100.0 / (1.0 + np.exp(-logit_interp.loc[inside_mask]))
			sub.loc[inside_mask, source_col] = interpolation_source
			sub.loc[inside_mask, proxy_col] = 0
			sub.loc[inside_mask, imputed_col] = 1

		# 2) 边界缺口优先用省内时序延展。
		still_missing = sub[metric].isna()
		if still_missing.any():
			carried = cast(pd.Series, sub[metric].ffill().bfill())
			carry_mask = still_missing & carried.notna()
			if carry_mask.any():
				sub.loc[carry_mask, metric] = np.clip(carried.loc[carry_mask], 0.01, 99.99)
				sub.loc[carry_mask, source_col] = carry_source
				sub.loc[carry_mask, proxy_col] = 0
				sub.loc[carry_mask, imputed_col] = 1

		# 3) 仅在省级仍缺失时用国家序列兜底。
		still_missing = sub[metric].isna()
		if still_missing.any():
			ref_ok = sub[nat_ref_col].notna() & (sub[nat_ref_col] > 0)
			overlap_mask = sub[metric].notna() & ref_ok
			ratio = 1.0
			if overlap_mask.any():
				ratios = cast(pd.Series, sub.loc[overlap_mask, metric] / sub.loc[overlap_mask, nat_ref_col])
				ratios = ratios.replace([np.inf, -np.inf], np.nan).dropna()
				if not ratios.empty:
					ratio = float(np.clip(ratios.median(), ratio_clip[0], ratio_clip[1]))

			fill_with_ref = still_missing & ref_ok
			if fill_with_ref.any():
				sub.loc[fill_with_ref, metric] = np.clip(
					sub.loc[fill_with_ref, nat_ref_col] * ratio,
					0.01,
					99.99,
				)
				sub.loc[fill_with_ref, source_col] = national_fallback_source
				sub.loc[fill_with_ref, proxy_col] = 1
				sub.loc[fill_with_ref, imputed_col] = 1

		df.loc[sub.index, [metric, source_col, proxy_col, imputed_col]] = sub[
			[metric, source_col, proxy_col, imputed_col]
		]

	df = df.drop(columns=[nat_ref_col], errors="ignore")
	df[metric] = pd.to_numeric(df[metric], errors="coerce")
	df[proxy_col] = pd.to_numeric(df[proxy_col], errors="coerce").fillna(0)
	df[imputed_col] = pd.to_numeric(df[imputed_col], errors="coerce").fillna(0)
	return sort_panel_by_province_year(df)


def fill_industry_with_interpolation_fit_and_anchor(
	panel: pd.DataFrame,
	industry_proxy_series: Optional[VariableSeries],
) -> pd.DataFrame:
	"""按省级优先原则补全Industry，国家序列仅做最后兜底。"""

	return fill_share_with_provincial_first(
		panel=panel,
		metric="Industry",
		national_proxy_series=industry_proxy_series,
		interpolation_source="Provincial_industry_logit_interpolation",
		carry_source="Provincial_industry_temporal_carry_fill",
		national_fallback_source="Industry_national_fallback_no_provincial_obs",
		ratio_clip=(0.2, 5.0),
	)


def read_provincial_urbanization_share(items: List[dict]) -> pd.DataFrame:
	"""读取省级城镇化率（城镇人口占比）观测值。"""

	path_str = select_single_sheet_path_by_keywords(items, ["省份维度", "人口相关", "城镇人口所占比重"])
	if not path_str:
		return pd.DataFrame(columns=["province", "year", "Urbanization_prov"])

	wb = load_workbook(path_str, read_only=False, data_only=True)
	try:
		ws = wb[wb.sheetnames[0]]

		year_cols: Dict[int, int] = {}
		for col in range(3, int(ws.max_column or 0) + 1):
			year = parse_year_cell(ws.cell(row=1, column=col).value)
			if year is not None and 1990 <= int(year) <= 2022:
				year_cols[int(year)] = col

		if not year_cols:
			return pd.DataFrame(columns=["province", "year", "Urbanization_prov"])

		records: List[dict] = []
		last_indicator: Optional[str] = None
		for ridx in range(2, int(ws.max_row or 0) + 1):
			indicator_cell = ws.cell(row=ridx, column=1).value
			if isinstance(indicator_cell, str) and indicator_cell.strip():
				last_indicator = indicator_cell.strip()
			indicator = last_indicator
			province_raw = ws.cell(row=ridx, column=2).value
			if not isinstance(indicator, str):
				continue
			if not isinstance(province_raw, str):
				continue
			if "城镇人口" not in indicator or "比重" not in indicator:
				continue

			province = normalize_province_name(province_raw)
			if not province:
				continue

			for year, col in sorted(year_cols.items()):
				v = to_float(ws.cell(row=ridx, column=col).value)
				if v is None:
					continue
				records.append(
					{
						"province": province,
						"year": int(year),
						"Urbanization_prov": float(v),
					}
				)
	finally:
		wb.close()

	df = pd.DataFrame(records)
	if df.empty:
		return pd.DataFrame(columns=["province", "year", "Urbanization_prov"])

	df = df.sort_values(["province", "year"]).drop_duplicates(subset=["province", "year"], keep="last")
	return df.reset_index(drop=True)


def fill_urbanization_with_interpolation_fit_and_anchor(
	panel: pd.DataFrame,
	urbanization_proxy_series: Optional[VariableSeries],
) -> pd.DataFrame:
	"""按省级优先原则补全Urbanization，国家序列仅做最后兜底。"""

	return fill_share_with_provincial_first(
		panel=panel,
		metric="Urbanization",
		national_proxy_series=urbanization_proxy_series,
		interpolation_source="Provincial_urbanization_logit_interpolation",
		carry_source="Provincial_urbanization_temporal_carry_fill",
		national_fallback_source="Urbanization_national_fallback_no_provincial_obs",
		ratio_clip=(0.2, 5.0),
		observed_col="Urbanization_prov",
		observed_source="Provincial_urbanization_share_1990_2025_observed",
	)


def safe_log(s: pd.Series) -> pd.Series:
	"""仅对正值取对数，其余保留为空。"""

	return cast(pd.Series, np.log(s.where(s > 0)))


def add_kaya_features(panel: pd.DataFrame) -> pd.DataFrame:
	"""基于原始变量构造Kaya恒等式分解因子。"""

	df = panel.copy()
	df["A"] = df["GDP"] / df["Population"]
	df["B"] = df["Energy"] / df["GDP"]
	df["C"] = df["CO2"] / df["Energy"]

	df["ln_CO2"] = safe_log(df["CO2"])
	df["ln_Population"] = safe_log(df["Population"])
	df["ln_A"] = safe_log(df["A"])
	df["ln_B"] = safe_log(df["B"])
	df["ln_C"] = safe_log(df["C"])
	return df


def log_mean(a: float, b: float) -> Optional[float]:
	"""计算LMDI使用的对数平均权重。"""

	if a is None or b is None:
		return None
	if a <= 0 or b <= 0:
		return None
	if abs(a - b) < 1e-12:
		return float(a)
	da = np.log(a)
	db = np.log(b)
	if abs(da - db) < 1e-12:
		return float(a)
	return float((a - b) / (da - db))


def compute_lmdi_time(panel: pd.DataFrame) -> pd.DataFrame:
	"""按省份逐年计算LMDI时序分解结果。"""

	rows: List[dict] = []

	for province, g in panel.sort_values(["province", "year"]).groupby("province"):
		g = g.reset_index(drop=True)
		for i in range(1, len(g)):
			prev = g.iloc[i - 1]
			curr = g.iloc[i]

			e0, et = prev["CO2"], curr["CO2"]
			p0, pt = prev["Population"], curr["Population"]
			a0, at = prev["A"], curr["A"]
			b0, bt = prev["B"], curr["B"]
			c0, ct = prev["C"], curr["C"]

			valid = all(
				pd.notna(v) and float(v) > 0
				for v in [e0, et, p0, pt, a0, at, b0, bt, c0, ct]
			)
			if not valid:
				continue

			lm = log_mean(float(et), float(e0))
			if lm is None:
				continue

			d_p = lm * np.log(float(pt) / float(p0))
			d_a = lm * np.log(float(at) / float(a0))
			d_b = lm * np.log(float(bt) / float(b0))
			d_c = lm * np.log(float(ct) / float(c0))
			d_co2 = float(et) - float(e0)
			resid = d_co2 - (d_p + d_a + d_b + d_c)

			rows.append(
				{
					"province": province,
					"year": int(curr["year"]),
					"base_year": int(prev["year"]),
					"delta_CO2": d_co2,
					"delta_P": d_p,
					"delta_A": d_a,
					"delta_B": d_b,
					"delta_C": d_c,
					"lmdi_residual": resid,
				}
			)

	return pd.DataFrame(rows)


def fill_energy_with_interpolation_and_fit(
	panel: pd.DataFrame,
	energy_proxy_series: Optional[VariableSeries],
) -> pd.DataFrame:
	"""按省级优先原则补全Energy，国家序列仅做最后兜底。"""

	df = panel.copy()
	df = ensure_metric_columns(df, "Energy", include_imputed=True)

	if energy_proxy_series is not None and not energy_proxy_series.data.empty:
		proxy = energy_proxy_series.data.rename(columns={"Energy": "Energy_national_ref"})
		df = df.merge(proxy, on="year", how="left")
	else:
		df["Energy_national_ref"] = np.nan

	for _, idx in df.groupby("province").groups.items():
		sub = cast(pd.DataFrame, df.loc[list(idx), :].sort_values("year").copy())
		sub["Energy"] = pd.to_numeric(sub["Energy"], errors="coerce")
		sub.loc[sub["Energy"] <= 0, "Energy"] = np.nan

		# 1) In-series interpolation in log space for internal gaps.
		log_energy = cast(pd.Series, np.log(sub["Energy"].where(sub["Energy"] > 0)))
		log_interp = log_energy.interpolate(method="linear", limit_area="inside")
		inside_mask = sub["Energy"].isna() & log_interp.notna()
		if inside_mask.any():
			sub.loc[inside_mask, "Energy"] = np.exp(log_interp.loc[inside_mask])
			sub.loc[inside_mask, "Energy_source"] = "Provincial_energy_log_interpolation"
			sub.loc[inside_mask, "Energy_is_national_proxy"] = 0
			sub.loc[inside_mask, "Energy_is_imputed"] = 1

		# 2) 边界缺口优先使用省内时序延展。
		still_missing = sub["Energy"].isna()
		if still_missing.any():
			carried = cast(pd.Series, sub["Energy"].ffill().bfill())
			carry_mask = still_missing & carried.notna()
			if carry_mask.any():
				sub.loc[carry_mask, "Energy"] = carried.loc[carry_mask]
				sub.loc[carry_mask, "Energy_source"] = "Provincial_energy_temporal_carry_fill"
				sub.loc[carry_mask, "Energy_is_national_proxy"] = 0
				sub.loc[carry_mask, "Energy_is_imputed"] = 1

		# 3) 仅在省级全缺时，使用国家序列兜底。
		still_missing = sub["Energy"].isna()
		if still_missing.any():
			ref_ok = sub["Energy_national_ref"].notna() & (sub["Energy_national_ref"] > 0)
			overlap_mask = sub["Energy"].notna() & ref_ok
			ratio = 1.0
			if overlap_mask.any():
				ratios = cast(pd.Series, sub.loc[overlap_mask, "Energy"] / sub.loc[overlap_mask, "Energy_national_ref"])
				ratios = ratios.replace([np.inf, -np.inf], np.nan).dropna()
				if not ratios.empty:
					ratio = float(np.clip(ratios.median(), 0.05, 20.0))

			fill_with_ref = still_missing & ref_ok
			if fill_with_ref.any():
				sub.loc[fill_with_ref, "Energy"] = sub.loc[fill_with_ref, "Energy_national_ref"] * ratio
				sub.loc[fill_with_ref, "Energy_source"] = "Energy_national_fallback_no_provincial_obs"
				sub.loc[fill_with_ref, "Energy_is_national_proxy"] = 1
				sub.loc[fill_with_ref, "Energy_is_imputed"] = 1

		df.loc[sub.index, ["Energy", "Energy_source", "Energy_is_national_proxy", "Energy_is_imputed"]] = sub[
			["Energy", "Energy_source", "Energy_is_national_proxy", "Energy_is_imputed"]
		]

	df["Energy"] = pd.to_numeric(df["Energy"], errors="coerce")
	df["Energy_is_national_proxy"] = df["Energy_is_national_proxy"].fillna(0)
	df["Energy_is_imputed"] = df["Energy_is_imputed"].fillna(0)
	df = df.drop(columns=["Energy_national_ref"], errors="ignore")
	return sort_panel_by_province_year(df)


def main() -> None:
	"""执行数据预处理主流程并写出结果文件。"""

	log_progress("开始执行预处理流程")
	ensure_output_dir()
	items = load_dataset_index()
	log_progress(f"已加载数据索引，共 {len(items)} 个条目")

	log_progress("阶段1/7：构建CO2主面板")
	co2_panel = build_co2_panel(items)
	log_progress(f"CO2主面板完成，记录数 {len(co2_panel)}")

	log_progress("阶段2/7：读取省级GDP与Population")
	prov_gdp = read_provincial_gdp(items)
	prov_population = read_provincial_population(items)
	log_observed_coverage("GDP", prov_gdp, "GDP")
	log_observed_coverage("Population", prov_population, "Population")
	panel = co2_panel.copy()
	if not prov_gdp.empty:
		panel = panel.merge(prov_gdp, on=["province", "year"], how="left")
	if not prov_population.empty:
		panel = panel.merge(prov_population, on=["province", "year"], how="left")

	log_progress("阶段3/7：构建国家兜底序列（仅缺口兜底）")
	controls = build_national_controls(items)
	log_progress(f"国家兜底序列准备完成，变量数 {len(controls)}")
	gdp_proxy_series = get_control_series(controls, "GDP")
	population_proxy_series = get_control_series(controls, "Population")
	energy_proxy_series = get_control_series(controls, "Energy")
	industry_proxy_series = get_control_series(controls, "Industry")
	urbanization_proxy_series = get_control_series(controls, "Urbanization")

	panel = fill_positive_with_provincial_first(
		panel,
		value_col="GDP",
		national_proxy_series=gdp_proxy_series,
		interpolation_source="Provincial_GDP_log_interpolation",
		carry_source="Provincial_GDP_temporal_carry_fill",
		national_fallback_source="GDP_national_fallback_no_provincial_obs",
	)
	panel = fill_positive_with_provincial_first(
		panel,
		value_col="Population",
		national_proxy_series=population_proxy_series,
		interpolation_source="Provincial_population_log_interpolation",
		carry_source="Provincial_population_temporal_carry_fill",
		national_fallback_source="Population_national_fallback_no_provincial_obs",
	)

	log_progress("阶段4/7：补全Industry")
	prov_industry = read_provincial_industry_share(items)
	log_observed_coverage("Industry", prov_industry, "Industry")
	if not prov_industry.empty:
		panel = panel.merge(prov_industry, on=["province", "year"], how="left")
	panel = ensure_metric_columns(panel, "Industry", include_imputed=True)

	panel = fill_industry_with_interpolation_fit_and_anchor(panel, industry_proxy_series)

	log_progress("阶段5/7：补全Urbanization")
	prov_urbanization = read_provincial_urbanization_share(items)
	log_observed_coverage("Urbanization", prov_urbanization, "Urbanization_prov")
	if not prov_urbanization.empty:
		panel = panel.merge(prov_urbanization, on=["province", "year"], how="left")
	panel = ensure_metric_columns(panel, "Urbanization", include_imputed=True)

	panel = fill_urbanization_with_interpolation_fit_and_anchor(panel, urbanization_proxy_series)

	log_progress("阶段6/7：补全Energy")
	prov_energy = read_provincial_energy_inventory(items)
	log_observed_coverage("Energy", prov_energy, "Energy")
	if not prov_energy.empty:
		panel = panel.merge(prov_energy, on=["province", "year"], how="left")
	panel = ensure_metric_columns(panel, "Energy", include_imputed=True)

	panel = fill_energy_with_interpolation_and_fit(panel, energy_proxy_series)

	panel["Energy_is_national_proxy"] = panel["Energy_is_national_proxy"].fillna(0)
	panel["Energy_is_imputed"] = panel["Energy_is_imputed"].fillna(0)
	panel = add_kaya_features(panel)

	core_cols = [
		"province",
		"year",
		"CO2",
		"GDP",
		"Population",
		"Energy",
		"Industry",
		"Urbanization",
		"A",
		"B",
		"C",
		"CO2_source",
	]
	panel_core = cast(pd.DataFrame, panel.loc[:, [c for c in core_cols if c in panel.columns]].copy())

	# Keep rows where CO2 exists; remaining variables are filled by available controls.
	panel_core = sort_panel_by_province_year(panel_core)

	log_progress("阶段7/7：计算LMDI并写出文件")
	lmdi_df = compute_lmdi_time(panel_core)

	panel_path = OUT_DIR / "panel_master.csv"
	lmdi_path = OUT_DIR / "lmdi_decomposition.csv"
	audit_path = OUT_DIR / "panel_source_audit.csv"
	summary_path = OUT_DIR / "panel_build_summary.json"

	panel_core.to_csv(panel_path, index=False, encoding="utf-8-sig")
	lmdi_df.to_csv(lmdi_path, index=False, encoding="utf-8-sig")

	audit_cols = [
		c
		for c in panel.columns
		if c.endswith("_source") or c.endswith("_is_national_proxy") or c.endswith("_is_imputed")
	]
	panel[["province", "year", "CO2"] + audit_cols].to_csv(audit_path, index=False, encoding="utf-8-sig")

	summary = {
		"panel_rows": int(len(panel_core)),
		"province_count": int(panel_core["province"].nunique()),
		"year_min": int(panel_core["year"].min()) if len(panel_core) else None,
		"year_max": int(panel_core["year"].max()) if len(panel_core) else None,
		"co2_source_counts": panel_core["CO2_source"].value_counts(dropna=False).to_dict(),
		"missing_ratio": {
			c: float(panel_core[c].isna().mean())
			for c in ["CO2", "GDP", "Population", "Energy", "Industry", "Urbanization", "A", "B", "C"]
			if c in panel_core.columns
		},
		"gdp_source_counts": panel["GDP_source"].value_counts(dropna=False).to_dict() if "GDP_source" in panel.columns else {},
		"population_source_counts": panel["Population_source"].value_counts(dropna=False).to_dict()
		if "Population_source" in panel.columns
		else {},
		"energy_source_counts": panel["Energy_source"].value_counts(dropna=False).to_dict()
		if "Energy_source" in panel.columns
		else {},
		"industry_source_counts": panel["Industry_source"].value_counts(dropna=False).to_dict()
		if "Industry_source" in panel.columns
		else {},
		"urbanization_source_counts": panel["Urbanization_source"].value_counts(dropna=False).to_dict()
		if "Urbanization_source" in panel.columns
		else {},
		"notes": [
			"CO2 uses MEIC provincial total emissions only (1990-2023).",
			"GDP and Population are read from provincial files first; fill order is in-province interpolation, temporal carry, and national fallback only if still missing.",
			"Energy uses provincial inventory (1997-2022); fill order is in-province log interpolation, temporal carry, and national fallback only if still missing.",
			"Industry and Urbanization use provincial share series first; fill order is in-province logit interpolation, temporal carry, and national fallback only if still missing.",
			"Panel rows are always output in stable province-year order.",
		],
	}
	summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

	log_progress("预处理流程完成")
	print("WROTE", panel_path)
	print("WROTE", lmdi_path)
	print("WROTE", audit_path)
	print("WROTE", summary_path)
	print("PANEL_ROWS", summary["panel_rows"], "PROVINCES", summary["province_count"], "YEARS", summary["year_min"], summary["year_max"])
	print("CO2_SOURCE_COUNTS", summary["co2_source_counts"])


if __name__ == "__main__":
	main()