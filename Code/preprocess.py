from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, cast

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
	name: str
	source: str
	data: pd.DataFrame


def load_dataset_index() -> List[dict]:
	payload = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
	return payload["items"]


def ensure_output_dir() -> None:
	OUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_province_name(name: str) -> Optional[str]:
	if not isinstance(name, str):
		return None
	name = name.strip()
	if name in ZH_PROVINCE_ALIAS:
		return ZH_PROVINCE_ALIAS[name]
	compact = re.sub(r"[^A-Za-z]", "", name).lower()
	return PROVINCE_ALIAS.get(compact)


def extract_year_from_text(text: str) -> Optional[int]:
	if not isinstance(text, str):
		return None
	m = re.search(r"(19\d{2}|20\d{2})", text)
	return int(m.group(1)) if m else None


def parse_year_cell(v) -> Optional[int]:
	if isinstance(v, (int, float)):
		iv = int(v)
		if 1900 <= iv <= 2100:
			return iv
	elif isinstance(v, str):
		return extract_year_from_text(v)
	return None


def find_header_row(ws) -> int:
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


def read_inventory_co2_nbs(items: List[dict]) -> pd.DataFrame:
	records: List[dict] = []

	inventory_files = sorted(
		[
			x["path"]
			for x in items
			if x.get("sheet_count") == 31 and "2001-2022" in x["path"]
		]
	)

	for path_str in inventory_files:
		year = extract_year_from_text(Path(path_str).name)
		if year is None:
			continue

		wb = load_workbook(path_str, read_only=True, data_only=True)
		for sheet_name in wb.sheetnames:
			if str(sheet_name).upper() == "NOTE":
				continue

			province = normalize_province_name(sheet_name)
			if not province:
				continue

			ws = wb[sheet_name]
			max_row = ws.max_row or 0
			max_col = ws.max_column or 0
			if max_row < 4 or max_col < 5:
				continue

			header = [ws.cell(row=1, column=c).value for c in range(1, max_col + 1)]
			header_norm = [str(h).strip() if h is not None else "" for h in header]

			try:
				total_row = None
				for r in range(2, max_row + 1):
					v = ws.cell(row=r, column=1).value
					if isinstance(v, str) and v.strip() == "TotalEmissions":
						total_row = r
						break
				if total_row is None:
					continue

				if "Scope_1_Total" in header_norm:
					cidx = header_norm.index("Scope_1_Total") + 1
					co2 = to_float(ws.cell(row=total_row, column=cidx).value)
				else:
					# Fallback: sum known fuel/process columns in the total row.
					cols = [
						i + 1
						for i, h in enumerate(header_norm)
						if h
						and h
						not in {
							"Emission_Inventory",
							"unit",
						}
						and not h.startswith("Scope_2")
					]
					vals = [to_float(ws.cell(row=total_row, column=c).value) for c in cols]
					co2 = float(np.nansum([v for v in vals if v is not None])) if vals else None

				if co2 is None:
					continue

				records.append(
					{
						"province": province,
						"year": int(year),
						"CO2": float(co2),
						"CO2_source": "NBS_2001_2022_inventory",
						"CO2_source_priority": 1,
					}
				)
			except Exception:
				continue

	return pd.DataFrame(records)


def read_meic_co2(items: List[dict]) -> pd.DataFrame:
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
	long_df = long_df[(long_df["year"] >= 1990) & (long_df["year"] <= 2000)]
	long_df["province"] = long_df["province_raw"].map(normalize_province_name)
	long_df["CO2"] = pd.to_numeric(long_df["CO2"], errors="coerce")
	long_df = long_df.dropna(subset=["province", "CO2"])

	out = long_df[["province", "year", "CO2"]].copy()
	out["CO2_source"] = "MEIC_1990_2000"
	out["CO2_source_priority"] = 2
	return out


def build_co2_panel(items: List[dict]) -> pd.DataFrame:
	nbs = read_inventory_co2_nbs(items)
	meic = read_meic_co2(items)
	combo = pd.concat([nbs, meic], ignore_index=True)

	combo = combo.sort_values(["province", "year", "CO2_source_priority"]).drop_duplicates(
		subset=["province", "year"],
		keep="first",
	)
	combo = combo[(combo["year"] >= 1990) & (combo["year"] <= 2022)].copy()

	return combo.sort_values(["province", "year"]).reset_index(drop=True)


def read_provincial_energy_inventory(items: List[dict]) -> pd.DataFrame:
	records: List[dict] = []

	energy_files = sorted(
		[
			x["path"]
			for x in items
			if Path(x["path"]).name.startswith("省级能源清单_") and Path(x["path"]).suffix.lower() == ".xlsx"
		]
	)

	for path_str in energy_files:
		year = extract_year_from_text(Path(path_str).name)
		if year is None or year < 1990 or year > 2022:
			continue

		wb = load_workbook(path_str, read_only=True, data_only=True)
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

	df = pd.DataFrame(records)
	if df.empty:
		return pd.DataFrame(columns=["province", "year", "Energy", "Energy_source", "Energy_is_national_proxy"])

	df = df.sort_values(["province", "year"]).drop_duplicates(subset=["province", "year"], keep="last")
	return df.reset_index(drop=True)


def select_single_sheet_path(items: List[dict], keyword: str) -> Optional[str]:
	for it in items:
		if it.get("sheet_count") == 1 and keyword in it["path"]:
			return it["path"]
	return None


def select_single_sheet_path_by_keywords(items: List[dict], keywords: Iterable[str]) -> Optional[str]:
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
	wb = load_workbook(path_str, read_only=True, data_only=True)
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


def build_national_controls(items: List[dict]) -> List[VariableSeries]:
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

	return series_list


def read_provincial_industry_share(items: List[dict]) -> pd.DataFrame:
	path_str = select_single_sheet_path_by_keywords(items, ["省份维度", "经济相关", "省份第二产业占比"])
	if not path_str:
		return pd.DataFrame(columns=["province", "year", "Industry", "Industry_source", "Industry_is_national_proxy"])

	wb = load_workbook(path_str, read_only=True, data_only=True)
	ws = wb[wb.sheetnames[0]]

	year_cols: Dict[int, int] = {}
	for col in range(3, int(ws.max_column or 0) + 1):
		year = parse_year_cell(ws.cell(row=1, column=col).value)
		if year is not None and 1990 <= int(year) <= 2022:
			year_cols[int(year)] = col

	if not year_cols:
		return pd.DataFrame(columns=["province", "year", "Industry", "Industry_source", "Industry_is_national_proxy"])

	records: List[dict] = []
	for ridx in range(2, int(ws.max_row or 0) + 1):
		indicator = ws.cell(row=ridx, column=1).value
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

	df = pd.DataFrame(records)
	if df.empty:
		return pd.DataFrame(columns=["province", "year", "Industry", "Industry_source", "Industry_is_national_proxy"])

	df = df.sort_values(["province", "year"]).drop_duplicates(subset=["province", "year"], keep="last")
	return df.reset_index(drop=True)


def fill_industry_with_interpolation_fit_and_anchor(
	panel: pd.DataFrame,
	industry_proxy_series: Optional[VariableSeries],
) -> pd.DataFrame:
	df = panel.copy()
	if "Industry" not in df.columns:
		df["Industry"] = np.nan
	if "Industry_source" not in df.columns:
		df["Industry_source"] = np.nan
	if "Industry_is_national_proxy" not in df.columns:
		df["Industry_is_national_proxy"] = np.nan
	if "Industry_is_imputed" not in df.columns:
		df["Industry_is_imputed"] = 0

	def clip_share(series: pd.Series) -> pd.Series:
		return series.clip(lower=0.01, upper=99.99)

	for province, idx in df.groupby("province").groups.items():
		sub = cast(pd.DataFrame, df.loc[list(idx), :].sort_values("year").copy())
		sub["Industry"] = pd.to_numeric(sub["Industry"], errors="coerce")
		sub.loc[(sub["Industry"] <= 0) | (sub["Industry"] >= 100), "Industry"] = np.nan

		# 1) In-series interpolation in logit space for internal gaps.
		base = clip_share(sub["Industry"])
		logit = np.log(base / (100.0 - base))
		logit_interp = logit.interpolate(method="linear", limit_area="inside")
		inside_mask = sub["Industry"].isna() & logit_interp.notna()
		if inside_mask.any():
			sub.loc[inside_mask, "Industry"] = 100.0 / (1.0 + np.exp(-logit_interp.loc[inside_mask]))
			sub.loc[inside_mask, "Industry_source"] = "Provincial_industry_logit_interpolation"
			sub.loc[inside_mask, "Industry_is_national_proxy"] = 0
			sub.loc[inside_mask, "Industry_is_imputed"] = 1

		# 2) Extrapolation for leading/trailing gaps using province-specific logit-linear fit.
		remain_mask = sub["Industry"].isna()
		obs_mask = sub["Industry"].notna() & (sub["Industry"] > 0) & (sub["Industry"] < 100)
		if remain_mask.any() and int(obs_mask.sum()) >= 4:
			x = sub.loc[obs_mask, "year"].astype(float).to_numpy()
			y_share = clip_share(sub.loc[obs_mask, "Industry"]).astype(float).to_numpy()
			y = np.log(y_share / (100.0 - y_share))
			slope, intercept = np.polyfit(x, y, 1)
			xm = sub.loc[remain_mask, "year"].astype(float).to_numpy()
			pred_logit = intercept + slope * xm
			pred = 100.0 / (1.0 + np.exp(-pred_logit))
			sub.loc[remain_mask, "Industry"] = pred
			sub.loc[remain_mask, "Industry_source"] = "Provincial_industry_logit_linear_fit_extrapolation"
			sub.loc[remain_mask, "Industry_is_national_proxy"] = 0
			sub.loc[remain_mask, "Industry_is_imputed"] = 1
		elif remain_mask.any() and int(obs_mask.sum()) >= 2:
			obs = cast(pd.DataFrame, sub.loc[obs_mask, ["year", "Industry"]].sort_values("year").copy())
			x1 = float(obs.iloc[0]["year"])
			x2 = float(obs.iloc[1]["year"])
			y1_share = float(np.clip(obs.iloc[0]["Industry"], 0.01, 99.99))
			y2_share = float(np.clip(obs.iloc[1]["Industry"], 0.01, 99.99))
			y1 = np.log(y1_share / (100.0 - y1_share))
			y2 = np.log(y2_share / (100.0 - y2_share))
			slope = (y2 - y1) / (x2 - x1) if abs(x2 - x1) > 0 else 0.0
			xm = sub.loc[remain_mask, "year"].astype(float).to_numpy()
			pred_logit = y1 + slope * (xm - x1)
			pred = 100.0 / (1.0 + np.exp(-pred_logit))
			sub.loc[remain_mask, "Industry"] = pred
			sub.loc[remain_mask, "Industry_source"] = "Provincial_industry_logit_backcast_extrapolation"
			sub.loc[remain_mask, "Industry_is_national_proxy"] = 0
			sub.loc[remain_mask, "Industry_is_imputed"] = 1

		df.loc[
			sub.index,
			["Industry", "Industry_source", "Industry_is_national_proxy", "Industry_is_imputed"],
		] = sub[["Industry", "Industry_source", "Industry_is_national_proxy", "Industry_is_imputed"]]

	# 3) National anchoring for imputed rows only (year-level mean calibration).
	if industry_proxy_series is not None and not industry_proxy_series.data.empty:
		proxy = industry_proxy_series.data.rename(columns={"Industry": "Industry_national_ref"})
		df = df.merge(proxy, on="year", how="left")

		for year, idx in df.groupby("year").groups.items():
			nat_vals = df.loc[list(idx), "Industry_national_ref"].dropna()
			if nat_vals.empty:
				continue
			nat = float(nat_vals.iloc[0])
			year_mask = df["year"] == year
			imputed_mask = year_mask & (df["Industry_is_imputed"] == 1) & df["Industry"].notna()
			if not imputed_mask.any():
				continue
			mean_val = float(df.loc[year_mask & df["Industry"].notna(), "Industry"].mean())
			if not np.isfinite(mean_val) or mean_val <= 0:
				continue
			ratio = nat / mean_val
			df.loc[imputed_mask, "Industry"] = np.clip(df.loc[imputed_mask, "Industry"] * ratio, 0.01, 99.99)
			df.loc[imputed_mask, "Industry_source"] = (
				df.loc[imputed_mask, "Industry_source"].astype(str)
				+ "_national_anchor"
			)

		missing_industry = df["Industry"].isna()
		df.loc[missing_industry, "Industry"] = df.loc[missing_industry, "Industry_national_ref"]
		df.loc[missing_industry, "Industry_source"] = industry_proxy_series.source
		df.loc[missing_industry, "Industry_is_national_proxy"] = 1
		df.loc[missing_industry, "Industry_is_imputed"] = 0
		df = df.drop(columns=["Industry_national_ref"], errors="ignore")

	df["Industry"] = pd.to_numeric(df["Industry"], errors="coerce")
	df["Industry_is_national_proxy"] = df["Industry_is_national_proxy"].fillna(0)
	df["Industry_is_imputed"] = df["Industry_is_imputed"].fillna(0)
	return df


def safe_log(s: pd.Series) -> pd.Series:
	return np.log(s.where(s > 0))


def add_kaya_features(panel: pd.DataFrame) -> pd.DataFrame:
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


def attach_controls(panel: pd.DataFrame, controls: List[VariableSeries]) -> pd.DataFrame:
	df = panel.copy()
	for s in controls:
		if s.data.empty:
			continue
		df = df.merge(s.data, on="year", how="left")
		df[f"{s.name}_source"] = s.source
		df[f"{s.name}_is_national_proxy"] = 1

	if "Urbanization" not in df.columns:
		df["Urbanization"] = np.nan

	if "UrbanPopulation" in df.columns and "Population" in df.columns:
		calc = (df["UrbanPopulation"] / df["Population"]) * 100.0
		df["Urbanization"] = df["Urbanization"].fillna(calc)
		if "Urbanization_source" not in df.columns:
			df["Urbanization_source"] = "Derived_from_National_UrbanPopulation_over_Population"
			df["Urbanization_is_national_proxy"] = 1

	if "UrbanPopulation" in df.columns:
		df = df.drop(columns=["UrbanPopulation"], errors="ignore")
		df = df.drop(columns=["UrbanPopulation_source", "UrbanPopulation_is_national_proxy"], errors="ignore")

	return df


def fill_energy_with_interpolation_and_fit(panel: pd.DataFrame) -> pd.DataFrame:
	df = panel.copy()
	if "Energy" not in df.columns:
		df["Energy"] = np.nan
	if "Energy_source" not in df.columns:
		df["Energy_source"] = np.nan
	if "Energy_is_national_proxy" not in df.columns:
		df["Energy_is_national_proxy"] = np.nan
	if "Energy_is_imputed" not in df.columns:
		df["Energy_is_imputed"] = 0

	for province, idx in df.groupby("province").groups.items():
		sub = cast(pd.DataFrame, df.loc[list(idx), :].sort_values("year").copy())
		sub["Energy"] = pd.to_numeric(sub["Energy"], errors="coerce")
		sub.loc[sub["Energy"] <= 0, "Energy"] = np.nan

		# 1) In-series interpolation in log space for internal gaps.
		log_energy = np.log(sub["Energy"].where(sub["Energy"] > 0))
		log_interp = log_energy.interpolate(method="linear", limit_area="inside")
		inside_mask = sub["Energy"].isna() & log_interp.notna()
		if inside_mask.any():
			sub.loc[inside_mask, "Energy"] = np.exp(log_interp.loc[inside_mask])
			sub.loc[inside_mask, "Energy_source"] = "Provincial_energy_log_interpolation"
			sub.loc[inside_mask, "Energy_is_national_proxy"] = 0
			sub.loc[inside_mask, "Energy_is_imputed"] = 1

		# 2) Extrapolation for leading/trailing gaps using province-specific log-linear fit.
		remain_mask = sub["Energy"].isna()
		obs_mask = sub["Energy"].notna() & (sub["Energy"] > 0)
		if remain_mask.any() and int(obs_mask.sum()) >= 4:
			x = sub.loc[obs_mask, "year"].astype(float).to_numpy()
			y = np.log(sub.loc[obs_mask, "Energy"].astype(float).to_numpy())
			slope, intercept = np.polyfit(x, y, 1)
			xm = sub.loc[remain_mask, "year"].astype(float).to_numpy()
			pred = np.exp(intercept + slope * xm)
			sub.loc[remain_mask, "Energy"] = pred
			sub.loc[remain_mask, "Energy_source"] = "Provincial_energy_log_linear_fit_extrapolation"
			sub.loc[remain_mask, "Energy_is_national_proxy"] = 0
			sub.loc[remain_mask, "Energy_is_imputed"] = 1
		elif remain_mask.any() and int(obs_mask.sum()) >= 2:
			obs = cast(pd.DataFrame, sub.loc[obs_mask, ["year", "Energy"]].sort_values("year").copy())
			x1 = float(obs.iloc[0]["year"])
			x2 = float(obs.iloc[1]["year"])
			y1 = float(obs.iloc[0]["Energy"])
			y2 = float(obs.iloc[1]["Energy"])
			slope = (np.log(y2) - np.log(y1)) / (x2 - x1) if abs(x2 - x1) > 0 else 0.0
			xm = sub.loc[remain_mask, "year"].astype(float).to_numpy()
			pred = np.exp(np.log(y1) + slope * (xm - x1))
			sub.loc[remain_mask, "Energy"] = pred
			sub.loc[remain_mask, "Energy_source"] = "Provincial_energy_cagr_backcast_extrapolation"
			sub.loc[remain_mask, "Energy_is_national_proxy"] = 0
			sub.loc[remain_mask, "Energy_is_imputed"] = 1

		df.loc[sub.index, ["Energy", "Energy_source", "Energy_is_national_proxy", "Energy_is_imputed"]] = sub[
			["Energy", "Energy_source", "Energy_is_national_proxy", "Energy_is_imputed"]
		]

	df["Energy"] = pd.to_numeric(df["Energy"], errors="coerce")
	return df


def main() -> None:
	ensure_output_dir()
	items = load_dataset_index()

	co2_panel = build_co2_panel(items)
	controls = build_national_controls(items)
	energy_proxy_series = next((s for s in controls if s.name == "Energy"), None)
	industry_proxy_series = next((s for s in controls if s.name == "Industry"), None)
	non_energy_controls = [s for s in controls if s.name not in {"Energy", "Industry"}]

	panel = attach_controls(co2_panel, non_energy_controls)

	prov_industry = read_provincial_industry_share(items)
	if not prov_industry.empty:
		panel = panel.merge(prov_industry, on=["province", "year"], how="left")
	else:
		if "Industry" not in panel.columns:
			panel["Industry"] = np.nan
		if "Industry_source" not in panel.columns:
			panel["Industry_source"] = np.nan
		if "Industry_is_national_proxy" not in panel.columns:
			panel["Industry_is_national_proxy"] = np.nan
	if "Industry_is_imputed" not in panel.columns:
		panel["Industry_is_imputed"] = 0

	panel = fill_industry_with_interpolation_fit_and_anchor(panel, industry_proxy_series)

	prov_energy = read_provincial_energy_inventory(items)
	if not prov_energy.empty:
		panel = panel.merge(prov_energy, on=["province", "year"], how="left")
	else:
		if "Energy" not in panel.columns:
			panel["Energy"] = np.nan
		if "Energy_source" not in panel.columns:
			panel["Energy_source"] = np.nan
		if "Energy_is_national_proxy" not in panel.columns:
			panel["Energy_is_national_proxy"] = np.nan
	if "Energy_is_imputed" not in panel.columns:
		panel["Energy_is_imputed"] = 0

	panel = fill_energy_with_interpolation_and_fit(panel)

	if energy_proxy_series is not None and not energy_proxy_series.data.empty:
		energy_proxy_df = energy_proxy_series.data.rename(columns={"Energy": "Energy_proxy"})
		panel = panel.merge(energy_proxy_df, on="year", how="left")

		if "Energy" not in panel.columns:
			panel["Energy"] = np.nan
		if "Energy_source" not in panel.columns:
			panel["Energy_source"] = np.nan
		if "Energy_is_national_proxy" not in panel.columns:
			panel["Energy_is_national_proxy"] = np.nan

		missing_energy = panel["Energy"].isna()
		panel.loc[missing_energy, "Energy"] = panel.loc[missing_energy, "Energy_proxy"]
		panel.loc[missing_energy, "Energy_source"] = energy_proxy_series.source
		panel.loc[missing_energy, "Energy_is_national_proxy"] = 1
		panel.loc[missing_energy, "Energy_is_imputed"] = 0
		panel.loc[~missing_energy & panel["Energy"].notna(), "Energy_is_national_proxy"] = panel.loc[
			~missing_energy & panel["Energy"].notna(), "Energy_is_national_proxy"
		].fillna(0)

		panel = panel.drop(columns=["Energy_proxy"], errors="ignore")

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
	panel_core = cast(pd.DataFrame, panel_core.sort_values(by="year").sort_values(by="province").reset_index(drop=True))

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
		"energy_source_counts": panel["Energy_source"].value_counts(dropna=False).to_dict()
		if "Energy_source" in panel.columns
		else {},
		"industry_source_counts": panel["Industry_source"].value_counts(dropna=False).to_dict()
		if "Industry_source" in panel.columns
		else {},
		"notes": [
			"CO2 source priority applied: NBS(2001-2022) > MEIC(1990-2000).",
			"Energy uses provincial inventory (1997-2022); internal gaps are log-interpolated and leading gaps are backcast by province log-linear fit.",
			"If energy still missing after imputation, national proxy fallback is applied.",
			"Industry uses provincial share series (observed mostly from 1996 onward), with logit interpolation/backcast and national year-level anchoring for imputed years.",
			"Urbanization remains national proxy because currently available provincial urbanization file has sparse early-year coverage.",
		],
	}
	summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

	print("WROTE", panel_path)
	print("WROTE", lmdi_path)
	print("WROTE", audit_path)
	print("WROTE", summary_path)
	print("PANEL_ROWS", summary["panel_rows"], "PROVINCES", summary["province_count"], "YEARS", summary["year_min"], summary["year_max"])
	print("CO2_SOURCE_COUNTS", summary["co2_source_counts"])


if __name__ == "__main__":
	main()