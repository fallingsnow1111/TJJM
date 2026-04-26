from __future__ import annotations

"""Build publication-style LMDI figures from the precomputed outputs.

This script intentionally stays separate from preprocessing. It only reads the
final CSV outputs and writes figures into Code/LMDI/output/figures.
"""

from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
LMDI_OUTPUT_DIR = ROOT / "Code" / "LMDI" / "output"
FIGURE_DIR = LMDI_OUTPUT_DIR / "figures"
LMDI_PATH = LMDI_OUTPUT_DIR / "lmdi_decomposition.csv"
PANEL_PATH = ROOT / "Code" / "STIRPAT" / "output" / "dataset" /"panel_with_residual.csv"

FACTOR_ORDER = ["delta_P", "delta_A", "delta_B", "delta_C"]
FACTOR_LABELS = {
	"delta_P": "Population",
	"delta_A": "GDP per capita",
	"delta_B": "Energy intensity",
	"delta_C": "Emission factor",
}
FACTOR_COLORS = {
	"delta_P": "#4C78A8",
	"delta_A": "#F58518",
	"delta_B": "#72B7B2",
	"delta_C": "#9D755D",
}

REGION_MAP = {
	"Beijing": "East",
	"Tianjin": "East",
	"Hebei": "East",
	"Liaoning": "East",
	"Shanghai": "East",
	"Jiangsu": "East",
	"Zhejiang": "East",
	"Fujian": "East",
	"Shandong": "East",
	"Guangdong": "East",
	"Hainan": "East",
	"Shanxi": "Central",
	"Jilin": "Central",
	"Heilongjiang": "Central",
	"Anhui": "Central",
	"Jiangxi": "Central",
	"Henan": "Central",
	"Hubei": "Central",
	"Hunan": "Central",
}

PERIODS: List[Tuple[str, int, int]] = [
	("1991-1995", 1991, 1995),
	("1996-2000", 1996, 2000),
	("2001-2005", 2001, 2005),
	("2006-2010", 2006, 2010),
	("2011-2015", 2011, 2015),
	("2016-2020", 2016, 2020),
	("2021-2022", 2021, 2022),
]


def ensure_dirs() -> None:
	FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
	if not LMDI_PATH.exists():
		raise FileNotFoundError(f"Missing input: {LMDI_PATH}")
	if not PANEL_PATH.exists():
		raise FileNotFoundError(f"Missing input: {PANEL_PATH}")
	lmdi = pd.read_csv(LMDI_PATH)
	panel = pd.read_csv(PANEL_PATH)
	return lmdi, panel


def annual_totals(lmdi: pd.DataFrame) -> pd.DataFrame:
	cols = FACTOR_ORDER + ["delta_CO2"]
	annual = lmdi.groupby("year", as_index=False)[cols].sum()
	annual = annual.sort_values("year").reset_index(drop=True)
	return annual


def province_totals(lmdi: pd.DataFrame) -> pd.DataFrame:
	cols = FACTOR_ORDER + ["delta_CO2"]
	province = lmdi.groupby("province", as_index=False)[cols].sum()
	province["mitigation_score"] = -(province["delta_B"] + province["delta_C"])
	province["net_change"] = province["delta_CO2"]
	province = province.sort_values("mitigation_score", ascending=False).reset_index(drop=True)
	return province


def period_totals(lmdi: pd.DataFrame) -> pd.DataFrame:
	rows = []
	for label, start, end in PERIODS:
		sub = lmdi[(lmdi["year"] >= start) & (lmdi["year"] <= end)]
		tot = sub[FACTOR_ORDER + ["delta_CO2"]].sum()
		row = {"period": label}
		row.update({col: float(tot[col]) for col in FACTOR_ORDER + ["delta_CO2"]})
		rows.append(row)
	return pd.DataFrame(rows)


def region_totals(lmdi: pd.DataFrame) -> pd.DataFrame:
	df = lmdi.copy()
	df["region"] = df["province"].map(REGION_MAP).fillna("West")
	return df.groupby("region", as_index=False)[FACTOR_ORDER + ["delta_CO2"]].sum()


def add_period_bands(ax: plt.Axes) -> None:
	bands = [
		(1991, 1995, "8th FYP"),
		(1996, 2000, "9th FYP"),
		(2001, 2005, "10th FYP"),
		(2006, 2010, "11th FYP"),
		(2011, 2015, "12th FYP"),
		(2016, 2020, "13th FYP"),
		(2021, 2022, "14th FYP"),
	]
	for idx, (start, end, label) in enumerate(bands):
		alpha = 0.04 if idx % 2 == 0 else 0.08
		ax.axvspan(start - 0.5, end + 0.5, color="#D8E3F0", alpha=alpha, lw=0)
		if end - start >= 2:
			ax.text((start + end) / 2, 0.985, label, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8.5, color="#222222")


def make_waterfall(annual: pd.DataFrame, out_path: Path) -> None:
	totals = annual[FACTOR_ORDER].sum()
	net_total = float(annual["delta_CO2"].sum())
	labels = [FACTOR_LABELS[c] for c in FACTOR_ORDER] + ["Net"]
	values = [float(totals[c]) for c in FACTOR_ORDER] + [net_total]

	fig, ax = plt.subplots(figsize=(11, 6.2))
	cumulative = 0.0
	for idx, (_label, value) in enumerate(zip(labels[:-1], values[:-1])):
		start = cumulative
		cumulative += value
		bottom = min(start, cumulative)
		height = abs(value)
		color = FACTOR_COLORS[FACTOR_ORDER[idx]]
		ax.bar(idx, height, bottom=bottom, width=0.68, color=color, edgecolor="white", linewidth=0.8)
		ax.plot([idx + 0.34, idx + 0.66], [cumulative, cumulative], color="#666666", lw=1.0)
		ax.text(idx, cumulative + (0.01 * max(abs(net_total), 1.0)), f"{value:,.1f}", ha="center", va="bottom", fontsize=9)

	ax.bar(len(labels) - 1, net_total, width=0.68, color="#2F2F2F", edgecolor="white", linewidth=0.8)
	ax.text(len(labels) - 1, net_total + (0.01 * max(abs(net_total), 1.0)), f"{net_total:,.1f}", ha="center", va="bottom", fontsize=9, color="#2F2F2F")
	ax.axhline(0, color="#444444", lw=0.8)
	ax.set_xticks(range(len(labels)))
	ax.set_xticklabels(labels, rotation=0)
	ax.set_ylabel(r"Cumulative $\Delta CO_2$")
	ax.set_title("Cumulative LMDI effects, 1990-2022")
	ax.grid(axis="y", alpha=0.2)
	ax.spines[["top", "right"]].set_visible(False)
	fig.tight_layout()
	fig.savefig(out_path, dpi=300, bbox_inches="tight")
	plt.close(fig)


def make_stacked_timeseries(annual: pd.DataFrame, out_path: Path) -> None:
	years = annual["year"].to_numpy()
	fig, ax = plt.subplots(figsize=(14, 7.4))
	pos_bottom = np.zeros(len(annual), dtype=float)
	neg_bottom = np.zeros(len(annual), dtype=float)

	for col in FACTOR_ORDER:
		values = annual[col].to_numpy(dtype=float)
		positive = np.clip(values, 0, None)
		negative = np.clip(values, None, 0)
		ax.bar(years, positive, bottom=pos_bottom, color=FACTOR_COLORS[col], width=0.8, label=FACTOR_LABELS[col], alpha=0.9)
		ax.bar(years, negative, bottom=neg_bottom, color=FACTOR_COLORS[col], width=0.8, alpha=0.9)
		pos_bottom += positive
		neg_bottom += negative

	net = annual["delta_CO2"].to_numpy(dtype=float)
	ax.plot(years, net, color="#1F1F1F", lw=2.0, marker="o", ms=3.5, label="Net change")
	add_period_bands(ax)
	ax.axhline(0, color="#333333", lw=0.9)
	ax.set_xlim(years.min() - 0.8, years.max() + 0.8)
	ax.set_xlabel("Year")
	ax.set_ylabel(r"Annual $\Delta CO_2$")
	ax.set_title("Annual stacked decomposition with net change line", pad=18)
	ax.legend(ncol=3, frameon=False, loc="upper left", bbox_to_anchor=(0.01, 0.965))
	ax.grid(axis="y", alpha=0.2)
	ax.spines[["top", "right"]].set_visible(False)
	fig.tight_layout()
	fig.savefig(out_path, dpi=300, bbox_inches="tight")
	plt.close(fig)


def make_spatial_chart(province: pd.DataFrame, out_path: Path) -> None:
	ordered = province.sort_values(["mitigation_score", "net_change"], ascending=[False, False]).reset_index(drop=True)
	provinces = ordered["province"].tolist()
	y = np.arange(len(ordered))

	fig, ax = plt.subplots(figsize=(13.5, 13))
	pos_left = np.zeros(len(ordered), dtype=float)
	neg_left = np.zeros(len(ordered), dtype=float)

	for col in FACTOR_ORDER:
		values = ordered[col].to_numpy(dtype=float)
		positive = np.clip(values, 0, None)
		negative = np.clip(values, None, 0)
		ax.barh(y, positive, left=pos_left, color=FACTOR_COLORS[col], edgecolor="white", linewidth=0.5, label=FACTOR_LABELS[col], alpha=0.92)
		ax.barh(y, negative, left=neg_left, color=FACTOR_COLORS[col], edgecolor="white", linewidth=0.5, alpha=0.92)
		pos_left += positive
		neg_left += negative

	ax.axvline(0, color="#333333", lw=0.9)
	ax.set_yticks(y)
	ax.set_yticklabels(provinces, fontsize=9)
	ax.invert_yaxis()
	ax.set_xlabel(r"Cumulative $\Delta CO_2$")
	ax.set_title("Spatial heterogeneity of cumulative LMDI effects, 1990-2022")
	ax.legend(ncol=3, frameon=False, loc="lower right")
	ax.grid(axis="x", alpha=0.18)
	ax.spines[["top", "right"]].set_visible(False)
	fig.tight_layout()
	fig.savefig(out_path, dpi=300, bbox_inches="tight")
	plt.close(fig)


def make_period_heatmap(period: pd.DataFrame, out_path: Path) -> None:
	matrix = period.set_index("period")[FACTOR_ORDER].T
	values = matrix.to_numpy(dtype=float)
	limit = np.nanmax(np.abs(values)) if values.size else 1.0

	fig, ax = plt.subplots(figsize=(12.5, 4.8))
	im = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
	ax.set_xticks(np.arange(matrix.shape[1]))
	ax.set_xticklabels(matrix.columns, rotation=35, ha="right")
	ax.set_yticks(np.arange(matrix.shape[0]))
	ax.set_yticklabels([FACTOR_LABELS[c] for c in matrix.index])
	ax.set_title("LMDI contribution heatmap by development period")

	for i in range(matrix.shape[0]):
		for j in range(matrix.shape[1]):
			v = values[i, j]
			text_color = "white" if abs(v) > 0.55 * limit else "#222222"
			ax.text(j, i, f"{v:,.0f}", ha="center", va="center", fontsize=8, color=text_color)

	cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
	cbar.set_label(r"$\Delta CO_2$")
	fig.tight_layout(rect=(0, 0, 1, 0.96))
	fig.savefig(out_path, dpi=300, bbox_inches="tight")
	plt.close(fig)


def make_region_stacked(region: pd.DataFrame, out_path: Path) -> None:
	ordered = region.copy()
	ordered["region"] = pd.Categorical(ordered["region"], categories=["East", "Central", "West"], ordered=True)
	ordered = ordered.sort_values("region").reset_index(drop=True)
	x = np.arange(len(ordered))

	fig, ax = plt.subplots(figsize=(12.2, 6.2))
	pos_bottom = np.zeros(len(ordered), dtype=float)
	neg_bottom = np.zeros(len(ordered), dtype=float)
	for col in FACTOR_ORDER:
		values = ordered[col].to_numpy(dtype=float)
		positive = np.clip(values, 0, None)
		negative = np.clip(values, None, 0)
		ax.bar(x, positive, bottom=pos_bottom, color=FACTOR_COLORS[col], width=0.62, label=FACTOR_LABELS[col])
		ax.bar(x, negative, bottom=neg_bottom, color=FACTOR_COLORS[col], width=0.62)
		pos_bottom += positive
		neg_bottom += negative

	ax.plot(x, ordered["delta_CO2"], color="#1F1F1F", marker="o", linewidth=2.0, label="Net change")
	ax.axhline(0, color="#333333", linewidth=0.9)
	ax.set_xticks(x)
	ax.set_xticklabels(ordered["region"])
	ax.set_ylabel(r"Cumulative $\Delta CO_2$")
	ax.set_title("Regional heterogeneity of cumulative LMDI effects")
	ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
	ax.grid(axis="y", alpha=0.2)
	ax.spines[["top", "right"]].set_visible(False)
	fig.tight_layout(rect=(0, 0, 0.82, 1))
	fig.savefig(out_path, dpi=300, bbox_inches="tight")
	plt.close(fig)


def make_province_quadrant(province: pd.DataFrame, out_path: Path) -> None:
	df = province.copy()
	df["growth_pressure"] = df["delta_P"] + df["delta_A"]
	df["mitigation_effect"] = -(df["delta_B"] + df["delta_C"])
	df["region"] = df["province"].map(REGION_MAP).fillna("West")
	colors = {"East": "#4C78A8", "Central": "#F58518", "West": "#54A24B"}

	fig, ax = plt.subplots(figsize=(10.5, 7.2))
	for region, grp in df.groupby("region", sort=False):
		ax.scatter(
			grp["growth_pressure"],
			grp["mitigation_effect"],
			s=np.clip(np.abs(grp["net_change"]) * 0.35, 25, 260),
			color=colors.get(region, "#777777"),
			alpha=0.72,
			edgecolor="white",
			linewidth=0.7,
			label=region,
		)

	top_labels = df.assign(label_score=np.abs(df["net_change"]) + df["mitigation_effect"]).nlargest(10, "label_score")
	for _, row in top_labels.iterrows():
		ax.text(row["growth_pressure"], row["mitigation_effect"], str(row["province"]), fontsize=8, ha="left", va="bottom")

	ax.axvline(df["growth_pressure"].median(), color="#777777", linestyle="--", linewidth=0.9)
	ax.axhline(df["mitigation_effect"].median(), color="#777777", linestyle="--", linewidth=0.9)
	ax.set_xlabel("Growth pressure: Population + GDP per capita effects")
	ax.set_ylabel("Mitigation effect: -(Energy intensity + Emission factor effects)")
	ax.set_title("Province typology from LMDI growth pressure and mitigation effect")
	ax.grid(alpha=0.22, linestyle="--")
	ax.legend(frameon=False, loc="best")
	ax.spines[["top", "right"]].set_visible(False)
	fig.tight_layout()
	fig.savefig(out_path, dpi=300, bbox_inches="tight")
	plt.close(fig)


def write_run_summary(annual: pd.DataFrame, province: pd.DataFrame) -> None:
	summary_path = LMDI_OUTPUT_DIR / "lmdi_visual_summary.md"
	full = annual[FACTOR_ORDER + ["delta_CO2"]].sum()
	top = province.head(8)[["province", "mitigation_score", "net_change"]]
	bottom = province.sort_values("net_change", ascending=True).head(8)[["province", "mitigation_score", "net_change"]]
	lines: List[str] = []
	lines.append("# LMDI Figure Summary")
	lines.append("")
	lines.append(f"- Cumulative totals: P {full['delta_P']:.3f}, A {full['delta_A']:.3f}, B {full['delta_B']:.3f}, C {full['delta_C']:.3f}, net {full['delta_CO2']:.3f}")
	lines.append(f"- Figure directory: {FIGURE_DIR}")
	lines.append("")
	lines.append("## Highest mitigation score provinces")
	for _, row in top.iterrows():
		lines.append(f"- {row['province']}: mitigation_score={row['mitigation_score']:.3f}, net_change={row['net_change']:.3f}")
	lines.append("")
	lines.append("## Lowest net-change provinces")
	for _, row in bottom.iterrows():
		lines.append(f"- {row['province']}: mitigation_score={row['mitigation_score']:.3f}, net_change={row['net_change']:.3f}")
	lines.append("")
	lines.append("## Suggested figure use")
	lines.append("- Main text: stacked_timeseries.png, period_heatmap.png, region_stacked.png")
	lines.append("- Supplement: cumulative_waterfall.png, spatial_heterogeneity.png, province_quadrant.png")
	summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
	ensure_dirs()
	lmdi, _panel = load_inputs()
	annual = annual_totals(lmdi)
	province = province_totals(lmdi)
	period = period_totals(lmdi)
	region = region_totals(lmdi)

	make_waterfall(annual, FIGURE_DIR / "cumulative_waterfall.png")
	make_stacked_timeseries(annual, FIGURE_DIR / "stacked_timeseries.png")
	make_spatial_chart(province, FIGURE_DIR / "spatial_heterogeneity.png")
	make_period_heatmap(period, FIGURE_DIR / "period_heatmap.png")
	make_region_stacked(region, FIGURE_DIR / "region_stacked.png")
	make_province_quadrant(province, FIGURE_DIR / "province_quadrant.png")
	write_run_summary(annual, province)

	print("WROTE", FIGURE_DIR / "cumulative_waterfall.png")
	print("WROTE", FIGURE_DIR / "stacked_timeseries.png")
	print("WROTE", FIGURE_DIR / "spatial_heterogeneity.png")
	print("WROTE", FIGURE_DIR / "period_heatmap.png")
	print("WROTE", FIGURE_DIR / "region_stacked.png")
	print("WROTE", FIGURE_DIR / "province_quadrant.png")
	print("WROTE", LMDI_OUTPUT_DIR / "lmdi_visual_summary.md")


if __name__ == "__main__":
	main()
