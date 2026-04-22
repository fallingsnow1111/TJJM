from __future__ import annotations

"""Build publication-style LMDI figures from the precomputed outputs.

This script intentionally stays separate from preprocessing. It only reads the
final CSV outputs and writes figures into Code/output/figures.
"""

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "Code" / "output"
FIGURE_DIR = OUTPUT_DIR / "figures"
LMDI_PATH = OUTPUT_DIR / "lmdi_decomposition.csv"
PANEL_PATH = OUTPUT_DIR / "panel_master.csv"

FACTOR_ORDER = ["delta_P", "delta_A", "delta_S", "delta_B", "delta_C"]
FACTOR_LABELS = {
	"delta_P": "Population",
	"delta_A": "GDP per capita",
	"delta_S": "Industry structure",
	"delta_B": "Energy intensity",
	"delta_C": "Emission factor",
}
FACTOR_COLORS = {
	"delta_P": "#4C78A8",
	"delta_A": "#F58518",
	"delta_S": "#54A24B",
	"delta_B": "#72B7B2",
	"delta_C": "#9D755D",
}


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
	province["mitigation_score"] = -(province["delta_S"] + province["delta_B"])
	province["net_change"] = province["delta_CO2"]
	province = province.sort_values("mitigation_score", ascending=False).reset_index(drop=True)
	return province


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
			ax.text((start + end) / 2, 1.02, label, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=9)


def make_waterfall(annual: pd.DataFrame, out_path: Path) -> None:
	totals = annual[FACTOR_ORDER].sum()
	net_total = float(annual["delta_CO2"].sum())
	labels = [FACTOR_LABELS[c] for c in FACTOR_ORDER] + ["Net"]
	values = [float(totals[c]) for c in FACTOR_ORDER] + [net_total]

	fig, ax = plt.subplots(figsize=(11, 6.2))
	cumulative = 0.0
	for idx, (label, value) in enumerate(zip(labels[:-1], values[:-1])):
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
	fig, ax = plt.subplots(figsize=(14, 7))
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
	ax.set_title("Annual stacked decomposition with net change line")
	ax.legend(ncol=3, frameon=False, loc="upper left", bbox_to_anchor=(0.01, 1.01))
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


def write_run_summary(annual: pd.DataFrame, province: pd.DataFrame) -> None:
	summary_path = OUTPUT_DIR / "lmdi_visual_summary.md"
	full = annual[FACTOR_ORDER + ["delta_CO2"]].sum()
	top = province.head(8)[["province", "mitigation_score", "net_change"]]
	bottom = province.sort_values("net_change", ascending=True).head(8)[["province", "mitigation_score", "net_change"]]
	lines: List[str] = []
	lines.append("# LMDI Figure Summary")
	lines.append("")
	lines.append(f"- Cumulative totals: P {full['delta_P']:.3f}, A {full['delta_A']:.3f}, S {full['delta_S']:.3f}, B {full['delta_B']:.3f}, C {full['delta_C']:.3f}, net {full['delta_CO2']:.3f}")
	lines.append(f"- Figure directory: {FIGURE_DIR}")
	lines.append("")
	lines.append("## Highest mitigation score provinces")
	for _, row in top.iterrows():
		lines.append(f"- {row['province']}: mitigation_score={row['mitigation_score']:.3f}, net_change={row['net_change']:.3f}")
	lines.append("")
	lines.append("## Lowest net-change provinces")
	for _, row in bottom.iterrows():
		lines.append(f"- {row['province']}: mitigation_score={row['mitigation_score']:.3f}, net_change={row['net_change']:.3f}")
	summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
	ensure_dirs()
	lmdi, _panel = load_inputs()
	annual = annual_totals(lmdi)
	province = province_totals(lmdi)

	make_waterfall(annual, FIGURE_DIR / "cumulative_waterfall.png")
	make_stacked_timeseries(annual, FIGURE_DIR / "stacked_timeseries.png")
	make_spatial_chart(province, FIGURE_DIR / "spatial_heterogeneity.png")
	write_run_summary(annual, province)

	print("WROTE", FIGURE_DIR / "cumulative_waterfall.png")
	print("WROTE", FIGURE_DIR / "stacked_timeseries.png")
	print("WROTE", FIGURE_DIR / "spatial_heterogeneity.png")
	print("WROTE", OUTPUT_DIR / "lmdi_visual_summary.md")


if __name__ == "__main__":
	main()
