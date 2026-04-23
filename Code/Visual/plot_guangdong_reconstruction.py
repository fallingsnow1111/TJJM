from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Ensure imports work when running as: python Code/Visual/plot_guangdong_reconstruction.py
CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from Model.stirpat_ee_gru import EntityEmbeddingGRU


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize original vs hybrid-predicted CO2 by recursive model inference."
    )
    parser.add_argument("--province", default="Guangdong", help="Province name in panel CSV")
    parser.add_argument("--start-year", type=int, default=1990, help="Start year")
    parser.add_argument("--end-year", type=int, default=2023, help="End year")
    parser.add_argument(
        "--input-csv",
        default=str(Path(__file__).resolve().parents[1] / "Dataset" / "output" / "panel_with_residual.csv"),
        help="Input panel_with_residual CSV path",
    )
    parser.add_argument(
        "--dataset-npz",
        default=str(Path(__file__).resolve().parents[1] / "Dataset" / "output" / "stirpat_ee_gru_dataset.npz"),
        help="Dataset NPZ path that stores feature order and standardizer stats",
    )
    parser.add_argument(
        "--model-ckpt",
        default=str(Path(__file__).resolve().parents[1] / "Model" / "output" / "best_ee_gru.pt"),
        help="Trained hybrid model checkpoint path",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "output"),
        help="Directory to save chart and extracted data",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device")
    return parser


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_ckpt: Path, device: torch.device) -> EntityEmbeddingGRU:
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


def main() -> None:
    args = build_parser().parse_args()

    input_csv = Path(args.input_csv)
    dataset_npz = Path(args.dataset_npz)
    model_ckpt = Path(args.model_ckpt)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if not dataset_npz.exists():
        raise FileNotFoundError(f"Dataset NPZ not found: {dataset_npz}")
    if not model_ckpt.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_ckpt}")

    df = pd.read_csv(input_csv)
    npz_data = np.load(dataset_npz, allow_pickle=True)
    model = load_model(model_ckpt=model_ckpt, device=device)

    required_cols = ["province", "province_id", "year", "CO2", "stirpat_log_pred"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in input CSV: {missing_cols}")

    if "gru_feature_names" not in npz_data or "standardizer_mean" not in npz_data or "standardizer_std" not in npz_data:
        raise KeyError("NPZ missing required metadata arrays: gru_feature_names/standardizer_mean/standardizer_std")
    if "train_dynamic_x" not in npz_data:
        raise KeyError("NPZ missing train_dynamic_x; cannot infer GRU window length")

    gru_feature_names = [str(x) for x in npz_data["gru_feature_names"].tolist()]
    mean = npz_data["standardizer_mean"].astype(np.float32)
    std = npz_data["standardizer_std"].astype(np.float32)
    window = int(npz_data["train_dynamic_x"].shape[1])

    if len(gru_feature_names) != len(mean) or len(mean) != len(std):
        raise ValueError("Feature metadata dimension mismatch in NPZ")

    feat_missing = [c for c in gru_feature_names if c not in df.columns]
    if feat_missing:
        raise ValueError(f"Missing GRU feature columns in input CSV: {feat_missing}")

    province_df = df.loc[
        (df["province"] == args.province)
        & (df["year"].between(args.start_year, args.end_year))
    ].copy()

    if province_df.empty:
        raise ValueError(
            f"No rows found for province={args.province}, years={args.start_year}-{args.end_year}."
        )

    province_df = province_df.sort_values("year", kind="stable").reset_index(drop=True)
    province_df["year"] = province_df["year"].astype(int)
    province_df["true_co2"] = province_df["CO2"].astype(float)
    province_df["stirpat_co2"] = province_df["stirpat_log_pred"].astype(float).map(lambda x: float(np.exp(x)))

    dynamic_raw = province_df[gru_feature_names].to_numpy(dtype=np.float32)
    dynamic_std = (dynamic_raw - mean) / std
    province_idx = int(province_df["province_id"].iloc[0])

    pred_residual = np.zeros(len(province_df), dtype=np.float32)
    pred_log_co2 = np.zeros(len(province_df), dtype=np.float32)

    with torch.no_grad():
        for i in range(len(province_df)):
            dyn_seq = np.zeros((window, len(gru_feature_names)), dtype=np.float32)
            lag_seq = np.zeros((window, 1), dtype=np.float32)

            for j in range(window):
                src_idx = i - window + j
                if src_idx < 0:
                    dyn_seq[j] = dynamic_std[0]
                    lag_seq[j, 0] = 0.0
                else:
                    dyn_seq[j] = dynamic_std[src_idx]
                    lag_seq[j, 0] = pred_residual[src_idx]

            x_dyn = torch.tensor(dyn_seq[None, :, :], dtype=torch.float32, device=device)
            x_lag = torch.tensor(lag_seq[None, :, :], dtype=torch.float32, device=device)
            x_pid = torch.tensor([province_idx], dtype=torch.long, device=device)

            res_hat = float(model(x_dyn, x_lag, x_pid).cpu().item())
            pred_residual[i] = res_hat
            pred_log_co2[i] = float(province_df.loc[i, "stirpat_log_pred"]) + res_hat

    province_df["pred_residual"] = pred_residual
    province_df["pred_log_co2"] = pred_log_co2
    province_df["pred_co2"] = province_df["pred_log_co2"].map(lambda x: float(np.exp(x)))

    mae = (province_df["true_co2"] - province_df["pred_co2"]).abs().mean()
    mape = ((province_df["true_co2"] - province_df["pred_co2"]).abs() / province_df["true_co2"]).mean() * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        province_df["year"],
        province_df["true_co2"],
        color="#1f77b4",
        linewidth=2.2,
        marker="o",
        markersize=4,
        label="Original CO2",
    )
    ax.plot(
        province_df["year"],
        province_df["stirpat_co2"],
        color="#6c757d",
        linewidth=1.8,
        linestyle=":",
        label="STIRPAT Baseline",
    )
    ax.plot(
        province_df["year"],
        province_df["pred_co2"],
        color="#d62728",
        linewidth=2.2,
        marker="s",
        markersize=4,
        linestyle="--",
        label="Predicted CO2 (Hybrid, Recursive Inference)",
    )

    ax.set_title(
        f"{args.province} CO2: Original vs Predicted ({args.start_year}-{args.end_year})",
        fontsize=13,
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("CO2")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend()

    metric_text = f"MAE={mae:.3f}    MAPE={mape:.2f}%    window={window}"
    ax.text(
        0.01,
        0.98,
        metric_text,
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="#bbbbbb"),
    )

    figure_path = output_dir / f"{args.province}_{args.start_year}_{args.end_year}_original_vs_pred.png"
    data_path = output_dir / f"{args.province}_{args.start_year}_{args.end_year}_original_vs_pred.csv"

    fig.tight_layout()
    fig.savefig(str(figure_path), dpi=300)
    plt.close(fig)

    province_df[
        [
            "province",
            "year",
            "true_co2",
            "stirpat_co2",
            "pred_residual",
            "pred_log_co2",
            "pred_co2",
        ]
    ].to_csv(data_path, index=False)

    print(f"Saved figure: {figure_path}")
    print(f"Saved data:   {data_path}")
    print(f"Rows: {len(province_df)}; MAE={mae:.3f}; MAPE={mape:.2f}%")
    print(f"Inference uses recursive residual prediction only (no true residual labels). Device: {device}")


if __name__ == "__main__":
    main()
