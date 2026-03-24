# -*- coding: utf-8 -*-
"""Plot 24-step vs 120-step comparison for the current AGC hybrid-transformer."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from benchmark_current_hybrid_transformer import (
    _apply_fair_budget_overrides,
    _build_bundle,
    _set_global_seed,
)
from config import AGCConfig
from data_processing.processor import AGCDataProcessor
from models.transformer_hybrid_forecaster import ConditionalTransformerHybridForecaster


PROJECT_ROOT = Path(__file__).resolve().parent
FIGURES_DIR = PROJECT_ROOT / "results" / "forecasting" / "figures"
ANALYSIS_DIR = PROJECT_ROOT / "results" / "forecasting" / "analysis"

TARGET_SPECS = [
    ("Tair", "温度", "°C"),
    ("Rhair", "湿度", "%"),
    ("CO2air", "CO2", "ppm"),
]


def _setup_matplotlib_chinese() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_window(horizon: int, checkpoint_name: str) -> dict:
    cfg = _apply_fair_budget_overrides(AGCConfig())
    cfg.horizon = horizon
    _set_global_seed(cfg.seed)
    processor = AGCDataProcessor(cfg)
    bundle = _build_bundle(processor, cfg, "joint_all", "Reference")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConditionalTransformerHybridForecaster(
        past_dim=bundle["X_past_train"].shape[-1],
        weather_dim=bundle["W_future_train"].shape[-1],
        control_dim=bundle["U_future_train"].shape[-1],
        target_dim=bundle["Y_future_train"].shape[-1],
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        nhead=cfg.transformer_heads,
        ff_dim=cfg.transformer_ff_dim,
        max_past_len=cfg.seq_len,
        max_future_len=cfg.horizon,
    )
    checkpoint_path = PROJECT_ROOT / "results" / "forecasting" / "checkpoints" / checkpoint_name
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    sample_idx = len(bundle["X_past_test"]) // 2
    xb = torch.tensor(bundle["X_past_test"][sample_idx : sample_idx + 1], dtype=torch.float32, device=device)
    wb = torch.tensor(bundle["W_future_test"][sample_idx : sample_idx + 1], dtype=torch.float32, device=device)
    ub = torch.tensor(bundle["U_future_test"][sample_idx : sample_idx + 1], dtype=torch.float32, device=device)

    with torch.no_grad():
        pred_norm = model(xb, wb, ub).cpu().numpy()[0]
    true_norm = bundle["Y_future_test"][sample_idx]
    pred_real = bundle["scalers"]["y"].inverse_transform(pred_norm)
    true_real = bundle["scalers"]["y"].inverse_transform(true_norm)
    return {
        "sample_idx": int(sample_idx),
        "true": true_real,
        "pred": pred_real,
    }


def _plot_metrics(summary_24: dict, summary_120: dict, out_path: Path) -> None:
    x = np.arange(len(TARGET_SPECS))
    labels = [zh_name for _, zh_name, _ in TARGET_SPECS]
    mae_24 = [summary_24["metrics_by_target"][key]["final_mae"] for key, _, _ in TARGET_SPECS]
    mae_120 = [summary_120["metrics_by_target"][key]["final_mae"] for key, _, _ in TARGET_SPECS]
    r2_24 = [summary_24["metrics_by_target"][key]["final_r2"] for key, _, _ in TARGET_SPECS]
    r2_120 = [summary_120["metrics_by_target"][key]["final_r2"] for key, _, _ in TARGET_SPECS]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    mae_ax, r2_ax = axes

    mae_ax.plot(x, mae_24, marker="o", linewidth=2.6, color="#1f9d73", label="24-step (120 min)")
    mae_ax.plot(x, mae_120, marker="o", linewidth=2.6, color="#c85c3a", label="120-step (600 min)")
    mae_ax.set_title("最终步 MAE 对比")
    mae_ax.set_xticks(x, labels)
    mae_ax.set_ylabel("MAE")
    mae_ax.grid(True, alpha=0.25)
    for idx, value in enumerate(mae_24):
        mae_ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9, color="#166f52")
    for idx, value in enumerate(mae_120):
        mae_ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9, color="#9a4128")

    r2_ax.plot(x, r2_24, marker="o", linewidth=2.6, color="#1f9d73", label="24-step (120 min)")
    r2_ax.plot(x, r2_120, marker="o", linewidth=2.6, color="#c85c3a", label="120-step (600 min)")
    r2_ax.set_title("最终步 R² 对比")
    r2_ax.set_xticks(x, labels)
    r2_ax.set_ylabel("R²")
    r2_ax.grid(True, alpha=0.25)
    for idx, value in enumerate(r2_24):
        r2_ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9, color="#166f52")
    for idx, value in enumerate(r2_120):
        r2_ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9, color="#9a4128")

    fig.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("AGC 当前 hybrid-transformer：24-step vs 120-step", fontsize=14, y=1.12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_windows(summary_24: dict, summary_120: dict, window_24: dict, window_120: dict, out_path: Path) -> None:
    true_24 = np.asarray(window_24["true"], dtype=float)
    pred_24 = np.asarray(window_24["pred"], dtype=float)
    true_120 = np.asarray(window_120["true"], dtype=float)
    pred_120 = np.asarray(window_120["pred"], dtype=float)
    minutes_24 = np.arange(1, true_24.shape[0] + 1, dtype=float) * 5.0
    minutes_120 = np.arange(1, true_120.shape[0] + 1, dtype=float) * 5.0

    fig, axes = plt.subplots(3, 2, figsize=(15, 11), squeeze=False)

    for row_idx, (key, zh_name, unit) in enumerate(TARGET_SPECS):
        ax_short = axes[row_idx, 0]
        ax_long = axes[row_idx, 1]
        idx = [name for name, _, _ in TARGET_SPECS].index(key)

        ax_short.plot(minutes_24, true_24[:, idx], color="black", linewidth=2.2, label="真实轨迹")
        ax_short.plot(minutes_24, pred_24[:, idx], color="#1f9d73", linewidth=2.0, linestyle="--", label="预测轨迹")
        ax_short.set_title(f"24-step (120 min) | {zh_name}")
        ax_short.set_xlabel("未来分钟")
        ax_short.set_ylabel(f"{zh_name} ({unit})")
        ax_short.grid(True, alpha=0.25)
        m24 = summary_24["metrics_by_target"][key]
        ax_short.text(
            0.02, 0.96,
            f"Final R² = {m24['final_r2']:.3f}\nFinal MAE = {m24['final_mae']:.2f}",
            transform=ax_short.transAxes,
            va="top", ha="left", fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.88, "edgecolor": "0.75"},
        )

        ax_long.plot(minutes_120, true_120[:, idx], color="black", linewidth=2.2, label="真实轨迹")
        ax_long.plot(minutes_120, pred_120[:, idx], color="#c85c3a", linewidth=2.0, linestyle="--", label="预测轨迹")
        ax_long.set_title(f"120-step (600 min) | {zh_name}")
        ax_long.set_xlabel("未来分钟")
        ax_long.set_ylabel(f"{zh_name} ({unit})")
        ax_long.grid(True, alpha=0.25)
        m120 = summary_120["metrics_by_target"][key]
        ax_long.text(
            0.02, 0.96,
            f"Final R² = {m120['final_r2']:.3f}\nFinal MAE = {m120['final_mae']:.2f}",
            transform=ax_long.transAxes,
            va="top", ha="left", fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.88, "edgecolor": "0.75"},
        )

        if row_idx == 0:
            ax_short.legend(loc="best")
            ax_long.legend(loc="best")

    fig.suptitle(
        "AGC 当前 hybrid-transformer：24-step 与 120-step 代表性预测窗\n"
        "说明：120-step 已经是 600 分钟任务，不再与旧 Strawberry 的 120 x 1min 等价",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    os.chdir(PROJECT_ROOT)
    _setup_matplotlib_chinese()

    summary_24 = _load_json(ANALYSIS_DIR / "current_hybrid_transformer_joint_all_reference_summary.json")
    summary_120 = _load_json(ANALYSIS_DIR / "current_hybrid_transformer_h120_joint_all_reference_summary.json")
    window_24 = _load_window(24, "current_hybrid_transformer_joint_all_reference.pt")
    window_120 = _load_window(120, "current_hybrid_transformer_h120_joint_all_reference.pt")

    metrics_path = FIGURES_DIR / "current_hybrid_transformer_h24_vs_h120_metrics_cn.png"
    windows_path = FIGURES_DIR / "current_hybrid_transformer_h24_vs_h120_windows_cn.png"
    _plot_metrics(summary_24, summary_120, metrics_path)
    _plot_windows(summary_24, summary_120, window_24, window_120, windows_path)

    print(f"Saved figure: {metrics_path}")
    print(f"Saved figure: {windows_path}")


if __name__ == "__main__":
    main()
