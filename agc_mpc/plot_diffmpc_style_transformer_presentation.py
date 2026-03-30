# -*- coding: utf-8 -*-
"""Generate presentation-friendly Chinese figures for DiffMPC-style Transformer comparison."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from benchmark_diffmpc_style_transformer import (
    _apply_diffmpc_style_overrides,
    _build_bundle,
    _set_global_seed,
)
from config import AGCConfig
from data_processing.processor import AGCDataProcessor
from figure_layout import comparison_figures_dir
from models.diffmpc_style_transformer import DiffMPCStyleTransformerHybridForecaster


PROJECT_ROOT = Path(__file__).resolve().parent
FIGURES_DIR = comparison_figures_dir(PROJECT_ROOT / "results" / "forecasting" / "figures")
ANALYSIS_DIR = PROJECT_ROOT / "results" / "forecasting" / "analysis"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _setup_matplotlib_chinese() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _build_joint_all_bundle() -> dict:
    cfg = _apply_diffmpc_style_overrides(AGCConfig())
    processor = AGCDataProcessor(cfg)
    return _build_bundle(processor, cfg, "joint_all", "Reference")


def _load_joint_all_representative_window() -> dict:
    cfg = _apply_diffmpc_style_overrides(AGCConfig())
    _set_global_seed(cfg.seed)
    bundle = _build_joint_all_bundle()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = PROJECT_ROOT / "results" / "forecasting" / "checkpoints" / "diffmpc_style_transformer_joint_all_reference.pt"
    model = DiffMPCStyleTransformerHybridForecaster(
        past_dim=bundle["X_past_train"].shape[-1],
        future_dim=bundle["W_future_train"].shape[-1] + bundle["U_future_train"].shape[-1],
        target_dim=bundle["Y_future_train"].shape[-1],
        seq_len=cfg.seq_len,
        horizon=cfg.horizon,
        d_model=64,
        nhead=4,
        num_layers=4,
        dim_feedforward=128,
        dropout=0.1,
        target_indices=[bundle["feature_groups"]["x_past"].index(col) for col in cfg.target_cols],
    )
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
        "true": true_real.tolist(),
        "pred": pred_real.tolist(),
    }


def _plot_best_vs_old_line(old_summary: dict, joint_summary: dict, out_path: Path) -> None:
    target_order = ["Temperature", "Humidity", "CO2"]
    x = np.arange(len(target_order))

    old_mae = old_summary["old_project"]["final_mae"]
    old_r2 = old_summary["old_project"]["final_r2"]
    agc_mae = [
        joint_summary["metrics_by_target"]["Tair"]["final_mae"],
        joint_summary["metrics_by_target"]["Rhair"]["final_mae"],
        joint_summary["metrics_by_target"]["CO2air"]["final_mae"],
    ]
    agc_r2 = [
        joint_summary["metrics_by_target"]["Tair"]["final_r2"],
        joint_summary["metrics_by_target"]["Rhair"]["final_r2"],
        joint_summary["metrics_by_target"]["CO2air"]["final_r2"],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    mae_ax, r2_ax = axes
    mae_ax.plot(x, old_mae, marker="o", linewidth=2.4, color="#9c6ade", label="旧 Strawberry / old Transformer-hybrid")
    mae_ax.plot(x, agc_mae, marker="o", linewidth=2.4, color="#5ad8a6", label="AGC / joint_all / diffmpc 风格 Transformer")
    mae_ax.set_title("最终步 MAE 对比")
    mae_ax.set_xticks(x, target_order)
    mae_ax.set_ylabel("MAE")
    mae_ax.grid(True, alpha=0.25)
    for idx, value in enumerate(old_mae):
        mae_ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9, color="#6f42c1")
    for idx, value in enumerate(agc_mae):
        mae_ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9, color="#1f7a5a")

    r2_ax.plot(x, old_r2, marker="o", linewidth=2.4, color="#9c6ade", label="旧 Strawberry / old Transformer-hybrid")
    r2_ax.plot(x, agc_r2, marker="o", linewidth=2.4, color="#5ad8a6", label="AGC / joint_all / diffmpc 风格 Transformer")
    r2_ax.set_title("最终步 R² 对比")
    r2_ax.set_xticks(x, target_order)
    r2_ax.set_ylabel("R²")
    r2_ax.grid(True, alpha=0.25)
    for idx, value in enumerate(old_r2):
        r2_ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9, color="#6f42c1")
    for idx, value in enumerate(agc_r2):
        r2_ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9, color="#1f7a5a")

    fig.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle(
        "相似 Transformer 风格协议下：旧 Strawberry 与 AGC（joint_all）对比",
        fontsize=14,
        y=1.12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_window_comparison(old_summary: dict, joint_summary: dict, agc_window: dict, out_path: Path) -> None:
    target_specs = [
        ("Temperature", "温度", "°C", "Tair"),
        ("Humidity", "湿度", "%", "Rhair"),
        ("CO2", "CO2", "ppm", "CO2air"),
    ]
    old_true = np.asarray(old_summary["old_project"]["representative_window"]["true"], dtype=float)
    old_pred = np.asarray(old_summary["old_project"]["representative_window"]["pred"], dtype=float)
    agc_true = np.asarray(agc_window["true"], dtype=float)
    agc_pred = np.asarray(agc_window["pred"], dtype=float)

    old_minutes = np.arange(1, old_true.shape[0] + 1, dtype=float) * old_summary["old_project"]["step_minutes"]
    agc_minutes = np.arange(1, agc_true.shape[0] + 1, dtype=float) * 5.0

    fig, axes = plt.subplots(3, 2, figsize=(14.5, 11), squeeze=False)

    for row_idx, (old_name, zh_name, unit, agc_name) in enumerate(target_specs):
        ax_old = axes[row_idx, 0]
        ax_agc = axes[row_idx, 1]

        ax_old.plot(old_minutes, old_true[:, row_idx], color="black", linewidth=2.2, label="真实轨迹")
        ax_old.plot(old_minutes, old_pred[:, row_idx], color="#9c6ade", linewidth=2.0, linestyle="--", label="预测轨迹")
        ax_old.set_title(f"旧 Strawberry | {zh_name}")
        ax_old.set_xlabel("未来分钟")
        ax_old.set_ylabel(f"{zh_name} ({unit})")
        ax_old.grid(True, alpha=0.25)
        old_metric = old_summary["old_project"]
        ax_old.text(
            0.02,
            0.96,
            f"Final R² = {old_metric['final_r2'][row_idx]:.3f}\nFinal MAE = {old_metric['final_mae'][row_idx]:.2f}",
            transform=ax_old.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.88, "edgecolor": "0.75"},
        )

        agc_idx = ["Tair", "Rhair", "CO2air"].index(agc_name)
        ax_agc.plot(agc_minutes, agc_true[:, agc_idx], color="black", linewidth=2.2, label="真实轨迹")
        ax_agc.plot(agc_minutes, agc_pred[:, agc_idx], color="#5ad8a6", linewidth=2.0, linestyle="--", label="预测轨迹")
        ax_agc.set_title(f"AGC joint_all | {zh_name}")
        ax_agc.set_xlabel("未来分钟")
        ax_agc.set_ylabel(f"{zh_name} ({unit})")
        ax_agc.grid(True, alpha=0.25)
        agc_metric = joint_summary["metrics_by_target"][agc_name]
        ax_agc.text(
            0.02,
            0.96,
            f"Final R² = {agc_metric['final_r2']:.3f}\nFinal MAE = {agc_metric['final_mae']:.2f}",
            transform=ax_agc.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.88, "edgecolor": "0.75"},
        )

        if row_idx == 0:
            ax_old.legend(loc="best")
            ax_agc.legend(loc="best")

    fig.suptitle(
        "代表性预测窗对比：旧 Strawberry vs AGC joint_all\n"
        "目的：肉眼比较轨迹形态与跟踪稳定性，而不是做严格样本对齐",
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

    old_summary = _load_json(ANALYSIS_DIR / "strawberry_vs_agc_dataset_switch_summary.json")
    joint_summary = _load_json(ANALYSIS_DIR / "diffmpc_style_transformer_joint_all_reference_summary.json")
    agc_window = _load_joint_all_representative_window()

    line_path = FIGURES_DIR / "diffmpc_style_transformer_best_vs_old_line_cn.png"
    _plot_best_vs_old_line(old_summary, joint_summary, line_path)

    window_path = FIGURES_DIR / "diffmpc_style_transformer_old_vs_agc_joint_all_windows_cn.png"
    _plot_window_comparison(old_summary, joint_summary, agc_window, window_path)

    print(f"Saved figure: {line_path}")
    print(f"Saved figure: {window_path}")


if __name__ == "__main__":
    main()
