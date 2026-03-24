# -*- coding: utf-8 -*-
"""Generate advisor-facing Chinese figures for old Strawberry vs current AGC hybrid-transformer."""

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
    ("Temperature", "温度", "°C", "Tair"),
    ("Humidity", "湿度", "%", "Rhair"),
    ("CO2", "CO2", "ppm", "CO2air"),
]


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


def _load_joint_all_representative_window() -> dict:
    cfg = _apply_fair_budget_overrides(AGCConfig())
    _set_global_seed(cfg.seed)
    processor = AGCDataProcessor(cfg)
    bundle = _build_bundle(processor, cfg, "joint_all", "Reference")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = (
        PROJECT_ROOT
        / "results"
        / "forecasting"
        / "checkpoints"
        / "current_hybrid_transformer_joint_all_reference.pt"
    )
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


def _plot_best_vs_old_line(old_summary: dict, agc_summary: dict, out_path: Path) -> None:
    x = np.arange(len(TARGET_SPECS))
    target_order = [zh_name for _, zh_name, _, _ in TARGET_SPECS]

    old_mae = old_summary["old_project"]["final_mae"]
    old_r2 = old_summary["old_project"]["final_r2"]
    agc_mae = [agc_summary["metrics_by_target"][key]["final_mae"] for _, _, _, key in TARGET_SPECS]
    agc_r2 = [agc_summary["metrics_by_target"][key]["final_r2"] for _, _, _, key in TARGET_SPECS]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    mae_ax, r2_ax = axes

    mae_ax.plot(x, old_mae, marker="o", linewidth=2.6, color="#9c6ade", label="旧 Strawberry / old hybrid-transformer")
    mae_ax.plot(x, agc_mae, marker="o", linewidth=2.6, color="#1f9d73", label="AGC / current hybrid-transformer")
    mae_ax.set_title("最终步 MAE 对比")
    mae_ax.set_xticks(x, target_order)
    mae_ax.set_ylabel("MAE")
    mae_ax.grid(True, alpha=0.25)
    for idx, value in enumerate(old_mae):
        mae_ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9, color="#6f42c1")
    for idx, value in enumerate(agc_mae):
        mae_ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9, color="#166f52")

    r2_ax.plot(x, old_r2, marker="o", linewidth=2.6, color="#9c6ade", label="旧 Strawberry / old hybrid-transformer")
    r2_ax.plot(x, agc_r2, marker="o", linewidth=2.6, color="#1f9d73", label="AGC / current hybrid-transformer")
    r2_ax.set_title("最终步 R² 对比")
    r2_ax.set_xticks(x, target_order)
    r2_ax.set_ylabel("R²")
    r2_ax.grid(True, alpha=0.25)
    for idx, value in enumerate(old_r2):
        r2_ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9, color="#6f42c1")
    for idx, value in enumerate(agc_r2):
        r2_ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9, color="#166f52")

    fig.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle(
        "同属 hybrid-transformer：旧数据集旧方法 vs 新数据集新方法",
        fontsize=14,
        y=1.12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_window_comparison(old_summary: dict, agc_summary: dict, agc_window: dict, out_path: Path) -> None:
    old_true = np.asarray(old_summary["old_project"]["representative_window"]["true"], dtype=float)
    old_pred = np.asarray(old_summary["old_project"]["representative_window"]["pred"], dtype=float)
    agc_true = np.asarray(agc_window["true"], dtype=float)
    agc_pred = np.asarray(agc_window["pred"], dtype=float)

    old_minutes = np.arange(1, old_true.shape[0] + 1, dtype=float) * old_summary["old_project"]["step_minutes"]
    agc_minutes = np.arange(1, agc_true.shape[0] + 1, dtype=float) * 5.0

    fig, axes = plt.subplots(3, 2, figsize=(14.5, 11), squeeze=False)

    for row_idx, (_, zh_name, unit, agc_key) in enumerate(TARGET_SPECS):
        ax_old = axes[row_idx, 0]
        ax_agc = axes[row_idx, 1]

        ax_old.plot(old_minutes, old_true[:, row_idx], color="black", linewidth=2.2, label="真实轨迹")
        ax_old.plot(old_minutes, old_pred[:, row_idx], color="#9c6ade", linewidth=2.0, linestyle="--", label="预测轨迹")
        ax_old.set_title(f"旧 Strawberry | {zh_name}")
        ax_old.set_xlabel("未来分钟")
        ax_old.set_ylabel(f"{zh_name} ({unit})")
        ax_old.grid(True, alpha=0.25)
        ax_old.text(
            0.02,
            0.96,
            f"Final R² = {old_summary['old_project']['final_r2'][row_idx]:.3f}\n"
            f"Final MAE = {old_summary['old_project']['final_mae'][row_idx]:.2f}",
            transform=ax_old.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.88, "edgecolor": "0.75"},
        )

        agc_idx = ["Tair", "Rhair", "CO2air"].index(agc_key)
        ax_agc.plot(agc_minutes, agc_true[:, agc_idx], color="black", linewidth=2.2, label="真实轨迹")
        ax_agc.plot(agc_minutes, agc_pred[:, agc_idx], color="#1f9d73", linewidth=2.0, linestyle="--", label="预测轨迹")
        ax_agc.set_title(f"AGC joint_all | {zh_name}")
        ax_agc.set_xlabel("未来分钟")
        ax_agc.set_ylabel(f"{zh_name} ({unit})")
        ax_agc.grid(True, alpha=0.25)
        metric = agc_summary["metrics_by_target"][agc_key]
        ax_agc.text(
            0.02,
            0.96,
            f"Final R² = {metric['final_r2']:.3f}\nFinal MAE = {metric['final_mae']:.2f}",
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
        "代表性预测窗对比：旧 Strawberry 的 old hybrid-transformer vs AGC 当前 hybrid-transformer\n"
        "目的：直观看轨迹贴合与系统性漂移，不做严格样本对齐",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_window_comparison_minute_aligned(old_summary: dict, agc_summary: dict, agc_window: dict, out_path: Path) -> None:
    old_true = np.asarray(old_summary["old_project"]["representative_window"]["true"], dtype=float)
    old_pred = np.asarray(old_summary["old_project"]["representative_window"]["pred"], dtype=float)
    agc_true = np.asarray(agc_window["true"], dtype=float)
    agc_pred = np.asarray(agc_window["pred"], dtype=float)

    old_minutes = np.arange(1, old_true.shape[0] + 1, dtype=float)
    agc_minutes = np.arange(1, agc_true.shape[0] + 1, dtype=float) * 5.0
    aligned_minutes = np.arange(1, 121, dtype=float)

    fig, axes = plt.subplots(3, 2, figsize=(14.5, 11), squeeze=False)

    for row_idx, (_, zh_name, unit, agc_key) in enumerate(TARGET_SPECS):
        ax_old = axes[row_idx, 0]
        ax_agc = axes[row_idx, 1]

        ax_old.plot(old_minutes, old_true[:, row_idx], color="black", linewidth=2.2, label="真实轨迹")
        ax_old.plot(old_minutes, old_pred[:, row_idx], color="#9c6ade", linewidth=2.0, linestyle="--", label="预测轨迹")
        ax_old.set_title(f"旧 Strawberry | {zh_name}")
        ax_old.set_xlabel("未来分钟")
        ax_old.set_ylabel(f"{zh_name} ({unit})")
        ax_old.grid(True, alpha=0.25)
        ax_old.text(
            0.02,
            0.96,
            f"Final R² = {old_summary['old_project']['final_r2'][row_idx]:.3f}\n"
            f"Final MAE = {old_summary['old_project']['final_mae'][row_idx]:.2f}",
            transform=ax_old.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.88, "edgecolor": "0.75"},
        )

        agc_idx = ["Tair", "Rhair", "CO2air"].index(agc_key)
        agc_true_interp = np.interp(aligned_minutes, agc_minutes, agc_true[:, agc_idx])
        agc_pred_interp = np.interp(aligned_minutes, agc_minutes, agc_pred[:, agc_idx])

        ax_agc.plot(aligned_minutes, agc_true_interp, color="black", linewidth=2.2, label="真实轨迹（5 分钟插值）")
        ax_agc.plot(aligned_minutes, agc_pred_interp, color="#1f9d73", linewidth=2.0, linestyle="--", label="预测轨迹（5 分钟插值）")
        ax_agc.scatter(agc_minutes, agc_true[:, agc_idx], color="black", s=18, alpha=0.75)
        ax_agc.scatter(agc_minutes, agc_pred[:, agc_idx], color="#1f9d73", s=18, alpha=0.75)
        ax_agc.set_title(f"AGC joint_all | {zh_name}")
        ax_agc.set_xlabel("未来分钟")
        ax_agc.set_ylabel(f"{zh_name} ({unit})")
        ax_agc.grid(True, alpha=0.25)
        metric = agc_summary["metrics_by_target"][agc_key]
        ax_agc.text(
            0.02,
            0.96,
            f"Final R² = {metric['final_r2']:.3f}\n"
            f"Final MAE = {metric['final_mae']:.2f}\n"
            "显示为 120 个分钟点；原始预测仍为 24 x 5min",
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
        "分钟对齐展示版：旧 Strawberry 的 120 x 1min vs AGC 的 24 x 5min\n"
        "说明：右侧仅做显示插值，便于肉眼比较，不代表额外预测了 120 步",
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

    old_summary = _load_json(FIGURES_DIR / "strawberry_vs_agc_dataset_switch_summary.json")
    agc_summary = _load_json(ANALYSIS_DIR / "current_hybrid_transformer_joint_all_reference_summary.json")
    agc_window = _load_joint_all_representative_window()

    line_path = FIGURES_DIR / "current_hybrid_transformer_best_vs_old_line_cn.png"
    _plot_best_vs_old_line(old_summary, agc_summary, line_path)

    window_path = FIGURES_DIR / "current_hybrid_transformer_old_vs_agc_joint_all_windows_cn.png"
    _plot_window_comparison(old_summary, agc_summary, agc_window, window_path)

    aligned_window_path = FIGURES_DIR / "current_hybrid_transformer_old_vs_agc_joint_all_windows_minute_aligned_cn.png"
    _plot_window_comparison_minute_aligned(old_summary, agc_summary, agc_window, aligned_window_path)

    print(f"Saved figure: {line_path}")
    print(f"Saved figure: {window_path}")
    print(f"Saved figure: {aligned_window_path}")


if __name__ == "__main__":
    main()
