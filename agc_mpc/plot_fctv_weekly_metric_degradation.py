# -*- coding: utf-8 -*-
"""Build a weekly-report figure showing FCTV metric explanatory degradation."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

from config import AGCConfig
from results_utils import ensure_results_layout


METRICS = [
    {
        "key": "co2_first_step_mae",
        "target": "mpc_co2_mae",
        "label": "CO2 first-step",
        "color": "#cf4f35",
        "early": {"spearman": 0.516, "pairwise": 0.681, "role": "secondary"},
    },
    {
        "key": "co2_constraint_near_mae_proxy",
        "target": "mpc_co2_mae",
        "label": "CO2 constraint-near",
        "color": "#e59f32",
        "early": {"spearman": 0.522, "pairwise": 0.676, "role": "secondary"},
    },
    {
        "key": "rhair_first_step_mae",
        "target": "mpc_rhair_mae",
        "label": "Rhair first-step",
        "color": "#2c7a7b",
        "early": {"spearman": 0.711, "pairwise": 0.787, "role": "primary"},
    },
    {
        "key": "multiobjective_transfer_selection_score",
        "target": "mpc_objective",
        "label": "Multi-objective score",
        "color": "#3c64a6",
        "early": {"spearman": 0.267, "pairwise": 0.618, "role": "weak"},
    },
    {
        "key": "tair_first_step_mae",
        "target": "mpc_tair_mae",
        "label": "Tair first-step",
        "color": "#777777",
        "early": {"spearman": -0.270, "pairwise": 0.412, "role": "diagnostic"},
    },
]

REFERENCE_ROLES = {
    ("rhair_first_step_mae", "mpc_rhair_mae"): "secondary",
    ("co2_first_step_mae", "mpc_co2_mae"): "diagnostic",
    ("co2_constraint_near_mae_proxy", "mpc_co2_mae"): "diagnostic",
    ("multiobjective_transfer_selection_score", "mpc_objective"): "diagnostic",
    ("tair_first_step_mae", "mpc_tair_mae"): "diagnostic",
}


def _setup_chinese_font() -> None:
    font_candidates = [
        Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ]
    for font_path in font_candidates:
        if font_path.exists():
            fm.fontManager.addfont(str(font_path))
            font_name = fm.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.family"] = font_name
            break
    plt.rcParams["axes.unicode_minus"] = False


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _transfer_lookup(rows: list[dict[str, str]], metric: str, target: str) -> dict[str, str]:
    for row in rows:
        if row.get("metric") == metric and row.get("control_target") == target:
            return row
    raise KeyError(f"Missing transfer row: {metric} -> {target}")


def _multi_start_lookup(rows: list[dict[str, str]], metric: str, target: str, start_idx: int) -> dict[str, str]:
    for row in rows:
        if (
            int(row.get("start_idx", -1)) == start_idx
            and row.get("metric") == metric
            and row.get("control_target") == target
        ):
            return row
    raise KeyError(f"Missing multi-start row: start={start_idx}, {metric} -> {target}")


def _metric_series(reference_rows: list[dict[str, str]], multistart_rows: list[dict[str, str]]) -> list[dict]:
    records = []
    for metric in METRICS:
        ref = _transfer_lookup(reference_rows, metric["key"], metric["target"])
        starts = [_multi_start_lookup(multistart_rows, metric["key"], metric["target"], start) for start in (0, 96, 192)]
        start_spearman = [float(row["full_spearman"]) for row in starts]
        start_pairwise = [float(row["full_pairwise_consistency"]) for row in starts]
        records.append(
            {
                **metric,
                "stage_spearman": [
                    metric["early"]["spearman"],
                    float(ref["spearman"]),
                    float(np.mean(start_spearman)),
                ],
                "stage_pairwise": [
                    metric["early"]["pairwise"],
                    float(ref["pairwise_consistency"]),
                    float(np.mean(start_pairwise)),
                ],
                "heatmap_values": [
                    metric["early"]["spearman"],
                    float(ref["spearman"]),
                    *start_spearman,
                ],
                "roles": [
                    metric["early"]["role"],
                    REFERENCE_ROLES[(metric["key"], metric["target"])],
                    *[row.get("role", "") for row in starts],
                ],
            }
        )
    return records


def _short_role(role: str) -> str:
    if "primary" in role:
        return "primary"
    if "secondary" in role:
        return "secondary"
    if "weak" in role:
        return "weak"
    return "diagnostic"


def _read_winner_pairs(path: Path) -> list[dict[str, object]]:
    rows = _read_csv_rows(path)
    output = []
    for start in (0, 96, 192):
        start_rows = [row for row in rows if int(row["start_idx"]) == start]
        objective_winner = min(start_rows, key=lambda row: float(row["objective_rank"]))
        co2_winner = min(start_rows, key=lambda row: float(row["co2_rank"]))
        output.append({"start": start, "kind": "Objective winner", **objective_winner})
        output.append({"start": start, "kind": "CO2 winner", **co2_winner})
    return output


def _pretty_model(name: str) -> str:
    replacements = {
        "current_hybrid_transformer": "current\nhybrid",
        "transformer_hybrid_residual": "transformer\nhybrid\nresidual",
        "itransformer_co2_residual": "iTransformer\nCO2\nresidual",
    }
    return replacements.get(name, name.replace("_forecaster", "").replace("_", "\n"))


def build_figure(prefix: str = "fctv_weekly_metric_degradation_summary") -> Path:
    _setup_chinese_font()
    cfg = AGCConfig()
    ensure_results_layout(cfg)
    analysis_dir = Path(cfg.forecast_analysis_dir)
    figure_dir = Path(cfg.forecast_figures_dir) / "comparisons"
    figure_dir.mkdir(parents=True, exist_ok=True)

    reference_rows = _read_csv_rows(analysis_dir / "forecast_to_control_transfer_reference.csv")
    multistart_rows = _read_csv_rows(analysis_dir / "forecast_to_control_transfer_multistart16_reference.csv")
    metric_records = _metric_series(reference_rows, multistart_rows)

    stages = ["17模型\nCO2指标归纳", "24模型\n扩池验证", "16模型×3起点\n多目标/多起点均值"]

    fig, (ax_spearman, ax_pairwise) = plt.subplots(1, 2, figsize=(16.8, 6.45), facecolor="#f7f4ee")
    fig.subplots_adjust(left=0.06, right=0.985, top=0.91, bottom=0.31, wspace=0.18)

    x = np.arange(len(stages))
    for record in metric_records:
        linewidth = 3.1 if record["key"].startswith("co2") else 2.3
        alpha = 1.0 if record["key"].startswith("co2") else 0.82
        ax_spearman.plot(
            x,
            record["stage_spearman"],
            marker="o",
            linewidth=linewidth,
            markersize=7,
            color=record["color"],
            alpha=alpha,
            label=record["label"],
        )
        ax_pairwise.plot(
            x,
            record["stage_pairwise"],
            marker="o",
            linewidth=linewidth,
            markersize=7,
            color=record["color"],
            alpha=alpha,
            label=record["label"],
        )

    for ax, ylabel, threshold in [
        (ax_spearman, "Spearman 相关：预测指标 vs 闭环收益", 0.35),
        (ax_pairwise, "两两模型排序一致率", 0.60),
    ]:
        ax.axhline(threshold, color="#34495e", linestyle="--", linewidth=1.2, alpha=0.65)
        ax.text(2.03, threshold + 0.015, "可用筛选参考线", fontsize=9, color="#34495e")
        ax.axhspan(-0.5, threshold, color="#d9534f", alpha=0.06)
        ax.set_xticks(x)
        ax.set_xticklabels(stages, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_ylim((-0.5, 0.85) if ax is ax_spearman else (0.30, 0.85))
        ax.grid(axis="y", alpha=0.25)
        ax.set_facecolor("#fffdf8")
    ax_spearman.set_title("扩大模型池后：指标解释性明显退化", loc="left", fontsize=15, fontweight="bold")
    ax_pairwise.set_title("排序一致率也退化：不能稳定筛选闭环收益", loc="left", fontsize=15, fontweight="bold")
    ax_pairwise.legend(loc="lower left", ncols=2, fontsize=9, frameon=True)

    fig.text(
        0.06,
        0.185,
        "Spearman 相关系数：衡量“预测指标排名”和“闭环控制收益排名”的单调一致性，取值范围为 -1 到 1。越接近 1，说明预测指标越能按正确方向解释模型闭环表现；接近 0 表示排序关系弱；负值表示预测指标给出的排序可能与闭环结果相反。",
        fontsize=10.2,
        color="#263238",
    )
    fig.text(
        0.06,
        0.135,
        "可用筛选参考线：Spearman 约 0.2-0.4 通常只能视为弱到中等相关；低于 0.2 多数情况下只适合诊断，高于 0.4 才更适合作为辅助筛选信号。本图取 0.35 作为保守的“开始可用”参考线，不是统计显著性阈值。",
        fontsize=10.2,
        color="#263238",
    )
    fig.text(
        0.06,
        0.085,
        "两两模型排序一致率：任意取两个模型，如果模型 A 的预测指标比模型 B 好，那么 A 的闭环控制表现是不是也比 B 好。这个比例越高，说明预测指标越能筛模型；接近 0.5 时基本等同随机判断。",
        fontsize=10.2,
        color="#263238",
    )
    fig.text(
        0.985,
        0.025,
        "数据源：17-model context baseline；24-model single-start FCTV；16-model multi-start GradientMPC 96-step",
        fontsize=8.5,
        color="#606c76",
        ha="right",
    )

    output = figure_dir / f"{prefix}.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)
    output = build_figure()
    print(f"Saved figure: {output}")


if __name__ == "__main__":
    main()
