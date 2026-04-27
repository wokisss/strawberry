# -*- coding: utf-8 -*-
"""Build the paper-facing Protected Horizon Fusion ablation table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import AGCConfig
from results_utils import ensure_results_layout


PHF_ABLATION_MODELS = [
    {
        "predictor": "itransformer_residual",
        "label": "Residual",
        "role": "generic residual backbone baseline",
        "question": "Generic residual backbone",
    },
    {
        "predictor": "itransformer_co2_late_residual",
        "label": "Late residual",
        "role": "CO2-aware backbone",
        "question": "Does a CO2-aware late adapter help?",
    },
    {
        "predictor": "itransformer_co2_frozen_expert",
        "label": "Frozen expert",
        "role": "naive frozen-expert fusion baseline",
        "question": "Does a frozen standalone expert help if blended directly?",
    },
    {
        "predictor": "itransformer_co2_late_frozen_expert",
        "label": "Late frozen expert",
        "role": "late-trust control baseline",
        "question": "Is horizon-dependent late trust useful?",
    },
    {
        "predictor": "itransformer_co2_teacher_distill",
        "label": "Teacher distill",
        "role": "distillation ablation",
        "question": "Is using the expert only as a teacher enough?",
    },
    {
        "predictor": "itransformer_co2_recoupled_expert",
        "label": "Recoupled expert",
        "role": "cross-target recoupling baseline",
        "question": "Does recoupling after expert correction improve control objective?",
    },
    {
        "predictor": "itransformer_co2_protected_expert",
        "label": "Protected expert",
        "role": "agreement-protection ablation",
        "question": "Is agreement protection useful?",
    },
    {
        "predictor": "itransformer_co2_protected_terminal",
        "label": "Protected terminal",
        "role": "terminal-loss ablation",
        "question": "Is terminal loss alone enough?",
    },
    {
        "predictor": "itransformer_co2_horizon_mixture",
        "label": "Horizon mixture",
        "role": "proposed offline PHF representative",
        "question": "Does protected horizon fusion with terminal pullback improve offline CO2?",
    },
    {
        "predictor": "itransformer_co2_frozen_backbone_horizon_mixture",
        "label": "Frozen-backbone mix",
        "role": "control-safety diagnostic",
        "question": "Does freezing the backbone improve MPC safety?",
    },
    {
        "predictor": "itransformer_co2_control_aware_fusion",
        "label": "Control-aware fusion",
        "role": "late-frozen anchor + PHF terminal candidate",
        "question": "Can we keep late-frozen control behavior while recovering PHF terminal gains?",
    },
]


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _forecast_summary(cfg: AGCConfig, predictor: str, compartment: str) -> dict:
    name = f"{predictor}_joint_all_{compartment.lower()}_summary.json"
    return _read_json(Path(cfg.forecast_analysis_dir) / name)


def _control_summary(cfg: AGCConfig, predictor: str, controller: str) -> dict:
    return _read_json(Path(cfg.control_summaries_dir) / f"{predictor}_{controller}_summary.json")


def _control_relevant_rank(cfg: AGCConfig, predictor: str, compartment: str) -> float:
    path = Path(cfg.forecast_analysis_dir) / f"control_relevant_validation_{compartment.lower()}.json"
    payload = _read_json(path)
    for record in payload.get("ranked_summary", []):
        if record.get("predictor") == predictor:
            return float(record.get("control_relevant_mean_rank", np.nan))
    return float("nan")


def _target_metrics(summary: dict, target: str) -> dict:
    metrics = summary.get("metrics_by_target", {}).get(target, {})
    return {
        f"{target.lower()}_full_mae": metrics.get("full_mae", np.nan),
        f"{target.lower()}_final_mae": metrics.get("final_mae", np.nan),
        f"{target.lower()}_full_r2": metrics.get("full_r2", np.nan),
        f"{target.lower()}_final_r2": metrics.get("final_r2", np.nan),
    }


def build_ablation(cfg: AGCConfig, compartment: str) -> dict:
    ensure_results_layout(cfg)
    rows = []
    previous_co2_full = np.nan
    previous_co2_final = np.nan

    for item in PHF_ABLATION_MODELS:
        predictor = item["predictor"]
        forecast = _forecast_summary(cfg, predictor, compartment)
        gradient = _control_summary(cfg, predictor, "gradient_mpc")
        cem = _control_summary(cfg, predictor, "cem_mpc")
        recorded = _control_summary(cfg, predictor, "recorded")

        row = {
            "predictor": predictor,
            "label": item["label"],
            "role": item["role"],
            "question": item["question"],
            "has_forecast_summary": bool(forecast),
            "has_gradient_mpc_summary": bool(gradient),
            "has_cem_mpc_summary": bool(cem),
            "control_relevant_mean_rank": _control_relevant_rank(cfg, predictor, compartment),
        }
        for target in ("Tair", "Rhair", "CO2air"):
            row.update(_target_metrics(forecast, target))

        row.update(
            {
                "gradient_objective": gradient.get("objective_mean", np.nan),
                "gradient_tair_mae": gradient.get("target_mae", {}).get("Tair", np.nan),
                "gradient_rhair_mae": gradient.get("target_mae", {}).get("Rhair", np.nan),
                "gradient_co2_mae": gradient.get("target_mae", {}).get("CO2air", np.nan),
                "gradient_action_tv": gradient.get("action_tv", np.nan),
                "cem_objective": cem.get("objective_mean", np.nan),
                "cem_co2_mae": cem.get("target_mae", {}).get("CO2air", np.nan),
                "recorded_co2_mae": recorded.get("target_mae", {}).get("CO2air", np.nan),
            }
        )

        co2_full = float(row.get("co2air_full_mae", np.nan))
        co2_final = float(row.get("co2air_final_mae", np.nan))
        row["delta_co2_full_mae_vs_previous"] = (
            co2_full - previous_co2_full
            if np.isfinite(co2_full) and np.isfinite(previous_co2_full)
            else np.nan
        )
        row["delta_co2_final_mae_vs_previous"] = (
            co2_final - previous_co2_final
            if np.isfinite(co2_final) and np.isfinite(previous_co2_final)
            else np.nan
        )
        if np.isfinite(co2_full):
            previous_co2_full = co2_full
        if np.isfinite(co2_final):
            previous_co2_final = co2_final
        rows.append(row)

    return {
        "compartment": compartment,
        "purpose": "Protected Horizon Fusion ablation summary for paper-facing model-story convergence.",
        "rows": rows,
        "interpretation": {
            "offline_leader": "itransformer_co2_horizon_mixture",
            "co2_control_leader": "itransformer_co2_late_frozen_expert",
            "overall_control_objective_leader": "itransformer_co2_recoupled_expert",
            "control_safe_diagnostic": "itransformer_co2_frozen_backbone_horizon_mixture",
            "control_aware_candidate": "itransformer_co2_control_aware_fusion",
        },
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _fmt(value) -> str:
    if isinstance(value, str):
        return value
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(value):
        return ""
    return f"{value:.3f}"


def _write_markdown(path: Path, rows: list[dict]) -> None:
    cols = [
        "label",
        "role",
        "co2air_full_mae",
        "co2air_final_mae",
        "gradient_objective",
        "gradient_co2_mae",
        "control_relevant_mean_rank",
        "question",
    ]
    lines = [
        "# Protected Horizon Fusion Ablation Summary",
        "",
        "Lower MAE/objective/rank values are better. Blank control cells mean the model has no recorded closed-loop summary.",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(col, "")) for col in cols) + " |")
    lines.extend(
        [
            "",
            "## Main Reading",
            "",
            "- `Horizon mixture` is the current offline PHF representative and CO2 forecasting leader.",
            "- `Late frozen expert` remains the strongest current closed-loop CO2 control baseline.",
            "- `Control-aware fusion` is the new single-candidate follow-up that should be judged by both control-relevant rank and `GradientMPC` transfer, not offline CO2 alone.",
            "- `Recoupled expert` remains the strongest current overall closed-loop objective baseline.",
            "- `Frozen-backbone mix` is a control-safety diagnostic, not the main offline forecasting method.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [row["label"] for row in rows]
    metrics = [
        ("co2air_full_mae", "CO2 full MAE"),
        ("co2air_final_mae", "CO2 final MAE"),
        ("gradient_co2_mae", "Closed-loop CO2 MAE"),
        ("gradient_objective", "Closed-loop objective"),
    ]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(len(metrics), 1, figsize=(15, 15), sharex=True)
    for ax, (metric, title) in zip(axes, metrics):
        values = np.asarray([row.get(metric, np.nan) for row in rows], dtype=np.float32)
        ax.bar(x, values, color="#2f6f8f")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        for idx, value in enumerate(values):
            if np.isfinite(value):
                ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    axes[-1].set_xticks(x, labels, rotation=25, ha="right")
    fig.suptitle("Protected Horizon Fusion Ablation", fontsize=16, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_outputs(cfg: AGCConfig, payload: dict) -> None:
    compartment = payload["compartment"].lower()
    analysis_dir = Path(cfg.forecast_analysis_dir)
    figure_dir = Path(cfg.forecast_figures_dir) / "comparisons"
    json_path = analysis_dir / f"phf_ablation_{compartment}.json"
    csv_path = analysis_dir / f"phf_ablation_{compartment}.csv"
    md_path = analysis_dir / f"phf_ablation_{compartment}.md"
    figure_path = figure_dir / f"phf_ablation_{compartment}.png"

    json_path.write_text(
        json.dumps(_json_ready(payload), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    _write_csv(csv_path, payload["rows"])
    _write_markdown(md_path, payload["rows"])
    _plot(figure_path, payload["rows"])
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved Markdown: {md_path}")
    print(f"Saved figure: {figure_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compartment", default="Reference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = AGCConfig()
    payload = build_ablation(cfg, args.compartment)
    write_outputs(cfg, payload)


if __name__ == "__main__":
    main()
