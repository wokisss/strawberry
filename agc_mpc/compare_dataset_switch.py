# -*- coding: utf-8 -*-
"""Build an advisor-facing comparison figure for Strawberry diffmpc vs AGC agc_mpc."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import AGCConfig
from results_utils import ensure_results_layout


OLD_MODEL_LABEL = "Strawberry / old Transformer-hybrid"
NEW_MODEL_LABELS = [
    "AGC / DLinear",
    "AGC / Transformer",
    "AGC / Transformer-hybrid",
]
WINDOW_MODEL_LABELS = [
    OLD_MODEL_LABEL,
    "AGC / Transformer",
    "AGC / Transformer-hybrid",
]
COMMON_VARS = ["Temperature", "Humidity", "CO2"]
COLOR_MAP = {
    OLD_MODEL_LABEL: "#7F8C8D",
    "AGC / DLinear": "#1E88E5",
    "AGC / Transformer": "#43A047",
    "AGC / Transformer-hybrid": "#E53935",
}


def _run_python_snippet(project_dir: Path, code: str) -> dict:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(project_dir),
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"No JSON output captured from snippet in {project_dir}")
    return json.loads(lines[-1])


def _collect_old_metrics(repo_root: Path) -> dict:
    code = f"""
import contextlib, io, json, random, sys
from pathlib import Path
import numpy as np
import torch

project_root = Path(r"{repo_root / 'diffmpc'}")
sys.path.insert(0, str(project_root))

from config import Config
from data_processing.processor import DataProcessor
from models.transformer_hybrid import TransformerHybridModel
from simulation.evaluator import PredictorEvaluator

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

cfg = Config()
setup_seed(cfg.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with contextlib.redirect_stdout(io.StringIO()):
    processor = DataProcessor(cfg)
    df = processor.load_and_preprocess()
    df = processor.merge_weather(df)
    df = processor.add_time_encoding(df)
    df = processor.add_energy_features(df)
    df = processor.add_ode_derivatives(df)
    data_scaled = processor.prepare_features(df)
    datasets = processor.prepare_datasets(data_scaled)

model = TransformerHybridModel(
    input_dim=len(processor.feature_order),
    seq_len=cfg.seq_len,
    future_dim=len(processor.future_indices),
    target_dim=3,
    forecast_horizon=cfg.horizon,
    target_indices=processor.target_indices,
    d_model=cfg.transformer_d_model,
    nhead=cfg.transformer_nhead,
    num_layers=cfg.transformer_num_layers,
    dim_feedforward=cfg.transformer_dim_feedforward,
    dropout=cfg.transformer_dropout,
).to(device)
state = torch.load(project_root / cfg.model_save_path, map_location=device)
model.load_state_dict(state)
model.eval()

evaluator = PredictorEvaluator(
    model,
    processor.scaler,
    processor.target_indices,
    processor.feature_order,
    cfg,
    device=device,
)
with contextlib.redirect_stdout(io.StringIO()):
    metrics = evaluator.evaluate(datasets["X_test_p"], datasets["X_test_f"], datasets["y_test"])

sample_idx = len(datasets["X_test_p"]) // 2

summary = {{
    "dataset": "Strawberry version2 + external weather",
    "label": "{OLD_MODEL_LABEL}",
    "horizon_steps": cfg.horizon,
    "step_minutes": 1,
    "horizon_minutes": cfg.horizon,
    "variables": {COMMON_VARS},
    "final_mae": metrics["mae_final"],
    "final_r2": metrics["r2_final"],
    "representative_window": {{
        "sample_idx": sample_idx,
        "true": np.asarray(metrics["plot_true_ar"]).tolist(),
        "pred": np.asarray(metrics["plot_pred_ar"]).tolist(),
    }},
}}
print(json.dumps(summary))
"""
    return _run_python_snippet(repo_root / "diffmpc", code)


def _collect_new_metrics(repo_root: Path) -> dict:
    code = f"""
import json, random, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, r2_score

project_root = Path(r"{repo_root / 'agc_mpc'}")
sys.path.insert(0, str(project_root))

from config import AGCConfig
from data_processing.processor import AGCDataProcessor
from models.dlinear_forecaster import ConditionalDLinearForecaster
from models.transformer_forecaster import ConditionalTransformerForecaster
from models.transformer_hybrid_forecaster import ConditionalTransformerHybridForecaster

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

cfg = AGCConfig()
setup_seed(cfg.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AGCDataProcessor(cfg)
bundle = processor.build_multi_compartment_bundle()

def inverse_targets(arr):
    shape = arr.shape
    flat = arr.reshape(-1, shape[-1])
    inv = bundle["scalers"]["y"].inverse_transform(flat)
    return inv.reshape(shape)

models = {{
    "AGC / DLinear": ConditionalDLinearForecaster(
        seq_len=cfg.seq_len,
        horizon=cfg.horizon,
        past_dim=bundle["X_past_test"].shape[-1],
        weather_dim=bundle["W_future_test"].shape[-1],
        control_dim=bundle["U_future_test"].shape[-1],
        target_dim=bundle["Y_future_test"].shape[-1],
        hidden_dim=cfg.hidden_dim,
    ),
    "AGC / Transformer": ConditionalTransformerForecaster(
        past_dim=bundle["X_past_test"].shape[-1],
        weather_dim=bundle["W_future_test"].shape[-1],
        control_dim=bundle["U_future_test"].shape[-1],
        target_dim=bundle["Y_future_test"].shape[-1],
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        nhead=cfg.transformer_heads,
        ff_dim=cfg.transformer_ff_dim,
        max_past_len=cfg.seq_len,
        max_future_len=cfg.horizon,
    ),
    "AGC / Transformer-hybrid": ConditionalTransformerHybridForecaster(
        past_dim=bundle["X_past_test"].shape[-1],
        weather_dim=bundle["W_future_test"].shape[-1],
        control_dim=bundle["U_future_test"].shape[-1],
        target_dim=bundle["Y_future_test"].shape[-1],
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        nhead=cfg.transformer_heads,
        ff_dim=cfg.transformer_ff_dim,
        max_past_len=cfg.seq_len,
        max_future_len=cfg.horizon,
    ),
}}
ckpt_names = {{
    "AGC / DLinear": "dlinear_baseline.pt",
    "AGC / Transformer": "transformer_baseline.pt",
    "AGC / Transformer-hybrid": "transformer_hybrid_baseline.pt",
}}

common_indices = [bundle["feature_groups"]["y_future"].index(name) for name in ["Tair", "Rhair", "CO2air"]]
results = {{}}
for label, model in models.items():
    state = torch.load(project_root / "results" / "forecasting" / "checkpoints" / ckpt_names[label], map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    preds = []
    batch_size = 512
    with torch.no_grad():
        for start in range(0, len(bundle["X_past_test"]), batch_size):
            end = min(start + batch_size, len(bundle["X_past_test"]))
            xb = torch.from_numpy(bundle["X_past_test"][start:end]).float().to(device)
            wb = torch.from_numpy(bundle["W_future_test"][start:end]).float().to(device)
            ub = torch.from_numpy(bundle["U_future_test"][start:end]).float().to(device)
            preds.append(model(xb, wb, ub).cpu().numpy())

    pred_real = inverse_targets(np.concatenate(preds, axis=0))
    true_real = inverse_targets(bundle["Y_future_test"])
    final_pred = pred_real[:, -1, :]
    final_true = true_real[:, -1, :]
    sample_idx = len(pred_real) // 2
    results[label] = {{
        "final_mae": [mean_absolute_error(final_true[:, i], final_pred[:, i]) for i in common_indices],
        "final_r2": [r2_score(final_true[:, i], final_pred[:, i]) for i in common_indices],
        "representative_window": {{
            "sample_idx": sample_idx,
            "true": true_real[sample_idx][:, common_indices].tolist(),
            "pred": pred_real[sample_idx][:, common_indices].tolist(),
        }},
    }}

summary = {{
    "dataset": "AGC 2019 multi-compartment benchmark",
    "horizon_steps": cfg.horizon,
    "step_minutes": 5,
    "horizon_minutes": cfg.horizon * 5,
    "variables": {COMMON_VARS},
    "models": results,
}}
print(json.dumps(summary))
"""
    return _run_python_snippet(repo_root / "agc_mpc", code)


def _draw_feature_table(ax):
    ax.axis("off")
    rows = [
        ("Future weather", "Yes", "Yes"),
        ("Future control plan", "Yes", "Yes"),
        ("Actuator / realized feedback", "Limited", "Yes"),
        ("Multiple compartments / policies", "No", "Yes"),
        ("Resource / economic signals", "No", "Yes"),
        ("Closed-loop benchmarking fit", "Weak", "Strong"),
    ]
    cell_text = [[r[0], r[1], r[2]] for r in rows]
    table = ax.table(
        cellText=cell_text,
        colLabels=["Criterion", "Strawberry", "AGC"],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.8)

    colors = {
        "Yes": "#D7F2D0",
        "Strong": "#D7F2D0",
        "Limited": "#FFF2CC",
        "Weak": "#F4CCCC",
        "No": "#F4CCCC",
    }
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#D9EAF7")
            cell.set_text_props(weight="bold")
            continue
        value = cell.get_text().get_text()
        cell.set_facecolor(colors.get(value, "white"))
    ax.set_title("Task / dataset suitability", fontsize=13, pad=12)


def _plot_mae_ratio(ax, old_metrics: dict, new_metrics: dict):
    x = np.arange(len(COMMON_VARS))
    width = 0.18

    all_labels = [OLD_MODEL_LABEL] + NEW_MODEL_LABELS
    old_mae = np.asarray(old_metrics["final_mae"], dtype=np.float32)
    ratio_map = {OLD_MODEL_LABEL: np.ones_like(old_mae)}
    for label in NEW_MODEL_LABELS:
        ratio_map[label] = np.asarray(new_metrics["models"][label]["final_mae"], dtype=np.float32) / old_mae

    for idx, label in enumerate(all_labels):
        ax.bar(
            x + (idx - 1.5) * width,
            ratio_map[label],
            width=width,
            label=label,
            color=COLOR_MAP[label],
            alpha=0.9,
        )

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(COMMON_VARS)
    ax.set_ylabel("Final MAE ratio vs old Strawberry")
    ax.set_title("Lower is better\n(reference line = 1.0)", fontsize=13)
    ax.grid(True, axis="y", alpha=0.25)


def _plot_final_r2(ax, old_metrics: dict, new_metrics: dict):
    x = np.arange(len(COMMON_VARS))
    width = 0.18
    all_labels = [OLD_MODEL_LABEL] + NEW_MODEL_LABELS
    r2_map = {OLD_MODEL_LABEL: np.asarray(old_metrics["final_r2"], dtype=np.float32)}
    for label in NEW_MODEL_LABELS:
        r2_map[label] = np.asarray(new_metrics["models"][label]["final_r2"], dtype=np.float32)

    for idx, label in enumerate(all_labels):
        ax.bar(
            x + (idx - 1.5) * width,
            r2_map[label],
            width=width,
            label=label,
            color=COLOR_MAP[label],
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(COMMON_VARS)
    ax.set_ylabel("Final-step R2")
    ax.set_title("Higher is better", fontsize=13)
    ax.grid(True, axis="y", alpha=0.25)


def _plot_representative_windows(old_metrics: dict, new_metrics: dict, fig_path: Path) -> None:
    model_payload = {
        OLD_MODEL_LABEL: old_metrics,
        "AGC / Transformer": new_metrics["models"]["AGC / Transformer"],
        "AGC / Transformer-hybrid": new_metrics["models"]["AGC / Transformer-hybrid"],
    }

    fig, axes = plt.subplots(
        len(COMMON_VARS),
        len(WINDOW_MODEL_LABELS),
        figsize=(15.5, 9.5),
        squeeze=False,
    )

    for col_idx, label in enumerate(WINDOW_MODEL_LABELS):
        payload = model_payload[label]
        rep = payload["representative_window"]
        step_minutes = payload.get("step_minutes", new_metrics["step_minutes"])
        true_arr = np.asarray(rep["true"], dtype=np.float32)
        pred_arr = np.asarray(rep["pred"], dtype=np.float32)
        time_axis = np.arange(1, len(true_arr) + 1, dtype=np.float32) * step_minutes

        for row_idx, var_name in enumerate(COMMON_VARS):
            ax = axes[row_idx][col_idx]
            ax.plot(time_axis, true_arr[:, row_idx], color="black", linewidth=2.0, label="True")
            ax.plot(
                time_axis,
                pred_arr[:, row_idx],
                color=COLOR_MAP[label],
                linewidth=2.0,
                linestyle="--",
                label="Pred",
            )
            if row_idx == 0:
                ax.set_title(label)
            if col_idx == 0:
                ax.set_ylabel(var_name)
            if row_idx == len(COMMON_VARS) - 1:
                ax.set_xlabel("Minutes ahead")
            ax.grid(True, alpha=0.25)
            if row_idx == 0 and col_idx == 0:
                ax.legend()

    fig.suptitle("Representative 2-hour forecast windows (mid-test sample)", fontsize=16)
    fig.text(
        0.5,
        0.01,
        "Each panel uses the midpoint test sample from its own dataset. "
        "The purpose is qualitative comparison, not sample-by-sample alignment.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_dataset_switch_figure(repo_root: Path) -> tuple[Path, Path, Path]:
    cfg = AGCConfig()
    ensure_results_layout(cfg)

    old_metrics = _collect_old_metrics(repo_root)
    new_metrics = _collect_new_metrics(repo_root)

    out_dir = repo_root / "agc_mpc" / "results" / "forecasting" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "strawberry_vs_agc_dataset_switch.png"
    window_fig_path = out_dir / "strawberry_vs_agc_forecast_windows.png"
    json_path = out_dir / "strawberry_vs_agc_dataset_switch_summary.json"

    fig = plt.figure(figsize=(18, 6.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.2, 1.2])
    ax_table = fig.add_subplot(gs[0, 0])
    ax_mae = fig.add_subplot(gs[0, 1])
    ax_r2 = fig.add_subplot(gs[0, 2])

    _draw_feature_table(ax_table)
    _plot_mae_ratio(ax_mae, old_metrics, new_metrics)
    _plot_final_r2(ax_r2, old_metrics, new_metrics)

    handles, labels = ax_r2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Why AGC Is More Suitable Than the Old Strawberry Dataset", fontsize=16, y=1.06)
    fig.text(
        0.5,
        -0.02,
        "Metrics compare common variables only (Temperature / Humidity / CO2). "
        "Both tasks represent a 2-hour horizon, while sampling differs: Strawberry = 120 x 1 min, AGC = 24 x 5 min.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    old_metrics["step_minutes"] = old_metrics["step_minutes"]
    new_metrics["step_minutes"] = new_metrics["step_minutes"]
    _plot_representative_windows(old_metrics, new_metrics, window_fig_path)

    summary = {
        "old_project": old_metrics,
        "new_project": new_metrics,
        "figure_path": str(fig_path),
        "window_figure_path": str(window_fig_path),
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return fig_path, window_fig_path, json_path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    fig_path, window_fig_path, json_path = build_dataset_switch_figure(repo_root)
    print(f"Saved comparison figure: {fig_path}")
    print(f"Saved representative windows figure: {window_fig_path}")
    print(f"Saved comparison summary: {json_path}")


if __name__ == "__main__":
    main()
