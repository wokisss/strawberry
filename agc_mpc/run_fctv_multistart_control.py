# -*- coding: utf-8 -*-
"""Run repeated GradientMPC rollouts for FCTV multi-start robustness."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from config import AGCConfig
from control.controller import GradientMPCController
from control.simulator import AGCClosedLoopSimulator
from control_main import (
    _apply_three_target_control_protocol,
    _build_model_specs,
    _load_checkpoint,
    _load_frozen_expert_if_needed,
    _load_main_if_needed,
    _set_global_seed,
)
from control_relevant_validation import DEFAULT_PREDICTORS
from control.controller import PredictiveControlAdapter
from data_processing.processor import AGCDataProcessor
from results_utils import ensure_results_layout


def _load_adapter(predictor: str, cfg: AGCConfig, scaled_bundle, raw_bundle, device: torch.device):
    specs = _build_model_specs(scaled_bundle, cfg)
    if predictor not in specs:
        raise ValueError(f"Unsupported predictor: {predictor}")
    model = specs[predictor]["builder"]()
    _load_frozen_expert_if_needed(model, predictor, cfg, device)
    _load_main_if_needed(model, predictor, cfg, device)
    _load_checkpoint(
        model,
        Path(cfg.forecast_checkpoints_dir) / specs[predictor]["checkpoint"],
        device,
    )
    return PredictiveControlAdapter(
        model=model,
        scalers=scaled_bundle["scalers"],
        feature_groups=scaled_bundle["feature_groups"],
        cfg=cfg,
        raw_bundle=raw_bundle,
        device=device,
    )


def run_multistart(cfg: AGCConfig, predictors: list[str], start_indices: list[int]) -> dict:
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)
    _set_global_seed(cfg.seed)
    ensure_results_layout(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AGCDataProcessor(cfg)
    raw_bundle = processor.build_compartment_raw_bundle(cfg.control_compartment)
    scaled_bundle = processor.build_compartment_bundle(cfg.control_compartment)

    records = []
    output_tag_prefix = str(getattr(cfg, "economic_profile_name", "") or "").strip()
    for start_idx in start_indices:
        cfg.control_start_idx = int(start_idx)
        start_tag = f"start{int(start_idx):05d}_{cfg.control_eval_steps}steps"
        cfg.control_output_tag = f"{output_tag_prefix}_{start_tag}" if output_tag_prefix else start_tag
        print(f"Running start_idx={start_idx} for {len(predictors)} predictors")
        for predictor in predictors:
            print(f"  {predictor}")
            adapter = _load_adapter(predictor, cfg, scaled_bundle, raw_bundle, device)
            simulator = AGCClosedLoopSimulator(adapter, raw_bundle, cfg)
            controller = GradientMPCController(adapter, cfg)
            summary = simulator.run(controller, predictor_name=predictor)
            record = asdict(summary)
            record["output_tag"] = cfg.control_output_tag
            records.append(record)

    suite = {
        "predictors": predictors,
        "controllers": ["gradient_mpc"],
        "compartment": cfg.control_compartment,
        "reference_mode": cfg.control_reference_mode,
        "rollout_mode": cfg.control_rollout_mode,
        "start_indices": start_indices,
        "steps": cfg.control_eval_steps,
        "target_cols": cfg.target_cols,
        "economic_profile_name": output_tag_prefix,
        "economic_resource_weight": float(getattr(cfg, "economic_resource_weight", 0.0)),
        "economic_action_weights": getattr(cfg, "economic_action_weights", {}),
        "records": records,
    }
    joined = "_".join(str(idx) for idx in start_indices)
    predictor_digest = hashlib.sha1("_".join(predictors).encode("utf-8")).hexdigest()[:10]
    out_path = Path(cfg.control_summaries_dir) / (
        f"fctv_multistart_gradient_mpc_{cfg.control_compartment.lower()}_"
        f"{cfg.control_eval_steps}steps_{len(predictors)}predictors_{predictor_digest}_"
        f"{output_tag_prefix + '_' if output_tag_prefix else ''}starts_{joined}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(suite, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved multi-start suite: {out_path}")
    return suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictors", nargs="+", default=DEFAULT_PREDICTORS)
    parser.add_argument("--start-indices", nargs="+", type=int, default=[0, 96, 192])
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--compartment", default="Reference")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _apply_three_target_control_protocol(AGCConfig())
    cfg.control_compartment = args.compartment
    cfg.control_eval_steps = args.steps
    cfg.seed = args.seed
    run_multistart(cfg, args.predictors, args.start_indices)


if __name__ == "__main__":
    main()
