# -*- coding: utf-8 -*-
"""Run the paper-facing final FCTV closed-loop benchmark.

This script fixes the cross-family predictor pool and repeated start indices
defined in FCTV_EXPERIMENT_DESIGN.md. It is intentionally a thin wrapper around
run_fctv_multistart_control.py so the formal benchmark can be reproduced with a
single command.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import AGCConfig
from control_main import _apply_three_target_control_protocol
from run_fctv_multistart_control import run_multistart


FINAL_FCTV_PREDICTORS = [
    "dlinear_forecaster",
    "nlinear_forecaster",
    "gru_forecaster",
    "lstm_forecaster",
    "segrnn_forecaster",
    "frequency_forecaster",
    "transformer_forecaster",
    "current_hybrid_transformer",
    "transformer_hybrid_residual",
    "patchtst_residual",
    "itransformer_residual",
    "itransformer_co2_residual",
    "itransformer_co2_late_residual",
    "itransformer_co2_late_frozen_expert",
    "itransformer_co2_horizon_mixture",
    "itransformer_co2_control_aware_fusion",
]

FINAL_FCTV_START_INDICES = [0, 96, 192, 288, 384]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--compartment", default="Reference")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--start-indices",
        nargs="+",
        type=int,
        default=FINAL_FCTV_START_INDICES,
        help="Repeated closed-loop starts for the formal benchmark.",
    )
    parser.add_argument(
        "--predictors",
        nargs="+",
        default=FINAL_FCTV_PREDICTORS,
        help="Override the fixed final FCTV predictor pool only for diagnostics.",
    )
    parser.add_argument(
        "--print-plan",
        action="store_true",
        help="Print the formal benchmark plan without running MPC rollouts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.print_plan:
        print("Final FCTV closed-loop benchmark")
        print(f"predictors ({len(args.predictors)}):")
        for predictor in args.predictors:
            print(f"  - {predictor}")
        print(f"start_indices: {args.start_indices}")
        print(f"steps: {args.steps}")
        print(f"compartment: {args.compartment}")
        print("analysis command after run:")
        print(
            "  python agc_mpc/analyze_fctv_multistart_transfer.py "
            "--suite-json <generated_suite_json> "
            "--prefix forecast_to_control_transfer_final_reference"
        )
        return

    cfg = _apply_three_target_control_protocol(AGCConfig())
    cfg.control_compartment = args.compartment
    cfg.control_eval_steps = args.steps
    cfg.seed = args.seed
    suite = run_multistart(cfg, args.predictors, args.start_indices)
    print("Generated suite with records:", len(suite.get("records", [])))
    print("Project root:", Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
