# -*- coding: utf-8 -*-
"""Run a small economic/resource-aware MPC probe.

The probe keeps the same surrogate and GradientMPC solver as the tracking
benchmark, but enables the resource proxy term in the control objective. It is
intended as the first executable E-stage check, not as a final economic study.
"""

from __future__ import annotations

import argparse

from config import AGCConfig
from control_main import _apply_three_target_control_protocol
from run_fctv_multistart_control import run_multistart


DEFAULT_ECONOMIC_PREDICTORS = [
    "current_hybrid_transformer",
    "itransformer_co2_residual",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictors", nargs="+", default=DEFAULT_ECONOMIC_PREDICTORS)
    parser.add_argument("--start-indices", nargs="+", type=int, default=[0])
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--compartment", default="Reference")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resource-weight", type=float, default=0.15)
    parser.add_argument("--profile-name", default="economic_probe_w015")
    parser.add_argument(
        "--print-plan",
        action="store_true",
        help="Print the economic probe plan without running MPC rollouts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _apply_three_target_control_protocol(AGCConfig())
    cfg.control_compartment = args.compartment
    cfg.control_eval_steps = args.steps
    cfg.seed = args.seed
    cfg.economic_resource_weight = float(args.resource_weight)
    cfg.economic_profile_name = args.profile_name

    if args.print_plan:
        print("Economic/resource-aware MPC probe")
        print(f"profile: {cfg.economic_profile_name}")
        print(f"resource_weight: {cfg.economic_resource_weight}")
        print(f"predictors ({len(args.predictors)}):")
        for predictor in args.predictors:
            print(f"  - {predictor}")
        print(f"start_indices: {args.start_indices}")
        print(f"steps: {args.steps}")
        print("action weights:")
        for key, value in cfg.economic_action_weights.items():
            print(f"  - {key}: {value}")
        return

    suite = run_multistart(cfg, args.predictors, args.start_indices)
    print("Generated economic/resource-aware suite records:", len(suite.get("records", [])))


if __name__ == "__main__":
    main()

