# -*- coding: utf-8 -*-
"""Minimal runtime smoke test for controller/simulator integration."""

import os
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces

from config import Config
from controllers.dpc_controller import DPCController
from controllers.sac.sac_agent import SAC
from controllers.sac_controller import SACController
from data_processing.processor import DataProcessor
from environment.physics_env import PhysicsGreenhouseEnv
from models.decision_model import DecisionControlModel
from models.transformer_hybrid import TransformerHybridModel
from simulation.simulator import Simulator


def main():
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)

    cfg = Config()
    cfg.sim_steps = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    checkpoint_path = Path(cfg.model_save_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing predictor checkpoint: {checkpoint_path}")
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    decision_model = DecisionControlModel(
        model,
        config=cfg,
        scaler=processor.scaler,
        target_indices=processor.target_indices,
    )
    dpc = DPCController(
        decision_model,
        processor.scaler,
        processor.target_indices,
        processor.future_indices,
        config=cfg,
    )

    obs_dim = 15
    action_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
    sac_agent = SAC(obs_dim, action_space, cfg, device)
    sac_controller = SACController(
        sac_agent,
        processor.scaler,
        processor.target_indices,
        processor.future_indices,
        feature_order=processor.feature_order,
        config=cfg,
    )

    sim = Simulator(
        dpc,
        sac_controller,
        PhysicsGreenhouseEnv,
        processor.feature_order,
        processor.scaler,
        processor.target_indices,
        config=cfg,
        device=device,
    )
    result = sim.run(datasets["X_test_p"], datasets["X_test_f"], config=cfg)
    print("smoke_ok", len(result.history_dpc), len(result.history_sac))


if __name__ == "__main__":
    main()
