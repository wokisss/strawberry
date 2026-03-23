# -*- coding: utf-8 -*-
"""Entry point for DiffMPC training and evaluation."""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

from config import Config
from controllers.dpc_controller import DPCController
from controllers.pwm_driver import PWMDriver, PWMSimulator
from controllers.sac.replay_buffer import ReplayBuffer
from controllers.sac.sac_agent import SAC
from controllers.sac_controller import SACController
from data_processing.processor import DataProcessor
from environment.gym_wrapper import GreenhouseGymEnv
from environment.physics_env import PhysicsGreenhouseEnv
from models.decision_model import DecisionControlModel
from models.transformer_hybrid import TransformerHybridModel
from simulation.evaluator import PredictorEvaluator
from simulation.simulator import Simulator
from simulation.visualizer import Visualizer
from training.trainer import Trainer

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def setup_seed(seed):
    """Set deterministic seeds for reproducible experiments."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)

    print("=" * 60)
    print("--- Greenhouse Control System ---")
    print("--- DPC vs SAC ---")
    print("=" * 60)

    cfg = Config()
    setup_seed(cfg.seed)
    print(f"---> Random seed fixed at {cfg.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"---> Using device: {device}")

    processor = DataProcessor(cfg)
    df = processor.load_and_preprocess()
    df = processor.merge_weather(df)
    df = processor.add_time_encoding(df)
    df = processor.add_energy_features(df)
    df = processor.add_ode_derivatives(df)
    data_scaled = processor.prepare_features(df)
    datasets = processor.prepare_datasets(data_scaled)

    print("\n---> Initializing Transformer predictor...")
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

    trainer = Trainer(model, config=cfg, device=device)
    trainer.train(datasets["X_train_p"], datasets["X_train_f"], datasets["y_train"])

    print("\n---> Running offline predictor diagnostics...")
    evaluator = PredictorEvaluator(
        model,
        processor.scaler,
        processor.target_indices,
        processor.feature_order,
        cfg,
        device=device,
    )
    val_metrics = evaluator.evaluate(
        datasets["X_test_p"], datasets["X_test_f"], datasets["y_test"]
    )

    temp_viz = Visualizer(config=cfg)
    temp_viz.plot_predictor_diagnostics(val_metrics, save=True)

    print("\n---> Initializing control surrogate...")
    decision_model = DecisionControlModel(
        model,
        config=cfg,
        scaler=processor.scaler,
        target_indices=processor.target_indices,
    )
    print("    Physical guidance layer attached.")

    print("\n---> Initializing DPC controller...")
    print(f"    lr={cfg.dpc_lr}, iterations={cfg.dpc_iterations}")
    dpc = DPCController(
        decision_model,
        processor.scaler,
        processor.target_indices,
        processor.future_indices,
        config=cfg,
    )

    print("\n---> Initializing SAC environment and agent...")
    print(
        f"    train_steps={cfg.sac_train_steps}, "
        f"batch={cfg.sac_batch_size}, lr={cfg.sac_lr}"
    )

    gym_env = GreenhouseGymEnv(
        PhysicsGreenhouseEnv(np.zeros(3), config=cfg),
        config=cfg,
        scaler=processor.scaler,
        target_indices=processor.target_indices,
    )

    sac_agent = SAC(gym_env.observation_space.shape[0], gym_env.action_space, cfg, device)
    memory = ReplayBuffer(cfg.sac_replay_size, cfg.seed)

    print("    [SAC] Starting offline environment interaction...")
    updates = 0
    episodes = 0
    total_numsteps = 0

    while total_numsteps < cfg.sac_train_steps:
        episode_reward = 0.0
        episode_steps = 0
        done = False
        state, _ = gym_env.reset(seed=cfg.seed + episodes)

        while not done:
            if cfg.sac_train_steps <= total_numsteps:
                break

            action = sac_agent.select_action(state)
            next_state, reward, terminated, truncated, _ = gym_env.step(action)
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward
            done = terminated or truncated

            mask = 1 if episode_steps == gym_env.max_steps else float(not done)
            memory.push(state, action, reward, next_state, mask)
            state = next_state

            if len(memory) > cfg.sac_batch_size:
                sac_agent.update_parameters(memory, cfg.sac_batch_size, updates)
                updates += 1

        episodes += 1
        if episodes % 20 == 0 or total_numsteps >= cfg.sac_train_steps:
            print(
                f"        Episode: {episodes}, total_steps: {total_numsteps}, "
                f"episode_reward: {episode_reward:.2f}"
            )

    print("---> SAC training complete; building SAC controller...")
    sac_controller = SACController(
        sac_agent,
        processor.scaler,
        processor.target_indices,
        processor.future_indices,
        feature_order=processor.feature_order,
        config=cfg,
    )

    print("\n---> Initializing PWM drivers...")
    print(
        f"    cycle={cfg.pwm_cycle} min, min_on={cfg.pwm_min_on} min, "
        f"min_off={cfg.pwm_min_off} min"
    )

    pwm_driver_dpc = PWMDriver(
        cycle_minutes=cfg.pwm_cycle,
        min_on_minutes=cfg.pwm_min_on,
        min_off_minutes=cfg.pwm_min_off,
    )
    pwm_sim_dpc = PWMSimulator(pwm_driver_dpc)

    pwm_driver_sac = PWMDriver(
        cycle_minutes=cfg.pwm_cycle,
        min_on_minutes=cfg.pwm_min_on,
        min_off_minutes=cfg.pwm_min_off,
    )
    pwm_sim_sac = PWMSimulator(pwm_driver_sac)

    print("\n---> Running DPC vs SAC simulation...")

    if hasattr(torch, "compile") and sys.platform != "win32":
        print("    [Hardware Accel] Compiling decision model with torch.compile...")
        try:
            decision_model = torch.compile(decision_model)
            dpc.model = decision_model
        except Exception as exc:
            print(f"    [Warning] torch.compile failed, falling back to eager mode: {exc}")
    else:
        print("    [Hardware Accel] Skipping torch.compile on this platform.")

    sim = Simulator(
        dpc,
        sac_controller,
        PhysicsGreenhouseEnv,
        processor.feature_order,
        processor.scaler,
        processor.target_indices,
        config=cfg,
        device=device,
        pwm_sim_dpc=pwm_sim_dpc,
        pwm_sim_sac=pwm_sim_sac,
    )
    result = sim.run(datasets["X_test_p"], datasets["X_test_f"], config=cfg)

    print("\n---> Plotting results...")
    viz = Visualizer(config=cfg)
    viz.plot_comparison(result)

    print("\nDONE")


if __name__ == "__main__":
    main()
