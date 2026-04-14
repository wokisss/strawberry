# -*- coding: utf-8 -*-
"""Helpers for AGC result directory layout and legacy migration."""

from __future__ import annotations

import shutil
from pathlib import Path

from figure_layout import (
    baseline_figures_dir,
    co2_specialist_figures_dir,
    comparison_figures_dir,
    current_hybrid_figures_dir,
    residual_figures_dir,
)

def _move_if_needed(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve() or dst.exists():
        return
    try:
        shutil.move(str(src), str(dst))
    except PermissionError:
        shutil.copy2(str(src), str(dst))


def ensure_results_layout(cfg) -> None:
    """Create the new results tree and migrate legacy files when present."""
    for path_str in [
        cfg.results_dir,
        cfg.forecast_results_dir,
        cfg.forecast_checkpoints_dir,
        cfg.forecast_figures_dir,
        cfg.forecast_analysis_dir,
        cfg.control_results_dir,
        cfg.control_summaries_dir,
        cfg.control_figures_dir,
    ]:
        Path(path_str).mkdir(parents=True, exist_ok=True)

    for subdir in [
        baseline_figures_dir(cfg.forecast_figures_dir),
        current_hybrid_figures_dir(cfg.forecast_figures_dir),
        residual_figures_dir(cfg.forecast_figures_dir),
        comparison_figures_dir(cfg.forecast_figures_dir),
        co2_specialist_figures_dir(cfg.forecast_figures_dir),
    ]:
        subdir.mkdir(parents=True, exist_ok=True)

    legacy_root = Path(cfg.results_dir)
    legacy_figures = legacy_root / "figures"
    legacy_control = legacy_root / "control"

    for file in legacy_root.glob("*.pt"):
        _move_if_needed(file, Path(cfg.forecast_checkpoints_dir) / file.name)

    if legacy_figures.exists():
        for file in legacy_figures.glob("*.png"):
            _move_if_needed(file, Path(cfg.forecast_figures_dir) / file.name)

    if legacy_control.exists():
        for file in legacy_control.glob("*_summary.json"):
            _move_if_needed(file, Path(cfg.control_summaries_dir) / file.name)
        legacy_control_figures = legacy_control / "figures"
        if legacy_control_figures.exists():
            for file in legacy_control_figures.glob("*.png"):
                _move_if_needed(file, Path(cfg.control_figures_dir) / file.name)
