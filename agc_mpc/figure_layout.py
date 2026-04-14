# -*- coding: utf-8 -*-
"""Centralized forecast/control figure subdirectory layout."""

from __future__ import annotations

from pathlib import Path


BASELINES_DIRNAME = "baselines"
CURRENT_HYBRID_DIRNAME = "current_hybrid_transformer"
RESIDUAL_VARIANTS_DIRNAME = "residual_variants"
COMPARISONS_DIRNAME = "comparisons"
CO2_SPECIALISTS_DIRNAME = "co2_specialists"


def baseline_figures_dir(root: str | Path) -> Path:
    return Path(root) / BASELINES_DIRNAME


def current_hybrid_figures_dir(root: str | Path) -> Path:
    return Path(root) / CURRENT_HYBRID_DIRNAME


def residual_figures_dir(root: str | Path) -> Path:
    return Path(root) / RESIDUAL_VARIANTS_DIRNAME


def comparison_figures_dir(root: str | Path) -> Path:
    return Path(root) / COMPARISONS_DIRNAME


def co2_specialist_figures_dir(root: str | Path) -> Path:
    return Path(root) / CO2_SPECIALISTS_DIRNAME
