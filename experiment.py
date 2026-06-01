"""Experiment directory layout: experiments/<name>/{input.txt, model.npz, plots/}."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"

# Per-regime defaults for corpus size, training, and visualization window.
EXPERIMENT_CONFIG: dict[str, dict[str, int]] = {
    "one_word": {"chars": 10_000, "steps": 8_000, "viz_length": 50},
    "shared_letters": {"chars": 50_000, "steps": 15_000, "viz_length": 100},
    "disjoint_letters": {"chars": 50_000, "steps": 15_000, "viz_length": 100},
    "ten_word_overlap": {"chars": 50_000, "steps": 15_000, "viz_length": 100},
}


def experiment_dir(name: str) -> Path:
    return EXPERIMENTS_ROOT / name


def input_path(name: str) -> Path:
    return experiment_dir(name) / "input.txt"


def model_path(name: str) -> Path:
    return experiment_dir(name) / "model.npz"


def plots_dir(name: str) -> Path:
    return experiment_dir(name) / "plots"


def plot_path(name: str, plot_name: str) -> Path:
    """e.g. plot_path('shared_letters', 'learning_curve.png')"""
    return plots_dir(name) / plot_name


def ensure_experiment_dirs(name: str) -> None:
    plots_dir(name).mkdir(parents=True, exist_ok=True)
