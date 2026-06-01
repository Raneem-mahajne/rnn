"""Experiment directory layout: experiments/<name>/{input.txt, model.npz, plots/}."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"

# Per-regime defaults for corpus size, training, and visualization window.
_BASE_CONFIG: dict[str, dict[str, int]] = {
    "one_word": {"chars": 10_000, "steps": 8_000, "viz_length": 50},
    "shared_letters": {"chars": 50_000, "steps": 15_000, "viz_length": 100},
    "disjoint_letters": {"chars": 50_000, "steps": 15_000, "viz_length": 100},
    "ten_word_overlap": {"chars": 50_000, "steps": 15_000, "viz_length": 100},
}


def spaced_experiment_name(regime: str) -> str:
    """Experiment folder name when words are separated by spaces in the corpus."""
    return f"{regime}_s"


def _build_experiment_config() -> dict[str, dict]:
    configs: dict[str, dict] = {}
    for regime, cfg in _BASE_CONFIG.items():
        configs[regime] = {**cfg, "regime": regime, "word_space": False}
        configs[spaced_experiment_name(regime)] = {
            **cfg,
            "regime": regime,
            "word_space": True,
        }
    return configs


EXPERIMENT_CONFIG: dict[str, dict] = _build_experiment_config()


def experiment_uses_word_space(name: str) -> bool:
    return bool(EXPERIMENT_CONFIG.get(name, {}).get("word_space", False))


def experiment_regime(name: str) -> str:
    """Underlying task regime (word list key in task.REGIMES)."""
    cfg = EXPERIMENT_CONFIG.get(name)
    if cfg and "regime" in cfg:
        return cfg["regime"]
    if name.endswith("_s"):
        return name[:-2]
    return name


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
