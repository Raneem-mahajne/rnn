"""
Train and visualize every task regime under experiments/<regime>/.

    python run_experiments.py
    python run_experiments.py --only shared_letters ten_word_overlap
    python run_experiments.py --skip-train   # visualize existing checkpoints only
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from experiment import EXPERIMENT_CONFIG, experiment_regime, model_path, input_path


def run(cmd: list[str]) -> None:
    print(f"\n>> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=list(EXPERIMENT_CONFIG.keys()),
        help="subset of experiments to run (include <regime>_s for word-space corpora)",
    )
    parser.add_argument("--skip-train", action="store_true",
                        help="only run visualize.py (requires existing model.npz)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    names = args.only if args.only else list(EXPERIMENT_CONFIG.keys())

    for name in names:
        cfg = EXPERIMENT_CONFIG.get(name, EXPERIMENT_CONFIG["shared_letters"])
        regime = experiment_regime(name)
        print(f"\n=== {name} ===")

        if not args.skip_train:
            run([
                sys.executable, "task.py", regime,
                "--exp", name,
                "--chars", str(cfg["chars"]),
                "--seed", str(args.seed),
            ])
            run([
                sys.executable, "min-char-rnn.py",
                "--input", str(input_path(name)),
                "--model", str(model_path(name)),
                "--steps", str(cfg["steps"]),
            ])

        run([
            sys.executable, "visualize.py",
            "--exp", name,
            "--length", str(cfg["viz_length"]),
        ])

    print(f"\nDone. Plots under experiments/<name>/plots/")


if __name__ == "__main__":
    main()
