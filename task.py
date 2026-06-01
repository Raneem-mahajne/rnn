"""
Statistical-learning task generator.

Task: sample a word uniformly from a small vocabulary, emit its characters,
repeat. Writes the resulting character stream to `input.txt` so it can be
consumed by `min-char-rnn.py` unchanged.

From: https://github.com/Raneem-mahajne/creating_transformer/tree/statistical_learning

Usage:
    python task.py                       # default: shared_letters, 50 chars
    python task.py shared_letters
    python task.py disjoint_letters --chars 200000
    python task.py one_word
    python task.py ten_word_overlap --chars 50000
    python task.py shared_letters --exp shared_letters   # -> experiments/shared_letters/input.txt
"""

from __future__ import annotations

import argparse
import random

from experiment import input_path as experiment_input_path

REGIMES: dict[str, list[str]] = {
    "one_word":         ["cat"],
    "disjoint_letters": ["cat", "mop", "red"],
    "shared_letters":   ["cat", "hat", "map"],
    # 10 words, length 3; overlap on -at/-et/-ea; vowels a, e, i.
    "ten_word_overlap": [
        "cat", "hat", "mat", "rat",
        "met", "pet", "net",
        "ate", "eat", "tea",
    ],
}


def generate_sequence(words: list[str], num_chars: int, seed: int = 0) -> str:
    rng = random.Random(seed)
    out: list[str] = []
    while len(out) < num_chars:
        out.extend(rng.choice(words))
    return "".join(out[:num_chars])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("regime", nargs="?", default="shared_letters",
                        choices=list(REGIMES.keys()))
    parser.add_argument("--chars", type=int, default=50,
                        help="total characters to emit (default: 50)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp", default=None,
                        help="experiment name (default: regime); writes experiments/<exp>/input.txt")
    parser.add_argument("--out", default=None,
                        help="output path (overrides --exp)")
    args = parser.parse_args()

    out_path = args.out
    if out_path is None:
        exp_name = args.exp or args.regime
        out_path = str(experiment_input_path(exp_name))
        experiment_input_path(exp_name).parent.mkdir(parents=True, exist_ok=True)

    words = REGIMES[args.regime]
    text = generate_sequence(words, args.chars, seed=args.seed)
    with open(out_path, "w") as f:
        f.write(text)

    vocab = sorted(set(text))
    print(f"Regime:  {args.regime}")
    print(f"Words:   {words}")
    print(f"Vocab:   {''.join(vocab)} ({len(vocab)} symbols)")
    print(f"Wrote:   {out_path} ({len(text):,} characters)")
    print(f"Preview: {text[:80]}")


if __name__ == "__main__":
    main()
