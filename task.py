"""
Statistical-learning task generator.

Task: sample a word uniformly from a small vocabulary, emit its characters,
repeat. Writes the resulting character stream to `input.txt` so it can be
consumed by `min-char-rnn.py` unchanged.

From: https://github.com/Raneem-mahajne/creating_transformer/tree/statistical_learning

Usage:
    python task.py ten_word_overlap --chars 50000
"""

from __future__ import annotations

import argparse
import random

from experiment import (
    EXPERIMENT_CONFIG,
    experiment_regime,
    experiment_uses_word_space,
    input_path as experiment_input_path,
    spaced_experiment_name,
)

REGIMES: dict[str, list[str]] = {
    # 10 words, length 3; overlap on -at/-et/-ea; vowels a, e, i.
    "ten_word_overlap": [
        "cat", "hat", "mat", "rat",
        "met", "pet", "net",
        "ate", "eat", "tea",
    ],
    # 12 words with shared prefixes and mixed lengths (richer minimal DFA).
    # Designed to have branching after the *first* character too (a→{n,t,x}, b→{a,e}).
    "twelve_word_overlap": [
        # a-family (splits after 'a')
        "an", "ant", "ante", "anti",
        "at",
        "ax",
        # b-family (splits after 'b')
        "ban", "band", "bank",
        "be", "bet", "bent",
    ],
    # 16 words with richer prefix branching and an extra vowel (o).
    # Also designed to have branching after the *first* character in each family.
    "sixteen_word_overlap": [
        # a-family
        "an", "ant", "ante", "anti",
        "at",
        # b-family
        "ban", "band", "bank",
        "bar", "bark", "barn",
        "be", "bet",
        # o-family (adds vowel + new branching after 'o')
        "on", "one", "or",
    ],
}


def generate_sequence(
    words: list[str],
    num_chars: int,
    seed: int = 0,
    *,
    word_space: bool = False,
) -> str:
    rng = random.Random(seed)
    out: list[str] = []
    while len(out) < num_chars:
        if word_space and out:
            out.append(" ")
            if len(out) >= num_chars:
                break
        for ch in rng.choice(words):
            if len(out) >= num_chars:
                break
            out.append(ch)
    return "".join(out[:num_chars])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("regime", nargs="?", default="ten_word_overlap",
                        choices=list(REGIMES.keys()))
    parser.add_argument("--chars", type=int, default=50,
                        help="total characters to emit (default: 50)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp", default=None,
                        help="experiment name (default: regime); writes experiments/<exp>/input.txt")
    parser.add_argument("--out", default=None,
                        help="output path (overrides --exp)")
    parser.add_argument(
        "--word-space",
        action="store_true",
        help="insert a space between sampled words (experiment name gets _s suffix)",
    )
    args = parser.parse_args()

    exp_name = args.exp or args.regime
    if args.exp and args.exp in EXPERIMENT_CONFIG:
        word_space = bool(EXPERIMENT_CONFIG[args.exp].get("word_space", False))
        regime = experiment_regime(args.exp)
    else:
        regime = args.regime
        word_space = args.word_space or experiment_uses_word_space(exp_name)
        if word_space and not exp_name.endswith("_s"):
            exp_name = spaced_experiment_name(regime)

    out_path = args.out
    if out_path is None:
        out_path = str(experiment_input_path(exp_name))
        experiment_input_path(exp_name).parent.mkdir(parents=True, exist_ok=True)

    words = REGIMES[regime]
    text = generate_sequence(words, args.chars, seed=args.seed, word_space=word_space)
    with open(out_path, "w") as f:
        f.write(text)

    vocab = sorted(set(text))
    print(f"Regime:  {regime}" + (" (word-space)" if word_space else ""))
    print(f"Exp:     {exp_name}")
    print(f"Words:   {words}")
    print(f"Vocab:   {''.join(vocab)} ({len(vocab)} symbols)")
    print(f"Wrote:   {out_path} ({len(text):,} characters)")
    print(f"Preview: {text[:80]}")


if __name__ == "__main__":
    main()
