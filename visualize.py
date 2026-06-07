"""
Visualize a trained min-char-rnn after training is done.

Loads the saved model from `model.npz`, runs a forward pass over the first
`--length` characters of `input.txt`, and plots:

  1) activation_heatmap.png
       Heatmap of every hidden unit's activation at every timestep.
       Y-axis: hidden units (h0, h1, ...). X-axis: input character.
       Works for any `hidden_size`.

  2) next_char_prob_sequence_heatmap.png
       Heatmap of the model's next-char probability distribution at every
       position, with the actual next character marked.

  3) state_trajectory_by_input.png   (only when hidden_size == 2)
       2D scatter of every hidden state visited, colored by the input
       character that produced it, with grey arrows showing the temporal
       trajectory through state space.

  4) state_trajectory_by_target.png    (only when hidden_size == 2)
       Same scatter, colored by the *next* (target) character.

  5) learning_curve.png
       Per-window training loss vs iteration (from model.npz).

  6) embedding_panels_context.png
       2D embeddings (PCA/UMAP/t-SNE/Isomap) of hidden states with context labels.

  7) next_char_regions_pca.png
       Two PCA panels: argmax next-char regions and prediction entropy (2D h).

  8) next_char_prob_panels_pca.png
       One panel per vocab char: P(next = char) over the PCA plane (softmax).

  9) activation_clustered_heatmap.png
       Heatmap of timesteps × hidden units with row/column dendrograms
       (average linkage). Row labels: two preceding chars + current char.

  10) state_correlation_clustered_heatmap.png
       Timestep × timestep Pearson correlation of hidden states, hierarchically
       clustered; row/column labels = prefix since last space; tick colors = min DFA state.

  10b) state_correlation_by_dfa_state.png
       Timesteps grouped by min DFA state; Pearson r within and between state blocks.

  11) dfa_state_distance_comparison.png
       Pairwise Euclidean distances between hidden states; bars = within vs between
       minimized DFA state (all timestep pairs in the test window).

  12) weights.png
       Side-by-side heatmaps of final input weights (char columns × hidden rows)
       and recurrent hidden→hidden weights (h0..h{n-1} in index order).

  13) weight_dynamics_over_training.png
       Eight E/I-block heatmaps of W_xh and W_hh weights over training snapshots.

Usage:
    python visualize.py --exp ten_word_overlap_s
    python visualize.py --exp ten_word_overlap --length 100
    python visualize.py --model path/to/model.npz --input path/to/input.txt --out-dir path/to/plots
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import ndimage

from experiment import (
    ensure_experiment_dirs,
    experiment_uses_word_space,
    input_path,
    model_path,
    plots_dir,
)
from task import REGIMES
from rnn_dyn import activation_label, no_input_hidden_step, rnn_hidden_step
from vocab_diagrams import (
    MinimizedVocabAutomaton,
    build_minimized_vocabulary_automaton,
    dfa_state_at_position,
    dfa_state_label,
    draw_minimized_dfa_on_axes,
    vocabulary_for_experiment,
    write_vocabulary_diagrams,
)


def load_model(path: str = "model.npz"):
    data = np.load(path, allow_pickle=False)
    model = {
        "weights_input_to_hidden":  data["weights_input_to_hidden"],
        "weights_hidden_to_hidden": data["weights_hidden_to_hidden"],
        "weights_hidden_to_output": data["weights_hidden_to_output"],
        "bias_hidden":              data["bias_hidden"],
        "bias_output":              data["bias_output"],
        "chars":                    [str(c) for c in data["chars"]],
        "hidden_size":              int(data["hidden_size"]),
        "vocab_size":               int(data["vocab_size"]),
    }
    if "loss_iterations" in data.files:
        model["loss_iterations"] = data["loss_iterations"]
        model["loss_smooth"] = data["loss_smooth"]
        model["loss_window"] = data["loss_window"]
    if "metric_iterations" in data.files:
        model["metric_iterations"] = data["metric_iterations"]
        model["metric_valid_vocab_letter_frac"] = data["metric_valid_vocab_letter_frac"]
    if "vocab_words" in data.files:
        model["vocab_words"] = [str(w) for w in data["vocab_words"]]
    if "sample_before" in data.files:
        model["sample_before"] = str(data["sample_before"])
    if "sample_after" in data.files:
        model["sample_after"] = str(data["sample_after"])
    if "demo_target" in data.files:
        model["demo_target"] = str(data["demo_target"])
    if "demo_prompt" in data.files:
        model["demo_prompt"] = str(data["demo_prompt"])
    if "demo_before" in data.files:
        model["demo_before"] = str(data["demo_before"])
    if "demo_after" in data.files:
        model["demo_after"] = str(data["demo_after"])
    if "demo_word_error_frac" in data.files:
        model["demo_word_error_frac"] = float(data["demo_word_error_frac"])
    if "demo_rng_seed" in data.files:
        model["demo_rng_seed"] = int(data["demo_rng_seed"])
    if "demo_seed_char" in data.files:
        model["demo_seed_char"] = str(data["demo_seed_char"])
    if "dale_law" in data.files:
        model["dale_law"] = bool(data["dale_law"])
    if "use_relu" in data.files:
        model["use_relu"] = bool(data["use_relu"])
    elif "dale_law" in model:
        model["use_relu"] = model["dale_law"]
    else:
        model["use_relu"] = False
    if "e_fraction" in data.files:
        model["e_fraction"] = float(data["e_fraction"])
    if "dale_sign" in data.files:
        ds = data["dale_sign"]
        model["dale_sign"] = ds if len(ds) else None
    if "weight_snap_iterations" in data.files:
        model["weight_snap_iterations"] = data["weight_snap_iterations"]
        model["weight_snap_outgoing"] = data["weight_snap_outgoing"]
        model["weight_snap_violation_frac"] = data["weight_snap_violation_frac"]
    if "metric_word_error_frac" in data.files:
        model["metric_word_error_frac"] = data["metric_word_error_frac"]
    return model


def forward_pass(model, text: str):
    """Run the trained RNN over `text` and return per-timestep states + probs."""
    hidden_size = model["hidden_size"]
    vocab_size  = model["vocab_size"]
    chars       = model["chars"]
    char_to_index = {c: i for i, c in enumerate(chars)}

    weights_input_to_hidden  = model["weights_input_to_hidden"]
    weights_hidden_to_hidden = model["weights_hidden_to_hidden"]
    weights_hidden_to_output = model["weights_hidden_to_output"]
    bias_hidden              = model["bias_hidden"]
    bias_output              = model["bias_output"]

    hidden_state = np.zeros((hidden_size, 1))
    hidden_states = np.zeros((len(text), hidden_size))
    output_probs  = np.zeros((len(text), vocab_size))

    for t, char in enumerate(text):
        input_one_hot = np.zeros((vocab_size, 1))
        input_one_hot[char_to_index[char]] = 1
        hidden_state, _ = rnn_hidden_step(
            hidden_state,
            input_one_hot,
            weights_input_to_hidden,
            weights_hidden_to_hidden,
            bias_hidden,
            use_relu=model.get("use_relu", False),
        )
        logits = weights_hidden_to_output @ hidden_state + bias_output
        exp = np.exp(logits - np.max(logits))
        probs = exp / np.sum(exp)

        hidden_states[t] = hidden_state.ravel()
        output_probs[t]  = probs.ravel()

    return hidden_states, output_probs


def plot_state_trajectory(hidden_states, color_by_chars, chars, title, save_path):
    """2D scatter of hidden states colored by some categorical char per timestep."""
    if hidden_states.shape[1] != 2:
        raise ValueError(
            f"This plot expects hidden_size == 2, got {hidden_states.shape[1]}. "
            f"Re-train with hidden_size = 2 (already the default in min-char-rnn.py)."
        )

    cmap = plt.get_cmap("tab10")
    char_to_color = {c: cmap(i) for i, c in enumerate(chars)}

    fig, ax = plt.subplots(figsize=(8, 7))

    xs, ys = hidden_states[:, 0], hidden_states[:, 1]
    ax.plot(xs, ys, color="lightgrey", linewidth=0.5, zorder=1)
    ax.quiver(
        xs[:-1], ys[:-1],
        xs[1:] - xs[:-1], ys[1:] - ys[:-1],
        angles="xy", scale_units="xy", scale=1,
        color="lightgrey", width=0.002, headwidth=4, alpha=0.6, zorder=1,
    )

    for c in chars:
        mask = np.array([ch == c for ch in color_by_chars])
        if not mask.any():
            continue
        ax.scatter(
            xs[mask], ys[mask],
            color=char_to_color[c], label=repr(c), s=30,
            edgecolor="black", linewidth=0.3, zorder=3,
        )

    ax.set_xlabel("hidden unit 0")
    ax.set_ylabel("hidden unit 1")
    ax.set_title(title)
    ax.legend(title="char", loc="best", framealpha=0.9)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"wrote {save_path}")


def plot_hidden_states_heatmap(text, hidden_states, save_path, *, act_label: str = "tanh"):
    """Heatmap of every hidden unit's activation over the sequence."""
    length, hidden_size = hidden_states.shape
    use_relu = act_label == "relu"
    cmap = "magma" if use_relu else "RdBu_r"
    vmin = 0.0 if use_relu else -1.0
    vmax = None if use_relu else 1.0

    fig, ax = plt.subplots(figsize=(max(12, length * 0.15),
                                    max(2.5, hidden_size * 0.35)))
    im = ax.imshow(
        hidden_states.T,
        aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
        interpolation="nearest", origin="lower",
    )

    ax.set_yticks(range(hidden_size))
    ax.set_yticklabels([f"h{i}" for i in range(hidden_size)])
    ax.set_xticks(range(length))
    ax.set_xticklabels(list(text), fontsize=7)
    ax.set_xlabel("timestep / input character")
    ax.set_ylabel("hidden unit")
    ax.set_title(f"Hidden state activations ({act_label} output) over the input sequence")

    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label=f"activation ({act_label})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"wrote {save_path}")


def average_linkage_hierarchy(rows):
    """Average-linkage clustering; returns (linkage, leaf_order).

    linkage has shape (n-1, 4) with columns [left_id, right_id, distance, count]
    in the same convention as scipy.cluster.hierarchy.linkage.
    """
    n_rows = rows.shape[0]
    if n_rows == 0:
        return np.zeros((0, 4)), []
    if n_rows == 1:
        return np.zeros((0, 4)), [0]

    distances = np.linalg.norm(rows[:, None, :] - rows[None, :, :], axis=2)
    clusters = [
        {"indices": [i], "members": [i], "cluster_id": i, "size": 1}
        for i in range(n_rows)
    ]
    linkage = []
    next_cluster_id = n_rows

    while len(clusters) > 1:
        best_pair = None
        best_distance = np.inf

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                member_distances = distances[np.ix_(
                    clusters[i]["members"],
                    clusters[j]["members"],
                )]
                distance = float(np.mean(member_distances))
                if distance < best_distance:
                    best_distance = distance
                    best_pair = (i, j)

        left, right = best_pair
        left_cluster, right_cluster = clusters[left], clusters[right]
        linkage.append([
            left_cluster["cluster_id"],
            right_cluster["cluster_id"],
            best_distance,
            left_cluster["size"] + right_cluster["size"],
        ])
        merged = {
            "indices": left_cluster["indices"] + right_cluster["indices"],
            "members": left_cluster["members"] + right_cluster["members"],
            "cluster_id": next_cluster_id,
            "size": left_cluster["size"] + right_cluster["size"],
        }
        next_cluster_id += 1
        clusters[left] = merged
        del clusters[right]

    return np.array(linkage), clusters[0]["indices"]


def average_linkage_cluster_order(rows):
    """Return row indices ordered by a small average-linkage clustering pass."""
    _, order = average_linkage_hierarchy(rows)
    return order


def display_char(char):
    """Format a character so labels stay readable for whitespace too."""
    if char == "\n":
        return "\\n"
    if char == "\t":
        return "\\t"
    if char == " ":
        return "space"
    return char


def argmax_region_glyph(char: str) -> str | None:
    """Single visible glyph for an argmax region label (None = skip region)."""
    if char == " ":
        return "␣"
    if len(char) == 1:
        return char
    return None


def corpus_uses_word_spacing(text: str, exp_name: str | None = None) -> bool:
    if exp_name is not None and experiment_uses_word_space(exp_name):
        return True
    return " " in text


def word_subsequent_label(text: str, index: int) -> str:
    """Spaced corpora: ' ' on a space; else prefix from after the last space through here."""
    if index < 0 or index >= len(text):
        return ""
    if text[index] == " ":
        return " "
    start = index
    while start > 0 and text[start - 1] != " ":
        start -= 1
    return text[start : index + 1]


def space_to_space_segments(text: str) -> list[tuple[int, int, str]]:
    """
    Inclusive timestep ranges from one space to the next (or document boundaries).

    Each segment includes both endpoint spaces when present.
    """
    n = len(text)
    if n == 0:
        return []

    space_ix = [i for i, c in enumerate(text) if c == " "]
    if not space_ix:
        return [(0, n - 1, text)]

    segments: list[tuple[int, int, str]] = []
    if space_ix[0] > 0:
        segments.append((0, space_ix[0], text[: space_ix[0] + 1]))
    for start, end in zip(space_ix, space_ix[1:]):
        segments.append((start, end, text[start : end + 1]))
    if space_ix[-1] < n - 1:
        segments.append((space_ix[-1], n - 1, text[space_ix[-1] :]))
    return segments


def segment_word_label(segment_text: str) -> str:
    """Readable label for a space-to-space path (stripped word, or ␣ for spaces only)."""
    stripped = segment_text.strip()
    return stripped if stripped else "␣"


def context_label(text, index, *, spaced: bool = False):
    if spaced:
        return word_subsequent_label(text, index)
    previous = "^" if index == 0 else display_char(text[index - 1])
    current = display_char(text[index])
    return f"{previous}{current}@{index}"


def timestep_context_label(text, index, *, spaced: bool = False):
    """Context string for plot labels (trigram, or growing prefix after last space)."""
    if spaced:
        return word_subsequent_label(text, index)
    if index < 0:
        return ""
    parts = []
    for offset in (2, 1, 0):
        pos = index - offset
        if pos < 0:
            parts.append("^")
        else:
            parts.append(display_char(text[pos]))
    return "".join(parts)


def timestep_axis_description(text: str, exp_name: str | None = None) -> str:
    if corpus_uses_word_spacing(text, exp_name):
        return "timestep (prefix after space, or ' ')"
    return "timestep (prev2 + current)"


def infer_task_words(text: str) -> list[str] | None:
    """Best-matching word vocabulary from task.py regimes for this corpus."""
    text_chars = set(text)
    allow_space = " " in text_chars
    best_words = None
    best_char_count = None
    for words in REGIMES.values():
        regime_chars = set("".join(words))
        if allow_space:
            regime_chars.add(" ")
        if text_chars <= regime_chars:
            n = len(regime_chars)
            if best_char_count is None or n < best_char_count:
                best_words = words
                best_char_count = n
    return best_words


def original_vocabulary_title(chars, text: str | None = None) -> str:
    """Suptitle text: task word vocabulary (inferred) and model character vocab."""
    parts = []
    if text:
        words = infer_task_words(text)
        if words:
            parts.append(f"vocabulary: {', '.join(words)}")
    parts.append(f"chars: {''.join(chars)}")
    return " · ".join(parts)


def plot_hidden_states_clustermap(
    text, hidden_states, chars, save_path, *, exp_name: str | None = None
):
    """Heatmap (hidden units × timesteps) with seaborn clustermap layout."""
    n_rows, n_cols = hidden_states.shape
    if n_rows == 0:
        return

    spaced = corpus_uses_word_spacing(text, exp_name)
    row_labels = [timestep_context_label(text, t, spaced=spaced) for t in range(n_rows)]
    col_labels = [f"h{i}" for i in range(n_cols)]
    # Flip orientation: units on rows, timesteps on columns (makes long sequences readable).
    data = pd.DataFrame(hidden_states, index=row_labels, columns=col_labels).T

    grid = sns.clustermap(
        data,
        method="average",
        metric="euclidean",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        figsize=(max(10, n_rows * 0.24), max(6, n_cols * 0.55)),
        dendrogram_ratio=(0.12, 0.1),
        cbar=False,
        cbar_pos=None,
        xticklabels=True,
        yticklabels=True,
    )
    grid.ax_heatmap.set_xlabel(timestep_axis_description(text, exp_name))
    grid.ax_heatmap.set_ylabel("hidden unit")
    grid.ax_heatmap.tick_params(axis="y", labelsize=8)
    grid.ax_heatmap.tick_params(axis="x", labelsize=7)
    grid.fig.suptitle(
        f"Hidden states clustered (units × timesteps) · {original_vocabulary_title(chars, text)}",
        y=1.02, fontsize=11,
    )
    grid.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(grid.fig)
    print(f"wrote {save_path}")


def prefix_tick_label(text: str, index: int, *, spaced: bool) -> str:
    """Axis tick text: in-word prefix since last space (␣ on spaces)."""
    label = prefix_annotation_label(text, index, spaced=spaced)
    return "␣" if label == " " else label


def plot_hidden_states_correlation_clustermap(
    text: str,
    hidden_states: np.ndarray,
    chars,
    save_path: str,
    *,
    spaced: bool = False,
    automaton: MinimizedVocabAutomaton | None = None,
):
    """One clustered matrix: Pearson r between hidden states at each timestep."""
    n = hidden_states.shape[0]
    if n < 2:
        return

    labels = [prefix_tick_label(text, t, spaced=spaced) for t in range(n)]
    corr = np.corrcoef(hidden_states)
    np.fill_diagonal(corr, 1.0)
    corr = np.nan_to_num(corr, nan=0.0)

    data = pd.DataFrame(corr, index=labels, columns=labels)

    state_ids = None
    state_colors = None
    if automaton is not None:
        state_ids = [
            dfa_state_at_position(text, t, automaton, spaced=spaced) for t in range(n)
        ]
        state_colors = _state_id_colors(state_ids)

    panel = max(10.0, n * 0.2)
    grid = sns.clustermap(
        data,
        method="average",
        metric="euclidean",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        figsize=(panel, panel),
        dendrogram_ratio=(0.12, 0.12),
        cbar=False,
        cbar_pos=None,
        xticklabels=True,
        yticklabels=True,
    )

    if state_ids is not None and state_colors is not None:
        row_order = grid.dendrogram_row.reordered_ind
        col_order = grid.dendrogram_col.reordered_ind
        for tick, idx in zip(grid.ax_heatmap.get_yticklabels(), row_order):
            tick.set_color(state_colors[state_ids[idx]])
        for tick, idx in zip(grid.ax_heatmap.get_xticklabels(), col_order):
            tick.set_color(state_colors[state_ids[idx]])

    axis_label = (
        "prefix since last space"
        if spaced
        else "stream prefix"
    )
    grid.ax_heatmap.set_xlabel(axis_label)
    grid.ax_heatmap.set_ylabel(axis_label)
    grid.ax_heatmap.tick_params(axis="both", labelsize=7)
    plt.setp(grid.ax_heatmap.get_xticklabels(), rotation=90, ha="center")

    title = "Hidden-state correlation"
    if automaton is not None:
        title += " · tick color = min DFA state"
    grid.fig.suptitle(
        f"{title} · {original_vocabulary_title(chars, text)}",
        y=1.02,
        fontsize=11,
    )
    grid.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(grid.fig)
    print(f"wrote {save_path}")


def _invalid_word_fraction(sampled_text: str, vocab: set[str]) -> float:
    if not vocab:
        return float("nan")
    tokens = [t for t in sampled_text.split(" ") if t]
    if not tokens:
        return float("nan")
    bad = sum(1 for t in tokens if t not in vocab)
    return bad / len(tokens)


def plot_dfa_grouped_state_correlation(
    text: str,
    hidden_states: np.ndarray,
    save_path: str,
    *,
    spaced: bool = False,
    automaton: MinimizedVocabAutomaton,
) -> None:
    """Pearson r between hidden vectors; rows/cols grouped by min DFA state (all blocks)."""
    n = hidden_states.shape[0]
    if n < 2:
        return

    state_ids = [
        dfa_state_at_position(text, t, automaton, spaced=spaced) for t in range(n)
    ]
    by_state: dict[int, list[int]] = {}
    for t, sid in enumerate(state_ids):
        by_state.setdefault(sid, []).append(t)

    order: list[int] = []
    boundaries: list[int] = [0]
    block_labels: list[str] = []
    for sid in sorted(by_state.keys()):
        idxs = sorted(
            by_state[sid],
            key=lambda t: prefix_tick_label(text, t, spaced=spaced),
        )
        order.extend(idxs)
        boundaries.append(len(order))
        block_labels.append(dfa_state_label(sid, automaton))

    if len(order) < 2:
        print(f"skip {save_path}: need ≥2 timesteps")
        return

    corr = np.corrcoef(hidden_states[order])
    np.fill_diagonal(corr, 1.0)
    corr = np.nan_to_num(corr, nan=0.0)

    state_colors = _state_id_colors(state_ids)

    panel = max(9.0, len(order) * 0.14)
    fig, ax = plt.subplots(figsize=(panel, panel * 0.92), constrained_layout=True)
    im = ax.imshow(
        corr,
        aspect="equal",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
        origin="lower",
    )
    block_sids = sorted(by_state.keys())
    for b in boundaries[1:-1]:
        ax.axhline(b - 0.5, color="black", lw=0.8)
        ax.axvline(b - 0.5, color="black", lw=0.8)

    tick_pos: list[float] = []
    tick_labels: list[str] = []
    for sid, lo, hi, lab in zip(block_sids, boundaries[:-1], boundaries[1:], block_labels):
        tick_pos.append((lo + hi - 1) / 2.0)
        tick_labels.append(f"q{sid}: {lab}")
    tick_fs = max(5, min(8, 120 // max(len(tick_labels), 1)))

    ax.set_xticks(tick_pos)
    ax.set_yticks(tick_pos)
    ax.set_xticklabels(tick_labels, fontsize=tick_fs, rotation=55, ha="right")
    ax.set_yticklabels(tick_labels, fontsize=tick_fs)
    for tick, sid in zip(ax.get_xticklabels(), block_sids):
        tick.set_color(state_colors[sid])
    for tick, sid in zip(ax.get_yticklabels(), block_sids):
        tick.set_color(state_colors[sid])

    ax.set_xlabel("min DFA state (accepted prefixes)")
    ax.set_ylabel("min DFA state (accepted prefixes)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="Pearson r")
    fig.suptitle(
        "Hidden-state correlation grouped by min DFA state "
        "(diagonal = within state, off-diagonal = vs other states)",
        fontsize=10,
        y=1.02,
    )
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def pairwise_hidden_state_distance_groups(
    text: str,
    hidden_states: np.ndarray,
    state_ids: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """L2 distance for each timestep pair (i < j): within DFA, between DFA, same input char."""
    n = hidden_states.shape[0]
    within: list[float] = []
    between: list[float] = []
    same_input: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.linalg.norm(hidden_states[i] - hidden_states[j]))
            if state_ids[i] == state_ids[j]:
                within.append(dist)
            else:
                between.append(dist)
            if text[i] == text[j]:
                same_input.append(dist)
    return np.asarray(within), np.asarray(between), np.asarray(same_input)


def plot_dfa_state_distance_comparison(
    text: str,
    hidden_states: np.ndarray,
    automaton: MinimizedVocabAutomaton,
    save_path: str,
    *,
    spaced: bool = False,
) -> None:
    """Subsampled pairwise distances + mean (diamond) ± std; y-axis clipped at 0."""
    n = hidden_states.shape[0]
    if n < 2:
        return

    state_ids = [
        dfa_state_at_position(text, t, automaton, spaced=spaced) for t in range(n)
    ]
    within, between, same_input = pairwise_hidden_state_distance_groups(
        text, hidden_states, state_ids
    )
    if len(within) == 0 or len(between) == 0:
        print("DFA distance comparison: need both within- and between-state pairs")
        return

    palette = {
        "Within DFA state": "#4c72b0",
        "Same input char": "#55a868",
        "Between DFA states": "#dd8452",
    }
    by_label = {
        "Within DFA state": within,
        "Same input char": same_input,
        "Between DFA states": between,
    }
    order = [
        label for label in ("Within DFA state", "Same input char", "Between DFA states")
        if len(by_label[label]) > 0
    ]
    specs = [(label, by_label[label]) for label in order]
    stats = {
        label: (float(np.mean(vals)), float(np.std(vals)), len(vals))
        for label, vals in specs
    }

    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    x = np.arange(len(order))
    rng = np.random.default_rng(0)
    max_points = 200
    for i, label in enumerate(order):
        vals = np.asarray(by_label[label], dtype=float)
        if len(vals) > max_points:
            idx = rng.choice(len(vals), size=max_points, replace=False)
            vals = vals[idx]
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        color = palette[label]
        ax.scatter(
            x[i] + jitter,
            vals,
            c=color,
            alpha=0.35,
            s=14,
            linewidths=0,
            zorder=1,
        )

        mean, std, _ = stats[label]
        err_lo = min(mean, std)
        ax.errorbar(
            x[i],
            mean,
            yerr=np.array([[err_lo], [std]]),
            fmt="D",
            color=color,
            ecolor=color,
            elinewidth=2.0,
            capsize=10,
            capthick=2.0,
            markersize=9,
            markerfacecolor="white",
            markeredgecolor="0.15",
            markeredgewidth=1.5,
            zorder=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_xlabel("")
    ax.set_ylabel("Euclidean distance ||h_i − h_j||")
    n_pairs = n * (n - 1) // 2
    ax.set_title(f"Pairwise hidden-state distance ({n_pairs} pairs, n={n} timesteps)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(bottom=0)
    within_mean = stats.get("Within DFA state", (float("nan"),))[0]
    between_mean = stats.get("Between DFA states", (float("nan"),))[0]
    ratio = within_mean / between_mean if between_mean > 0 else float("inf")
    parts = [
        f"{label}: n={n_} mean={m:.4f} std={s:.4f}"
        for label, (m, s, n_) in stats.items()
    ]
    print("pairwise L2: " + " | ".join(parts) + f" | ratio within/between={ratio:.3f}")

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def fit_pca_2d(points):
    """PCA fit: return 2D coords, mean, and (2, D) principal axes for reconstruction."""
    mean = np.mean(points, axis=0)
    centered = points - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:2]
    coords = centered @ components.T
    return coords, mean, components


def fit_pca_2d_with_evr(points):
    """PCA fit + explained variance ratio for PC1/PC2."""
    mean = np.mean(points, axis=0)
    centered = points - mean
    _, s, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:2]
    coords = centered @ components.T
    denom = float(np.sum(s * s)) if len(s) else 1.0
    evr = (s[:2] * s[:2]) / denom if denom > 0 else np.array([0.0, 0.0])
    return coords, mean, components, evr


def pca_2d(points):
    """Project points to two dimensions with PCA using NumPy's SVD."""
    return fit_pca_2d(points)[0]


def reconstruct_from_pca(coords, mean, components):
    """Approximate hidden states from PC1/PC2 (other PCs set to zero)."""
    return mean + coords @ components


def argmax_next_char(model, hidden_states):
    """Most likely next character index for each row of hidden_states."""
    return np.argmax(next_char_probabilities(model, hidden_states), axis=1)


def next_char_probabilities(model, hidden_states):
    """Softmax next-char distribution for each row of hidden_states."""
    weights = model["weights_hidden_to_output"]
    bias = model["bias_output"].ravel()
    logits = hidden_states @ weights.T + bias
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def prediction_entropy(probs):
    """Shannon entropy (nats) of each row of a probability matrix."""
    p = np.clip(probs, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def build_pca_plane_grid(text, hidden_states, grid_resolution=120):
    """PCA mesh and 2D-reconstructed hidden states on a grid covering data + labels."""
    projected, mean, components = fit_pca_2d(hidden_states)
    x_min, x_max = projected[:, 0].min(), projected[:, 0].max()
    y_min, y_max = projected[:, 1].min(), projected[:, 1].max()
    x_pad = max((x_max - x_min) * 0.12, 1e-3)
    y_pad = max((y_max - y_min) * 0.12, 1e-3)
    if text:
        _, _, _, label_positions = layout_trigram_labels(text, projected)
        if label_positions:
            text_positions = np.array(list(label_positions.values()))
            x_min = min(x_min, text_positions[:, 0].min())
            x_max = max(x_max, text_positions[:, 0].max())
            y_min = min(y_min, text_positions[:, 1].min())
            y_max = max(y_max, text_positions[:, 1].max())
            x_pad = max(x_pad, (x_max - x_min) * 0.08)
            y_pad = max(y_pad, (y_max - y_min) * 0.08)
    xlim = (x_min - x_pad, x_max + x_pad)
    ylim = (y_min - y_pad, y_max + y_pad)

    xs = np.linspace(xlim[0], xlim[1], grid_resolution)
    ys = np.linspace(ylim[0], ylim[1], grid_resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_coords = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    grid_hidden = reconstruct_from_pca(grid_coords, mean, components)
    return grid_x, grid_y, grid_hidden, projected, xlim, ylim


def _rolling_median(y: np.ndarray, win: int) -> np.ndarray:
    """Centered rolling median; edges use available samples only."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0 or win <= 1:
        return y
    out = np.empty(n)
    half = win // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = float(np.median(y[lo:hi]))
    return out


def plot_learning_curve(model, save_path):
    """Per-window CE (rolling median) and stochastic word-validity metric."""
    if "loss_iterations" not in model:
        print(f"skip {save_path}: re-run min-char-rnn.py to record loss history")
        return

    iters = np.asarray(model["loss_iterations"], dtype=int)
    window = np.asarray(model["loss_window"], dtype=float)
    ce_plot = _rolling_median(window, 51)

    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    ax.plot(iters, ce_plot, color="steelblue", linewidth=1.2, label="CE (51-iter median)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("cross-entropy (sum over BPTT window)")
    ax.set_title("Training: cross-entropy vs word-validity rollout")
    ax.grid(True, linestyle=":", alpha=0.4)
    finite = ce_plot[np.isfinite(ce_plot)]
    if finite.size:
        hi = float(np.percentile(finite, 99.5))
        ax.set_ylim(0, max(hi * 1.05, 1.0))

    if "metric_iterations" in model and "metric_word_error_frac" in model:
        ax2 = ax.twinx()
        ax2.plot(
            model["metric_iterations"],
            100.0 * np.asarray(model["metric_word_error_frac"], dtype=float),
            color="darkorange",
            linewidth=1.2,
            alpha=0.9,
        )
        ax2.set_ylabel("% invalid words (mean stochastic rollout)")
        ax2.set_ylim(0, 100)
        ax.legend(loc="upper right", fontsize=8)
    elif "metric_iterations" in model and "metric_valid_vocab_letter_frac" in model:
        ax2 = ax.twinx()
        ax2.plot(
            model["metric_iterations"],
            100.0 * (1.0 - model["metric_valid_vocab_letter_frac"]),
            color="darkorange",
            linewidth=1.2,
            alpha=0.9,
        )
        ax2.set_ylabel("% letters out of vocab (rollout)")
        ax2.set_ylim(0, 100)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"wrote {save_path}")


def _token_letter_valid_mask(text: str, vocab: set[str]) -> list[bool]:
    """Per-character mask: True if char is in a token that is exactly in vocab."""
    mask = [False] * len(text)
    if not vocab or not text:
        return mask
    i = 0
    n = len(text)
    while i < n:
        if text[i] == " ":
            i += 1
            continue
        j = i
        while j < n and text[j] != " ":
            j += 1
        token = text[i:j]
        ok = token in vocab
        for k in range(i, j):
            mask[k] = ok
        i = j
    return mask


SAMPLE_DISPLAY_WORDS = 15


def _vocab_word_tokens(text: str, vocab: set[str], max_words: int) -> tuple[list[str], int]:
    """Whitespace tokens that are whole in-vocabulary words."""
    all_vocab = [t for t in text.split() if t in vocab]
    return all_vocab[:max_words], len(all_vocab)


def plot_sample_before_after(model, save_path: str) -> None:
    """Stochastic samples before/after training; first N words on one line."""
    if "sample_before" not in model or "sample_after" not in model:
        print(f"skip {save_path}: re-run min-char-rnn.py to record samples")
        return

    vocab = set(map(str, model.get("vocab_words", [])))
    demo_prompt = str(model.get("demo_prompt", ""))
    demo_target = str(model.get("demo_target", ""))
    demo_before = str(model.get("demo_before", "")) or str(model["sample_before"])
    demo_after = str(model.get("demo_after", "")) or str(model["sample_after"])
    training_demo = (demo_prompt + demo_target).strip()

    training_tokens, training_n = _vocab_word_tokens(training_demo, vocab, SAMPLE_DISPLAY_WORDS)

    after_err = model.get("demo_word_error_frac")
    if after_err is None or not np.isfinite(after_err):
        after_err = _invalid_word_fraction(demo_after, vocab)
    after_title = (
        f"Generated after learning — {100.0 * after_err:.1f}% invalid words (full rollout)"
    )
    if "metric_word_error_frac" in model and len(model["metric_word_error_frac"]):
        train_err = float(model["metric_word_error_frac"][-1])
        after_title += f"; training metric: {100.0 * train_err:.1f}%"

    rows = [
        ("Demo snippet from training sequence", training_tokens, None, training_n),
        ("Generated before learning — green=in vocab, red=not in vocab", demo_before, vocab, None),
        (after_title + " — green=in vocab, red=not in vocab", demo_after, vocab, None),
    ]

    fig, axes = plt.subplots(len(rows), 1, figsize=(14, 4.2), constrained_layout=True)
    for ax, (title, text_or_tokens, word_vocab, n_vocab_words) in zip(axes, rows):
        ax.set_axis_off()
        if isinstance(text_or_tokens, list):
            tokens = text_or_tokens
            n_total = n_vocab_words if n_vocab_words is not None else len(tokens)
        else:
            tokens = text_or_tokens.split()[:SAMPLE_DISPLAY_WORDS]
            n_total = len(text_or_tokens.split())
        suffix = (
            f" (first {SAMPLE_DISPLAY_WORDS} of {n_total} words)"
            if n_total > SAMPLE_DISPLAY_WORDS
            else ""
        )
        ax.text(0.0, 0.92, title + suffix, transform=ax.transAxes, fontsize=10, va="top")

        if not tokens:
            continue
        if word_vocab is None:
            ax.text(
                0.0, 0.35, "   ".join(tokens),
                transform=ax.transAxes,
                fontfamily="monospace",
                fontsize=11,
                color="0.15",
                va="center",
                ha="left",
            )
        else:
            shown = [t if len(t) <= 10 else t[:9] + "…" for t in tokens]
            line = "   ".join(shown)
            if all(t not in word_vocab for t in tokens):
                ax.text(
                    0.0, 0.35, line,
                    transform=ax.transAxes,
                    fontfamily="monospace",
                    fontsize=10,
                    color="#d62728",
                    va="center",
                    ha="left",
                )
            else:
                n = len(shown)
                xs = np.linspace(0.0, 0.98, n) if n > 1 else np.array([0.0])
                for x, tok, raw in zip(xs, shown, tokens):
                    color = "#2ca02c" if raw in word_vocab else "#d62728"
                    ax.text(
                        x, 0.35, tok,
                        transform=ax.transAxes,
                        fontfamily="monospace",
                        fontsize=10,
                        color=color,
                        va="center",
                        ha="left" if n == 1 else "center",
                    )

    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def trigram_sequence_colors(labels):
    """Stable color per unique 3-char context label (tab10, full saturation)."""
    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab10", max(len(unique_labels), 1))
    return {label: cmap(i) for i, label in enumerate(unique_labels)}


def _state_id_colors(state_ids: list[int]) -> dict[int, tuple]:
    unique = sorted(set(state_ids))
    cmap = plt.get_cmap("tab20", max(len(unique), 1))
    return {state: cmap(i) for i, state in enumerate(unique)}


def prefix_annotation_label(text: str, index: int, *, spaced: bool) -> str:
    """Text on annotation boxes: in-word prefix since last space, or stream prefix."""
    if spaced:
        return word_subsequent_label(text, index)
    return text[: index + 1]


def _layout_group_label_positions(
    projected, groups: dict[str, list[int]]
) -> dict[str, np.ndarray]:
    center = projected.mean(axis=0)
    span = max(
        float(np.ptp(projected[:, 0])),
        float(np.ptp(projected[:, 1])),
        1e-3,
    )
    label_offset = span * 0.14
    label_positions = {}

    def _rot(v: np.ndarray, deg: float) -> np.ndarray:
        t = np.deg2rad(deg)
        c, s = float(np.cos(t)), float(np.sin(t))
        return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])

    # Try a small set of angles so leader labels don't overlap each other.
    # (We approximate overlap using label center distance in data units.)
    angle_candidates_deg = [0, 18, -18, 36, -36, 54, -54, 72, -72, 90, -90, 120, -120, 150, -150, 180]
    min_sep = label_offset * 0.65

    # Place "harder" groups first (those nearer the center tend to collide more).
    items = list(groups.items())
    items.sort(key=lambda kv: float(np.linalg.norm(projected[kv[1]].mean(axis=0) - center)))

    for key, indices in items:
        points = projected[indices]
        centroid = points.mean(axis=0)
        outward = centroid - center
        norm = float(np.linalg.norm(outward))
        if norm < 1e-9:
            outward = np.array([0.0, 1.0])
        else:
            outward = outward / norm

        best = centroid + outward * label_offset
        for deg in angle_candidates_deg:
            cand = centroid + _rot(outward, deg) * label_offset
            if not label_positions:
                best = cand
                break
            if all(float(np.linalg.norm(cand - p)) >= min_sep for p in label_positions.values()):
                best = cand
                break
        label_positions[key] = best
    return label_positions


def layout_trigram_labels(text, projected, *, spaced: bool = False):
    """Label positions and grouping for context annotations on PCA plots."""
    labels = [timestep_context_label(text, i, spaced=spaced) for i in range(len(text))]
    sequence_color = trigram_sequence_colors(labels)
    by_sequence: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        by_sequence[label].append(i)
    label_positions = _layout_group_label_positions(projected, by_sequence)
    return labels, sequence_color, by_sequence, label_positions


CONTEXT_LABEL_FONTSIZE = 9


def _context_annotation_bbox(edge_color: str) -> dict:
    """Opaque label box readable on top of colored contour regions."""
    return dict(
        boxstyle="round,pad=0.22",
        facecolor="#ffffff",
        edgecolor=edge_color,
        linewidth=1.0,
        alpha=1.0,
    )


def _context_annotation_effects():
    """Thin outline so small labels stay legible without looking heavy."""
    return [
        path_effects.withStroke(linewidth=2.5, foreground="#ffffff"),
        path_effects.Normal(),
    ]


def _draw_annotation_groups(
    ax,
    projected,
    groups: dict,
    label_positions: dict,
    point_colors: list,
    label_text: dict,
    *,
    point_size: float = 40,
    label_fontsize: float = CONTEXT_LABEL_FONTSIZE,
    leader_linewidth: float = 1.4,
) -> list[tuple[float, float]]:
    """Scatter + leader lines + one label per group (shared by trigram / DFA modes)."""
    ax.scatter(
        projected[:, 0], projected[:, 1],
        s=point_size, c=point_colors, edgecolors="black", linewidths=0.5,
        zorder=6,
    )

    for key, indices in groups.items():
        text_pos = label_positions[key]
        color = point_colors[indices[0]]
        for point in projected[indices]:
            ax.plot(
                [text_pos[0], point[0]], [text_pos[1], point[1]],
                color=color, linewidth=leader_linewidth, solid_capstyle="round", zorder=5,
            )
        ax.text(
            text_pos[0], text_pos[1], label_text[key],
            fontsize=label_fontsize, color="#1a1a1a",
            ha="center", va="center",
            bbox=_context_annotation_bbox(color),
            path_effects=_context_annotation_effects(),
            zorder=10,
        )

    return list(label_positions.values())


def _add_dfa_state_color_legend(
    ax, automaton: MinimizedVocabAutomaton, state_colors: dict[int, tuple]
) -> None:
    handles = [
        Patch(
            facecolor=state_colors[state],
            edgecolor="#333333",
            label=dfa_state_label(state, automaton),
        )
        for state in sorted(state_colors)
    ]
    ax.legend(
        handles=handles,
        title="min DFA state",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=7,
        title_fontsize=8,
        framealpha=0.95,
        borderaxespad=0.0,
    )


def add_dfa_state_annotations(
    ax,
    text,
    projected,
    automaton: MinimizedVocabAutomaton,
    *,
    spaced: bool,
    state_colors: dict[int, tuple] | None = None,
    show_legend: bool = False,
    point_size: float = 40,
    label_fontsize: float = CONTEXT_LABEL_FONTSIZE,
    leader_linewidth: float = 1.4,
    annot_style: str = "leaders",
):
    """Point color = min DFA state; annotation text = prefix since last space."""
    n = len(text)
    state_ids = [
        dfa_state_at_position(text, i, automaton, spaced=spaced) for i in range(n)
    ]
    prefixes = [
        prefix_annotation_label(text, i, spaced=spaced) for i in range(n)
    ]
    if state_colors is None:
        state_colors = _state_id_colors(state_ids)
    point_colors = [state_colors[s] for s in state_ids]

    by_prefix: dict[str, list[int]] = defaultdict(list)
    for i, prefix in enumerate(prefixes):
        by_prefix[prefix].append(i)
    label_positions = _layout_group_label_positions(projected, by_prefix)
    label_text = {p: ("␣" if p == " " else p) for p in by_prefix}

    annot_style = (annot_style or "leaders").lower()
    if annot_style == "none":
        ax.scatter(
            projected[:, 0],
            projected[:, 1],
            c=point_colors,
            s=point_size,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.92,
            zorder=4,
        )
        text_positions = []
    elif annot_style == "annots_only":
        # Put the prefix labels directly at each point (no box, no leader lines).
        fs = max(8, int(label_fontsize * 0.65))
        for i, prefix in enumerate(prefixes):
            label = "␣" if prefix == " " else prefix
            ax.text(
                projected[i, 0],
                projected[i, 1],
                label,
                fontsize=fs,
                color=point_colors[i],
                ha="center",
                va="center",
                zorder=10,
            )
        text_positions = projected.tolist()
    else:
        text_positions = _draw_annotation_groups(
            ax, projected, by_prefix, label_positions, point_colors, label_text,
            point_size=point_size,
            label_fontsize=label_fontsize,
            leader_linewidth=leader_linewidth,
        )
    if show_legend:
        _add_dfa_state_color_legend(ax, automaton, state_colors)
    return text_positions


def add_trigram_annotations(ax, text, projected, *, spaced: bool = False):
    """Context-colored points, leader lines, one label per context group."""
    labels, sequence_color, by_sequence, label_positions = layout_trigram_labels(
        text, projected, spaced=spaced
    )
    label_text = {"␣" if label == " " else label for label in by_sequence}
    point_colors = [sequence_color[label] for label in labels]
    return _draw_annotation_groups(
        ax, projected, by_sequence, label_positions, point_colors, label_text
    )


def add_pca_point_annotations(
    ax,
    text,
    projected,
    *,
    spaced: bool = False,
    automaton: MinimizedVocabAutomaton | None = None,
    show_dfa_legend: bool = False,
    annot_style: str = "leaders",
):
    if automaton is not None:
        return add_dfa_state_annotations(
            ax,
            text,
            projected,
            automaton,
            spaced=spaced,
            show_legend=show_dfa_legend,
            annot_style=annot_style,
        )
    return add_trigram_annotations(ax, text, projected, spaced=spaced)


def _expand_limits_for_annotations(ax, projected, text_positions, base_xlim, base_ylim):
    """Union of grid limits and annotation positions."""
    all_x = [projected[:, 0].min(), projected[:, 0].max(), base_xlim[0], base_xlim[1]]
    all_y = [projected[:, 1].min(), projected[:, 1].max(), base_ylim[0], base_ylim[1]]
    if text_positions:
        all_x.extend(p[0] for p in text_positions)
        all_y.extend(p[1] for p in text_positions)
    x_pad = max((max(all_x) - min(all_x)) * 0.1, 1e-3)
    y_pad = max((max(all_y) - min(all_y)) * 0.1, 1e-3)
    ax.set_xlim(min(all_x) - x_pad, max(all_x) + x_pad)
    ax.set_ylim(min(all_y) - y_pad, max(all_y) + y_pad)


def plot_2d_hidden_state_labels(
    text,
    hidden_states,
    chars,
    projected,
    save_path,
    title,
    xlabel,
    ylabel,
    fig_suptitle=None,
    *,
    spaced: bool = False,
    automaton: MinimizedVocabAutomaton | None = None,
):
    """Scatter points with one context label per group, lines to its points."""
    _ = chars
    if len(text) == 0:
        return

    fig, ax = plt.subplots(figsize=(14, 11), constrained_layout=True)
    text_positions = add_pca_point_annotations(
        ax, text, projected, spaced=spaced, automaton=automaton
    )
    _expand_limits_for_annotations(
        ax, projected, text_positions,
        (projected[:, 0].min(), projected[:, 0].max()),
        (projected[:, 1].min(), projected[:, 1].max()),
    )

    ax.axhline(0, color="lightgrey", linewidth=0.6, zorder=0)
    ax.axvline(0, color="lightgrey", linewidth=0.6, zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.35)
    if fig_suptitle:
        fig.suptitle(fig_suptitle, fontsize=11, y=1.02)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"wrote {save_path}")


def _plot_2d_hidden_state_labels_on_ax(
    ax,
    text: str,
    projected: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    spaced: bool = False,
    automaton: MinimizedVocabAutomaton | None = None,
    annot_style: str = "leaders",
) -> None:
    if len(text) == 0:
        return
    text_positions = add_pca_point_annotations(
        ax,
        text,
        projected,
        spaced=spaced,
        automaton=automaton,
        annot_style=annot_style,
    )
    _expand_limits_for_annotations(
        ax, projected, text_positions,
        (projected[:, 0].min(), projected[:, 0].max()),
        (projected[:, 1].min(), projected[:, 1].max()),
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.35)


def plot_dimred_context_panels(
    text: str,
    hidden_states: np.ndarray,
    chars,
    save_path: str,
    *,
    spaced: bool = False,
    automaton: MinimizedVocabAutomaton | None = None,
    annot_style: str = "leaders",
) -> None:
    """Compare multiple 2D embeddings with the same annotation/coloring scheme."""
    _ = chars
    n = len(text)
    if n < 1:
        return

    # Lazy imports: these are optional in some environments.
    from sklearn.manifold import TSNE, Isomap  # type: ignore
    from umap import UMAP  # type: ignore

    # --- Embeddings -----------------------------------------------------------
    pca_xy, _, _, evr = fit_pca_2d_with_evr(hidden_states)
    pc1 = 100.0 * float(evr[0]) if len(evr) > 0 else 0.0
    pc2 = 100.0 * float(evr[1]) if len(evr) > 1 else 0.0

    umap_xy = UMAP(
        n_components=2,
        n_neighbors=min(15, max(2, n - 1)),
        min_dist=0.1,
        random_state=0,
    ).fit_transform(hidden_states)

    perplexity = min(30, max(5, (n - 1) // 3))
    tsne_xy = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=0,
    ).fit_transform(hidden_states)

    isomap_xy = Isomap(
        n_components=2,
        n_neighbors=min(10, max(2, n - 1)),
    ).fit_transform(hidden_states)

    panels = [
        ("PCA", pca_xy, f"PC1 ({pc1:.1f}%)", f"PC2 ({pc2:.1f}%)",
         f"PCA\nvariance explained: PC1 {pc1:.1f}%, PC2 {pc2:.1f}%"),
        ("UMAP", umap_xy, "UMAP-1", "UMAP-2", "UMAP"),
        ("t-SNE", tsne_xy, "t-SNE-1", "t-SNE-2", f"t-SNE (perplexity={perplexity})"),
        ("Isomap", isomap_xy, "Isomap-1", "Isomap-2", "Isomap"),
    ]

    ctx = "prefix since last space" if spaced else "stream prefix"
    if automaton is not None:
        scheme = f"min DFA state · {ctx}"
    else:
        scheme = "prefix after space" if spaced else "prev2+current (3-char)"

    fig, axes = plt.subplots(2, 2, figsize=(18, 14), constrained_layout=True)
    axes = axes.ravel()
    for ax, (_, xy, xlabel, ylabel, title) in zip(axes, panels):
        _plot_2d_hidden_state_labels_on_ax(
            ax,
            text,
            xy,
            title=f"{title}\n({scheme})",
            xlabel=xlabel,
            ylabel=ylabel,
            spaced=spaced,
            automaton=automaton,
            annot_style=annot_style,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(top=False, right=False)

    fig.suptitle(original_vocabulary_title(chars, text), fontsize=12, y=1.01)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def plot_per_char_hidden_state_heatmaps(
    text, hidden_states, chars, save_path, cluster_rows=True, *, spaced: bool = False
):
    """Combined per-input-char heatmaps, rows = hidden units, columns = occurrences."""
    hidden_size = hidden_states.shape[1]
    groups = []

    for char in chars:
        indices = np.array([i for i, text_char in enumerate(text) if i > 0 and text_char == char])
        if len(indices) == 0:
            continue

        rows = hidden_states[indices]
        labels = [context_label(text, int(i), spaced=spaced) for i in indices]

        if cluster_rows and len(indices) > 2:
            order = average_linkage_cluster_order(rows)
            rows = rows[order]
            labels = [labels[i] for i in order]
            title_suffix = "clustered by hidden-state similarity"
        else:
            title_suffix = "in sequence order"

        groups.append((char, rows, labels, title_suffix))

    if not groups:
        return

    fig, axes = plt.subplots(
        len(groups), 1,
        figsize=(max(12, max(len(labels) for _, _, labels, _ in groups) * 0.28),
                 max(3, len(groups) * max(2.1, hidden_size * 0.24))),
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    last_image = None

    for ax, (char, rows, labels, title_suffix) in zip(axes, groups):
        im = ax.imshow(
            rows.T,
            aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
            interpolation="nearest", origin="lower",
        )
        last_image = im

        ax.set_yticks(range(hidden_size))
        ax.set_yticklabels([f"h{i}" for i in range(hidden_size)])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=6, rotation=90)
        ax.set_ylabel("hidden unit")
        ax.set_title(
            f"Hidden states for input {display_char(char)!r} "
            f"({len(labels)} occurrences, {title_suffix})"
        )

    axes[-1].set_xlabel(
        "prefix after space (or ' ') @ timestep"
        if spaced
        else "previous + current character @ timestep"
    )
    fig.suptitle(
        f"Hidden states by input character · {original_vocabulary_title(chars, text)}",
        y=0.995,
    )
    fig.colorbar(last_image, ax=axes, fraction=0.015, pad=0.01, label="activation")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"wrote {save_path}")


def trigram_avoidance_points(
    text,
    projected,
    *,
    spaced: bool = False,
    automaton: MinimizedVocabAutomaton | None = None,
):
    """PC coordinates to keep region letters away from (scatter + label boxes)."""
    if automaton is not None:
        prefixes = [prefix_annotation_label(text, i, spaced=spaced) for i in range(len(text))]
        by_prefix: dict[str, list[int]] = defaultdict(list)
        for i, prefix in enumerate(prefixes):
            by_prefix[prefix].append(i)
        label_positions = _layout_group_label_positions(projected, by_prefix)
    else:
        _, _, _, label_positions = layout_trigram_labels(text, projected, spaced=spaced)
    blocks = [projected]
    if label_positions:
        blocks.append(np.array(list(label_positions.values())))
    return np.vstack(blocks)


def region_interior_point(
    grid_x, grid_y, class_mask,
    avoid_xy=None, avoid_radius=0.0,
    xlim=None, ylim=None, edge_margin_frac=0.08,
    min_area_frac=0.02, erosion_iters=4,
):
    """Point deep inside the largest argmax blob, away from labels and plot edges."""
    if not class_mask.any():
        return None

    labeled, num_features = ndimage.label(class_mask)
    if num_features == 0:
        return None

    component_sizes = ndimage.sum(
        class_mask, labeled, index=np.arange(1, num_features + 1),
    )
    largest_label = 1 + int(np.argmax(component_sizes))
    component = labeled == largest_label

    if component.sum() < min_area_frac * class_mask.size:
        return None

    interior = component
    for _ in range(erosion_iters):
        shrunk = ndimage.binary_erosion(interior)
        if shrunk.any():
            interior = shrunk

    depth = ndimage.distance_transform_edt(interior)
    if depth.max() < 1.0:
        return None

    rows, cols = np.where(interior)
    depth_vals = depth[rows, cols]
    gx = grid_x[rows, cols]
    gy = grid_y[rows, cols]

    if avoid_xy is not None and len(avoid_xy):
        avoid = np.asarray(avoid_xy, dtype=float)
        diff = np.stack([gx, gy], axis=1)[:, None, :] - avoid[None, :, :]
        clearance = np.linalg.norm(diff, axis=2).min(axis=1) - avoid_radius
        clearance = np.maximum(clearance, 0.0)
    else:
        clearance = np.ones(len(rows), dtype=float)

    if xlim is not None and ylim is not None:
        plane_span = max(float(xlim[1] - xlim[0]), float(ylim[1] - ylim[0]), 1e-3)
        margin = plane_span * edge_margin_frac
        edge_clear = np.minimum(
            np.minimum(gx - xlim[0], xlim[1] - gx),
            np.minimum(gy - ylim[0], ylim[1] - gy),
        ) - margin
        edge_clear = np.maximum(edge_clear, 0.0)
    else:
        edge_clear = np.ones(len(rows), dtype=float)

    score = depth_vals * np.sqrt(clearance + 1e-6) * edge_clear
    if score.max() <= 0:
        return None

    best = int(np.argmax(score))
    row, col = rows[best], cols[best]
    return float(grid_x[row, col]), float(grid_y[row, col])


def add_argmax_region_labels(
    ax, grid_x, grid_y, grid_pred, chars,
    avoid_xy=None, avoid_radius=0.0, xlim=None, ylim=None,
):
    """Large white letter at the interior of each argmax region."""
    stroke = path_effects.withStroke(linewidth=2.5, foreground="#1a1a1a")
    for index, char in enumerate(chars):
        mask = grid_pred == index
        if not mask.any():
            continue
        position = region_interior_point(
            grid_x, grid_y, mask,
            avoid_xy=avoid_xy, avoid_radius=avoid_radius,
            xlim=xlim, ylim=ylim,
        )
        if position is None:
            continue
        glyph = argmax_region_glyph(char)
        if glyph is None:
            continue
        ax.text(
            position[0], position[1], glyph,
            fontsize=26, color="white",
            ha="center", va="center", zorder=8,
            path_effects=[
                path_effects.withStroke(linewidth=3.5, foreground="#ffffff"),
                stroke,
            ],
        )


def plot_pca_context_labels(
    text,
    hidden_states,
    chars,
    save_path,
    *,
    spaced: bool = False,
    automaton: MinimizedVocabAutomaton | None = None,
):
    """4-panel dim-reduction comparison with shared annotations/colors."""
    plot_dimred_context_panels(
        text,
        hidden_states,
        chars,
        save_path,
        spaced=spaced,
        automaton=automaton,
        annot_style="leaders",
    )


def _word_trajectory_colors(segments: list[tuple[int, int, str]]) -> dict[str, tuple]:
    """Stable color per distinct word label across space-to-space segments."""
    words = sorted({segment_word_label(seg) for _, _, seg in segments})
    cmap = plt.get_cmap("tab20", max(len(words), 1))
    return {word: cmap(i) for i, word in enumerate(words)}


def _square_data_limits(*xy_arrays: np.ndarray, padding_frac: float = 0.12):
    """Square x/y limits from trajectory data (ignore annotation label offsets)."""
    xs: list[float] = []
    ys: list[float] = []
    for arr in xy_arrays:
        if arr is None or len(arr) == 0:
            continue
        xs.extend([float(arr[:, 0].min()), float(arr[:, 0].max())])
        ys.extend([float(arr[:, 1].min()), float(arr[:, 1].max())])
    if not xs:
        return (-1.0, 1.0), (-1.0, 1.0)
    cx = 0.5 * (min(xs) + max(xs))
    cy = 0.5 * (min(ys) + max(ys))
    half = 0.5 * max(max(xs) - min(xs), max(ys) - min(ys), 1e-3)
    half *= 1.0 + padding_frac
    return (cx - half, cx + half), (cy - half, cy + half)


def plot_space_to_space_trajectories(
    text: str,
    hidden_states: np.ndarray,
    save_path: str,
    *,
    model=None,
    free_rollout_steps: int = 10,
    closed_loop_steps: int | None = None,
    closed_loop_seed: int = 0,
    spaced: bool = False,
    automaton: MinimizedVocabAutomaton | None = None,
    annot_style: str = "leaders",
):
    """PCA plot of every hidden-state path from one space timestep to the next.

    If `model` is provided, draw the no-input recurrent vector field in PCA
    as a faint background quiver grid.
    """
    segments = space_to_space_segments(text)
    if len(text) < 2 or not segments:
        return

    projected, mean, components, evr = fit_pca_2d_with_evr(hidden_states)
    pc1 = 100.0 * float(evr[0]) if len(evr) > 0 else 0.0
    pc2 = 100.0 * float(evr[1]) if len(evr) > 1 else 0.0
    xlabel = f"PC1 ({pc1:.1f}%)"
    ylabel = f"PC2 ({pc2:.1f}%)"
    word_colors = _word_trajectory_colors(segments)

    if closed_loop_steps is None:
        closed_loop_steps = len(text)

    ncols = 3 if model is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(22 if ncols == 2 else 30, 11), constrained_layout=True)
    axes = np.atleast_1d(axes)
    # Panel order: internal (no input), trained (observed), closed-loop (self-fed).
    ax_free = axes[0] if ncols >= 2 else None
    ax_paths = axes[1] if ncols >= 2 else axes[0]
    ax_gen = axes[2] if ncols >= 3 else None

    rollout_paths: list[np.ndarray] = []
    gen_z = projected

    # Panel: trained (observed) trajectories colored by true word segment.
    for start, end, segment_text in segments:
        path = projected[start : end + 1]
        if len(path) == 0:
            continue
        color = word_colors[segment_word_label(segment_text)]

        ax_paths.plot(
            path[:, 0], path[:, 1],
            color=color, linewidth=1.6, alpha=0.55, solid_capstyle="round", zorder=2,
        )

    # Panel 2: free dynamics rollouts from each observed hidden state (no input).
    if ax_free is not None and model is not None and free_rollout_steps > 0:
        W_hh = np.asarray(model["weights_hidden_to_hidden"])
        b_h = np.asarray(model["bias_hidden"]).ravel()

        # Map each timestep to its containing word label for coloring.
        word_at_t = [""] * len(text)
        for start, end, segment_text in segments:
            word = segment_word_label(segment_text)
            for t in range(start, end + 1):
                if 0 <= t < len(word_at_t):
                    word_at_t[t] = word

        use_relu = bool(model.get("use_relu", False))
        for t, h0 in enumerate(hidden_states):
            h = np.asarray(h0, dtype=float)
            zs = [projected[t]]  # start exactly at the observed point
            for _ in range(int(free_rollout_steps)):
                h = no_input_hidden_step(h, W_hh, b_h, use_relu=use_relu)
                z = (h - mean) @ components.T
                zs.append(z)
            zs = np.asarray(zs, dtype=float)
            if zs.shape[0] < 2:
                continue
            rollout_paths.append(zs)
            color = word_colors.get(word_at_t[t], "0.15")
            ax_free.plot(zs[:, 0], zs[:, 1], color=color, linewidth=1.0, alpha=0.22, zorder=3)

    # Panel 3: closed-loop generation (sampled; previous output fed back as input).
    if ax_gen is not None and model is not None and closed_loop_steps > 1:
        rng = np.random.default_rng(int(closed_loop_seed))
        chars = list(model["chars"])
        char_to_index = {c: i for i, c in enumerate(chars)}
        vocab_size = len(chars)

        W_xh = np.asarray(model["weights_input_to_hidden"])
        W_hh = np.asarray(model["weights_hidden_to_hidden"])
        W_ho = np.asarray(model["weights_hidden_to_output"])
        b_h = np.asarray(model["bias_hidden"]).ravel()
        b_o = np.asarray(model["bias_output"]).ravel()

        # Seed with the first character of the observed window (keeps vocab consistent).
        seed_char = text[0] if text else chars[0]
        if seed_char not in char_to_index:
            seed_char = chars[0]

        h = np.zeros((hidden_states.shape[1], 1), dtype=float)
        generated = [seed_char]
        gen_h = []
        use_relu = bool(model.get("use_relu", False))
        b_h_col = np.asarray(model["bias_hidden"])

        prev_char = seed_char
        for _ in range(int(closed_loop_steps)):
            x = np.zeros((vocab_size, 1), dtype=float)
            x[char_to_index[prev_char], 0] = 1.0
            h, _ = rnn_hidden_step(
                h, x, W_xh, W_hh, b_h_col, use_relu=use_relu,
            )
            gen_h.append(h.ravel().copy())

            logits = W_ho @ h.ravel() + b_o
            logits = logits - np.max(logits)
            probs = np.exp(logits)
            probs = probs / np.sum(probs)
            next_ix = int(rng.choice(vocab_size, p=probs))
            next_char = chars[next_ix]
            generated.append(next_char)
            prev_char = next_char

        gen_h = np.asarray(gen_h, dtype=float)
        gen_z = (gen_h - mean) @ components.T

        # Break generated characters into word segments between spaces and color by word
        gen_text = "".join(generated[: len(gen_z)])
        gen_segments = space_to_space_segments(gen_text)
        if not gen_segments:
            gen_segments = [(0, len(gen_text) - 1, gen_text)]

        for start, end, seg in gen_segments:
            if start < 0 or end < 0 or start >= len(gen_z):
                continue
            end = min(end, len(gen_z) - 1)
            path = gen_z[start : end + 1]
            if len(path) == 0:
                continue
            word = segment_word_label(seg)
            color = word_colors.get(word, (0.2, 0.2, 0.2, 1.0))

            ax_gen.plot(path[:, 0], path[:, 1], color=color, linewidth=1.3, alpha=0.40, zorder=2)
            if len(path) >= 2:
                ax_gen.scatter(
                    path[:, 0], path[:, 1],
                    s=18, c=[color] * len(path),
                    alpha=0.55, edgecolors="black", linewidths=0.25, zorder=3,
                )

    limit_arrays = [projected, gen_z]
    limit_arrays.extend(rollout_paths)
    xlim, ylim = _square_data_limits(*limit_arrays)

    if model is not None:
        grid_resolution = 26
        xs = np.linspace(xlim[0], xlim[1], grid_resolution)
        ys = np.linspace(ylim[0], ylim[1], grid_resolution)
        grid_x, grid_y = np.meshgrid(xs, ys)
        z_grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])

        h = reconstruct_from_pca(z_grid, mean, components)
        W_hh = np.asarray(model["weights_hidden_to_hidden"])
        b_h = np.asarray(model["bias_hidden"]).ravel()
        use_relu = bool(model.get("use_relu", False))
        h_next = no_input_hidden_step(h, W_hh, b_h, use_relu=use_relu)
        z_next = (h_next - mean) @ components.T
        d = z_next - z_grid
        U = d[:, 0].reshape(grid_resolution, grid_resolution)
        V = d[:, 1].reshape(grid_resolution, grid_resolution)

        for ax in (a for a in (ax_paths, ax_free, ax_gen) if a is not None):
            ax.quiver(
                grid_x,
                grid_y,
                U,
                V,
                angles="xy",
                scale_units="xy",
                scale=35.0,
                width=0.0022,
                headwidth=3.6,
                headlength=4.6,
                headaxislength=3.6,
                color="#000000",
                alpha=0.18,
                zorder=1,
            )

    # Observed test-window prefix labels at their trained PCA positions on every panel.
    for ax in (a for a in (ax_paths, ax_free, ax_gen) if a is not None):
        add_pca_point_annotations(
            ax,
            text,
            projected,
            spaced=spaced,
            automaton=automaton,
            annot_style=annot_style,
        )

    handles = [
        Patch(facecolor=word_colors[w], edgecolor="#333333", label=w)
        for w in sorted(word_colors)
    ]
    ax_paths.legend(
        handles=handles,
        title="word",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=7,
        title_fontsize=8,
        framealpha=0.95,
    )

    for ax in (a for a in (ax_paths, ax_free, ax_gen) if a is not None):
        ax.axhline(0, color="lightgrey", linewidth=0.6, zorder=0)
        ax.axvline(0, color="lightgrey", linewidth=0.6, zorder=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")

    ax_paths.set_title(f"Trained (observed) trajectories (PCA)\n{len(segments)} segments, {len(text)} chars")
    if ax_free is not None:
        ax_free.set_title(
            f"Internal dynamics (no input)\n"
            f"{len(text)} start states × {free_rollout_steps} steps"
        )
    if ax_gen is not None:
        ax_gen.set_title(
            f"Closed-loop generation (sampled; self-fed)\n"
            f"{closed_loop_steps} steps (seed={closed_loop_seed})"
        )

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def plot_pca_vector_field(
    text: str,
    hidden_states: np.ndarray,
    model,
    save_path: str,
    *,
    grid_resolution: int = 26,
    stride: int = 1,
    scale: float = 35.0,
) -> None:
    """Grid vector field in PCA: z -> z' from no-input recurrent dynamics.

    We reconstruct h from each (PC1,PC2) grid point, apply one recurrent step with x=0,
    then project back to PCA to get the vector z' - z.
    """
    if len(text) < 3 or hidden_states.shape[0] < 3:
        return

    projected, mean, components, evr = fit_pca_2d_with_evr(hidden_states)
    z = projected

    x_min, x_max = float(np.min(z[:, 0])), float(np.max(z[:, 0]))
    y_min, y_max = float(np.min(z[:, 1])), float(np.max(z[:, 1]))
    x_pad = max((x_max - x_min) * 0.08, 1e-3)
    y_pad = max((y_max - y_min) * 0.08, 1e-3)
    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad

    xs = np.linspace(x_min, x_max, grid_resolution)
    ys = np.linspace(y_min, y_max, grid_resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)
    z_grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # z -> h (2D PCA reconstruction)
    h = reconstruct_from_pca(z_grid, mean, components)

    W_hh = np.asarray(model["weights_hidden_to_hidden"])
    b_h = np.asarray(model["bias_hidden"])
    use_relu = bool(model.get("use_relu", False))
    h_next = no_input_hidden_step(h, W_hh, b_h, use_relu=use_relu)

    # h' -> z' via the same PCA projection
    z_next = (h_next - mean) @ components.T
    d = z_next - z_grid

    U = d[:, 0].reshape(grid_resolution, grid_resolution)
    V = d[:, 1].reshape(grid_resolution, grid_resolution)
    mask = np.ones_like(U, dtype=bool)
    if stride > 1:
        mask[:] = False
        mask[::stride, ::stride] = True

    fig, ax = plt.subplots(figsize=(10.5, 9.0), constrained_layout=True)
    ax.scatter(
        z[:, 0],
        z[:, 1],
        s=14,
        c="0.4",
        alpha=0.22,
        edgecolors="none",
        zorder=1,
    )
    ax.quiver(
        grid_x[mask],
        grid_y[mask],
        U[mask],
        V[mask],
        angles="xy",
        scale_units="xy",
        scale=max(scale, 1e-6),
        width=0.0026,
        headwidth=4.0,
        headlength=5.0,
        headaxislength=4.0,
        color="#000000",
        alpha=0.9,
        zorder=3,
    )

    pc1 = 100.0 * float(evr[0]) if len(evr) > 0 else 0.0
    pc2 = 100.0 * float(evr[1]) if len(evr) > 1 else 0.0
    ax.set_xlabel(f"PC1 ({pc1:.1f}%)")
    ax.set_ylabel(f"PC2 ({pc2:.1f}%)")
    ax.set_title("Vector field in PCA (grid; no-input recurrent dynamics)")
    ax.axhline(0, color="lightgrey", linewidth=0.6, zorder=0)
    ax.axvline(0, color="lightgrey", linewidth=0.6, zorder=0)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, linestyle=":", alpha=0.35)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def plot_pca_dfa_analysis(
    text,
    hidden_states,
    chars,
    words: list[str],
    save_path,
    automaton: MinimizedVocabAutomaton,
    *,
    spaced: bool = False,
    annot_style: str = "leaders",
    embedding: str = "pca",
):
    """Embedding (default: PCA) beside the min-DFA with matching state colors."""
    if len(text) < 2:
        return
    embedding = (embedding or "umap").lower()
    if embedding == "pca":
        projected, _, _, evr = fit_pca_2d_with_evr(hidden_states)
        xlabel = f"PC1 ({100.0 * float(evr[0]):.1f}%)" if len(evr) > 0 else "PC1"
        ylabel = f"PC2 ({100.0 * float(evr[1]):.1f}%)" if len(evr) > 1 else "PC2"
        embed_title = "2D PCA"
        embed_subtitle = (
            f"variance explained: PC1 {100.0 * float(evr[0]):.1f}%, PC2 {100.0 * float(evr[1]):.1f}%"
            if len(evr) > 1
            else ""
        )
    else:
        # Lazy import: optional dependency.
        from umap import UMAP  # type: ignore

        n = hidden_states.shape[0]
        projected = UMAP(
            n_components=2,
            n_neighbors=min(15, max(2, n - 1)),
            min_dist=0.1,
            random_state=0,
        ).fit_transform(hidden_states)
        xlabel, ylabel = "UMAP-1", "UMAP-2"
        embed_title = "UMAP"
        embed_subtitle = ""
    state_ids = [
        dfa_state_at_position(text, i, automaton, spaced=spaced) for i in range(len(text))
    ]
    state_colors = _state_id_colors(state_ids)

    fig, axes = plt.subplots(1, 2, figsize=(28, 11), constrained_layout=True)
    ax_dfa, ax_embed = axes[0], axes[1]

    draw_minimized_dfa_on_axes(ax_dfa, automaton, words, state_colors=state_colors)
    ax_dfa.set_title("Minimal DFA", fontsize=12, pad=12)

    text_positions = add_dfa_state_annotations(
        ax_embed, text, projected, automaton,
        spaced=spaced, state_colors=state_colors,
        point_size=160,
        label_fontsize=18,
        leader_linewidth=2.8,
        annot_style=annot_style,
    )
    _expand_limits_for_annotations(
        ax_embed, projected, text_positions,
        (projected[:, 0].min(), projected[:, 0].max()),
        (projected[:, 1].min(), projected[:, 1].max()),
    )
    ax_embed.axhline(0, color="lightgrey", linewidth=0.6, zorder=0)
    ax_embed.axvline(0, color="lightgrey", linewidth=0.6, zorder=0)
    ax_embed.set_xlabel(xlabel)
    ax_embed.set_ylabel(ylabel)
    ctx = "prefix since last space" if spaced else "stream prefix"
    subtitle = f"\n{embed_subtitle}" if embed_subtitle else ""
    ax_embed.set_title(f"{embed_title} (min DFA state · {ctx}){subtitle}")
    ax_embed.grid(True, linestyle=":", alpha=0.35)
    ax_embed.spines["top"].set_visible(False)
    ax_embed.spines["right"].set_visible(False)
    ax_embed.tick_params(top=False, right=False)

    fig.suptitle(", ".join(words), fontsize=12, y=1.01)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def plot_pca_prediction_regions(
    model,
    text,
    hidden_states,
    chars,
    save_path,
    grid_resolution=120,
    *,
    spaced: bool = False,
    automaton: MinimizedVocabAutomaton | None = None,
):
    """PCA panels: argmax next-char regions and softmax entropy, with context labels."""
    n_points, hidden_size = hidden_states.shape
    vocab_size = len(chars)
    if n_points < 2 or hidden_size < 1 or len(text) == 0:
        return

    grid_x, grid_y, grid_hidden, projected, xlim, ylim = build_pca_plane_grid(
        text, hidden_states, grid_resolution,
    )
    probs = next_char_probabilities(model, grid_hidden)
    grid_pred = np.argmax(probs, axis=1).reshape(grid_resolution, grid_resolution)
    grid_entropy = prediction_entropy(probs).reshape(grid_resolution, grid_resolution)
    max_entropy = float(np.log(vocab_size))
    avoid_xy = trigram_avoidance_points(
        text, projected, spaced=spaced, automaton=automaton
    )
    plane_span = max(
        float(np.ptp(grid_x)),
        float(np.ptp(grid_y)),
        float(np.ptp(projected[:, 0])),
        float(np.ptp(projected[:, 1])),
        1e-3,
    )
    avoid_radius = plane_span * 0.12

    pred_cmap = plt.get_cmap("tab10", vocab_size)
    fig, axes = plt.subplots(1, 2, figsize=(24, 11), constrained_layout=True)
    panel_specs = [
        (
            axes[0],
            grid_pred,
            dict(
                levels=np.arange(-0.5, vocab_size, 1),
                cmap=pred_cmap,
                vmin=None,
                vmax=None,
            ),
            "Argmax next-char (2D-reconstructed h)",
        ),
        (
            axes[1],
            grid_entropy,
            dict(
                levels=20,
                cmap="magma",
                alpha=0.85,
                vmin=0.0,
                vmax=max_entropy,
            ),
            f"Prediction entropy (max = ln {vocab_size} ≈ {max_entropy:.2f} nats)",
        ),
    ]

    for ax, field, contour_kw, title in panel_specs:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        im = ax.contourf(grid_x, grid_y, field, antialiased=True, zorder=1, **contour_kw)
        if ax is axes[0]:
            add_argmax_region_labels(
                ax, grid_x, grid_y, grid_pred, chars,
                avoid_xy=avoid_xy, avoid_radius=avoid_radius,
                xlim=xlim, ylim=ylim,
            )
        add_pca_point_annotations(
            ax, text, projected, spaced=spaced, automaton=automaton
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.35)
        if ax is axes[1]:
            fig.colorbar(im, ax=ax, label="entropy (nats)", fraction=0.046, pad=0.02)

    if automaton is not None:
        pca_ctx = "min DFA state (prefix since last space)" if spaced else "min DFA state"
    else:
        pca_ctx = "prefix after space" if spaced else "prev2+current, 3-char"
    fig.suptitle(
        f"PCA plane ({pca_ctx}) · {original_vocabulary_title(chars, text)}",
        fontsize=12, y=1.01,
    )
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def plot_pca_next_char_probability_panels(
    model,
    text,
    hidden_states,
    chars,
    save_path,
    grid_resolution=120,
    *,
    spaced: bool = False,
    automaton: MinimizedVocabAutomaton | None = None,
):
    """One panel per vocab char: P(next = char) over the PCA plane (from softmax)."""
    n_points, hidden_size = hidden_states.shape
    vocab_size = len(chars)
    if n_points < 2 or hidden_size < 1:
        return

    grid_x, grid_y, grid_hidden, projected, xlim, ylim = build_pca_plane_grid(
        text, hidden_states, grid_resolution,
    )
    probs = next_char_probabilities(model, grid_hidden)

    ncols = min(3, vocab_size)
    nrows = (vocab_size + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.4 * ncols, 3.9 * nrows),
        sharex=True, sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()
    last_im = None

    for char_index, (ax, char) in enumerate(zip(axes, chars)):
        field = probs[:, char_index].reshape(grid_resolution, grid_resolution)
        last_im = ax.contourf(
            grid_x, grid_y, field,
            levels=np.linspace(0, 1, 21),
            cmap="viridis", vmin=0, vmax=1, antialiased=True,
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f"P(next = {display_char(char)!r})")
        ax.set_aspect("equal", adjustable="box")
        add_pca_point_annotations(
            ax, text, projected, spaced=spaced, automaton=automaton
        )

    for ax in axes[vocab_size:]:
        ax.axis("off")

    axes[0].set_ylabel("PC2")
    axes[(nrows - 1) * ncols].set_xlabel("PC1")
    if nrows > 1:
        for row in range(1, nrows):
            axes[row * ncols].set_ylabel("PC2")
        for col in range(1, ncols):
            bottom = min((nrows - 1) * ncols + col, vocab_size - 1)
            if bottom < vocab_size:
                axes[bottom].set_xlabel("PC1")

    fig.colorbar(last_im, ax=axes[:vocab_size], label="probability", shrink=0.92)
    fig.suptitle(
        f"P(next char | 2D-reconstructed h) over PCA · {original_vocabulary_title(chars, text)}",
        fontsize=11, y=1.02,
    )
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def char_axis_labels(chars):
    """Tick labels for the vocabulary axis (readable for whitespace)."""
    return [display_char(c) for c in chars]


def symmetric_abs_vmax(*matrices):
    return float(max(np.max(np.abs(m)) for m in matrices))


def hidden_unit_labels(dale_sign, hidden_size: int) -> list[str]:
    if dale_sign is None or len(dale_sign) != hidden_size:
        return [f"h{i}" for i in range(hidden_size)]
    return [f"h{i}({'E' if s > 0 else 'I'})" for i, s in enumerate(dale_sign)]


def ei_block_boundary(dale_sign) -> int | None:
    """Index between E and I blocks (line drawn between n_E-1 and n_E)."""
    if dale_sign is None:
        return None
    n_exc = int(np.sum(np.asarray(dale_sign) > 0))
    if 0 < n_exc < len(dale_sign):
        return n_exc
    return None


def weights_for_plot(model: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    """Return W_xh, W_hh, W_ho (and dale_sign) in E-first / I-last order."""
    from rnn_dyn import dale_signs_ordered, permute_hidden_by_dale

    W_in = np.asarray(model["weights_input_to_hidden"])
    W_rec = np.asarray(model["weights_hidden_to_hidden"])
    W_out = np.asarray(model["weights_hidden_to_output"])
    b_h = np.asarray(model["bias_hidden"])
    dale_sign = model.get("dale_sign")
    if dale_sign is not None and len(dale_sign) == W_in.shape[0]:
        dale_sign = np.asarray(dale_sign, dtype=float)
        if not dale_signs_ordered(dale_sign):
            W_in, W_rec, W_out, b_h, dale_sign = permute_hidden_by_dale(
                W_in, W_rec, W_out, b_h, dale_sign,
            )
    return W_in, W_rec, W_out, dale_sign


def _draw_ei_guides(ax, boundary: int | None, *, horizontal: bool, vertical: bool) -> None:
    if boundary is None:
        return
    if horizontal:
        ax.axhline(boundary - 0.5, color="black", lw=1.0, ls="--")
    if vertical:
        ax.axvline(boundary - 0.5, color="black", lw=1.0, ls="--")


def plot_learned_weights(model, out_dir):
    """Input (W_xh) and hidden recurrent (W_hh); E columns red, I blue, 0 white."""
    W_in, W_rec, _W_out, dale_sign = weights_for_plot(model)
    chars = model["chars"]
    hidden_size, vocab_size = W_in.shape
    unit_labels = hidden_unit_labels(dale_sign, hidden_size)
    boundary = ei_block_boundary(dale_sign)

    # Hidden units are columns: E block (red) then I block (blue).
    W_input = W_in.T
    W_hidden = W_rec
    vmax = max(symmetric_abs_vmax(W_input, W_hidden), 1e-9)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(max(8, vocab_size * 0.5 + hidden_size * 0.45), max(3.5, hidden_size * 0.55)),
        constrained_layout=True,
    )
    cmap = plt.cm.RdBu_r

    im0 = axes[0].imshow(
        W_input, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax,
        interpolation="nearest", origin="lower",
    )
    axes[0].set_title("Input")
    axes[0].set_xlabel("hidden unit (E | I)")
    axes[0].set_ylabel("input character")
    axes[0].set_xticks(range(hidden_size))
    axes[0].set_xticklabels(unit_labels, fontsize=6, rotation=90)
    axes[0].set_yticks(range(vocab_size))
    axes[0].set_yticklabels(char_axis_labels(chars), fontsize=8)
    _draw_ei_guides(axes[0], boundary, horizontal=False, vertical=True)

    im1 = axes[1].imshow(
        W_hidden, aspect="equal", cmap=cmap, vmin=-vmax, vmax=vmax,
        interpolation="nearest", origin="lower",
    )
    axes[1].set_title("Hidden")
    axes[1].set_xlabel("source h (E | I)")
    axes[1].set_ylabel("target h (E | I)")
    axes[1].set_xticks(range(hidden_size))
    axes[1].set_xticklabels(unit_labels, fontsize=6, rotation=90)
    axes[1].set_yticks(range(hidden_size))
    axes[1].set_yticklabels(unit_labels, fontsize=6)
    _draw_ei_guides(axes[1], boundary, horizontal=True, vertical=True)

    fig.colorbar(im1, ax=axes, fraction=0.03, pad=0.02, label="weight (E red, I blue)")
    fig.suptitle("Learned weights", y=1.02)
    save_path = os.path.join(out_dir, "weights.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def _mean_in_per_unit(W_in: np.ndarray, W_rec: np.ndarray) -> np.ndarray:
    """Mean weight over all connections into each hidden unit (input + recurrent row)."""
    hidden_size = W_in.shape[0]
    return np.array([
        np.mean(np.concatenate([W_in[i], W_rec[i]]))
        for i in range(hidden_size)
    ])


def _mean_out_per_unit(W_rec: np.ndarray, W_out: np.ndarray) -> np.ndarray:
    """Mean weight over all connections out of each hidden unit (recurrent col + readout col)."""
    hidden_size = W_rec.shape[0]
    return np.array([
        np.mean(np.concatenate([W_rec[:, j], W_out[:, j]]))
        for j in range(hidden_size)
    ])


def _extract_ei_block(
    W_in: np.ndarray,
    W_hh: np.ndarray,
    *,
    dale_sign: np.ndarray,
    layer: str,
    post: str,
    pre: str,
    vocab_size: int,
) -> np.ndarray:
    """Flatten one E/I submatrix (target row E/I × source E/I)."""
    from rnn_dyn import dale_ei_blocks

    exc, inh = dale_ei_blocks(dale_sign)
    post_idx = exc if post == "E" else inh
    pre_idx = exc if pre == "E" else inh
    if layer == "xh":
        # Input has no E/I; map pre E/I to first/second half of character alphabet.
        mid = max(vocab_size // 2, 1)
        cols = np.arange(0, mid) if pre == "E" else np.arange(mid, vocab_size)
        if len(post_idx) == 0 or len(cols) == 0:
            return np.array([])
        return W_in[np.ix_(post_idx, cols)].ravel()
    if len(post_idx) == 0 or len(pre_idx) == 0:
        return np.array([])
    return W_hh[np.ix_(post_idx, pre_idx)].ravel()


def _collect_block_weights(
    snaps: np.ndarray,
    *,
    hidden_size: int,
    vocab_size: int,
    dale_sign: np.ndarray,
    layer: str,
    post: str,
    pre: str,
) -> np.ndarray:
    """Weight trajectories for one block; shape (n_snap, n_syn), sorted by |w| range."""
    from rnn_dyn import unpack_weight_snapshot

    rows = []
    for vec in snaps:
        W_in, W_hh, _ = unpack_weight_snapshot(vec, hidden_size, vocab_size)
        block = _extract_ei_block(
            W_in, W_hh, dale_sign=dale_sign, layer=layer, post=post, pre=pre,
            vocab_size=vocab_size,
        )
        rows.append(block)

    max_len = max((s.size for s in rows), default=0)
    if max_len == 0:
        return np.zeros((len(rows), 0))
    out = np.full((len(rows), max_len), np.nan)
    for t, s in enumerate(rows):
        out[t, : s.size] = s
    spread = np.nanmax(out, axis=0) - np.nanmin(out, axis=0)
    order = np.argsort(spread)[::-1]
    return out[:, order]


def _panel_vmax(data: np.ndarray, pct: float = 99.0) -> float:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 1e-3
    return max(float(np.percentile(np.abs(finite), pct)), 1e-4)


def _plot_ei_block_panel(ax, data, iters, title: str) -> object | None:
    if data.size == 0 or data.shape[1] == 0:
        ax.text(0.5, 0.5, "no synapses", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return None
    vmax = _panel_vmax(data)
    im = ax.imshow(
        data.T,
        aspect="auto",
        cmap=plt.cm.RdBu_r,
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
        origin="lower",
    )
    ax.set_title(f"{title}\n(n={data.shape[1]} syns)", fontsize=9)
    ax.set_xlabel("iteration")
    ax.set_ylabel("synapse (sorted by |w| range)")
    if len(iters) > 0:
        tick_idx = np.linspace(0, len(iters) - 1, min(6, len(iters)), dtype=int)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([str(iters[i]) for i in tick_idx], fontsize=7)
    return im


def plot_weight_dynamics_over_training(model, save_path: str) -> None:
    """Eight weight heatmaps over training: 4× W_xh + 4× W_hh (EE/EI/IE/II)."""
    if "weight_snap_outgoing" not in model:
        print(f"skip {save_path}: re-run min-char-rnn.py to record weight snapshots")
        return

    snaps = np.asarray(model["weight_snap_outgoing"], dtype=float)
    iters = np.asarray(model["weight_snap_iterations"], dtype=int)
    if snaps.ndim != 2 or snaps.shape[0] < 2:
        print(f"skip {save_path}: insufficient weight snapshot history")
        return

    dale_sign = model.get("dale_sign")
    if dale_sign is None or len(dale_sign) != int(model["hidden_size"]):
        print(f"skip {save_path}: Dale sign vector required for E/I blocks")
        return

    from rnn_dyn import snapshot_vector_layout

    hidden_size = int(model["hidden_size"])
    vocab_size = int(model["vocab_size"])
    layout = snapshot_vector_layout(hidden_size, vocab_size, snaps.shape[1])
    if layout == "outgoing":
        print(
            f"skip {save_path}: re-run training for full snapshots "
            "(need W_xh + W_hh in weight_snap_outgoing)",
        )
        return

    viol = np.asarray(model.get("weight_snap_violation_frac", []), dtype=float)
    dale_sign = np.asarray(dale_sign, dtype=float)

    xh_blocks = [("E", "E"), ("E", "I"), ("I", "E"), ("I", "I")]
    hh_blocks = [("E", "E"), ("E", "I"), ("I", "E"), ("I", "I")]
    # post = target row; pre = source (vocab half for xh, hidden unit for hh).
    xh_titles = [r"$W_{xh}$ EE", r"$W_{xh}$ EI", r"$W_{xh}$ IE", r"$W_{xh}$ II"]
    hh_titles = [r"$W_{hh}$ EE", r"$W_{hh}$ EI", r"$W_{hh}$ IE", r"$W_{hh}$ II"]

    block_data = []
    for post, pre in xh_blocks:
        block_data.append(
            _collect_block_weights(
                snaps,
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                dale_sign=dale_sign,
                layer="xh",
                post=post,
                pre=pre,
            )
        )
    for post, pre in hh_blocks:
        block_data.append(
            _collect_block_weights(
                snaps,
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                dale_sign=dale_sign,
                layer="hh",
                post=post,
                pre=pre,
            )
        )

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    fig.suptitle(
        r"Weight per synapse over training (per-panel scale; E red, I blue) — "
        r"$W_{xh}$: E/I row $\times$ input half; $W_{hh}$: E/I row $\times$ E/I column",
        fontsize=10,
        y=1.02,
    )

    ims = []
    for ax, data, title in zip(axes[0], block_data[:4], xh_titles):
        im = _plot_ei_block_panel(ax, data, iters, title)
        if im is not None:
            ims.append(im)
    for ax, data, title in zip(axes[1], block_data[4:], hh_titles):
        im = _plot_ei_block_panel(ax, data, iters, title)
        if im is not None:
            ims.append(im)

    if ims:
        fig.colorbar(
            ims[-1], ax=axes.ravel().tolist(), fraction=0.02, pad=0.02,
            label="weight (E red, I blue; scale varies per panel)",
        )

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def plot_weight_eigenspectra(model, save_path: str) -> None:
    """Spectra, pooled weight histogram, and per-unit mean |in| / |out|."""
    W_in, W_rec, W_out, dale_sign = weights_for_plot(model)
    b_h = np.asarray(model["bias_hidden"]).ravel()
    if dale_sign is not None and len(dale_sign) == W_in.shape[0]:
        from rnn_dyn import dale_signs_ordered, permute_hidden_by_dale
        if not dale_signs_ordered(dale_sign):
            _, _, _, b_h, _ = permute_hidden_by_dale(W_in, W_rec, W_out, b_h, dale_sign)
    b_o = np.asarray(model["bias_output"]).ravel()
    hidden_size = W_in.shape[0]
    unit_labels = hidden_unit_labels(dale_sign, hidden_size)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)

    eigs = np.linalg.eigvals(W_rec)
    ax = axes[0, 0]
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="#888888", lw=0.9, ls="--", zorder=1)
    ax.scatter(
        eigs.real, eigs.imag,
        c=np.abs(eigs), cmap="viridis", s=55, edgecolors="black", linewidths=0.4, zorder=3,
    )
    ax.axhline(0, color="lightgrey", lw=0.6)
    ax.axvline(0, color="lightgrey", lw=0.6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Re(λ)")
    ax.set_ylabel("Im(λ)")
    ax.set_title(r"$W_{hh}$ eigenvalues")
    ax.grid(True, linestyle=":", alpha=0.35)

    for ax, name, W in zip(
        axes[0, 1:],
        (r"$W_{xh}$ singular values", r"$W_{ho}$ singular values"),
        (W_in, W_out),
    ):
        singular = np.linalg.svd(W, compute_uv=False)
        ax.bar(np.arange(len(singular)), singular, color="#4c72b0", edgecolor="black", linewidth=0.3)
        ax.set_xlabel("index")
        ax.set_ylabel("σ")
        ax.set_title(name)
        ax.grid(True, axis="y", linestyle=":", alpha=0.35)

    all_weights = np.concatenate([
        W_in.ravel(), W_rec.ravel(), W_out.ravel(), b_h, b_o,
    ])
    ax_hist = axes[1, 0]
    bins = np.linspace(-np.max(np.abs(all_weights)), np.max(np.abs(all_weights)), 41)
    ax_hist.hist(
        all_weights, bins=bins, color="#888888", alpha=0.55,
        edgecolor="white", linewidth=0.4, density=True, label="all",
    )
    ax_hist.hist(
        W_in.ravel(), bins=bins, color="#4c72b0", alpha=0.45,
        edgecolor="white", linewidth=0.3, density=True, label=r"$W_{xh}$ (in)",
    )
    ax_hist.hist(
        W_out.ravel(), bins=bins, color="#dd8452", alpha=0.45,
        edgecolor="white", linewidth=0.3, density=True, label=r"$W_{ho}$ (out)",
    )
    ax_hist.axvline(0, color="black", lw=0.8)
    ax_hist.set_xlabel("weight value")
    ax_hist.set_ylabel("density")
    ax_hist.set_title("Weight distributions")
    ax_hist.legend(fontsize=7, framealpha=0.9)
    ax_hist.grid(True, axis="y", linestyle=":", alpha=0.35)

    mean_in = _mean_in_per_unit(W_in, W_rec)
    mean_out = _mean_out_per_unit(W_rec, W_out)
    x = np.arange(hidden_size)
    width = 0.38

    ax_in = axes[1, 1]
    ax_in.bar(x - width / 2, mean_in, width=width, color="#4c72b0", edgecolor="black", linewidth=0.3)
    ax_in.set_xticks(x)
    ax_in.set_xticklabels(unit_labels)
    ax_in.axhline(0, color="black", lw=0.8)
    ax_in.set_ylabel("mean weight")
    ax_in.set_title("Mean incoming per unit")
    ax_in.grid(True, axis="y", linestyle=":", alpha=0.35)

    ax_out = axes[1, 2]
    ax_out.bar(x + width / 2, mean_out, width=width, color="#dd8452", edgecolor="black", linewidth=0.3)
    ax_out.set_xticks(x)
    ax_out.set_xticklabels(unit_labels)
    ax_out.axhline(0, color="black", lw=0.8)
    ax_out.set_ylabel("mean weight")
    ax_out.set_title("Mean outgoing per unit")
    ax_out.grid(True, axis="y", linestyle=":", alpha=0.35)

    fig.suptitle("Weight spectra and distributions (final model)", y=1.01)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def plot_output_probs(text, output_probs, chars, save_path):
    """Heatmap of P(next char) over time; overlay the true next char."""
    vocab_size = len(chars)
    length = len(text)
    targets = list(text[1:]) + [text[0]]
    target_indices = np.array([chars.index(c) for c in targets])

    fig, ax = plt.subplots(figsize=(max(12, length * 0.15), 4))
    im = ax.imshow(
        output_probs.T,
        aspect="auto", cmap="viridis", vmin=0, vmax=1,
        interpolation="nearest", origin="lower",
    )

    ax.set_yticks(range(vocab_size))
    ax.set_yticklabels(chars)
    ax.set_xticks(range(length))
    ax.set_xticklabels(list(text), fontsize=7)
    ax.set_xlabel("timestep / input character")
    ax.set_ylabel("predicted next char")
    ax.set_title("P(next char | input so far)  —  red dots = actual next char")

    ax.scatter(
        np.arange(length), target_indices,
        color="red", s=18, edgecolor="white", linewidth=0.5, zorder=3,
    )

    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="probability")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"wrote {save_path}")


def resolve_paths(args):
    """Return (model_path, input_path, out_dir) from --exp or explicit paths."""
    if args.exp:
        ensure_experiment_dirs(args.exp)
        return (
            str(model_path(args.exp)),
            str(input_path(args.exp)),
            str(plots_dir(args.exp)),
        )
    out = args.out_dir if args.out_dir is not None else "plots"
    return args.model, args.input, out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", default=None,
                        help="experiment name under experiments/<exp>/")
    parser.add_argument("--model", default="model.npz")
    parser.add_argument("--input", default="input.txt")
    parser.add_argument("--length", type=int, default=50,
                        help="how many characters of the corpus to visualize (default: 50)")
    parser.add_argument("--out-dir", default=None,
                        help="plot output directory (default: experiments/<exp>/plots or plots)")
    parser.add_argument("--no-cluster-per-char", action="store_true",
                        help="keep per-character heatmap rows in sequence order")
    parser.add_argument(
        "--dfa-annot-style",
        default="leaders",
        choices=["leaders", "none", "annots_only"],
        help="DFA annotation style for hidden_states_pca_dfa_analysis.png",
    )
    args = parser.parse_args()

    model_file, input_file, out_dir = resolve_paths(args)
    os.makedirs(out_dir, exist_ok=True)
    if args.exp:
        print(f"experiment: {args.exp} -> {out_dir}")

    model = load_model(model_file)
    print(f"loaded model: hidden_size={model['hidden_size']}, "
          f"vocab_size={model['vocab_size']}, chars={''.join(model['chars'])}")

    plot_learned_weights(model, out_dir)
    plot_weight_eigenspectra(
        model, save_path=os.path.join(out_dir, "weights_eigenspectra.png")
    )
    plot_weight_dynamics_over_training(
        model, os.path.join(out_dir, "weight_dynamics_over_training.png")
    )
    plot_learning_curve(
        model,
        save_path=os.path.join(out_dir, "learning_curve.png"),
    )
    plot_sample_before_after(
        model,
        save_path=os.path.join(out_dir, "samples_before_after.png"),
    )

    with open(input_file, "r") as f:
        text = f.read()[: args.length]
    print(f"running forward pass over {len(text)} characters of {input_file}")

    spaced = corpus_uses_word_spacing(text, args.exp)
    words = vocabulary_for_experiment(args.exp) if args.exp else infer_task_words(text)
    automaton = build_minimized_vocabulary_automaton(words) if words else None
    if automaton is not None:
        print(
            "PCA point colors: minimized DFA state after in-word prefix since last space"
            if spaced
            else "PCA point colors: minimized DFA state (stream prefix)"
        )
    elif spaced:
        print("annotation mode: prefix after space (e.g. h, ha, hat; ' ' on spaces)")

    hidden_states, output_probs = forward_pass(model, text)
    targets = list(text[1:]) + [text[0]]
    act_label = activation_label(use_relu=bool(model.get("use_relu", False)))

    plot_hidden_states_heatmap(
        text, hidden_states,
        save_path=os.path.join(out_dir, "activation_heatmap.png"),
        act_label=act_label,
    )

    plot_output_probs(
        text, output_probs, model["chars"],
        save_path=os.path.join(out_dir, "next_char_prob_sequence_heatmap.png"),
    )

    plot_per_char_hidden_state_heatmaps(
        text, hidden_states, model["chars"],
        save_path=os.path.join(out_dir, "activation_by_input_char.png"),
        cluster_rows=not args.no_cluster_per_char,
        spaced=spaced,
    )

    plot_pca_context_labels(
        text, hidden_states, model["chars"],
        save_path=os.path.join(out_dir, "embedding_panels_context.png"),
        spaced=spaced,
        automaton=automaton,
    )

    plot_pca_prediction_regions(
        model, text, hidden_states, model["chars"],
        save_path=os.path.join(out_dir, "next_char_regions_pca.png"),
        spaced=spaced,
        automaton=automaton,
    )

    if automaton is not None and words:
        plot_pca_dfa_analysis(
            text, hidden_states, model["chars"], words,
            save_path=os.path.join(out_dir, "dfa_and_embedding_pca.png"),
            automaton=automaton,
            spaced=spaced,
            annot_style=args.dfa_annot_style,
        )

    if spaced:
        plot_space_to_space_trajectories(
            text, hidden_states,
            save_path=os.path.join(
                out_dir, "word_trajectories_pca.png"
            ),
            model=model,
            spaced=spaced,
            automaton=automaton,
            annot_style=args.dfa_annot_style,
        )

    plot_pca_next_char_probability_panels(
        model, text, hidden_states, model["chars"],
        save_path=os.path.join(out_dir, "next_char_prob_panels_pca.png"),
        spaced=spaced,
        automaton=automaton,
    )

    plot_pca_vector_field(
        text,
        hidden_states,
        model,
        os.path.join(out_dir, "vector_field_grid_pca_no_input.png"),
    )

    plot_hidden_states_clustermap(
        text, hidden_states, model["chars"],
        save_path=os.path.join(out_dir, "activation_clustered_heatmap.png"),
        exp_name=args.exp,
    )

    plot_hidden_states_correlation_clustermap(
        text, hidden_states, model["chars"],
        save_path=os.path.join(out_dir, "state_correlation_clustered_heatmap.png"),
        spaced=spaced,
        automaton=automaton,
    )

    if automaton is not None:
        plot_dfa_grouped_state_correlation(
            text,
            hidden_states,
            save_path=os.path.join(out_dir, "state_correlation_by_dfa_state.png"),
            spaced=spaced,
            automaton=automaton,
        )
        plot_dfa_state_distance_comparison(
            text, hidden_states, automaton,
            save_path=os.path.join(out_dir, "dfa_state_distance_comparison.png"),
            spaced=spaced,
        )

    if model["hidden_size"] == 2:
        plot_state_trajectory(
            hidden_states,
            color_by_chars=list(text),
            chars=model["chars"],
            title=f"Hidden state trajectory over {len(text)} chars (colored by INPUT char)",
            save_path=os.path.join(out_dir, "state_trajectory_by_input.png"),
        )
        plot_state_trajectory(
            hidden_states,
            color_by_chars=targets,
            chars=model["chars"],
            title=f"Hidden state trajectory over {len(text)} chars (colored by TARGET / next char)",
            save_path=os.path.join(out_dir, "state_trajectory_by_target.png"),
        )

    correct = np.sum(np.argmax(output_probs, axis=1) ==
                     np.array([model["chars"].index(c) for c in targets]))
    print(f"top-1 next-char accuracy over the {len(text)}-char window: "
          f"{correct}/{len(text)} = {100*correct/len(text):.1f}%")

    if words:
        trie_path, dfa_path = write_vocabulary_diagrams(words, Path(out_dir))
        print(f"wrote {trie_path}")
        print(f"wrote {dfa_path}")


if __name__ == "__main__":
    main()
