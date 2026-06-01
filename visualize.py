"""
Visualize a trained min-char-rnn after training is done.

Loads the saved model from `model.npz`, runs a forward pass over the first
`--length` characters of `input.txt`, and plots:

  1) hidden_states_heatmap.png
       Heatmap of every hidden unit's activation at every timestep.
       Y-axis: hidden units (h0, h1, ...). X-axis: input character.
       Works for any `hidden_size`.

  2) output_probabilities.png
       Heatmap of the model's next-char probability distribution at every
       position, with the actual next character marked.

  3) hidden_states_trajectory.png   (only when hidden_size == 2)
       2D scatter of every hidden state visited, colored by the input
       character that produced it, with grey arrows showing the temporal
       trajectory through state space.

  4) hidden_states_by_target.png    (only when hidden_size == 2)
       Same scatter, colored by the *next* (target) character.

  5) learning_curve.png
       Per-window training loss vs iteration (from model.npz).

  6) hidden_states_pca_context_labels.png
       2D PCA of hidden states; annotation = prev2 + current char (3 chars).

  7) hidden_states_pca_prediction_regions.png
       Two PCA panels: argmax next-char regions and prediction entropy (2D h).

  8) hidden_states_pca_next_char_prob_panels.png
       One panel per vocab char: P(next = char) over the PCA plane (softmax).

  9) hidden_states_clustermap.png
       Heatmap of timesteps × hidden units with row/column dendrograms
       (average linkage). Row labels: two preceding chars + current char.

  10) weights.png
       Side-by-side heatmaps of final input weights (char columns × hidden rows)
       and recurrent hidden→hidden weights (h0..h{n-1} in index order).

Usage:
    python visualize.py --exp shared_letters
    python visualize.py --exp ten_word_overlap --length 100
    python visualize.py --model path/to/model.npz --input path/to/input.txt --out-dir path/to/plots
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import ndimage

from experiment import ensure_experiment_dirs, input_path, model_path, plots_dir
from task import REGIMES


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
        hidden_state = np.tanh(
            weights_input_to_hidden  @ input_one_hot +
            weights_hidden_to_hidden @ hidden_state  +
            bias_hidden
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


def plot_hidden_states_heatmap(text, hidden_states, save_path):
    """Heatmap of every hidden unit's tanh activation over the sequence.

    rows = hidden units, columns = timesteps, color = activation in [-1, 1].
    """
    length, hidden_size = hidden_states.shape

    fig, ax = plt.subplots(figsize=(max(12, length * 0.15),
                                    max(2.5, hidden_size * 0.35)))
    im = ax.imshow(
        hidden_states.T,
        aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
        interpolation="nearest", origin="lower",
    )

    ax.set_yticks(range(hidden_size))
    ax.set_yticklabels([f"h{i}" for i in range(hidden_size)])
    ax.set_xticks(range(length))
    ax.set_xticklabels(list(text), fontsize=7)
    ax.set_xlabel("timestep / input character")
    ax.set_ylabel("hidden unit")
    ax.set_title("Hidden state activations (tanh output) over the input sequence")

    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="activation (tanh)")
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


def context_label(text, index):
    previous = "^" if index == 0 else display_char(text[index - 1])
    current = display_char(text[index])
    return f"{previous}{current}@{index}"


def timestep_context_label(text, index):
    """Two preceding characters plus the current input character (3 chars, ^-padded)."""
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


def infer_task_words(text: str) -> list[str] | None:
    """Best-matching word vocabulary from task.py regimes for this corpus."""
    text_chars = set(text)
    best_words = None
    best_char_count = None
    for words in REGIMES.values():
        regime_chars = set("".join(words))
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


def plot_hidden_states_clustermap(text, hidden_states, chars, save_path):
    """Heatmap (timesteps × hidden units) with seaborn clustermap layout."""
    n_rows, n_cols = hidden_states.shape
    if n_rows == 0:
        return

    row_labels = [timestep_context_label(text, t) for t in range(n_rows)]
    col_labels = [f"h{i}" for i in range(n_cols)]
    data = pd.DataFrame(hidden_states, index=row_labels, columns=col_labels)

    grid = sns.clustermap(
        data,
        method="average",
        metric="euclidean",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        figsize=(max(9, n_cols * 0.55), max(6, n_rows * 0.25)),
        dendrogram_ratio=(0.12, 0.1),
        cbar=False,
        xticklabels=True,
        yticklabels=True,
    )
    grid.ax_heatmap.set_xlabel("hidden unit")
    grid.ax_heatmap.set_ylabel("timestep (prev2 + current)")
    grid.ax_heatmap.tick_params(axis="y", labelsize=7)
    grid.ax_heatmap.tick_params(axis="x", labelsize=8)
    grid.fig.suptitle(
        f"Hidden states clustered (timesteps × units) · {original_vocabulary_title(chars, text)}",
        y=1.02, fontsize=11,
    )
    grid.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(grid.fig)
    print(f"wrote {save_path}")


def fit_pca_2d(points):
    """PCA fit: return 2D coords, mean, and (2, D) principal axes for reconstruction."""
    mean = np.mean(points, axis=0)
    centered = points - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:2]
    coords = centered @ components.T
    return coords, mean, components


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


def plot_learning_curve(model, save_path):
    """Plot per-window training loss recorded during training."""
    if "loss_iterations" not in model:
        print(f"skip {save_path}: re-run min-char-rnn.py to record loss history")
        return

    iters = model["loss_iterations"]
    window = model["loss_window"]

    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    ax.plot(iters, window, color="steelblue", linewidth=1.0)
    ax.set_xlabel("iteration")
    ax.set_ylabel("cross-entropy (sum over BPTT window)")
    ax.set_title("Training loss")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"wrote {save_path}")


def trigram_sequence_colors(labels):
    """Stable color per unique 3-char context label (tab10, full saturation)."""
    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab10", max(len(unique_labels), 1))
    return {label: cmap(i) for i, label in enumerate(unique_labels)}


def layout_trigram_labels(text, projected):
    """Label positions and grouping for prev2+current (3-char) annotations."""
    labels = [timestep_context_label(text, i) for i in range(len(text))]
    sequence_color = trigram_sequence_colors(labels)
    by_sequence = defaultdict(list)
    for i, label in enumerate(labels):
        by_sequence[label].append(i)

    center = projected.mean(axis=0)
    span = max(
        float(np.ptp(projected[:, 0])),
        float(np.ptp(projected[:, 1])),
        1e-3,
    )
    label_offset = span * 0.14
    label_positions = {}

    for label, indices in by_sequence.items():
        points = projected[indices]
        centroid = points.mean(axis=0)
        outward = centroid - center
        norm = float(np.linalg.norm(outward))
        if norm < 1e-9:
            outward = np.array([0.0, 1.0])
        else:
            outward = outward / norm
        label_positions[label] = centroid + outward * label_offset

    return labels, sequence_color, by_sequence, label_positions


def add_trigram_annotations(ax, text, projected):
    """Context-colored points, leader lines, one label per 3-char sequence."""
    labels, sequence_color, by_sequence, label_positions = layout_trigram_labels(text, projected)
    point_colors = [sequence_color[label] for label in labels]

    ax.scatter(
        projected[:, 0], projected[:, 1],
        s=48, c=point_colors, edgecolors="black", linewidths=0.6,
        zorder=4,
    )

    for label, indices in by_sequence.items():
        text_pos = label_positions[label]
        color = sequence_color[label]
        for point in projected[indices]:
            ax.plot(
                [text_pos[0], point[0]], [text_pos[1], point[1]],
                color=color, linewidth=1.4, solid_capstyle="round", zorder=3,
            )
        ax.text(
            text_pos[0], text_pos[1], label,
            fontsize=10, fontweight="bold", color=color, ha="center", va="center",
            bbox=dict(
                boxstyle="round,pad=0.25", facecolor="white",
                edgecolor=color, linewidth=1.2,
            ),
            zorder=5,
        )

    return list(label_positions.values())


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
    text, hidden_states, chars, projected, save_path, title, xlabel, ylabel,
    fig_suptitle=None,
):
    """Scatter points with one 3-char context label per sequence, lines to its points."""
    _ = chars
    if len(text) == 0:
        return

    fig, ax = plt.subplots(figsize=(14, 11), constrained_layout=True)
    text_positions = add_trigram_annotations(ax, text, projected)
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


def plot_per_char_hidden_state_heatmaps(text, hidden_states, chars, save_path, cluster_rows=True):
    """Combined per-input-char heatmaps, rows = hidden units, columns = occurrences."""
    hidden_size = hidden_states.shape[1]
    groups = []

    for char in chars:
        indices = np.array([i for i, text_char in enumerate(text) if i > 0 and text_char == char])
        if len(indices) == 0:
            continue

        rows = hidden_states[indices]
        labels = [context_label(text, int(i)) for i in indices]

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

    axes[-1].set_xlabel("previous + current character @ timestep")
    fig.suptitle(
        f"Hidden states by input character · {original_vocabulary_title(chars, text)}",
        y=0.995,
    )
    fig.colorbar(last_image, ax=axes, fraction=0.015, pad=0.01, label="activation (tanh)")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"wrote {save_path}")


def trigram_avoidance_points(text, projected):
    """PC coordinates to keep region letters away from (scatter + label boxes)."""
    _, _, _, label_positions = layout_trigram_labels(text, projected)
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
    stroke = path_effects.withStroke(linewidth=4, foreground="#1a1a1a")
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
        letter = display_char(char)
        if len(letter) != 1:
            continue
        ax.text(
            position[0], position[1], letter,
            fontsize=34, fontweight="bold", color="white",
            ha="center", va="center", zorder=6,
            path_effects=[stroke],
        )


def plot_pca_context_labels(text, hidden_states, chars, save_path):
    """2D PCA of hidden states; labels show prev2 + current char (3 chars)."""
    if len(text) < 1:
        return
    plot_2d_hidden_state_labels(
        text, hidden_states, chars,
        pca_2d(hidden_states),
        save_path,
        title="2D PCA (prev2+current, 3-char)",
        xlabel="PC1",
        ylabel="PC2",
        fig_suptitle=original_vocabulary_title(chars, text),
    )


def plot_pca_prediction_regions(model, text, hidden_states, chars, save_path, grid_resolution=120):
    """PCA panels: argmax next-char regions and softmax entropy, with 3-char context labels."""
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
    avoid_xy = trigram_avoidance_points(text, projected)
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
        add_trigram_annotations(ax, text, projected)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.35)
        if ax is axes[1]:
            fig.colorbar(im, ax=ax, label="entropy (nats)", fraction=0.046, pad=0.02)

    fig.suptitle(
        f"PCA plane (prev2+current, 3-char) · {original_vocabulary_title(chars, text)}",
        fontsize=12, y=1.01,
    )
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def plot_pca_next_char_probability_panels(
    model, text, hidden_states, chars, save_path, grid_resolution=120,
):
    """One panel per vocab char: P(next = char) over the PCA plane (from softmax)."""
    n_points, hidden_size = hidden_states.shape
    vocab_size = len(chars)
    if n_points < 2 or hidden_size < 1:
        return

    grid_x, grid_y, grid_hidden, _, xlim, ylim = build_pca_plane_grid(
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


def plot_learned_weights(model, out_dir):
    """Side-by-side Wxh and Whh heatmaps; hidden units in index order h0..h{n-1}."""
    W_in = np.asarray(model["weights_input_to_hidden"])
    W_rec = np.asarray(model["weights_hidden_to_hidden"])
    chars = model["chars"]
    hidden_size, vocab_size = W_in.shape
    vmax = symmetric_abs_vmax(W_in, W_rec)
    unit_labels = [f"h{i}" for i in range(hidden_size)]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(max(8, vocab_size * 0.5 + hidden_size * 0.5), max(3.5, hidden_size * 0.55)),
        constrained_layout=True,
    )

    axes[0].imshow(
        W_in, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        interpolation="nearest", origin="lower",
    )
    axes[0].set_xticks(range(vocab_size))
    axes[0].set_xticklabels(char_axis_labels(chars), fontsize=8)
    axes[0].set_yticks(range(hidden_size))
    axes[0].set_yticklabels(unit_labels)
    axes[0].set_xlabel("input character")
    axes[0].set_ylabel("hidden unit")
    axes[0].set_title("Input → hidden (Wxh)")

    im1 = axes[1].imshow(
        W_rec, aspect="equal", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        interpolation="nearest", origin="lower",
    )
    axes[1].set_xticks(range(hidden_size))
    axes[1].set_xticklabels(unit_labels, fontsize=7, rotation=90)
    axes[1].set_yticks(range(hidden_size))
    axes[1].set_yticklabels(unit_labels)
    axes[1].set_xlabel("source h (t−1)")
    axes[1].set_ylabel("target h (t)")
    axes[1].set_title("Hidden → hidden (Whh)")

    fig.colorbar(im1, ax=axes, fraction=0.03, pad=0.02, label="weight")
    fig.suptitle("Learned weights (final model)", y=1.02)
    save_path = os.path.join(out_dir, "weights.png")
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
    args = parser.parse_args()

    model_file, input_file, out_dir = resolve_paths(args)
    os.makedirs(out_dir, exist_ok=True)
    if args.exp:
        print(f"experiment: {args.exp} -> {out_dir}")

    model = load_model(model_file)
    print(f"loaded model: hidden_size={model['hidden_size']}, "
          f"vocab_size={model['vocab_size']}, chars={''.join(model['chars'])}")

    plot_learned_weights(model, out_dir)
    plot_learning_curve(
        model,
        save_path=os.path.join(out_dir, "learning_curve.png"),
    )

    with open(input_file, "r") as f:
        text = f.read()[: args.length]
    print(f"running forward pass over {len(text)} characters of {input_file}")

    hidden_states, output_probs = forward_pass(model, text)
    targets = list(text[1:]) + [text[0]]

    plot_hidden_states_heatmap(
        text, hidden_states,
        save_path=os.path.join(out_dir, "hidden_states_heatmap.png"),
    )

    plot_output_probs(
        text, output_probs, model["chars"],
        save_path=os.path.join(out_dir, "output_probabilities.png"),
    )

    plot_per_char_hidden_state_heatmaps(
        text, hidden_states, model["chars"],
        save_path=os.path.join(out_dir, "hidden_states_by_input_char.png"),
        cluster_rows=not args.no_cluster_per_char,
    )

    plot_pca_context_labels(
        text, hidden_states, model["chars"],
        save_path=os.path.join(out_dir, "hidden_states_pca_context_labels.png"),
    )

    plot_pca_prediction_regions(
        model, text, hidden_states, model["chars"],
        save_path=os.path.join(out_dir, "hidden_states_pca_prediction_regions.png"),
    )

    plot_pca_next_char_probability_panels(
        model, text, hidden_states, model["chars"],
        save_path=os.path.join(out_dir, "hidden_states_pca_next_char_prob_panels.png"),
    )

    plot_hidden_states_clustermap(
        text, hidden_states, model["chars"],
        save_path=os.path.join(out_dir, "hidden_states_clustermap.png"),
    )

    if model["hidden_size"] == 2:
        plot_state_trajectory(
            hidden_states,
            color_by_chars=list(text),
            chars=model["chars"],
            title=f"Hidden state trajectory over {len(text)} chars (colored by INPUT char)",
            save_path=os.path.join(out_dir, "hidden_states_trajectory.png"),
        )
        plot_state_trajectory(
            hidden_states,
            color_by_chars=targets,
            chars=model["chars"],
            title=f"Hidden state trajectory over {len(text)} chars (colored by TARGET / next char)",
            save_path=os.path.join(out_dir, "hidden_states_by_target.png"),
        )

    correct = np.sum(np.argmax(output_probs, axis=1) ==
                     np.array([model["chars"].index(c) for c in targets]))
    print(f"top-1 next-char accuracy over the {len(text)}-char window: "
          f"{correct}/{len(text)} = {100*correct/len(text):.1f}%")


if __name__ == "__main__":
    main()
