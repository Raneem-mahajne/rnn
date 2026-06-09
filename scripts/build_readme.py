"""Regenerate README.md and sync figures into docs/figures/ for preview + GitHub."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_PLOTS = ROOT / "experiments" / "ten_word_overlap_s" / "plots"
FIGURES = ROOT / "docs" / "figures"

# Ordered list: (readme number, source filename, dest basename)
FIGURE_FILES: list[tuple[int, str, str]] = [
    (1, "vocabulary_trie.png", "01_vocabulary_trie.png"),
    (2, "vocabulary_min_dfa.png", "02_vocabulary_min_dfa.png"),
    (3, "learning_curve.png", "03_learning_curve.png"),
    (4, "samples_before_after.png", "04_samples_before_after.png"),
    (5, "weights.png", "05_weights.png"),
    (6, "weights_eigenspectra.png", "06_weights_eigenspectra.png"),
    (7, "activation_heatmap.png", "07_activation_heatmap.png"),
    (8, "next_char_prob_sequence_heatmap.png", "08_next_char_prob_sequence.png"),
    (9, "activation_by_input_char.png", "09_activation_by_input_char.png"),
    (10, "activation_clustered_heatmap.png", "10_activation_clustered_heatmap.png"),
    (11, "embedding_panels_context.png", "11_embedding_panels_context.png"),
    (12, "dfa_and_embedding_pca.png", "12_dfa_and_embedding_pca.png"),
    (13, "next_char_regions_pca.png", "13_next_char_regions_pca.png"),
    (14, "next_char_prob_panels_pca.png", "14_next_char_prob_panels_pca.png"),
    (15, "vector_field_grid_pca_no_input.png", "15_vector_field_grid_pca.png"),
    (16, "word_trajectories_pca.png", "16_word_trajectories_pca.png"),
    (17, "state_correlation_clustered_heatmap.png", "17_state_correlation_clustered.png"),
    (18, "state_correlation_by_dfa_state.png", "18_state_correlation_by_dfa_state.png"),
    (19, "dfa_state_distance_comparison.png", "19_dfa_state_distance_comparison.png"),
]


def sync_figures() -> None:
    """Copy experiment plots into docs/figures/ (paths README and GitHub both resolve)."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    for _n, src_name, dest_name in FIGURE_FILES:
        src = SRC_PLOTS / src_name
        dest = FIGURES / dest_name
        if not src.is_file():
            raise FileNotFoundError(f"missing plot for README: {src}")
        shutil.copy2(src, dest)


def fig(n: int, dest_name: str, caption: str) -> str:
    path = f"docs/figures/{dest_name}"
    return (
        f"![Figure {n}]({path})\n\n"
        f"*Figure {n}. {caption}*\n\n"
    )


def main() -> None:
    sync_figures()

    cap14 = r"Per-character fields $P(\text{next} = c \mid \mathbf{h})$ over the PCA plane."
    cap15 = (
        r"No-input recurrent vector field "
        r"$\mathbf{h}_{t+1} = \tanh(W_{hh}\mathbf{h}_t)$ projected to PC1-PC2."
    )
    cap5 = r"Input weights $W_{xh}$ and recurrent weights $W_{hh}$ after training."

    captions: dict[int, str] = {
        1: "Trie for ten_word_overlap. Shared prefixes and suffixes merge into common paths.",
        2: "Minimal DFA for the ten training words. State labels show surviving prefix sets.",
        3: "Training cross-entropy (blue) and invalid-word rate on long stochastic rollouts (orange).",
        4: "Fifty-character training excerpt (top) and stochastic generations before and after learning.",
        5: cap5,
        6: "Eigenvalue spectra of recurrent weights - timescales available to the hidden state.",
        7: "Hidden-unit activations at each timestep.",
        8: "Softmax next-character distribution vs ground truth at each timestep.",
        9: "Hidden states grouped by input character; columns labeled by in-word prefix since last space.",
        10: "Hierarchically clustered timesteps x hidden units; row labels = in-word prefix.",
        11: "2D embeddings (PCA, UMAP, t-SNE, Isomap) with prefix annotations.",
        12: "Central result: minimized DFA (left) and PCA of hidden states colored by DFA state (right).",
        13: "Argmax next-character regions and prediction entropy over the PCA plane.",
        14: cap14,
        15: cap15,
        16: "PCA trajectories between word boundaries - the first organizing axis (prefix / position in word).",
        17: "Pearson correlation between hidden states at all timesteps; tick color = DFA state.",
        18: "Correlation matrix with timesteps grouped by minimized DFA state.",
        19: "Pairwise hidden-state distances: within DFA state vs between DFA states vs same input character.",
    }

    readme = f"""# Statistical Word Learning in a Minimal Character-Level RNN

This repository trains a **small vanilla recurrent neural network** on synthetic text streams designed to mimic infant **statistical learning** of words. Words are sampled from a finite vocabulary and concatenated (optionally with spaces). After training, we visualize how the RNN hidden state relates to:

1. **Position within the current word** - the in-word prefix since the last boundary (`h`, `ha`, `hat` after a space).
2. **State in the minimal vocabulary DFA** - where you are in the automaton that accepts exactly the training vocabulary.

The central claim: a generic next-character predictor develops hidden-state geometry that **factorizes along both axes**. Prefix length captures how far into the word you are; DFA state captures which lexical hypotheses remain alive given overlapping spellings.

Related work: [creating_transformer (statistical learning)](https://github.com/Raneem-mahajne/creating_transformer/tree/statistical_learning) uses the same synthetic regimes with a minimal transformer. This repo is the **RNN** counterpart.

---

## Quick start

**Requirements:** Python 3.10+, NumPy, Matplotlib, Seaborn, Pandas, SciPy; optional UMAP.

```bash
pip install -r requirements.txt
pip install matplotlib seaborn pandas scipy umap-learn
python run_experiments.py --only ten_word_overlap_s
python scripts/build_readme.py   # sync docs/figures/ for README preview
```

Training plots: `experiments/<name>/plots/`. README figures: `docs/figures/` (copied from `ten_word_overlap_s`).

---

## Repository structure

```
task.py              # Statistical-learning corpora
min-char-rnn.py      # Vanilla RNN training
visualize.py         # Post-training figures
vocab_diagrams.py    # Trie + minimal DFA
experiment.py        # Hyperparameters per regime
run_experiments.py   # Batch train + visualize
docs/figures/        # README figure copies (run scripts/build_readme.py)
experiments/<name>/  # input.txt, model.npz, plots/
```

---

## Available experiments

| Folder | Words | Notes |
|--------|-------|-------|
| `ten_word_overlap` / `_s` | 10 x 3-letter words | Primary demo |
| `ten_four_letter_overlap` / `_s` | 10 x 4-letter words | Longer words |
| `ten_four_letter_overlap_s_dale` | Same 4-letter vocab | Dale's law + ReLU |
| `six_word_overlap` / `_s` | 6 words | Smaller vocab |
| `twelve_word_overlap` / `_s` | 12 words | Mixed overlap |
| `sixteen_word_overlap` / `_s` | 16 words | Largest regime |

Default: **50k chars**, **15k steps**, $h=32$, BPTT $=40$, **50-char viz window**.

---

## Paper: Learning words with an RNN

Walkthrough experiment: **`ten_word_overlap_s`**. Figures below are in `docs/figures/`.

**Abstract.** Infants segment speech using distributional statistics alone (Saffran, Aslin, & Newport, 1996). We train a 32-unit $\\tanh$ RNN on a ten-word corpus with overlapping trigrams and ask whether hidden states align with **word-internal position** and **lexical DFA state**. Two principles emerge: clustering by **in-word prefix** and by **minimized DFA state**.

---

## 1. Introduction

### 1.1 Statistical learning and word segmentation

Jenny Saffran showed that infants discover word-like units from continuous streams where transitional probabilities are high within words and low across boundaries.

```
... hat cat met rat tea eat cat hat ...
```

The objective is next-character prediction only. Does $\\mathbf{{h}}_t$ encode position inside the current word and which vocabulary items remain possible?

### 1.2 Why an RNN?

$$
\\mathbf{{h}}_t = \\tanh\\!\\left( W_{{xh}} \\mathbf{{x}}_t + W_{{hh}} \\mathbf{{h}}_{{t-1}} + \\mathbf{{b}}_h \\right)
$$

$$
P(\\text{{next}} = c \\mid \\mathbf{{x}}_{{\\leq t}}) = \\mathrm{{softmax}}\\!\\left( W_{{hy}} \\mathbf{{h}}_t + \\mathbf{{b}}_y \\right)_c
$$

### 1.3 Two axes of organization

| Axis | Encodes | Labels | Key plots |
|------|---------|--------|-----------|
| Word-boundary / prefix | Letters since last space | `h`, `ha`, `hat` | Trajectories, prefix heatmaps |
| DFA state | Surviving lexical branches | $q_k$ | DFA-colored PCA, grouped correlation |

---

## 2. Task and vocabulary structure

### 2.1 The `ten_word_overlap` regime

| Group | Words |
|-------|-------|
| `-at` family | cat, hat, mat, rat |
| `-et` family | met, pet, net |
| vowel overlap | ate, eat, tea |

### 2.2 Trie and minimal DFA

{fig(1, FIGURE_FILES[0][2], captions[1])}
{fig(2, FIGURE_FILES[1][2], captions[2])}

---

## 3. Model and training

| Hyperparameter | Value |
|----------------|-------|
| Hidden units $h$ | 32 |
| Activation | $\\tanh$ |
| Optimizer | Adagrad ($\\eta = 0.1$) |
| BPTT window | 40 characters |
| Training steps | 15,000 |
| Corpus size | 50,000 characters |

---

## 4. Results

Analysis window: first **50 characters** of the trained corpus.

### 4.1 The model learns the stream

{fig(3, FIGURE_FILES[2][2], captions[3])}
{fig(4, FIGURE_FILES[3][2], captions[4])}

### 4.2 Learned parameters

{fig(5, FIGURE_FILES[4][2], captions[5])}
{fig(6, FIGURE_FILES[5][2], captions[6])}

### 4.3 Activations and predictions over time

{fig(7, FIGURE_FILES[6][2], captions[7])}
{fig(8, FIGURE_FILES[7][2], captions[8])}
{fig(9, FIGURE_FILES[8][2], captions[9])}
{fig(10, FIGURE_FILES[9][2], captions[10])}

### 4.4 Hidden-state geometry

{fig(11, FIGURE_FILES[10][2], captions[11])}
{fig(12, FIGURE_FILES[11][2], captions[12])}
{fig(13, FIGURE_FILES[12][2], captions[13])}
{fig(14, FIGURE_FILES[13][2], captions[14])}
{fig(15, FIGURE_FILES[14][2], captions[15])}
{fig(16, FIGURE_FILES[15][2], captions[16])}

### 4.5 Correlation structure: prefix and DFA

{fig(17, FIGURE_FILES[16][2], captions[17])}
{fig(18, FIGURE_FILES[17][2], captions[18])}
{fig(19, FIGURE_FILES[18][2], captions[19])}

---

## 5. Discussion

### 5.1 Two organizations, one hidden space

1. **In-word prefix.** As the network reads $h \\to ha \\to hat$, $\\mathbf{{h}}_t$ tracks distance from the last word boundary (Figures 9-10, 16).
2. **DFA state.** Each prefix maps to $q_k$ in the minimal automaton (Figures 12, 18-19).

### 5.2 Statistical learning connection

High within-word and lower cross-boundary transitional probabilities drive prefix structure in $\\mathbf{{h}}_t$. The DFA axis tracks which lexical branches remain open.

### 5.3 Limitations

Small synthetic vocabulary; PCA projects $h=32$ to 2D; vanilla RNN; single trained run.

---

## 6. Reproduce

```bash
python run_experiments.py --only ten_word_overlap_s
python scripts/export_diagram_pngs.py
python scripts/build_readme.py
```

---

## 7. References

- Saffran, J. R., Aslin, R. N., & Newport, E. L. (1996). Statistical learning by 8-month-old infants. *Science*, 274(5294), 1926-1928.
- Karpathy, A. [Minimal character-level RNN](https://gist.github.com/karpathy/d4dee566867f8291f086).
- [creating_transformer - statistical learning](https://github.com/Raneem-mahajne/creating_transformer/tree/statistical_learning).

---

## License

`min-char-rnn.py` retains the BSD license from the Karpathy gist.
"""
    out = ROOT / "README.md"
    out.write_bytes(readme.encode("utf-8"))
    print(f"wrote {out}")
    print(f"synced {len(FIGURE_FILES)} figures -> {FIGURES}")


if __name__ == "__main__":
    main()
