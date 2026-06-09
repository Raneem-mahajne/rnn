"""Regenerate README.md and sync figures into docs/figures/."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_PLOTS = ROOT / "experiments" / "ten_word_overlap_s" / "plots"
FIGURES = ROOT / "docs" / "figures"

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
    FIGURES.mkdir(parents=True, exist_ok=True)
    for _n, src_name, dest_name in FIGURE_FILES:
        src = SRC_PLOTS / src_name
        dest = FIGURES / dest_name
        if not src.is_file():
            raise FileNotFoundError(f"missing plot for README: {src}")
        shutil.copy2(src, dest)


def figure_block(n: int, dest_name: str, caption: str, lead: str = "") -> str:
    path = f"docs/figures/{dest_name}"
    parts = []
    if lead.strip():
        parts.append(lead.strip())
        parts.append("")
    parts.append(f"![Figure {n}]({path})")
    parts.append("")
    parts.append(f"*Figure {n}. {caption}*")
    parts.append("")
    return "\n".join(parts)


# (number, dest, caption, explanatory text before figure)
FIGURE_SECTIONS: list[tuple[int, str, str, str]] = [
    (
        1,
        "01_vocabulary_trie.png",
        "Trie over the ten-word vocabulary. Nodes are prefixes; double circles are complete words. "
        "Edges are labeled by the consumed character.",
        "Before training, we compile the word list into a **trie**: a rooted tree whose edges are "
        "characters and whose paths spell valid prefixes. Every training word appears as a root-to-terminal "
        "path. Overlap is explicit: `cat`, `hat`, `mat`, and `rat` share the suffix `at`; `met`, `pet`, and "
        "`net` share `et`. The trie is the literal lexical hypothesis tree the model must implicitly navigate "
        "when predicting the next character.",
    ),
    (
        2,
        "02_vocabulary_min_dfa.png",
        "Minimized deterministic finite automaton (DFA) for the same vocabulary. Each state is labeled "
        "with the set of vocabulary words still consistent with the prefix read since the last space.",
        "The trie is folded into a **minimal DFA** by merging states with identical future continuations. "
        "This is the canonical reference machine for our second organizing axis: at each timestep we walk "
        "the DFA on the **in-word prefix** (characters since the last space) and record the current state "
        "$q_k$. Two timesteps with the same prefix always share a DFA state; two timesteps with the same "
        "input character but different prefixes generally do not.",
    ),
    (
        3,
        "03_learning_curve.png",
        "Training loss (blue, 51-iteration rolling median of per-window cross-entropy) and stochastic "
        "word-error rate (orange, right axis: percent of space-delimited tokens not in the vocabulary "
        "during long sampled rollouts).",
        "We first verify that optimization succeeds. Cross-entropy falls steadily over 15,000 iterations. "
        "In parallel we track **word error rate**: sample long strings from the model and count how often "
        "whitespace-delimited chunks are not exact vocabulary words. This metric is our analogue of "
        "\"does the generator respect the statistical word units?\" After training, invalid-word rate "
        "approaches zero: the model is not merely memorizing local trigrams; it has learned to emit legal words.",
    ),
    (
        4,
        "04_samples_before_after.png",
        "Three 50-character rows: excerpt from the training corpus (top), stochastic sample at "
        "initialization (middle), stochastic sample after training (bottom). Green/red per-character "
        "coloring marks in-vocabulary vs out-of-vocabulary segments in the generated rows.",
        "Figure 4 compares **what the model actually generates** before and after learning. The top row "
        "is ground-truth stream structure. Before training, characters are essentially unstructured noise "
        "with respect to the lexicon. After training, almost every character lies inside a valid vocabulary "
        "word. The display uses fixed 50-character windows so before/after comparisons are apples-to-apples.",
    ),
    (
        5,
        "05_weights.png",
        "Learned weight matrices after training. Left: input weights $W_{xh}$ (character columns $\\times$ "
        "hidden rows). Right: recurrent weights $W_{hh}$ (hidden $\\times$ hidden).",
        "The raw parameters reveal which letters drive which hidden units and how units recurrently mix. "
        "Input columns show letter-specific tuning; recurrent blocks show long-timescale coupling. With "
        "$h=32$ the matrices are small enough to inspect directly. There is no hand-designed word feature: "
        "any boundary or lexical structure must be implemented through these weights.",
    ),
    (
        6,
        "06_weights_eigenspectra.png",
        "Eigenvalue spectra of the recurrent and cross-weight blocks, summarizing effective memory "
        "timescales and stability of the trained dynamics.",
        "The spectrum of $W_{hh}$ (and related blocks) indicates how many past characters the recurrence "
        "can integrate and whether dynamics are contractive or expansive in different directions. For word "
        "learning, we expect nontrivial structure here: the network must preserve prefix information across "
        "several timesteps without a dedicated counter.",
    ),
    (
        7,
        "07_activation_heatmap.png",
        "Hidden activations over a 50-character analysis window. Rows are hidden units $h_0\\ldots h_{31}$; "
        "columns are timesteps (input characters shown along the bottom).",
        "This is the raw activation trace from which all geometry plots are derived. Each column is "
        "$\\mathbf{h}_t$ after reading one more character. Visual inspection already suggests structure: "
        "activations repeat with similar patterns when the model is at analogous positions inside words, "
        "even when the absolute corpus index differs.",
    ),
    (
        8,
        "08_next_char_prob_sequence.png",
        "Softmax next-character probabilities at every timestep (columns), with the true next character "
        "highlighted. Brighter cells are higher predicted probability.",
        "Behaviorally, the model is a next-character predictor. Where the trie branches are narrow (late in "
        "a word, or after an informative prefix), probability mass concentrates on one or few characters. "
        "At ambiguous early prefixes (`c` could start `cat`; `a` is shared widely), mass spreads. Comparing "
        "to ground truth shows where the trained network is confident vs uncertain.",
    ),
    (
        9,
        "09_activation_by_input_char.png",
        "For each input character, all timesteps where that character was read. Columns are labeled by "
        "**in-word prefix** (e.g. `h`, `ha`, `hat` after a space). Rows are hidden units; columns are "
        "occurrences, optionally clustered by activation similarity.",
        "This panel is the first direct evidence for **prefix-axis organization**. Fix an input letter "
        "such as `a`. Every occurrence is shown, but columns are sorted/labeled by how far into the "
        "current word that `a` appeared. Timesteps with the same prefix produce similar activation "
        "profiles even when they occur at unrelated positions in the corpus. The network encodes "
        "\"where am I inside this word?\" not merely \"what letter did I just see?\"",
    ),
    (
        10,
        "10_activation_clustered_heatmap.png",
        "Hierarchically clustered heatmap of all timesteps $\\times$ hidden units in the analysis window. "
        "Row and column dendrograms group similar timesteps; tick labels are in-word prefixes.",
        "Clustering across the full 50-timestep window reveals blocks of timesteps with near-identical "
        "hidden vectors. Many blocks align with shared prefixes or suffixes (`at`, `et`, etc.). This is "
        "unsupervised structure in $\\mathbf{h}_t$ using only prefix labels for interpretation - the "
        "clustering itself is driven purely by activation similarity.",
    ),
    (
        11,
        "11_embedding_panels_context.png",
        "Four 2D embeddings of the same 50 hidden states: PCA, UMAP, t-SNE, and Isomap. Points are "
        "annotated with in-word prefix labels; colors follow embedding-specific layout.",
        "Because $h=32$, we project $\\mathbf{h}_t$ to the plane for visualization. Different nonlinear "
        "methods stress different aspects (global variance, local neighborhoods, geodesics), but all show "
        "annotated prefixes grouping into coherent regions. PCA is used consistently in subsequent panels "
        "so trajectories and vector fields live in a single linear subspace.",
    ),
    (
        12,
        "12_dfa_and_embedding_pca.png",
        "Left: minimized DFA from Figure 2. Right: PCA of hidden states with point color = DFA state and "
        "text label = in-word prefix. Leader lines connect grouped prefix annotations.",
        "This is the **central figure**. The DFA is the discrete lexical reference; the PCA scatter is "
        "the continuous representation the RNN actually uses. Points with the same DFA color cluster "
        "together even when prefix labels differ in length. Conversely, along a single word, the trajectory "
        "visits multiple DFA states as more letters disambiguate the lexical hypothesis. The geometry "
        "**implements the automaton**: the second organizing axis is not an artifact of coloring.",
    ),
    (
        13,
        "13_next_char_regions_pca.png",
        "PCA plane with 2D-reconstructed hidden states. Left: argmax next-character label in each region. "
        "Right: prediction entropy (nats). Overlaid points carry prefix annotations.",
        "These panels answer: **what would the model predict at each location in hidden space?** Low-entropy "
        "regions are predictable continuations inside words; high-entropy regions sit at trie branch points "
        "where several next characters remain viable. The overlaid real trajectory samples these regions as "
        "it moves through prefixes.",
    ),
    (
        14,
        "14_next_char_prob_panels_pca.png",
        "One panel per vocabulary character, showing $P(\\text{next}=c\\mid\\mathbf{h})$ over the PCA plane "
        "(from softmax on 2D-reconstructed $\\mathbf{h}$).",
        "Decomposing the output layer per character shows how each letter's logit carves a different region "
        "of hidden space. Vowels and consonants that appear in overlapping words (`a`, `t`, `e`, ...) have "
        "complex, interleaved regions - reflecting the competition among `-at`, `-et`, and `-ea` families.",
    ),
    (
        15,
        "15_vector_field_grid_pca.png",
        "Recurrent vector field in PCA coordinates with **no external input**: "
        "$\\mathbf{h}_{t+1}=\\tanh(W_{hh}\\mathbf{h}_t)$, projected to PC1-PC2. Quiver arrows show local flow.",
        "Between explicit character inputs, the hidden state still evolves under $W_{hh}$ alone. The vector "
        "field shows attractor-like structure and drift directions in the PCA plane. This is the intrinsic "
        "dynamics the network would follow if characters stopped arriving - relevant for understanding "
        "transients at word boundaries and spaces.",
    ),
    (
        16,
        "16_word_trajectories_pca.png",
        "PCA trajectories from one space timestep to the next (space-to-space segments) in the 50-character "
        "window. Each word path is colored by word identity; faint background shows optional no-input flow.",
        "This is the trajectory-level view of the **first organizing axis**. Each word is a path through "
        "hidden space starting just after a space. Paths with the same prefix length tend to occupy similar "
        "\"lanes\"; completing a word returns toward a boundary region. The figure makes word segmentation "
        "visible as repeated geometric motifs without ever supervising boundaries.",
    ),
    (
        17,
        "17_state_correlation_clustered.png",
        "Pearson correlation matrix between hidden vectors at all pairs of timesteps, with hierarchical "
        "clustering on rows/columns. Tick labels: in-word prefix; label color = minimized DFA state.",
        "Correlation complements PCA: it measures linear similarity of full 32-dimensional states. Blocks "
        "of high correlation appear when both prefix and DFA state align. Tick colors show that DFA state "
        "often cuts across prefix clusters - two timesteps can share a prefix length but differ in DFA "
        "state if the letters diverge (`ca` vs `ha`).",
    ),
    (
        18,
        "18_state_correlation_by_dfa_state.png",
        "Same correlation matrix, but timesteps are **grouped by DFA state** (all states shown). Diagonal "
        "blocks are within-state correlations; off-diagonal blocks are between-state.",
        "Reordering by DFA state exposes block structure directly. High values on the diagonal mean hidden "
        "states in the same automaton state are similar; lower off-diagonal values mean states representing "
        "different lexical hypotheses are separated. This is the correlation analogue of Figure 12's coloring.",
    ),
    (
        19,
        "19_dfa_state_distance_comparison.png",
        "Pairwise Euclidean distances between hidden vectors (subsampled pairs). Three distributions: "
        "within the same DFA state, between different DFA states, and pairs with the same input character.",
        "The quantitative summary: within-state distances are sharply smaller than between-state distances, "
        "even though same-input-character pairs can be far apart. The DFA partition captures variance in "
        "$\\mathbf{h}_t$ that raw character identity alone cannot. Error bars / overlays show means; "
        "scatter shows individual pairs.",
    ),
]


def build_readme_body() -> str:
    parts: list[str] = []

    parts.append("""# Statistical Word Learning in a Minimal Character-Level RNN

This repository trains a **small vanilla recurrent neural network** on synthetic text streams designed to mimic infant **statistical learning** of words (Saffran, Aslin, & Newport, 1996). Words are sampled uniformly from a finite vocabulary and concatenated; in the `_s` regimes a space is inserted between words. The model receives **no word labels**, **no boundary tokens**, and **no auxiliary losses** - only next-character cross-entropy.

After training, an exhaustive visualization pipeline asks how the hidden state $\\mathbf{h}_t$ relates to two complementary structures:

1. **Position within the current word** - the in-word prefix since the last space (`h`, `ha`, `hat`, or a space token).
2. **State in the minimal vocabulary DFA** - which equivalence class of lexical continuations remains open.

**Central claim:** a generic next-character predictor develops geometry that **factorizes along both axes**. Prefix length answers "how many letters into the current word?"; DFA state answers "which vocabulary items are still possible given overlapping spellings?"

Related work: [creating_transformer (statistical learning)](https://github.com/Raneem-mahajne/creating_transformer/tree/statistical_learning) uses the same synthetic regimes with a minimal transformer. This repo is the **RNN** counterpart (Karpathy char-RNN in NumPy).

---

## Quick start

**Requirements:** Python 3.10+, NumPy, Matplotlib, Seaborn, Pandas, SciPy; optional UMAP for one embedding panel.

```bash
pip install -r requirements.txt
pip install matplotlib seaborn pandas scipy umap-learn
python run_experiments.py --only ten_word_overlap_s
```

`run_experiments.py` trains, visualizes, and syncs README figures automatically for `ten_word_overlap_s`.

| Path | Contents |
|------|----------|
| `experiments/<name>/plots/` | Full plot suite from `visualize.py` |
| `docs/figures/` | Copies used by this README (synced by `scripts/build_readme.py`) |

---

## Repository structure

```
task.py              # Corpus generator (word-sampling regimes)
min-char-rnn.py      # Vanilla RNN, BPTT, Adagrad, metrics
visualize.py         # All analysis figures
vocab_diagrams.py    # Trie + minimal DFA construction
experiment.py        # Per-regime hyperparameters
run_experiments.py   # Batch train + visualize
docs/figures/        # README figure copies
experiments/<name>/  # input.txt, model.npz, plots/
```

---

## Available experiments

| Folder | Words | Notes |
|--------|-------|-------|
| `ten_word_overlap` / `_s` | 10 x 3-letter words | Primary demo; heavy `-at`/`-et` overlap |
| `ten_four_letter_overlap` / `_s` | 10 x 4-letter words | Longer words, more branching |
| `ten_four_letter_overlap_s_dale` | Same 4-letter vocab | Dale's law + ReLU, $h=50$ |
| `six_word_overlap` / `_s` | 6 words | Smaller vocab |
| `twelve_word_overlap` / `_s` | 12 words | Mixed overlap patterns |
| `sixteen_word_overlap` / `_s` | 16 words | Largest default regime |

Each regime has spaced (`_s`) and unspaced variants. Labeling uses explicit spaces when present, otherwise implicit vocabulary word boundaries (with a $\\leq 3$-character fallback).

Default training for main overlap tasks: **50k characters**, **15k steps**, **$h=32$**, **BPTT $=40$**, **50-character visualization window**.

---

## Paper: Learning words with an RNN

The walkthrough below uses **`ten_word_overlap_s`**. All figures are in `docs/figures/`.

> **Abstract.** Infants appear to segment fluent speech into words using only distributional statistics - transitional probabilities between syllables - without explicit boundaries (Saffran, Aslin, & Newport, 1996). We train a 32-unit $\\tanh$ RNN on a ten-word corpus with overlapping trigrams and visualize hidden activations, next-character predictions, PCA embeddings, correlation structure, and recurrent vector fields. Two organizing principles emerge in hidden space: timesteps cluster by **in-word prefix** (distance from the last space), and orthogonally by **minimized DFA state** after that prefix. Pairwise distances are substantially smaller within DFA state than between states, even when the current input character matches. The framework links infant statistical-learning theory to mechanistic RNN interpretability.

---

## 1. Introduction

### 1.1 Statistical learning and word segmentation

Jenny Saffran and colleagues showed that eight-month-old infants can discover word-like units from continuous artificial speech streams. Within putative words, adjacent syllables have high transitional probability (TP); across word boundaries, TP drops. No pauses, stress, or semantic cues are required - only **distributional structure**.

Our synthetic task instantiates the same logic at the character level. The generator repeatedly samples a word uniformly from a small vocabulary and appends its letters (with a space between words in `_s` corpora):

```
... hat cat met rat tea eat cat hat ...
```

The learner sees one long string. The only supervision is: predict the next character. The scientific question is whether $\\mathbf{h}_t$ - the RNN's only memory - implicitly encodes (i) **where you are inside the current word** and (ii) **which vocabulary items remain consistent** with the letters read so far.

### 1.2 Why an RNN?

A vanilla RNN compresses the past into a fixed vector and updates it causally:

$$
\\mathbf{h}_t = \\tanh\\!\\left( W_{xh} \\mathbf{x}_t + W_{hh} \\mathbf{h}_{t-1} + \\mathbf{b}_h \\right)
$$

$$
P(\\text{next} = c \\mid \\mathbf{x}_{\\leq t}) = \\mathrm{softmax}\\!\\left( W_{hy} \\mathbf{h}_t + \\mathbf{b}_y \\right)_c
$$

Here $\\mathbf{x}_t$ is the one-hot encoding of the current character. There is no attention, no stack, and no built-in word counter. If word-like structure appears in $\\mathbf{h}_t$, it is because next-character prediction on overlapping words **demands** it.

This makes the RNN a conservative model of incremental statistical learning: one pass, bounded memory, local supervision.

### 1.3 Two axes of organization (preview)

| Axis | What it encodes | How we label timesteps | Signature figures |
|------|-----------------|------------------------|-------------------|
| **Word-boundary / prefix** | Letter index within the current word | `h`, `ha`, `hat`, space | 9, 10, 16 |
| **DFA state** | Equivalence class of surviving words | Minimized automaton state $q_k$ | 12, 17, 18, 19 |

The first axis is **syntactic position** inside a word (how far from the last boundary). The second is **lexical uncertainty** given shared spellings (`cat` vs `hat` vs `mat`). A successful statistical learner needs both.

---

## 2. Task and vocabulary structure

### 2.1 The `ten_word_overlap` regime

Ten three-letter English words with controlled overlap:

| Group | Words |
|-------|-------|
| `-at` family | cat, hat, mat, rat |
| `-et` family | met, pet, net |
| `-ea` / vowel overlap | ate, eat, tea |

Character set: $\\{a,c,e,h,m,n,p,r,t\\}$ plus space. A model that only tracks the last one or two characters confuses branches that share prefixes or suffixes.

### 2.2 Trie and minimal DFA

We compile the vocabulary into classical finite-state structures **before** training and use them only for **analysis labels** (not as model inputs).""")

    for n, dest, caption, lead in FIGURE_SECTIONS[:2]:
        parts.append(figure_block(n, dest, caption, lead))

    parts.append("""
---

## 3. Model and training

We use Andrej Karpathy's minimal character-level RNN (NumPy, from scratch).

| Hyperparameter | `ten_word_overlap_s` value |
|----------------|----------------------------|
| Hidden units $h$ | 32 |
| Activation | $\\tanh$ |
| Optimizer | Adagrad ($\\eta = 0.1$) |
| BPTT window | 40 characters |
| Training steps | 15,000 |
| Corpus size | 50,000 characters |

Training minimizes sum cross-entropy over each BPTT window. Every 100 steps we also draw stochastic samples and estimate **word error rate** on long rollouts: the fraction of space-delimited tokens not in the vocabulary. This metric asks whether free generation respects the same word units infants extract from streams.

Optional **Dale's law** mode (`--dale` in `ten_four_letter_overlap_s_dale`): fixed excitatory/inhibitory outgoing signs, ReLU activations, softer learning rate.

---

## 4. Results

All panels use the first **50 characters** of the trained corpus unless noted. Tick labels on later figures use **in-word prefix since last space**.

### 4.1 The model learns the stream

We begin by confirming that training succeeds and that generation improves.""")

    for n, dest, caption, lead in FIGURE_SECTIONS[2:4]:
        parts.append(figure_block(n, dest, caption, lead))

    parts.append("""
### 4.2 Learned parameters

Next we inspect the weights directly - the only place lexical structure can be stored.""")

    for n, dest, caption, lead in FIGURE_SECTIONS[4:6]:
        parts.append(figure_block(n, dest, caption, lead))

    parts.append("""
### 4.3 Activations and predictions over time

With parameters fixed, we turn to hidden activations and outputs along the corpus.""")

    for n, dest, caption, lead in FIGURE_SECTIONS[6:10]:
        parts.append(figure_block(n, dest, caption, lead))

    parts.append("""
### 4.4 Hidden-state geometry

Projecting $\\mathbf{h}_t \\in \\mathbb{R}^{32}$ to the plane exposes the two organizing axes visually.""")

    for n, dest, caption, lead in FIGURE_SECTIONS[10:16]:
        parts.append(figure_block(n, dest, caption, lead))

    parts.append("""
### 4.5 Correlation structure: prefix and DFA

Finally we quantify similarity between hidden vectors - within and across DFA states.""")

    for n, dest, caption, lead in FIGURE_SECTIONS[16:19]:
        parts.append(figure_block(n, dest, caption, lead))

    parts.append("""
---

## 5. Discussion

### 5.1 Two organizations, one hidden space

The figures support a two-factor description of the learned representation:

1. **Distance from word boundary (in-word prefix).** As the network reads $h \\to ha \\to hat$, $\\mathbf{h}_t$ moves along paths that depend on how many characters have been consumed since the last space. Figures 9-10 and 16 make this explicit: prefix labels predict similarity across the corpus. This mirrors **positional uncertainty** in infant segmentation - early characters in a word are more ambiguous than later ones.

2. **DFA state (lexical hypothesis class).** After merging equivalent prefixes, each timestep has a well-defined state in the minimal vocabulary automaton. Figure 12 shows these states occupy distinct PCA regions; Figures 18-19 show correlation and distance respect DFA partitions even when the current input character is held constant. This is **competition among overlapping words** made geometric: `ca` could still become `cat`; `at` after `c` is a different state than `at` after `h`.

The axes are not redundant. Prefix length is a scalar progress variable; DFA state is a finite partition of lexical knowledge. The RNN must implement both to minimize loss on overlapping trigrams.

### 5.2 Connection to statistical learning theory

Saffran's learners track transitional probabilities and use troughs at boundaries to segment streams. Our model never sees explicit boundaries in the loss, but spaces in `_s` corpora induce bimodal statistics: high TP within words, lower TP across boundaries. The emergent prefix structure in $\\mathbf{h}_t$ is the network's solution to that problem.

The DFA axis goes further: the RNN tracks **which branch of the lexical tree** it occupies - a discrete state machine embedded in continuous hidden space.

### 5.3 Limitations and extensions

- **Small vocabulary, synthetic data** - ten words is a laboratory setting.
- **PCA is a projection** - 32 dimensions collapsed to 2 for plotting; UMAP/t-SNE panels are qualitative.
- **Vanilla RNN** - no LSTM/GRU; long-range dependencies may be harder than in modern architectures.
- **Single run** - figures are illustrative; multiple seeds would strengthen quantitative claims.
- **Unspaced corpora** - implicit word boundaries via vocabulary segmentation work for labeling; see unspaced experiment folders.

---

## 6. Reproduce

```bash
python run_experiments.py --only ten_word_overlap_s
```

This generates the corpus, trains the RNN, writes all plots under `experiments/ten_word_overlap_s/plots/`, and runs `scripts/build_readme.py` to refresh `docs/figures/`.

---

## 7. References

- Saffran, J. R., Aslin, R. N., & Newport, E. L. (1996). Statistical learning by 8-month-old infants. *Science*, 274(5294), 1926-1928.
- Karpathy, A. [Minimal character-level RNN](https://gist.github.com/karpathy/d4dee566867f8291f086).
- Mahajne, R., et al. [creating_transformer - statistical learning](https://github.com/Raneem-mahajne/creating_transformer/tree/statistical_learning).

---

## License

`min-char-rnn.py` retains the BSD license from the Karpathy gist.
""")

    return "\n\n".join(parts)


def main() -> None:
    sync_figures()
    out = ROOT / "README.md"
    out.write_bytes(build_readme_body().encode("utf-8"))
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    print(f"synced {len(FIGURE_FILES)} figures -> {FIGURES}")


if __name__ == "__main__":
    main()
