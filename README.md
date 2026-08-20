# G2I: Generating Intervention Hypotheses using Explainable Explanations on Graphs

G2I reframes counterfactual explanation on graphs as an *intervention design* problem, in two
stages:

1. **Node-level counterfactual generation (CF-Greedy).** A greedy discrete search over node
   features, edge modifications, and neighbor feature perturbations finds the minimal
   perturbation that flips a node's predicted risk, then abstracts it into an interpretable
   clause with threshold-based *neighbor conditions*.
2. **Graph-level intervention design.** Local clauses are aggregated into a Disjunctive Normal
   Form (DNF) policy by greedily maximizing a monotone, approximately submodular coverage
   function under a budget constraint.

---

## Installation

```bash
git clone <this-repository>
cd G2I
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

Install PyTorch first, matched to your platform (https://pytorch.org/get-started/locally/),
then the rest:

```bash
pip install -r requirements.txt
```

Reproducing **Table 2** additionally needs `dgl` and a checkout of the CF² reference
implementation — see [Table 2](#table-2-explanation-quality-on-benchmark-datasets) below.

Tested with Python 3.10, PyTorch 2.x and PyTorch Geometric 2.x. All commands below are run
from the repository root.

---

## Quick start

A single small configuration, all three methods, about a minute on a laptop:

```bash
python run.py --synthetic_type neighbor_feature --n_nodes 100 --total_edges 150 --m_attrs 10 --N 1
```

This writes `results/synthetic_neighbor_feature/node100_attr10_edge150_N1_....json`. Print its
Table 3 row with:

```bash
python scripts/make_table3.py
```

The two notebooks `notebooks/quickstart_neighbor_feature.ipynb` and
`notebooks/quickstart_neighbor_only.ipynb` do the same thing interactively and plot the
coverage curves.

---

## Reproducing the paper

### Table 2: explanation quality on benchmark datasets

Table 2 compares CF-Greedy against GNNExplainer, CF-GNNExplainer and CF² on BA-Shapes,
Tree-Cycles and Mutag₀ using ground-truth explanation motifs.

To keep the comparison fair, CF-Greedy is evaluated **on the baselines' own pre-trained models
and test splits**, taken from the CF² reference implementation. Set it up next to this
repository:

```bash
git clone https://github.com/chrisjtan/gnn_cff.git
```

Follow that repository's README to train the three models, so that
`gnn_cff/log/{Mutagenicity_0,BA_Shapes,Tree_Cycles}_logs/` exist. Then run

```bash
pip install dgl
jupyter notebook notebooks/table2_explanation_benchmarks.ipynb
```

The notebook's final summary cell prints, per dataset, `Pre` (the **Pr%** column of Table 2),
`avg_exp` (the **#exp** column, i.e. the Minimum Information Perturbation) and the elapsed
time. If `gnn_cff` lives elsewhere, point the notebook at it with the `GNN_CFF_DIR`
environment variable.

The **baseline rows** of Table 2 are produced by `gnn_cff`'s and CF-GNNExplainer's own
evaluation scripts, not by this repository.

### Table 3: intervention evaluation on synthetic graphs

Twelve configurations — six `(n_nodes, total_edges)` pairs × two synthetic families:

| Family | Label rule | `--synthetic_type` | `--m_attrs` | Table 3 suffix |
|---|---|---|---|---|
| Neighbor-Feature | diathesis–stress: own attributes **and** neighborhood | `neighbor_feature` | 10 | `-D10` |
| Neighbor-Only | neighborhood composition alone | `neighbor_only` | 6 | `-D6` |

Run the whole sweep:

```bash
bash scripts/run_table3.sh
```

then build the table:

```bash
python scripts/make_table3.py            # prints the table
python scripts/make_table3.py --csv table3.csv --show_std
```

The columns match Table 3 exactly: AUCC under `Rand.` / `Freq.` / `DNF` clause selection,
final `Cov.(DNF)`, and `Time`. `notebooks/results_summary.ipynb` shows the same numbers plus
the underlying coverage-vs-cost curves.

Defaults used by `scripts/run_table3.sh`, overridable by environment variable:

| Variable | Default | Meaning |
|---|---|---|
| `N` | `10` | repetitions per configuration (seeds `42 … 42+N-1`) |
| `BUDGET` | `5.0` | budget `B`; the effective limit is `B · ln(1/ε) = 11.5` |
| `EPSILON` | `0.1` | ε |
| `WORKERS` | `8` | worker processes for the CF and CF² baselines |
| `DEVICE` | `cpu` | `cpu` or `cuda` |

```bash
N=10 BUDGET=5.0 EPSILON=0.1 WORKERS=30 bash scripts/run_table3.sh
```

> **Note on hyperparameters.** The paper states that the budget was selected from
> `{5, 10, 15, 20}` depending on dataset scale, and that CF's β and CF²'s mask thresholds were
> tuned per dataset (β ∈ `{0.01, 0.1, 1}`, thresholds in `[0.2, 0.8]`). The per-configuration
> choices behind the published table were not recorded in the code. The defaults above
> (`B = 5`, `ε = 0.1`, `cf_beta = 1`, `cff_feat_TH = cff_edge_TH = 0.7`, `cff_lam = 100`) are
> the settings that appear in the surviving experiment scripts and result filenames, and they
> reproduce the qualitative pattern of Table 3; individual cells may differ from the published
> values. Every one of these is exposed as a `run.py` flag — see `python run.py --help`.

Runtime warning: the CF and CF² baselines dominate the cost. The 500-node / 3000-edge
configurations take hours each even on a many-core machine. CF-Greedy alone finishes in
seconds; use `--explanation_method cf-greedy` to skip the baselines.

### Figure 3: runtime scalability

Figure 3 measures explanation-generation time only, on an **untrained** (randomly initialized)
GCN, so that model training does not enter the measurement. Generate the command list, run it,
and plot:

```bash
bash scripts/gen_scalability_commands.sh
# executes 24 jobs; run them however your cluster prefers, e.g. sequentially:
cut -d'|' -f2- all_commands_runtime.txt | while read -r c; do eval "$c"; done

python scripts/plot_scalability.py --out figures/scalability.png
```

The sweep spans 100 → 102,400 nodes. CF and CF² exhaust memory beyond roughly 6,400 nodes
(the paper reports >248 GB required), so only CF-Greedy is scheduled at 25,600 and 102,400
nodes — that is why their curves stop early in Figure 3. `scripts/plot_scalability.py` also
prints the mean runtimes it plotted, so the numbers can be checked directly.

The published version of this figure is checked in as
[`figures/scalability_largerfont.png`](figures/scalability_largerfont.png).

### Figure 2: additivity and submodularity validation

```bash
python scripts/submodularity_analysis.py --n_trials 5000 --out figures/submodularity.png
```

For each trial the script samples a correctly predicted at-risk node and two disjoint
intervention sets `A`, `B` (each of cardinality < `k = 5`), and records

- **additivity ratio** `f(A ∪ B) / (f(A) + f(B))`, and
- **modularity ratio** `Σ_{x∈B} [f(A ∪ {x}) − f(A)] / [f(A ∪ B) − f(A)]`,

where `f(S)` is the cumulative risk reduction `p(v) − p(v | S applied)`. It prints summary
statistics (mean, median, fraction in `[0.9, 1.1]`, fraction ≤ 1) alongside the two histograms.

As noted above, the figure in the paper was computed on the Military network; this script runs
the identical analysis on a synthetic graph. Use `--synthetic_type neighbor_only --m_attrs 6`
for the Neighbor-Only family.

---

## Repository layout

```
run.py                       Main experiment driver (train -> explain -> select DNF policy)
run_test_runtime.py          Runtime-only benchmark, skips training (Figure 3)

src/
  model.py                   GCN_Mulin backbone (3-layer JK GCN, optional fixed PCA layer),
                             perturbable variant, CF-GNNExplainer and CF^2 explainer modules
  intervention_design_model.py       CF-Greedy: local counterfactual search, neighbor-condition
                                     abstraction, coverage, and greedy/frequency/random
                                     clause selection (Algorithms 1 and 2)
  intervention_design_model_cf.py    CF-GNNExplainer -> clause conversion
  intervention_design_model_cff.py   CF^2 -> clause conversion
  experiment_runner.py       run_ALL: one trained model, all methods, N repetitions
  parallel_utils.py          Multiprocessing for the CF and CF^2 baselines
  utils.py                   AUCC computation, summary tables, coverage plots

data/
  gen_graph_data.py          Synthetic SBM + homophily graph generator and data splits

scripts/
  run_table3.sh              All twelve Table 3 configurations
  make_table3.py             Aggregate result JSONs into Table 3
  gen_scalability_commands.sh  Emit the Figure 3 runtime sweep
  plot_scalability.py        Build Figure 3 from the sweep output
  submodularity_analysis.py  Figure 2

notebooks/
  quickstart_neighbor_feature.ipynb   Interactive single-configuration run (D10)
  quickstart_neighbor_only.ipynb      Interactive single-configuration run (D6)
  results_summary.ipynb               Table 3 numbers and coverage curves
  table2_explanation_benchmarks.ipynb Table 2 (requires gnn_cff + dgl)

docs/
  methods.md                 Method write-up: architecture, Algorithms 1 and 2
  graph_generation.md        Synthetic graph generation, step by step

figures/
  scalability_largerfont.png Figure 3 as published
```

Results are written under `results/` (git-ignored):
`results/synthetic_neighbor_feature/`, `results/synthetic_neighbor_only/` for `run.py`, and
`results/test_results/` for `run_test_runtime.py`.

---

## Key options

`python run.py --help` lists everything. The ones that matter most:

| Flag | Default | Meaning |
|---|---|---|
| `--synthetic_type` | `neighbor_feature` | Which synthetic label rule to use |
| `--n_nodes` / `--total_edges` / `--m_attrs` | `250` / `300` / `5` | Graph size and feature dimension |
| `--label_percent` | `0.17` | Fraction of nodes labeled at-risk |
| `--N` | `1` | Repetitions; run `i` uses seed `42 + i` |
| `--budget` / `--epsilon` | `10.0` / `0.1` | Budget constraint; limit is `budget · ln(1/ε)` |
| `--explanation_method` | `all` | Any subset of `cf-greedy`, `cf`, `cff` |
| `--cf_greedy_edge_mode` | `edge_features` | `none` = target-node features only; `edges` = also add/remove edges; `edge_features` = also perturb neighbor features |
| `--cf_greedy_max_interventions` | `5` | `K` in Algorithm 1 |
| `--parallel` / `--num_workers` | off | Multiprocessing for CF and CF² (CF-Greedy is single-process and already fast) |
| `--device` | `cpu` | `cpu` or `cuda` |

Model training is deterministic given the seed; `run_ALL` re-seeds NumPy and PyTorch with
`42 + run_id` at the start of every repetition and re-splits the data, so `--N 10` gives ten
independent train/explain cycles on the same generated graph.

---

## Citation

```bibtex
@inproceedings{tian2026g2i,
  title     = {Generating Intervention Hypotheses using Explainable Explanations on Graphs:
               G2I, a Two-Stage Greedy Framework},
  author    = {Tian, Mulin and Srivastava, Ajitesh},
  booktitle = {Proceedings of the 35th ACM International Conference on Information and
               Knowledge Management (CIKM '26)},
  year      = {2026},
  publisher = {ACM},
  address   = {Rome, Italy},
  doi       = {10.1145/3799682.3840645}
}
```

---

## Important: restricted data has been removed

The paper evaluates on two real-world networks — a **Military** peer network and a **Youth**
homeless-youth dataset. **Neither dataset, nor any code or notebook that loads them, is
included in this repository**, because both are governed by IRB agreements that do not permit
redistribution.

Consequently the following results **cannot be reproduced from this repository**:

| Paper artifact | Status |
|---|---|
| Table 3, `Military` and `Youth` rows | Not reproducible (restricted data) |
| Table 4 (unconstrained CF-Greedy clauses, Military) | Not reproducible (restricted data) |
| Table 5 (constrained CF-Greedy clauses, Military) | Not reproducible (restricted data) |
| Section 5.1 discussion of risk factors | Not reproducible (restricted data) |
| Figure 2 (additivity / modularity ratios) | Reproducible **on synthetic graphs only** — see the note in [`scripts/submodularity_analysis.py`](scripts/submodularity_analysis.py); the published figure was computed on the Military network |

Everything else — Table 2, the twelve synthetic rows of Table 3, and Figure 3 — is fully
reproducible with the code and instructions below.

The pipeline itself is dataset-agnostic. To apply it to your own tabular + edge-list data,
write a loader returning a PyTorch Geometric `Data` object (plus optional PCA components,
one-hot `feature_partitions`, and `unchangeable_indices` / `forbidden_indices` for
immutability constraints) and pass it to `src.experiment_runner.run_ALL`; see the `run_ALL`
docstring for the full contract.


## Acknowledgments

This work was supported by the Army Research Office grant W911NF-23-1-0354.

## License

This work is licensed under a Creative Commons Attribution 4.0 International License,
matching the paper.
