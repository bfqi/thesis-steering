# Steering vectors against ROME knowledge edits

Master's thesis experiments: can a steering vector, computed with simple
operations (mean, PCA, projection), reverse ROME knowledge edits in GPT-2 XL?

## Structure

- `steer_utils.py` – shared helpers (data loading, editing, activations, steering hooks, scoring)
- `notebooks/01_counterfact_overview.ipynb` – CounterFact statistics per relation (no GPU)
- `notebooks/02_collect_edits.ipynb` – ROME edits on P103 / P30, saves activation differences; within- vs across-relation similarity
- `results/` – small result files (CSV, figures). Large `.pt` files are not tracked.

## Running on Kaggle

GPU T4 x2, internet on. Each notebook clones EasyEdit and this repository,
installs the requirements, and imports `steer_utils`.

## Conventions

- `hidden_states[i]` is the output of block `i - 1`; ROME edits block 17, so its output is `hidden_states[18]`.
- Steering vector `v = mean(h_original - h_edited)`; steering adds `alpha * v`.
- Differences are taken at the last token of the prompt.
