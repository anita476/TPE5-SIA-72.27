# TPE5-SIA-72.27

Autoencoders. Fifth practical assignment for Sistemas de Inteligencia Artificial.

Everything is implemented from scratch with NumPy (manual forward/backprop,
Adam/SGD, early stopping, best-model checkpoint, He/Xavier init). The assignment
has two parts:

- **Part 1 — Autoencoder.** Basic AE with a 2-D latent space, generation of new
  characters from the latent space, and a Denoising AE with a noise-robustness
  study, all over the binary font characters in `data/font.h`.
- **Part 2 — Variational Autoencoder (VAE).** A dense VAE (reparametrisation +
  `recon + KL` loss, Kingma & Welling 2014) that **generates new images** by
  sampling the prior `N(0, I)`, trained over four image datasets.

## Autoencoder Types

| Type          | Class                    | Description                                               | Implemented |
| ------------- | ------------------------ | --------------------------------------------------------- | ----------- |
| `simple`      | `SimpleAutoencoder`      | Basic encoder/decoder, MSE or BCE loss                    | yes         |
| `denoising`   | `DenoisingAutoencoder`   | Corrupts the input, reconstructs the clean image          | yes         |
| `variational` | `VariationalAutoencoder` | mu/logvar heads, reparametrisation, KL term; can generate | yes         |

## Entry points

Run every command from the **repository root** so that the `font`/`out`/`data`
paths inside the JSON configs resolve correctly (`scripts/_bootstrap.py` adds
`src/` to the path and anchors relative paths to the root).

### Part 1 — Autoencoder

| Script                            | Purpose                                                                       |
| --------------------------------- | ----------------------------------------------------------------------------- |
| `scripts/main.py`                 | Quick demo: small AE on `data/font.h`, prints reconstructions                 |
| `scripts/plot_dataset.py`         | Grid of the 32 input characters → `output/simple/dataset_grid.png`            |
| `scripts/test_autoencoder.py`     | Train/evaluate the basic AE (multi-seed; `grid` section triggers grid search) |
| `scripts/generate_letter.py`      | Generate new letters from the latent space (interpolation + centroid)         |
| `scripts/seed_similarity.py`      | Cross-seed structural similarity of the 2-D latent space (RSA heatmap)        |
| `scripts/denoising_experiment.py` | Train the DAE and study reconstruction vs noise level                         |
| `scripts/compare_experiment.py`   | Compare AE architectures / optimisers (same budget)                           |
| `scripts/compare_denoising.py`    | Compare noise types / DAE architectures (mode chosen by config)               |
| `scripts/arch_zoom_plots.py`      | Redraw the DAE comparisons with a low-noise zoom inset (no retraining)        |

### Part 2 — VAE

| Script                            | Purpose                                                                     |
| --------------------------------- | --------------------------------------------------------------------------- |
| `scripts/train_vae.py`            | Architecture sweep (latent / depth / lr), multi-seed, + plain-AE comparison |
| `scripts/generate_vae.py`         | Generate from a trained VAE: prior samples, latent traversals, latent grid  |
| `scripts/olivetti_vae.py`         | Single-run VAE on Olivetti faces: train once, emit all plots                |
| `scripts/ae_vs_vae_generation.py` | AE vs VAE on emojis: why a plain AE cannot generate                         |
| `scripts/ae_vs_vae_fashion.py`    | Same argument on Fashion-MNIST                                              |
| `scripts/compare_vae_configs.py`  | BCE vs MSE and ReLU+He vs tanh+Xavier                                       |

---

## Requirements

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `scipy`, `matplotlib`, `seaborn`, `pillow`, `scikit-learn`.
Fashion-MNIST, Olivetti and CelebA are **downloaded automatically** on first use
(Fashion-MNIST IDX files to `data/fashion/`, Olivetti via scikit-learn's cache,
and a small CelebA subset from the public Hugging Face mirror). If a download
fails, the loaders print where to drop the files manually.

---

## Data format

### Font characters (`data/font.h`)

32 characters, each a 7×5 = 35-pixel binary bitmap stored as C row bitmasks
(MSB = leftmost column). Loaded by `utils/font_loader.py` into a `(32, 35)`
float array in `{0, 1}`. This is the Part 1 dataset.

### Emojis (`data/emojis.h`)

48 emoji glyphs at 20×20, each stored both as a B&W row bitmask and as a
`0xRRGGBB` colour grid. `utils/emoji_loader.py` returns a 400-dim B&W vector and
a 1200-dim (20×20×3) colour vector per emoji. The file is generated from system
emoji fonts by `utils/emoji_to_h.py`.

### Fashion-MNIST / Olivetti / CelebA (auto-download)

| Dataset        | Loader                          | Shape                         | Notes                                                    |
| -------------- | ------------------------------- | ----------------------------- | -------------------------------------------------------- |
| Fashion-MNIST  | `utils/fashion_mnist_loader.py` | 28×28 = 784-dim, 10 classes   | IDX `.gz` in `data/fashion/`, subsampled (`--n-samples`) |
| Olivetti faces | `utils/olivetti_loader.py`      | 64×64 = 4096-dim, 40 subjects | 400 images via scikit-learn                              |
| CelebA         | `utils/celeba_loader.py`        | 40×40 grayscale = 1600-dim    | subset fetched from a Hugging Face mirror                |

All loaders return `float32` arrays in `[0, 1]` flattened to the shape the rest
of the pipeline expects.

---

## Conventions

**Multi-seed by default (Part 1 and the VAE sweeps).** Scripts run over **several
seeds** and report **mean ± std**, so claims rest on a sample rather than a lucky
run. Seeds are an explicit list in the config, e.g.
`"seeds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`. The number of **simultaneous** runs
is set with `--workers N` (default `1`, portable; not tied to the host CPU
count). If the config has no `"seeds"`, `--seeds N` is used as a fallback,
generating `[1..N]`.

---

## Library (`src/`)

Importable packages used by the scripts above.

**`src/autoencoders/`**

| Module                      | Contents                                                                                   |
| --------------------------- | ------------------------------------------------------------------------------------------ |
| `Autoencoder.py`            | Base class: training loop, Adam/SGD, early stopping, best-model checkpoint, He/Xavier init |
| `SimpleAutoencoder.py`      | Basic autoencoder (forward/backprop)                                                       |
| `DenoisingAutoencoder.py`   | Denoising variant (corrupts the input)                                                     |
| `VariationalAutoencoder.py` | Dense VAE: mu/logvar heads, reparametrisation, KL term                                     |

**`src/utils/`**

| Module                    | Contents                                                 |
| ------------------------- | -------------------------------------------------------- |
| `activations.py`          | Activation functions and their gradients                 |
| `font_loader.py`          | Parser for `data/font.h`                                 |
| `emoji_loader.py`         | Parser for `data/emojis.h` (B&W + color)                 |
| `emoji_to_h.py`           | Generates `emojis.h` by rendering glyphs with PIL        |
| `fashion_mnist_loader.py` | Fashion-MNIST loader (IDX, auto-download)                |
| `olivetti_loader.py`      | Olivetti loader (via scikit-learn)                       |
| `celeba_loader.py`        | CelebA loader (auto-download from a HF mirror)           |
| `config_loader.py`        | Config loading/validation + grid expansion               |
| `grid_runner.py`          | Parallel grid search + plots + CSV                       |
| `single_run.py`           | One training/evaluation run                              |
| `multiseed.py`            | Runs N seeds and aggregates mean ± std + plots           |
| `comparison.py`           | Comparative study of variants (multi-seed)               |
| `noise.py`                | Noise generators: `gaussian` / `salt_pepper` / `masking` |
| `denoising_eval.py`       | Denoising study per noise level                          |
| `latent_generate.py`      | Generation from the latent space (Part 1)                |
| `plot_style.py`           | Shared plot style for the TP                             |
| `emoji_grid.py`           | Grid of all emojis in `emojis.h`                         |
| `fashion_mnist_grid.py`   | Grid of Fashion-MNIST samples                            |
| `olivetti_grid.py`        | Grid of the 400 Olivetti faces                           |

---

## Part 1 — Autoencoder

### `scripts/test_autoencoder.py`

Trains and evaluates the basic AE over the config's seeds, aggregating mean ±
std. A multi-combination `"grid"` section (see `configs/combination_simple.json`)
switches to a classic hyperparameter grid search instead.

```bash
python scripts/test_autoencoder.py --config configs/default_simple.json --workers 8
```

| Argument        | Type | Default        | Description                                                                      |
| --------------- | ---- | -------------- | -------------------------------------------------------------------------------- |
| `--config`      | str  | `None`         | Path to a JSON config. A multi-combination `"grid"` section triggers grid search |
| `--seeds`       | int  | `10`           | Fallback seed count `[1..N]` when the config has no `"seeds"` list               |
| `--workers`     | int  | `1`            | Simultaneous runs (1 = sequential)                                               |
| `--latent-show` | int  | `6`            | How many seeds to display in the latent-space grid                               |
| `--font`        | str  | `data/font.h`  | Path to the font bitmap file                                                     |
| `--out`         | str  | config's `out` | Output directory                                                                 |

Outputs in `output/simple/`: `loss.png` (mean±std band), `errors.png` (per-char
error with std), `latent_seeds.png` (2-D latent space per seed), `summary.csv`
(passed + error per seed plus summary).

### `scripts/generate_letter.py`

Generates new letters from the latent space: an interpolation between two
characters (intermediate steps are letters that do not exist in the dataset) and
a character decoded from the latent centroid.

```bash
python scripts/generate_letter.py --config configs/default_simple.json --from c --to e --workers 6
```

| Argument    | Type | Default  | Description                                                        |
| ----------- | ---- | -------- | ------------------------------------------------------------------ |
| `--config`  | str  | required | Path to JSON config                                                |
| `--from`    | str  | `c`      | Source character                                                   |
| `--to`      | str  | `e`      | Target character                                                   |
| `--seeds`   | int  | `6`      | Fallback seed count `[1..N]` when the config has no `"seeds"` list |
| `--workers` | int  | `1`      | Simultaneous runs                                                  |

Outputs in `output/simple/`: `latent_interpolation_seeds.png` (clean
single-seed strip of the `from→to` morph, using the representative seed — the
first one, matching `latent_single.png`), `latent_generated_point_seeds.png`
(centroid decode, one per seed) and `latent_grid.png` (full latent-plane decode
for the representative seed).

### `scripts/seed_similarity.py`

Quantifies how much of the 2-D latent **geometry** is shared across independent
seeds (Representational Similarity Analysis). For each seed it builds the matrix
of pairwise distances between characters — invariant to rotation, reflection,
translation and scaling of the plane — then correlates those matrices between
every pair of seeds. The mean off-diagonal correlation summarises the shared
structure (≈0 = geometry dictated by the random init; >0 = partly dictated by
the data).

```bash
python scripts/seed_similarity.py --config configs/chosen_model.json --workers 10
```

| Argument    | Type | Default        | Description                                          |
| ----------- | ---- | -------------- | ---------------------------------------------------- |
| `--config`  | str  | required       | Path to JSON config (must have a 2-D bottleneck)     |
| `--seeds`   | int  | `10`           | Fallback seed count `[1..N]` when no `"seeds"` list  |
| `--workers` | int  | `1`            | Simultaneous runs                                    |
| `--out`     | str  | config's `out` | Output directory                                     |

Outputs: `latent_seed_similarity.png` (N×N heatmap with the mean off-diagonal in
the title) and `latent_seed_similarity.csv` (the matrix + the mean).

### `scripts/denoising_experiment.py`

Trains the DAE by corrupting the input (target = clean image) and evaluates
reconstruction at several noise levels, aggregating mean ± std over seeds.

```bash
python scripts/denoising_experiment.py --config configs/default_denoising.json --workers 8
```

| Argument    | Type | Default  | Description                  |
| ----------- | ---- | -------- | ---------------------------- |
| `--config`  | str  | required | Path to JSON config          |
| `--seeds`   | int  | `10`     | Fallback seed count `[1..N]` |
| `--workers` | int  | `1`      | Simultaneous runs            |

Outputs in `output/denoising/`: `denoising_vs_noise.png` (band), `loss.png`,
`denoising_examples_<level>.png` (representative seed), `denoising_metrics.csv`.

### `scripts/compare_experiment.py`

Compares AE architectures / optimisers under the same budget. Variants are
defined in the config's `"variants"` key; each is trained over all seeds.

```bash
python scripts/compare_experiment.py --config configs/compare_simple.json --workers 6
```

| Argument        | Type | Default  | Description                                        |
| --------------- | ---- | -------- | -------------------------------------------------- |
| `--config`      | str  | required | Path to JSON config (with a `"variants"` list)     |
| `--seeds`       | int  | `10`     | Fallback seed count `[1..N]`                       |
| `--workers`     | int  | `1`      | Simultaneous runs                                  |
| `--latent-show` | int  | `6`      | How many seeds to display in the latent-space grid |

Outputs in `output/compare_simple/`: `compare_loss.png` (bands),
`compare_metrics.png` (bars with error bars), `compare_results.csv` (mean ± std
per variant). Related configs: `simple_optimization_compare.json`,
`simple_activation_compare.json`, `compare_init.json`,
`simple_arquitecture_compare_in_out.json`, …

### `scripts/compare_denoising.py`

Comparative DAE studies; the **mode is selected by the config**.

```bash
python scripts/compare_denoising.py --config configs/default_denoising.json --workers 6
```

| Argument    | Type | Default  | Description                                    |
| ----------- | ---- | -------- | ---------------------------------------------- |
| `--config`  | str  | required | Path to JSON config (its keys select the mode) |
| `--seeds`   | int  | `10`     | Fallback seed count `[1..N]`                   |
| `--workers` | int  | `1`      | Simultaneous runs                              |

| Mode                  | Config key                                   | Output (under `output/denoising/`)                                                            |
| --------------------- | -------------------------------------------- | --------------------------------------------------------------------------------------------- |
| noise types (default) | —                                            | `noise_compare/denoising_noise_comparison.png`, `recon_vs_actual.png`, `fraction_removed.png` |
| cross-robustness      | `cross_robustness`                           | `noise_compare/cross_robustness.png` (train-on-X, test-on-Y 3×3 heatmap)                      |
| qualitative panel     | `qualitative` (+ `qual_chars`, `qual_level`) | `noise_compare/noise_qualitative.png`                                                         |
| architectures         | `arch_variants`                              | `arch_compare/denoising_arch_comparison.png`                                                  |
| training level        | `train_levels`                               | `trainlevel_compare/trainlevel_<type>.png`                                                    |

Configs: `denoising_crossrobust.json`, `denoising_qualitative.json`,
`denoising_arch.json`, `denoising_trainlevel.json`, plus the per-axis sweeps
`denoising_hidden.json`, `denoising_depth.json`, `denoising_activation.json`,
`denoising_batch.json`, `denoising_lr.json`, `denoising_init.json`,
`denoising_simplearch.json`.

### `scripts/arch_zoom_plots.py`

Redraws the DAE architecture comparisons (latent / hidden width / depth /
activation / batch / lr / training level) adding a **zoom inset** over the
low-noise region where the curves separate. Reads the already-generated CSVs — it
does **not** retrain.

```bash
python scripts/arch_zoom_plots.py
python scripts/arch_zoom_plots.py --zoom 0.25
```

| Argument | Type  | Default | Description                                                 |
| -------- | ----- | ------- | ----------------------------------------------------------- |
| `--zoom` | float | `0.2`   | Upper bound of the low-noise zoom inset (x-range `0..zoom`) |

### `scripts/plot_dataset.py`

No arguments. Renders the grid of the 32 input characters.

```bash
python scripts/plot_dataset.py   # → output/simple/dataset_grid.png
```

---

## Part 2 — VAE

The VAE generates **new images** by sampling the prior `N(0, I)` and decoding.
The network layout is given in `layer_dims` with the bottleneck (latent
dimension) in the centre, e.g. `[400, 128, 2, 128, 400]`. Outputs carry a
dataset prefix: `""` (emoji B&W), `color_`, `fashion_`, `olivetti_`, `celeba_`.

### `scripts/train_vae.py`

Runs up to three sweeps over all seeds, then trains a plain AE with the best 2-D
config and compares the two latent spaces side by side. Per-config outputs (loss
curves, reconstructions, **KL-per-dim** to reveal posterior collapse, and the
latent scatter when 2-D) go to `output/<prefix>vae_configs/<arch>/`.

```bash
python scripts/train_vae.py --config configs/vae_sweep.json                 # emoji B&W (default)
python scripts/train_vae.py --color --config configs/vae_sweep_color.json   # emoji color (1200-dim)
python scripts/train_vae.py --dataset fashion  --config configs/vae_sweep_fashion.json
python scripts/train_vae.py --dataset olivetti --config configs/vae_sweep_olivetti.json
python scripts/train_vae.py --dataset celeba   --config configs/vae_sweep_celeba.json
```

| Argument      | Type | Default             | Description                                                            |
| ------------- | ---- | ------------------- | ---------------------------------------------------------------------- |
| `--config`    | str  | per-dataset default | Path to the sweep config JSON                                          |
| `--dataset`   | str  | `emoji`             | `emoji` / `fashion` / `olivetti` / `celeba`                            |
| `--color`     | flag | off                 | Train on color RGB (1200-dim) instead of B&W (400-dim) — emoji only    |
| `--n-samples` | int  | `None`              | Number of samples to load (fashion/celeba only)                        |
| `--sweep`     | str  | `all`               | `all` (latent+arch+lr+AE comparison), or only `latent` / `arch` / `lr` |

Sweeps (config sections): **A** `latent_dim_sweep` (MSE vs L, plus the dual-axis
MSE↓/KL↑ tradeoff plot), **B** `arch_sweep` (MSE vs depth/width at latent=2),
**C** `lr_sweep` (MSE vs lr, log x; optional). Global outputs:
`output/<prefix>vae_latent_dim_sweep.png`, `..._arch_sweep.png`,
`..._vae_arch_tuning_table.txt`, `..._best_vae_config.json`.

### `scripts/generate_vae.py`

Generates from a trained VAE. Run **after** `train_vae.py` — it reuses the
`<prefix>best_vae_config.json` it saved (falling back to sensible per-dataset
defaults if absent).

```bash
python scripts/generate_vae.py --dataset emoji      # or fashion / olivetti / celeba (+ --color)
```

| Argument      | Type | Default | Description                                     |
| ------------- | ---- | ------- | ----------------------------------------------- |
| `--dataset`   | str  | `emoji` | `emoji` / `fashion` / `olivetti` / `celeba`     |
| `--color`     | flag | off     | Use color (RGB 1200-dim) mode — emoji only      |
| `--n-samples` | int  | `None`  | Number of samples to load (fashion/celeba only) |

Outputs (prefixed): `vae_prior_samples.png` (+ binarised B&W version + a map of
where samples land in the 2-D latent), `vae_traversal_annotated_*.png`
(interpolations with the path drawn over the latent space), `vae_latent_grid.png`
(decoded grid; for L>2 it sweeps the two highest-KL dims), and a class/subject
scatter for the non-emoji datasets.

### `scripts/compare_vae_configs.py`

Trains 4 VAE configs (same architecture and seed): BCE/MSE × {ReLU+He,
tanh+Xavier}.

```bash
python scripts/compare_vae_configs.py --dataset emoji    # or --dataset fashion
```

| Argument      | Type | Default | Description                              |
| ------------- | ---- | ------- | ---------------------------------------- |
| `--dataset`   | str  | `emoji` | `emoji` or `fashion`                     |
| `--n-samples` | int  | `None`  | Number of samples to load (fashion only) |

Per-config outputs in `output/vae_compare_<dataset>/<name>/` (loss curves,
reconstructions, latent scatter, latent grid) plus a bar-chart summary and a text
table in `output/`.

### `scripts/olivetti_vae.py`, `ae_vs_vae_generation.py`, `ae_vs_vae_fashion.py`

No arguments — tunables live at the top of each file.

```bash
python scripts/olivetti_vae.py            # single-run Olivetti VAE, all plots (olivetti_ prefix)
python scripts/ae_vs_vae_generation.py    # AE vs VAE on emojis (ae_vs_vae_ prefix)
python scripts/ae_vs_vae_fashion.py       # AE vs VAE on Fashion-MNIST (fashion_ae_vs_vae_ prefix)
```

The two `ae_vs_vae_*` scripts train an AE and a VAE with the **same
architecture** and contrast them: the two latent spaces, decoded random `z` (AE
→ noise, VAE → recognisable images), and a decoded latent grid (AE has
meaningless gaps, VAE morphs smoothly).

---

## Config options

**Common (AE and VAE).**

- `layer_dims`: architecture; the central value is the bottleneck (= latent dim).
- `activation`, `optimizer` (`adam`/`sgd`), `weight_init` (`"he"`/`"xavier"`),
  `lr`, `batch_size`, `epochs`, `seed`.
- `patience` / `min_delta`: early stopping (omit `patience` to disable).
- `seeds`: list of seeds for the multi-seed study, e.g. `[1, 2, …, 10]`.

**Part 1 (basic AE and denoising).**

- `loss`: output loss, `"mse"` or `"bce"`. With the sigmoid output, `bce` pushes
  pixels to 0/1 and reaches error-free reconstruction on 100% of seeds; `mse`
  only ~40% (some letters keep borderline pixels). Both the loss curve and the
  best-model checkpoint use the chosen loss. Default: `bce` in
  `default_simple.json`, `mse` in `default_denoising.json`.
- `threshold`, `max_errors`: binarisation and pixel tolerance.
- Denoising: `noise_type` (`gaussian`/`salt_pepper`/`masking`), `noise_level`
  (training noise) and `noise_levels` (levels to evaluate).

**Part 2 (VAE).** Sweep configs have `shared` (common hyperparameters), `seeds`,
and one or more of `latent_dim_sweep` / `arch_sweep` / `lr_sweep`, each with a
`configs` list of `layer_dims` overrides. `recon_loss` is the ELBO
reconstruction term (`"mse"` or `"bce"`); `log_every` sets the logging cadence.

---

## Config index

Every file under `configs/`, grouped by the script that consumes it.

**Part 1 — basic AE** (`test_autoencoder.py`, `generate_letter.py`, `seed_similarity.py`):

| Config                    | Role                                                              |
| ------------------------- | ---------------------------------------------------------------- |
| `default_simple.json`     | Default basic-AE run (BCE)                                       |
| `chosen_model.json`       | Final chosen architecture `[35,16,8,2,8,16,35]` (BCE, He, Adam) |
| `combination_simple.json` | `grid` section → hyperparameter grid search                     |

**Part 1 — AE comparison** (`compare_experiment.py`, `"variants"` configs):

| Config                                    | Compares                                    |
| ----------------------------------------- | ------------------------------------------- |
| `compare_simple.json`                     | Reference architecture/optimiser comparison |
| `compare_init.json`                       | Weight initialisers                         |
| `simple_optimization_compare.json`        | Optimisers (Adam vs SGD)                    |
| `simple_activation_compare.json`          | Activations (tanh vs logistic)              |
| `simple_activation_compare_full.json`     | Activations, longer schedule (MSE)          |
| `simple_activation_compare_full_bce.json` | Activations, longer schedule (BCE)          |
| `simple_bce.json`                         | BCE-loss variant set                        |
| `simple_arquitecture_compare_in_out.json` | Architecture, encoder→bottleneck taper      |
| `simple_arquitecture_compare_out_in.json` | Architecture, bottleneck→decoder taper      |

**Part 1 — denoising** (`denoising_experiment.py` / `compare_denoising.py`):

| Config                       | Role                                        |
| ---------------------------- | ------------------------------------------- |
| `default_denoising.json`     | Default DAE run / noise-types comparison    |
| `denoising_crossrobust.json` | Cross-robustness (train-on-X, test-on-Y)    |
| `denoising_qualitative.json` | Qualitative clean/noisy/reconstructed panel |
| `denoising_arch.json`        | Bottleneck-width sweep                      |
| `denoising_hidden.json`      | Hidden-layer width sweep                    |
| `denoising_depth.json`       | Depth sweep                                 |
| `denoising_activation.json`  | Activation sweep                            |
| `denoising_batch.json`       | Batch-size sweep                            |
| `denoising_lr.json`          | Learning-rate sweep                         |
| `denoising_init.json`        | Weight-init sweep                           |
| `denoising_simplearch.json`  | Minimal-architecture variant                |
| `denoising_trainlevel.json`  | Training-noise-level sweep                  |

**Part 2 — VAE** (`train_vae.py`):

| Config                    | Dataset                 |
| ------------------------- | ----------------------- |
| `vae_sweep.json`          | Emoji (B&W, 400-dim)    |
| `vae_sweep_color.json`    | Emoji (color, 1200-dim) |
| `vae_sweep_fashion.json`  | Fashion-MNIST           |
| `vae_sweep_olivetti.json` | Olivetti faces          |
| `vae_sweep_celeba.json`   | CelebA                  |
