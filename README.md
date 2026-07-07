# SLD Resurrection

Modern reanalysis tooling for the **SLAC Large Detector** (SLD) experiment,
plus cross-experiment OmniLearned foundation-model studies on $e^+e^-$ /
$ep$ / $pp$ datasets. This repository accompanies the paper
*An AI-ready, Polarized Electron-Positron Collision Dataset*
([arXiv:2606.00224](https://arxiv.org/abs/2606.00224)); the demo
notebooks reproduce its physics-validation and measurement baselines.

The underlying release -- approximately 660,000 reconstructed $Z$-pole
events ($\sqrt{s} \approx 91.2$ GeV) recorded with a highly polarized
electron beam during the 1996--1998 SLD runs -- is hosted on
[Zenodo](https://zenodo.org/records/21199778), translated from the
legacy Jazelle binary format by the open-source
[`jazelle`](https://github.com/HEP-KE/jazelle_reader) reader.

The codebase supports a full analysis pipeline:

1. Read SLD parquet shards (translated from the legacy Jazelle binary
   format).
2. Apply selection presets that reproduce published SLD analyses
   ($A_{LR}$, $\alpha_s$, $R_b$, $A_b$/$A_c$, leptonic asymmetries).
3. Convert events to OmniLearned-compatible point clouds via three
   coordinate-mapping strategies (super-jet, hemisphere, boosted-frame).
   Per-particle features are $(\Delta\eta,\ \Delta\phi,\ \log p_T,\ \log E)$
   relative to the strategy's reference axis, $p_T$-sorted and zero-padded
   to a fixed particle count.
4. Run pretrained OmniLearned checkpoints to extract embeddings or
   classifier softmax probabilities.
5. Reduce embeddings to 2D via t-SNE / UMAP and visualise alongside
   simulated ALEPH, H1, and JetClass1 reference samples (SLD is the
   only reconstructed-detector-data sample in the comparison).

The SLD data analysis is documented in three staged notebooks, stored
**with executed outputs** so the cutflows, validation plots, and
measurement results are viewable directly on GitHub:

| Notebook | Contents |
|---|---|
| [`demos/SLD_01_DataPreparation.ipynb`](demos/SLD_01_DataPreparation.ipynb) | Published selections and cutflows, Durham jet clustering, OmniLearned inputs (stages 1–3) |
| [`demos/SLD_02_ValidationPlots.ipynb`](demos/SLD_02_ValidationPlots.ipynb) | Kinematic validation: reconstructed $m_Z$, event shapes, $\cos\theta_T$ by beam helicity |
| [`demos/SLD_03_Measurements.ipynb`](demos/SLD_03_Measurements.ipynb) | $A_{LR}$ and leptonic coupling asymmetries, combined into $\sin^2\theta_W^{\mathrm{eff}}$ |

For SLD, stages 2–5 are also exposed through the `sld-resurrect` CLI
(the SLD path applies the default hadronic selection before building
point clouds), plus a visualisation notebook
[`demos/OmniLearn_Embeddings.ipynb`](demos/OmniLearn_Embeddings.ipynb).

---

## Installation

Requires **Python 3.10+**.

```bash
git clone https://github.com/HEP-KE/sld-resurrect.git
cd sld-resurrect
pip install -e .
```

The base install is enough to run the three SLD data-analysis notebooks.
Optional dependency groups cover the OmniLearned pipeline:

```bash
# CPU (or CUDA) inference + classifier; needed for `sld-resurrect inference`
pip install -e ".[ml]"

# scikit-learn / openTSNE / umap-learn CPU stack; needed for
# `sld-resurrect reduce-embeddings` on CPU
pip install -e ".[reduce]"

# [ml] + [reduce] at once
pip install -e ".[all]"

# [ml] plus scikit-learn, for GPU nodes where the reduction runs on cuML
# (cuML itself still installs separately -- see below)
pip install -e ".[ml-gpu]"

# Development tools (ruff, mypy, pytest, jupyter)
pip install -e ".[dev]"
```

### GPU acceleration via cuML

With `--device cuda` (the default) and a CUDA-enabled torch,
`sld-resurrect reduce-embeddings` runs t-SNE / UMAP on cuML. cuML lives
on the RAPIDS index, not on PyPI, so no extra (including `[ml-gpu]`)
installs it automatically:

```bash
# CUDA 12 example -- pin the CUDA major to whatever your driver supports
pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cuml-cu12==<version>" \
    "cudf-cu12==<version>"
```

See <https://docs.rapids.ai/install> for current version pins. The
automatic CPU fallback triggers only when torch is missing or reports no
CUDA device; on a CUDA machine **without cuML installed, the GPU path
fails** rather than falling back -- either install cuML or pass
`--device cpu` to use the `[reduce]` CPU stack.

---

## Environment variables

Two environment variables control the on-disk layout:

| Variable | Default | What it controls |
|---|---|---|
| `SLD_BASE` | `./sld` | Root of all dataset I/O. Subdirectories `datasets/minidst_translated/parquet` (raw shards), `datasets/minidst_processed` (selected events), `omnilearned/{inputs,embeddings,predictions,reduced}`, and `analysis/{plots,measurements}` hang off this. |
| `OMNILEARN_CHECKPOINT_DIR` | `./checkpoints/omnilearned` | Where the pretrained `.pt` files live. |

Practical notes:

* The CLI and the demo notebooks share the same `./sld` fallback, so
  with `SLD_BASE` unset both operate on the same tree relative to where
  they run. Still, **export `SLD_BASE` as an absolute path** when
  working from more than one directory.
* Defaults are resolved against the current working directory when
  `sld_resurrect.paths` is **first imported** and are then fixed for the
  session -- export the variables before launching Python or Jupyter,
  and run from the repository root if relying on the defaults.
* CLI overrides: `download-dataset` takes `--output-dir`;
  `download-checkpoints` and `inference` take `--checkpoint-dir`;
  `reduce-embeddings` takes `--embedding-dir` and `--output-dir`.
  `process-dataset` takes its input and output locations as positional
  arguments, so the env var does not affect it.

---

## Event-selection API

The selection layer reproduces published SLD selections as named presets.
The typical entry point:

```python
from sld_resurrect.kinematics import build_particles
from sld_resurrect.selector import EventSelector

particles = build_particles(data)          # PHPSUM banks -> Lorentz vectors
selector = EventSelector.from_preset("hadronic_default", data, particles)
selector.print_cutflow()
selected = data[selector.mask()]
```

* `selector_presets.PRESETS` registers the presets by name:
  `hadronic_default` (= `alr_2000`), `alr_1994`, `alphas_1995`,
  `rb_1998`, `abc_2005`, `leptonic_default`, `leptonic_{ee,mumu,tautau}`,
  and the legacy `leptonic_1997_{ee,mumu,tautau}`.
* Each preset pairs an event-level cut list with a paper-specific
  `track_quality.TrackQualityCuts` model ($p_T$, polar-angle, and
  impact-parameter requirements measured against the per-event IP);
  `with_overrides()` produces variations for systematics studies.
* `event_view.EventView` computes and memoises the event-level
  observables the cuts consume (thrust axes, hemisphere multiplicities /
  net charges / invariant masses, LAC energies, ...), and `kinematics`
  holds the observable toolkit (thrust, oblateness, sphericity,
  aplanarity, $C$-parameter, heavy jet mass, hemisphere splitting).

---

## CLI overview

A single `sld-resurrect` console script dispatches five subcommands:

```text
sld-resurrect <command> [options]

  download-dataset       Fetch and unpack the released SLD parquet dataset
  download-checkpoints   Fetch pretrained OmniLearned .pt files
  process-dataset        Convert raw experimental data to OmniLearned point clouds
  inference              Run an OmniLearned checkpoint on a point cloud
  reduce-embeddings      Run t-SNE or UMAP on saved OmniLearned embeddings
```

`--help` works at every level (`sld-resurrect --help`,
`sld-resurrect process-dataset --help`,
`sld-resurrect process-dataset sld --help`).

### 1. Download the dataset

```bash
# ~4.9 GB zip from Zenodo, unpacked into $SLD_BASE/datasets/minidst_translated/parquet
sld-resurrect download-dataset

# Custom location
sld-resurrect download-dataset --output-dir /path/to/parquet
```

The 68 parquet shards land where notebook 01 reads them. The zip is
removed after successful extraction (pass `--keep-zip` to retain it),
an interrupted download resumes where it stopped, and a directory that
already holds shards is skipped unless `--overwrite` is given.

### 2. Download checkpoints

```bash
# All three sizes (s, m, l). Cached files are skipped on size match.
sld-resurrect download-checkpoints

# Just one size, custom directory
sld-resurrect download-checkpoints --sizes m --checkpoint-dir ./my-ckpts
```

The files land as `best_model_pretrain_{s,m,l}.pt`, roughly 100–300 MB
each.

### 3. Convert raw data to OmniLearned input

Four sub-subcommands, one per experiment. The non-SLD ones write a single
HDF5 file containing the `(n_events, max_particles, 4)` point cloud under
the dataset key `data`:

```bash
# JetClass1 (pp, ROOT input, pre-clustered jets)
sld-resurrect process-dataset jetclass1 \
    /path/to/JetClass/ZToQQ_*.root \
    "$SLD_BASE/omnilearned/inputs/omnilearned_input_jetclass1_ZToQQ.h5" \
    --max-events 100000 --max-particles 128

# H1 DIS (ep, HDF5 input)
sld-resurrect process-dataset h1 \
    /path/to/H1/Django_Eminus06.h5 \
    "$SLD_BASE/omnilearned/inputs/omnilearned_input_h1_Django_Eminus06.h5"

# ALEPH (ee, ROOT input; --strategy is required)
sld-resurrect process-dataset aleph \
    /path/to/aleph/LEP1MCZQQ94YMCE1994.root \
    "$SLD_BASE/omnilearned/inputs/omnilearned_input_aleph_LEP1MCZQQ94YMCE1994_superjet.h5" \
    --strategy superjet
```

For ALEPH the `hemisphere` strategy writes **two** files
(`<stem>_leading.h5` and `<stem>_subleading.h5`) at the path you specify.

The SLD subcommand first applies the default hadronic event selection --
the `hadronic_default` preset, i.e. the 2000 $A_{LR}$ selection -- then
performs the 2-jet clustering and emits OmniLearned input files for
whichever strategies you ask for (default: all three):

```bash
# All three strategies (default)
sld-resurrect process-dataset sld \
    "$SLD_BASE/datasets/minidst_translated/parquet" \
    "$SLD_BASE/omnilearned/inputs/" \
    --pattern "*.parquet"

# Only the super-jet view
sld-resurrect process-dataset sld \
    "$SLD_BASE/datasets/minidst_translated/parquet" \
    "$SLD_BASE/omnilearned/inputs/" \
    --strategies superjet
```

Outputs are named `omnilearned_input_sld_<strategy>.h5` (override the
prefix with `--name-prefix`); `--strategies hemisphere` triggers the
Durham 2-jet clustering and emits two files
(`*_hemisphere_leading.h5`, `*_hemisphere_subleading.h5`), while the
other two strategies only need the inclusive particle list. `--max-events`
defaults to `-1` (all events) and `--max-particles` to `128`.

For a custom selection (different cut values, an alternative preset,
etc.), use
[`demos/SLD_01_DataPreparation.ipynb`](demos/SLD_01_DataPreparation.ipynb)
or the event-selection API directly rather than the CLI.

### 4. OmniLearned inference

Two-stage workflow: first extract body **embeddings**, then optionally
run the **classifier** head on those embeddings (a softmax over the 210
OmniLearned pre-training classes):

```bash
# Stage 1 -- single-process embedding extraction (GPU, or CPU if no CUDA)
sld-resurrect inference \
    "$SLD_BASE/omnilearned/inputs/omnilearned_input_sld_superjet.h5" \
    "$SLD_BASE/omnilearned/embeddings/omnilearned_embedding_s_sld_superjet.h5" \
    --size s --task embed --batch-size 128

# Stage 1 -- multi-GPU via torchrun (4 GPUs, full statistics)
torchrun --nproc_per_node=4 -m sld_resurrect.cli._main inference \
    "$SLD_BASE/omnilearned/inputs/omnilearned_input_sld_superjet.h5" \
    "$SLD_BASE/omnilearned/embeddings/omnilearned_embedding_s_sld_superjet.h5" \
    --size s --task embed --batch-size 256 --distributed --max-events -1

# Stage 2 -- run the classifier on the embeddings
sld-resurrect inference \
    "$SLD_BASE/omnilearned/embeddings/omnilearned_embedding_s_sld_superjet.h5" \
    "$SLD_BASE/omnilearned/predictions/omnilearned_prediction_s_sld_superjet.h5" \
    --size s --task classify
```

Two flags to keep in mind: `--max-events` defaults to `10000` **even in
distributed mode** -- pass `-1` for a full-statistics run -- and an
existing output file is skipped (with a notice) unless `--overwrite` is
given.

### 5. Dimensionality reduction

`reduce-embeddings` finds the available datasets by inspecting the
embedding directory. Each embedding file is named
`omnilearned_embedding_{size}_{dataset}.h5`, and the trailing `{dataset}`
portion is the dataset name you pass on the command line:

```bash
# What's available for size 'm'? (--method is required even for --list)
sld-resurrect reduce-embeddings --method tsne --size m --list

# Reduce all available datasets with t-SNE on GPU
sld-resurrect reduce-embeddings --method tsne --size m --device cuda

# Just two datasets, with UMAP on CPU
sld-resurrect reduce-embeddings sld_superjet jetclass1_ZToQQ \
    --method umap --size m --device cpu
```

Embeddings are mean-pooled over the token axis, standard-scaled, and
PCA-reduced to 50 dimensions before the manifold step (tunable via
`--no-scale` / `--pca-components`); `--max-events` caps each dataset at
`3000` events by default. Output is a single HDF5 file
`$SLD_BASE/omnilearned/reduced/reduced_{tsne,umap}_{s,m,l}.h5` with one
top-level dataset per input dataset name, holding the 2D coordinates.

---

## End-to-end example

A short walkthrough of the full pipeline on a small subset of the data:

```bash
export SLD_BASE=$PWD/sld
export OMNILEARN_CHECKPOINT_DIR=$PWD/checkpoints/omnilearned

# 1. Fetch the dataset (~4.9 GB) from Zenodo
sld-resurrect download-dataset

# 2. Get the small model (~100 MB)
sld-resurrect download-checkpoints --sizes s

# 3. Select + convert SLD parquet shards (limited to a small subset for speed)
sld-resurrect process-dataset sld \
    "$SLD_BASE/datasets/minidst_translated/parquet" \
    "$SLD_BASE/omnilearned/inputs/" \
    --max-events 50000

# 4. Embed the SLD super-jet point cloud (uses default --max-events 10000)
sld-resurrect inference \
    "$SLD_BASE/omnilearned/inputs/omnilearned_input_sld_superjet.h5" \
    "$SLD_BASE/omnilearned/embeddings/omnilearned_embedding_s_sld_superjet.h5" \
    -s s -t embed

# 5. Reduce with t-SNE
sld-resurrect reduce-embeddings sld_superjet \
    --method tsne --size s --device cpu --max-events 1000
```

The reduced output then feeds the visualisation notebook
[`demos/OmniLearn_Embeddings.ipynb`](demos/OmniLearn_Embeddings.ipynb).

---

## References

The SLD publications reproduced (or used as reference) in this codebase;
each of the first seven corresponds to a selection preset in
`selector_presets`:

| Year | Reference | Measurement |
|---|---|---|
| 1994 | [hep-ex/9404001](https://arxiv.org/abs/hep-ex/9404001) | First $A_{LR}$ |
| 1995 | [hep-ex/9501003](https://arxiv.org/abs/hep-ex/9501003) | $\alpha_s(M_Z^2)$ from event shapes |
| 1997 | [hep-ex/9704012](https://arxiv.org/abs/hep-ex/9704012) | Leptonic coupling asymmetries |
| 1998 | [hep-ex/9708015](https://arxiv.org/abs/hep-ex/9708015) | $R_b$ via vertex mass |
| 2000 | [hep-ex/0004026](https://arxiv.org/abs/hep-ex/0004026) | High-precision $A_{LR}$ |
| 2001 | [hep-ex/0010015](https://arxiv.org/abs/hep-ex/0010015) | Leptonic coupling asymmetries (final) |
| 2005 | [hep-ex/0410042](https://arxiv.org/abs/hep-ex/0410042) | $A_b$, $A_c$ |
| 2006 | [hep-ex/0509008](https://arxiv.org/abs/hep-ex/0509008) | Precision EW on the $Z$ resonance (background reference) |

The released dataset and translation toolkit:

* **SLD dataset (Zenodo)** -- <https://zenodo.org/records/21199778>
* **`jazelle` reader** -- <https://github.com/HEP-KE/jazelle_reader>

The OmniLearned foundation model:

* OmniLearned paper: [arXiv:2510.24066](https://arxiv.org/abs/2510.24066)
* Pretrained checkpoints: <https://portal.nersc.gov/cfs/dasrepo/omnilearned/checkpoints/>

The reference experimental datasets used for the cross-experiment
comparison:

* **JetClass1** -- <https://zenodo.org/records/6619768>
* **H1 DIS Open Data** -- <https://portal.nersc.gov/cfs/dasrepo/omnilearned/h1/>
* **ALEPH MC Z-pole sample** -- Not Available Yet

---

## Citation

If you use this code or the accompanying dataset, please cite:

```bibtex
@article{Cheng:2026zvl,
    author = "Cheng, Chi Lung and Corrodi, Simon and Hobbs, T. J. and Mete, Alaettin Serhan and Nachman, Benjamin",
    title = "{An AI-ready, Polarized Electron-Positron Collision Dataset}",
    eprint = "2606.00224",
    archivePrefix = "arXiv",
    primaryClass = "hep-ex",
    month = "5",
    year = "2026"
}
```
