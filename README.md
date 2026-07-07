# SLD Resurrection

Modern reanalysis tooling for the **SLAC Large Detector** (SLD) experiment,
plus cross-experiment OmniLearn foundation-model studies on $e^+e^-$ /
$ep$ / $pp$ datasets.

The codebase supports a full analysis pipeline:

1. Read SLD parquet shards (translated from the legacy jazelle format).
2. Apply event-shape and selection presets that reproduce published SLD
   analyses ($A_{LR}$, $\alpha_s$, $R_b$, leptonic asymmetries).
3. Convert events to OmniLearn-compatible point clouds via three
   coordinate-mapping strategies (super-jet, hemisphere, boosted-frame).
4. Run pretrained OmniLearn checkpoints to extract embeddings or
   classifier softmax probabilities.
5. Reduce embeddings to 2D via t-SNE / UMAP and visualise alongside
   ALEPH, H1, and JetClass1 reference samples.

The SLD data analysis is documented in three staged notebooks:
[`demos/SLD_01_DataPreparation.ipynb`](demos/SLD_01_DataPreparation.ipynb)
(stages 1-3: selections, jet clustering, OmniLearn inputs),
[`demos/SLD_02_ValidationPlots.ipynb`](demos/SLD_02_ValidationPlots.ipynb)
(kinematic validation), and
[`demos/SLD_03_Measurements.ipynb`](demos/SLD_03_Measurements.ipynb)
(asymmetry measurements and the $\sin^2\theta_W^{\mathrm{eff}}$ fit).
Stages 3-5 are also exposed as a single `sld-resurrect` CLI plus a
visualisation notebook
[`demos/OmniLearn_Embeddings.ipynb`](demos/OmniLearn_Embeddings.ipynb).

---

## Installation

Requires **Python 3.10+**.

```bash
git clone https://github.com/AlkaidCheng/sld_resurrect.git
cd sld_resurrect
pip install -e .
```

The base install is enough to run the SLD data-analysis notebooks. Two
optional dependency groups cover the OmniLearn pipeline:

```bash
# CPU (or CUDA) inference + classifier; needed for `sld-resurrect inference`
pip install -e ".[ml]"

# t-SNE / UMAP CPU implementations; needed for `sld-resurrect reduce-embeddings`
pip install -e ".[reduce]"

# Both at once
pip install -e ".[all]"

# Development tools (ruff, mypy, pytest, jupyter)
pip install -e ".[dev]"
```

### GPU acceleration via cuML

The reduction step (`sld-resurrect reduce-embeddings`) supports cuML for
GPU-accelerated t-SNE / UMAP. cuML lives on the RAPIDS index, not on PyPI,
so it cannot be installed with a plain `pip install`. The supported route
is:

```bash
# CUDA 12 example -- pin the CUDA major to whatever your driver supports
pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cuml-cu12==<version>" \
    "cudf-cu12==<version>"
```

See <https://docs.rapids.ai/install> for current version pins. If cuML is
unavailable at runtime, `reduce-embeddings` automatically falls back to
the CPU implementations (openTSNE / umap-learn) installed via the
`[reduce]` extra.

---

## Environment variables

Two environment variables control the on-disk layout. They are
**optional** -- the package falls back to project-relative defaults so a
fresh clone runs out of the box.

| Variable | Default | What it controls |
|---|---|---|
| `SLD_BASE` | `./data` | Root of all dataset I/O. Subdirectories `datasets/`, `omnilearned/{inputs,embeddings,predictions,reduced}`, `analysis/{plots,measurements}` hang off this. |
| `OMNILEARN_CHECKPOINT_DIR` | `./checkpoints/omnilearned` | Where the pretrained `.pt` files live. |

Every CLI subcommand also takes the relevant directory as an explicit
flag, which overrides the env var.

---

## CLI overview

A single `sld-resurrect` console script dispatches four pipeline stages:

```text
sld-resurrect <command> [options]

  download-checkpoints   Fetch pretrained OmniLearn .pt files
  process-dataset        Convert raw experimental data to OmniLearn point clouds
  inference              Run an OmniLearn checkpoint on a point cloud
  reduce-embeddings      Run t-SNE or UMAP on saved OmniLearn embeddings
```

`--help` works at every level (`sld-resurrect --help`,
`sld-resurrect process-dataset --help`,
`sld-resurrect process-dataset sld --help`).

### 1. Download checkpoints

```bash
# All three sizes (s, m, l). Cached files are skipped on size match.
sld-resurrect download-checkpoints

# Just one size, custom directory
sld-resurrect download-checkpoints --sizes m --checkpoint-dir ./my-ckpts
```

### 2. Convert raw data to OmniLearn input

Four sub-subcommands, one per experiment. The non-SLD ones write a single
HDF5 file containing the `(n_events, max_particles, 4)` point cloud:

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

# ALEPH (ee, ROOT input, choose strategy)
sld-resurrect process-dataset aleph \
    /path/to/aleph/LEP1MCZQQ94YMCE1994.root \
    "$SLD_BASE/omnilearned/inputs/omnilearned_input_aleph_LEP1MCZQQ94YMCE1994_superjet.h5" \
    --strategy superjet
```

For ALEPH the `hemisphere` strategy writes **two** files
(`<stem>_leading.h5` and `<stem>_subleading.h5`) at the path you specify.

The SLD subcommand runs the default hadronic event selection and 2-jet
clustering, then emits OmniLearn input files for whichever strategies you
ask for (default: all three):

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

`--strategies hemisphere` triggers the Durham 2-jet clustering and
emits two files (`*_hemisphere_leading.h5`, `*_hemisphere_subleading.h5`);
the other two strategies only need the inclusive particle list.

For a custom hadronic selection (different cut values, an alternative
preset, etc.), use the data-analysis notebook directly rather than the CLI.

### 3. OmniLearn inference

Two-stage workflow: first extract body **embeddings**, then optionally
run the **classifier** head on those embeddings:

```bash
# Stage 1 -- single-GPU embedding extraction
sld-resurrect inference \
    "$SLD_BASE/omnilearned/inputs/omnilearned_input_sld_superjet.h5" \
    "$SLD_BASE/omnilearned/embeddings/omnilearned_embedding_s_sld_superjet.h5" \
    --size s --task embed --batch-size 128

# Stage 1 -- multi-GPU via torchrun (4 GPUs)
torchrun --nproc_per_node=4 -m sld_resurrect.cli._main inference \
    "$SLD_BASE/omnilearned/inputs/omnilearned_input_sld_superjet.h5" \
    "$SLD_BASE/omnilearned/embeddings/omnilearned_embedding_s_sld_superjet.h5" \
    --size s --task embed --batch-size 256 --distributed

# Stage 2 -- run classifier on the embeddings (always single-GPU)
sld-resurrect inference \
    "$SLD_BASE/omnilearned/embeddings/omnilearned_embedding_s_sld_superjet.h5" \
    "$SLD_BASE/omnilearned/predictions/omnilearned_prediction_s_sld_superjet.h5" \
    --size s --task classify
```

`--max-events` defaults to `10000`; pass `-1` for a full-statistics run.

### 4. Dimensionality reduction

`reduce-embeddings` finds the available datasets by inspecting the
embedding directory. Each embedding file is named
`omnilearned_embedding_{size}_{dataset}.h5`, and the trailing `{dataset}`
portion is the dataset name you pass on the command line:

```bash
# What's available for size 'm'?
sld-resurrect reduce-embeddings --size m --list

# Reduce all available datasets with t-SNE on GPU
sld-resurrect reduce-embeddings --method tsne --size m --device cuda

# Just two datasets, with UMAP on CPU
sld-resurrect reduce-embeddings sld_superjet jetclass1_ZToQQ \
    --method umap --size m --device cpu
```

Output is a single HDF5 file
`$SLD_BASE/omnilearned/reduced/reduced_{tsne,umap}_{s,m,l}.h5` with one
top-level dataset per input dataset name, holding the 2D coordinates.

---

## End-to-end example

A short walkthrough of all four stages on a small subset of the data:

```bash
export SLD_BASE=$PWD/data
export OMNILEARN_CHECKPOINT_DIR=$PWD/checkpoints/omnilearned

# 1. Get the small model (~100 MB)
sld-resurrect download-checkpoints --sizes s

# 2. Convert SLD parquet shards (limited to a small subset for speed)
sld-resurrect process-dataset sld \
    "$SLD_BASE/datasets/minidst_translated/parquet" \
    "$SLD_BASE/omnilearned/inputs/" \
    --max-events 50000

# 3. Embed the SLD super-jet point cloud (uses default --max-events 10000)
sld-resurrect inference \
    "$SLD_BASE/omnilearned/inputs/omnilearned_input_sld_superjet.h5" \
    "$SLD_BASE/omnilearned/embeddings/omnilearned_embedding_s_sld_superjet.h5" \
    -s s -t embed

# 4. Reduce with t-SNE
sld-resurrect reduce-embeddings sld_superjet \
    --method tsne --size s --device cpu --max-events 1000
```

The reduced output then feeds the visualisation notebook
`examples/OmniLearn_Embeddings.ipynb`.

---

## References

The SLD publications reproduced (or partially reproduced) in this
codebase:

| Year | Reference | Measurement |
|---|---|---|
| 1994 | [hep-ex/9404001](https://arxiv.org/abs/hep-ex/9404001) | First $A_{LR}$ |
| 1995 | [hep-ex/9501003](https://arxiv.org/abs/hep-ex/9501003) | $\alpha_s(M_Z^2)$ from event shapes |
| 1997 | [hep-ex/9704012](https://arxiv.org/abs/hep-ex/9704012) | Leptonic coupling asymmetries |
| 1998 | [hep-ex/9708015](https://arxiv.org/abs/hep-ex/9708015) | $R_b$ via vertex mass |
| 2000 | [hep-ex/0004026](https://arxiv.org/abs/hep-ex/0004026) | High-precision $A_{LR}$ |
| 2001 | [hep-ex/0010015](https://arxiv.org/abs/hep-ex/0010015) | Leptonic coupling asymmetries (final) |
| 2005 | [hep-ex/0410042](https://arxiv.org/abs/hep-ex/0410042) | $A_b$, $A_c$ |
| 2006 | [hep-ex/0509008](https://arxiv.org/abs/hep-ex/0509008) | Precision EW on the $Z$ resonance |

The OmniLearn foundation model:

* OmniLearned paper: [arXiv:2510.24066](https://arxiv.org/abs/2510.24066)
* Pretrained checkpoints: <https://portal.nersc.gov/cfs/dasrepo/omnilearned/checkpoints/>

The reference experimental datasets used for the cross-experiment
comparison:

* **JetClass1** -- <https://zenodo.org/records/6619768>
* **H1 DIS Open Data** -- <https://portal.nersc.gov/cfs/dasrepo/omnilearned/h1/>
* **ALEPH MC Z-pole sample** -- Not Available Yet
