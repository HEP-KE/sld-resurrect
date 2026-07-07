"""Filesystem-path resolution for sld_resurrect.

* ``SLD_BASE`` -- root directory for datasets, intermediate outputs,
  and analysis results. Used to construct
  :data:`DATASET_DIR`, :data:`OMNILEARN_INPUT_DIR`,
  :data:`OMNILEARN_EMBEDDING_DIR`, :data:`OMNILEARN_REDUCED_DIR`,
  and :data:`ANALYSIS_DIR`. Defaults to ``./sld``, matching the
  fallback used by the demo notebooks.
* ``OMNILEARN_CHECKPOINT_DIR`` -- where the OmniLearned ``.pt``
  checkpoints live. Defaults to ``./checkpoints/omnilearned``.

"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "ANALYSIS_DIR",
    "DATASET_DIR",
    "MEASUREMENTS_DIR",
    "MINIDST_PARQUET_DIR",
    "OMNILEARN_CHECKPOINT_DIR",
    "OMNILEARN_EMBEDDING_DIR",
    "OMNILEARN_INPUT_DIR",
    "OMNILEARN_PREDICTION_DIR",
    "OMNILEARN_REDUCED_DIR",
    "PLOTS_DIR",
    "SLD_BASE",
]


def _resolve(env_var: str, default: Path) -> Path:
    """Return ``Path(os.environ[env_var])`` if set, else ``default``."""
    value = os.environ.get(env_var)
    return Path(value) if value else default


# ---------------------------------------------------------------------------
# Top-level roots
# ---------------------------------------------------------------------------

SLD_BASE: Path = _resolve("SLD_BASE", Path.cwd() / "sld")
"""Root directory for SLD datasets, intermediate files, and analysis outputs.

The ``./sld`` fallback matches the demo notebooks, so the CLI and the
notebooks operate on the same tree when ``SLD_BASE`` is unset.
"""

OMNILEARN_CHECKPOINT_DIR: Path = _resolve(
    "OMNILEARN_CHECKPOINT_DIR",
    Path.cwd() / "checkpoints" / "omnilearned",
)
"""Directory containing the OmniLearned ``.pt`` checkpoint files."""


# ---------------------------------------------------------------------------
# Subdirectories under SLD_BASE
# ---------------------------------------------------------------------------

DATASET_DIR: Path = SLD_BASE / "datasets"
"""Raw and translated datasets (e.g. ``minidst_translated/parquet``)."""

MINIDST_PARQUET_DIR: Path = DATASET_DIR / "minidst_translated" / "parquet"
"""Translated mini-DST parquet shards -- the input the notebooks read."""

OMNILEARN_INPUT_DIR: Path = SLD_BASE / "omnilearned" / "inputs"
"""Point-cloud ``.h5`` files prepared for OmniLearned inference."""

OMNILEARN_EMBEDDING_DIR: Path = SLD_BASE / "omnilearned" / "embeddings"
"""OmniLearned body-output embeddings."""

OMNILEARN_PREDICTION_DIR: Path = SLD_BASE / "omnilearned" / "predictions"
"""OmniLearned classifier-head softmax probabilities."""

OMNILEARN_REDUCED_DIR: Path = SLD_BASE / "omnilearned" / "reduced"
"""t-SNE / UMAP-reduced 2D coordinates."""

ANALYSIS_DIR: Path = SLD_BASE / "analysis"
"""Analysis outputs."""

PLOTS_DIR: Path = ANALYSIS_DIR / "plots"
"""PDF plots from the analysis notebooks."""

MEASUREMENTS_DIR: Path = ANALYSIS_DIR / "measurements"
"""JSON files with measurement results."""
