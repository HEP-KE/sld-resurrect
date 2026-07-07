"""Dataset parsers for OmniLearn input preparation.

Each parser takes raw input file(s) for one experiment and produces a
fixed-size array point cloud of shape ``(n_events, max_particles, 4)``.
The four features are ``(delta_eta, delta_phi, log pT, log E)``.

The four currently-supported datasets:

* :func:`sld_resurrect.datasets.sld.parse_sld_dataset` -- SLD ``e+ e-``
  ($\\sqrt s = 91$ GeV, parquet input).
* :func:`sld_resurrect.datasets.aleph.parse_aleph` -- ALEPH ``e+ e-``
  ($\\sqrt s = 91$ GeV, ROOT input).
* :func:`sld_resurrect.datasets.h1.parse_h1` -- H1 DIS ``e+- p``
  (HEP-formatted HDF5 input).
* :func:`sld_resurrect.datasets.jetclass1.parse_jetclass1` --
  JetClass1 ``p p`` (ROOT input, pre-clustered jets).

The three jet coordinate-mapping strategies (``superjet``,
``hemisphere``, ``boosted_frame``) are dataset-agnostic and live in
:mod:`sld_resurrect.datasets.strategies`.
"""

from __future__ import annotations

from .aleph import parse_aleph
from .h1 import parse_h1
from .jetclass1 import parse_jetclass1
from .sld import parse_sld_dataset, save_strategy_outputs
from .strategies import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_PARTICLES,
    Strategy,
    prepare_boosted_frame,
    prepare_hemisphere,
    prepare_superjet,
)

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_MAX_PARTICLES",
    # Strategy primitives
    "Strategy",
    # Per-dataset parsers
    "parse_aleph",
    "parse_h1",
    "parse_jetclass1",
    "parse_sld_dataset",
    "prepare_boosted_frame",
    "prepare_hemisphere",
    "prepare_superjet",
    "save_strategy_outputs",
]
