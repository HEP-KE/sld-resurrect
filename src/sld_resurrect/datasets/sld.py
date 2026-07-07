"""SLD dataset parser"""

from __future__ import annotations

import os
from collections.abc import Iterable

import awkward as ak
import numpy as np

from sld_resurrect.datasets.strategies import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_PARTICLES,
    Strategy,
    prepare_boosted_frame,
    prepare_hemisphere,
    prepare_superjet,
)

__all__ = [
    "DEFAULT_SHARD_PATTERN",
    "SLD_REQUIRED_BANKS",
    "parse_sld_dataset",
    "save_strategy_outputs",
]


SLD_REQUIRED_BANKS: tuple[str, ...] = (
    "IEVENTH",  # event header: run, event, trigger, timestamp
    "PHBM",  # beam info: ecm, per-event polarisation, IP position
    "PHPSUM",  # inclusive reconstructed particles
    "PHKLUS",  # calorimeter clusters (LAC)
    "PHPOINT",  # PHPSUM -> (track, cluster) pointer bank
    "PHWIC",  # muon iron-calorimeter info
)
"""Mini-DST bank families the SLD selection/point-cloud pipeline reads."""

DEFAULT_SHARD_PATTERN: str = "*nrec*.parquet"
"""Glob pattern matching the released mini-DST parquet shards."""


def parse_sld_dataset(
    constituents: ak.Array | None,
    particles: ak.Array | None,
    strategy: Strategy,
    max_particles: int = DEFAULT_MAX_PARTICLES,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Convert SLD events to OmniLearn-compatible point clouds.

    Parameters
    ----------
    constituents : ak.Array or None
        Jet-clustered constituents, shape ``[events, 2, particles]``,
        pT-sorted at the jet level. Required for ``"hemisphere"``.
    particles : ak.Array or None
        Inclusive (non-jet-clustered) particle list per event. Required
        for ``"superjet"`` and ``"boosted_frame"``.
    strategy : {"superjet", "hemisphere", "boosted_frame"}
    max_particles : int
    batch_size : int
        Batch size for ``boosted_frame``.

    Returns
    -------
    np.ndarray or tuple of np.ndarray
        ``"superjet"`` / ``"boosted_frame"``: a single point cloud of shape
        ``(n_events, max_particles, 4)``.
        ``"hemisphere"``: a pair ``(leading, subleading)`` of point clouds.
    """
    if strategy == "superjet":
        if particles is None:
            raise ValueError("strategy='superjet' requires `particles`.")
        return prepare_superjet(particles, max_particles=max_particles)

    if strategy == "hemisphere":
        if constituents is None:
            raise ValueError("strategy='hemisphere' requires `constituents`.")
        return prepare_hemisphere(constituents, max_particles=max_particles)

    if strategy == "boosted_frame":
        if particles is None:
            raise ValueError("strategy='boosted_frame' requires `particles`.")
        return prepare_boosted_frame(particles, max_particles=max_particles, batch_size=batch_size)

    raise ValueError(
        f"Unknown strategy {strategy!r}. Choose from: 'superjet', 'hemisphere', 'boosted_frame'."
    )


def save_strategy_outputs(
    constituents: ak.Array | None,
    particles: ak.Array | None,
    output_dir: str | os.PathLike[str],
    strategies: Iterable[Strategy] = ("superjet", "hemisphere", "boosted_frame"),
    max_particles: int = DEFAULT_MAX_PARTICLES,
    name_prefix: str = "omnilearned_input_sld",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, str]:
    """Run each requested strategy and save the resulting arrays.

    Parameters
    ----------
    constituents, particles : ak.Array or None
        See :func:`parse_sld_dataset`.
    output_dir : path-like
        Directory in which to save the output files (created if missing).
    strategies : iterable of {"superjet", "hemisphere", "boosted_frame"}
        Strategies to run, in order.
    max_particles : int
    name_prefix : str
    batch_size : int
        Batch size for the ``boosted_frame`` strategy.

    Returns
    -------
    dict[str, str]
        Mapping from a label (e.g. ``"superjet"``,
        ``"hemisphere_leading"``, ...) to the absolute output filepath.
    """

    os.makedirs(output_dir, exist_ok=True)
    written: dict[str, str] = {}

    for strategy in strategies:
        result = parse_sld_dataset(
            constituents=constituents,
            particles=particles,
            strategy=strategy,
            max_particles=max_particles,
            batch_size=batch_size,
        )
        if strategy == "hemisphere":
            cloud_leading, cloud_subleading = result
            for label, array in (
                ("hemisphere_leading", cloud_leading),
                ("hemisphere_subleading", cloud_subleading),
            ):
                path = _write_array(output_dir, f"{name_prefix}_{label}", array)
                written[label] = path
        else:
            assert isinstance(result, np.ndarray)  # tuple only for "hemisphere"
            path = _write_array(output_dir, f"{name_prefix}_{strategy}", result)
            written[strategy] = path

    return written


def _write_array(
    output_dir: str | os.PathLike[str],
    stem: str,
    array: np.ndarray,
) -> str:
    """Write ``array`` to ``output_dir/stem.h5`` and return the absolute path."""
    import h5py

    path = os.path.join(output_dir, f"{stem}.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=array)
    return path
