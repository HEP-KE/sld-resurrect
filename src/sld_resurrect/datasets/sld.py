"""SLD dataset parser"""

from __future__ import annotations

import os
from typing import Iterable, Optional, Union

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


__all__ = ["parse_sld_dataset", "save_strategy_outputs"]


def parse_sld_dataset(
    constituents: Optional[ak.Array],
    particles: Optional[ak.Array],
    strategy: Strategy,
    max_particles: int = DEFAULT_MAX_PARTICLES,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
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
        return prepare_boosted_frame(
            particles, max_particles=max_particles, batch_size=batch_size
        )

    raise ValueError(
        f"Unknown strategy {strategy!r}. "
        "Choose from: 'superjet', 'hemisphere', 'boosted_frame'."
    )


def save_strategy_outputs(
    constituents: Optional[ak.Array],
    particles: Optional[ak.Array],
    output_dir: Union[str, "os.PathLike[str]"],
    strategies: Iterable[Strategy] = ("superjet", "hemisphere", "boosted_frame"),
    max_particles: int = DEFAULT_MAX_PARTICLES,
    name_prefix: str = "omnilearned_input_sld",
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
        )
        if strategy == "hemisphere":
            cloud_leading, cloud_subleading = result
            for label, array in (
                ("hemisphere_leading", cloud_leading),
                ("hemisphere_subleading", cloud_subleading),
            ):
                path = _write_array(
                    output_dir, f"{name_prefix}_{label}", array
                )
                written[label] = path
        else:
            path = _write_array(
                output_dir, f"{name_prefix}_{strategy}", result
            )
            written[strategy] = path

    return written


def _write_array(
    output_dir: Union[str, "os.PathLike[str]"],
    stem: str,
    array: np.ndarray,
) -> str:
    """Write ``array`` to ``output_dir/stem.h5`` and return the absolute path."""
    import h5py

    path = os.path.join(output_dir, f"{stem}.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=array)
    return os.path.abspath(path)