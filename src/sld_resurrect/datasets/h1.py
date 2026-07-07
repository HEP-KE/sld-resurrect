"""H1 DIS dataset parser.

The H1 ``.h5`` files released for the OmniLearn benchmark are already
in the (n_events, n_particles, n_features) layout that OmniLearn
consumes -- the per-particle features are
``(delta_eta, delta_phi, log pT, log E, ...)`` with extra trailing
columns we drop here. The parser concatenates files, truncates to the
first ``max_events``, drops the trailing feature columns, and pads.

"""

from __future__ import annotations

from collections.abc import Iterable

import awkward as ak
import h5py
import numpy as np
from aliad.interface import awkward as aku

__all__ = ["parse_h1"]


# Number of features OmniLearn consumes:
# (delta_eta, delta_phi, log pT, log E)
_N_FEATURES: int = 4


def parse_h1(
    filepaths: str | Iterable[str],
    max_events: int = -1,
    max_particles: int = 128,
) -> np.ndarray:
    """Parse H1 ``.h5`` files into an OmniLearn point cloud.

    Parameters
    ----------
    filepaths : str or iterable of str
        H5 files to read.
    max_events : int
        If positive, truncate to the first ``max_events`` events
        (across all files combined).
    max_particles : int
        Pad/clip each event to this many particles.

    Returns
    -------
    np.ndarray, shape (n_events, max_particles, 4), dtype float32
    """
    if isinstance(filepaths, str):
        filepaths = [filepaths]

    arrays: list[ak.Array] = []
    total = 0

    for filepath in filepaths:
        with h5py.File(filepath, "r") as f:
            chunk = ak.Array(f["data"][:])

        if max_events > 0:
            remaining = max_events - total
            if remaining <= 0:
                break
            chunk = chunk[:remaining]

        arrays.append(chunk)
        total += len(chunk)

    if not arrays:
        return np.empty((0, max_particles, _N_FEATURES), dtype=np.float32)

    combined = ak.concatenate(arrays)[:, :, :_N_FEATURES]
    padded = aku.pad_and_fill(combined, pad_size=max_particles, axis=1, value=0, clip=True)
    return ak.to_numpy(padded).astype(np.float32)
