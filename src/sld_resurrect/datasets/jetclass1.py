"""JetClass1 dataset parser.

JetClass1 stores per-jet ROOT files with already-computed
``part_deta``, ``part_dphi`` (relative to the jet axis), and per-particle
energies, so no thrust calculation or jet clustering is needed -- we
just need to compute ``log pT`` / ``log E`` and pad.

"""

from __future__ import annotations

from collections.abc import Iterable

import awkward as ak
import numpy as np
import uproot
from aliad.interface import awkward as aku

__all__ = ["parse_jetclass1"]


_BRANCHES: list[str] = [
    "part_px",
    "part_py",
    "part_energy",
    "part_deta",
    "part_dphi",
]


def parse_jetclass1(
    filepaths: str | Iterable[str],
    max_events: int = -1,
    max_particles: int = 128,
) -> np.ndarray:
    """Parse JetClass1 ROOT files into an OmniLearned point cloud.

    Parameters
    ----------
    filepaths : str or iterable of str
        ROOT file(s) to read.
    max_events : int
        If positive, truncate to the first ``max_events`` events.
    max_particles : int
        Pad/clip each event to this many particles.

    Returns
    -------
    np.ndarray, shape (n_events, max_particles, 4), dtype float32
        Features ``(delta_eta, delta_phi, log pT, log E)``, pT-sorted.

    Notes
    -----
    The output is pT-sorted within each event. JetClass1 files are usually
    already produced in pT-descending order, so this is a no-op for stock
    files; the explicit sort is kept for consistency with the other
    dataset parsers.
    """
    events = uproot.concatenate(filepaths, filter_name=_BRANCHES)
    if max_events > 0:
        events = events[:max_events]

    pt = np.hypot(events["part_px"], events["part_py"])
    log_pt = np.log(pt)
    log_e = np.log(events["part_energy"])
    deta = events["part_deta"]
    dphi = events["part_dphi"]

    features = aku.stack(deta, dphi, log_pt, log_e)
    pt_order = ak.argsort(pt, axis=-1, ascending=False)
    features = features[pt_order]

    padded = aku.pad_and_fill(features, pad_size=max_particles, axis=1, value=0, clip=True)
    return ak.to_numpy(padded).astype(np.float32)
