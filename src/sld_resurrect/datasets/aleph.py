"""ALEPH dataset parser.

ALEPH ROOT files store one event per row with per-particle
``(pt, eta, phi, mass)``. We reconstruct ``Momentum4D`` collections,
cluster into two exclusive Durham jets (for the hemisphere strategy),
and dispatch to the same three strategies as SLD.
"""

from __future__ import annotations

from collections.abc import Iterable

import awkward as ak
import numpy as np
import uproot
import vector

from sld_resurrect.datasets.strategies import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_PARTICLES,
    Strategy,
    prepare_boosted_frame,
    prepare_hemisphere,
    prepare_superjet,
)

__all__ = ["parse_aleph"]


vector.register_awkward()


_TREE_NAME: str = "t"
_PARTICLE_BRANCHES: list[str] = ["pt", "eta", "phi", "mass"]


def _load_aleph_particles(
    filepaths: str | Iterable[str],
    max_events: int,
) -> ak.Array:
    """Load and assemble ``Momentum4D`` particle collections from ALEPH ROOT files.

    Parameters
    ----------
    filepaths : str or iterable of str
    max_events : int
        If positive, stop after this many events.

    Returns
    -------
    ak.Array of Momentum4D
    """
    if isinstance(filepaths, str):
        filepaths = [filepaths]

    files = {fp: _TREE_NAME for fp in filepaths}
    chunks: list[ak.Array] = []
    total = 0

    for chunk in uproot.iterate(files, expressions=_PARTICLE_BRANCHES, library="awkward"):
        if max_events > 0 and total >= max_events:
            break
        chunks.append(chunk)
        total += len(chunk)

    if not chunks:
        empty = ak.Array([])
        return ak.zip(
            {"pt": empty, "eta": empty, "phi": empty, "m": empty},
            with_name="Momentum4D",
        )

    arrays = ak.concatenate(chunks)
    if max_events > 0:
        arrays = arrays[:max_events]

    return ak.zip(
        {
            "pt": arrays["pt"],
            "eta": arrays["eta"],
            "phi": arrays["phi"],
            "m": arrays["mass"],
        },
        with_name="Momentum4D",
    )


def _cluster_two_jets(particles: ak.Array) -> ak.Array:
    """Cluster particles into exactly two exclusive Durham jets, pT-sorted.

    Returns
    -------
    constituents : ak.Array, shape ``[n_events, 2, n_particles]``
        Constituents of the two jets. Leading jet at index 0.
    """
    # Imported here so the module is importable without fastjet installed
    # for users who only need the superjet/boosted_frame strategies.
    import fastjet

    # The swig-level JetDefinition is needed because fastjet's public
    # wrapper requires an R parameter, which ee_kt does not take.
    from fastjet._swig import JetDefinition

    jet_def = JetDefinition(fastjet.ee_kt_algorithm)
    cluster_seq = fastjet.ClusterSequence(particles, jet_def)
    constituents = cluster_seq.exclusive_jets_constituents(2)
    jets = ak.sum(constituents, axis=2)
    pt_order = ak.argsort(jets.pt, axis=1, ascending=False)
    return constituents[pt_order]


def parse_aleph(
    filepaths: str | Iterable[str],
    strategy: Strategy,
    max_events: int = -1,
    max_particles: int = DEFAULT_MAX_PARTICLES,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Parse ALEPH ROOT files into an OmniLearned point cloud.

    Parameters
    ----------
    filepaths : str or iterable of str
        ROOT file(s) to read.
    strategy : {"superjet", "hemisphere", "boosted_frame"}
        Coordinate-mapping strategy. See
        :mod:`sld_resurrect.datasets.strategies`.
    max_events : int
        If positive, truncate to the first ``max_events`` events.
    max_particles : int
        Pad/clip each event to this many particles.
    batch_size : int
        Batch size for ``boosted_frame`` (ignored for other strategies).

    Returns
    -------
    np.ndarray or tuple of np.ndarray
        ``"superjet"`` / ``"boosted_frame"``: a single point cloud of shape
        ``(n_events, max_particles, 4)``.
        ``"hemisphere"``: a pair ``(leading, subleading)`` of point clouds.
    """
    particles = _load_aleph_particles(filepaths, max_events=max_events)

    if strategy == "superjet":
        return prepare_superjet(particles, max_particles=max_particles)

    if strategy == "hemisphere":
        constituents = _cluster_two_jets(particles)
        return prepare_hemisphere(constituents, max_particles=max_particles)

    if strategy == "boosted_frame":
        return prepare_boosted_frame(particles, max_particles=max_particles, batch_size=batch_size)

    raise ValueError(
        f"Unknown strategy {strategy!r}. Choose from: 'superjet', 'hemisphere', 'boosted_frame'."
    )
