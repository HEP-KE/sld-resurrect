"""OmniLearn point-cloud feature builders for SLD events.

This module turns the inclusive particle list of an event into the fixed-size
``(n_events, max_particles, 4)`` arrays that the OmniLearn foundation model
expects, plus the four-component jet-level summary that travels alongside.

Three coordinate-mapping strategies are provided:

* :func:`prepare_superjet` -- treat the whole event as a single wide-angle
  jet centred on the thrust axis.
* :func:`prepare_hemisphere` -- embed the leading and subleading Durham jets
  separately, each centred on its own jet axis.  This matches the angular
  scale OmniLearn was trained on.
* :func:`prepare_boosted_frame` -- rigidly rotate every event into a frame
  whose ``+z`` axis lies along the thrust axis, then embed the result.

The four point-cloud features are
``(delta_eta, delta_phi, log(pT), log(E))``, computed relative to the chosen
reference axis.  Each event is pT-sorted and zero-padded (or clipped) to
exactly ``max_particles`` entries.

The high-level :func:`parse_sld_dataset` dispatches between the three
strategies and is the one most analysis code should call.
"""

from __future__ import annotations

import os
from typing import Literal

import awkward as ak
import numpy as np
from aliad.interface import awkward as aku
from tqdm.auto import tqdm

from sld_resurrect.kinematics import thrust


Strategy = Literal["superjet", "hemisphere", "boosted_frame"]
"""Point-cloud preparation strategies available in this module."""

DEFAULT_MAX_PARTICLES: int = 128
"""Default padding length for the per-event point cloud."""

DEFAULT_BATCH_SIZE: int = 10_000
"""Default batch size for memory-bounded operations (boosted-frame rotation)."""

_NUMERICAL_EPS: float = 1e-10
"""Numerical floor used when normalising rotation axes / arctanh arguments."""


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _wrap_dphi(dphi: ak.Array) -> ak.Array:
    """Wrap ``delta phi`` into the canonical ``[-pi, pi]`` interval."""
    return np.remainder(dphi + np.pi, 2 * np.pi) - np.pi


def _thrust_eta_phi(
    thrust_vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert per-event thrust 3-vectors into ``(eta, phi)`` reference axes.

    Parameters
    ----------
    thrust_vectors : np.ndarray, shape (n_events, 3)
        Unit vectors along the thrust axis.

    Returns
    -------
    eta : np.ndarray, shape (n_events,)
    phi : np.ndarray, shape (n_events,)
    """
    tx = thrust_vectors[:, 0]
    ty = thrust_vectors[:, 1]
    tz = thrust_vectors[:, 2]
    p_total = np.sqrt(tx**2 + ty**2 + tz**2)
    eta = np.arctanh(np.clip(tz / p_total, -1 + _NUMERICAL_EPS, 1 - _NUMERICAL_EPS))
    phi = np.arctan2(ty, tx)
    return eta, phi


def _rotation_matrix_to_z(thrust_vectors: np.ndarray) -> np.ndarray:
    """Per-event rotation matrices that send each thrust vector onto z-hat.

    Uses the Rodrigues rotation formula
    :math:`R = I + \\sin(\\alpha)\\,K + (1-\\cos(\\alpha))\\,K^2`
    where :math:`K` is the skew-symmetric cross-product matrix of the
    rotation axis :math:`\\hat n_T \\times \\hat z`. Two degenerate cases
    are handled explicitly:

    * thrust already along ``+z``  -> identity matrix;
    * thrust anti-parallel to ``+z`` -> ``diag(-1, -1, +1)``.

    Parameters
    ----------
    thrust_vectors : np.ndarray, shape (n_events, 3)
        Unit vectors along the thrust axis.

    Returns
    -------
    np.ndarray, shape (n_events, 3, 3)
        Rotation matrices, one per event.
    """
    n_events = len(thrust_vectors)
    z_hat = np.array([0.0, 0.0, 1.0])

    # Rotation axis k = thrust x z_hat
    kx = thrust_vectors[:, 1] * z_hat[2] - thrust_vectors[:, 2] * z_hat[1]
    ky = thrust_vectors[:, 2] * z_hat[0] - thrust_vectors[:, 0] * z_hat[2]
    kz = thrust_vectors[:, 0] * z_hat[1] - thrust_vectors[:, 1] * z_hat[0]

    sin_angle = np.sqrt(kx**2 + ky**2 + kz**2)
    cos_angle = thrust_vectors @ z_hat

    # Normalise the rotation axis (guard against sin~0).
    safe_sin = np.where(sin_angle > _NUMERICAL_EPS, sin_angle, 1.0)
    kx_n = kx / safe_sin
    ky_n = ky / safe_sin
    kz_n = kz / safe_sin

    identity = np.tile(np.eye(3), (n_events, 1, 1))

    K = np.zeros((n_events, 3, 3))
    K[:, 0, 1] = -kz_n
    K[:, 0, 2] = ky_n
    K[:, 1, 0] = kz_n
    K[:, 1, 2] = -kx_n
    K[:, 2, 0] = -ky_n
    K[:, 2, 1] = kx_n

    K_squared = np.einsum("nij,njk->nik", K, K)

    rotations = (
        identity
        + sin_angle[:, None, None] * K
        + (1.0 - cos_angle[:, None, None]) * K_squared
    )

    # Degenerate cases.
    degenerate = sin_angle < _NUMERICAL_EPS
    rotations[degenerate] = np.eye(3)
    anti_parallel = degenerate & (cos_angle < 0)
    rotations[anti_parallel] = np.diag([-1.0, -1.0, 1.0])

    return rotations


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------

def _build_omnilearn_features(
    particles: ak.Array,
    ref_eta: np.ndarray,
    ref_phi: np.ndarray,
    max_particles: int = DEFAULT_MAX_PARTICLES,
) -> np.ndarray:
    """Build a ``(n_events, max_particles, 4)`` point cloud.

    Per-particle features, expressed relative to ``(ref_eta, ref_phi)``:

    * ``delta_eta = eta - ref_eta``
    * ``delta_phi = wrap(phi - ref_phi)``
    * ``log(pT)``
    * ``log(E)``

    Within each event, particles are sorted by descending ``pT`` and then
    zero-padded (or clipped) to exactly ``max_particles`` entries.

    Parameters
    ----------
    particles : ak.Array
        Momentum4D collection with ``pt``, ``eta``, ``phi``, ``e`` fields.
    ref_eta, ref_phi : np.ndarray, shape (n_events,)
        Reference axis per event.
    max_particles : int, default 128
        Output sequence length.

    Returns
    -------
    np.ndarray, shape (n_events, max_particles, 4), dtype float32
        Padded point cloud.
    """
    d_eta = particles.eta - ref_eta
    d_phi = _wrap_dphi(particles.phi - ref_phi)
    log_pt = np.log(particles.pt)
    log_e = np.log(particles.e)

    features = aku.stack(d_eta, d_phi, log_pt, log_e)

    # pT-sort, then pad/clip to max_particles.
    pt_order = ak.argsort(particles.pt, axis=-1, ascending=False)
    features = features[pt_order]
    padded = aku.pad_and_fill(
        features, pad_size=max_particles, axis=1, value=0, clip=True
    )

    return ak.to_numpy(padded).astype(np.float32)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def prepare_superjet(
    particles: ak.Array,
    max_particles: int = DEFAULT_MAX_PARTICLES,
) -> np.ndarray:
    """Strategy 1: super-jet -- one wide-angle jet centred on the thrust axis.

    Every particle in the event is expressed relative to the thrust-axis
    ``(eta, phi)``. The whole event is treated as a single jet.

    Parameters
    ----------
    particles : ak.Array
        Inclusive (non-jet-clustered) particle list per event.
    max_particles : int
        Padding length.

    Returns
    -------
    np.ndarray, shape (n_events, max_particles, 4)
        Point cloud.
    """
    _, thrust_vecs, _ = thrust(particles)
    ref_eta, ref_phi = _thrust_eta_phi(thrust_vecs)
    return _build_omnilearn_features(
        particles, ref_eta, ref_phi, max_particles=max_particles
    )


def prepare_hemisphere(
    constituents: ak.Array,
    max_particles: int = DEFAULT_MAX_PARTICLES,
) -> tuple[np.ndarray, np.ndarray]:
    """Strategy 2: hemisphere embedding -- two jets, each on its own axis.

    Each of the two Durham jets is centred on its own ``(eta, phi)``,
    producing a point cloud whose angular scale matches what OmniLearn was
    pre-trained on. The two embeddings should be combined downstream
    (concatenation, averaging, ...).

    Parameters
    ----------
    constituents : ak.Array, shape ``[events, 2, particles]``
        Jet constituents, **pT-sorted at the jet level** (leading jet at
        index 0).
    max_particles : int
        Padding length per jet.

    Returns
    -------
    cloud_leading : np.ndarray, shape (n_events, max_particles, 4)
    cloud_subleading : np.ndarray, shape (n_events, max_particles, 4)
    """
    jets = ak.sum(constituents, axis=2)

    cloud_leading = _build_omnilearn_features(
        constituents[:, 0],
        ak.to_numpy(jets[:, 0].eta),
        ak.to_numpy(jets[:, 0].phi),
        max_particles=max_particles,
    )
    cloud_subleading = _build_omnilearn_features(
        constituents[:, 1],
        ak.to_numpy(jets[:, 1].eta),
        ak.to_numpy(jets[:, 1].phi),
        max_particles=max_particles,
    )
    return cloud_leading, cloud_subleading


def _rotate_event_batch(
    batch_particles: ak.Array,
    max_particles: int,
) -> np.ndarray:
    """Rotate one batch of events into the thrust-frame and build features."""
    _, thrust_vecs, _ = thrust(batch_particles)
    rotations = _rotation_matrix_to_z(thrust_vecs)
    counts = ak.to_numpy(ak.num(batch_particles, axis=-1))

    # Flatten 3-momenta to one big (sum_n, 3) block, rotate, then unflatten.
    px_flat = ak.to_numpy(ak.flatten(batch_particles.px))
    py_flat = ak.to_numpy(ak.flatten(batch_particles.py))
    pz_flat = ak.to_numpy(ak.flatten(batch_particles.pz))
    e_flat = ak.to_numpy(ak.flatten(batch_particles.e))

    p3_flat = np.stack([px_flat, py_flat, pz_flat], axis=-1)
    rotations_flat = np.repeat(rotations, counts, axis=0)
    p3_rotated = np.einsum("nij,nj->ni", rotations_flat, p3_flat)

    rotated_p4 = ak.zip(
        {
            "px": ak.unflatten(p3_rotated[:, 0], counts),
            "py": ak.unflatten(p3_rotated[:, 1], counts),
            "pz": ak.unflatten(p3_rotated[:, 2], counts),
            "m": ak.unflatten(np.zeros_like(e_flat), counts),
        },
        with_name="Momentum4D",
    )
    rotated_cylindrical = ak.zip(
        {
            "pt": rotated_p4.pt,
            "eta": rotated_p4.eta,
            "phi": rotated_p4.phi,
            "e": ak.unflatten(e_flat, counts),
        },
        with_name="Momentum4D",
    )

    # Reference axis is the new +z direction, i.e. (eta, phi) = (0, 0).
    n_events = len(batch_particles)
    ref_eta = np.zeros(n_events)
    ref_phi = np.zeros(n_events)
    return _build_omnilearn_features(
        rotated_cylindrical, ref_eta, ref_phi, max_particles=max_particles
    )


def prepare_boosted_frame(
    particles: ak.Array,
    max_particles: int = DEFAULT_MAX_PARTICLES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    show_progress: bool = True,
) -> np.ndarray:
    """Strategy 3: rotate each event so the thrust axis lies along ``+z``.

    Done in batches to keep the flattened ``(sum_n, 3)`` block small
    enough for the dense 3x3 rotation step.

    Parameters
    ----------
    particles : ak.Array
        Inclusive (non-jet-clustered) particle list per event.
    max_particles : int
        Padding length.
    batch_size : int
        Number of events processed per batch.
    show_progress : bool
        If True, display a tqdm progress bar.

    Returns
    -------
    np.ndarray, shape (n_events, max_particles, 4)
        Point cloud.
    """
    n_events = len(particles)
    batch_starts = range(0, n_events, batch_size)
    iterator = (
        tqdm(batch_starts, desc="Rotating to thrust frame")
        if show_progress
        else batch_starts
    )

    point_clouds = [
        _rotate_event_batch(
            particles[start : min(start + batch_size, n_events)],
            max_particles=max_particles,
        )
        for start in iterator
    ]
    return np.concatenate(point_clouds, axis=0)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def parse_sld_dataset(
    constituents: ak.Array | None,
    particles: ak.Array | None,
    strategy: Strategy,
    max_particles: int = DEFAULT_MAX_PARTICLES,
    **kwargs,
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
        Coordinate-mapping strategy.
    max_particles : int
        Padding length.
    **kwargs
        Passed to the underlying ``prepare_*`` function (e.g.
        ``batch_size`` for the boosted-frame strategy).

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
            particles, max_particles=max_particles, **kwargs
        )

    raise ValueError(
        f"Unknown strategy {strategy!r}. "
        "Choose from: 'superjet', 'hemisphere', 'boosted_frame'."
    )


def save_strategy_outputs(
    constituents: ak.Array | None,
    particles: ak.Array | None,
    output_dir: str,
    strategies: tuple[Strategy, ...] = ("superjet", "hemisphere", "boosted_frame"),
    max_particles: int = DEFAULT_MAX_PARTICLES,
    name_prefix: str = "omnilearned_input_sld",
) -> dict[str, str]:
    """Run each requested strategy and save the resulting arrays as ``.h5``.

    Parameters
    ----------
    constituents, particles : ak.Array or None
        See :func:`parse_sld_dataset`.
    output_dir : str
        Directory in which to save the output files.
    strategies : tuple of str
        Strategies to run, in order.
    max_particles : int
        Padding length.
    name_prefix : str
        Filename prefix.

    Returns
    -------
    dict[str, str]
        Mapping from a label (e.g. ``"superjet"``,
        ``"hemisphere_leading"``, ...) to the absolute output filepath.
    """
    import h5py

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
                path = os.path.join(output_dir, f"{name_prefix}_{label}.h5")
                with h5py.File(path, "w") as f:
                    f.create_dataset("data", data=array)
                written[label] = path
        else:
            path = os.path.join(output_dir, f"{name_prefix}_{strategy}.h5")
            with h5py.File(path, "w") as f:
                f.create_dataset("data", data=result)
            written[strategy] = path

    return written
