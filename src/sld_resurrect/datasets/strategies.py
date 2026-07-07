"""OmniLearned coordinate-mapping strategies for ``e+ e-`` collision events.

The OmniLearned foundation model expects fixed-size point clouds in
``(delta_eta, delta_phi, log pT, log E)`` coordinates relative to a
reference axis. This module provides three different choices of that
reference axis, each with different physics motivation:

* :func:`prepare_superjet` -- treat the whole event as one wide-angle
  jet centred on the thrust axis.
* :func:`prepare_hemisphere` -- embed the leading and subleading Durham
  jets separately on their own axes. This matches the angular scale on
  which OmniLearned was pre-trained.
* :func:`prepare_boosted_frame` -- rigidly rotate every event into a
  frame whose ``+z`` axis lies along the thrust axis, then embed.

These are dataset-agnostic: anything that exposes a ``Momentum4D``
collection plus a thrust function can use them. Per-dataset wrappers
live in :mod:`sld_resurrect.datasets.sld`,
:mod:`sld_resurrect.datasets.aleph`, etc.

The four point-cloud features per particle are
``(delta_eta, delta_phi, log pT, log E)``; particles are pT-sorted and
each event is zero-padded (or clipped) to ``max_particles`` entries.
"""

from __future__ import annotations

from typing import Literal

import awkward as ak
import numpy as np
from aliad.interface import awkward as aku
from tqdm.auto import tqdm

from sld_resurrect.kinematics import thrust

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_MAX_PARTICLES",
    "Strategy",
    "prepare_boosted_frame",
    "prepare_hemisphere",
    "prepare_superjet",
]


Strategy = Literal["superjet", "hemisphere", "boosted_frame"]
"""Available coordinate-mapping strategies."""

DEFAULT_MAX_PARTICLES: int = 128
"""Default padding length per event."""

DEFAULT_BATCH_SIZE: int = 10_000
"""Default batch size for the boosted-frame rotation pass."""

_NUMERICAL_EPS: float = 1e-10
"""Numerical floor used when normalising rotation axes / arctanh arguments."""

_BOOSTED_FRAME_PT_FLOOR: float = 1e-6
"""Minimum :math:`p_T` (GeV) imposed on particles after rotation to the
thrust frame, to avoid ``log(0)`` in the feature builder.

The thrust axis maximises longitudinal energy flow by construction, so
the highest-momentum particles in an event tend to land near the new
:math:`\\pm z` after rotation -- giving them a numerically small (and
sometimes exactly zero) :math:`p_T^{\\rm new}`. The floor sits five
orders of magnitude below the smallest physical SLD track momentum
(:math:`\\sim 100` MeV), so it cannot bias the learned representation,
but is large enough to keep ``log(pt)`` finite.
"""


# ---------------------------------------------------------------------------
# Geometry helpers (private)
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
    """
    n_events = len(thrust_vectors)
    z_hat = np.array([0.0, 0.0, 1.0])

    # Rotation axis k = thrust x z_hat
    kx = thrust_vectors[:, 1] * z_hat[2] - thrust_vectors[:, 2] * z_hat[1]
    ky = thrust_vectors[:, 2] * z_hat[0] - thrust_vectors[:, 0] * z_hat[2]
    kz = thrust_vectors[:, 0] * z_hat[1] - thrust_vectors[:, 1] * z_hat[0]

    sin_angle = np.sqrt(kx**2 + ky**2 + kz**2)
    cos_angle = thrust_vectors @ z_hat

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
        identity + sin_angle[:, None, None] * K + (1.0 - cos_angle[:, None, None]) * K_squared
    )

    degenerate = sin_angle < _NUMERICAL_EPS
    rotations[degenerate] = np.eye(3)
    anti_parallel = degenerate & (cos_angle < 0)
    rotations[anti_parallel] = np.diag([-1.0, -1.0, 1.0])
    return rotations


def _build_omnilearn_features(
    particles: ak.Array,
    ref_eta: np.ndarray,
    ref_phi: np.ndarray,
    max_particles: int = DEFAULT_MAX_PARTICLES,
) -> np.ndarray:
    """Build a ``(n_events, max_particles, 4)`` point cloud.

    Per-particle features, expressed relative to ``(ref_eta, ref_phi)``:
    ``delta_eta``, ``delta_phi`` (wrapped to ``[-pi, pi]``), ``log pT``,
    ``log E``. Within each event, particles are pT-sorted and zero-padded
    (or clipped) to exactly ``max_particles`` entries.

    Parameters
    ----------
    particles : ak.Array
        Momentum4D collection with ``pt``, ``eta``, ``phi``, ``e`` fields.
    ref_eta, ref_phi : np.ndarray, shape (n_events,)
    max_particles : int

    Returns
    -------
    np.ndarray, shape (n_events, max_particles, 4), dtype float32
    """
    d_eta = particles.eta - ref_eta
    d_phi = _wrap_dphi(particles.phi - ref_phi)
    log_pt = np.log(particles.pt)
    log_e = np.log(particles.e)

    features = aku.stack(d_eta, d_phi, log_pt, log_e)
    pt_order = ak.argsort(particles.pt, axis=-1, ascending=False)
    features = features[pt_order]
    padded = aku.pad_and_fill(features, pad_size=max_particles, axis=1, value=0, clip=True)
    return ak.to_numpy(padded).astype(np.float32)


# ---------------------------------------------------------------------------
# Public strategy implementations
# ---------------------------------------------------------------------------


def prepare_superjet(
    particles: ak.Array,
    max_particles: int = DEFAULT_MAX_PARTICLES,
) -> np.ndarray:
    """Strategy 1: treat the whole event as a single wide-angle jet.

    Every particle is expressed relative to the thrust-axis ``(eta, phi)``.

    Parameters
    ----------
    particles : ak.Array
        Inclusive (non-jet-clustered) particle list per event.
    max_particles : int

    Returns
    -------
    np.ndarray, shape (n_events, max_particles, 4)
    """
    _, thrust_vecs, _ = thrust(particles)
    ref_eta, ref_phi = _thrust_eta_phi(thrust_vecs)
    return _build_omnilearn_features(particles, ref_eta, ref_phi, max_particles=max_particles)


def prepare_hemisphere(
    constituents: ak.Array,
    max_particles: int = DEFAULT_MAX_PARTICLES,
) -> tuple[np.ndarray, np.ndarray]:
    """Strategy 2: embed the leading and subleading jets on their own axes.

    Two embeddings per event, one per Durham jet. Each is centred on its
    own jet axis ``(eta, phi)`` so the angular scale matches what
    OmniLearned was pre-trained on. The two embeddings should be combined
    downstream (typically: pass each through OmniLearned separately,
    concatenate or average the resulting class scores).

    Parameters
    ----------
    constituents : ak.Array, shape ``[events, 2, particles]``
        Jet constituents, **pT-sorted at the jet level** (leading jet at
        index 0).
    max_particles : int

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

    px_flat = ak.to_numpy(ak.flatten(batch_particles.px))
    py_flat = ak.to_numpy(ak.flatten(batch_particles.py))
    pz_flat = ak.to_numpy(ak.flatten(batch_particles.pz))
    e_flat = ak.to_numpy(ak.flatten(batch_particles.e))

    p3_flat = np.stack([px_flat, py_flat, pz_flat], axis=-1)
    rotations_flat = np.repeat(rotations, counts, axis=0)
    p3_rotated = np.einsum("nij,nj->ni", rotations_flat, p3_flat)

    # Particles aligned with the new +/- z (originally near the thrust
    # axis) end up with px = py = 0 after rotation, giving pt = 0 and
    # eta = +/- inf. Clamp the rotated transverse momentum to a small
    # floor along the particle's original (px, py) direction. For the
    # degenerate case where (px, py) are exactly zero, we set
    # (px, py) = (floor, 0); the choice of phi is arbitrary here.
    px_rot = p3_rotated[:, 0].copy()
    py_rot = p3_rotated[:, 1].copy()
    pt_rot = np.hypot(px_rot, py_rot)
    too_small = pt_rot < _BOOSTED_FRAME_PT_FLOOR
    if np.any(too_small):
        exactly_zero = too_small & (pt_rot == 0)
        # Scale below-floor (but non-zero) (px, py) up to the floor.
        scale = np.where(
            too_small & ~exactly_zero,
            _BOOSTED_FRAME_PT_FLOOR / np.where(pt_rot > 0, pt_rot, 1.0),
            1.0,
        )
        px_rot = px_rot * scale
        py_rot = py_rot * scale
        # Pick an arbitrary direction for particles that were exactly on +/- z.
        px_rot = np.where(exactly_zero, _BOOSTED_FRAME_PT_FLOOR, px_rot)
        py_rot = np.where(exactly_zero, 0.0, py_rot)

    rotated_p4 = ak.zip(
        {
            "px": ak.unflatten(px_rot, counts),
            "py": ak.unflatten(py_rot, counts),
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
    """Strategy 3: rotate each event so its thrust axis lies along ``+z``.

    Done in batches to keep the flattened ``(sum_n, 3)`` block small
    enough for the dense 3x3 rotation step.

    Parameters
    ----------
    particles : ak.Array
        Inclusive (non-jet-clustered) particle list per event.
    max_particles : int
    batch_size : int
        Number of events processed per batch.
    show_progress : bool
        If True, show a tqdm progress bar.

    Returns
    -------
    np.ndarray, shape (n_events, max_particles, 4)
    """
    n_events = len(particles)
    batch_starts = range(0, n_events, batch_size)
    iterator = (
        tqdm(batch_starts, desc="Rotating to thrust frame") if show_progress else batch_starts
    )

    point_clouds = [
        _rotate_event_batch(
            particles[start : min(start + batch_size, n_events)],
            max_particles=max_particles,
        )
        for start in iterator
    ]
    return np.concatenate(point_clouds, axis=0)
