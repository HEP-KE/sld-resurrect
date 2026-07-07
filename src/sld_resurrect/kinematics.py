"""Event-shape and kinematic observables for the SLD analysis.

This module is the physics-helper layer of the SLD analysis: it builds
particle 4-vectors from the ``PHPSUM`` bank and provides the standard
event-shape observables (thrust, sphericity, C-parameter, oblateness,
heavy jet mass, ...), plus the hemisphere-splitting and two-object
kinematics helpers that the event selection and asymmetry analyses
depend on.

Conventions
-----------
* All particle collections are awkward arrays of ``Momentum4D`` records
  with the fields ``pt``, ``eta``, ``phi``, ``e`` plus ``charge``,
  ``vx``, ``vy``, ``vz``. The ``v*`` prefix is used because the bare
  ``x``, ``y``, ``z`` names collide with vector library accessors.
* Hemisphere helpers that need only a subset of particles (e.g. just
  charged tracks) take that subset *as input*. They do NOT re-filter
  the collection for you. This keeps track-quality selection in one
  place: the caller decides which tracks count, the helper does the
  geometry.
* Functions returning numpy arrays produce shape ``(n_events,)`` (or
  ``(n_events, 3)`` for 3-vectors). Functions returning awkward arrays
  preserve the per-event jagged structure of their input.
"""

from __future__ import annotations

import awkward as ak
import numba as nb
import numpy as np
import vector

vector.register_awkward()


# ---------------------------------------------------------------------------
# Physics constants
# ---------------------------------------------------------------------------

E_CM: float = 91.18
"""Z-pole centre-of-mass energy [GeV]."""

M_PION: float = 0.13957
"""Charged-pion mass [GeV/c^2], the standard SLD track mass hypothesis."""


# ---------------------------------------------------------------------------
# Particle 4-momentum construction
# ---------------------------------------------------------------------------


def build_particles(data: ak.Array) -> ak.Array:
    """Build Momentum4D particles from the ``PHPSUM`` bank.

    Charged particles are given the charged-pion mass; neutral particles are
    treated as massless. The output is in cylindrical (pt, eta, phi, e) form
    with extra fields attached: ``charge`` and the origin (vertex)
    coordinates ``vx``, ``vy``, ``vz`` of each particle. The vertex
    coordinates are needed by the track-quality cuts.

    The fields are named ``vx``, ``vy``, ``vz`` rather than ``x``, ``y``,
    ``z`` because ``x``/``y``/``z`` collide with the Cartesian-coordinate
    accessors that the ``vector`` library exposes on a Momentum4D record.

    Parameters
    ----------
    data : ak.Array
        Full event record containing at least the ``PHPSUM`` bank.

    Returns
    -------
    ak.Array
        Per-event Momentum4D particle list with fields
        ``pt``, ``eta``, ``phi``, ``e``, ``charge``, ``vx``, ``vy``, ``vz``.
    """
    charge = data.PHPSUM.charge
    mass = ak.where(charge != 0, M_PION, 0.0)

    cartesian = ak.zip(
        {
            "px": data.PHPSUM.px,
            "py": data.PHPSUM.py,
            "pz": data.PHPSUM.pz,
            "m": mass,
        },
        with_name="Momentum4D",
    )
    particles = ak.zip(
        {
            "pt": cartesian.pt,
            "eta": cartesian.eta,
            "phi": cartesian.phi,
            "e": cartesian.e,
        },
        with_name="Momentum4D",
    )
    particles["charge"] = charge
    particles["vx"] = data.PHPSUM.x
    particles["vy"] = data.PHPSUM.y
    particles["vz"] = data.PHPSUM.z
    return particles


def build_tracks(data: ak.Array) -> ak.Array:
    """Build Momentum4D 4-vectors from the ``PHCHRG`` (charged track) bank.

    Unlike :func:`build_particles` which uses the inclusive ``PHPSUM``
    list (the reconstruction's best-effort merging of tracks and
    clusters into physics-object 4-vectors), this helper rebuilds the
    track 4-vector *directly* from the raw helix parameters. Useful for
    track-level analyses where the PHPSUM merging is not appropriate
    (e.g. impact-parameter studies, vertex-based flavour tagging) or
    for cross-checking PHPSUM against its underlying detector inputs
    via the PHPOINT pointer bank.

    The 6-element ``hlxpar`` helix parameter vector is laid out as
    ``[phi, 1/pt, tan(lambda), x, y, z]``, where ``phi`` is the
    track-momentum azimuth at the reference point, ``1/pt`` is the
    signed inverse transverse momentum, ``tan(lambda) = pz / pt`` with
    ``lambda`` the dip angle, and ``(x, y, z)`` is the helix reference
    point. We take the magnitude of ``1/pt`` (sign is redundant with
    the ``charge`` field) and assemble (px, py, pz) cylindrically.
    Tracks are given the charged-pion mass, consistent with
    :func:`build_particles`.

    Parameters
    ----------
    data : ak.Array
        Full event record containing at least the ``PHCHRG`` bank.

    Returns
    -------
    ak.Array
        Per-event Momentum4D track list with fields
        ``pt``, ``eta``, ``phi``, ``e``, ``charge``, ``vx``, ``vy``, ``vz``.
        The ``v*`` fields are the per-track helix reference point (the
        "origin" of the helix, typically near the IP or a secondary
        vertex), analogous to PHPSUM's per-particle origin.
    """
    hlxpar = data.PHCHRG.hlxpar
    phi = hlxpar[..., 0]
    inv_pt = hlxpar[..., 1]
    tan_lambda = hlxpar[..., 2]
    ref_x = hlxpar[..., 3]
    ref_y = hlxpar[..., 4]
    ref_z = hlxpar[..., 5]

    # Magnitude of pt from |1/pt|, with a safety guard for the
    # (unphysical) inv_pt == 0 case.
    pt = 1.0 / ak.where(inv_pt != 0, abs(inv_pt), 1e-12)

    cartesian = ak.zip(
        {
            "px": pt * np.cos(phi),
            "py": pt * np.sin(phi),
            "pz": pt * tan_lambda,
            "m": M_PION,
        },
        with_name="Momentum4D",
    )
    tracks = ak.zip(
        {
            "pt": cartesian.pt,
            "eta": cartesian.eta,
            "phi": cartesian.phi,
            "e": cartesian.e,
        },
        with_name="Momentum4D",
    )
    tracks["charge"] = data.PHCHRG.charge
    tracks["vx"] = ref_x
    tracks["vy"] = ref_y
    tracks["vz"] = ref_z
    return tracks


def build_clusters(
    data: ak.Array,
    *,
    energy_weighted: bool = True,
    energy_scale: float = 1.0,
) -> ak.Array:
    """Build Momentum4D 4-vectors from the ``PHKLUS`` (calo cluster) bank.

    Calorimeter clusters are treated as massless light-cone modes: the
    cluster energy is ``eraw`` (optionally rescaled), and the
    direction is set by the cluster centroid ``(cos theta, phi)``.
    PHKLUS records both *geometric* (``cth``, ``phi``) and
    *energy-weighted* (``wcth``, ``wphi``) centroids; the
    energy-weighted ones are physically more meaningful for a 4-vector
    and are the default.

    The full-cluster centroid is used. PHKLUS also exposes EM-only
    (``cth2``/``wcth2``/``phi2``/``wphi2``) and HAD-only
    (``cth3``/``wcth3``/``phi3``/``wphi3``) subcluster variants; use
    those directly if you need to build subcluster 4-vectors.

    Parameters
    ----------
    data : ak.Array
        Full event record containing at least the ``PHKLUS`` bank.
    energy_weighted : bool, default True
        If True, use the energy-weighted centroid (``wcth``, ``wphi``);
        otherwise use the geometric centroid (``cth``, ``phi``).
    energy_scale : float, default 1.0
        Multiplicative calibration applied to ``eraw``. The default
        leaves the raw cluster energy unchanged; the caller is
        responsible for any LAC calibration appropriate to their analysis.

    Returns
    -------
    ak.Array
        Per-event Momentum4D cluster list with fields
        ``pt``, ``eta``, ``phi``, ``e``.
    """
    eraw = data.PHKLUS.eraw * energy_scale
    cos_theta = data.PHKLUS.wcth if energy_weighted else data.PHKLUS.cth
    phi = data.PHKLUS.wphi if energy_weighted else data.PHKLUS.phi

    # |p| = E for massless clusters; (px, py, pz) follows from
    # (E, cos_theta, phi) via pt = E sin(theta). Apply a tiny floor on
    # sin(theta) so that clusters exactly along the beam axis (cos_theta
    # = +/- 1) still get a finite pt -- and therefore a finite eta when
    # the record is converted to cylindrical form. The floor sits many
    # orders of magnitude below physical cluster pt (~10 MeV minimum)
    # so it cannot affect downstream physics.
    _SIN_THETA_FLOOR = 1e-6
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta**2))
    sin_theta = ak.where(sin_theta > _SIN_THETA_FLOOR, sin_theta, _SIN_THETA_FLOOR)
    pt = eraw * sin_theta

    cartesian = ak.zip(
        {
            "px": pt * np.cos(phi),
            "py": pt * np.sin(phi),
            "pz": eraw * cos_theta,
            "m": ak.zeros_like(eraw),
        },
        with_name="Momentum4D",
    )
    clusters = ak.zip(
        {
            "pt": cartesian.pt,
            "eta": cartesian.eta,
            "phi": cartesian.phi,
            "e": cartesian.e,
        },
        with_name="Momentum4D",
    )
    return clusters


# ---------------------------------------------------------------------------
# Thrust
# ---------------------------------------------------------------------------


@nb.njit(parallel=True, cache=True)
def _thrust_kernel(
    px: np.ndarray,
    py: np.ndarray,
    pz: np.ndarray,
    offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Thrust by enumeration over single-particle and particle-pair-sum axes.

    Exact for back-to-back two-jet topologies; for events whose true
    thrust axis is not aligned with any particle or pair sum, the
    enumeration is a tight lower-bound approximation.
    """
    n_events = len(offsets) - 1
    thrust_vals = np.zeros(n_events, dtype=np.float64)
    thrust_axes = np.zeros((n_events, 3), dtype=np.float64)

    for i in nb.prange(n_events):
        lo, hi = offsets[i], offsets[i + 1]
        n = hi - lo

        if n < 2:
            thrust_vals[i] = 1.0
            if n == 1:
                mag = np.sqrt(px[lo] ** 2 + py[lo] ** 2 + pz[lo] ** 2)
                if mag > 0.0:
                    thrust_axes[i, 0] = px[lo] / mag
                    thrust_axes[i, 1] = py[lo] / mag
                    thrust_axes[i, 2] = pz[lo] / mag
            continue

        px_i = px[lo:hi]
        py_i = py[lo:hi]
        pz_i = pz[lo:hi]

        pmag = np.sqrt(px_i**2 + py_i**2 + pz_i**2)
        total_p = pmag.sum()
        if total_p <= 0.0:
            continue

        best_t = 0.0
        best_ax = 0.0
        best_ay = 0.0
        best_az = 0.0

        for k in range(n):
            nk = pmag[k]
            if nk <= 0.0:
                continue

            ax = px_i[k] / nk
            ay = py_i[k] / nk
            az = pz_i[k] / nk

            s = 0.0
            for m in range(n):
                s += abs(px_i[m] * ax + py_i[m] * ay + pz_i[m] * az)

            if s > best_t:
                best_t = s
                best_ax = ax
                best_ay = ay
                best_az = az

            for j in range(k + 1, n):
                cx = px_i[k] + px_i[j]
                cy = py_i[k] + py_i[j]
                cz = pz_i[k] + pz_i[j]
                cn = np.sqrt(cx * cx + cy * cy + cz * cz)
                if cn <= 0.0:
                    continue
                ax2 = cx / cn
                ay2 = cy / cn
                az2 = cz / cn

                s2 = 0.0
                for m in range(n):
                    s2 += abs(px_i[m] * ax2 + py_i[m] * ay2 + pz_i[m] * az2)

                if s2 > best_t:
                    best_t = s2
                    best_ax = ax2
                    best_ay = ay2
                    best_az = az2

        thrust_vals[i] = best_t / total_p
        thrust_axes[i, 0] = best_ax
        thrust_axes[i, 1] = best_ay
        thrust_axes[i, 2] = best_az

    return thrust_vals, thrust_axes


def thrust(
    particles: ak.Array,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thrust value, thrust axis, and signed cos(theta_T) for every event.

    The axis is found by enumerating single-particle and particle-pair-sum
    candidates -- exact for two-jet topologies, a tight approximation
    otherwise. The thrust axis has an intrinsic two-fold ambiguity (n and
    -n give the same T), so the sign of cos(theta_T) is only meaningful if
    a convention is applied externally -- see
    :func:`orient_thrust_by_charge`.

    Parameters
    ----------
    particles : ak.Array
        Momentum4D collection. The caller is responsible for any prior
        track-quality filtering; the function uses every particle it
        receives.

    Returns
    -------
    T : np.ndarray, shape (n_events,)
        Thrust value in [0.5, 1].
    thrust_vec : np.ndarray, shape (n_events, 3)
        Unit vector along the thrust axis.
    cos_theta_T : np.ndarray, shape (n_events,)
        Signed ``thrust_vec[:, 2]``, in [-1, 1].
    """
    px = ak.to_numpy(ak.flatten(particles.px)).astype(np.float64)
    py = ak.to_numpy(ak.flatten(particles.py)).astype(np.float64)
    pz = ak.to_numpy(ak.flatten(particles.pz)).astype(np.float64)
    counts = ak.to_numpy(ak.num(particles, axis=-1))
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    T, thrust_vec = _thrust_kernel(px, py, pz, offsets)
    cos_theta_T = thrust_vec[:, 2]
    return T, thrust_vec, cos_theta_T


def thrust_value(particles: ak.Array) -> np.ndarray:
    """Just the scalar thrust T per event.

    Convenience wrapper around :func:`thrust` for cases where the axis is
    not needed (e.g. the 2005 :math:`A_b/A_c` ``T > 0.8`` cut).
    """
    T, _, _ = thrust(particles)
    return T


def orient_thrust_by_charge(
    thrust_vec: np.ndarray,
    charged: ak.Array,
) -> np.ndarray:
    """Resolve the thrust-axis sign using the hemisphere net charges.

    The thrust axis is symmetric under :math:`\\hat n \\to -\\hat n`, so
    the sign of ``cos(theta_T)`` carries no physical information by
    default. This function fixes the sign so the axis points along the
    **positive**-net-charge hemisphere (for dilepton events, along the
    positively-charged lepton): if the forward hemisphere defined by the
    candidate axis has net charge ``q_f < 0``, the axis is flipped. The
    convention is validated end to end by the sign of the extracted
    leptonic asymmetries, which reproduce the published SLD values.

    Events whose hemisphere charges are ambiguous (same-sign or zero) are
    returned with NaN axis components, so downstream cuts naturally drop
    them.

    Parameters
    ----------
    thrust_vec : np.ndarray, shape (n_events, 3)
        Raw thrust axis from :func:`thrust`.
    charged : ak.Array
        The charged-track collection used to compute ``thrust_vec``. The
        same set of tracks must be used to define the hemispheres so the
        sign convention is internally consistent.

    Returns
    -------
    np.ndarray, shape (n_events, 3)
        Sign-corrected thrust axes, with ambiguous events set to NaN.
    """
    q_f, q_b = hemisphere_net_charge(charged, thrust_vec)
    flip = np.where(q_f < 0, -1.0, 1.0)
    oriented = thrust_vec * flip[:, np.newaxis]

    ambiguous = (q_f * q_b >= 0) | (q_f == 0)
    if np.any(ambiguous):
        oriented = oriented.copy()
        oriented[ambiguous] = np.nan
    return oriented


# ---------------------------------------------------------------------------
# Thrust-major / thrust-minor / oblateness
# ---------------------------------------------------------------------------


@nb.njit(parallel=True, cache=True)
def _thrust_major_minor_kernel(
    px: np.ndarray,
    py: np.ndarray,
    pz: np.ndarray,
    offsets: np.ndarray,
    tx: np.ndarray,
    ty: np.ndarray,
    tz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute thrust-major and thrust-minor for each event."""
    n_events = len(offsets) - 1
    T_maj = np.zeros(n_events, dtype=np.float64)
    T_min = np.zeros(n_events, dtype=np.float64)

    for i in nb.prange(n_events):
        lo, hi = offsets[i], offsets[i + 1]
        n = hi - lo

        px_i = px[lo:hi]
        py_i = py[lo:hi]
        pz_i = pz[lo:hi]

        pmag = np.sqrt(px_i**2 + py_i**2 + pz_i**2)
        total_p = pmag.sum()
        if total_p <= 0.0:
            continue

        nx, ny, nz = tx[i], ty[i], tz[i]

        best_maj = 0.0
        best_mx = 0.0
        best_my = 0.0
        best_mz = 0.0

        for k in range(n):
            dot_k = px_i[k] * nx + py_i[k] * ny + pz_i[k] * nz
            qx = px_i[k] - dot_k * nx
            qy = py_i[k] - dot_k * ny
            qz = pz_i[k] - dot_k * nz
            qmag = np.sqrt(qx * qx + qy * qy + qz * qz)
            if qmag <= 0.0:
                continue

            mx = qx / qmag
            my = qy / qmag
            mz = qz / qmag

            s = 0.0
            for m in range(n):
                dot_m = px_i[m] * nx + py_i[m] * ny + pz_i[m] * nz
                rx = px_i[m] - dot_m * nx
                ry = py_i[m] - dot_m * ny
                rz = pz_i[m] - dot_m * nz
                s += abs(rx * mx + ry * my + rz * mz)

            if s > best_maj:
                best_maj = s
                best_mx = mx
                best_my = my
                best_mz = mz

        T_maj[i] = best_maj / total_p

        # n_min = n_T x n_maj  (unit vector since both inputs are unit)
        min_x = ny * best_mz - nz * best_my
        min_y = nz * best_mx - nx * best_mz
        min_z = nx * best_my - ny * best_mx

        s_min = 0.0
        for m in range(n):
            s_min += abs(px_i[m] * min_x + py_i[m] * min_y + pz_i[m] * min_z)
        T_min[i] = s_min / total_p

    return T_maj, T_min


def thrust_major_minor(
    particles: ak.Array,
    thrust_vec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Thrust-major :math:`T_\text{maj}` and thrust-minor :math:`T_\text{min}`.

    The same particle collection used to compute ``thrust_vec`` must be
    passed in here so the major/minor axes are anchored to a consistent
    thrust axis.

    Parameters
    ----------
    particles : ak.Array
        Momentum4D collection.
    thrust_vec : np.ndarray, shape (n_events, 3)
        Thrust axis from :func:`thrust`.

    Returns
    -------
    T_maj, T_min : np.ndarray, shape (n_events,)
    """
    px = ak.to_numpy(ak.flatten(particles.px)).astype(np.float64)
    py = ak.to_numpy(ak.flatten(particles.py)).astype(np.float64)
    pz = ak.to_numpy(ak.flatten(particles.pz)).astype(np.float64)
    counts = ak.to_numpy(ak.num(particles, axis=-1))
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    tx = thrust_vec[:, 0].astype(np.float64)
    ty = thrust_vec[:, 1].astype(np.float64)
    tz = thrust_vec[:, 2].astype(np.float64)

    return _thrust_major_minor_kernel(px, py, pz, offsets, tx, ty, tz)


def oblateness(
    particles: ak.Array,
    thrust_vec: np.ndarray,
) -> np.ndarray:
    r"""Oblateness :math:`O = T_\text{maj} - T_\text{min}`. In :math:`[0, 1/\sqrt 3]`."""
    T_maj, T_min = thrust_major_minor(particles, thrust_vec)
    return T_maj - T_min


# ---------------------------------------------------------------------------
# Sphericity, aplanarity, C-parameter
# ---------------------------------------------------------------------------


def event_momentum_tensor(particles: ak.Array) -> np.ndarray:
    r"""Per-event 3x3 momentum tensors :math:`M_{jk} = \sum_i p_j^{(i)} p_k^{(i)}`.

    Parameters
    ----------
    particles : ak.Array
        Momentum4D array with ``px``, ``py``, ``pz`` fields.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_events, 3, 3)``.
    """
    px, py, pz = particles.px, particles.py, particles.pz
    mxx = ak.to_numpy(ak.sum(px * px, axis=-1))
    myy = ak.to_numpy(ak.sum(py * py, axis=-1))
    mzz = ak.to_numpy(ak.sum(pz * pz, axis=-1))
    mxy = ak.to_numpy(ak.sum(px * py, axis=-1))
    mxz = ak.to_numpy(ak.sum(px * pz, axis=-1))
    myz = ak.to_numpy(ak.sum(py * pz, axis=-1))

    return np.stack(
        [
            np.stack([mxx, mxy, mxz], axis=-1),
            np.stack([mxy, myy, myz], axis=-1),
            np.stack([mxz, myz, mzz], axis=-1),
        ],
        axis=-2,
    )


def sphericity_tensor_eigenvalues(particles: ak.Array) -> np.ndarray:
    """Three eigenvalues of the normalised sphericity tensor, descending.

    Returns
    -------
    np.ndarray, shape (n_events, 3)
        Eigenvalues sorted in descending order
        (:math:`\\lambda_1 \\ge \\lambda_2 \\ge \\lambda_3`), satisfying
        :math:`\\lambda_1 + \\lambda_2 + \\lambda_3 = 1`. Events with zero
        total momentum yield NaN eigenvalues.
    """
    tensors = event_momentum_tensor(particles)
    p2 = np.trace(tensors, axis1=-2, axis2=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        normalised = tensors / p2[:, None, None]
    eigvals = np.linalg.eigvalsh(normalised)  # ascending
    return eigvals[:, ::-1]


def sphericity(particles: ak.Array) -> np.ndarray:
    r"""Sphericity :math:`S = \tfrac{3}{2}(\lambda_2 + \lambda_3)`. In :math:`[0, 1]`."""
    eigvals = sphericity_tensor_eigenvalues(particles)
    return 1.5 * (eigvals[:, 1] + eigvals[:, 2])


def aplanarity(particles: ak.Array) -> np.ndarray:
    r"""Aplanarity :math:`A = \tfrac{3}{2}\lambda_3`. In :math:`[0, 0.5]`."""
    eigvals = sphericity_tensor_eigenvalues(particles)
    return 1.5 * eigvals[:, 2]


def c_parameter(particles: ak.Array) -> np.ndarray:
    r"""C-parameter from the eigenvalues of the linearised momentum tensor.

    .. math::

        \theta_{\rho\sigma}
            = \frac{\sum_i p_i^\rho p_i^\sigma / |\vec p_i|}{\sum_i |\vec p_i|},
        \qquad
        C = 3 (\lambda_1\lambda_2 + \lambda_2\lambda_3 + \lambda_3\lambda_1).

    In :math:`[0, 1]`.
    """
    px, py, pz = particles.px, particles.py, particles.pz
    pmag = particles.p
    safe_pmag = ak.where(pmag > 0, pmag, 1.0)

    def _elem(a: ak.Array, b: ak.Array) -> np.ndarray:
        return ak.to_numpy(ak.sum(a * b / safe_pmag, axis=-1))

    total_p = ak.to_numpy(ak.sum(pmag, axis=-1))

    mxx = _elem(px, px)
    myy = _elem(py, py)
    mzz = _elem(pz, pz)
    mxy = _elem(px, py)
    mxz = _elem(px, pz)
    myz = _elem(py, pz)

    tensors = np.stack(
        [
            np.stack([mxx, mxy, mxz], axis=-1),
            np.stack([mxy, myy, myz], axis=-1),
            np.stack([mxz, myz, mzz], axis=-1),
        ],
        axis=-2,
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        tensors = tensors / total_p[:, None, None]

    eigvals = np.linalg.eigvalsh(tensors)
    l1, l2, l3 = eigvals[:, 0], eigvals[:, 1], eigvals[:, 2]
    return 3.0 * (l1 * l2 + l2 * l3 + l3 * l1)


# ---------------------------------------------------------------------------
# Hemisphere splitting
# ---------------------------------------------------------------------------


def split_by_beam_hemisphere(
    particles: ak.Array,
) -> tuple[ak.Array, ak.Array]:
    r"""Split each event into :math:`p_z > 0` and :math:`p_z \le 0` halves.

    This is the topological-filter definition used by the :math:`A_{LR}`
    selections to reject Bhabha and beam-related backgrounds. For
    physics-level hemisphere quantities, use
    :func:`split_by_thrust_hemisphere` instead.
    """
    return particles[particles.pz > 0], particles[particles.pz <= 0]


def split_by_thrust_hemisphere(
    particles: ak.Array,
    thrust_vec: np.ndarray,
) -> tuple[ak.Array, ak.Array]:
    r"""Split each event by the plane perpendicular to the thrust axis.

    Parameters
    ----------
    particles : ak.Array
        Event-indexed particle collection.
    thrust_vec : np.ndarray, shape (n_events, 3)
        Thrust axis per event from :func:`thrust`.

    Returns
    -------
    forward, backward : ak.Array
        Particles with :math:`\vec p \cdot \hat n_T > 0` and :math:`\le 0`
        respectively.
    """
    nx = ak.Array(thrust_vec[:, 0])
    ny = ak.Array(thrust_vec[:, 1])
    nz = ak.Array(thrust_vec[:, 2])
    dot = particles.px * nx + particles.py * ny + particles.pz * nz
    return particles[dot > 0], particles[dot <= 0]


def hemisphere_charge(
    particles: ak.Array,
    thrust_vec: np.ndarray,
    kappa: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Momentum-weighted hemisphere charges about the thrust axis.

    :math:`Q_H = \sum_{i \in H} q_i |\vec p_i|^{\kappa}`. The
    ``kappa = 0.5`` default matches the convention used in most SLD
    heavy-flavour analyses.

    Parameters
    ----------
    particles : ak.Array
        The particle collection to use. The caller controls track
        quality.
    thrust_vec : np.ndarray, shape (n_events, 3)
        Thrust axis defining the hemisphere split.
    kappa : float, optional
        Momentum-weighting exponent. Default 0.5.
    """
    forward, backward = split_by_thrust_hemisphere(particles, thrust_vec)

    def _hemisphere_sum(hemi: ak.Array) -> np.ndarray:
        return ak.to_numpy(ak.sum(hemi.charge * hemi.p**kappa, axis=-1))

    return _hemisphere_sum(forward), _hemisphere_sum(backward)


def hemisphere_net_charge(
    particles: ak.Array,
    thrust_vec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Net integer charge in each thrust-axis hemisphere.

    Used by the leptonic-asymmetry selection to require one hemisphere
    with net charge :math:`+1` and the other with :math:`-1`. The caller
    is responsible for passing only charged tracks (or quality-selected
    charged tracks); neutral entries contribute zero charge anyway, so
    passing the full particle list also works -- but the count of
    "non-zero net charge" tracks is determined by the input.
    """
    forward, backward = split_by_thrust_hemisphere(particles, thrust_vec)
    q_f = ak.to_numpy(ak.sum(forward.charge, axis=-1))
    q_b = ak.to_numpy(ak.sum(backward.charge, axis=-1))
    return q_f, q_b


def hemisphere_momentum_sums(
    particles: ak.Array,
    thrust_vec: np.ndarray,
) -> tuple[ak.Array, ak.Array]:
    r"""4-momentum sums of each thrust-axis hemisphere.

    Returns two Momentum4D arrays, one per hemisphere, containing the
    4-vector sum of the input particles on that side. Used by the
    :math:`\tau^+\tau^-` selection to compute the angle between the two
    hemisphere-visible momenta and each hemisphere's invariant mass.
    The caller controls whether neutrals are included.
    """
    forward, backward = split_by_thrust_hemisphere(particles, thrust_vec)
    return ak.sum(forward, axis=-1), ak.sum(backward, axis=-1)


def heavy_jet_mass(
    particles: ak.Array,
    thrust_vec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Normalised heavy jet mass :math:`\rho = M_H^2 / E_\text{vis}^2`.

    Events are split into two hemispheres by the plane perpendicular to
    the thrust axis. The heavy jet mass is the larger of the two
    hemisphere invariant masses. The result is normalised by the square
    of the total visible energy of the input collection.

    Parameters
    ----------
    particles : ak.Array
        Momentum4D collection (typically the inclusive list, to match
        the paper's definition of E_vis).
    thrust_vec : np.ndarray, shape (n_events, 3)
        Thrust axis from :func:`thrust`.

    Returns
    -------
    rho : np.ndarray, shape (n_events,)
        Normalised heavy jet mass squared in [0, ~0.25].
    M_H : np.ndarray, shape (n_events,)
        Unnormalised heavy jet mass in GeV.
    """
    fwd, bwd = split_by_thrust_hemisphere(particles, thrust_vec)
    p4_fwd = ak.sum(fwd, axis=-1)
    p4_bwd = ak.sum(bwd, axis=-1)

    M_a = np.asarray(p4_fwd.mass)
    M_b = np.asarray(p4_bwd.mass)
    M_H = np.maximum(M_a, M_b)

    E_vis = visible_energy(particles, charged_only=False)

    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.where(E_vis > 0, M_H**2 / E_vis**2, 0.0)

    return rho, M_H


# ---------------------------------------------------------------------------
# Per-event scalar observables
# ---------------------------------------------------------------------------


def charged_multiplicity(particles: ak.Array) -> np.ndarray:
    """Number of charged particles per event.

    Counts ``charge != 0`` in the input. To use a quality-selected set,
    pass that set in directly.
    """
    return ak.to_numpy(ak.num(particles[particles.charge != 0], axis=-1))


def visible_energy(particles: ak.Array, charged_only: bool = True) -> np.ndarray:
    """Visible energy per event [GeV].

    Parameters
    ----------
    particles : ak.Array
        Particle list to sum.
    charged_only : bool, optional
        If True (default), sum only particles with non-zero charge.
        If False, sum all particles in the input.
    """
    selected = particles[particles.charge != 0] if charged_only else particles
    return ak.to_numpy(ak.sum(selected.e, axis=-1))


def normalised_energy_imbalance(particles: ak.Array) -> np.ndarray:
    r"""Longitudinal energy imbalance :math:`|\sum p_z| / \sum |\vec p|`.

    SLD :math:`A_{LR}` selections require this to be :math:`< 0.6` to
    suppress beam-gas and initial-state-radiation events.
    """
    pz_sum = ak.to_numpy(ak.sum(particles.pz, axis=-1))
    pmag_sum = ak.to_numpy(ak.sum(particles.p, axis=-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(pmag_sum > 0, np.abs(pz_sum) / pmag_sum, 0.0)


def charged_invariant_mass(particles: ak.Array) -> np.ndarray:
    """Invariant mass of the sum of all charged particles in each event."""
    charged = particles[particles.charge != 0]
    total = ak.sum(charged, axis=-1)
    return np.asarray(total.mass)


def max_charged_momentum(particles: ak.Array) -> np.ndarray:
    """Maximum 3-momentum among the charged particles of each event [GeV]."""
    charged = particles[particles.charge != 0]
    return ak.to_numpy(ak.fill_none(ak.max(charged.p, axis=-1, mask_identity=True), 0.0))


# ---------------------------------------------------------------------------
# Two-object kinematics
# ---------------------------------------------------------------------------


def opening_angle(p1: ak.Array, p2: ak.Array) -> np.ndarray:
    """Opening angle between two 4-vectors [rad]."""
    return np.asarray(p1.deltaangle(p2))


def acollinearity(p1: ak.Array, p2: ak.Array) -> np.ndarray:
    r"""Acollinearity :math:`\pi - \alpha`, opening-angle complement [rad]."""
    return np.pi - np.asarray(p1.deltaangle(p2))


def momentum_imbalance(p1: ak.Array, p2: ak.Array) -> np.ndarray:
    """Fractional momentum imbalance between two 4-vectors."""
    mag1 = np.asarray(p1.p)
    mag2 = np.asarray(p2.p)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.abs(mag1 - mag2) / (mag1 + mag2)


def invariant_mass(*particles: ak.Array) -> np.ndarray:
    """Invariant mass of the sum of the given 4-vectors."""
    if not particles:
        raise ValueError("invariant_mass requires at least one input")
    total = particles[0]
    for p in particles[1:]:
        total = total + p
    return np.asarray(total.mass)
