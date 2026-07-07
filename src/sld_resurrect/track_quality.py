"""Per-track quality cuts for SLD charged-particle observables.

This module owns the *track-quality* layer of the analysis: the
:class:`TrackQualityCuts` dataclass that declares per-track requirements
(:math:`p_T`, :math:`|\\cos\\theta|`, impact-parameter cuts), and the
:func:`build_track_quality_mask` function that materialises the
corresponding per-particle boolean mask against an event record.

Impact-parameter cuts are measured relative to the per-event interaction
point read from the ``PHBM.pos`` bank. SLD beam-position monitors track
the IP per fill, so this is the physically meaningful reference for every
published selection. The few events with an empty ``PHBM.pos`` (no
measured IP) are treated as failing every IP-relative cut.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import awkward as ak
import numpy as np

# Pre-computed cos(30 deg), used by the 1994/2000 A_LR track-quality cuts.
COS_30_DEG: float = float(np.cos(np.deg2rad(30.0)))


@dataclass(frozen=True)
class TrackQualityCuts:
    """Per-track quality requirements for charged-particle selections.

    All distance cuts (``max_r``, ``max_abs_z``, ``max_d3``) are measured
    relative to the per-event interaction point read from ``PHBM.pos``.
    SLD beam-position monitors track the IP per fill, so this is the
    physically meaningful reference for every published selection.

    A cut is *disabled* by leaving its parameter at the sentinel value
    documented below. The defaults are deliberately permissive so that an
    analyzer created without an explicit ``TrackQualityCuts`` behaves
    like the unfiltered ``particles[particles.charge != 0]`` selection.

    Parameters
    ----------
    min_pt : float, optional
        Minimum transverse momentum [GeV]. Disabled when ``<= 0``.
    max_abs_cos_theta : float, optional
        Maximum :math:`|\\cos\\theta|` with respect to the beam axis.
        Disabled when ``>= 1``.
    max_r : float or None, optional
        Maximum radial impact parameter [cm] in the beam-perpendicular
        plane, measured from the IP. Disabled when ``None``.
    max_abs_z : float or None, optional
        Maximum longitudinal impact parameter [cm], measured from the
        IP. Disabled when ``None``.
    max_d3 : float or None, optional
        Maximum 3D distance [cm] from the IP. Disabled when ``None``.
    name : str, optional
        Short identifier used in log messages and cutflow tables.
    """

    min_pt: float = 0.0
    max_abs_cos_theta: float = 1.0
    max_r: float | None = None
    max_abs_z: float | None = None
    max_d3: float | None = None
    name: str = "track_quality"

    def is_active(self) -> bool:
        """Return True if any per-track cut is configured."""
        return (
            self.min_pt > 0.0
            or self.max_abs_cos_theta < 1.0
            or self.max_r is not None
            or self.max_abs_z is not None
            or self.max_d3 is not None
        )

    def with_overrides(self, **changes: object) -> TrackQualityCuts:
        """Return a copy with the given fields replaced.

        Useful for systematics studies, e.g.::

            preset_cuts = TrackQualityCuts(min_pt=0.10, max_r=5.0, max_abs_z=10.0)
            tight = preset_cuts.with_overrides(max_r=2.5, max_abs_z=5.0)
        """
        return replace(self, **changes)  # type: ignore[arg-type]


def _ip_position(
    data: ak.Array,
) -> tuple[ak.Array, ak.Array, ak.Array, ak.Array]:
    """Return per-event IP ``(x, y, z)`` from ``PHBM.pos`` plus a validity mask.

    ``PHBM.pos`` is variable-length per event in jazelle output: a small
    fraction of events have an empty ``pos`` (i.e.
    ``ak.num(data.PHBM.pos) == 0``). For those events no measured IP is
    available, and any IP-relative quality cut should fail every track
    in the event. The returned validity mask flags such events; the
    caller (:func:`build_track_quality_mask`) uses it to suppress every
    track in an invalid-IP event.

    For events with valid ``pos``, the layout is ``[3, 6]`` -- three
    coordinates plus a 6-element packed error matrix. The IP coordinates
    are the first column.

    Returns
    -------
    ip_x, ip_y, ip_z : ak.Array
        Per-event IP coordinates [cm]. Set to 0 for invalid-IP events
        (the value is irrelevant since every track in those events is
        already masked out via ``valid_ip``).
    valid_ip : ak.Array
        Per-event boolean mask, True when ``PHBM.pos`` has at least one
        entry along its outer (coordinate) axis.
    """
    pos = data.PHBM.pos
    valid_ip = ak.num(pos, axis=1) > 0

    dummy_row: list[float] = [0.0] * 3
    pos_padded = ak.fill_none(
        ak.pad_none(pos, target=1, axis=1, clip=True),
        dummy_row,
        axis=1,
    )
    ip_x = pos_padded[:, 0, 0]
    ip_y = pos_padded[:, 0, 1]
    ip_z = pos_padded[:, 0, 2]
    return ip_x, ip_y, ip_z, valid_ip


def build_track_quality_mask(
    particles: ak.Array,
    cuts: TrackQualityCuts,
    data: ak.Array,
) -> ak.Array:
    """Build the per-particle boolean mask for a given :class:`TrackQualityCuts`.

    The mask is True for *charged* particles that pass every active cut.
    Neutral particles are always excluded. All distance-based cuts are
    measured relative to the per-event IP from ``PHBM.pos``.

    Parameters
    ----------
    particles : ak.Array
        Inclusive particle list (output of
        :func:`kinematics.build_particles`).
    cuts : TrackQualityCuts
        Cut configuration.
    data : ak.Array
        Full event record. Used to read ``PHBM.pos`` for the IP position.

    Returns
    -------
    ak.Array
        Boolean mask jagged-aligned to ``particles``.
    """
    mask = particles.charge != 0

    if cuts.min_pt > 0.0:
        mask = mask & (particles.pt > cuts.min_pt)

    if cuts.max_abs_cos_theta < 1.0:
        pmag = particles.p
        cos_theta = ak.where(pmag > 0, particles.pz / pmag, 0.0)
        mask = mask & (np.abs(cos_theta) < cuts.max_abs_cos_theta)

    # All distance cuts are measured relative to the per-event IP. Read it
    # once so the three branches below share the same coordinates.
    needs_ip = cuts.max_r is not None or cuts.max_abs_z is not None or cuts.max_d3 is not None
    if needs_ip:
        ip_x, ip_y, ip_z, valid_ip = _ip_position(data)
        dx = particles.vx - ip_x
        dy = particles.vy - ip_y
        dz = particles.vz - ip_z

        if cuts.max_r is not None:
            r = np.sqrt(dx**2 + dy**2)
            mask = mask & (r < cuts.max_r)

        if cuts.max_abs_z is not None:
            mask = mask & (np.abs(dz) < cuts.max_abs_z)

        if cuts.max_d3 is not None:
            d3 = np.sqrt(dx**2 + dy**2 + dz**2)
            mask = mask & (d3 < cuts.max_d3)

        # Events with no measured IP fail every track. ``valid_ip`` is a
        # per-event boolean; broadcasting it against the per-particle
        # jagged mask sets all tracks in invalid-IP events to False.
        mask = mask & valid_ip

    return mask
