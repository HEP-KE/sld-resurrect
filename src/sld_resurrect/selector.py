"""Event-selection framework for the SLD analysis.

Design
------
Track-quality selection is a property of the *selector*, configured at
construction time via a :class:`TrackQualityCuts` object. The selector
materialises the per-particle quality mask once and then uses the
quality-selected charged subset as the universal input for every
charged-particle observable -- multiplicities, hemisphere charges,
invariant masses, leading-track LAC matching, and so on. This guarantees
that all cuts in a given selection see the same set of tracks.

Concretely:

* Charged-particle quantities (``n_charged``, ``e_vis_charged``,
  ``charged_mass``, ``max_charged_p``, ``hem_*``,
  ``hem_top_track_lac_*``, ``hem_charged_track_lac_max``,
  ``n_charged_beam_*``,
  ``thrust_vec_charged``, ``abs_cos_theta_thrust_charged``) all use
  ``self._quality_charged``.
* Detector-global quantities (``lac_total_energy``, ``n_lac_clusters``,
  ``energy_imbalance``, ``e_vis_total``, ``thrust_vec``,
  ``abs_cos_theta_thrust``, ``thrust_value``, ``n_wic_*``,
  ``event_year``) ignore the quality mask -- they are intrinsic to the
  event, not derived from tracks.

The :class:`TrackQualityCuts` dataclass is flexible enough to express
every per-track requirement that appears in published SLD selections:
``min_pt``, ``max_abs_cos_theta``, ``max_r`` and ``max_abs_z`` (cylinder
in IP coordinates), and ``max_d3`` (3D distance to the IP). When
``relative_to_measured_ip=True`` the IP position is read per event from
``PHBM.pos`` rather than the origin -- this is what the leptonic
papers' "1 cm of the e+e- interaction point" cut requires.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Literal

import awkward as ak
import numpy as np

from .kinematics import (
    charged_invariant_mass,
    hemisphere_momentum_sums,
    hemisphere_net_charge,
    max_charged_momentum,
    normalised_energy_imbalance,
    opening_angle,
    orient_thrust_by_charge,
    split_by_beam_hemisphere,
    thrust,
    thrust_value,
    visible_energy,
)


# ---------------------------------------------------------------------------
# Constants & PHKLUS layer indices
# ---------------------------------------------------------------------------

LAC_EM_LAYERS: tuple[int, ...] = (0, 1)
"""``PHKLUS.elayer`` indices belonging to the EM section of the LAC."""

LAC_HAD_LAYERS: tuple[int, ...] = (2, 3, 4, 5, 6, 7)
"""``PHKLUS.elayer`` indices belonging to the HAD section of the LAC."""

COS_30_DEG: float = float(np.cos(np.deg2rad(30.0)))
"""Pre-computed :math:`\\cos(30^\\circ)`, used by the A_LR track-quality cut."""


# ---------------------------------------------------------------------------
# Track quality
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrackQualityCuts:
    """Per-track quality requirements for charged-particle selections.

    All distance cuts (``max_r``, ``max_abs_z``, ``max_d3``) are measured
    relative to the per-event interaction point read from ``PHBM.pos``.
    SLD beam-position monitors track the IP per fill, so this is the
    physically meaningful reference for every published selection.

    A cut is *disabled* by leaving its parameter at the sentinel value
    documented below. The defaults are deliberately permissive so that a
    selector created without an explicit ``TrackQualityCuts`` behaves
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

    def with_overrides(self, **changes: object) -> "TrackQualityCuts":
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
    caller (``build_track_quality_mask``) uses it to suppress every
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

    # Pad short events to length 3 along the coordinate axis with a
    # dummy 6-element row, then take column 0. Events with empty pos
    # get all-None for the three coords; we replace those with 0
    # because they will be suppressed via valid_ip downstream.
    dummy_row: list[float] = [0.0] * 3
    pos_padded = ak.fill_none(
        ak.pad_none(pos, target=1, axis=1, clip=True),
        dummy_row,
        axis=1
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
    """Build the per-particle boolean mask for a given ``TrackQualityCuts``.

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
    needs_ip = (
        cuts.max_r is not None
        or cuts.max_abs_z is not None
        or cuts.max_d3 is not None
    )
    if needs_ip:
        ip_x, ip_y, ip_z, valid_ip = _ip_position(data)
        dx = particles.vx - ip_x
        dy = particles.vy - ip_y
        dz = particles.vz - ip_z

        if cuts.max_r is not None:
            r = np.sqrt(dx ** 2 + dy ** 2)
            mask = mask & (r < cuts.max_r)

        if cuts.max_abs_z is not None:
            mask = mask & (np.abs(dz) < cuts.max_abs_z)

        if cuts.max_d3 is not None:
            d3 = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            mask = mask & (d3 < cuts.max_d3)

        # Events with no measured IP fail every track. ``valid_ip`` is a
        # per-event boolean; broadcasting it against the per-particle
        # jagged mask sets all tracks in invalid-IP events to False.
        mask = mask & valid_ip

    return mask


# ---------------------------------------------------------------------------
# Cluster-energy lookup helpers (used by the LAC-per-track quantities)
# ---------------------------------------------------------------------------

def _flat_id_lookup(
    ref_ids_flat: np.ndarray,
    ref_event_idx: np.ndarray,
    src_ids: ak.Array,
    src_values: ak.Array,
) -> np.ndarray:
    """Look up per-event id references in a source bank.

    Parameters
    ----------
    ref_ids_flat : np.ndarray
        Flat array of id values to resolve.
    ref_event_idx : np.ndarray
        Event index for each entry of ``ref_ids_flat``.
    src_ids : ak.Array
        Jagged array of ids in the source bank, one list per event.
    src_values : ak.Array
        Jagged array of the same shape as ``src_ids`` holding the values
        to return on a successful match.

    Returns
    -------
    np.ndarray
        Flat array of values, with 0 where no match is found.
    """
    src_ids_flat = ak.to_numpy(ak.flatten(src_ids, axis=1))
    src_values_flat = ak.to_numpy(ak.flatten(src_values, axis=1))
    src_offsets = ak.to_numpy(src_ids.layout.offsets)
    n_src_flat = len(src_ids_flat)

    src_counts = np.diff(src_offsets)
    src_event_idx = np.repeat(np.arange(len(src_counts)), src_counts)

    mult = np.int64(max(
        int(src_ids_flat.max()) + 1 if n_src_flat else 1,
        int(ref_ids_flat.max()) + 1 if len(ref_ids_flat) else 1,
        1,
    ))
    src_composite = src_event_idx.astype(np.int64) * mult + src_ids_flat.astype(np.int64)
    ref_composite = ref_event_idx.astype(np.int64) * mult + ref_ids_flat.astype(np.int64)

    sort_idx = np.argsort(src_composite)
    src_sorted = src_composite[sort_idx]
    positions = np.searchsorted(src_sorted, ref_composite)
    positions_clamped = np.clip(positions, 0, max(n_src_flat - 1, 0))
    matched = (n_src_flat > 0) & (src_sorted[positions_clamped] == ref_composite)
    src_row = sort_idx[positions_clamped]

    return np.where(matched, src_values_flat[src_row], 0.0)


# ---------------------------------------------------------------------------
# Cut specifications and groups
# ---------------------------------------------------------------------------

CompareOp = Literal["<", "<=", ">", ">=", "==", "!=", "between"]


@dataclass(frozen=True)
class CutSpec:
    """Declarative description of a single event-level cut.

    Parameters
    ----------
    name : str
        Short identifier used in cutflow tables and log messages.
    quantity : str
        Key of the quantity computed by :class:`EventSelector`. Either a
        built-in or a custom quantity registered via
        :meth:`EventSelector.register_quantity`.
    op : CompareOp
        Comparison to apply against ``threshold``. ``"between"`` takes a
        ``(lo, hi)`` tuple (inclusive on both ends).
    threshold : float | int | tuple
        Right-hand side of the comparison.
    description : str
        Free-form human-readable description (printed in cutflow tables).
    """

    name: str
    quantity: str
    op: CompareOp
    threshold: object
    description: str = ""

    def apply(self, values: np.ndarray) -> np.ndarray:
        threshold = self.threshold
        if self.op == "<":
            return values < threshold
        if self.op == "<=":
            return values <= threshold
        if self.op == ">":
            return values > threshold
        if self.op == ">=":
            return values >= threshold
        if self.op == "==":
            return values == threshold
        if self.op == "!=":
            return values != threshold
        if self.op == "between":
            lo, hi = threshold  # type: ignore[misc]
            return (values >= lo) & (values <= hi)
        raise ValueError(f"Unknown comparison operator: {self.op!r}")


@dataclass(frozen=True)
class CutGroup:
    """Collection of cuts combined with a common logical operator.

    Parameters
    ----------
    name : str
        Identifier used in cutflow tables.
    members : list of CutSpec or CutGroup
        The cuts (possibly nested groups) to combine.
    combine : {"and", "or"}
        How to combine the member results. Defaults to ``"or"``.
    description : str
        Human-readable description.
    """

    name: str
    members: list  # list[CutSpec | CutGroup]
    combine: Literal["and", "or"] = "or"
    description: str = ""


Selection = list  # list[CutSpec | CutGroup]


# ---------------------------------------------------------------------------
# Event selector
# ---------------------------------------------------------------------------

class EventSelector:
    """Apply event-level cuts to an awkward event sample.

    All charged-particle-derived quantities are computed from the
    *quality-selected* charged tracks defined by ``track_quality``. This
    guarantees consistency: the same set of tracks feeds the multiplicity
    cut, the hemisphere charges, the invariant mass, the leading-track
    LAC matching, and so on.

    Parameters
    ----------
    data : ak.Array
        Full event record. Needed for quantities that depend on raw
        detector banks (LAC, WIC, PHPOINT, ...).
    particles : ak.Array
        Inclusive particle collection (from ``build_particles``).
    cuts : list of CutSpec or CutGroup
        Cut specifications. Top-level elements are combined with AND.
    track_quality : TrackQualityCuts, optional
        Per-track quality cuts. Defaults to a no-op configuration (all
        charged particles pass).

    Notes
    -----
    Use :meth:`track_mask` to retrieve the per-particle quality mask,
    :meth:`quality_charged` to retrieve the corresponding charged-track
    collection, and :meth:`get` to read any cached event-level quantity.
    """

    # Quantities that intentionally bypass track quality.
    _GLOBAL_QUANTITIES: frozenset[str] = frozenset({
        "energy_imbalance",
        "e_vis_total",
        "thrust_vec",
        "abs_cos_theta_thrust",
        "thrust_value",
        "n_lac_clusters",
        "lac_total_energy",
        "lac_em_energy",
        "lac_em_fraction",
        "n_wic_matches",
        "wic_total_hits",
        "wic_min_nlayexp",
        "wic_max_matchChi2",
        "event_year",
    })

    def __init__(
        self,
        data: ak.Array,
        particles: ak.Array,
        cuts: Selection,
        *,
        track_quality: TrackQualityCuts | None = None,
    ) -> None:
        self._data = data
        self._particles = particles
        self._cuts = list(cuts)
        self._track_quality = track_quality or TrackQualityCuts()
        self._cache: dict[str, np.ndarray] = {}
        self._extra_quantities: dict[str, Callable[[], np.ndarray]] = {}

        # Build the quality mask once at construction time, so all
        # subsequent quantities see exactly the same set of tracks.
        self._quality_mask: ak.Array = build_track_quality_mask(
            particles=self._particles,
            cuts=self._track_quality,
            data=self._data,
        )
        self._quality_charged: ak.Array = self._particles[self._quality_mask]

    # ------------------------------------------------------------------ access
    @property
    def track_quality(self) -> TrackQualityCuts:
        """The track-quality configuration used by this selector."""
        return self._track_quality

    def track_mask(self) -> ak.Array:
        """Per-particle boolean mask of tracks passing the quality cuts."""
        return self._quality_mask

    def quality_charged(self) -> ak.Array:
        """Per-event quality-selected charged-particle list."""
        return self._quality_charged

    def register_quantity(
        self,
        name: str,
        func: Callable[[], np.ndarray],
    ) -> None:
        """Register a custom quantity builder, overriding any built-in."""
        self._extra_quantities[name] = func
        self._cache.pop(name, None)

    def get(self, name: str) -> np.ndarray:
        """Public accessor for a (possibly cached) derived quantity."""
        return self._get(name)

    # ------------------------------------------------------------------ quantities
    def _get(self, name: str) -> np.ndarray:
        cached = self._cache.get(name)
        if cached is not None:
            return cached
        if name in self._extra_quantities:
            values = self._extra_quantities[name]()
        else:
            values = self._compute_builtin(name)
        self._cache[name] = values
        return values

    def _compute_builtin(self, name: str) -> np.ndarray:
        # Quality-selected charged tracks are the universal input for all
        # charged-particle-derived quantities. Detector-global quantities
        # listed in _GLOBAL_QUANTITIES use the unfiltered event record.
        charged = self._quality_charged
        particles = self._particles
        data = self._data

        # ----- basic multiplicities & energies ------------------------------
        if name == "n_charged":
            return ak.to_numpy(ak.num(charged, axis=-1))
        if name == "e_vis_charged":
            return ak.to_numpy(ak.sum(charged.e, axis=-1))
        if name == "e_vis_total":
            return visible_energy(particles, charged_only=False)
        if name == "max_charged_p":
            # ``charged`` has only quality tracks, all of which are charged
            # by construction, so this is the leading-quality-track p.
            return ak.to_numpy(ak.fill_none(
                ak.max(charged.p, axis=-1, mask_identity=True), 0.0
            ))
        if name == "charged_mass":
            total = ak.sum(charged, axis=-1)
            return np.asarray(total.mass)

        # ----- thrust axis & |cos theta| (one eigendecomposition per pair) --
        if name in ("thrust_vec", "abs_cos_theta_thrust"):
            _, thrust_vec, cos_theta_t = thrust(particles)
            self._cache["thrust_vec"] = thrust_vec
            self._cache["abs_cos_theta_thrust"] = np.abs(cos_theta_t)
            return self._cache[name]
        if name in ("thrust_vec_charged", "abs_cos_theta_thrust_charged"):
            # Charged thrust uses the quality-selected charged subset for
            # consistency with the rest of the charged-particle stack. The
            # axis is sign-resolved by the hemisphere net charges so that
            # ``cos(theta_T_charged)`` carries physical meaning (axis
            # points along the negative-charge fermion direction). Events
            # whose hemisphere charges are ambiguous are written with NaN
            # axis components, so any downstream cut on
            # ``hem_charges_opposite_unit == 1`` naturally filters them
            # out before the NaNs propagate further.
            _, thrust_vec_raw, _ = thrust(charged)
            thrust_vec = orient_thrust_by_charge(thrust_vec_raw, charged)
            self._cache["thrust_vec_charged"] = thrust_vec
            self._cache["abs_cos_theta_thrust_charged"] = np.abs(thrust_vec[:, 2])
            return self._cache[name]
        if name == "thrust_value":
            return thrust_value(particles)
        if name == "energy_imbalance":
            return normalised_energy_imbalance(particles)

        # ----- beam-axis hemisphere counts (quality-track aware) ------------
        if name in ("n_charged_beam_fwd", "n_charged_beam_bwd"):
            fwd, bwd = split_by_beam_hemisphere(charged)
            self._cache["n_charged_beam_fwd"] = ak.to_numpy(ak.num(fwd, axis=-1))
            self._cache["n_charged_beam_bwd"] = ak.to_numpy(ak.num(bwd, axis=-1))
            return self._cache[name]

        # ----- year of data taking, derived from IEVENTH.evttime ------------
        # The Unix-epoch zero (evttime == 0) shows up for a handful of
        # events in the 1996-97 sample; treat those as 1997 since that is
        # where the bulk of the dataset lies.
        if name == "event_year":
            import pandas as pd
            evttime = ak.to_numpy(data.IEVENTH.evttime)
            year = pd.to_datetime(evttime, unit="s").year.values.astype(np.int32)
            return np.where(year == 1970, 1997, year)

        # ----- thrust-axis hemisphere quantities ----------------------------
        if name in (
            "hem_net_charge_fwd",
            "hem_net_charge_bwd",
            "hem_charges_opposite_unit",
        ):
            thrust_vec = self._get("thrust_vec_charged")
            q_f, q_b = hemisphere_net_charge(charged, thrust_vec)
            self._cache["hem_net_charge_fwd"] = q_f
            self._cache["hem_net_charge_bwd"] = q_b
            self._cache["hem_charges_opposite_unit"] = (
                ((q_f == 1) & (q_b == -1)) | ((q_f == -1) & (q_b == 1))
            ).astype(np.int8)
            return self._cache[name]

        if name in (
            "hem_opening_angle",
            "hem_invariant_mass_max",
            "hem_invariant_mass_min",
        ):
            thrust_vec = self._get("thrust_vec_charged")
            hem_fwd, hem_bwd = hemisphere_momentum_sums(charged, thrust_vec)
            angle = opening_angle(hem_fwd, hem_bwd)
            mass_f = np.asarray(hem_fwd.mass)
            mass_b = np.asarray(hem_bwd.mass)
            self._cache["hem_opening_angle"] = angle
            self._cache["hem_invariant_mass_max"] = np.maximum(mass_f, mass_b)
            self._cache["hem_invariant_mass_min"] = np.minimum(mass_f, mass_b)
            return self._cache[name]

        # ----- LAC cluster-energy-per-track quantities ----------------------
        if name in (
            "hem_top_track_lac_max",
            "hem_top_track_lac_min",
            "hem_top_track_lac_sum",
            "hem_charged_track_lac_max",
        ):
            self._compute_track_lac_quantities()
            return self._cache[name]

        # ----- LAC global cluster quantities --------------------------------
        if name == "n_lac_clusters":
            return ak.to_numpy(ak.num(data.PHKLUS.eraw, axis=-1))
        if name == "lac_total_energy":
            return ak.to_numpy(ak.sum(data.PHKLUS.eraw, axis=-1))
        if name == "lac_em_energy":
            em_per_cluster = sum(
                data.PHKLUS.elayer[:, :, i] for i in LAC_EM_LAYERS
            )
            return ak.to_numpy(ak.sum(em_per_cluster, axis=-1))
        if name == "lac_em_fraction":
            em = self._get("lac_em_energy")
            total = self._get("lac_total_energy")
            with np.errstate(invalid="ignore", divide="ignore"):
                return np.where(total > 0, em / total, 0.0)

        # ----- WIC muon quantities ------------------------------------------
        if name == "n_wic_matches":
            return ak.to_numpy(ak.num(data.PHWIC.nhit, axis=-1))
        if name == "wic_total_hits":
            return ak.to_numpy(ak.sum(data.PHWIC.nhit, axis=-1))
        if name == "wic_min_nlayexp":
            return ak.to_numpy(ak.fill_none(
                ak.min(data.PHWIC.nlayexp, axis=-1, mask_identity=True), 0,
            ))
        if name == "wic_max_matchChi2":
            return ak.to_numpy(ak.fill_none(
                ak.max(data.PHWIC.matchChi2, axis=-1, mask_identity=True), np.inf,
            ))

        raise KeyError(
            f"Unknown quantity {name!r}. Register it via "
            "EventSelector.register_quantity."
        )

    def _compute_track_lac_quantities(self) -> None:
        """Populate the per-thrust-hemisphere LAC-energy cache entries.

        Two distinct per-hemisphere notions of "track LAC energy" are
        computed in this method, and they are NOT interchangeable:

        * **Leading-momentum track**, used by the ``ee`` selection (sum
          across hemispheres exceeds 45 GeV) and the ``mu mu``
          ee-veto (max across hemispheres below 14 GeV in the 2001
          paper). The relevant track per hemisphere is the
          highest-momentum quality charged track. The associated LAC
          energy is zero when that track has no matched cluster.
          Cached as ``hem_top_track_lac_{max,min,sum}``.

        * **Maximum-LAC track**, used by the 2001 ``tau tau`` ee-veto
          ("the maximum energy per hemisphere in the LAC associated to
          a charged track"). The relevant track per hemisphere is the
          quality charged track with the largest associated LAC cluster
          energy -- not necessarily the leading-momentum one. A tau
          decay can produce several charged hadrons of which the most
          electromagnetic-looking is rarely the highest-momentum one,
          so this veto needs to inspect every track. Cached as
          ``hem_charged_track_lac_max``: the larger of the two
          hemisphere maxima (a single cut on this scalar enforces the
          paper's "each hemisphere" requirement).

        ``PHPOINT`` is row-aligned to ``PHPSUM`` 1:1, so the same
        per-particle quality mask selects the matching pointer entries.
        Cluster-energy lookups are therefore performed only over the
        quality-selected charged subset.
        """
        data = self._data
        charged = self._quality_charged
        quality_mask = self._quality_mask

        phpoint_quality = data.PHPOINT[quality_mask]
        klus_ids_flat = ak.to_numpy(ak.flatten(phpoint_quality.phklus_id, axis=1))
        counts = ak.to_numpy(ak.num(phpoint_quality, axis=-1))
        event_idx = np.repeat(np.arange(len(counts)), counts)

        lac_e_flat = _flat_id_lookup(
            klus_ids_flat,
            event_idx,
            data.PHKLUS.id,
            data.PHKLUS.eraw,
        )
        lac_e_flat = np.where(klus_ids_flat != 0, lac_e_flat, 0.0)
        lac_e = ak.unflatten(lac_e_flat, counts)  # aligned to ``charged``

        # Per-thrust-hemisphere split (shared by both notions below).
        charged_p = charged.p
        thrust_vec = self._get("thrust_vec_charged")
        nx = ak.Array(thrust_vec[:, 0])
        ny = ak.Array(thrust_vec[:, 1])
        nz = ak.Array(thrust_vec[:, 2])
        dot = charged.px * nx + charged.py * ny + charged.pz * nz
        in_fwd = dot > 0

        # ----- Leading-momentum-track LAC, per hemisphere -----
        def _lead_lac(hem_mask: ak.Array) -> np.ndarray:
            p_sel = ak.where(hem_mask, charged_p, np.float64(-np.inf))
            lac_sel = ak.where(hem_mask, lac_e, 0.0)
            has_any = ak.any(hem_mask, axis=-1)
            idx = ak.argmax(p_sel, axis=-1, keepdims=True)
            picked = ak.flatten(lac_sel[idx])
            return ak.to_numpy(ak.where(has_any, picked, 0.0))

        lead_fwd = _lead_lac(in_fwd)
        lead_bwd = _lead_lac(~in_fwd)
        self._cache["hem_top_track_lac_max"] = np.maximum(lead_fwd, lead_bwd)
        self._cache["hem_top_track_lac_min"] = np.minimum(lead_fwd, lead_bwd)
        self._cache["hem_top_track_lac_sum"] = lead_fwd + lead_bwd

        # ----- Max-LAC over any quality charged track, per hemisphere -----
        def _max_lac(hem_mask: ak.Array) -> np.ndarray:
            lac_sel = ak.where(hem_mask, lac_e, 0.0)
            return ak.to_numpy(ak.fill_none(
                ak.max(lac_sel, axis=-1, mask_identity=True), 0.0,
            ))

        max_fwd = _max_lac(in_fwd)
        max_bwd = _max_lac(~in_fwd)
        self._cache["hem_charged_track_lac_max"] = np.maximum(max_fwd, max_bwd)

    # ------------------------------------------------------------------ evaluate
    def _eval_element(self, element: object) -> np.ndarray:
        """Evaluate a CutSpec or CutGroup into a boolean per-event mask."""
        if isinstance(element, CutSpec):
            return element.apply(self._get(element.quantity))
        if isinstance(element, CutGroup):
            member_masks = [self._eval_element(m) for m in element.members]
            if not member_masks:
                return np.ones(len(self._particles), dtype=bool)
            if element.combine == "and":
                return np.logical_and.reduce(member_masks)
            if element.combine == "or":
                return np.logical_or.reduce(member_masks)
            raise ValueError(f"Unknown combine mode: {element.combine!r}")
        raise TypeError(
            f"Selection element must be CutSpec or CutGroup, "
            f"got {type(element).__name__}"
        )

    # ------------------------------------------------------------------ application
    def mask(self) -> np.ndarray:
        """Return the AND of all top-level cuts/groups as a boolean mask."""
        if not self._cuts:
            return np.ones(len(self._particles), dtype=bool)
        return np.logical_and.reduce([self._eval_element(c) for c in self._cuts])

    def apply(self) -> ak.Array:
        """Return the inclusive particle array filtered by :meth:`mask`."""
        return self._particles[self.mask()]

    def cutflow(self) -> list[dict[str, object]]:
        """Per-element yields assuming elements are applied sequentially."""
        total = len(self._particles)
        running = np.ones(total, dtype=bool)
        rows: list[dict[str, object]] = [
            {"cut": "initial", "description": "", "passed": total, "efficiency": 1.0}
        ]
        for element in self._cuts:
            running = running & self._eval_element(element)
            n_pass = int(running.sum())
            rows.append(
                {
                    "cut": element.name,
                    "description": element.description,
                    "passed": n_pass,
                    "efficiency": n_pass / total if total else 0.0,
                }
            )
        return rows

    def print_cutflow(self) -> None:
        """Pretty-print the cutflow table to stdout."""
        rows = self.cutflow()
        name_w = max(len(str(r["cut"])) for r in rows) + 2
        desc_w = max(len(str(r["description"])) for r in rows) + 2
        print(f"{'Cut':<{name_w}}{'Description':<{desc_w}}"
              f"{'Passed':>10}{'Efficiency':>14}")
        print("-" * (name_w + desc_w + 24))
        for row in rows:
            print(
                f"{str(row['cut']):<{name_w}}"
                f"{str(row['description']):<{desc_w}}"
                f"{row['passed']:>10,d}"
                f"{row['efficiency']:>13.2%}"
            )