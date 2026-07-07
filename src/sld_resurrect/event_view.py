"""Event-level observable computation for the SLD analysis.

This module owns the *observable* layer of the analysis:
:class:`EventView` takes raw event data, an inclusive particle list,
and a :class:`TrackQualityCuts` configuration, then provides on-demand,
memoised access to every event-level physics observable used by SLD
analyses (multiplicities, hemisphere quantities, thrust axis, LAC and
WIC aggregates, year of data taking, ...).

No event-level cuts are applied here -- this layer is purely
*computational*. :class:`sld_resurrect.selector.EventSelector` adds
cut application on top by inheritance, so anywhere only the observables
are needed (validation-plot notebooks, post-selection measurements,
custom studies on already-filtered data) an :class:`EventView` can
be used directly without supplying a dummy cut list.

Charged-particle quantities (``n_charged``, ``charged_mass``,
``thrust_vec_charged``, ``hem_*``, ``hem_top_track_lac_*``, ...) are
computed from the *quality-selected* charged tracks defined by
``track_quality``. Detector-global quantities (LAC and WIC totals,
``energy_imbalance``, ``event_year``, the inclusive thrust family, ...)
are computed from the unfiltered event record and bypass the quality
mask.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import awkward as ak
import numpy as np

from .kinematics import (
    hemisphere_momentum_sums,
    hemisphere_net_charge,
    normalised_energy_imbalance,
    opening_angle,
    orient_thrust_by_charge,
    split_by_beam_hemisphere,
    thrust,
    thrust_value,
    visible_energy,
)
from .track_quality import TrackQualityCuts, build_track_quality_mask

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# PHKLUS layer indices for LAC energy decomposition
# ---------------------------------------------------------------------------

LAC_EM_LAYERS: tuple[int, ...] = (0, 1)
"""``PHKLUS.elayer`` indices belonging to the EM section of the LAC."""

LAC_HAD_LAYERS: tuple[int, ...] = (2, 3, 4, 5, 6, 7)
"""``PHKLUS.elayer`` indices belonging to the HAD section of the LAC."""


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
    n_src_flat = len(src_ids_flat)

    # ak.num is layout-agnostic; reading .layout.offsets breaks on the
    # IndexedArray layout produced by row-slicing an event record.
    src_counts = ak.to_numpy(ak.num(src_ids, axis=1))
    src_event_idx = np.repeat(np.arange(len(src_counts)), src_counts)

    mult = np.int64(
        max(
            int(src_ids_flat.max()) + 1 if n_src_flat else 1,
            int(ref_ids_flat.max()) + 1 if len(ref_ids_flat) else 1,
            1,
        )
    )
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
# EventView
# ---------------------------------------------------------------------------


class EventView:
    """Compute (and cache) event-level physics observables.

    Takes raw event data + an inclusive particle list + a
    :class:`TrackQualityCuts` configuration; materialises the per-particle
    quality mask once at construction time; then exposes every event-level
    observable through :meth:`get`. Results are memoised, so requesting
    the same quantity twice is free.

    This is the *observable* layer of the analysis. To apply event-level
    cuts on top, use :class:`sld_resurrect.selector.EventSelector`, which
    inherits from this class.

    Parameters
    ----------
    data : ak.Array
        Full event record. Needed for quantities that depend on raw
        detector banks (LAC, WIC, PHPOINT, IEVENTH, ...).
    particles : ak.Array
        Inclusive particle collection (from
        :func:`sld_resurrect.kinematics.build_particles`).
    track_quality : TrackQualityCuts, optional
        Per-track quality cuts. Defaults to a no-op configuration in which
        every charged particle passes.

    Examples
    --------
    Build with a custom :class:`TrackQualityCuts`::

        ea = EventView(data, particles, track_quality=tq)
        cos_theta = ea.get('thrust_vec_charged')[:, 2]
    """

    def __init__(
        self,
        data: ak.Array,
        particles: ak.Array,
        *,
        track_quality: TrackQualityCuts | None = None,
    ) -> None:
        self._data = data
        self._particles = particles
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
        """The track-quality configuration in use."""
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

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"n_events={len(self._particles)}, "
            f"track_quality={self._track_quality.name!r}, "
            f"cached={sorted(self._cache)})"
        )

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
        # (the branches reading `data` below) use the unfiltered record.
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
            return ak.to_numpy(ak.fill_none(ak.max(charged.p, axis=-1, mask_identity=True), 0.0))
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
            # Charged thrust uses the quality-selected charged subset. The
            # axis is sign-resolved by the hemisphere net charges so that
            # cos(theta_T_charged) carries physical meaning (axis points
            # along the positive-net-charge hemisphere). Events whose
            # hemisphere charges are ambiguous are written with NaN axis
            # components, so any downstream cut on
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
            em_per_cluster = sum(data.PHKLUS.elayer[:, :, i] for i in LAC_EM_LAYERS)
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
            return ak.to_numpy(
                ak.fill_none(
                    ak.min(data.PHWIC.nlayexp, axis=-1, mask_identity=True),
                    0,
                )
            )
        if name == "wic_max_matchChi2":
            return ak.to_numpy(
                ak.fill_none(
                    ak.max(data.PHWIC.matchChi2, axis=-1, mask_identity=True),
                    np.inf,
                )
            )

        raise KeyError(
            f"Unknown quantity {name!r}. Register it via {type(self).__name__}.register_quantity."
        )

    def _compute_track_lac_quantities(self) -> None:
        """Populate the per-thrust-hemisphere LAC-energy cache entries.

        Two distinct per-hemisphere notions of "track LAC energy" are
        computed in this method, and they are NOT interchangeable:

        * **Leading-momentum track**, used by the ``ee`` selection (sum
          across hemispheres exceeds 45 GeV) and the ``mu mu`` ee-veto
          (max across hemispheres below 14 GeV in the 2001 paper). The
          relevant track per hemisphere is the highest-momentum quality
          charged track. The associated LAC energy is zero when that
          track has no matched cluster. Cached as
          ``hem_top_track_lac_{max,min,sum}``.

        * **Maximum-LAC track**, used by the 2001 ``tau tau`` ee-veto
          ("the maximum energy per hemisphere in the LAC associated to a
          charged track"). The relevant track per hemisphere is the
          quality charged track with the largest associated LAC cluster
          energy -- not necessarily the leading-momentum one. A tau
          decay can produce several charged hadrons of which the most
          electromagnetic-looking is rarely the highest-momentum one, so
          this veto needs to inspect every track. Cached as
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
            return ak.to_numpy(
                ak.fill_none(
                    ak.max(lac_sel, axis=-1, mask_identity=True),
                    0.0,
                )
            )

        max_fwd = _max_lac(in_fwd)
        max_bwd = _max_lac(~in_fwd)
        self._cache["hem_charged_track_lac_max"] = np.maximum(max_fwd, max_bwd)
