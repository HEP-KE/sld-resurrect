"""SLD event-selection presets, one per published analysis.

Each ``preset_*`` function returns a ``(Selection, TrackQualityCuts)`` pair
that reproduces the cuts used in one published SLD paper. Presets are
registered in :data:`PRESETS` and resolved by name through
:func:`get_preset`, which backs the classmethods
:meth:`sld_resurrect.event_view.EventView.from_preset` (observables only)
and :meth:`sld_resurrect.selector.EventSelector.from_preset` (observables
plus the preset's cut list).

Preset registry
---------------
Hadronic
~~~~~~~~
* ``alr_1994``        -- Calorimeter-led :math:`A_{LR}` selection
  (`hep-ex/9404001 <https://arxiv.org/abs/hep-ex/9404001>`__).
* ``alphas_1995``     -- Event-shape selection for :math:`\\alpha_s`
  (`hep-ex/9501003 <https://arxiv.org/abs/hep-ex/9501003>`__).
* ``rb_1998``         -- Vertex-mass :math:`R_b` selection
  (`hep-ex/9708015 <https://arxiv.org/abs/hep-ex/9708015>`__).
* ``alr_2000``        -- High-precision :math:`A_{LR}` (default hadronic)
  (`hep-ex/0004026 <https://arxiv.org/abs/hep-ex/0004026>`__).
* ``abc_2005``        -- :math:`A_b/A_c` vertex+kaon
  (`hep-ex/0410042 <https://arxiv.org/abs/hep-ex/0410042>`__).

Leptonic (2001 defaults, `hep-ex/0010015 <https://arxiv.org/abs/hep-ex/0010015>`__)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ``leptonic_default``  -- Common 2001 preselection only.
* ``leptonic_ee``       -- :math:`Z \\to e^+e^-`.
* ``leptonic_mumu``     -- :math:`Z \\to \\mu^+\\mu^-`.
* ``leptonic_tautau``   -- :math:`Z \\to \\tau^+\\tau^-`.

Leptonic (legacy 1997, `hep-ex/9704012 <https://arxiv.org/abs/hep-ex/9704012>`__)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ``leptonic_1997_ee``, ``leptonic_1997_mumu``, ``leptonic_1997_tautau``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .cuts import CutGroup, CutSpec, Selection
from .track_quality import COS_30_DEG, TrackQualityCuts

LEPTONIC_FIDUCIAL_BY_YEAR: dict[int, float] = {1996: 0.80, 1997: 0.90, 1998: 0.90}
"""Year-dependent ``|cos(theta_T)|`` fiducial of the 2001 leptonic preselection.

The 1996 run used the pre-upgrade central tracker (fiducial 0.8); the
1997-98 runs share the upgraded acceptance (0.9). The leptonic presets
build their fiducial cuts from this table, and the measurement layer
derives its geometric acceptance correction from the same values.
"""

# ---------------------------------------------------------------------------
# Hadronic presets
# ---------------------------------------------------------------------------


def preset_alr_1994() -> tuple[Selection, TrackQualityCuts]:
    r"""1994 :math:`A_{LR}` selection (hep-ex/9404001).

    Calorimeter-only -- no per-track quality cuts.
    """
    cuts: Selection = [
        CutGroup(
            name="n_lac_threshold",
            combine="or",
            description="LAC cluster multiplicity (barrel/endcap)",
            members=[
                CutGroup(
                    name="barrel_branch",
                    combine="and",
                    description=">=9 LAC in barrel",
                    members=[
                        CutSpec(
                            name="cos_t_le_08",
                            quantity="abs_cos_theta_thrust",
                            op="<=",
                            threshold=0.8,
                            description="|cos(theta_T)| <= 0.8",
                        ),
                        CutSpec(
                            name="n_lac_ge_9",
                            quantity="n_lac_clusters",
                            op=">=",
                            threshold=9,
                            description=">=9 LAC clusters",
                        ),
                    ],
                ),
                CutSpec(
                    name="n_lac_ge_12",
                    quantity="n_lac_clusters",
                    op=">=",
                    threshold=12,
                    description=">=12 LAC clusters (endcap)",
                ),
            ],
        ),
        CutSpec(
            name="e_lac_ge_22",
            quantity="lac_total_energy",
            op=">=",
            threshold=22.0,
            description="Total LAC energy [GeV]",
        ),
        CutSpec(
            name="eimb_lt_06",
            quantity="energy_imbalance",
            op="<",
            threshold=0.6,
            description="Normalised energy imbalance",
        ),
    ]
    return cuts, TrackQualityCuts(name="alr_1994_no_track_cuts")


def preset_alr_2000() -> tuple[Selection, TrackQualityCuts]:
    r"""2000 high-precision :math:`A_{LR}` selection (hep-ex/0004026).

    Track quality (all distances IP-relative): pT > 100 MeV, angle > 30 deg
    from beam, IP within 5 cm radially / 10 cm longitudinally.

    Beam-axis hemisphere track-count condition:
      (n_fwd >= 2 AND n_bwd >= 2) OR n_fwd >= 4 OR n_bwd >= 4.
    """
    quality = TrackQualityCuts(
        min_pt=0.100,
        max_abs_cos_theta=COS_30_DEG,
        max_r=5.0,
        max_abs_z=10.0,
        name="alr_2000_quality",
    )
    cuts: Selection = [
        CutGroup(
            name="beam_hem_topology",
            combine="or",
            description="Beam-hemisphere track topology",
            members=[
                CutGroup(
                    name="two_per_hem",
                    combine="and",
                    description=">=2 quality tracks per beam hemisphere",
                    members=[
                        CutSpec(
                            name="n_fwd_ge_2",
                            quantity="n_charged_beam_fwd",
                            op=">=",
                            threshold=2,
                            description=">=2 fwd (quality)",
                        ),
                        CutSpec(
                            name="n_bwd_ge_2",
                            quantity="n_charged_beam_bwd",
                            op=">=",
                            threshold=2,
                            description=">=2 bwd (quality)",
                        ),
                    ],
                ),
                CutSpec(
                    name="n_fwd_ge_4",
                    quantity="n_charged_beam_fwd",
                    op=">=",
                    threshold=4,
                    description=">=4 quality tracks in fwd hem",
                ),
                CutSpec(
                    name="n_bwd_ge_4",
                    quantity="n_charged_beam_bwd",
                    op=">=",
                    threshold=4,
                    description=">=4 quality tracks in bwd hem",
                ),
            ],
        ),
        CutSpec(
            name="e_lac_ge_22",
            quantity="lac_total_energy",
            op=">=",
            threshold=22.0,
            description="Total LAC energy [GeV]",
        ),
        CutSpec(
            name="eimb_lt_06",
            quantity="energy_imbalance",
            op="<",
            threshold=0.6,
            description="Normalised energy imbalance",
        ),
    ]
    return cuts, quality


def preset_alphas_1995() -> tuple[Selection, TrackQualityCuts]:
    r"""1995 :math:`\alpha_s` selection (hep-ex/9501003).

    Track quality (all distances IP-relative): pT > 150 MeV,
    |cos(theta)| < 0.80, IP within 5 cm radially / 10 cm longitudinally.
    """
    quality = TrackQualityCuts(
        min_pt=0.150,
        max_abs_cos_theta=0.80,
        max_r=5.0,
        max_abs_z=10.0,
        name="alphas_1995_quality",
    )
    cuts: Selection = [
        CutSpec(
            name="nch_ge_5",
            quantity="n_charged",
            op=">=",
            threshold=5,
            description=">=5 quality charged tracks",
        ),
        CutSpec(
            name="evis_ge_20",
            quantity="e_vis_charged",
            op=">=",
            threshold=20.0,
            description="Quality-track visible energy [GeV]",
        ),
        CutSpec(
            name="cos_theta_lt_071",
            quantity="abs_cos_theta_thrust",
            op="<",
            threshold=0.71,
            description="|cos(theta_thrust)|",
        ),
    ]
    return cuts, quality


def preset_rb_1998() -> tuple[Selection, TrackQualityCuts]:
    r"""1998 :math:`R_b` vertex-mass tag selection (hep-ex/9708015)."""
    cuts: Selection = [
        CutSpec(
            name="nch_ge_7",
            quantity="n_charged",
            op=">=",
            threshold=7,
            description="Number of charged particles",
        ),
        CutSpec(
            name="evis_ge_18",
            quantity="e_vis_charged",
            op=">=",
            threshold=18.0,
            description="Visible charged energy [GeV]",
        ),
        CutSpec(
            name="cos_theta_lt_071",
            quantity="abs_cos_theta_thrust",
            op="<",
            threshold=0.71,
            description="|cos(theta_thrust)|",
        ),
    ]
    return cuts, TrackQualityCuts(name="rb_1998_no_track_cuts")


def preset_abc_2005() -> tuple[Selection, TrackQualityCuts]:
    r"""2005 :math:`A_b` / :math:`A_c` vertex+kaon selection (hep-ex/0410042)."""
    cuts: Selection = [
        CutSpec(
            name="nch_ge_7",
            quantity="n_charged",
            op=">=",
            threshold=7,
            description="Number of charged particles",
        ),
        CutSpec(
            name="evis_ge_18",
            quantity="e_vis_charged",
            op=">=",
            threshold=18.0,
            description="Visible charged energy [GeV]",
        ),
        CutSpec(
            name="cos_theta_lt_070",
            quantity="abs_cos_theta_thrust",
            op="<",
            threshold=0.70,
            description="|cos(theta_thrust)|",
        ),
        CutSpec(
            name="T_gt_08", quantity="thrust_value", op=">", threshold=0.80, description="Thrust T"
        ),
    ]
    return cuts, TrackQualityCuts(name="abc_2005_no_track_cuts")


def preset_hadronic_default() -> tuple[Selection, TrackQualityCuts]:
    """Default hadronic selection: the 2000 high-precision A_LR setup."""
    return preset_alr_2000()


# ---------------------------------------------------------------------------
# Leptonic presets (2001 defaults)
# ---------------------------------------------------------------------------

# Both 1997 and 2001 leptonic-coupling papers require charged tracks to be
# within 1 cm of the e+e- interaction point.
LEPTONIC_QUALITY_2001 = TrackQualityCuts(max_d3=1.0, name="leptonic_2001_quality")
LEPTONIC_QUALITY_1997 = TrackQualityCuts(max_d3=1.0, name="leptonic_1997_quality")


def _leptonic_preselection_2001() -> Selection:
    """2001 common preselection: 2-8 quality tracks, opposite hem charges,
    year-dependent thrust fiducial."""
    return [
        CutSpec(
            name="nch_between",
            quantity="n_charged",
            op="between",
            threshold=(2, 8),
            description="2 <= n_charged (quality, IP <= 1 cm) <= 8",
        ),
        CutSpec(
            name="hem_q_pm1",
            quantity="hem_charges_opposite_unit",
            op="==",
            threshold=1,
            description="Thrust-hem net charges = {+1, -1}",
        ),
        CutGroup(
            name="fiducial_by_year",
            combine="or",
            description="Year-dependent |cos(theta_thrust)| fiducial",
            members=[
                CutGroup(
                    name="y1996",
                    combine="and",
                    description="1996: |cos(theta_T)| < 0.8",
                    members=[
                        CutSpec(
                            name="is_1996",
                            quantity="event_year",
                            op="==",
                            threshold=1996,
                            description="year == 1996",
                        ),
                        CutSpec(
                            name="cos_lt_08",
                            quantity="abs_cos_theta_thrust_charged",
                            op="<",
                            threshold=LEPTONIC_FIDUCIAL_BY_YEAR[1996],
                            description="|cos(theta_T_charged)| < 0.8",
                        ),
                    ],
                ),
                CutGroup(
                    name="y1997_98",
                    combine="and",
                    description="1997-98: |cos(theta_T)| < 0.9",
                    members=[
                        CutSpec(
                            name="is_1997_98",
                            quantity="event_year",
                            op="between",
                            threshold=(1997, 1998),
                            description="1997 <= year <= 1998",
                        ),
                        CutSpec(
                            name="cos_lt_09",
                            quantity="abs_cos_theta_thrust_charged",
                            op="<",
                            threshold=LEPTONIC_FIDUCIAL_BY_YEAR[1997],
                            description="|cos(theta_T_charged)| < 0.9",
                        ),
                    ],
                ),
            ],
        ),
    ]


def preset_leptonic_default() -> tuple[Selection, TrackQualityCuts]:
    """Inclusive leptonic preselection only (2001) -- no channel separation."""
    return _leptonic_preselection_2001(), LEPTONIC_QUALITY_2001


def preset_leptonic_ee() -> tuple[Selection, TrackQualityCuts]:
    r"""2001 :math:`Z \to e^+ e^-` selection (hep-ex/0010015)."""
    cuts = [
        *_leptonic_preselection_2001(),
        CutSpec(
            name="hem_lac_sum_gt_45",
            quantity="hem_top_track_lac_sum",
            op=">",
            threshold=45.0,
            description="Sum of per-hem leading-track LAC [GeV]",
        ),
    ]
    return cuts, LEPTONIC_QUALITY_2001


def preset_leptonic_mumu() -> tuple[Selection, TrackQualityCuts]:
    r"""2001 :math:`Z \to \mu^+ \mu^-` selection (hep-ex/0010015)."""
    cuts = [
        *_leptonic_preselection_2001(),
        CutSpec(
            name="mass_gt_70",
            quantity="charged_mass",
            op=">",
            threshold=70.0,
            description="Charged-track invariant mass [GeV]",
        ),
        CutSpec(
            name="hem_lac_lt_14",
            quantity="hem_top_track_lac_max",
            op="<",
            threshold=14.0,
            description="Max hem leading-track LAC energy [GeV]",
        ),
    ]
    return cuts, LEPTONIC_QUALITY_2001


def preset_leptonic_tautau() -> tuple[Selection, TrackQualityCuts]:
    r"""2001 :math:`Z \to \tau^+ \tau^-` selection (hep-ex/0010015)."""
    cuts = [
        *_leptonic_preselection_2001(),
        CutSpec(
            name="mass_lt_70",
            quantity="charged_mass",
            op="<",
            threshold=70.0,
            description="Charged-track invariant mass [GeV]",
        ),
        CutGroup(
            name="ee_veto_lac",
            combine="or",
            description="ee veto: cos-dependent LAC threshold",
            members=[
                CutGroup(
                    name="central_branch",
                    combine="and",
                    description="central: LAC < 39 GeV",
                    members=[
                        CutSpec(
                            name="cos_lt_07",
                            quantity="abs_cos_theta_thrust",
                            op="<",
                            threshold=0.7,
                            description="|cos(theta_T)| < 0.7",
                        ),
                        CutSpec(
                            name="lac_lt_39",
                            quantity="hem_charged_track_lac_max",
                            op="<",
                            threshold=39.0,
                            description="Max per-hem charged-track LAC < 39 GeV",
                        ),
                    ],
                ),
                CutGroup(
                    name="forward_branch",
                    combine="and",
                    description="forward: LAC < 33 GeV",
                    members=[
                        CutSpec(
                            name="cos_ge_07",
                            quantity="abs_cos_theta_thrust",
                            op=">=",
                            threshold=0.7,
                            description="|cos(theta_T)| >= 0.7",
                        ),
                        CutSpec(
                            name="lac_lt_33",
                            quantity="hem_charged_track_lac_max",
                            op="<",
                            threshold=33.0,
                            description="Max per-hem charged-track LAC < 33 GeV",
                        ),
                    ],
                ),
            ],
        ),
        CutSpec(
            name="hem_angle_gt_160",
            quantity="hem_opening_angle",
            op=">",
            threshold=float(np.deg2rad(160.0)),
            description="Hemisphere opening angle > 160 deg",
        ),
        CutSpec(
            name="max_p_gt_4",
            quantity="max_charged_p",
            op=">",
            threshold=4.0,
            description="Highest-p charged track > 4 GeV",
        ),
        CutSpec(
            name="hem_mass_lt_16",
            quantity="hem_invariant_mass_max",
            op="<",
            threshold=1.6,
            description="Max thrust-hem invariant mass < 1.6 GeV",
        ),
    ]
    return cuts, LEPTONIC_QUALITY_2001


# ---------------------------------------------------------------------------
# Leptonic presets (legacy 1997)
# ---------------------------------------------------------------------------


def _leptonic_preselection_1997() -> Selection:
    """1997 common preselection: tighter |cos| < 0.7, no year split."""
    return [
        CutSpec(
            name="nch_between",
            quantity="n_charged",
            op="between",
            threshold=(2, 8),
            description="2 <= n_charged (quality, IP <= 1 cm) <= 8",
        ),
        CutSpec(
            name="hem_q_pm1",
            quantity="hem_charges_opposite_unit",
            op="==",
            threshold=1,
            description="Thrust-hem net charges = {+1, -1}",
        ),
        CutSpec(
            name="cos_theta_lt_070",
            quantity="abs_cos_theta_thrust",
            op="<",
            threshold=0.70,
            description="|cos(theta_thrust)|",
        ),
    ]


def preset_leptonic_1997_ee() -> tuple[Selection, TrackQualityCuts]:
    r"""1997 :math:`Z \to e^+ e^-` (hep-ex/9704012)."""
    cuts = [
        *_leptonic_preselection_1997(),
        CutSpec(
            name="hem_lac_sum_gt_45",
            quantity="hem_top_track_lac_sum",
            op=">",
            threshold=45.0,
            description="Sum of per-hem leading-track LAC [GeV]",
        ),
    ]
    return cuts, LEPTONIC_QUALITY_1997


def preset_leptonic_1997_mumu() -> tuple[Selection, TrackQualityCuts]:
    r"""1997 :math:`Z \to \mu^+ \mu^-` (hep-ex/9704012)."""
    cuts = [
        *_leptonic_preselection_1997(),
        CutSpec(
            name="mass_gt_70",
            quantity="charged_mass",
            op=">",
            threshold=70.0,
            description="Charged-track invariant mass [GeV]",
        ),
        CutSpec(
            name="hem_lac_lt_10",
            quantity="hem_top_track_lac_max",
            op="<",
            threshold=10.0,
            description="Max hem leading-track LAC energy [GeV]",
        ),
        CutSpec(
            name="hem_lac_gt_0",
            quantity="hem_top_track_lac_min",
            op=">",
            threshold=0.0,
            description="Min hem leading-track LAC energy [GeV]",
        ),
    ]
    return cuts, LEPTONIC_QUALITY_1997


def preset_leptonic_1997_tautau() -> tuple[Selection, TrackQualityCuts]:
    r"""1997 :math:`Z \to \tau^+ \tau^-` (hep-ex/9704012)."""
    cuts = [
        *_leptonic_preselection_1997(),
        CutSpec(
            name="mass_lt_70",
            quantity="charged_mass",
            op="<",
            threshold=70.0,
            description="Charged-track invariant mass [GeV]",
        ),
        CutSpec(
            name="hem_lac_lt_275",
            quantity="hem_top_track_lac_max",
            op="<",
            threshold=27.5,
            description="Max hem leading-track LAC energy [GeV]",
        ),
        CutSpec(
            name="hem_lac_gt_0",
            quantity="hem_top_track_lac_min",
            op=">",
            threshold=0.0,
            description="Min hem leading-track LAC energy [GeV]",
        ),
        CutSpec(
            name="hem_angle_gt_160",
            quantity="hem_opening_angle",
            op=">",
            threshold=float(np.deg2rad(160.0)),
            description="Opening angle of hem momentum sums > 160 deg",
        ),
        CutSpec(
            name="max_p_gt_3",
            quantity="max_charged_p",
            op=">",
            threshold=3.0,
            description="Highest-p charged track [GeV]",
        ),
        CutSpec(
            name="hem_mass_lt_18",
            quantity="hem_invariant_mass_max",
            op="<",
            threshold=1.8,
            description="Max thrust-hem invariant mass [GeV]",
        ),
    ]
    return cuts, LEPTONIC_QUALITY_1997


# ---------------------------------------------------------------------------
# Registry & dispatcher
# ---------------------------------------------------------------------------

PresetFactory = Callable[[], tuple[Selection, TrackQualityCuts]]

PRESETS: dict[str, PresetFactory] = {
    # hadronic
    "hadronic_default": preset_hadronic_default,
    "alr_1994": preset_alr_1994,
    "alphas_1995": preset_alphas_1995,
    "rb_1998": preset_rb_1998,
    "alr_2000": preset_alr_2000,
    "abc_2005": preset_abc_2005,
    # leptonic (2001 defaults)
    "leptonic_default": preset_leptonic_default,
    "leptonic_ee": preset_leptonic_ee,
    "leptonic_mumu": preset_leptonic_mumu,
    "leptonic_tautau": preset_leptonic_tautau,
    # leptonic (legacy 1997)
    "leptonic_1997_ee": preset_leptonic_1997_ee,
    "leptonic_1997_mumu": preset_leptonic_1997_mumu,
    "leptonic_1997_tautau": preset_leptonic_1997_tautau,
}


def get_preset(name: str) -> tuple[Selection, TrackQualityCuts]:
    """Look up a preset by name and build its cut list and track quality.

    The single home for preset resolution: the ``from_preset``
    classmethods on :class:`~sld_resurrect.event_view.EventView` and
    :class:`~sld_resurrect.selector.EventSelector` both delegate here.

    Parameters
    ----------
    name : str
        Preset name (key of :data:`PRESETS`).

    Returns
    -------
    (Selection, TrackQualityCuts)
        Freshly-built cut list and the preset's track-quality model.

    Raises
    ------
    KeyError
        If ``name`` is not a registered preset.
    """
    if name not in PRESETS:
        raise KeyError(f"Unknown preset {name!r}. Available: {sorted(PRESETS)}")
    return PRESETS[name]()
