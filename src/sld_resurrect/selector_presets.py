"""SLD event-selection presets, one per published analysis.

Each ``preset_*`` function returns a ``(Selection, TrackQualityCuts)`` pair
that reproduces the cuts used in one published SLD paper. Presets are
registered in :data:`PRESETS` and instantiated through the classmethods
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

from typing import Callable

import numpy as np

from .selector import CutGroup, CutSpec, Selection
from .track_quality import COS_30_DEG, TrackQualityCuts


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
                        CutSpec("cos_t_le_08", "abs_cos_theta_thrust", "<=", 0.8,
                                "|cos(theta_T)| <= 0.8"),
                        CutSpec("n_lac_ge_9", "n_lac_clusters", ">=", 9,
                                ">=9 LAC clusters"),
                    ],
                ),
                CutSpec("n_lac_ge_12", "n_lac_clusters", ">=", 12,
                        ">=12 LAC clusters (endcap)"),
            ],
        ),
        CutSpec("e_lac_ge_22", "lac_total_energy", ">=", 22.0,
                "Total LAC energy [GeV]"),
        CutSpec("eimb_lt_06", "energy_imbalance", "<", 0.6,
                "Normalised energy imbalance"),
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
                        CutSpec("n_fwd_ge_2", "n_charged_beam_fwd", ">=", 2,
                                ">=2 fwd (quality)"),
                        CutSpec("n_bwd_ge_2", "n_charged_beam_bwd", ">=", 2,
                                ">=2 bwd (quality)"),
                    ],
                ),
                CutSpec("n_fwd_ge_4", "n_charged_beam_fwd", ">=", 4,
                        ">=4 quality tracks in fwd hem"),
                CutSpec("n_bwd_ge_4", "n_charged_beam_bwd", ">=", 4,
                        ">=4 quality tracks in bwd hem"),
            ],
        ),
        CutSpec("e_lac_ge_22", "lac_total_energy", ">=", 22.0,
                "Total LAC energy [GeV]"),
        CutSpec("eimb_lt_06", "energy_imbalance", "<", 0.6,
                "Normalised energy imbalance"),
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
        CutSpec("nch_ge_5", "n_charged", ">=", 5,
                ">=5 quality charged tracks"),
        CutSpec("evis_ge_20", "e_vis_charged", ">=", 20.0,
                "Quality-track visible energy [GeV]"),
        CutSpec("cos_theta_lt_071", "abs_cos_theta_thrust", "<", 0.71,
                "|cos(theta_thrust)|"),
    ]
    return cuts, quality


def preset_rb_1998() -> tuple[Selection, TrackQualityCuts]:
    r"""1998 :math:`R_b` vertex-mass tag selection (hep-ex/9708015)."""
    cuts: Selection = [
        CutSpec("nch_ge_7", "n_charged", ">=", 7,
                "Number of charged particles"),
        CutSpec("evis_ge_18", "e_vis_charged", ">=", 18.0,
                "Visible charged energy [GeV]"),
        CutSpec("cos_theta_lt_071", "abs_cos_theta_thrust", "<", 0.71,
                "|cos(theta_thrust)|"),
    ]
    return cuts, TrackQualityCuts(name="rb_1998_no_track_cuts")


def preset_abc_2005() -> tuple[Selection, TrackQualityCuts]:
    r"""2005 :math:`A_b` / :math:`A_c` vertex+kaon selection (hep-ex/0410042)."""
    cuts: Selection = [
        CutSpec("nch_ge_7", "n_charged", ">=", 7,
                "Number of charged particles"),
        CutSpec("evis_ge_18", "e_vis_charged", ">=", 18.0,
                "Visible charged energy [GeV]"),
        CutSpec("cos_theta_lt_070", "abs_cos_theta_thrust", "<", 0.70,
                "|cos(theta_thrust)|"),
        CutSpec("T_gt_08", "thrust_value", ">", 0.80,
                "Thrust T"),
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
        CutSpec("nch_between", "n_charged", "between", (2, 8),
                "2 <= n_charged (quality, IP <= 1 cm) <= 8"),
        CutSpec("hem_q_pm1", "hem_charges_opposite_unit", "==", 1,
                "Thrust-hem net charges = {+1, -1}"),
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
                        CutSpec("is_1996", "event_year", "==", 1996,
                                "year == 1996"),
                        CutSpec("cos_lt_08", "abs_cos_theta_thrust_charged", "<", 0.80,
                                "|cos(theta_T_charged)| < 0.8"),
                    ],
                ),
                CutGroup(
                    name="y1997_98",
                    combine="and",
                    description="1997-98: |cos(theta_T)| < 0.9",
                    members=[
                        CutSpec("is_1997_98", "event_year", "between", (1997, 1998),
                                "1997 <= year <= 1998"),
                        CutSpec("cos_lt_09", "abs_cos_theta_thrust_charged", "<", 0.90,
                                "|cos(theta_T_charged)| < 0.9"),
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
    cuts = _leptonic_preselection_2001() + [
        CutSpec("hem_lac_sum_gt_45", "hem_top_track_lac_sum", ">", 45.0,
                "Sum of per-hem leading-track LAC [GeV]"),
    ]
    return cuts, LEPTONIC_QUALITY_2001


def preset_leptonic_mumu() -> tuple[Selection, TrackQualityCuts]:
    r"""2001 :math:`Z \to \mu^+ \mu^-` selection (hep-ex/0010015)."""
    cuts = _leptonic_preselection_2001() + [
        CutSpec("mass_gt_70", "charged_mass", ">", 70.0,
                "Charged-track invariant mass [GeV]"),
        CutSpec("hem_lac_lt_14", "hem_top_track_lac_max", "<", 14.0,
                "Max hem leading-track LAC energy [GeV]"),
    ]
    return cuts, LEPTONIC_QUALITY_2001


def preset_leptonic_tautau() -> tuple[Selection, TrackQualityCuts]:
    r"""2001 :math:`Z \to \tau^+ \tau^-` selection (hep-ex/0010015)."""
    cuts = _leptonic_preselection_2001() + [
        CutSpec("mass_lt_70", "charged_mass", "<", 70.0,
                "Charged-track invariant mass [GeV]"),
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
                        CutSpec("cos_lt_07", "abs_cos_theta_thrust", "<", 0.7,
                                "|cos(theta_T)| < 0.7"),
                        CutSpec("lac_lt_39", "hem_charged_track_lac_max", "<", 39.0,
                                "Max per-hem charged-track LAC < 39 GeV"),
                    ],
                ),
                CutGroup(
                    name="forward_branch",
                    combine="and",
                    description="forward: LAC < 33 GeV",
                    members=[
                        CutSpec("cos_ge_07", "abs_cos_theta_thrust", ">=", 0.7,
                                "|cos(theta_T)| >= 0.7"),
                        CutSpec("lac_lt_33", "hem_charged_track_lac_max", "<", 33.0,
                                "Max per-hem charged-track LAC < 33 GeV"),
                    ],
                ),
            ],
        ),
        CutSpec("hem_angle_gt_160", "hem_opening_angle", ">",
                float(np.deg2rad(160.0)),
                "Hemisphere opening angle > 160 deg"),
        CutSpec("max_p_gt_4", "max_charged_p", ">", 4.0,
                "Highest-p charged track > 4 GeV"),
        CutSpec("hem_mass_lt_16", "hem_invariant_mass_max", "<", 1.6,
                "Max thrust-hem invariant mass < 1.6 GeV"),
    ]
    return cuts, LEPTONIC_QUALITY_2001


# ---------------------------------------------------------------------------
# Leptonic presets (legacy 1997)
# ---------------------------------------------------------------------------

def _leptonic_preselection_1997() -> Selection:
    """1997 common preselection: tighter |cos| < 0.7, no year split."""
    return [
        CutSpec("nch_between", "n_charged", "between", (2, 8),
                "2 <= n_charged (quality, IP <= 1 cm) <= 8"),
        CutSpec("hem_q_pm1", "hem_charges_opposite_unit", "==", 1,
                "Thrust-hem net charges = {+1, -1}"),
        CutSpec("cos_theta_lt_070", "abs_cos_theta_thrust", "<", 0.70,
                "|cos(theta_thrust)|"),
    ]


def preset_leptonic_1997_ee() -> tuple[Selection, TrackQualityCuts]:
    r"""1997 :math:`Z \to e^+ e^-` (hep-ex/9704012)."""
    cuts = _leptonic_preselection_1997() + [
        CutSpec("hem_lac_sum_gt_45", "hem_top_track_lac_sum", ">", 45.0,
                "Sum of per-hem leading-track LAC [GeV]"),
    ]
    return cuts, LEPTONIC_QUALITY_1997


def preset_leptonic_1997_mumu() -> tuple[Selection, TrackQualityCuts]:
    r"""1997 :math:`Z \to \mu^+ \mu^-` (hep-ex/9704012)."""
    cuts = _leptonic_preselection_1997() + [
        CutSpec("mass_gt_70", "charged_mass", ">", 70.0,
                "Charged-track invariant mass [GeV]"),
        CutSpec("hem_lac_lt_10", "hem_top_track_lac_max", "<", 10.0,
                "Max hem leading-track LAC energy [GeV]"),
        CutSpec("hem_lac_gt_0", "hem_top_track_lac_min", ">", 0.0,
                "Min hem leading-track LAC energy [GeV]"),
    ]
    return cuts, LEPTONIC_QUALITY_1997


def preset_leptonic_1997_tautau() -> tuple[Selection, TrackQualityCuts]:
    r"""1997 :math:`Z \to \tau^+ \tau^-` (hep-ex/9704012)."""
    cuts = _leptonic_preselection_1997() + [
        CutSpec("mass_lt_70", "charged_mass", "<", 70.0,
                "Charged-track invariant mass [GeV]"),
        CutSpec("hem_lac_lt_275", "hem_top_track_lac_max", "<", 27.5,
                "Max hem leading-track LAC energy [GeV]"),
        CutSpec("hem_lac_gt_0", "hem_top_track_lac_min", ">", 0.0,
                "Min hem leading-track LAC energy [GeV]"),
        CutSpec("hem_angle_gt_160", "hem_opening_angle", ">",
                float(np.deg2rad(160.0)),
                "Opening angle of hem momentum sums > 160 deg"),
        CutSpec("max_p_gt_3", "max_charged_p", ">", 3.0,
                "Highest-p charged track [GeV]"),
        CutSpec("hem_mass_lt_18", "hem_invariant_mass_max", "<", 1.8,
                "Max thrust-hem invariant mass [GeV]"),
    ]
    return cuts, LEPTONIC_QUALITY_1997


# ---------------------------------------------------------------------------
# Registry & dispatcher
# ---------------------------------------------------------------------------

PresetFactory = Callable[[], tuple[Selection, TrackQualityCuts]]

PRESETS: dict[str, PresetFactory] = {
    # hadronic
    "hadronic_default":     preset_hadronic_default,
    "alr_1994":             preset_alr_1994,
    "alphas_1995":          preset_alphas_1995,
    "rb_1998":              preset_rb_1998,
    "alr_2000":             preset_alr_2000,
    "abc_2005":             preset_abc_2005,
    # leptonic (2001 defaults)
    "leptonic_default":     preset_leptonic_default,
    "leptonic_ee":          preset_leptonic_ee,
    "leptonic_mumu":        preset_leptonic_mumu,
    "leptonic_tautau":      preset_leptonic_tautau,
    # leptonic (legacy 1997)
    "leptonic_1997_ee":     preset_leptonic_1997_ee,
    "leptonic_1997_mumu":   preset_leptonic_1997_mumu,
    "leptonic_1997_tautau": preset_leptonic_1997_tautau,
}