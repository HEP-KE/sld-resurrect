"""Durham (ee-kT) jet clustering helpers.

The single home for the exclusive 2-jet clustering recipe used by the
SLD hadronic pipeline, the ALEPH parser, and the demo notebooks.
"""

from __future__ import annotations

import awkward as ak

__all__ = ["cluster_two_jets"]


def cluster_two_jets(particles: ak.Array) -> tuple[ak.Array, ak.Array]:
    """Cluster each event into exactly two exclusive Durham jets.

    Parameters
    ----------
    particles : ak.Array
        Per-event ``Momentum4D`` particle collections.

    Returns
    -------
    jets : ak.Array, shape ``[n_events, 2]``
        Jet 4-vectors, pT-sorted with the leading jet at index 0.
    constituents : ak.Array, shape ``[n_events, 2, n_particles]``
        Constituent particles aligned with ``jets``.
    """
    # Imported here so the module is importable without fastjet installed.
    import fastjet

    # The swig-level JetDefinition is needed because fastjet's public
    # wrapper requires an R parameter, which ee_kt does not take.
    from fastjet._swig import JetDefinition

    jet_def = JetDefinition(fastjet.ee_kt_algorithm)
    constituents = fastjet.ClusterSequence(particles, jet_def).exclusive_jets_constituents(2)
    jets = ak.sum(constituents, axis=2)
    pt_order = ak.argsort(jets.pt, axis=1, ascending=False)
    return jets[pt_order], constituents[pt_order]
