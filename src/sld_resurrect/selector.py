"""Event-level selection on top of :class:`EventView`.

This module owns the *selection* layer of the analysis: declarative
:class:`CutSpec` / :class:`CutGroup` dataclasses for individual cuts and
:class:`EventSelector`, a specialisation of
:class:`sld_resurrect.event_view.EventView` that evaluates a
selection (list of cuts/groups) against the cached observables.

Charged-particle observables are still computed from the
quality-selected charged tracks defined by ``track_quality``, so a
multiplicity cut, a hemisphere-charge cut, and a leading-track LAC cut
all see exactly the same set of tracks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import awkward as ak
import numpy as np

from .event_view import EventView
from .track_quality import TrackQualityCuts

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
        Key of the quantity computed by :class:`EventView`. Either a
        built-in or a custom quantity registered via
        :meth:`EventView.register_quantity`.
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
            lo, hi = cast("tuple[float, float]", threshold)
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


class EventSelector(EventView):
    """Apply event-level cuts to an awkward event sample.

    Specialisation of :class:`EventView` that adds a list of
    :class:`CutSpec` / :class:`CutGroup` elements and the machinery to
    evaluate them (cut mask, sequential cutflow, pretty-printed table).
    Inherits the entire observable-computation machinery from the base
    class -- all charged-particle observables are computed from the same
    quality-selected charged tracks that the cuts see.

    Parameters
    ----------
    data, particles, track_quality
        Same as :class:`EventView`.
    cuts : list of CutSpec or CutGroup
        Cut specifications. Top-level elements are combined with AND. An
        empty list is a no-op selection (``mask()`` returns all-True);
        in that case this class is equivalent to :class:`EventView`.
    """

    def __init__(
        self,
        data: ak.Array,
        particles: ak.Array,
        cuts: Selection,
        *,
        track_quality: TrackQualityCuts | None = None,
    ) -> None:
        super().__init__(data=data, particles=particles, track_quality=track_quality)
        self._cuts = list(cuts)

    # ------------------------------------------------------------------ factories
    @classmethod
    def from_preset(
        cls,
        preset: str,
        data: ak.Array,
        particles: ak.Array,
        *,
        track_quality: TrackQualityCuts | None = None,
    ) -> EventSelector:
        """Build an :class:`EventSelector` configured from a named preset.

        Uses both the preset's *event-level cut list* and its
        *track-quality* configuration. ``track_quality`` overrides the
        preset's default when provided -- useful for systematics studies.

        Parameters
        ----------
        preset : str
            Preset name (key of
            :data:`sld_resurrect.selector_presets.PRESETS`).
        data, particles, track_quality
            Same as :meth:`__init__`.

        Raises
        ------
        KeyError
            If ``preset`` is not in
            :data:`sld_resurrect.selector_presets.PRESETS`.
        """
        # Imported lazily here to avoid the
        # ``selector_presets -> selector -> event_view`` import cycle.
        from .selector_presets import PRESETS

        if preset not in PRESETS:
            raise KeyError(f"Unknown preset {preset!r}. Available: {sorted(PRESETS)}")
        cuts, default_quality = PRESETS[preset]()
        return cls(
            data=data,
            particles=particles,
            cuts=cuts,
            track_quality=(track_quality if track_quality is not None else default_quality),
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"n_events={len(self._particles)}, "
            f"n_cuts={len(self._cuts)}, "
            f"track_quality={self._track_quality.name!r})"
        )

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
            f"Selection element must be CutSpec or CutGroup, got {type(element).__name__}"
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
        print(f"{'Cut':<{name_w}}{'Description':<{desc_w}}{'Passed':>10}{'Efficiency':>14}")
        print("-" * (name_w + desc_w + 24))
        for row in rows:
            print(
                f"{row['cut']!s:<{name_w}}"
                f"{row['description']!s:<{desc_w}}"
                f"{row['passed']:>10,d}"
                f"{row['efficiency']:>13.2%}"
            )
