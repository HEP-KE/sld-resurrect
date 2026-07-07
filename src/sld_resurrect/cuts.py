"""Declarative cut vocabulary for event selections.

:class:`CutSpec` describes one event-level cut on a named
:class:`~sld_resurrect.event_view.EventView` quantity; :class:`CutGroup`
combines cuts (or nested groups) with a common logical operator; a
``Selection`` is the list of top-level elements a
:class:`~sld_resurrect.selector.EventSelector` evaluates (combined with
AND). The published selections in
:mod:`sld_resurrect.selector_presets` are built from these primitives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np

__all__ = ["CompareOp", "CutGroup", "CutSpec", "Selection"]


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
