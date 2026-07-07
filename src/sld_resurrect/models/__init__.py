"""OmniLearn model loading, inference, and checkpoint management.

Submodules:

* :mod:`sld_resurrect.models.checkpoints` -- download/cache helpers (no torch).
* :mod:`sld_resurrect.models.loader` -- model loading from .pt files.
* :mod:`sld_resurrect.models.inference` -- single-GPU + distributed inference.
"""

from __future__ import annotations

__all__: list[str] = []
