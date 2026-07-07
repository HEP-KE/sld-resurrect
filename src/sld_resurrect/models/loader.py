"""Load pretrained OmniLearn (PET2) models from checkpoint files."""

from __future__ import annotations

from pathlib import Path

import torch

from sld_resurrect.models.checkpoints import CHECKPOINT_FILES
from sld_resurrect.paths import OMNILEARN_CHECKPOINT_DIR

__all__ = [
    "SIZE_ALIASES",
    "checkpoint_path_for",
    "load_omnilearned_model",
    "print_parameter_counts",
]


SIZE_ALIASES: dict[str, str] = {"s": "small", "m": "medium", "l": "large"}
"""One-letter to full-name mapping for model sizes."""


def _resolve_size(size: str) -> str:
    """Accept either a one-letter alias or a full name and return the full name."""
    if size in CHECKPOINT_FILES:
        return size
    if size in SIZE_ALIASES:
        return SIZE_ALIASES[size]
    raise ValueError(
        f"Unknown size {size!r}. Expected one of {list(SIZE_ALIASES)} or {list(CHECKPOINT_FILES)}."
    )


def checkpoint_path_for(
    size: str,
    checkpoint_dir: str | Path = OMNILEARN_CHECKPOINT_DIR,
) -> Path:
    """Return the checkpoint file path for a given model size.

    Parameters
    ----------
    size : str
        Model size, either a one-letter alias (``'s'``, ``'m'``, ``'l'``)
        or full name (``'small'``, ``'medium'``, ``'large'``).
    checkpoint_dir : path-like
        Directory containing the checkpoint files.
    """
    full = _resolve_size(size)
    return Path(checkpoint_dir) / CHECKPOINT_FILES[full]


def load_omnilearned_model(
    size: str,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device | None = None,
    *,
    input_dim: int = 4,
    num_classes: int = 210,
) -> torch.nn.Module:
    """Load the OmniLearn ``PET2`` model from a checkpoint.

    Parameters
    ----------
    size : str
        Model size: ``'s'``, ``'m'``, ``'l'`` (or full names).
    checkpoint_path : path-like or None
        Path to the ``.pt`` file. If ``None``, derive from
        :data:`sld_resurrect.paths.OMNILEARN_CHECKPOINT_DIR` and
        :data:`sld_resurrect.models.checkpoints.CHECKPOINT_FILES`.
    device : str, torch.device, or None
        Target device. If ``None``, ``"cuda"`` is used when available.
    input_dim : int
        Number of per-particle input features. OmniLearn was pre-trained
        with ``input_dim=4`` (``delta_eta, delta_phi, log pT, log E``).
    num_classes : int
        Classifier head output dimension. Default 210 matches the
        published pretraining classes.

    Returns
    -------
    torch.nn.Module
        The loaded ``PET2`` model in eval mode.
    """
    # Local imports keep ``import sld_resurrect.models`` cheap when
    # downstream code only needs the checkpoints utilities.
    from omnilearned.network import PET2
    from omnilearned.utils import get_model_parameters

    if checkpoint_path is None:
        checkpoint_path = checkpoint_path_for(size)
    else:
        checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    full = _resolve_size(size)
    model_params = get_model_parameters(full)
    model = PET2(input_dim=input_dim, **model_params, num_classes=num_classes)

    base_model = model.module if hasattr(model, "module") else model
    base_model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "body" in checkpoint:
        base_model.body.load_state_dict(checkpoint["body"], strict=False)
    if "classifier_head" in checkpoint:
        base_model.classifier.load_state_dict(checkpoint["classifier_head"], strict=False)

    base_model.eval()
    return base_model


def print_parameter_counts(model: torch.nn.Module) -> int:
    """Print total + trainable parameter counts in millions, return trainable count."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total / 1e6:.2f}M")
    print(f"Trainable params: {trainable / 1e6:.2f}M")
    return trainable
