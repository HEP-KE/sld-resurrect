"""Dimensionality reduction (t-SNE / UMAP) for OmniLearn embeddings.

The two public entry points :func:`get_tsne_embedding` and
:func:`get_umap_embedding` accept any preloaded NumPy embedding,
optionally apply standard scaling + PCA, and run the requested
reduction with GPU acceleration via cuML when available (falling back
to scikit-learn / openTSNE / umap-learn on CPU).

For the SLD use case there are also two convenience helpers:
:func:`load_pooled_embedding` reads an OmniLearn embedding ``.h5`` file
and applies mean pooling over the token axis;
:func:`embedding_path_for` builds the conventional file path under
:data:`sld_resurrect.paths.OMNILEARN_EMBEDDING_DIR`.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from tqdm.auto import tqdm

from sld_resurrect.paths import OMNILEARN_EMBEDDING_DIR

__all__ = [
    "DATASET_KEY",
    "embedding_path_for",
    "get_tsne_embedding",
    "get_umap_embedding",
    "load_pooled_embedding",
]


DATASET_KEY: str = "data"
"""HDF5 dataset key inside the embedding files."""


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def embedding_path_for(
    dataset: str,
    size: str,
    base_dir: str | Path = OMNILEARN_EMBEDDING_DIR,
) -> Path:
    """Return the conventional embedding file path for ``(dataset, size)``.

    Parameters
    ----------
    dataset : str
        Dataset key, e.g. ``"jetclass1_HToBB"``.
    size : str
        Model size, one of ``"s"``, ``"m"``, ``"l"``.
    base_dir : path-like
        Directory containing the embedding files.
    """
    return Path(base_dir) / f"omnilearned_embedding_{size}_{dataset}.h5"


def load_pooled_embedding(
    path: str | Path,
    max_events: int | None = None,
) -> np.ndarray:
    """Load an embedding file and mean-pool over the token axis.

    Parameters
    ----------
    path : path-like
        ``.h5`` file containing a dataset of shape
        ``(n_events, n_particles, hidden_size)`` under
        :data:`DATASET_KEY`.
    max_events : int or None
        Load at most this many events; ``None`` loads all events.

    Returns
    -------
    np.ndarray, shape (n_events, hidden_size), dtype float32
        Mean-pooled embedding with NaN rows dropped.

    Raises
    ------
    ValueError
        If ``max_events`` is neither ``None`` nor a positive integer.
    """
    if max_events is not None and max_events <= 0:
        raise ValueError(
            f"max_events must be a positive count or None for all events; got {max_events}"
        )
    path = Path(path)
    with h5py.File(path, "r") as f:
        raw = f[DATASET_KEY][:] if max_events is None else f[DATASET_KEY][:max_events]
    pooled = raw.mean(axis=1).astype(np.float32)

    nan_mask = np.isnan(pooled).any(axis=1)
    n_nan = int(nan_mask.sum())
    if n_nan > 0:
        tqdm.write(f"  WARNING: dropped {n_nan} NaN rows from {path.name}")
        pooled = pooled[~nan_mask]
    return pooled


# ---------------------------------------------------------------------------
# Compute helpers
# ---------------------------------------------------------------------------


def _resolve_gpu(device: str, *, verbose: bool = False) -> bool:
    """Return ``True`` if GPU should be used, else ``False`` (with fallback)."""
    if device != "cuda":
        return False
    try:
        import torch

        if not torch.cuda.is_available():
            if verbose:
                tqdm.write("CUDA not available, falling back to CPU.")
            return False
    except ImportError:
        if verbose:
            tqdm.write("torch not available, falling back to CPU.")
        return False
    return True


def _stage(message: str, pbar: tqdm | None) -> None:
    """Update a tqdm bar's postfix, or write to stdout if no bar."""
    if pbar is not None:
        pbar.set_postfix(stage=message)
    else:
        tqdm.write(message)


def _preprocess_embedding(
    embedding: np.ndarray,
    *,
    scale: bool = True,
    pca_components: int | None = 50,
    use_gpu: bool = False,
    pbar: tqdm | None = None,
) -> np.ndarray:
    """Optionally apply standard scaling and PCA to an embedding.

    Parameters
    ----------
    embedding : np.ndarray, shape ``(n_samples, n_features)``
    scale : bool
        Apply standard scaling.
    pca_components : int or None
        Reduce to this many PCA components. ``None`` skips PCA.
    use_gpu : bool
        Use cuML implementations (must be installed).
    pbar : tqdm or None
    """
    if use_gpu:
        from cuml.decomposition import PCA as _PCA
        from cuml.preprocessing import StandardScaler as _StandardScaler
    else:
        from sklearn.decomposition import PCA as _PCA
        from sklearn.preprocessing import StandardScaler as _StandardScaler

    if scale:
        _stage("scaling embeddings", pbar)
        embedding = _StandardScaler().fit_transform(embedding)

    if pca_components is not None:
        if pca_components >= embedding.shape[1]:
            _stage(
                f"skipping PCA: pca_components ({pca_components}) "
                f">= n_features ({embedding.shape[1]})",
                pbar,
            )
        else:
            _stage(
                f"applying PCA: {embedding.shape[1]} -> {pca_components} dims",
                pbar,
            )
            embedding = _PCA(n_components=pca_components).fit_transform(embedding)

    return embedding


# ---------------------------------------------------------------------------
# Public reduction entry points
# ---------------------------------------------------------------------------


def get_tsne_embedding(
    embedding: np.ndarray,
    *,
    device: str = "cuda",
    scale: bool = True,
    pca_components: int | None = 50,
    perplexity: int = 30,
    metric: str = "cosine",
    method: str = "fft",
    verbose: bool = False,
    pbar: tqdm | None = None,
    **kwargs,
) -> np.ndarray:
    """Run t-SNE on a preloaded embedding, with optional GPU acceleration.

    Parameters
    ----------
    embedding : np.ndarray, shape ``(n_samples, n_features)``
    device : str
        ``"cuda"`` falls back to CPU if no GPU is available.
    scale : bool
    pca_components : int or None
    perplexity : int
    metric : str
    method : str
        Approximation method (``"fft"`` or ``"exact"``). Only used by
        cuML; openTSNE picks its own method.
    verbose : bool
    pbar : tqdm or None
    **kwargs
        Forwarded to ``cuml.manifold.TSNE`` or ``openTSNE.TSNE``.

    Returns
    -------
    np.ndarray, shape (n_samples, 2)

    Notes
    -----
    cuML requires ``n_neighbors >= 3 * perplexity``; we silently raise
    ``n_neighbors`` if necessary.
    """
    use_gpu = _resolve_gpu(device, verbose=verbose)

    _stage("preprocessing", pbar)
    embedding = _preprocess_embedding(
        embedding,
        scale=scale,
        pca_components=pca_components,
        use_gpu=use_gpu,
        pbar=pbar,
    )

    if use_gpu:
        from cuml.manifold import TSNE as cumlTSNE

        n_neighbors = max(kwargs.pop("n_neighbors", 0), 3 * perplexity)
        _stage("fitting t-SNE (GPU)", pbar)
        result = cumlTSNE(
            perplexity=perplexity,
            metric=metric,
            method=method,
            n_neighbors=n_neighbors,
            verbose=verbose,
            **kwargs,
        ).fit_transform(embedding)
    else:
        from openTSNE import TSNE as openTSNE

        _stage("fitting t-SNE (CPU)", pbar)
        result = openTSNE(
            perplexity=perplexity,
            metric=metric,
            verbose=verbose,
            **kwargs,
        ).fit(embedding)

    _stage("done", pbar)
    return np.asarray(result)


def get_umap_embedding(
    embedding: np.ndarray,
    *,
    device: str = "cuda",
    scale: bool = True,
    pca_components: int | None = 50,
    n_neighbors: int = 15,
    metric: str = "cosine",
    min_dist: float = 0.1,
    verbose: bool = False,
    pbar: tqdm | None = None,
    **kwargs,
) -> np.ndarray:
    """Run UMAP on a preloaded embedding, with optional GPU acceleration.

    Parameters
    ----------
    embedding : np.ndarray, shape ``(n_samples, n_features)``
    device : str
    scale : bool
    pca_components : int or None
    n_neighbors : int
    metric : str
    min_dist : float
    verbose : bool
    pbar : tqdm or None
    **kwargs
        Forwarded to ``cuml.manifold.UMAP`` or ``umap.UMAP``.

    Returns
    -------
    np.ndarray, shape (n_samples, 2)
    """
    use_gpu = _resolve_gpu(device, verbose=verbose)

    _stage("preprocessing", pbar)
    embedding = _preprocess_embedding(
        embedding,
        scale=scale,
        pca_components=pca_components,
        use_gpu=use_gpu,
        pbar=pbar,
    )

    if use_gpu:
        from cuml.manifold import UMAP as cumlUMAP

        _stage("fitting UMAP (GPU)", pbar)
        result = cumlUMAP(
            n_neighbors=n_neighbors,
            metric=metric,
            min_dist=min_dist,
            verbose=verbose,
            **kwargs,
        ).fit_transform(embedding)
    else:
        from umap import UMAP

        _stage("fitting UMAP (CPU)", pbar)
        result = UMAP(
            n_neighbors=n_neighbors,
            metric=metric,
            min_dist=min_dist,
            verbose=verbose,
            **kwargs,
        ).fit_transform(embedding)

    _stage("done", pbar)
    return np.asarray(result)
