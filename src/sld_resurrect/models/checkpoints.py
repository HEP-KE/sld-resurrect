"""Download and cache OmniLearned checkpoint files.

The OmniLearned pretrained weights are published as plain ``.pt`` files
on a NERSC HTTP endpoint. :func:`fetch_checkpoints` downloads one or
more of them in parallel, writing to a target directory and skipping
files that already exist with the expected size.

The HTTP endpoint and filenames are baked in below. Edit
:data:`CHECKPOINT_URL_BASE` and :data:`CHECKPOINT_FILES` if the
OmniLearned project moves them.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from tqdm.auto import tqdm

from sld_resurrect.paths import OMNILEARN_CHECKPOINT_DIR

__all__ = [
    "CHECKPOINT_FILES",
    "CHECKPOINT_URL_BASE",
    "MODEL_SIZES",
    "MODEL_SIZE_ALIASES",
    "checkpoint_url",
    "fetch_checkpoints",
]


MODEL_SIZE_ALIASES: dict[str, str] = {"s": "small", "m": "medium", "l": "large"}
"""One-letter to full-name mapping for the published model sizes."""

MODEL_SIZES: tuple[str, ...] = tuple(MODEL_SIZE_ALIASES)
"""The one-letter model-size vocabulary used across the CLI."""

CHECKPOINT_URL_BASE: str = "https://portal.nersc.gov/cfs/dasrepo/omnilearned/checkpoints/"
"""Public NERSC URL prefix where the OmniLearned ``.pt`` files are hosted."""

CHECKPOINT_FILES: dict[str, str] = {
    "small": "best_model_pretrain_s.pt",
    "medium": "best_model_pretrain_m.pt",
    "large": "best_model_pretrain_l.pt",
}
"""Mapping from human-readable size to checkpoint filename."""


def checkpoint_url(filename: str) -> str:
    """Return the full URL for a checkpoint filename."""
    return urljoin(CHECKPOINT_URL_BASE, filename)


def _download_one(
    url: str,
    save_path: Path,
    chunk_size: int,
    position: int,
) -> bool:
    """Download a single URL to ``save_path`` with caching by size match.

    Returns
    -------
    bool
        True if a download happened, False if the cached file was reused.
    """
    # Size-only cache check via HEAD, so a cached file never opens a
    # streamed GET whose body would go unconsumed.
    head = requests.head(url, allow_redirects=True, timeout=15)
    head.raise_for_status()
    total_size = int(head.headers.get("content-length", 0))

    if save_path.exists():
        local_size = save_path.stat().st_size
        if total_size > 0 and local_size == total_size:
            tqdm.write(f"Cached: {save_path.name}")
            return False
        if total_size == 0 and local_size > 0:
            tqdm.write(f"Cached (size unknown): {save_path.name}")
            return False

    with (
        requests.get(url, stream=True, timeout=15) as response,
        tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=save_path.name,
            position=position,
            leave=False,
        ) as pbar,
        open(save_path, "wb") as fh,
    ):
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                fh.write(chunk)
                pbar.update(len(chunk))
    return True


def fetch_checkpoints(
    urls: str | Iterable[str],
    target_dir: str | Path = OMNILEARN_CHECKPOINT_DIR,
    chunk_size: int = 8192,
    max_workers: int = 5,
) -> list[Path]:
    """Download files from one or more URLs in parallel, with caching.

    Parameters
    ----------
    urls : str or iterable of str
        URL(s) to download.
    target_dir : path-like
        Local directory where files are saved (created if missing).
    chunk_size : int
        Streaming chunk size in bytes.
    max_workers : int
        Maximum number of concurrent downloads.

    Returns
    -------
    list[Path]
        Local paths of all successfully-downloaded (or cached) files.
    """
    if isinstance(urls, str):
        urls = [urls]
    urls = list(urls)

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []

    def _task(url: str, position: int) -> Path:
        filename = os.path.basename(urlparse(url).path)
        if not filename:
            raise ValueError(f"Cannot derive filename from URL: {url}")
        save_path = target_dir / filename
        try:
            _download_one(url, save_path, chunk_size, position)
        except Exception as exc:
            tqdm.write(f"Error downloading {url}: {exc}")
            raise
        return save_path

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_task, url, i): url for i, url in enumerate(urls)}
        for future in as_completed(futures):
            paths.append(future.result())

    return paths


def fetch_all_pretrained(
    target_dir: str | Path = OMNILEARN_CHECKPOINT_DIR,
) -> list[Path]:
    """Download all three (small / medium / large) pretrained checkpoints."""
    urls = [checkpoint_url(name) for name in CHECKPOINT_FILES.values()]
    return fetch_checkpoints(urls, target_dir=target_dir)
