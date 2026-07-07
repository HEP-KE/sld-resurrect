"""Download pretrained OmniLearned checkpoint files."""

from __future__ import annotations

import argparse
from pathlib import Path

from sld_resurrect.models.checkpoints import (
    CHECKPOINT_FILES,
    MODEL_SIZE_ALIASES,
    MODEL_SIZES,
    checkpoint_url,
    fetch_checkpoints,
)
from sld_resurrect.paths import OMNILEARN_CHECKPOINT_DIR

__all__ = ["add_parser", "run"]


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "download-checkpoints",
        help="Download pretrained OmniLearned checkpoints.",
        description=(
            "Download one or more pretrained OmniLearned .pt files from the "
            "public NERSC endpoint, with size-based caching (already-present "
            "files matching the remote size are skipped)."
        ),
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        choices=MODEL_SIZES,
        default=list(MODEL_SIZES),
        help=(
            "Model sizes to download. Default: all three (s, m, l). "
            "Each size is roughly 100-300 MB."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=OMNILEARN_CHECKPOINT_DIR,
        help=(
            "Output directory for the .pt files. Defaults to "
            "$OMNILEARN_CHECKPOINT_DIR if set, else "
            "./checkpoints/omnilearned/."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Maximum number of concurrent downloads (default: 5).",
    )
    parser.set_defaults(run=run)
    return parser


def run(args: argparse.Namespace) -> int:
    urls = [checkpoint_url(CHECKPOINT_FILES[MODEL_SIZE_ALIASES[s]]) for s in args.sizes]

    print(f"Downloading {len(urls)} checkpoint(s) to {args.checkpoint_dir}:")
    for url in urls:
        print(f"  {url}")

    paths = fetch_checkpoints(
        urls,
        target_dir=args.checkpoint_dir,
        max_workers=args.max_workers,
    )
    print("\nFinished. Files at:")
    for path in paths:
        print(f"  {path}")
    return 0
