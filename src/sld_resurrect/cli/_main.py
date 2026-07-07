"""Top-level dispatcher for the ``sld-resurrect`` command.

Subcommands:

* ``download-dataset``     -- fetch and unpack the released SLD
  parquet dataset from Zenodo.
* ``download-checkpoints`` -- fetch pretrained OmniLearn ``.pt`` files.
* ``process-dataset``      -- convert raw experimental data to
  OmniLearn point clouds.
* ``inference``            -- run an OmniLearn checkpoint on a point
  cloud (extract embeddings or classifier scores).
* ``reduce-embeddings``    -- run t-SNE / UMAP on saved embeddings.

Run ``sld-resurrect --help`` for argument details on any subcommand.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from sld_resurrect import __version__
from . import (
    download_checkpoints,
    download_dataset,
    inference,
    process_dataset,
    reduce_embeddings,
)


__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sld-resurrect",
        description=(
            "End-to-end pipeline for the SLD reanalysis: dataset preparation, "
            "OmniLearn inference, and dimensionality reduction."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(
        title="subcommands",
        dest="command",
        metavar="<command>",
        required=True,
    )

    download_dataset.add_parser(subparsers)
    download_checkpoints.add_parser(subparsers)
    process_dataset.add_parser(subparsers)
    inference.add_parser(subparsers)
    reduce_embeddings.add_parser(subparsers)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    code = args.run(args)
    return int(code) if isinstance(code, int) else 0


if __name__ == "__main__":
    sys.exit(main())