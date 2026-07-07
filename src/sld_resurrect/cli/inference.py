"""Run an OmniLearned checkpoint on a point cloud.

Use ``--task embed`` to extract per-token body embeddings (the default,
typical first step for visualisation). Use ``--task classify`` to run
the classifier head on an existing embedding and produce softmax
probabilities over the 210 pre-training classes.

For multi-GPU runs, launch under ``torchrun`` and pass ``--distributed``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from sld_resurrect.paths import OMNILEARN_CHECKPOINT_DIR

__all__ = ["add_parser", "run"]


from sld_resurrect.models.checkpoints import MODEL_SIZES as _SIZE_CHOICES


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "inference",
        help="Run an OmniLearned checkpoint on a point cloud (embed or classify).",
        description=(
            "Loads an OmniLearned checkpoint and runs inference on a point-cloud "
            "HDF5 file. Output is written as HDF5 with a single 'data' dataset. "
            "Use --task embed to extract body embeddings (typical first step), "
            "then re-run with --task classify on the embedding file to get the "
            "210-class softmax."
        ),
    )
    parser.add_argument("input", type=Path, help="Input HDF5 file (point cloud or embedding).")
    parser.add_argument("output", type=Path, help="Output HDF5 file.")
    parser.add_argument(
        "--size",
        "-s",
        choices=_SIZE_CHOICES,
        default="s",
        help="Model size: 's' (small), 'm' (medium), 'l' (large). Default: 's'.",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=128,
        help="Batch size per GPU (default: 128).",
    )
    parser.add_argument(
        "--max-events",
        "-n",
        type=int,
        default=10_000,
        help=(
            "Maximum number of events to process. Default 10000 to keep "
            "single-shot inference cheap; pass -1 to process all events. "
            "Note that this default applies even in distributed mode."
        ),
    )
    parser.add_argument(
        "--task",
        "-t",
        choices=("embed", "classify"),
        default="embed",
        help=(
            "'embed' runs model.body (default; produces token-level "
            "embeddings). 'classify' runs model.classifier on the embedding "
            "and applies softmax."
        ),
    )
    parser.add_argument(
        "--distributed",
        "-d",
        action="store_true",
        help=(
            "Run inference across multiple GPUs. Requires invocation under "
            "torchrun, e.g. 'torchrun --nproc_per_node=4 -m "
            "sld_resurrect.cli._main inference ... -d'."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader worker count for distributed mode (default: 2).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=OMNILEARN_CHECKPOINT_DIR,
        help=(
            "Directory containing the .pt checkpoint files. Defaults to "
            "$OMNILEARN_CHECKPOINT_DIR if set, else "
            "./checkpoints/omnilearned/."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Re-run and overwrite the output even if it already exists. "
            "By default, an existing output file is left untouched."
        ),
    )
    parser.set_defaults(run=run)
    return parser


def run(args: argparse.Namespace) -> int:
    from sld_resurrect.models.inference import (
        cleanup_distributed,
        setup_distributed,
    )

    if args.distributed:
        setup_distributed()

    try:
        return _run_inference(args)
    finally:
        if args.distributed:
            cleanup_distributed()


def _log(message: str) -> None:
    """Print only on the main (rank-0) process."""
    from sld_resurrect.models.inference import is_main_process

    if is_main_process():
        print(message)


def _run_inference(args: argparse.Namespace) -> int:
    import h5py
    import numpy as np
    import torch

    from sld_resurrect.models.inference import (
        batched_inference,
        batched_inference_distributed,
        release_memory,
    )
    from sld_resurrect.models.loader import (
        checkpoint_path_for,
        load_omnilearned_model,
    )

    checkpoint_path = checkpoint_path_for(args.size, args.checkpoint_dir)

    _log(f"Task:           {args.task}")
    _log(f"Input:          {args.input}")
    _log(f"Output:         {args.output}")
    _log(f"Model size:     {args.size}")
    _log(f"Batch size:     {args.batch_size}")
    _log(f"Distributed:    {args.distributed}")
    _log(f"Checkpoint:     {checkpoint_path}")

    if args.output.exists() and not args.overwrite:
        _log("Output already exists -- skipping (pass --overwrite to re-run).")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # ---- Load input ----
    with h5py.File(args.input, "r") as f:
        array = f["data"][: args.max_events] if args.max_events > 0 else f["data"][:]
    data = torch.from_numpy(np.asarray(array)).float()
    _log(f"Loaded input:   shape={tuple(data.shape)}")

    # ---- Load model ----
    model = load_omnilearned_model(args.size, checkpoint_path)
    # nn.Module attribute access types as Tensor | Module; both heads are modules.
    submodel = cast(
        "torch.nn.Module",
        model.body if args.task == "embed" else model.classifier,
    )

    # ---- Run inference ----
    if args.distributed:
        results = batched_inference_distributed(
            submodel,
            data,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    else:
        results = batched_inference(submodel, data, batch_size=args.batch_size)

    # Apply softmax for classify task. In distributed mode, only rank 0
    # holds the gathered tensor.
    if args.task == "classify" and results is not None:
        results = results.softmax(dim=-1)

    # ---- Save output (rank-0 only in distributed mode) ----
    if results is not None:
        with h5py.File(args.output, "w") as f:
            f.create_dataset("data", data=results.numpy())
        _log(f"Saved {args.output} | shape={tuple(results.shape)}")

    del model, results
    release_memory()
    return 0
