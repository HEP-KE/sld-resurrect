"""Run t-SNE or UMAP on saved OmniLearn embeddings.

Loads embedding files for one or more datasets, applies mean pooling
over the token axis, then runs the chosen non-linear reduction down to
2D. The 2D coordinates for each input dataset are written under that
dataset's own key in a single combined output HDF5 file.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from sld_resurrect.paths import OMNILEARN_EMBEDDING_DIR, OMNILEARN_REDUCED_DIR

__all__ = ["add_parser", "run"]


_FILENAME_RE = re.compile(r"^omnilearned_embedding_(?P<size>[sml])_(?P<dataset>.+)\.h5$")


def _max_events(value: str) -> int | None:
    """Argparse type for --max-events: a positive integer, or -1 for all events."""
    try:
        count = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from None
    if count == -1:
        return None
    if count <= 0:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer or -1 (all events), got {count}"
        )
    return count


def _discover_datasets(embedding_dir: Path, size: str) -> list[str]:
    """Return the sorted list of dataset names available for ``size``.

    A "dataset name" is the trailing portion of the filename
    ``omnilearned_embedding_{size}_{name}.h5``.
    """
    names: list[str] = []
    for path in sorted(embedding_dir.glob(f"omnilearned_embedding_{size}_*.h5")):
        match = _FILENAME_RE.match(path.name)
        if match and match["size"] == size:
            names.append(match["dataset"])
    return names


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "reduce-embeddings",
        help="Run t-SNE or UMAP on saved OmniLearn embeddings.",
        description=(
            "Loads embedding files for one or more datasets, mean-pools "
            "over the token axis, runs t-SNE or UMAP, and writes the 2D "
            "coordinates to a single output HDF5 with one dataset key "
            "per input dataset."
        ),
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help=(
            "Dataset names to process (e.g. 'sld_superjet', "
            "'jetclass1_ZToQQ'). Use --list to see what's available. "
            "If empty, all discovered datasets are processed."
        ),
    )
    parser.add_argument(
        "--method",
        choices=("tsne", "umap"),
        required=True,
        help="Dimensionality reduction method.",
    )
    parser.add_argument(
        "--size",
        "-s",
        choices=("s", "m", "l"),
        default="m",
        help="OmniLearn model size whose embeddings to load (default: m).",
    )
    parser.add_argument(
        "--max-events",
        type=_max_events,
        default=3000,
        help=(
            "Per-dataset event cap (default: 3000). Use -1 for all "
            "events. t-SNE/UMAP runtime scales steeply with sample "
            "count; 3000 is a reasonable default for visualisation on "
            "a single GPU."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help=(
            "Compute device for the reduction (default: 'cuda'). 'cuda' "
            "falls back to CPU if no GPU is available."
        ),
    )
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=OMNILEARN_EMBEDDING_DIR,
        help=(
            "Directory containing the embedding files. Defaults to "
            "$SLD_BASE/omnilearned/embeddings."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OMNILEARN_REDUCED_DIR,
        help=(
            "Directory to write the reduced output into. Defaults to "
            "$SLD_BASE/omnilearned/reduced. The output filename is "
            "'reduced_{method}_{size}.h5'."
        ),
    )
    parser.add_argument(
        "--scale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply standard scaling before reduction (default: yes).",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=50,
        help=(
            "Reduce embeddings to this many PCA components before "
            "running t-SNE/UMAP (default: 50). Pass 0 to skip PCA."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discovered datasets for the chosen size and exit.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose progress output from the underlying t-SNE/UMAP fit.",
    )
    parser.set_defaults(run=run)
    return parser


def _resolve_datasets(args: argparse.Namespace) -> list[str]:
    """Resolve the list of datasets to process, validating user input."""
    available = _discover_datasets(args.embedding_dir, args.size)
    if not available:
        raise SystemExit(
            f"No embedding files matching "
            f"'omnilearned_embedding_{args.size}_*.h5' found under "
            f"{args.embedding_dir!r}.\n"
            f"Did you run `sld-resurrect inference --task embed -s {args.size}` first?"
        )

    if not args.datasets:
        return available

    unknown = [d for d in args.datasets if d not in available]
    if unknown:
        raise SystemExit(
            f"Unknown dataset(s): {unknown!r}\nAvailable for size {args.size!r}: {available!r}"
        )
    return list(args.datasets)


def run(args: argparse.Namespace) -> int:
    import h5py
    import numpy as np
    from tqdm.auto import tqdm

    from sld_resurrect.reduction import (
        embedding_path_for,
        get_tsne_embedding,
        get_umap_embedding,
        load_pooled_embedding,
    )

    if args.list:
        available = _discover_datasets(args.embedding_dir, args.size)
        print(f"Datasets available for size {args.size!r} under {args.embedding_dir}:")
        for name in available:
            print(f"  {name}")
        if not available:
            print("  (none)")
        return 0

    datasets = _resolve_datasets(args)
    pca_components: int | None = args.pca_components if args.pca_components > 0 else None
    output_path = args.output_dir / f"reduced_{args.method}_{args.size}.h5"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Method:        {args.method}")
    print(f"Model size:    {args.size}")
    print(f"Datasets:      {len(datasets)}")
    print(f"Max events:    {args.max_events}")
    print(f"Device:        {args.device}")
    print(f"PCA components: {pca_components}")
    print(f"Output:        {output_path}")

    fit_fn = get_tsne_embedding if args.method == "tsne" else get_umap_embedding

    results: dict[str, np.ndarray] = {}
    with tqdm(total=len(datasets), desc=f"Reducing ({args.method})") as pbar:
        for name in datasets:
            pbar.set_description(f"Reducing {name}")
            path = embedding_path_for(name, args.size, args.embedding_dir)
            embedding = load_pooled_embedding(path, max_events=args.max_events)
            reduced = fit_fn(
                embedding,
                device=args.device,
                scale=args.scale,
                pca_components=pca_components,
                verbose=args.verbose,
                pbar=pbar,
            )
            results[name] = reduced
            pbar.update(1)

    with h5py.File(output_path, "w") as f:
        for name, coords in results.items():
            f.create_dataset(name, data=coords)
    print(f"\nSaved {output_path} (keys: {list(results.keys())})")
    return 0
