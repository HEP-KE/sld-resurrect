"""Convert raw experimental data to OmniLearn-compatible point clouds.

One sub-subcommand per experiment (jetclass1, h1, aleph, sld). Outputs
are HDF5 files of shape ``(n_events, max_particles, 4)`` with a single
``data`` dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path


__all__ = ["add_parser"]


# ---------------------------------------------------------------------------
# Shared argument groups
# ---------------------------------------------------------------------------

def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add ``--max-events`` and ``--max-particles`` to a sub-subparser."""
    parser.add_argument(
        "--max-events",
        type=int,
        default=-1,
        help=(
            "Maximum number of events to process. Use -1 for all events "
            "(default: -1). Useful for development on a fast subset."
        ),
    )
    parser.add_argument(
        "--max-particles",
        type=int,
        default=128,
        help=(
            "Padding length per event (default: 128). OmniLearn was "
            "pre-trained at 128 -- changing this only makes sense if "
            "you are also fine-tuning the model."
        ),
    )


def _add_strategy_args(parser: argparse.ArgumentParser) -> None:
    """Add ``--strategy`` (required) and ``--batch-size`` (boosted-frame only)."""
    parser.add_argument(
        "--strategy",
        choices=("superjet", "hemisphere", "boosted_frame"),
        required=True,
        help=(
            "Coordinate-mapping strategy for OmniLearn input. "
            "'superjet': whole event on the thrust axis. "
            "'hemisphere': two Durham jets, each on its own axis. "
            "'boosted_frame': rotate the event so thrust lies along +z."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
        help=(
            "Events per rotation batch for 'boosted_frame' strategy "
            "(default: 10000). Ignored for the other two strategies."
        ),
    )


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def _write_h5(path: Path, array) -> None:
    """Write a single (n, p, f) array under the conventional ``data`` key."""
    import h5py  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=array)
    print(f"  wrote {path}  shape={array.shape}, dtype={array.dtype}")


# ---------------------------------------------------------------------------
# JetClass1
# ---------------------------------------------------------------------------

def _jetclass1_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        "jetclass1",
        help="Convert JetClass1 ROOT files to an OmniLearn point cloud.",
        description=(
            "JetClass1 stores per-jet ROOT files with already-computed "
            "(part_deta, part_dphi) relative to each jet axis, so no "
            "thrust calculation or jet clustering is needed."
        ),
    )
    p.add_argument("inputs", nargs="+", type=Path, help="Input ROOT file(s).")
    p.add_argument("output", type=Path, help="Output HDF5 file.")
    _add_common_args(p)
    p.set_defaults(run=_run_jetclass1)
    return p


def _run_jetclass1(args: argparse.Namespace) -> int:
    from sld_resurrect.datasets import parse_jetclass1

    cloud = parse_jetclass1(
        [str(p) for p in args.inputs],
        max_events=args.max_events,
        max_particles=args.max_particles,
    )
    _write_h5(args.output, cloud)
    return 0


# ---------------------------------------------------------------------------
# H1
# ---------------------------------------------------------------------------

def _h1_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        "h1",
        help="Convert H1 DIS HDF5 files to an OmniLearn point cloud.",
        description=(
            "H1 .h5 files are already in OmniLearn point-cloud format; "
            "this subcommand concatenates files, drops trailing feature "
            "columns, and pads."
        ),
    )
    p.add_argument("inputs", nargs="+", type=Path, help="Input HDF5 file(s).")
    p.add_argument("output", type=Path, help="Output HDF5 file.")
    _add_common_args(p)
    p.set_defaults(run=_run_h1)
    return p


def _run_h1(args: argparse.Namespace) -> int:
    from sld_resurrect.datasets import parse_h1

    cloud = parse_h1(
        [str(p) for p in args.inputs],
        max_events=args.max_events,
        max_particles=args.max_particles,
    )
    _write_h5(args.output, cloud)
    return 0


# ---------------------------------------------------------------------------
# ALEPH
# ---------------------------------------------------------------------------

def _aleph_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        "aleph",
        help="Convert ALEPH ROOT files to an OmniLearn point cloud.",
        description=(
            "ALEPH events are processed exactly like SLD: raw particles "
            "are loaded, then dispatched through one of the three "
            "coordinate-mapping strategies. The 'hemisphere' strategy "
            "writes two output files (leading + subleading), with "
            "'_leading'/'_subleading' suffixes appended to the output "
            "filename's stem."
        ),
    )
    p.add_argument("inputs", nargs="+", type=Path, help="Input ROOT file(s).")
    p.add_argument(
        "output",
        type=Path,
        help=(
            "Output HDF5 file (or stem for the two hemisphere outputs)."
        ),
    )
    _add_common_args(p)
    _add_strategy_args(p)
    p.set_defaults(run=_run_aleph)
    return p


def _run_aleph(args: argparse.Namespace) -> int:
    from sld_resurrect.datasets import parse_aleph

    result = parse_aleph(
        [str(p) for p in args.inputs],
        strategy=args.strategy,
        max_events=args.max_events,
        max_particles=args.max_particles,
        batch_size=args.batch_size,
    )

    if args.strategy == "hemisphere":
        cloud_leading, cloud_subleading = result
        out = args.output.with_suffix("")  # drop .h5 if present
        _write_h5(out.with_name(f"{out.name}_leading.h5"), cloud_leading)
        _write_h5(out.with_name(f"{out.name}_subleading.h5"), cloud_subleading)
    else:
        _write_h5(args.output, result)
    return 0


# ---------------------------------------------------------------------------
# SLD -- runs the full pipeline (load + select + cluster + all 3 strategies)
# ---------------------------------------------------------------------------

def _sld_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        "sld",
        help="Run the SLD hadronic pipeline and emit one or more strategies.",
        description=(
            "Reads SLD parquet shards via jazelle, applies the default "
            "hadronic event selection, runs Durham 2-jet clustering on "
            "the surviving events, and emits OmniLearn input files for "
            "the requested strategies. For custom hadronic selections, "
            "use the data-analysis notebook directly."
        ),
    )
    p.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing SLD parquet shards.",
    )
    p.add_argument(
        "output_dir",
        type=Path,
        help=(
            "Directory to write the output HDF5 files into. "
            "File names follow the convention "
            "omnilearned_input_sld_<strategy>.h5."
        ),
    )
    p.add_argument(
        "--strategies",
        nargs="+",
        choices=("superjet", "hemisphere", "boosted_frame"),
        default=("superjet", "hemisphere", "boosted_frame"),
        help=(
            "Which strategies to run (default: all three). "
            "The 'hemisphere' strategy emits two files "
            "(_leading and _subleading)."
        ),
    )
    p.add_argument(
        "--pattern",
        default="*.parquet",
        help="Pattern to match parquet shards in input_dir (default: '*.parquet').",
    )
    p.add_argument(
        "--max-events",
        type=int,
        default=-1,
        help=(
            "Maximum number of events to process. Use -1 for all events "
            "(default: -1). When set, parquet shards are read until the "
            "limit is reached, then the result is truncated."
        ),
    )
    p.add_argument(
        "--name-prefix",
        default="omnilearned_input_sld",
        help="Output filename prefix (default: 'omnilearned_input_sld').",
    )
    p.add_argument(
        "--max-particles",
        type=int,
        default=128,
        help="Padding length per event (default: 128).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
        help="Events per batch for the boosted-frame strategy (default: 10000).",
    )
    p.set_defaults(run=_run_sld)
    return p


def _run_sld(args: argparse.Namespace) -> int:
    import glob

    import awkward as ak
    import jazelle
    from tqdm.auto import tqdm

    import fastjet
    from fastjet._pyjet import AwkwardClusterSequence
    from fastjet._swig import JetDefinition

    from sld_resurrect.datasets import save_strategy_outputs
    from sld_resurrect.kinematics import build_particles
    from sld_resurrect.selector_presets import make_selector

    # ---- Load parquet shards (lazy: stop once max_events is reached) ----
    files = sorted(glob.glob(str(args.input_dir / args.pattern)))
    if not files:
        raise FileNotFoundError(
            f"No files matching {args.pattern!r} under {args.input_dir!r}"
        )

    requested_banks = [
        "IEVENTH", "PHBM", "PHPSUM", "PHCHRG",
        "PHKLUS", "PHPOINT", "PHWIC",
    ]
    arrays: list = []
    n_loaded = 0
    file_iter = tqdm(files, desc="Reading parquet")
    for path in file_iter:
        chunk = jazelle.from_parquet(path, columns=requested_banks)
        arrays.append(chunk)
        n_loaded += len(chunk)
        if args.max_events > 0 and n_loaded >= args.max_events:
            file_iter.close()
            break

    data = ak.concatenate(arrays)
    if args.max_events > 0:
        data = data[: args.max_events]
    print(f"Loaded {len(data):,} events from {len(arrays)} file(s).")

    # ---- Hadronic selection ----
    particles_all = build_particles(data)
    selector = make_selector("hadronic_default", data, particles_all)
    mask = selector.mask()
    n_sel, n_tot = int(mask.sum()), len(data)
    print(f"Hadronic selection: {n_sel:,} / {n_tot:,} events ({100 * n_sel / n_tot:.1f}%)")

    particles = build_particles(data[mask])

    # The hemisphere strategy needs Durham 2-jet clustering; the other
    # two only need the inclusive particle list. Skip the cluster step
    # if the user didn't ask for hemisphere.
    constituents = None
    if "hemisphere" in args.strategies:
        print("Clustering into 2 exclusive Durham jets...")
        jet_def = JetDefinition(fastjet.ee_kt_algorithm)
        cluster_seq = AwkwardClusterSequence(particles, jet_def)
        constituents = cluster_seq.exclusive_jets_constituents(2)
        jets = ak.sum(constituents, axis=2)
        pt_order = ak.argsort(jets.pt, axis=1, ascending=False)
        constituents = constituents[pt_order]

    # ---- Run + save the requested strategies ----
    written = save_strategy_outputs(
        constituents=constituents,
        particles=particles,
        output_dir=args.output_dir,
        strategies=tuple(args.strategies),
        max_particles=args.max_particles,
        name_prefix=args.name_prefix
    )
    print("\nFinished. Output files:")
    for label, path in written.items():
        print(f"  {label:25s} -> {path}")
    return 0


# ---------------------------------------------------------------------------
# Top-level wiring
# ---------------------------------------------------------------------------

def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "process-dataset",
        help="Convert raw experimental data to OmniLearn point clouds.",
        description=(
            "Converts raw experimental data files into OmniLearn-compatible "
            "point clouds (HDF5 with a single 'data' dataset of shape "
            "(n_events, max_particles, 4)). Use one of the four "
            "sub-subcommands to choose the experiment."
        ),
    )
    nested = parser.add_subparsers(
        title="experiment", dest="dataset", metavar="<dataset>", required=True
    )
    _jetclass1_parser(nested)
    _h1_parser(nested)
    _aleph_parser(nested)
    _sld_parser(nested)

    # No parent ``run`` is needed: the required sub-subparser always
    # populates ``args.run`` with one of the dataset-specific runners
    # (``_run_jetclass1`` etc.), and ``_main.py`` calls ``args.run(args)``.
    return parser