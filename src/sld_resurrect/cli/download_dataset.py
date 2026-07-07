"""Download and unpack the released SLD parquet dataset from Zenodo.

Fetches the zipped 1996-1998 mini-DST parquet release, extracts the
shards into the directory the analysis notebooks read from
(``$SLD_BASE/datasets/minidst_translated/parquet``), and removes the
zip once every member has been extracted successfully.
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import requests
from tqdm.auto import tqdm

from sld_resurrect.paths import MINIDST_PARQUET_DIR

__all__ = ["add_parser", "run"]


ZENODO_DATASET_URL = "https://zenodo.org/records/21199778/files/sld_minidsts_parquet_1996_1998.zip"

_CHUNK_BYTES = 1 << 20  # 1 MiB
_TIMEOUT = 60  # seconds, per read


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "download-dataset",
        help="Download the released SLD parquet dataset from Zenodo.",
        description=(
            "Download the zipped 1996-1998 SLD mini-DST parquet release "
            "(~4.9 GB), unpack the shards into the directory the analysis "
            "notebooks read from, and remove the zip after successful "
            "extraction. An interrupted download resumes where it left off."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MINIDST_PARQUET_DIR,
        help=(
            "Directory to place the parquet shards in. Defaults to "
            "$SLD_BASE/datasets/minidst_translated/parquet (the location "
            "the analysis notebooks read), with $SLD_BASE falling back "
            "to ./sld."
        ),
    )
    parser.add_argument(
        "--url",
        default=ZENODO_DATASET_URL,
        help="Zenodo file URL of the dataset zip (default: %(default)s).",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the downloaded zip instead of deleting it after extraction.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=("Download and extract even if the output directory already contains parquet shards."),
    )
    parser.set_defaults(run=run)
    return parser


def _remote_size(url: str) -> int | None:
    """Content length of ``url`` in bytes, or ``None`` if unavailable."""
    try:
        response = requests.head(url, allow_redirects=True, timeout=_TIMEOUT)
        response.raise_for_status()
        length = response.headers.get("Content-Length")
        return int(length) if length else None
    except (requests.RequestException, ValueError):
        return None


def _check_disk_space(output_dir: Path, zip_bytes: int) -> None:
    """Raise if the filesystem cannot hold the zip plus its contents."""
    # During extraction the zip and the (similarly sized) shards coexist,
    # so require roughly twice the archive size plus slack.
    required = int(zip_bytes * 2.1)
    free = shutil.disk_usage(output_dir).free
    if free < required:
        raise OSError(
            f"Not enough free space in {output_dir}: need about "
            f"{required / 1e9:.1f} GB (zip + extracted shards), have "
            f"{free / 1e9:.1f} GB. Free up space or pass --output-dir "
            f"pointing at a larger filesystem."
        )


def _download(url: str, zip_path: Path, total: int | None) -> None:
    """Stream ``url`` to ``zip_path``, resuming a partial download if possible."""
    part_path = zip_path.with_suffix(zip_path.suffix + ".part")
    offset = part_path.stat().st_size if part_path.exists() else 0

    headers = {}
    if offset:
        headers["Range"] = f"bytes={offset}-"
        print(f"Resuming download at {offset / 1e9:.2f} GB")

    response = requests.get(url, headers=headers, stream=True, timeout=_TIMEOUT)
    response.raise_for_status()
    if offset and response.status_code != 206:
        # Server ignored the Range request; start over.
        print("Server does not support resume -- restarting download")
        offset = 0

    mode = "ab" if offset else "wb"
    progress = tqdm(
        total=total,
        initial=offset,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc="Downloading",
    )
    with open(part_path, mode) as fh, progress:
        for chunk in response.iter_content(chunk_size=_CHUNK_BYTES):
            fh.write(chunk)
            progress.update(len(chunk))

    written = part_path.stat().st_size
    if total is not None and written != total:
        raise OSError(
            f"Incomplete download: got {written:,} of {total:,} bytes. "
            f"Re-run to resume from where it stopped."
        )
    part_path.rename(zip_path)


def _extract(zip_path: Path, output_dir: Path) -> list[Path]:
    """Extract every member of ``zip_path`` into ``output_dir``.

    Members are extracted flat (no directory structure is expected in
    the release zip) and unsafe member names are rejected.
    """
    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path) as archive:
        members = [m for m in archive.infolist() if not m.is_dir()]
        for member in tqdm(members, desc="Extracting", unit="file"):
            name = Path(member.filename)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"Refusing unsafe zip member: {member.filename!r}")
            target = output_dir / name.name
            with archive.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, _CHUNK_BYTES)
            extracted.append(target)
    return extracted


def run(args: argparse.Namespace) -> int:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(output_dir.glob("*.parquet"))
    if existing and not args.overwrite:
        print(
            f"Output directory already holds {len(existing)} parquet "
            f"shard(s) -- skipping (pass --overwrite to re-download).\n"
            f"  {output_dir}"
        )
        return 0

    zip_path = output_dir / Path(args.url).name
    total = _remote_size(args.url)
    if total is not None and not zip_path.exists():
        _check_disk_space(output_dir, total)

    if zip_path.exists():
        print(f"Using previously downloaded archive: {zip_path}")
    else:
        print(f"Downloading {args.url}")
        print(f"  -> {zip_path}")
        _download(args.url, zip_path, total)

    extracted = _extract(zip_path, output_dir)
    print(f"Extracted {len(extracted)} file(s) to {output_dir}")

    if args.keep_zip:
        print(f"Keeping archive at {zip_path}")
    else:
        zip_path.unlink()
        print("Removed archive after successful extraction")

    print(
        "\nThe analysis notebooks read this location when SLD_BASE points "
        "at the same tree (see demos/SLD_01_DataPreparation.ipynb)."
    )
    return 0
