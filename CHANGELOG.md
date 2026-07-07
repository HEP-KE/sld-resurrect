# Changelog

Notable changes to `sld_resurrect`. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions are
git tags read by `setuptools_scm`.

## [0.2.0] - 2026-07-07

The post-publication hardening release, accompanying the arXiv paper
([2606.00224](https://arxiv.org/abs/2606.00224)): the analysis notebook is
split into a staged, executed pipeline, the package is restructured around
single authoritative homes for every analysis fact, and the codebase gains
CI, packaging, and a full reviewed cleanup. All measured results are
unchanged — every restructuring step was verified against the previous
outputs (selection presets byte-identical throughout; the full measurement
chain reproduced on re-execution).

### Added

- `sld-resurrect download-dataset`: fetches and unpacks the Zenodo parquet
  release into the tree the notebooks read, with resume, an existing-output
  skip, a disk-space check, and safe zip handling (#9).
- Continuous integration: ruff format/lint and mypy gates plus an
  install-and-import smoke matrix on Python 3.10–3.14 (#10).
- Executed demo notebooks: the staged pipeline ships with its cutflows,
  validation plots, and measurement results embedded, with portable
  (`./sld`) paths and sequential execution counts (#6, refreshed in #13).
- Selection-layer API: `selector_presets.get_preset()`,
  `EventSelector.from_preset` / observables-only `EventView.from_preset`,
  the `cuts` module (typed, keyword-only `CutSpec`/`CutGroup`), and the
  `jets.cluster_two_jets` Durham helper (#3, #13, #15).
- Event-metadata API: `event_view.event_year`,
  `event_view.beam_polarisation`, and `POLARISATION_VALIDITY_THRESHOLD`,
  shared by the package and the notebooks (#13).
- Shared conventions as constants: `SLD_REQUIRED_BANKS`,
  `DEFAULT_SHARD_PATTERN`, `MINIDST_PARQUET_DIR`,
  `LEPTONIC_FIDUCIAL_BY_YEAR`, `EMBEDDING_FILENAME_FORMAT` (+
  `parse_embedding_filename`), and `MODEL_SIZES`/`MODEL_SIZE_ALIASES`
  (#9, #11, #13).
- Bank-decoding utilities for dataset consumers: `kinematics.build_tracks`
  (PHCHRG helix rebuild) and `kinematics.build_clusters` (PHKLUS
  4-vectors) (#3).
- README: event-selection API section, dataset provenance (Zenodo +
  `jazelle` reader), and a citation section with the paper BibTeX (#7).

### Changed

- The monolithic `SLD_DataAnalysis` notebook is split into three staged
  notebooks — data preparation, validation plots, measurements — and
  retired; the embeddings notebook joins the series as
  `SLD_04_OmniLearnedEmbeddings` (#3, #5, #13).
- Selection layer restructured: `selector.py` now holds only the cut
  engine, with per-track quality in `track_quality.py`, observables in
  `event_view.py`, and the cut vocabulary in `cuts.py`;
  `make_selector` is replaced by `EventSelector.from_preset` (#3, #13).
- `SLD_BASE` defaults to `./sld` everywhere, so the CLI and notebooks
  operate on one tree out of the box (#9).
- Versioning is SCM-driven (`setuptools_scm`; tag `vX.Y.Z` to release)
  and Python 3.10–3.14 are declared and exercised in CI (#10).
- The foundation model is named OmniLearned throughout, matching the
  paper and the `omnilearned` package (#12, #14).
- Jet clustering goes through fastjet's public `ClusterSequence`; the
  swig-level `JetDefinition` remains as a documented necessity for the
  R-less ee_kt constructor (#12).
- CLI conventions unified: existing outputs are skipped unless
  `--overwrite`, expected user errors exit with clean messages, and
  `reduce-embeddings --list` no longer requires `--method` (#15).

### Fixed

- Tau-channel bias correction uses the Table II year mapping of
  hep-ex/0010015 (1996: −0.0182, 1997–98: −0.0183) (#3).
- The ALEPH reader's branch filter now applies — `uproot.iterate` takes
  `expressions`, and the previous `columns` kwarg was silently ignored
  (#11).
- Checkpoint loading fails loudly on foreign files and missing model
  parameters instead of silently yielding randomly-initialised layers;
  training-time extras in the published checkpoints remain tolerated
  (#11).
- The LAC cluster-id lookup survives row-sliced event records
  (layout-agnostic `ak.num` instead of `.layout.offsets`) (#11).
- `max_events` is validated (positive or `None` at the API, `-1` at the
  CLI for all events); the previous truthiness gate silently dropped the
  last event for `-1` and loaded everything for `0` (#11).
- `process-dataset sld --batch-size` is threaded through to the
  boosted-frame strategy instead of being ignored; the unused `PHCHRG`
  bank is no longer read (#11).
- Streamed HTTP downloads close their responses deterministically (#11).
- The charge-oriented thrust-axis convention is documented as
  implemented (axis along the positive-net-charge hemisphere) and the
  confusion-flagged double sign flip is collapsed; thrust docstrings no
  longer overclaim exactness beyond two-jet topologies (#12).
- The thrust-major kernel no longer recomputes axis-invariant
  projections inside its candidate scan (identical results, a third of
  the inner-loop arithmetic) (#15).

### Removed

- The monolithic reference notebook (preserved in history); local
  working notebooks live untracked under `demos/private/` (#4, #5).
- Dead code: the unread `_GLOBAL_QUANTITIES` table and
  `release_memory`'s no-op reference-dropping loop (#12).

## [0.1.0] - 2026-04-30

Initial codebase: SLD parquet reading via `jazelle`, published selection
presets, kinematic observables, OmniLearned point-cloud preparation,
inference, and embedding-reduction CLI, and the original
`SLD_DataAnalysis` notebook.
