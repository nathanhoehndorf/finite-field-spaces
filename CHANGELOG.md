# Changelog

## [0.3.0] - 2026-07-09

### Added
- Added a dedicated low-memory module at `src/ffspaces/lowmem.py` for large-
  dimension experiments over finite fields, including:
  `generate_space_chunked`, `_fwht_inplace`, `compute_sumset_lowmem`,
  `compute_sumset_lowmem_from_ints`, `generate_ball_ints_lowmem`,
  `generate_covering_ints_lowmem`, `complement_ints_lowmem`,
  `find_maximum_subspace_dimension_lowmem`, and `estimate_memory_gb`.
- Added a command-line interface at `src/ffspaces/cli.py` with two subcommands:
  `ffspaces run` for running the covering experiment and
  `ffspaces memory-check` for printing estimated peak memory requirements.
- Added a `--low-memory` execution mode to the CLI for binary experiments,
  routing trial execution through the integer-based low-memory implementations.
- Added a console script entry point in `pyproject.toml`:
  `ffspaces = "ffspaces.cli:main"`.
- Added pytest coverage for the new low-memory and CLI surfaces in
  `tests/test_lowmem_and_cli.py`, including regression coverage for:
  chunked space generation, in-place FWHT equivalence, low-memory sumset
  computation, integer-based covering/complement helpers, low-memory subspace
  detection, CLI output, CLI warnings, and output-file creation.

### Changed
- Bumped the project version from `0.2.2` to `0.3.0` in `pyproject.toml`.
- Expanded the public API exports in `src/ffspaces/__init__.py` to include the
  low-memory helpers so they can be imported directly from `ffspaces`.
- Updated `README.md` with:
  low-memory API documentation, end-to-end large-`n` usage examples,
  CLI usage for `ffspaces run` and `ffspaces memory-check`,
  and revised performance guidance comparing standard and low-memory modes.
- Extended the public API regression test in `tests/test_public_api.py` to
  assert that the new low-memory helpers are exported.

### Fixed
- Added a low-memory execution path that avoids materialising the full
  `p^n × n` universe array for binary covering experiments, which previously
  made large-`n` runs prone to out-of-memory crashes.
- Reduced peak memory usage for binary sumset computation by replacing the
  dense complex128 FFT path with a real float64 in-place Walsh-Hadamard
  transform in the low-memory implementation.
- Reduced complement-construction overhead in the large-`n` path by using a
  boolean mask rather than dense integer arrays, and reduced low-memory
  subspace-membership overhead by using sorted integer encodings instead of
  Python tuple sets.

## [0.2.2] - 2026-07-03

### Added
- Added `setup.py` to enable editable installs via `pip install -e .[test]`.
- Added optional `test` extras in `pyproject.toml` for `pytest` installation.

### Fixed
- `find_maximum_subspace_dimension` (both the default greedy path and the
  `exhaustive=True` path) previously checked linear independence using
  `np.linalg.matrix_rank` on floating-point data, which does not agree with
  rank over F_p in general. Replaced with an exact `rank_mod_p` routine using
  modular Gaussian elimination. This could previously overstate the maximum
  embedded subspace dimension for some inputs; see `rank_mod_p` in `core.py`
  and its regression test for a concrete example.
- Consolidated duplicate module copies under `src/` into the single
  `src/ffspaces` package to remove an import-shadowing hazard.

### Changed
- Bumped project version from `0.1.0` to `0.2.2` in `pyproject.toml` and `setup.py`.
- `is_invertible` now delegates to the shared `rank_mod_p` for consistency
  with the subspace-dimension code.
- README updated with the underlying math (Hamming balls, sumsets, subspace
  dimension) for the current API.
