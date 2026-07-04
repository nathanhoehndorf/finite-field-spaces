# Changelog

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
