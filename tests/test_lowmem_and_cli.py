import json

import numpy as np

from ffspaces.cli import main as cli_main
from ffspaces.cli import _run_trial_lowmem, _run_trial_standard
from ffspaces.core import generate_random_basis, generate_space
from ffspaces.covers import complement, generate_covering
from ffspaces.fwht_operators import fwht, ints_to_vectors, vectors_to_ints
from ffspaces.lowmem import (
    _fwht_inplace,
    complement_ints_lowmem,
    compute_sumset_lowmem,
    compute_sumset_lowmem_from_ints,
    estimate_memory_gb,
    find_maximum_subspace_dimension_lowmem,
    generate_ball_ints_lowmem,
    generate_covering_ints_lowmem,
    generate_space_chunked,
)
from ffspaces.operators import compute_sumset, find_maximum_subspace_dimension


def test_generate_space_chunked_matches_generate_space_for_binary_and_nonbinary_cases():
    full_binary = generate_space(6, p=2)
    chunked_binary = np.vstack(list(generate_space_chunked(6, p=2, chunk_size=11)))
    assert np.array_equal(chunked_binary, full_binary)

    full_ternary = generate_space(3, p=3)
    chunked_ternary = np.vstack(list(generate_space_chunked(3, p=3, chunk_size=5)))
    assert np.array_equal(chunked_ternary, full_ternary)


def test_fwht_inplace_matches_reference_fwht():
    rng = np.random.default_rng(7)
    values = rng.standard_normal(64)

    expected = fwht(values)
    actual = values.copy()
    _fwht_inplace(actual)

    assert np.allclose(actual, expected)


def test_compute_sumset_lowmem_matches_standard_for_binary_inputs():
    rng = np.random.default_rng(11)
    universe = generate_space(7, p=2)
    subset = universe[rng.choice(len(universe), size=23, replace=False)]

    expected_ints = np.sort(vectors_to_ints(compute_sumset(subset, p=2)))
    actual_ints = np.sort(vectors_to_ints(compute_sumset_lowmem(subset, p=2)))

    assert np.array_equal(actual_ints, expected_ints)


def test_compute_sumset_lowmem_falls_back_for_nonbinary_inputs():
    subset = np.array(
        [
            [0, 0],
            [1, 0],
            [2, 1],
        ],
        dtype=np.int8,
    )

    expected = compute_sumset(subset, p=3)
    actual = compute_sumset_lowmem(subset, p=3)

    assert set(map(tuple, actual)) == set(map(tuple, expected))


def test_compute_sumset_lowmem_from_ints_matches_vector_path():
    rng = np.random.default_rng(17)
    universe = generate_space(6, p=2)
    subset = universe[rng.choice(len(universe), size=19, replace=False)]
    subset_ints = vectors_to_ints(subset)

    expected = np.sort(vectors_to_ints(compute_sumset_lowmem(subset, p=2)))
    actual = np.sort(compute_sumset_lowmem_from_ints(subset_ints, 6, p=2))

    assert np.array_equal(actual, expected)


def test_lowmem_covering_and_complement_match_standard_covering_path():
    rng = np.random.default_rng(23)
    n = 8
    radius = 2
    k = 3
    universe = generate_space(n, p=2)
    centers = [universe[rng.integers(0, len(universe))] for _ in range(k)]
    bases = [generate_random_basis(n, p=2, rng=rng) for _ in range(k)]

    standard_covered = generate_covering(centers, radius, bases=bases, p=2, universe=None)
    standard_covered_ints = np.sort(vectors_to_ints(standard_covered))
    standard_complement_ints = np.sort(vectors_to_ints(complement(universe, standard_covered)))

    lowmem_covered_ints = generate_covering_ints_lowmem(n, centers, radius, bases=bases, p=2)
    lowmem_complement_ints = complement_ints_lowmem(n, lowmem_covered_ints, p=2)

    assert np.array_equal(lowmem_covered_ints, standard_covered_ints)
    assert np.array_equal(lowmem_complement_ints, standard_complement_ints)


def test_generate_ball_ints_lowmem_matches_single_ball_covering():
    rng = np.random.default_rng(29)
    n = 7
    center = rng.integers(0, 2, size=n, dtype=np.int8)
    basis = generate_random_basis(n, p=2, rng=rng)

    expected = np.sort(
        vectors_to_ints(
            generate_covering([center], 2, bases=[basis], p=2, universe=None)
        )
    )
    actual = generate_ball_ints_lowmem(n, center, 2, linear_transform=basis, p=2)

    assert np.array_equal(actual, expected)


def test_find_maximum_subspace_dimension_lowmem_matches_standard():
    rng = np.random.default_rng(31)
    universe = generate_space(7, p=2)
    covered = generate_covering(
        [universe[rng.integers(0, len(universe))] for _ in range(3)],
        1,
        bases=[generate_random_basis(7, p=2, rng=rng) for _ in range(3)],
        p=2,
        universe=None,
    )
    S = complement(universe, covered)
    S_plus_S = compute_sumset(S, p=2)

    expected = find_maximum_subspace_dimension(S_plus_S, p=2)
    actual = find_maximum_subspace_dimension_lowmem(vectors_to_ints(S_plus_S), 7, p=2)

    assert actual == expected


def test_estimate_memory_gb_shows_lowmem_advantage_for_large_binary_dimensions():
    est = estimate_memory_gb(30, p=2)

    assert est["compute_sumset_lowmem_GB"] < est["compute_sumset_standard_GB"]
    assert est["complement_bool_mask_GB"] < est["S_ints_worst_case_GB"]


def test_run_trial_lowmem_matches_standard_for_same_seed():
    rng_standard = np.random.default_rng(101)
    rng_lowmem = np.random.default_rng(101)

    expected = _run_trial_standard(10, 3, 1, 2, rng_standard, exhaustive=False)
    actual = _run_trial_lowmem(10, 3, 1, 2, rng_lowmem, exhaustive=False)

    assert actual == expected


def test_cli_memory_check_prints_expected_sections(capsys):
    cli_main(["memory-check", "--n", "20", "24"])

    captured = capsys.readouterr()
    assert "n=20, p=2" in captured.out
    assert "compute_sumset (low-memory FWHT)" in captured.out
    assert captured.err == ""


def test_cli_run_low_memory_writes_json_output(tmp_path, capsys):
    output_path = tmp_path / "results.json"

    cli_main(
        [
            "run",
            "--n",
            "8",
            "--k",
            "3",
            "--radius",
            "1",
            "--trials",
            "5",
            "--seed",
            "7",
            "--low-memory",
            "--no-progress",
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(output_path.read_text())

    assert "[low-memory mode]" in captured.out
    assert "Completed 5 trials" in captured.out
    assert payload["summary"]["mode"] == "low-memory"
    assert payload["summary"]["trials_completed"] == 5
    assert len(payload["results"]) == 5


def test_cli_run_standard_warns_for_large_n_when_trial_execution_is_mocked(monkeypatch, capsys):
    def fake_run_trial_standard(n, k, radius, p, rng, exhaustive):
        return {"S_size": 0, "max_subspace_dim": None}

    monkeypatch.setattr("ffspaces.cli._run_trial_standard", fake_run_trial_standard)

    cli_main([
        "run",
        "--n",
        "21",
        "--trials",
        "1",
        "--no-progress",
    ])

    captured = capsys.readouterr()
    assert "WARNING: n=21 with standard mode may require many GB of RAM" in captured.err
    assert "mode=standard" in captured.out