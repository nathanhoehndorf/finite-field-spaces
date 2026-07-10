import numpy as np

from ffspaces.core import generate_random_basis, is_invertible
from ffspaces.fwht_operators import compute_sumset_fwht
from ffspaces.operators import _compute_sumset_original, compare_sumset_methods


def test_random_subset_matches_fwht_and_original():
    subset, fwht_sumset, original_sumset = compare_sumset_methods(n=6, subset_size=10, seed=7)

    assert subset.shape == (10, 6)
    assert {tuple(row) for row in fwht_sumset} == {tuple(row) for row in original_sumset}


def test_generate_random_basis_accepts_rng():
    rng = np.random.default_rng(17)
    basis = generate_random_basis(4, p=2, rng=rng)
    basis_again = generate_random_basis(4, p=2, rng=np.random.default_rng(17))

    assert basis.shape == (4, 4)
    assert is_invertible(basis, p=2)
    assert np.array_equal(basis, basis_again)


def test_general_fourier_sumset_matches_original_for_p3():
    subset = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
        [2, 0],
        [0, 2],
    ], dtype=np.int8)

    fwht_sumset = compute_sumset_fwht(subset, p=3)
    original_sumset = _compute_sumset_original(subset, p=3)

    assert {tuple(row) for row in fwht_sumset} == {tuple(row) for row in original_sumset}


def test_is_invertible_uses_rank_over_f2():
    invertible = np.array([[1, 1], [1, 0]], dtype=np.int8)
    singular = np.array([[1, 1], [1, 1]], dtype=np.int8)

    assert is_invertible(invertible, p=2)
    assert not is_invertible(singular, p=2)
