import numpy as np
import pytest

from ffspaces.core import generate_space, generate_random_basis, is_invertible, rank_mod_p
from ffspaces.operators import (
    _compute_sumset_original,
    compute_sumset,
    find_maximum_subspace_dimension,
)
from ffspaces.geometries import generate_standard_ball


@pytest.mark.parametrize(
    ("n", "p", "expected"),
    [
        (0, 2, 1),
        (1, 2, 2),
        (2, 2, 4),
        (3, 3, 27),
    ],
)
def test_generate_space_edge_cases(n, p, expected):
    space = generate_space(n, p=p)
    assert space.shape == (expected, n)
    assert np.array_equal(space[0], np.zeros(n, dtype=np.int8))


def test_generate_space_full_space_contains_all_vectors():
    space = generate_space(3, p=2)
    assert set(map(tuple, space)) == {
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
    }


def test_is_invertible_edge_cases():
    assert is_invertible(np.array([[1]], dtype=np.int8), p=2)
    assert not is_invertible(np.array([[0]], dtype=np.int8), p=2)
    assert not is_invertible(np.array([[1, 0], [0, 0]], dtype=np.int8), p=2)
    assert is_invertible(np.array([[1, 1], [1, 0]], dtype=np.int8), p=2)


def test_compute_sumset_handles_empty_and_singleton_sets():
    empty = np.empty((0, 3), dtype=np.int8)
    singleton = np.array([[1, 0, 1]], dtype=np.int8)

    assert compute_sumset(empty, p=2).shape == (0, 3)
    assert _compute_sumset_original(empty, p=2).shape == (0, 3)

    expected_singleton = np.array([[0, 0, 0]], dtype=np.int8)
    computed_singleton = compute_sumset(singleton, p=2)
    assert computed_singleton.shape == (1, 3)
    assert np.array_equal(computed_singleton, expected_singleton)


def test_compute_sumset_full_space_matches_original():
    full_space = generate_space(3, p=2)
    computed = compute_sumset(full_space, p=2)
    original = _compute_sumset_original(full_space, p=2)
    assert set(map(tuple, computed)) == set(map(tuple, original))
    assert set(map(tuple, computed)) == set(map(tuple, full_space))


@pytest.mark.parametrize(
    ("sumset", "p", "expected"),
    [
        (np.array([[0, 0]], dtype=np.int8), 2, 0),
        (np.array([[0, 0], [1, 0]], dtype=np.int8), 2, 1),
        (np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int8), 2, 2),
    ],
)
def test_find_maximum_subspace_dimension_edge_cases(sumset, p, expected):
    assert find_maximum_subspace_dimension(sumset, p=p) == expected


def test_find_maximum_subspace_dimension_exhaustive_matches_greedy():
    sumset = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.int8,
    )
    assert find_maximum_subspace_dimension(sumset, p=2) == 3
    assert find_maximum_subspace_dimension(sumset, p=2, exhaustive=True) == 3


def test_find_maximum_subspace_dimension_uses_field_rank_over_f2():
    sumset = np.array(
        [
            [0, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 1, 0],
        ],
        dtype=np.int8,
    )
    assert find_maximum_subspace_dimension(sumset, p=2) == 2
    assert find_maximum_subspace_dimension(sumset, p=2, exhaustive=True) == 2


def test_find_maximum_subspace_dimension_exhaustive_respects_combinatorial_limit():
    sumset = generate_space(6, p=2)
    greedy = find_maximum_subspace_dimension(sumset, p=2)
    assert find_maximum_subspace_dimension(sumset, p=2, exhaustive=True, max_combinations=10) == greedy


def test_generate_standard_ball_regression_matches_expected_vectors():
    ball = generate_standard_ball(3, 1)
    expected = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=np.int8,
    )
    assert ball.shape == (4, 3)
    assert np.array_equal(ball, expected)


def test_generate_standard_ball_radius_zero_is_singleton_zero_vector():
    ball = generate_standard_ball(4, 0)
    assert ball.shape == (1, 4)
    assert np.array_equal(ball, np.zeros((1, 4), dtype=np.int8))

def test_rank_mod_p_catches_dependency_that_looks_indepenent_over_reals():
    a, b, c = [1,1,0,0], [1,0,1,0], [0,1,1,0]
    assert rank_mod_p(np.array([a,b,c]), p=2) == 2
