from itertools import combinations, product
from typing import Optional, Tuple

import numpy as np

from .core import rank_mod_p
from .fwht_operators import compute_sumset_fwht


def _compute_sumset_original(set_elements: np.ndarray, p: int = 2) -> np.ndarray:
    """
    Computes the unique elements of the sumset S + S over F_p
    using the direct broadcast-based approach.
    """
    if len(set_elements) == 0:
        return np.empty((0, set_elements.shape[1]), dtype=np.int8)

    broadcasted_sum = (set_elements[:, np.newaxis, :] + set_elements[np.newaxis, :, :]) % p
    reshaped_sums = broadcasted_sum.reshape(-1, set_elements.shape[1])

    return np.unique(reshaped_sums, axis=0)


def compute_sumset(set_elements: np.ndarray, p: int = 2) -> np.ndarray:
    """
    Computes the unique elements of the sumset S + S over Z_p^n.
    This uses the generalized Fourier-transform fast path for all p >= 2.
    """
    if p <= 1:
        raise ValueError("p must be at least 2")

    return compute_sumset_fwht(set_elements, p=p)


def compare_sumset_methods(n: int = 6, subset_size: int = 10, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a random subset of F_2^n and compares the FWHT-based sumset against
    the original direct implementation.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    if subset_size < 1:
        raise ValueError("subset_size must be at least 1")

    universe = np.array(list(product([0, 1], repeat=n)), dtype=np.int8)
    rng = np.random.default_rng(seed)
    subset_indices = rng.choice(len(universe), size=subset_size, replace=False)
    subset = universe[subset_indices]

    fwht_sumset = compute_sumset(subset, p=2)
    original_sumset = _compute_sumset_original(subset, p=2)

    return subset, fwht_sumset, original_sumset


def _find_maximum_subspace_dimension_greedy(sumset: np.ndarray, p: int = 2) -> int:
    """Greedily build a maximal linearly independent set whose span is contained in sumset.

    The span of accepted generators is maintained incrementally: when a new
    generator is accepted it is extended once, rather than rebuilt from scratch
    for every candidate.
    """
    n = sumset.shape[1]
    sumset_set = {tuple(row) for row in sumset}
    independent_generators: list = []
    current_span = [np.zeros(n, dtype=np.int8)]

    for candidate in sumset:
        if np.all(candidate == 0):
            continue

        # Check whether the span extended by `candidate` stays inside sumset.
        valid = True
        for s in current_span:
            for k in range(1, p):
                if tuple((s + k * candidate) % p) not in sumset_set:
                    valid = False
                    break
            if not valid:
                break

        if not valid:
            continue

        # Check linear independence before accepting.
        if independent_generators:
            test_matrix = np.array(independent_generators + [candidate])
            if rank_mod_p(test_matrix, p) != len(independent_generators) + 1:
                continue

        # Accept: extend the maintained span with the new generator.
        current_span += [(candidate * k + s) % p for k in range(1, p) for s in current_span]
        independent_generators.append(candidate)

    return len(independent_generators)


def find_maximum_subspace_dimension(
    sumset: np.ndarray,
    p: int = 2,
    exhaustive: bool = False,
    max_combinations: Optional[int] = 10_000,
) -> int:
    """
    Finds the dimension of the largest linear subspace completely contained within S+S.
    By default uses a greedy iterative check of linearly independent generators. If
    `exhaustive=True` it will search combinations of generators (starting from the
    largest possible dimension) to find the maximum dimension whose span is fully
    contained in `sumset`. A `max_combinations` limit can be supplied to avoid
    combinatorial blowups in large experiments; if reached, the greedy result is returned.
    """
    n = sumset.shape[1]
    sumset_set = {tuple(row) for row in sumset}

    if tuple(np.zeros(n, dtype=np.int8)) not in sumset_set:
        return -1

    max_possible_d = 0
    size = len(sumset)
    if size > 0:
        while p ** (max_possible_d + 1) <= size and max_possible_d < n:
            max_possible_d += 1

    greedy_dimension = _find_maximum_subspace_dimension_greedy(sumset, p)

    if exhaustive:
        checked_combinations = 0
        nonzero_rows = [row for row in sumset if not np.all(row == 0)]
        for d in range(max_possible_d, 0, -1):
            for combo in combinations(nonzero_rows, d):
                if max_combinations is not None and checked_combinations >= max_combinations:
                    return greedy_dimension

                checked_combinations += 1
                mat = np.vstack(combo)
                if rank_mod_p(mat, p) != d:
                    continue

                all_in_sumset = True
                for coeffs in product(range(p), repeat=d):
                    vec_arr = sum(coeffs[i] * mat[i] for i in range(d)) % p
                    if tuple(vec_arr.astype(np.int8)) not in sumset_set:
                        all_in_sumset = False
                        break

                if all_in_sumset:
                    return d

        return 0

    return greedy_dimension
