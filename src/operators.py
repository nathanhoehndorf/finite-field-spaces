from itertools import product, combinations
from typing import Optional, Tuple

import numpy as np

from src.fwht_operators import compute_sumset_fwht


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


def find_maximum_subspace_dimension(sumset: np.ndarray, p: int = 2, exhaustive: bool = False) -> int:
    """
    Finds the dimension of the largest linear subspace completely contained within S+S.
    By default uses a greedy iterative check of linearly independent generators. If
    `exhaustive=True` it will search all combinations of generators (starting from
    the largest possible dimension) to find the maximum dimension whose span is
    fully contained in `sumset`.
    """
    n = sumset.shape[1]
    sumset_set = {tuple(row) for row in sumset}

    # Origin must be in sumset for valid subspace
    if tuple(np.zeros(n, dtype=np.int8)) not in sumset_set:
        return -1

    # Quick bound: a subspace of dimension d has size p**d, so d <= floor(log_p(|sumset|))
    max_possible_d = 0
    size = len(sumset)
    if size > 0:
        while p ** (max_possible_d + 1) <= size and max_possible_d < n:
            max_possible_d += 1

    if exhaustive:
        # Try dimensions from largest feasible down to 0
        for d in range(max_possible_d, 0, -1):
            # iterate over all combinations of d non-zero candidates from sumset
            nonzero_rows = [row for row in sumset if not np.all(row == 0)]
            for combo in combinations(nonzero_rows, d):
                mat = np.vstack(combo)
                if np.linalg.matrix_rank(mat.astype(float)) != d:
                    continue

                # build span
                all_in_sumset = True
                for coeffs in product(range(p), repeat=d):
                    if all(c == 0 for c in coeffs):
                        vec = tuple(np.zeros(n, dtype=np.int8))
                    else:
                        vec_arr = sum((coeffs[i] * mat[i]) for i in range(d)) % p
                        vec = tuple(vec_arr.astype(np.int8))
                    if vec not in sumset_set:
                        all_in_sumset = False
                        break

                if all_in_sumset:
                    return d

        # if none found, only the trivial subspace (dimension 0) remains
        return 0

    # Greedy default behaviour
    independent_generators = []

    for candidate in sumset:
        if np.all(candidate == 0):
            continue
        tuple_cand = tuple(candidate)

        valid_generator = True
        current_span = [np.zeros(n, dtype=np.int8)]

        for gen in independent_generators:
            current_span += [(gen * k + s) % p for k in range(1, p) for s in current_span]

        for vec in current_span:
            for k in range(1, p):
                test_vec = (vec + k * candidate) % p
                if tuple(test_vec) not in sumset_set:
                    valid_generator = False
                    break
            if not valid_generator:
                break

        if valid_generator:
            # Verify linear independence
            test_matrix = np.array(independent_generators + [candidate])
            if np.linalg.matrix_rank(test_matrix.astype(float)) == len(independent_generators) + 1:
                independent_generators.append(candidate)

    return len(independent_generators)

