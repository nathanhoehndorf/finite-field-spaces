import itertools
from typing import Optional

import numpy as np


def generate_standard_ball(n: int, r: int) -> np.ndarray:
    """
    Generates all vectors in F_2^n with Hamming weight <= r.
    """
    vectors = []
    for weight in range(r + 1):
        for combo in itertools.combinations(range(n), weight):
            vec = np.zeros(n, dtype=np.int8)
            vec[list(combo)] = 1
            vectors.append(vec)
    return np.array(vectors, dtype=np.int8)


def compute_hamming_weight(vectors: np.ndarray) -> np.ndarray:
    """
    Computes the Hamming weight for an array of vectors.
    Input shape: (M, n) -> Output shape (M,)
    """
    return np.count_nonzero(vectors, axis=1)


def generate_hamming_ball(
    universe: np.ndarray,
    center: np.ndarray,
    radius: int,
    linear_transform: Optional[np.ndarray] = None,
    p: int = 2,
) -> np.ndarray:
    """
    Generates a Hamming ball centered at `center` with a given `radius`
    over F_p^n. If `linear_transform` is provided, distances are measured
    in the transformed basis: B_L(v, r) = {x : weight(L(x-v) mod p) <= r}.
    """
    shifted = (universe - center) % p
    if linear_transform is not None:
        shifted = (shifted @ linear_transform.T) % p

    weights = compute_hamming_weight(shifted)
    mask = weights <= radius

    return universe[mask]
