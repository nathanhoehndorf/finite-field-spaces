import numpy as np
from typing import Sequence, Optional

from src.geometries import generate_hamming_ball
from src.fwht_operators import vectors_to_ints


def _generate_weight_ball(n: int, radius: int) -> np.ndarray:
    """
    Generate all binary vectors in F_2^n with Hamming weight <= radius.
    """
    import itertools

    vectors = []
    for weight in range(radius + 1):
        for combo in itertools.combinations(range(n), weight):
            vec = np.zeros(n, dtype=np.int8)
            vec[list(combo)] = 1
            vectors.append(vec)
    if len(vectors) == 0:
        return np.empty((0, n), dtype=np.int8)
    return np.array(vectors, dtype=np.int8)


def generate_covering(centers: Sequence[np.ndarray], radii, bases: Optional[Sequence[np.ndarray]] = None,
                      p: int = 2, universe: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Produce the union of Hamming balls defined by `centers`, `radii` and `bases`.

    - `centers`: sequence of center vectors (each length n)
    - `radii`: either a single int or sequence matching centers
    - `bases`: either None, a single linear transform, or sequence of linear transforms
    - `universe`: optional full universe array; when provided `generate_hamming_ball`
      will be used (supports arbitrary p). When omitted and p==2 a combinatorial
      weight-based generation is used for speed.

    Returns a numpy array of unique covered vectors (dtype=int8) with shape (M,n).
    """
    if isinstance(radii, int):
        radii = [radii] * len(centers)
    if bases is None:
        bases = [None] * len(centers)
    if not (len(centers) == len(radii) == len(bases)):
        raise ValueError("centers, radii and bases must have the same length")

    centers = [np.array(c, dtype=np.int8) for c in centers]
    n = centers[0].size if len(centers) > 0 else 0

    covered_ints = None
    covered_rows = []

    # Fast path for binary when no universe provided: generate weight vectors and transform
    if universe is None and p == 2:
        # Group centers by radius to avoid regenerating weight-balls
        weight_balls = {}
        for c, r, B in zip(centers, radii, bases):
            if r not in weight_balls:
                weight_balls[r] = _generate_weight_ball(n, r)
            W_r = weight_balls[r]
            if B is None:
                A = W_r.copy()
            else:
                A = (W_r @ B.T) % 2
            ball = (A ^ c).astype(np.int8)
            # use integer encoding for fast uniqueness
            ints = vectors_to_ints(ball)
            if covered_ints is None:
                covered_ints = ints
            else:
                covered_ints = np.concatenate([covered_ints, ints])

        if covered_ints is None:
            return np.empty((0, n), dtype=np.int8)

        unique_ints = np.unique(covered_ints)
        # convert back to vectors
        from src.fwht_operators import ints_to_vectors

        return ints_to_vectors(unique_ints, n)

    # Generic path using provided universe and generate_hamming_ball (works for any p)
    for c, r, B in zip(centers, radii, bases):
        ball = generate_hamming_ball(universe, c, r, B, p)
        covered_rows.append(ball.astype(np.int8))

    if len(covered_rows) == 0:
        return np.empty((0, n), dtype=np.int8)

    all_covered = np.vstack(covered_rows)
    # Deduplicate
    if p == 2:
        ints = vectors_to_ints(all_covered)
        unique_ints = np.unique(ints)
        from src.fwht_operators import ints_to_vectors

        return ints_to_vectors(unique_ints, all_covered.shape[1])
    else:
        return np.unique(all_covered, axis=0)


def complement(universe: np.ndarray, covered: np.ndarray) -> np.ndarray:
    """
    Return rows of `universe` that are not in `covered`.
    Both inputs are arrays of vectors (rows). Result preserves order from `universe`.
    """
    if len(universe) == 0:
        return universe.copy()
    if len(covered) == 0:
        return universe.copy()

    n = universe.shape[1]
    # Fast binary path
    try:
        covered_ints = vectors_to_ints(covered)
        universe_ints = vectors_to_ints(universe)
        covered_set = set(covered_ints.tolist())
        mask = [i not in covered_set for i in universe_ints.tolist()]
        return universe[np.array(mask, dtype=bool)]
    except Exception:
        covered_set = {tuple(row) for row in covered}
        return np.array([row for row in universe if tuple(row) not in covered_set], dtype=np.int8)
