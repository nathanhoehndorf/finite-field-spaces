import numpy as np
import itertools

def generate_space(n: int, p: int = 2) -> np.ndarray:
    """
    Generates all p^n vectors in the vector space F_p^n.
    Returns a 2D numpy array of shape (p^n, n).
    """
    elements = list(itertools.product(range(p), repeat=n))
    return np.array(elements, dtype=np.int8)

def is_invertible(matrix: np.ndarray, p: int = 2) -> bool:
    """Checks if an n x n matrix is invertible over the finite field F_p."""
    matrix = np.array(matrix, dtype=np.int64, copy=True)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False

    n = matrix.shape[0]
    if p <= 1:
        raise ValueError("p must be a prime greater than 1")

    rank = 0
    for col in range(n):
        pivot_row = None
        for row in range(rank, n):
            if matrix[row, col] % p != 0:
                pivot_row = row
                break

        if pivot_row is None:
            continue

        if pivot_row != rank:
            matrix[[rank, pivot_row], :] = matrix[[pivot_row, rank], :]

        pivot_value = matrix[rank, col] % p
        pivot_inverse = pow(int(pivot_value), -1, p)

        for row in range(rank + 1, n):
            factor = (matrix[row, col] % p) * pivot_inverse % p
            if factor != 0:
                matrix[row, :] = (matrix[row, :] - factor * matrix[rank, :]) % p

        rank += 1
        if rank == n:
            return True

    return False

def generate_random_basis(n: int, p: int = 2, rng=None) -> np.ndarray:
    """Generates a random invertible n x n matrix over F_p to represent a basis change."""
    if rng is None:
        rng = np.random.default_rng()

    while True:
        matrix = rng.integers(0, p, size=(n, n)).astype(np.int8)
        if is_invertible(matrix, p):
            return matrix