import itertools

import numpy as np


def generate_space(n: int, p: int = 2) -> np.ndarray:
    """
    Generates all p^n vectors in the vector space F_p^n.
    Returns a 2D numpy array of shape (p^n, n).
    """
    elements = list(itertools.product(range(p), repeat=n))
    return np.array(elements, dtype=np.int8)

def rank_mod_p(matrix, p: int = 2) -> int:
    """
    Computes the rank of `matrix` over the finite field F_p using Gaussian elimination
    with modular arithmetic. Works for any prime p; matrix need not be square
    """
    if p<=1:
        raise ValueError("p must be a prime greater than 1")
    
    A = np.array(matrix, dtype=np.int64, copy=True) % p
    if A.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")
    
    m, n = A.shape
    rank = 0
    for col in range(n):
        pivot_row = None
        for row in range(rank, m):
            if A[row, col] % p != 0:
                pivot_row = row
                break
        if pivot_row is None:
            continue
                
        if pivot_row != rank:
            A[[rank, pivot_row], :] = A[[pivot_row, rank], :]

        pivot_inverse = pow(int(A[rank, col]), -1, p)
        for row in range(m):
            if row == rank:
                continue
            factor = (A[row, col] * pivot_inverse) % p
            if factor != 0:
                A[row, :] = (A[row, :] - factor * A[rank, :]) % p

        rank += 1
        if rank == min(m, n):
            break

    return rank

def is_invertible(matrix: np.ndarray, p: int = 2) -> bool:
    """Checks if an n x n matrix is invertible over the finite field F_p."""
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    return rank_mod_p(matrix, p) == matrix.shape[0]


def generate_random_basis(n: int, p: int = 2, rng=None) -> np.ndarray:
    """Generates a random invertible n x n matrix over F_p to represent a basis change."""
    if rng is None:
        rng = np.random.default_rng()

    while True:
        matrix = rng.integers(0, p, size=(n, n)).astype(np.int8)
        if is_invertible(matrix, p):
            return matrix

