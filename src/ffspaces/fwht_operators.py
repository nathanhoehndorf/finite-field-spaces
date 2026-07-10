import numpy as np


def vectors_to_ints(vectors: np.ndarray, p: int = 2) -> np.ndarray:
    """
    Converts a matrix of Z_p^n vectors (shape M x n) to a 1d array of integers.
    """
    if len(vectors) == 0:
        return np.array([], dtype=np.int64)

    if p <= 1:
        raise ValueError("p must be at least 2")

    n = vectors.shape[1]
    if np.any(vectors < 0) or np.any(vectors >= p):
        raise ValueError("All vector entries must lie in {0, ..., p-1}")

    powers = p ** np.arange(n - 1, -1, -1, dtype=np.int64)
    return np.dot(vectors.astype(np.int64), powers).astype(np.int64)


def ints_to_vectors(ints: np.ndarray, n: int, p: int = 2) -> np.ndarray:
    """
    Converts a 1d array of integers back to a matrix of Z_p^n vectors.
    """
    if p <= 1:
        raise ValueError("p must be at least 2")

    if len(ints) == 0:
        return np.empty((0, n), dtype=np.int8)

    ints_expanded = ints.astype(np.int64)[:, np.newaxis]
    digits = np.empty((len(ints), n), dtype=np.int8)
    remaining = ints_expanded.copy()

    for axis in range(n - 1, -1, -1):
        digits[:, axis] = (remaining[:, 0] % p).astype(np.int8)
        remaining = remaining // p

    return digits


def fwht(a: np.ndarray) -> np.ndarray:
    """
    Fast Walsh-Hadamard Transform of 1d array `a`.
    Returns a new float64 array. Array size must be exactly 2^n.
    Fully vectorized in NumPy.
    """
    n_elements = len(a)
    n_bits = n_elements.bit_length() - 1

    if 1 << n_bits != n_elements:
        raise ValueError("Array length must be a power of 2")

    res = a.astype(np.float64)

    for d in range(n_bits):
        half_step = 1 << d
        shape = (-1, 2, half_step)
        res_reshaped = res.reshape(shape)

        a0 = res_reshaped[:, 0, :].copy()
        a1 = res_reshaped[:, 1, :].copy()

        res_reshaped[:, 0, :] = a0 + a1
        res_reshaped[:, 1, :] = a0 - a1

    return res


def compute_sumset_fwht(set_elements: np.ndarray, p: int = 2) -> np.ndarray:
    """
    Computes S+S over Z_p^n using the Fourier transform on the finite abelian group.
    Time Complexity: O(p^n * n * log p) for a dense transform over the universe size p^n.
    Space Complexity: O(p^n)
    """
    if len(set_elements) == 0:
        return np.empty((0, set_elements.shape[1]), dtype=np.int8)

    if p <= 1:
        raise ValueError("p must be at least 2")

    n = set_elements.shape[1]

    ints = vectors_to_ints(set_elements, p=p)

    indicator = np.zeros((p,) * n, dtype=np.complex128)
    indicator.reshape(-1)[ints] = 1.0

    transformed = np.fft.fftn(indicator, axes=tuple(range(n)))
    transformed_squared = transformed * transformed

    convolution = np.fft.ifftn(transformed_squared, axes=tuple(range(n))).real
    sumset_ints = np.flatnonzero(np.rint(convolution).astype(np.int64) > 0)
    sumset_vectors = ints_to_vectors(sumset_ints, n, p=p)

    return sumset_vectors
