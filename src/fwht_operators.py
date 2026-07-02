import numpy as np

def vectors_to_ints(vectors: np.ndarray) -> np.ndarray:
    """
    Converts a matrix of F_2^n vectors (shape M x n) to a 1d array of integers
    """
    if len(vectors) == 0:
        return np.array([], dtype=np.int32)
    
    n = vectors.shape[1]
    powers_of_two = 1 << np.arange(n-1, -1, -1, dtype=np.int32)
    return np.dot(vectors, powers_of_two).astype(np.int32)

def ints_to_vectors(ints: np.ndarray, n: int) -> np.ndarray:
    """
    Converts a 1d array of integers back to a matrix of F_2^n vectors
    """
    if len(ints) == 0:
        return np.empty((0,n), dtype=np.int8)
    
    ints_expanded = ints[:, np.newaxis]
    powers_of_two = 1 << np.arange(n-1, -1, -1, dtype=np.int32)

    return ((ints_expanded & powers_of_two) > 0).astype(np.int8)

def fwht(a: np.ndarray) -> np.ndarray:
    """
    In-place Fast Walsh-Hadamard Transform of 1d array `a`.
    Array size must be exactly 2^n. Fully vectorized in NumPy.
    """
    n_elements = len(a)
    n_bits = int(np.log2(n_elements))

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

def compute_sumset_fwht(set_elements: np.ndarray) -> np.ndarray:
    """
    Computes S+S over F_2^n using the Fast Walsh-Hadamard Transform.
    Time Complexity: O(N log n) where N = 2^n
    Space Complexity: O(N)
    """
    if len(set_elements) == 0:
        return np.empty((0, set_elements.shape[1]), dtype=np.int8)
    
    n = set_elements.shape[1]
    universe_size = 1 << n

    ints = vectors_to_ints(set_elements)

    indicator = np.zeros(universe_size, dtype=np.float64)
    indicator[ints] = 1.0

    transformed = fwht(indicator)
    transformed_squared = transformed ** 2

    convolution = fwht(transformed_squared) / universe_size
    sumset_ints = np.where(convolution > 0.5)[0]
    sumset_vectors = ints_to_vectors(sumset_ints, n)

    return sumset_vectors 