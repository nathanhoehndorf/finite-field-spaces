"""
Memory-efficient implementations for large n (n > 20).

For the standard (dense) implementations see core.py, operators.py,
geometries.py, and covers.py.  The functions here avoid materialising
the full universe array (p^n × n bytes) and minimise peak heap allocation,
at the cost of some overhead.
"""

from __future__ import annotations

import itertools
from typing import Generator, Iterable, Optional, Sequence

import numpy as np

from .fwht_operators import vectors_to_ints, ints_to_vectors

def generate_space_chunked(
    n: int,
    p: int = 2,
    chunk_size: int = 1 << 14,
) -> Generator[np.ndarray, None, None]:
    """Yield F_p^n in successive chunks without materialising the full space.

    Args:
        n: Dimension of the space.
        p: Field characteristic (prime).
        chunk_size: Maximum number of vectors per yielded chunk.
    """
    total = p ** n
    if p == 2:
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            ints = np.arange(start, end, dtype=np.int64)
            yield ints_to_vectors(ints, n, p=2)
    else:
        batch: list = []
        for vec in itertools.product(range(p), repeat=n):
            batch.append(vec)
            if len(batch) == chunk_size:
                yield np.array(batch, dtype=np.int8)
                batch.clear()
        if batch:
            yield np.array(batch, dtype=np.int8)


def _fwht_inplace(a: np.ndarray) -> None:
    """In-place Walsh-Hadamard Transform of 1-D float64 array ``a`` (length 2^k).

    Equivalent to ``fwht_operators.fwht`` but modifies ``a`` directly,
    avoiding the initial full-array copy. Peak overhead per stage: N/2 × 8 bytes.
    """
    N = len(a)
    k = N.bit_length() - 1
    if 1 << k != N:
        raise ValueError("Array length must be a power of 2")

    for d in range(k):
        half = 1 << d
        view = a.reshape(-1, 2, half)
        top = view[:, 0, :]  
        bot = view[:, 1, :]  
        tmp = top + bot        
        bot[:] = top - bot     
        top[:] = tmp
        # tmp goes out of scope here; memory can be reclaimed

def compute_sumset_lowmem(
    set_elements: np.ndarray,
    p: int = 2,
) -> np.ndarray:
    """Compute S+S over F_p^n with reduced peak memory.

    For p==2 the Fourier convolution uses a real float64 FWHT instead of a
    complex128 DFT, halving the bytes per element in the indicator and
    transform arrays.  The transform is applied in-place (_fwht_inplace) with
    only one N/2 temporary per stage.

    For p>2 falls back to ``fwht_operators.compute_sumset_fwht``.

    Args:
        set_elements: Shape (M, n) int8 array of set vectors.
        p: Field characteristic.

    Returns:
        Shape (K, n) int8 array of sumset vectors (unique, sorted).
    """
    if p != 2:
        from .fwht_operators import compute_sumset_fwht
        return compute_sumset_fwht(set_elements, p=p)

    if len(set_elements) == 0:
        return np.empty((0, set_elements.shape[1]), dtype=np.int8)

    n = set_elements.shape[1]
    N = 1 << n

    ints = vectors_to_ints(set_elements, p=2)

    indicator = np.zeros(N, dtype=np.float64)
    indicator[ints] = 1.0
    del ints

    _fwht_inplace(indicator)         # in-place, one N/2 temp per stage
    indicator *= indicator            # square in-place
    _fwht_inplace(indicator)         # inverse FWHT (self-inverse up to scale)
    indicator /= N

    sumset_ints = np.flatnonzero(np.rint(indicator).astype(np.int32) > 0)
    del indicator  # release before decoding

    return ints_to_vectors(sumset_ints, n, p=2)


def compute_sumset_lowmem_from_ints(
    set_ints: np.ndarray,
    n: int,
    p: int = 2,
) -> np.ndarray:
    """Like compute_sumset_lowmem but accepts the set as integer encodings.

    Avoids converting from integer encodings to a vector matrix and back,
    which saves an M x n int8 array allocation.

    Args:
        set_ints: 1-D array of integer encodings of set elements.
        n: Dimension of the space.
        p: Field characteristic.
    """
    if p != 2:
        raise NotImplementedError(
            "Integer-input sumset path is only implemented for p=2; "
            "convert to vectors and use compute_sumset_lowmem for p>2."
        )
    if len(set_ints) == 0:
        return np.empty(0, dtype=np.int64)

    N = 1 << n

    indicator = np.zeros(N, dtype=np.float64)
    indicator[set_ints] = 1.0

    _fwht_inplace(indicator)
    indicator *= indicator
    _fwht_inplace(indicator)
    indicator /= N

    result = np.flatnonzero(np.rint(indicator).astype(np.int32) > 0).astype(np.int64)
    del indicator
    return result


def _ball_offsets_p2(
    n: int,
    radius: int,
    linear_transform: Optional[np.ndarray],
) -> Iterable[np.ndarray]:
    """Yield XOR offsets for a p=2 Hamming ball.

    Each yielded offset o satisfies: center XOR o is a member of the ball
    B_L(center, radius) = {x : weight(L*(x XOR center)) <= radius}.
    The convention matches generate_covering's weight-ball path:
        offset = (w @ L.T) % 2  for each weight-ball vector w.
    """
    yield np.zeros(n, dtype=np.int8)

    for weight in range(1, radius + 1):
        for combo in itertools.combinations(range(n), weight):
            if linear_transform is not None:
                offset = linear_transform[:, list(combo)].sum(axis=1).astype(np.int8) % 2
            else:
                offset = np.zeros(n, dtype=np.int8)
                for i in combo:
                    offset[i] = 1
            yield offset


def generate_ball_ints_lowmem(
    n: int,
    center: np.ndarray,
    radius: int,
    linear_transform: Optional[np.ndarray] = None,
    p: int = 2,
) -> np.ndarray:
    """Return integer encodings of the Hamming ball B_L(center, radius).

    This mirrors geometries.generate_hamming_ball but does NOT require the
    full universe array.  Instead the ball is enumerated combinatorially,
    iterating over weight-ball offset vectors.

    Args:
        n: Dimension.
        center: Length-n int8 array (centre of the ball).
        radius: Ball radius (Hamming weight threshold).
        linear_transform: Optional matrix L (n×n int8).  Ball is measured
            in the L-transformed basis.  Corresponds to the `bases` parameter
            of generate_covering.
        p: Field characteristic (currently only p=2 is supported).
    """
    if p != 2:
        raise NotImplementedError(
            "Combinatorial ball generation without universe is only implemented "
            "for p=2.  For p>2, pass a universe to geometries.generate_hamming_ball."
        )

    center = np.asarray(center, dtype=np.int8)
    powers = np.int64(2) ** np.arange(n - 1, -1, -1, dtype=np.int64)

    ints_list: list[int] = []
    for offset in _ball_offsets_p2(n, radius, linear_transform):
        # XOR offset with center to get the ball vector
        vec_int = int(np.dot((offset ^ center).astype(np.int64), powers))
        ints_list.append(vec_int)

    return np.unique(np.array(ints_list, dtype=np.int64))

def generate_covering_ints_lowmem(
    n: int,
    centers: Sequence,
    radii: int | Sequence[int],
    bases: Optional[Sequence[Optional[np.ndarray]]] = None,
    p: int = 2,
) -> np.ndarray:
    """Return sorted integer indices of the union of Hamming balls.

    Equivalent to generate_covering(..., universe=None) for p=2 followed by
    vectors_to_ints, but does not materialise the intermediate vector arrays.

    Args:
        n: Dimension.
        centers: Sequence of length-n center vectors.
        radii: Single radius or one per center.
        bases: Optional sequence of linear transforms (None = identity).
        p: Field characteristic (only p=2 currently supported).
    """
    if p != 2:
        raise NotImplementedError(
            "Integer-only covering path is only implemented for p=2."
        )

    if isinstance(radii, int):
        radii = [radii] * len(centers)
    if bases is None:
        bases = [None] * len(centers)

    if not (len(centers) == len(radii) == len(bases)):
        raise ValueError("centers, radii, and bases must have the same length")

    parts: list[np.ndarray] = []
    for c, r, B in zip(centers, radii, bases):
        ball_ints = generate_ball_ints_lowmem(
            n, np.asarray(c, dtype=np.int8), r, B, p
        )
        parts.append(ball_ints)

    if not parts:
        return np.empty(0, dtype=np.int64)

    return np.unique(np.concatenate(parts))


def complement_ints_lowmem(
    n: int,
    covered_ints: np.ndarray,
    p: int = 2,
) -> np.ndarray:
    """Return integer indices of vectors NOT in covered_ints.

    Uses a boolean mask (1 byte/element) rather than an int64 array
    (8 bytes/element), reducing memory by 8x.

    Args:
        n: Dimension of the space.
        covered_ints: 1-D array of covered integer indices.
        p: Field characteristic.
    """
    total = p ** n
    is_covered = np.zeros(total, dtype=bool)
    is_covered[covered_ints] = True
    return np.flatnonzero(~is_covered).astype(np.int64)

def _in_sorted(arr: np.ndarray, val: int) -> bool:
    """Binary search: True iff val is in sorted 1-D array arr."""
    idx = np.searchsorted(arr, val)
    return idx < len(arr) and arr[idx] == val


def find_maximum_subspace_dimension_lowmem(
    sumset_ints: np.ndarray,
    n: int,
    p: int = 2,
) -> int:
    """Greedy maximum subspace dimension using an integer-encoded sumset.

    This mirrors operators.find_maximum_subspace_dimension (greedy path) but
    replaces the Python tuple-set membership test with sorted-array binary
    search.

    Args:
        sumset_ints: Sorted (or unsorted; will be sorted internally) 1-D int64
            array of integer encodings of S+S elements.
        n: Dimension of the space.
        p: Field characteristic.
    """
    from .core import rank_mod_p

    ss = np.sort(sumset_ints)

    if not _in_sorted(ss, 0):
        return -1

    n_ss = len(ss)
    max_possible_d = 0
    while p ** (max_possible_d + 1) <= n_ss and max_possible_d < n:
        max_possible_d += 1

    powers = p ** np.arange(n - 1, -1, -1, dtype=np.int64)

    def in_ss(vec: np.ndarray) -> bool:
        return _in_sorted(ss, int(np.dot(vec.astype(np.int64), powers)))

    generators: list[np.ndarray] = []
    current_span: list[np.ndarray] = [np.zeros(n, dtype=np.int8)]

    for cand_int in ss:
        cand = ints_to_vectors(np.array([cand_int], dtype=np.int64), n, p)[0]
        if np.all(cand == 0):
            continue

        # Check: does adding cand keep the extended span inside S+S?
        valid = True
        for s in current_span:
            for k in range(1, p):
                if not in_ss((s + k * cand) % p):
                    valid = False
                    break
            if not valid:
                break

        if not valid:
            continue

        # Check linear independence
        if generators:
            mat = np.vstack(generators + [cand])
            if rank_mod_p(mat, p) != len(generators) + 1:
                continue

        # Accept: extend the maintained span.
        current_span += [(cand * k + s) % p for k in range(1, p) for s in current_span]
        generators.append(cand)

    return len(generators)

def estimate_memory_gb(n: int, p: int = 2) -> dict[str, float]:
    """Estimate peak memory (GB) for key low-memory operations at dimension n.

    Returns a dictionary with entries for the main operations so users can
    judge whether the low-memory mode is sufficient for their hardware.
    """
    universe_size = float(p ** n)

    return {
        # indicator (N float64) + one N/2 temp per FWHT stage = 1.5 × N × 8 bytes
        "compute_sumset_lowmem_GB": universe_size * 8 * 1.5 / 1e9,
        "complement_bool_mask_GB": universe_size * 1 / 1e9,
        "S_ints_worst_case_GB": universe_size * 0.5 * 8 / 1e9,
        "sumset_ints_worst_case_GB": universe_size * 8 / 1e9,
        "generate_space_standard_GB": universe_size * n / 1e9,
        "compute_sumset_standard_GB": universe_size * 16 * 4 / 1e9,
    }
