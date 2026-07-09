from .core import generate_space, generate_random_basis, is_invertible
from .fwht_operators import vectors_to_ints, ints_to_vectors, compute_sumset_fwht
from .operators import (
    _compute_sumset_original,
    compute_sumset,
    compare_sumset_methods,
    find_maximum_subspace_dimension,
)
from .covers import generate_covering, complement
from .geometries import compute_hamming_weight, generate_hamming_ball, generate_standard_ball
from .lowmem import (
    generate_space_chunked,
    compute_sumset_lowmem,
    compute_sumset_lowmem_from_ints,
    generate_ball_ints_lowmem,
    generate_covering_ints_lowmem,
    complement_ints_lowmem,
    find_maximum_subspace_dimension_lowmem,
    estimate_memory_gb,
)

__all__ = [
    "generate_space",
    "generate_random_basis",
    "is_invertible",
    "vectors_to_ints",
    "ints_to_vectors",
    "compute_sumset_fwht",
    "_compute_sumset_original",
    "compute_sumset",
    "compare_sumset_methods",
    "find_maximum_subspace_dimension",
    "generate_covering",
    "complement",
    "compute_hamming_weight",
    "generate_hamming_ball",
    "generate_standard_ball",
    # low-memory API
    "generate_space_chunked",
    "compute_sumset_lowmem",
    "compute_sumset_lowmem_from_ints",
    "generate_ball_ints_lowmem",
    "generate_covering_ints_lowmem",
    "complement_ints_lowmem",
    "find_maximum_subspace_dimension_lowmem",
    "estimate_memory_gb",
]
