# Finite-Field Spaces (`ffspaces`)

`ffspaces` is a lightweight Python toolkit for working with vectors over finite fields $\mathbb{F}_p^n$, with a focus on sumset analysis, linear-subspace structure, and Hamming-ball geometry.

The package is designed for small-to-medium computational experiments in additive combinatorics, coding theory, and related discrete geometry settings. It provides both direct implementations and faster specialized routines for common operations over finite fields.

## Current functionality

- **Finite-field vector spaces:** generate the full vector space $\mathbb{F}_p^n$ or construct random invertible basis changes.
- **Finite-field rank:** compute matrix rank over $\mathbb{F}_p$ using modular Gaussian elimination.
- **Sumsets:** compute $S+S$ for a set of vectors over $\mathbb{F}_p^n$, including a direct reference method and a faster FWHT-based implementation.
- **Subspace detection:** find the largest linear subspace contained in a sumset, with both greedy and exhaustive search modes.
- **Hamming-ball geometry:** generate standard Hamming balls, compute Hamming weights, and build balls under arbitrary linear transforms.
- **Cover constructions:** build unions of Hamming balls and compute complements relative to a universe.

## Features

- **NumPy-based operations:** vectorized generation and manipulation of finite-field vectors.
- **Field-aware linear algebra:** rank and invertibility checks performed over $\mathbb{F}_p$, not over the reals.
- **Flexible geometry tools:** support for binary and higher-characteristic experiments.
- **Reusable experiment helpers:** scripts and utilities in the `experiments/` directory for one-off analysis.

## Installation

Install from PyPI:

```bash
pip install ffspaces
```

Or clone the repository and install in editable mode:

```bash
git clone https://github.com/nathanhoehndorf/finite-field-spaces
cd finite-field-spaces
pip install -e .
```

The package requires Python 3.10+ and NumPy.

## Package API

After installing with `pip install ffspaces`, import the public API from the package root:

```python
from ffspaces import (
    generate_space,
    generate_random_basis,
    is_invertible,
    vectors_to_ints,
    ints_to_vectors,
    compute_sumset_fwht,
    _compute_sumset_original,
    compute_sumset,
    compare_sumset_methods,
    find_maximum_subspace_dimension,
    generate_covering,
    complement,
    compute_hamming_weight,
    generate_hamming_ball,
    generate_standard_ball,
)
```

### Core utilities

- `generate_space(n, p=2)`: returns all vectors in $\mathbb{F}_p^n$ as a NumPy array of shape `(p**n, n)`.
- `generate_random_basis(n, p=2, rng=None)`: returns a random invertible `n x n` matrix over $\mathbb{F}_p$.
- `is_invertible(matrix, p=2)`: checks whether a square matrix is invertible over $\mathbb{F}_p$.

### Vector encoding

- `vectors_to_ints(vectors, p=2)`: maps each row of a matrix in $\mathbb{Z}_p^n$ to a unique integer.
- `ints_to_vectors(ints, n, p=2)`: converts integer codes back into $\mathbb{Z}_p^n$ vectors.

### Sumset operations

- `compute_sumset_fwht(set_elements, p=2)`: computes the sumset $S+S$ using a Fourier-based transform over the finite field.
- `_compute_sumset_original(set_elements, p=2)`: computes $S+S$ by direct pairwise sums and unique filtering.
- `compute_sumset(set_elements, p=2)`: alias for the fast FWHT-based sumset computation.
- `compare_sumset_methods(n=6, subset_size=10, seed=None)`: generates a random binary subset and compares FWHT and direct sumset methods.

### Subspace analysis

- `find_maximum_subspace_dimension(sumset, p=2, exhaustive=False, max_combinations=10000)`: returns the dimension of the largest linear subspace contained in a sumset.

### Hamming geometry and covering

- `generate_standard_ball(n, r)`: returns all binary vectors in $\mathbb{F}_2^n$ with Hamming weight at most `r`.
- `compute_hamming_weight(vectors)`: computes Hamming weights for each row of a matrix.
- `generate_hamming_ball(universe, center, radius, linear_transform, p=2)`: builds a Hamming ball in $\mathbb{F}_p^n$ under a linear basis transform.
- `generate_covering(centers, radii, bases=None, p=2, universe=None)`: returns the union of Hamming balls defined by `centers`, `radii`, and optional basis transforms.
- `complement(universe, covered)`: returns vectors in `universe` that are not present in `covered`.

### Example usage

```python
import numpy as np
from ffspaces import generate_space, compute_sumset, find_maximum_subspace_dimension

space = generate_space(3, p=2)
subset = space[[0, 1, 2, 4]]
sumset = compute_sumset(subset, p=2)
dimension = find_maximum_subspace_dimension(sumset, p=2)
print("sumset shape:", sumset.shape)
print("max subspace dimension:", dimension)
```

```python
from ffspaces import generate_standard_ball, generate_covering, complement

ball = generate_standard_ball(4, 2)
universe = generate_space(4, p=2)
covered = generate_covering([np.zeros(4, dtype=int)], [2], p=2)
missing = complement(universe, covered)
print("covered size:", covered.shape[0])
print("missing size:", missing.shape[0])
```

## Mathematical background

This package works in the vector space $\mathbb{F}_p^n$, where each coordinate is taken modulo a prime $p$. In particular, $\mathbb{F}_2^n$ is the set of binary vectors of length $n$, with addition done coordinatewise modulo $2$.

- **What is $\mathbb{F}_p^n$?** A finite-dimensional vector space over the finite field with $p$ elements. For $p=2$, this is the space of binary vectors.
- **What is $S+S$?** For a set $S \subseteq \mathbb{F}_p^n$, the sumset $S+S$ is the set of all pairwise sums $x+y$ with $x,y \in S$, taken modulo $p$.
- **What is a Hamming ball?** A Hamming ball is the set of vectors within a given distance from a center, where distance is the number of coordinates in which two vectors differ (known as Hamming weight, or the Hamming metric).
- **What is the covering problem?** A covering problem asks whether a collection of balls (or other subsets) can be chosen so that their union contains all points in a given universe, or perhaps all points of a particular structure.
- **Why does this matter?** These objects are central in additive combinatorics, coding theory, and discrete geometry, and they provide a natural setting for studying how small sets expand under addition or how geometric configurations cover a space.


## Quick example

```python
import numpy as np
from ffspaces import generate_space, compute_sumset, find_maximum_subspace_dimension

space = generate_space(3, p=2)
subset = space[[0, 1, 2, 4]]
sumset = compute_sumset(subset, p=2)
dimension = find_maximum_subspace_dimension(sumset, p=2)
print(dimension)
```

## Experiments

One-off analysis and experiment scripts live in the `experiments/` directory so they stay separate from the installable package code in `src/`.
