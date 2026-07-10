# Finite-Field Spaces (`ffspaces`)

`ffspaces` is a lightweight Python toolkit for working with vectors over finite fields $\mathbb{F}_p^n$, with a focus on sumset analysis, linear-subspace structure, and Hamming-ball geometry.

The package is designed for small-to-medium computational experiments in additive combinatorics, coding theory, and related discrete geometry settings. It provides both direct implementations and faster specialized routines for common operations over finite fields.

## Current functionality

- **Finite-field vector spaces:** generate the full vector space $\mathbb{F}_p^n$ or stream it in memory-bounded chunks.
- **Finite-field rank:** compute matrix rank over $\mathbb{F}_p$ using modular Gaussian elimination.
- **Sumsets:** compute $S+S$ for a set of vectors over $\mathbb{F}_p^n$, including a direct reference method and a faster FWHT-based implementation.
- **Subspace detection:** find the largest linear subspace contained in a sumset, with both greedy and exhaustive search modes.
- **Hamming-ball geometry:** generate standard Hamming balls, compute Hamming weights, and build balls under arbitrary linear transforms — with or without a pre-built universe array.
- **Cover constructions:** build unions of Hamming balls and compute complements relative to a universe.
- **Low-memory mode:** run all of the above for $n > 20$ without out-of-memory crashes, via a dedicated low-memory API and a `--low-memory` CLI flag.
- **Command-line interface:** run the $k$-balls covering experiment and inspect memory estimates directly from the shell.

## Features

- **NumPy-based operations:** vectorized generation and manipulation of finite-field vectors.
- **Field-aware linear algebra:** rank and invertibility checks performed over $\mathbb{F}_p$, not over the reals.
- **Flexible geometry tools:** support for binary and higher-characteristic experiments.
- **Low-memory implementations:** real float64 FWHT (vs complex128 FFT), in-place transforms, combinatorial ball generation, and boolean-mask complement — all reducing peak RAM by 5–50× for large $n$.
- **CLI entry point:** `ffspaces run` and `ffspaces memory-check` for shell-based experiments.
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
    compute_sumset,
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
- `compute_sumset(set_elements, p=2)`: alias for the fast FWHT-based sumset computation.

### Subspace analysis

- `find_maximum_subspace_dimension(sumset, p=2, exhaustive=False, max_combinations=10000)`: returns the dimension of the largest linear subspace contained in a sumset.

### Hamming geometry and covering

- `generate_standard_ball(n, r)`: returns all binary vectors in $\mathbb{F}_2^n$ with Hamming weight at most `r`.
- `compute_hamming_weight(vectors)`: computes Hamming weights for each row of a matrix.
- `generate_hamming_ball(universe, center, radius, linear_transform=None, p=2)`: builds a Hamming ball in $\mathbb{F}_p^n$; when `linear_transform` is provided distances are measured in the transformed basis, otherwise the standard Hamming metric is used.
- `generate_covering(centers, radii, bases=None, p=2, universe=None)`: returns the union of Hamming balls defined by `centers`, `radii`, and optional basis transforms. For `p != 2` a `universe` array must be supplied.
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

### Low-memory API

For $n > 20$ use the functions in `ffspaces.lowmem` (also exported from the package root). These avoid materialising the full $p^n \times n$ universe array and replace the complex128 FFT with a real float64 in-place FWHT.

```python
from ffspaces import (
    generate_space_chunked,       
    generate_ball_ints_lowmem,      
    generate_covering_ints_lowmem,   
    complement_ints_lowmem,          
    compute_sumset_lowmem,           
    compute_sumset_lowmem_from_ints,
    find_maximum_subspace_dimension_lowmem,
    estimate_memory_gb
)
```

**Check memory requirements before running:**

```python
from ffspaces import estimate_memory_gb

for n in [24, 28, 30]:
    est = estimate_memory_gb(n)
    print(f"n={n}: sumset FWHT {est['compute_sumset_lowmem_GB']:.1f} GB  "
          f"(standard FFT {est['compute_sumset_standard_GB']:.0f} GB)")
# n=24: sumset FWHT 0.2 GB  (standard FFT 1 GB)
# n=28: sumset FWHT 3.2 GB  (standard FFT 17 GB)
# n=30: sumset FWHT 12.9 GB  (standard FFT 69 GB)
```

**End-to-end low-memory workflow for large n:**

```python
import numpy as np
from ffspaces import (
    generate_random_basis,
    generate_covering_ints_lowmem,
    complement_ints_lowmem,
    compute_sumset_lowmem_from_ints,
    find_maximum_subspace_dimension_lowmem,
)

n, k, radius = 25, 3, 1
rng = np.random.default_rng(0)

# Random centers as integer codes — no 2^25 × 25 universe array
center_ints = rng.integers(0, 1 << n, size=k)
from ffspaces import ints_to_vectors
centers = [ints_to_vectors(np.array([ci]), n)[0] for ci in center_ints]
bases   = [generate_random_basis(n, rng=rng) for _ in range(k)]

# Covered set and complement via bool mask (32 MB for n=25)
covered = generate_covering_ints_lowmem(n, centers, radius, bases=bases)
S_ints  = complement_ints_lowmem(n, covered)

# S+S via real float64 FWHT (256 MB for n=25)
ss_ints = compute_sumset_lowmem_from_ints(S_ints, n)

# Subspace dimension via binary-search membership
dim = find_maximum_subspace_dimension_lowmem(ss_ints, n)
print(f"|S|={len(S_ints)}, |S+S|={len(ss_ints)}, max subspace dim={dim}")
```

**Iterate over F_p^n without loading it all:**

```python
from ffspaces import generate_space_chunked

# Process F_2^28 (268 M vectors) 16 384 rows at a time
count = 0
for chunk in generate_space_chunked(28, chunk_size=1 << 14):
    count += len(chunk)   # chunk is a (≤16384, 28) int8 array
print(count)  # 268435456
```

## Command-line interface

After `pip install ffspaces`, the `ffspaces` command is available in your shell.

### `ffspaces run` — $k$-balls covering experiment

Generates $K$ random Hamming balls in $\mathbb{F}_p^n$, computes the complement $S$, and finds the maximum linear-subspace dimension in $S+S$.

```
usage: ffspaces run [--n N] [--k K] [--radius R] [--p P]
                   [--trials T] [--seed S] [--jobs J]
                   [--low-memory] [--exhaustive]
                   [--output FILE] [--no-progress]
```

**Standard mode** (fast; fine for $n \le 20$):

```bash
ffspaces run --n 16 --k 3 --radius 2 --trials 500 --seed 0
```

**Low-memory mode** (required for $n > 20$ on most machines):

```bash
ffspaces run --n 25 --k 3 --radius 1 --trials 200 --low-memory
ffspaces run --n 28 --k 4 --radius 1 --trials 50  --low-memory --jobs 4
ffspaces run --n 25 --k 3 --radius 1 --trials 100 --low-memory --output results.npz
```

Both modes produce **identical results for the same `--seed`**. With `--low-memory` a per-trial memory estimate is printed before the run begins.

Key options:

| Option | Default | Description |
|---|---|---|
| `--n` | 8 | Dimension of $\mathbb{F}_p^n$ |
| `--k` | 3 | Number of Hamming balls |
| `--radius` | 1 | Ball radius |
| `--p` | 2 | Field characteristic |
| `--trials` | 100 | Number of random trials |
| `--seed` | 0 | Master random seed |
| `--jobs` | 1 | Parallel workers |
| `--low-memory` | off | Use memory-efficient path |
| `--exhaustive` | off | Exhaustive subspace search |
| `--output` | — | Save results (`.npz` or `.json`) |

### `ffspaces memory-check` — RAM estimates

Print estimated peak memory for both standard and low-memory modes before committing to a long run.

```bash
ffspaces memory-check --n 24 28 30
```

```
── n=28, p=2 ──
  generate_space (standard)               7.52 GB  ███████████████
  compute_sumset (standard FFT)          17.18 GB  ███████████████████████████████
  compute_sumset (low-memory FWHT)        3.22 GB  ██████  ← low-mem
  complement bool mask (low-mem)          0.27 GB  █       ← low-mem
  S integer array (worst case)            1.07 GB  ██
  sumset integer array (worst case)       2.15 GB  ████
```

## Mathematical background

This package works in the vector space $\mathbb{F}_p^n$, where each coordinate is taken modulo a prime $p$. In particular, $\mathbb{F}_2^n$ is the set of binary vectors of length $n$, with addition done coordinatewise modulo $2$.

- **What is $\mathbb{F}_p^n$?** A finite-dimensional vector space over the finite field with $p$ elements. For $p=2$, this is the space of binary vectors.
- **What is $S+S$?** For a set $S \subseteq \mathbb{F}_p^n$, the sumset $S+S$ is the set of all pairwise sums $x+y$ with $x,y \in S$, taken modulo $p$.
- **What is a Hamming ball?** A Hamming ball is the set of vectors within a given distance from a center, where distance is the number of coordinates in which two vectors differ (known as Hamming weight, or the Hamming metric).
- **What is the covering problem?** A covering problem asks whether a collection of balls (or other subsets) can be chosen so that their union contains all points in a given universe, or perhaps all points of a particular structure.
- **Why does this matter?** These objects are central in additive combinatorics, coding theory, and discrete geometry, and they provide a natural setting for studying how small sets expand under addition or how geometric configurations cover a space.


## ⚠️ Performance & scale

Memory consumption scales exponentially with $n$: the universe has $p^n$ vectors, and most dense operations allocate arrays of that size.

| Range | Standard mode | Low-memory mode |
|---|---|---|
| $n \le 16$, $p=2$ | Near-instant, <1 GB | — |
| $16 < n \le 22$, $p=2$ | Gigabytes of RAM, may be slow | Not required |
| $22 < n \le 28$, $p=2$ | Likely OOM | Feasible with ≥4–8 GB RAM |
| $28 < n \le 32$, $p=2$ | OOM on most machines | Needs 16–64 GB RAM |
| $n > 32$, $p=2$ | Infeasible | Multi-day runtime; server class |

The **low-memory mode** reduces peak RAM by 5–50× relative to the standard path by:
- Replacing the complex128 FFT with a real float64 in-place FWHT for sumset computation.
- Generating Hamming balls combinatorially without loading the full universe.
- Using a 1-byte boolean mask for complement instead of an 8-byte int64 array.
- Using sorted-array binary search for subspace-dimension membership instead of Python tuple sets.

Run `ffspaces memory-check --n <N>` to see per-operation estimates before starting a long job.

## Experiments

One-off analysis and experiment scripts live in the `experiments/` directory so they stay separate from the installable package code in `src/`.
