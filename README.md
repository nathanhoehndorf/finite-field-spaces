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

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/nathanhoehndorf/finite-field-spaces
cd finite-field-spaces
pip install -e .
```

The package requires Python 3.10+ and NumPy.

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
