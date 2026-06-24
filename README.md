# Finite-Field Spaces (`ffspaces`)

An open-source computational discrete mathematics laboratory engineered to investigate partitioning questions, sumset structures, and arithmetic configurations over finite field spaces $\mathbb{F}_p^n$.

This suite is explicitly optimized to test multi-ball configurations, asymmetric radius boundaries, and higher-characteristic analogues related to problems in additive combinatorics (such as Ben Green's 100 Open Problems and recent follow-ups on Hamming Ball coverings).

## Features
- **Vectorized Generation:** Quick instantiation of $\mathbb{F}_p^n$ vector universes using NumPy matrix multiplication.
- **Basis Arbitrage:** Generate random invertible maps over $\mathbb{F}_p^n$ to evaluate configurations under varying geometric orientations.
- **Sumset Structural Trackers:** Tools to compute $S+S$ and identify embedded maximum linear subspaces or calculate codimensions.

## Installation
Clone this repository locally and install it in editable mode:
```bash
git clone https://github.com/nathanhoehndorf/finite-field-spaces
cd finite-field-spaces
pip install -e
```
