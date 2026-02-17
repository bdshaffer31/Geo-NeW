![Geo-NeW logo](images/geo_new_logo.png)
General Geometry Neural Whitney Forms (Geo-New) for PDEs on complex geometries

Code accompanying ICML 2026 Submission **Structure-Preserving Learning Improves Geometry Generalization in Neural PDEs**.

## Abstract:
We aim to develop *physics foundation models* for science and engineering that provide real-time solutions to Partial Differential Equations (PDEs) which preserve structure and accuracy under adaptation to unseen geometries. To this end, we introduce *General-Geometry Neural Whitney Forms* (Geo-NeW): a data-driven finite element method. We jointly learn a differential operator and compatible reduced finite element spaces defined on the underlying geometry. The resulting model is solved to generate predictions, while exactly preserving physical conservation laws through Finite Element Exterior Calculus. Geometry enters the model as a discretized mesh both through a transformer-based encoding and as the basis for the learned finite element spaces. This explicitly connects the underlying geometry and imposed boundary conditions to the solution, providing a powerful inductive bias for learning neural PDEs, which we demonstrate improves generalization to unseen domains. We provide a novel parameterization of the constitutive model ensuring the existence and uniqueness of the solution. Our approach demonstrates state-of-the-art performance on several steady-state PDE benchmarks, and provides a significant improvement over conventional baselines on out-of-distribution geometries.

This repository provides a minimal implementation of the core methodology described in the paper, it will be udpated with a training and evaluations examples shortly.

## What’s in this repo

This is a small, self-contained implementation of the Geo-NeW forward model:

- A **geometry encoder** that maps per-node input features to latent tokens.
- Learned **coarse / reduced spaces** via a partition-of-unity weight operator $W$ (with boundary handling).
- A learned **constitutive / flux model** operating on the coarse latent fields.
- A differentiable **batched Newton solver** (implicit function theorem in the backward pass) to solve the nonlinear residual.

The goal of this repo is to be a readable working version. The current `demo.py` runs a smoke test that exercises the full forward pass and nonlinear solve; a simple train/eval loop will be added next.

## Repository layout

- `geo_new.py`: Geo-NeW wrapper module. Defines the nonlinear residual and the forward pass.
- `nonlinear_solver.py`: differentiable batched Newton solver used by Geo-NeW.
- `models.py`: encoder / $W$ / flux / source models used by the demo.
- `utils.py`: small FEEC / masking / projection utilities (plus some metrics helpers).
- `demo.py`: minimal entrypoint for instantiation and a forward smoke test.

## Dependencies

Tested with:

- Python 3.10+
- PyTorch 2.0+ (uses `torch.func.vmap` / `torch.func.jacrev`)
- NumPy
- SciPy
- Scikit-FEM

Minimal install (CPU):

```bash
pip install torch numpy scipy skfem
```

For GPU / CUDA PyTorch, install PyTorch from the official selector for your platform.

## Quickstart

Run the included demonstration:

```bash
python demo.py
```
