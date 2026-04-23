# QuantumBilliards.jl

A Julia library for computing eigenvalues, eigenfunctions and Husimi functions of 2D quantum billiards using boundary integral and basis-expansion methods.

## Overview

`QuantumBilliards.jl` targets high-frequency spectral computations on smooth and piecewise-smooth domains. It combines:

- Boundary Integral Methods (DLP / CFIE / Alpert)
    1. Boundary Integral Equations in time-harmonic acoustic scattering, Kress R., 1991
    2. (Habilitationsschrift) Eigenfunctions in chaotic quantum systems, Backer A., 2007
    3. HYBRID GAUSS-TRAPEZOIDAL QUADRATURE RULES, Alpert B., 1999
- Local accelerated solvers (EBIM, Vergini–Saraceno)
    1. Calculation by scaling of highly excited states of billiards, Vergini E., Saraceno M. 1995
    2. (PhD thesis) https://users.flatironinstitute.org/~ahb/thesis_html/node71.html, Barnett A.
- Contour methods (Beyn)
    1. An integral method for solving nonlinear eigenvalue problems, Wolf-Jürgen Beyn, 2012
- Chebyshev-accelerated kernel assembly
    1. Greengard's hank106.f code - panelization implementation

Focus is on the balance between **performance** and **spectral resolution**, with a high-level API and low-level optimizations.

---

## Methods

### Boundary Integral Methods (BIM / DLP / CFIE)

The Helmholtz problem

\[
(\Delta + k^2)\psi = 0, \quad \psi|_{\partial\Omega}=0
\]

is reduced to

\[
A(k)\sigma = 0.
\]

CFIE variants use

\[
(\alpha I + D(k) + i\eta S(k))\sigma = 0,
\]

typically in the presence of holes.

**+** Accurate, flexible, stable (CFIE)  
**−** Dense matrices, expensive assembly  

---

### EBIM (Expanded Boundary Integral Method)

Local expansion:

\[
A(k+\varepsilon) \approx A + \varepsilon A' + \tfrac12 \varepsilon^2 A''.
\]

Solve

\[
A v = \lambda A' v, \quad \lambda \approx -\varepsilon,
\]

with second-order corrections.

**+** Efficient for sweeps, works with Krylov  
**−** Finicky to deal with, requires checking interval size to not lose eigenvalues  

---

### Vergini–Saraceno (Scaling Method)

Basis expansion near \(k_0\):

\[
F c = \mu G c.
\]

**+** Extremely fast, gets many nearby levels per solve
**−** Basis-dependent, less robust for complex (non-convex) geometries  

---

### Beyn Method

Contour-based extraction:

\[
M_p = \frac{1}{2\pi i} \oint_\Gamma z^p T(z)^{-1} V \, dz.
\]

**+** Finds all eigenvalues in a region  
**+** Very effective when Vergini–Saraceno fails  
**+** Naturally supports desymmetrized domains via subspace projections
**−** Slower than Vergini–Saraceno, as for desymmetirzed domains still needs to form and invert the full size matrix 
**−** Requires contour tuning (nq, but it varies slowly and is typically very small around 40-45) 

---

### Chebyshev Acceleration

Kernel approximation:

\[
f(x) \approx \sum a_n T_n(x).
\]

**+** Faster repeated assembly, prioriziting smaller panels with lower degrees
**−** Slightly higher RAM usage due to panelization  

---

## Geometry

Supports many billiards:
- circle, ellipse, stadium, mushroom, Robnik, star, multi-hole domains
- symmetry-reduced and composite geometries

---

## Practical Guidance

- **Many levels in a small window and large-scale computation →** Vergini–Saraceno  
- **Smaller number of eigenvalues →** EBIM (Krylov for large systems)  
- **Large-scale / difficult spectra →** Beyn  
- **Assembly dominates →** Chebyshev acceleration  

In practice:
- VS is fastest when it applies.
- Beyn is the most robust fallback and works on desymmetrized domains.
- EBIM is best for systematic checking in medium sized intervals.

---

## Symmetries

Symmetry reduction reduces problem size and improves performance, but:

- the spectrum splits into symmetry sectors -> removing degeneracies.
- level counting must match the reduced domain -> only get parts of the full spectrum for that irrep.

Some solvers (e.g. Kress, Alpert) assume periodic parametrizations and do not directly support desymmetrization. Beyn provides a natural workaround in these cases, but still requires full matrix assembly and inversion (lu!).

**Use symmetry whenever possible for improved performance and removal of degeneracies**

---

## Typical Workflow

1. Define billiard -> billiard, basis = make_billiard_and_basis(...)
2. Choose solver -> geometry and problem dependant
3. Discretize boundary -> evaluate_points(...) (done internally)
4. Compute spectrum  -> compute_spectrum_beyn(...), compute_spectrum_ebim(...), compute_spectrum_with_state_data(...)

---

## Status

Active development focused on:
- integration into the QuantumChaosJulia ecosystem
    1. https://github.com/Quantum-Chaos-Julia/BilliardGeometry.jl
- neutrino billiards
    1. Relativistic Quantum Chaos in Neutrino Billiards, Dietz B. (https://arxiv.org/pdf/2604.13003)
- Hyperbolic kernel - Legendre Q via mpmath and seeding w/ Taylor series center expansion (Done, currently used in a paper, add after publication)  
- QBX,QB2X
    1. Quadrature by Expansion: A New Method for the Evaluation of Layer Potentials, Kl¨ockner A., Barnett A., Greendard L., O'Neil M. (https://arxiv.org/pdf/1207.4461)
    2. Quadrature by Two Expansions for Evaluating
    elmholtz Layer Potentials, Weed J., Ding. L,... (https://arxiv.org/pdf/2207.13762)
- OTOC, wavepacket dynamics...