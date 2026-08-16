# QuantumBilliards

[![Build Status](https://github.com/clozej/QuantumBilliards.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/clozej/QuantumBilliards.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://Quantum-Chaos-Julia.github.io/QuantumBilliards.jl/)

Welcome to the QuantumBilliards.jl - a high performance Julia library for computing the states and spectra of quantum billiards. The goal is to provide a library that is a learning tool for students as well as full computational suite for seasoned quantum chaos researchers that want to attain that last bit of performance for their research publications. We hope this tool will invigorate the study of quantum billiards and related problems and make it more accessible to a wider research community. The documentation is available [here](https://Quantum-Chaos-Julia.github.io/QuantumBilliards.jl/).

## Introduction
Quantum billiards are fundamental models, widely studied in the context of quantum chaos, quantum classical correspondence and semiclassical analysis. 
The package contains a collection of the most effective solution methods for solving quantum billiard eigenvalue problems with state of the art performance and an intuitive API.
In the canonical example we are interested in the quantum equivalent of the mathematical billiard problem where a particle moves freely on the 2D billiard table of a given shape $\mathcal{B} \subset \mathbb{R}^2 $. It is specularly reflected when it hits the boundary of the table. In the quantum case we are interested in the dynamics of a quantum particle in an equivalent setting. We are solving the Schrödinger equation in a 2D infinite potential well of the same shape. The potential can be regarded as $V = 0$ inside the table and $$V = \infty$$ outside.
To compute the spectrum and eigenstates we must solve the stationary problem, that is the Helmholtz equation

$\left(\nabla^{2}+k_n^{2}\right)\psi_n(x)=0,$

with the boundary condition $\psi_n|_{\partial \mathcal{B}}=0$. Thus we find the eigenenergies $E_n=k_n^2$, where $k_n$ is the wavenumber of the n-th eigenstate.

The library implements extremely efficient numerical methods for computing the spectra (wavenumbers) and states (represented as wavefunctions) allowing us to compute up to a millions of eigenstates!
Each method has its own strengths and limitations and is more suitable for specific use cases for instance either sacrificing some accuracy for computation speed or vice versa. 

# Package features
- General purpose solvers that can handle arbitrary billiard tables - seamless integration with BilliardGeomery.jl
- State of the art billiard solvers featuring all the most efficient solution methods like VerginiSaraceno...
- Eigenstates, wavefunctions, boundary functions, and Husimi (phase-space) representations.
- Highly tunable numerical parameters, integration schemes and basis constructions for fine tuning to specific geometries.
- Discrete symmetry reduction schemes, allows separation fo spectral symmetry sectors and improved computation efficiency.  
- Extensible API ready for future updates to related problems like microwave resonators, neutrino billiards etc.