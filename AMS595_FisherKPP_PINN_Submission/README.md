# AMS 595 Final Project
## Inverse Fisher–KPP Equation using Physics-Informed Neural Networks

This repository contains the source code for the AMS 595 course project
on solving an inverse problem for the Fisher–KPP equation using
Physics-Informed Neural Networks (PINNs).

## Framework
The implementation is built on top of the **pinnstorch** framework,
a mature and well-tested PyTorch-based library for PINNs.
The framework provides automatic differentiation, neural network
modules, and training utilities.

This project **does not reimplement the PINN framework itself**.
Instead, it focuses on extending the framework to a **non-trivial
inverse PDE setting**, including:
- Learning unknown PDE parameters (diffusion coefficient D and reaction rate r)
- Enforcing boundary and initial conditions through physics-based loss terms
- Handling sparse and noisy observation data

## Governing Equation
The Fisher–KPP equation is given by
\[
u_t = D u_{xx} + r u (1 - u).
\]

## Boundary and Initial Conditions
The problem setup matches the forward solver used to generate data:
- Dirichlet boundary conditions:  
  \(u(0,t) = u(10,t) = 0\)
- Initial condition:  
  \(u(x,0) = \exp(-(x - 5)^2)\)

## Files
- `FisherKPP_inverse.ipynb`  
  Main notebook implementing the inverse PINN.
- `data/kpp_training_data.csv`  
  Training data generated from a finite-difference forward solver.

## How to Run
Open `FisherKPP_inverse.ipynb` and execute all cells sequentially.
