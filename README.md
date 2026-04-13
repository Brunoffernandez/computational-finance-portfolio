# Computational Finance & Quantitative Modeling Portfolio

Welcome to my computational finance portfolio. This repository contains Python implementations of advanced numerical methods, stochastic simulations, and pricing algorithms used in modern quantitative finance. 

The projects within showcase my ability to translate complex mathematical frameworks—such as stochastic calculus, Fourier transforms, and variance reduction techniques—into functional, optimized, and well-documented code.

## Portfolio Structure

This repository is organized into distinct project folders, each containing its own source code, datasets, and detailed mathematical reports.

### [1. Volatility Extraction and Monte Carlo Simulation](./01_option_pricing_models/)
This project focuses on extracting market expectations from option chains and pricing multi-asset derivatives.
* **Implied Volatility Solver:** Reverses the Black-Scholes formula using Bisection and Newton-Raphson root-finding algorithms to extract implied volatility from S&P 500 options data.
* **Basket Option Pricing:** Uses Monte Carlo simulations with Cholesky decomposition to price correlated multi-asset options.
* **Variance Reduction:** Demonstrates a Change of Numeraire technique to significantly reduce standard error compared to standard risk-neutral measures.

### [2. Advanced Pricing: Fourier Methods and Importance Sampling](./02_fourier_and_importance_sampling/)
This project implements highly advanced numerical techniques for probability density recovery and extreme tail-risk estimation.
* **Fourier-Cosine (COS) Method:** Utilizes characteristic functions and Fourier series expansions to recover probability densities, compute CDFs, and accurately price European options.
* **Tail Risk Estimation:** Estimates the 95th percentile of a multi-asset payoff distribution for a large basket of stocks.
* **Importance Sampling:** Applies mean and variance shifts to the underlying sampling distribution (using Radon-Nikodym derivatives) to drastically accelerate convergence and reduce variance for rare-event tail risks.

## Core Skills & Technology Stack

**Programming & Data Science:**
* **Python:** Core language for all implementations.
* **NumPy & SciPy:** For high-performance vectorized operations, matrix decompositions, and complex-number mathematics.
* **Pandas:** For financial dataset ingestion and manipulation.
* **Matplotlib:** For visual analysis of convergence rates, volatility smiles, and probability densities.

**Quantitative & Mathematical Skills:**
* Option Pricing (Black-Scholes, Monte Carlo, Fourier Methods)
* Variance Reduction (Importance Sampling, Change of Numeraire)
* Numerical Root-Finding (Newton-Raphson, Bisection)
* Stochastic Processes (Correlated Geometric Brownian Motion)
