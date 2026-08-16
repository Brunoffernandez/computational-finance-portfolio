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

### [3. Multi-Asset Basket Option Pricing: Monte Carlo, Moment Matching and 2D COS](./03_multi_asset_pricing_methods/)
This project prices a European basket call written on two underlyings — one log-normal (GBM) and, in the second setup, one square-root (CIR-type) diffusion — from three complementary angles.
* **Monte Carlo with Euler–Maruyama:** Exact GBM simulation for the log-normal setup and a **full-truncation Euler–Maruyama** scheme for the CIR leg (needed because $\rho \neq 0$ blocks exact joint simulation). Isolates and fits the weak-convergence order of the scheme and the statistical $\mathcal{O}(1/\sqrt{M})$ decay.
* **Moment-matching approximations:** Fits log-normal (2 moments), shifted log-normal (3 moments) and **Johnson SU** (4 moments) distributions to the basket by matching closed-form basket moments (cross log-normal / non-central $\chi^2_0$), pricing analytically or by Gauss–Hermite quadrature.
* **Two-dimensional COS method** (Ruijter & Oosterlee, 2012): Expands the joint density in a 2D cosine series using marginal characteristic functions, computes the non-separable payoff coefficients via a trapezoidal $C_1^{\top} P\, C_2$ product, and observes spectral convergence in the number of cosine terms $N$.

### [4. American Option Pricing: Least-Squares Monte Carlo and Finite Differences](./04_american_options_lsmc_and_fd/)
This project prices American put options with two complementary numerical methods and cross-validates them on a 20-configuration parameter grid $(S_0, \sigma, T)$.
* **Longstaff–Schwartz LSMC:** Backward recursion of the optimal-stopping problem with cross-sectional regression of the continuation value on the first three **Laguerre polynomials** (as in Longstaff & Schwartz 2001). Includes a step-by-step reproduction of the paper's 8-path worked example and a full parameter sweep with 100k paths and antithetic variates.
* **Finite Differences (FTCS):** Explicit forward-Euler scheme on a $\log$-price grid for the Black–Scholes PDE. The European put is validated against the closed-form price; the American extension imposes early exercise via $U \leftarrow \max(U, K - S)$ at every time slice.
* **Cross-validation:** LSMC and FTCS American prices are placed side by side and their absolute gap is reported, together with the early-exercise premium $V_A - V_E$ from both methods.

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
