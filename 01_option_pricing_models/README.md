# Quantitative Finance & Option Pricing Models

This repository contains Python implementations of numerical methods and simulation techniques used in computational finance. The projects focus on option pricing, volatility surface extraction, and variance reduction techniques.

## Projects Included

### 1. Implied Volatility Solver (`implied_volatility_solver.py`)
Extracts the implied volatility from market data (SPX options) by reversing the Black-Scholes formula. 
* **Data:** Processes a real-world CSV dataset of SPX option chains.
* **Algorithms Used:** Implements both the **Bisection Method** and the **Newton-Raphson Method** to find the roots of the non-linear Black-Scholes equation.
* **Analysis:** Compares the convergence and accuracy of both methods against market-quoted implied volatility, outputting visual comparisons using Matplotlib.

### 2. Basket Option Monte Carlo Simulation (`basket_option_monte_carlo.py`)
Prices a Basket Call Option using Monte Carlo simulations under different probability measures to demonstrate variance reduction.
* **Techniques:** Uses Cholesky decomposition to simulate correlated Geometric Brownian Motions (GBM) for multiple underlying assets.
* **Variance Reduction:** Compares the standard risk-neutral measure (Q) with a change of numeraire measure (QS1).
* **Analysis:** Evaluates the standard error decay rates across various simulation path lengths ($10,000$ to $100,000$ paths), visually confirming the Central Limit Theorem convergence.

## Technology Stack
* **Python 3**
* **NumPy & SciPy:** For statistical distributions, matrix operations, and vectorized math.
* **Pandas:** For financial data ingestion and cleaning.
* **Matplotlib:** For plotting convergence rates and volatility curves.

## Documentation
Detailed mathematical derivations and theoretical explanations for these implementations can be found in the `docs/` directory.

## How to Run
1. Clone the repository.
2. Ensure you have the required libraries installed: `pip install numpy scipy pandas matplotlib`.
3. Run the scripts directly from the `src/` directory.
