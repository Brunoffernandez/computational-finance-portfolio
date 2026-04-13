# Quantitative Finance: Volatility Extraction and Monte Carlo Simulation

This repository contains Python implementations of numerical methods used in computational finance. The project focuses on implied volatility extraction using root-finding algorithms and pricing multi-asset options using Monte Carlo simulations with variance reduction techniques.

## Files in this Repository

### `implied_volatility_solver.py`
This script extracts implied volatility from real-world market data by reversing the Black-Scholes formula.
* **Data:** Processes S&P 500 (SPX) option chains from `spx_options_data.csv`.
* **Algorithms:** Implements the **Bisection Method** and the **Newton-Raphson Method** to solve the non-linear Black-Scholes equation for $\sigma$.
* **Analysis:** Calculates the "Vega" of the options and plots the implied volatility "smile"/curve across different strike prices to compare the convergence and accuracy of the numerical methods.

### `basket_option_monte_carlo.py`
This script prices a Basket Call Option (an option dependent on two underlying assets) using Monte Carlo simulations.
* **Techniques:** Uses Cholesky decomposition to simulate correlated Geometric Brownian Motions (GBM) for the two underlying assets.
* **Variance Reduction:** Computes prices under the standard risk-neutral measure (Q) and compares it with a change of numeraire measure (QS1).
* **Analysis:** Simulates varying path lengths ($10,000$ to $100,000$ paths) and plots the standard error decay. It verifies the Central Limit Theorem by mapping the $1/\sqrt{M}$ convergence rate.

### `spx_options_data.csv`
The raw dataset containing S&P 500 option chain data (strike prices, midpoints, quoted implied volatilities) used by the volatility solver.

### `Option_Pricing_and_Simulation_Report.pdf`
A comprehensive technical report detailing the mathematical derivations, theoretical frameworks, and analysis of the results produced by the Python scripts.

## Requirements
To run the scripts, you will need Python 3 and the following scientific libraries:
* `numpy`
* `scipy`
* `pandas`
* `matplotlib`

## How to Run
Ensure all files are in the same directory.
1. Install dependencies: `pip install numpy scipy pandas matplotlib`
2. Run the implied volatility solver: `python implied_volatility_solver.py`
3. Run the Monte Carlo simulation: `python basket_option_monte_carlo.py`
