# Advanced Option Pricing: Fourier Methods and Variance Reduction

This subfolder contains Python implementations of advanced numerical techniques in computational finance. The first section focuses on pricing European options and recovering probability densities using Fourier-based methods (inspired by the COS method). The second section explores Monte Carlo simulations for estimating tail risk (95th percentiles) of basket options, utilizing Importance Sampling to drastically reduce variance.

## Project Structure & Files

### Part 1: Fourier Series Methods (Section 2)
These scripts utilize characteristic functions to recover probability densities, cumulative distribution functions (CDFs), and option prices.

* **`2b_fourier_density_recovery.py`** *(Original: 2bCF.py)*
  * **Goal:** Recovers the probability density function of a Normal distribution using a Fourier-sine series expansion.
  * **Technique:** Evaluates the characteristic function and constructs the density over a truncated domain, comparing the numerical approximation against the exact analytical density.

* **`2d_cos_method_put_pricing.py`** *(Original: 2dCF.py)*
  * **Goal:** Prices a European Put option under the Black-Scholes framework using a Fourier-based approach.
  * **Technique:** Implements the Fourier-sine coefficients to numerically evaluate the risk-neutral pricing formula, comparing the result against the analytical Black-Scholes put price.

* **`2e_fourier_cdf_computation.py`** *(Original: 2eCF.py)*
  * **Goal:** Computes the Cumulative Distribution Function (CDF) of the log-return distribution.
  * **Technique:** Uses the characteristic function to approximate the CDF via a truncated Fourier series, mapping convergence across different expansion terms ($N$).

### Part 2: Monte Carlo & Importance Sampling (Section 3)
These scripts estimate the 95th percentile of a multi-asset payoff (tail risk) and demonstrate how Importance Sampling accelerates convergence for rare events.

* **`3a_mc_plain_percentile.py`** *(Original: 3aCF.py)*
  * **Goal:** Computes the 95th percentile of a terminal payoff using standard (plain) Monte Carlo simulation.
  * **Technique:** Simulates thousands of correlated paths for a large basket of stocks to find the empirical quantile. Serves as the baseline for variance reduction comparisons.

* **`3b_is_mean_shift.py`** *(Original: 3bCF.py)*
  * **Goal:** Applies Importance Sampling by applying a **mean shift** to the underlying distribution.
  * **Technique:** Shifts the sampling distribution closer to the 95th percentile region to sample the "tail" more frequently. Adjusts the final estimates using the likelihood ratio (Radon-Nikodym derivative) to maintain unbiasedness.

* **`3c_is_mean_variance_shift.py`** *(Original: 3cCF.py)*
  * **Goal:** Applies Importance Sampling using both a **mean and variance shift**.
  * **Technique:** Flattens the distribution by increasing the variance of the common factor alongside the mean shift. Analyzes why over-dispersing the distribution can actually harm the efficiency of the quantile estimation compared to a pure mean shift.

### Documentation
* **`Fourier_and_IS_Report.pdf`** *(Original: Computational_Finance_Assignment_1.pdf)*
  * The full mathematical report containing the derivations of the characteristic functions, the proofs for the Importance Sampling likelihood ratios, and visual plots of the numerical convergence.

## Technology Stack
* **Python 3**
* **NumPy:** Heavily used for vectorizing large-scale Monte Carlo paths and computing complex numbers in Fourier series.
* **SciPy:** Used for baseline Black-Scholes analytics and standard normal distributions.
* **Matplotlib:** For plotting density recovery and standard error convergence.

## How to Run
Ensure all files are in the same directory.
1. Install dependencies: `pip install numpy scipy matplotlib`
2. Run any script directly from the terminal. For example:
   `python 2d_cos_method_put_pricing.py`
