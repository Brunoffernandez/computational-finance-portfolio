# American Option Pricing: Least-Squares Monte Carlo and Finite Differences

This subfolder implements two complementary numerical methods for pricing **American put options** (and their European counterparts as sanity checks):

1. **Least-Squares Monte Carlo (LSMC)** — the Longstaff–Schwartz (2001) algorithm, first walked through step-by-step on the paper's 8-path worked example and then applied to a full 20-configuration parameter grid $(S_0, \sigma, T)$ using Laguerre-polynomial regressors, antithetic variates and 100k simulated paths per case.
2. **Finite Differences (FTCS)** — a forward-time / central-space explicit scheme on a $\log$-price grid for both the European put (validated against Black–Scholes) and the American put (early-exercise imposed by projection onto the intrinsic value at every time step).

Both methods are then **cross-checked against each other** on the same 20-configuration grid, and the early-exercise premium is reported.

## Files in this Repository

### `ex1_longstaff_schwartz_lsmc.py`
Longstaff–Schwartz LSMC pricing of American puts.
* **Worked example (paper Table 1):** Reproduces the Longstaff–Schwartz (2001) toy example with 8 paths, $K = 1.10$, $r = 0.06$, $T = 3$ using a degree-2 polynomial basis. Prints the cash-flow matrix at each backward step, the regression table, the exercise-vs-continuation decision, the stopping rule and finally both the American and European put prices.
* **Parameter sweep (Table 1(ii)):** Prices 20 combinations of $(S_0 \in \{36,38,40,42,44\}, \sigma \in \{0.20, 0.40\}, T \in \{1, 2\})$ with $K = 40$, $r = 0.06$, $M = 50$ exercise dates per year, $N = 100{,}000$ paths (50k standard + 50k antithetic), regressing continuation values on the **first three Laguerre polynomials** normalised by the strike (as in the paper).
* **Reporting:** Simulated American price, standard error, closed-form European (Black–Scholes) benchmark, and early-exercise value $V_A - V_E$ for every configuration.

### `ex2_finite_differences.py`
FTCS finite-difference pricing of European and American puts, and cross-check against LSMC.
* **European put (FTCS):** Explicit forward-Euler scheme on a log-price grid centred at $\log K$ with $N_x = 1000$ space steps and $N_t = 50 T$ time steps. The Dirichlet boundary at the lower end contributes the discounted-strike source term $p_0$; the upper boundary is 0 (deep OTM). Interpolates back to $\log S_0$ and compares against the closed-form Black–Scholes put.
* **American put (FTCS with early exercise):** Same explicit scheme, plus a **projection step** $U \leftarrow \max(U, \text{intrinsic})$ at every time slice, giving the American continuation value. Reports the early-exercise premium against the European Black–Scholes benchmark.
* **Cross-validation (FD vs LSMC):** Reruns the LSMC pipeline of Exercise 1 on the same 20-configuration grid and produces a side-by-side comparison table `|FD American − LSM American|`, plus a summary table with the FD early-exercise value and the FD–LSMC gap on that quantity.

### `Exercise_II_Statement.pdf`
Original assignment statement listing the American-put contract specification, the numerical schemes to implement, and the 20-configuration parameter grid.

### `American_Options_Report.pdf`
Full written report with mathematical derivations (Longstaff–Schwartz recursion, FTCS stencil for the Black–Scholes PDE, boundary conditions, stability considerations) and the interpretation of the numerical tables produced by the scripts.

## Technology Stack
* **Python 3**
* **NumPy:** vectorised Monte Carlo paths, tridiagonal FTCS stencils, backward recursion on cash-flow matrices.
* **Pandas:** formatted per-step regression / stopping-rule / cash-flow tables.
* **SciPy:** `scipy.stats.norm` for the Black–Scholes benchmark.
* **scikit-learn:** `LinearRegression` and `PolynomialFeatures` / `make_pipeline` for the Longstaff–Schwartz cross-sectional regressions.

## How to Run
Ensure all files are in the same directory.
1. Install dependencies: `pip install numpy scipy pandas scikit-learn`
2. Run either script directly from the terminal:
   * `python ex1_longstaff_schwartz_lsmc.py`
   * `python ex2_finite_differences.py`

Both scripts print their numerical tables to stdout — no figures are produced or saved to disk, so the project is fully portable across machines.
