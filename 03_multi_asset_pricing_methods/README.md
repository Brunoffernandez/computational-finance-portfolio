# Multi-Asset Basket Option Pricing: Monte Carlo, Moment Matching and 2D COS

This subfolder tackles the pricing of a European **basket call option** written on two underlyings from three complementary angles: (1) Monte Carlo simulation with exact GBM and Euler–Maruyama for a CIR process, (2) analytic approximations built by matching the first moments of the basket distribution, and (3) the two-dimensional Fourier-cosine (COS) method of Ruijter & Oosterlee (2012). All three approaches price the same contract:

$$
V(0) = e^{-rT}\, \mathbb{E}^{\mathbb{Q}}\!\left[\max\!\big(w_1 S_1(T) + w_2 S_2(T) - K,\ 0\big)\right],
$$

with weights $w_1 = w_2 = 0.5$, strike $K = 100$, maturity $T = 1$ and risk-free rate $r = 2\%$.

Two market setups are studied throughout:
* **(a)** both underlyings are log-normal (GBM) with correlation $\rho = 0.30$ (or $\rho = 0.10$ for Q1(b) initial test)
* **(b)** $S_1$ is log-normal and $S_2$ follows a square-root (CIR-type) diffusion, $dS_2 = r S_2\, dt + \sigma_2 \sqrt{S_2}\, dW_2$

## Files in this Repository

### `q1_basket_monte_carlo.py`
Monte Carlo pricing of the basket call under both setups.
* **Setup (a) – exact GBM:** Simulates correlated terminal prices via Cholesky decomposition of two standard normals. Verifies the $\mathcal{O}(1/\sqrt{M})$ standard-error decay by log-log regression of SE against the number of paths.
* **Setup (b) – GBM + CIR with $\rho = 0.10$:** Uses an **Euler–Maruyama** scheme with **full truncation** for $S_2$ (exact joint simulation is not available when $\rho \neq 0$). Reports the price with a 95% confidence interval.
* **Weak convergence test:** Reuses the *same* fine Brownian increments across all coarser grids to isolate the discretization bias from Monte Carlo noise, then fits the weak order of the Euler scheme (expected $\approx 1$).
* **Statistical convergence test:** Fixes the time grid and varies $M$ to verify the $\mathcal{O}(1/\sqrt{M})$ MC-error decay.

### `q2_moment_matching.py`
Approximates the basket distribution by matching its first few raw moments to a parametric family, and prices the call analytically.
* **2 moments – Log-normal:** Fits a log-normal by matching the forward and the second moment; prices via Black's formula.
* **3 moments – Shifted log-normal:** Matches mean, variance and skewness with $B = \gamma + L$, $L\sim\mathrm{LN}$; inverts the log-normal skewness formula with `brentq` and prices via a shifted-strike log-normal call.
* **4 moments – Johnson SU:** Matches mean, variance, skewness and kurtosis using $B = \xi + \lambda \sinh((Z-\gamma)/\delta)$, $Z\sim N(0,1)$; solves the shape parameters with `least_squares` on Gauss–Hermite moments and prices by Gauss–Hermite quadrature.
* **Exact basket moments:** Derived in closed form: cross log-normal moments for setup (a) and non-central $\chi^2_0$ moments (from the assignment's technical hint $S_2(T) = c Z$, $Z \sim \chi'^2_0(\lambda)$) for setup (b).
* **High-precision MC benchmark:** Antithetic Monte Carlo with 4M paths using exact simulation is used as the reference against which the moment-matching errors are measured.

### `q3_basket_cos_2d.py`
Two-dimensional COS-method pricer following Ruijter & Oosterlee (2012).
* **Structure:** Expands the joint density of $(y_1, y_2)$ in a 2D cosine series on $[a_1,b_1]\times[a_2,b_2]$. When $\rho = 0$ the density coefficients factorise into a product of one-dimensional coefficients $\chi_k = \mathrm{Re}\{\phi(\omega_k) e^{-i\omega_k a}\}$ recovered from the marginal characteristic functions.
* **Payoff coefficients:** Since the basket payoff is non-separable, the coefficients $V_{k_1,k_2}$ are computed by a dense trapezoidal rule written as a matrix product $C_1^{\top} P\, C_2$.
* **Truncation:** Uses the COS-paper cumulant rule $[a,b] = \kappa_1 \pm L\sqrt{\kappa_2 + \sqrt{\kappa_4}}$ with $L = 12$. For setup (b) the lower endpoint of the $S_2$-domain is clipped at $0$ (non-negativity of the CIR level).
* **Characteristic functions:** Normal log-price for the GBM legs; the assignment's hint gives $\phi_{S_2(T)}(\omega) = \exp\!\big(\lambda\, i\omega c / (1 - 2i\omega c)\big)$ for the square-root process.
* **Convergence:** Prints a table of $|V_N - V_{\mathrm{ref}}|$ versus $N$ and observes the exponential (spectral) convergence characteristic of the COS method for smooth densities.

### `figures/`
Empty by default (contains a `.gitkeep`); the scripts write their PNG convergence plots here at run-time via a script-relative path, so the project is portable across machines — no absolute paths hard-coded.

### `Assignment_II_Basket_Options.pdf`
The original assignment statement with the mathematical specification of the problem, the technical hint for the CIR marginal, and the questions each script addresses.

## Technology Stack
* **Python 3**
* **NumPy:** vectorised Monte Carlo paths, cosine-basis matrix assembly, cumulant computations.
* **SciPy:** `scipy.stats.norm` for Black-style CDFs, `scipy.optimize.brentq` and `least_squares` for the moment-matching root-finding, `numpy.polynomial.hermite_e.hermegauss` for Gauss–Hermite nodes used by the Johnson SU pricer.
* **Matplotlib:** log-log and semi-log convergence plots.

## How to Run
Ensure all files are in the same directory.
1. Install dependencies: `pip install numpy scipy matplotlib`
2. Run any script directly from the terminal. For example:
   * `python q1_basket_monte_carlo.py`
   * `python q2_moment_matching.py`
   * `python q3_basket_cos_2d.py`

Each script prints its numerical tables to stdout and saves the corresponding convergence figures inside `figures/` next to the script.
