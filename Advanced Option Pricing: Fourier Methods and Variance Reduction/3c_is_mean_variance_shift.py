import numpy as np
import matplotlib.pyplot as plt

# Parameters
N_stocks = 1000
K = 2
S_0 = 1
r = 0.01
sigma1 = 0.8
sigma2 = 0.6
T = 1
drift = r - 0.5 * (sigma1**2 + sigma2**2)
N_Paths = [1000, 5000, 10000, 50000, 100000, 500000]
quantiles_meanvariance = []

# Importance Sampling parameters
mu_shift = 1.5
var_q    = 2 * T   

def weighted_quantile(values, weights, q):
    """
    we calculate the weighted empirical quantile.
    We sort by L, then find the smallest q such that weighted mass of {L >= q} <= alpha.
    """
    sorter = np.argsort(values) #returns the indices that would made the values ordered from small to large.
    v_sorted = values[sorter]
    w_sorted = weights[sorter]
    
    # we compute the cumulative sum of normalized weights
    cdf = np.cumsum(w_sorted) / np.sum(w_sorted)
    
    # we find the index where cdf crosses the requested quantile
    idx = np.searchsorted(cdf, q)
    return v_sorted[idx]

np.random.seed(42)

print(f"{'N_paths':>8}  {'95th pct':>10}")
print("-" * 22)

for paths in N_Paths:
    # Sample W from q = N(1.5, 2T)  
    W_q = np.random.normal(mu_shift, np.sqrt(var_q), paths) # we generate common factor W ~ N(1.5, 2T) for M paths
    Bi  = np.random.normal(0, np.sqrt(T), size=(paths, N_stocks))# we generate idiosyncratic factors B_1^i ~ N(0, T) for M paths and N stocks

    #we compute terminal prices of the stock St
    S_T = np.exp(drift * T + sigma1 * W_q[:, None] + sigma2 * Bi)

    #we compute the payoff Lt
    L_T = np.sum(S_T > K, axis=1)

    #we compute the ratio p(w)/q(w) = exp(-1.5 * w + 1.125)
    # p/q = sqrt(2) + exp((-w^2 - 3w + 2.25) / (4T))
    ratio  = np.sqrt(2)*np.exp((-W_q**2 - 3 * W_q + 2.25) / (4 * T))

    q_95 = weighted_quantile(L_T, ratio, 0.95)
    quantiles_meanvariance.append(q_95)
    print(f"{paths:>8}  {q_95:>10.2f}")

plt.figure(figsize=(8, 5))
plt.plot(N_Paths, quantiles_meanvariance, marker='^', linestyle='-',
             color='red', label='Estimated 95th pct')
plt.axhline(quantiles_meanvariance[-1], color='gray', linestyle='--',
            label='Reference (Last value)')
plt.title("Importance Sampling (Mean+Var shift): Convergence of 95% Quantile")
plt.xlabel("Number of Simulations (N_paths)")
plt.ylabel("Estimated 95% Quantile")
plt.legend()
plt.grid(True, which='both', alpha=0.4)
plt.tight_layout()
plt.show()