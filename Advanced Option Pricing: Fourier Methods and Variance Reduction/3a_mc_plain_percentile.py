import numpy as np
import matplotlib.pyplot as plt

#parameters
N_stocks = 1000
S_0 = 1
r = 0.01
sigma1 = 0.8
sigma2 = 0.6
T = 1
K = 2 
alpha = 1-0.95 #we want the 95% percentile

drift = r-0.5*(sigma1**2 + sigma2**2)

def simulate_plainMC(paths):
    """
    We simulate M paths of L_T = sum_i 1{S_i(T) > K}.
    Returns an array of shape (M,)
    """
    
    Wt = np.random.normal(0, np.sqrt(T), size = paths) # we generate common factor W ~ N(0, T) for M paths
    Bi = np.random.normal(0, np.sqrt(T), size = (paths,N_stocks)) # we generate idiosyncratic factors B_1^i ~ N(0, T) for M paths and N stocks
    
    #we compute terminal prices of the stock St
    S_T = S_0 * np.exp(drift * T + sigma1 * Wt[:, None] + sigma2 * Bi)
    
    #we compute the payoff Lt
    L_T = np.sum(S_T > K, axis = 1)
    
    return L_T

N_Paths = [1000, 5000, 10000, 50000, 100000, 500000]
quantiles_mc = []

np.random.seed(42) #to repeat experiments

print(f"{'N_paths':>8}  {'95th pct':>10}")
print("-"*22)
for paths in N_Paths:
    L_T = simulate_plainMC(paths)
    q = np.quantile(L_T, 1 - alpha) #estimate the 95% percentile
    quantiles_mc.append(q)
    print(f"{paths:>8}  {q:>10.2f}")


plt.figure(figsize=(8,5))
plt.semilogx(N_Paths, quantiles_mc, marker='o', linestyle='-', color='blue', label='Estimated 95th pct')
plt.axhline(quantiles_mc[-1], color='gray', linestyle='--', label='Reference (Last value)')
plt.xlabel('Number of simulations (N_paths)')
plt.ylabel('Estimated 95% quantile of L_T')
plt.title('Plain Monte Carlo — convergence of 95th percentile of Lt')
plt.legend(); plt.grid(True, which='both', alpha=0.4)
plt.tight_layout()
plt.show()