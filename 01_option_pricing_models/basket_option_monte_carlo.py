"""Exercise 4 - Computational Finance
"""

import numpy as np
import matplotlib.pyplot as plt


def simulate_underQ(S0, volatilities, r, rho, T, K, M):
  '''
   We simulate M paths of the basket call under the risk-neutral measure Q
    and compute the Monte Carlo price estimator and standard error.
  '''

  np.random.seed(44)

  sigma1 = volatilities[0]
  sigma2 = volatilities[1]

  # Generate two independent standard normal samples
  Z1 = np.random.standard_normal(M)
  Z2 = np.random.standard_normal(M)

  # Construct correlated Brownian motions using Cholesky decomposition:
  # W1 and W2 satisfy dW1dW2 = rho dt
  W1 = Z1
  W2 = rho*Z1 + np.sqrt(1 -rho**2)*Z2

  # Simulate terminal stock prices using the exact GBM solution under Q
  S1 = S0[0]*np.exp((r-0.5*sigma1**2)*T + sigma1*np.sqrt(T)*W1)
  S2 = S0[1]*np.exp((r-0.5*sigma2**2)*T + sigma2*np.sqrt(T)*W2)

  AT = 0.5*(S1 + S2)

  # Discounted payoff for each path
  V = np.exp(-r*T)*np.maximum(AT-K, 0)
  
  #monte carlo estimator and SE 
  V_aprox = np.mean(V)
  SE_aprox = np.std(V, ddof = 1)/np.sqrt(M)

  return V_aprox, SE_aprox

def simulate_underQS1(S0, volatilities, r, rho, T, K, M):
    
  '''
   We simulate M paths of the basket call under the risk-neutral measure QS1
    and compute the Monte Carlo price estimator and standard error.
  '''

  np.random.seed(44)

  sigma1 = volatilities[0]
  sigma2 = volatilities[1]
  
  # Generate two independent standard normal samples
  Z1 = np.random.standard_normal(M)
  Z2 = np.random.standard_normal(M)

  # Construct correlated Brownian motions using Cholesky decomposition:
  # W1 and W2 satisfy dW1dW2 = rho dt
  W1 = Z1
  W2 = np.sqrt(1-rho**2)*Z2 + rho*Z1

  # Simulate terminal stock prices using the exact GBM solution under Q
  S1 = S0[0]*np.exp((r + 0.5*sigma1**2)*T + sigma1*np.sqrt(T)*W1)
  S2 = S0[1]*np.exp((r+ rho*sigma1*sigma2 -  0.5*sigma2**2)*T + sigma2*np.sqrt(T)*W2)

  AT = 0.5*(S1 + S2)
  
  # Discounted payoff for each path
  V = S0[0]*np.maximum(AT-K,0)/S1
  
  #monte carlo estimator and SE
  V_aprox = np.mean(V)
  SE_aprox = np.std(V, ddof = 1)/np.sqrt(M)

  return V_aprox, SE_aprox

def simulate_scenarios(S0, volatilities, r, rho, T, K):
    
  '''
   We run Monte Carlo simulations under Q and QS1 for a range of different paths M,
   we print a summary table of prices and standard errors, and create a
   dual-axis plot showing convergence of prices and 1/sqrt(M) decay of SEs.
  '''

  # Run simulations for each value of M under both measures
  values_M = [10_000, 25_000, 50_000, 75_000, 100_000]
  scenarios1 = [simulate_underQ(S0,volatilities,r, rho, T, K, m) for m in values_M]
  scenarios2 = [simulate_underQS1(S0,volatilities, r, rho, T, K ,m) for m in values_M]
  
  # Prices and standard errors
  VQ = [scenario[0] for scenario in scenarios1]
  SEQ = [scenario[1] for scenario in scenarios1]
  VQS1 = [scenario[0] for scenario in scenarios2]
  SEQS1 = [scenario[1] for scenario in scenarios2]

  # Printing summary table
  print(f"\n{'='*65}")
  print(f"{'M':>10} | {'Q price':>10} | {'Q SE':>9} | {'QS1 price':>11} | {'QS1 SE':>9}")
  print("-"*65)
  for i, M in enumerate(values_M):
      print(f"{M:>10,} | {VQ[i]:>10.4f} | {SEQ[i]:>9.5f} | {VQS1[i]:>11.4f} | {SEQS1[i]:>9.5f}")

  
  fig, ax1 = plt.subplots(figsize=(9, 5))
  ax2 = ax1.twinx()
  
  #we plot price estimates (left axis)
  ax1.plot(values_M, VQ,  'b-o', label='Price Q')
  ax1.plot(values_M, VQS1, 'r-s', label='Price QS1')
  
  #we plot standard errors (right axis)
  ax2.plot(values_M, SEQ,      'b--o', alpha=0.5, label='SE Q')
  ax2.plot(values_M, SEQS1,     'r--s', alpha=0.5, label='SE QS1')

  #we add 1/sqrt(M) reference curves to verify the central limit theorem
  M_arr = np.array(values_M, dtype=float)
  c_Q  = SEQ[0]  * np.sqrt(values_M[0])
  c_QS = SEQS1[0] * np.sqrt(values_M[0])
  ax2.plot(values_M, c_Q/np.sqrt(M_arr),  'b:', label='1/√M Q')
  ax2.plot(values_M, c_QS/np.sqrt(M_arr), 'r:', label='1/√M QS1')

  #axis labels and legends
  ax1.set_xlabel('M (number of paths)')
  ax1.set_ylabel('Estimated Price', color='black')
  ax2.set_ylabel('Standard Error', color='gray')
  lines1, labs1 = ax1.get_legend_handles_labels()
  lines2, labs2 = ax2.get_legend_handles_labels()
  ax1.legend(lines1+lines2, labs1+labs2, loc='upper right', fontsize=8)
  plt.title('Basket Call MC')
  plt.tight_layout()
  plt.show()

#simulation of first scenario
S0_1 = [100, 100]
volatilities_1 = (0.20, 0.20)
r_1    = 0.06
rho_1  = 0.9999
T_1    = 2
K_1    = 100

simulate_scenarios(S0_1, volatilities_1, r_1, rho_1, T_1, K_1)

#simulation of second scenario
S0_2   = [443, 73]
volatilities_2 = (0.10, 0.20)
r_2    = 0.06
rho_2  = 0.2
T_2    = 2
K_2    = (443 + 73)*0.5

simulate_scenarios(S0_2, volatilities_2, r_2, rho_2, T_2, K_2)