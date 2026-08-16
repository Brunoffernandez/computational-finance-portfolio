import numpy as np
import pandas as pd
from scipy.stats import norm

K  = 40
r  = 0.06

params = [
    (36, 0.20, 1), (36, 0.20, 2), (36, 0.40, 1), (36, 0.40, 2),
    (38, 0.20, 1), (38, 0.20, 2), (38, 0.40, 1), (38, 0.40, 2),
    (40, 0.20, 1), (40, 0.20, 2), (40, 0.40, 1), (40, 0.40, 2),
    (42, 0.20, 1), (42, 0.20, 2), (42, 0.40, 1), (42, 0.40, 2),
    (44, 0.20, 1), (44, 0.20, 2), (44, 0.40, 1), (44, 0.40, 2),
]

results = []

for j in range(len(params)):
    S0 = params[j][0]
    sigma = params[j][1]
    T = params[j][2]

    # exact price - Black-Scholes formula
    d1 = ( np.log(S0/K) + (r + 0.5*sigma**2) * T ) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    bs_put = norm.cdf(-d2)*K*np.exp(-r*T) - norm.cdf(-d1)*S0

    # forward finite difference
    L  = 2 * np.log(K) + 2
    Nx = 1000 #1,000 steps for the stock price
    Nt = 50 * T #50 time steps per year
    h  = L / Nx
    k  = T / Nt

    T1 = np.diag([1]*(Nx-2), 1) - np.diag([1]*(Nx-2), -1)
    T2 = -2*np.diag([1]*(Nx-1)) + np.diag([1]*(Nx-2), 1) + np.diag([1]*(Nx-2), -1)

    F = ( (1 - r*k) * np.diag([1]*(Nx-1))
        + 0.5*k*(sigma**2)/(h**2) * T2
        + k*(r - 0.5*(sigma**2))/(2*h) * T1 )

    mvec = np.linspace(start=-L/2 + h, stop=L/2 - h, num=Nx-1)
    mvec = mvec + np.log(K)   # shift to center grid at log(K)

    U = np.zeros((Nx-1, Nt+1))
    U[:, 0] = np.maximum(K - np.exp(mvec), 0)

    for i in range(Nt):
        time2mat = i * k
        p = np.zeros(Nx-1)
        p[0] = ( 0.5*k*(sigma**2)/(h**2) - k*(r-0.5*(sigma**2))/(2*h) ) * K * np.exp(-r*time2mat)
        U[:, i+1] = np.dot(F, U[:, i]) + p

    ftcs_price = np.interp(np.log(S0), mvec, U[:, Nt])

    results.append({
        'S': S0, 'sigma': sigma, 'T': T,
        'FD European': round(ftcs_price, 4),
        'BS European': round(bs_put, 4),
        'Difference':  round(ftcs_price - bs_put, 4)
    })

df = pd.DataFrame(results)
print(df.to_string(index=False))

results_american = []

for j in range(len(params)):
    S0 = params[j][0]
    sigma = params[j][1]
    T = params[j][2]

    # exact price - Black-Scholes formula (European)
    d1 = ( np.log(S0/K) + (r + 0.5*sigma**2) * T ) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    bs_put = norm.cdf(-d2)*K*np.exp(-r*T) - norm.cdf(-d1)*S0

    # forward finite difference
    L  = 2 * np.log(K) + 2
    Nx = 1000
    Nt = 50 * T
    h  = L / Nx
    k  = T / Nt

    T1 = np.diag([1]*(Nx-2), 1) - np.diag([1]*(Nx-2), -1)
    T2 = -2*np.diag([1]*(Nx-1)) + np.diag([1]*(Nx-2), 1) + np.diag([1]*(Nx-2), -1)

    F = ( (1 - r*k) * np.diag([1]*(Nx-1))
        + 0.5*k*(sigma**2)/(h**2) * T2
        + k*(r - 0.5*(sigma**2))/(2*h) * T1 )

    mvec = np.linspace(start=-L/2 + h, stop=L/2 - h, num=Nx-1)
    mvec = mvec + np.log(K)

    intrinsic = np.maximum(K - np.exp(mvec), 0)  # payoff in each node

    U = np.zeros((Nx-1, Nt+1))
    U[:, 0] = intrinsic

    for i in range(Nt):
        time2mat = i * k
        p = np.zeros(Nx-1)
        p[0] = ( 0.5*k*(sigma**2)/(h**2) - k*(r-0.5*(sigma**2))/(2*h) ) * K * np.exp(-r*time2mat)
        U[:, i+1] = np.dot(F, U[:, i]) + p
        U[:, i+1] = np.maximum(U[:, i+1], intrinsic)   #change, we exercise or we hold

    ftcs_american = np.interp(np.log(S0), mvec, U[:, Nt])
    ee = ftcs_american - bs_put

    results_american.append({
        'S': S0, 'sigma': sigma, 'T': T,
        'FD American': round(ftcs_american, 4),
        'BS European': round(bs_put, 4),
        'Early exercise value': round(ee, 4)
    })

df_american = pd.DataFrame(results_american)
print(df_american.to_string(index=False))

"""Code from 1(ii) to make the testing"""

import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import LinearRegression
warnings.filterwarnings('ignore', category=FutureWarning)


#Parameters
K = 1.10
r = 0.06
dt = 1
T = 3
S = np.array([
    [1.00, 1.09, 1.08, 1.34],
    [1.00, 1.16, 1.26, 1.54],
    [1.00, 1.22, 1.07, 1.03],
    [1.00, 0.93, 0.97, 0.92],
    [1.00, 1.11, 1.56, 1.52],
    [1.00, 0.76, 0.77, 0.90],
    [1.00, 0.92, 0.84, 1.01],
    [1.00, 0.88, 1.22, 1.34],
])

from scipy.stats import norm

#Parameters
K = 40
r = 0.06
M = 50
N = 100000 #fiftyk normal and fiftyk antithetic

def simulate_paths(S0, r, sigma, T, M, N, dt):
  #we simulate the paths
  Z = np.random.standard_normal((N//2,M)) #we simulate N//2 different paths (the other N//2 are fot the antithetic ones), and with M columns (possibilities of exercising)
  Z = np.vstack([Z,-Z]) #antithetic paths, now we have a matrix with N paths and M columns
  S = np.zeros((N,M+1))
  S[:,0] = S0
  for i in range (M):
    S[:,i+1] = S[:,i]*np.exp((r-0.5*sigma**2)*dt + Z[:,i]*sigma*np.sqrt(dt))
  return S

def laguerre_basis(S, K):
  #we compute the first three degree Laguerre polynomials with the normalization constant as the strike price (as they did in the paper)
    x = S / K
    L0 = np.exp(-x/2)
    L1 = np.exp(-x/2) * (1 - x)
    L2 = np.exp(-x/2) * (1 - 2*x + x**2/2)
    return np.column_stack([L0, L1, L2])

def black_scholes_put(S0, K, r, sigma, T):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S0*norm.cdf(-d1)

def LSM_pricing(S, dt, M_total, K, r):
    n = len(S)
    C = np.zeros((n, M_total))
    C[:, M_total-1] = np.maximum(K - S[:, M_total], 0)

    for i in range(M_total-2, -1, -1):
        u = np.arange(i+1, C.shape[1])
        discount_factors = np.exp(-r * dt * (u - i))
        Y = C[:, i+1:] @ discount_factors
        itm_indexes = S[:, i+1] < K
        X_itm = laguerre_basis(S[itm_indexes, i+1], K)
        Y_itm = Y[itm_indexes]
        model = LinearRegression()
        model.fit(X_itm, Y_itm)
        continuation = np.zeros(n)
        exercise = np.zeros(n)
        continuation[itm_indexes] = model.predict(laguerre_basis(S[itm_indexes, i+1], K))
        exercise[itm_indexes] = K - S[itm_indexes, i+1]
        for j in range(n):
            if exercise[j] > continuation[j]:
                C[j, i] = exercise[j]
                C[j, i+1:] = 0.00
            else:
                C[j, i] = 0.00

    discount_times = np.arange(1, M_total+1)
    discount_factors_all = np.exp(-r * dt * discount_times)
    cashflows = np.sum(C * discount_factors_all, axis=1)
    return cashflows

params = [
    (36, 0.20, 1), (36, 0.20, 2), (36, 0.40, 1), (36, 0.40, 2),
    (38, 0.20, 1), (38, 0.20, 2), (38, 0.40, 1), (38, 0.40, 2),
    (40, 0.20, 1), (40, 0.20, 2), (40, 0.40, 1), (40, 0.40, 2),
    (42, 0.20, 1), (42, 0.20, 2), (42, 0.40, 1), (42, 0.40, 2),
    (44, 0.20, 1), (44, 0.20, 2), (44, 0.40, 1), (44, 0.40, 2),
]

#now we compute the simulations
results_lsmc = []
for S0, sigma, T in params:
    M_total = M * T
    dt = 1/M
    S_paths = simulate_paths(S0, r, sigma, T, M_total, N, dt)
    cashflow = LSM_pricing(S_paths, dt, M_total, K, r)
    american_value = np.mean(cashflow)
    european_value = black_scholes_put(S0, K, r, sigma, T)
    ee = american_value - european_value
    se = np.std(cashflow) / np.sqrt(N)
    results_lsmc.append({
        'S': S0, 'sigma': sigma, 'T': T,
        'Simulated American': round(american_value, 4),
        's.e.': f"({round(se, 4)})",
        'Closed form European': round(european_value, 4),
        'Early exercise value': round(ee, 4)
    })

# test

test_rows = []
for r_fd, r_lsm in zip(results_american, results_lsmc):
    diff = abs(r_fd['FD American'] - r_lsm['Simulated American'])
    test_rows.append({
        'S':            r_fd['S'],
        'sigma':        r_fd['sigma'],
        'T':            r_fd['T'],
        'FD American':  r_fd['FD American'],
        'LSM American': r_lsm['Simulated American'],
        '|Difference|': round(diff, 4),
    })

df_test = pd.DataFrame(test_rows)
print(df_test.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

df_outside_box = pd.DataFrame({
    'S':                       [r['S']                    for r in results_american],
    'sigma':                   [r['sigma']                for r in results_american],
    'T':                       [r['T']                    for r in results_american],
    'FD American':             [r['FD American']          for r in results_american],
    'Closed form European':    [r['BS European']          for r in results_american],
    'FD Early exercise value': [r['Early exercise value'] for r in results_american],
    'Difference in EE value':  [round(r_fd['Early exercise value'] -
                                      r_lsm['Early exercise value'], 4)
                                for r_fd, r_lsm in zip(results_american, results_lsmc)],
})
print(df_outside_box.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
