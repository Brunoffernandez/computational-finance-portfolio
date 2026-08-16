import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
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

#now, we create the cashflow matrix
C = np.zeros(( S.T[0].size, S[0].size-1))
print(C)

#Now we create the functions

def print_formatted_table(C, step_t):
    # we create the dataframe with column namese t = 1, t = 2, ...
    cols = [f't = {j+1}' for j in range(C.shape[1])]
    df = pd.DataFrame(C, columns=cols, index=range(1, len(C) + 1))

    # we write "-" in future columns
    for j in range(step_t - 1):
        df.iloc[:, j] = "—"

    # we write the column with two decimals
    df.iloc[:, step_t - 1] = df.iloc[:, step_t - 1].apply(lambda x: f"{x:.5f}")

    print(f"Cash-flow matrix at time {step_t}")
    print(df)
    print("\n")

def print_regression_table(S, Y_future, itm, t):
  #we create the dataframe for the regression table
    rows = []
    for path_idx in range(len(S)):
        if itm[path_idx]:
            rows.append({'Path': path_idx+1,
                        'Y': Y_future[path_idx],
                        'X': S[path_idx, t]})
        else:
            rows.append({'Path': path_idx+1,
                        'Y': np.nan,
                        'X': np.nan})

    df = pd.DataFrame(rows)
    print(f"Regression at time {t}")
    print(df.to_string(index=False,
                       float_format=lambda x: f"{x:.5f}",
                       na_rep='—'))
    print()

def optimal_table(S,continuation, exercise, itm, t):
  #we create the dataframe for the regression table
    rows = []
    for path_idx in range(len(S)):
        if itm[path_idx]:
            rows.append({'Path': path_idx+1,
                        'Exercise': exercise[path_idx],
                        'Continuation': continuation[path_idx]})
        else:
            rows.append({'Path': path_idx+1,
                        'Exercise': np.nan,
                        'Continuation': np.nan})

    df = pd.DataFrame(rows)
    print(f"Optimal early exercise decision at time {t}")
    print(df.to_string(index=False,
                       float_format=lambda x: f"{x:.5f}",
                       na_rep='—'))
    print()


def LSQM(S,C,dt,T, K, r):
  #we initialize last column
  C[:, T-1] = np.maximum(K - S[:, T], 0)
  print_formatted_table(C, T)
  C_european = C[:,T-1].copy()
  stopping_matrix = np.zeros((len(S),len(S.T)-1),dtype=int)
  #backwards recursion
  for i in range (T-2,-1,-1):
    #we create vector X and Y (for Y we need to discount all possible future cash flows)
    X = (S[:,i+1]*(S[:,i+1]<K)).reshape(-1,1) #must be a 2D array
    u = np.arange(i+1, C.shape[1])
    discount_factors = np.exp(-r * dt * (u - i))
    Y = C[:, i+1:] @ discount_factors #vector product for discount factor vector and cash flow vector
    itm_indexes = S[:,i+1]<K
    X_itm = X[itm_indexes]
    Y_itm = Y[itm_indexes]
    #Y = (S[:,i+1]<K)*C[:,i+1]*np.exp(-r*dt)
    #we make the regression, but only with the in the money indexes (when S(t)<K)
    model = make_pipeline(PolynomialFeatures(degree = 2, include_bias=False), LinearRegression(fit_intercept=True))
    model.fit(X_itm,Y_itm)
    #regression parameters
    linear_regression_step = model.named_steps['linearregression']
    a0 = linear_regression_step.intercept_
    a1, a2 = linear_regression_step.coef_
    #we print regression table and conditional expected value expression
    print_regression_table(S, Y, itm_indexes, i+1)
    print(f"E[Y|X] = {a0:.5f} + {a1:.5f}*X + {a2:.5f}*X^2\n")
    #now, we need to decide if we decide to exercise or continue based on the predictions of our model(linear regression)
    continuation = np.zeros(len(S))
    exercise = np.zeros(len(S))
    continuation[itm_indexes] = model.predict(X_itm)
    exercise[itm_indexes] = K - S[itm_indexes,i+1]
    optimal_table(S,continuation, exercise, itm_indexes, i+1)
    for j in range (len(S)):
      if exercise[j] > continuation[j]:
        C[j,i] = exercise[j]
        C[j,i+1:] = 0.00 #we need to eliminate all the future values
      else:
        C[j,i] = 0.00
    print_formatted_table(C,i+1)

  #we construct the stopping matrix by searching the values in which the cash flow matrix is different from 0
  stopping_matrix = np.where(C!=0, 1, stopping_matrix)
  # Stopping rule
  cols = [f't = {j+1}' for j in range(T)]
  df_stop = pd.DataFrame(stopping_matrix, columns=cols, index=range(1, len(S)+1))
  print("Stopping rule")
  print(df_stop.to_string(float_format=lambda x: f"{x:.5f}"))
  print()

  # Option cash flow matrix
  cols2 = [f't = {j+1}' for j in range(C.shape[1])]
  df_cf = pd.DataFrame(C, columns=cols2, index=range(1, len(S)+1))
  print("Option cash flow matrix")
  print(df_cf.to_string(float_format=lambda x: f"{x:.5f}"))
  print()

  #American put option price rounded to 5 decimals
  discount_times = np.arange(1,T+1)
  discount_factors = np.exp(-r*dt*discount_times)

  price_american = np.sum(C*discount_factors, axis = 1)
  price_american = np.mean(price_american)
  print("The american put option price rounded to 5 decimals is: ", f"{price_american:.5f}")


  #European put option price rounded to 5 decimals
  price_european = np.mean(C_european*np.exp(-r*dt*T))
  print("The european put option price rounded to 5 decimals is: ", f"{price_european:.5f}")




output = LSQM(S,C,dt,T,K,r)

"""Now we continue with 1(ii)"""

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

def LSM_pricing(S, dt, T, K, r):
    # we change LSMC for this section
    # we initialize last column
    C = np.zeros((len(S),
                  T))
    C[:, T-1] = np.maximum(K - S[:, T], 0)

    # backwards recursion
    for i in range(T-2, -1, -1):
        u = np.arange(i+1, C.shape[1])
        discount_factors = np.exp(-r * dt * (u - i))
        Y = C[:, i+1:] @ discount_factors

        itm_indexes = S[:, i+1] < K
        X_itm = laguerre_basis(S[itm_indexes, i+1], K)  #we use laguerre polynomials instead of polynomial basis
        Y_itm = Y[itm_indexes]

        model = LinearRegression()  #we do not need to make a pipeline as we have written directly the polynomials, however, we could have imported the laguerre polynomials
        #from sklearn library and we would have made the same
        model.fit(X_itm, Y_itm)

        continuation = np.zeros(len(S))
        exercise = np.zeros(len(S))
        continuation[itm_indexes] = model.predict(X_itm)
        exercise[itm_indexes] = K - S[itm_indexes, i+1]

        for j in range(len(S)):
            if exercise[j] > continuation[j]:
                C[j, i] = exercise[j]
                C[j, i+1:] = 0.00
            else:
                C[j, i] = 0.00

    # cashflows descontados por path (para precio y s.e.)
    discount_times = np.arange(1, T+1)
    discount_factors = np.exp(-r * dt * discount_times)
    cashflows = np.sum(C * discount_factors, axis=1)
    return cashflows

params = [
    (36, 0.20, 1), (36, 0.20, 2), (36, 0.40, 1), (36, 0.40, 2),
    (38, 0.20, 1), (38, 0.20, 2), (38, 0.40, 1), (38, 0.40, 2),
    (40, 0.20, 1), (40, 0.20, 2), (40, 0.40, 1), (40, 0.40, 2),
    (42, 0.20, 1), (42, 0.20, 2), (42, 0.40, 1), (42, 0.40, 2),
    (44, 0.20, 1), (44, 0.20, 2), (44, 0.40, 1), (44, 0.40, 2),
]

#now we compute the simulations
results = []
for i in range (len(params)):
  T = params[i][2]
  S0 = params[i][0]
  sigma = params[i][1]
  M_total = M*T #total number of exercise points
  dt = 1/M
  S = simulate_paths(S0, r, sigma, T, M_total, N, dt)
  cashflow = LSM_pricing(S,dt,M_total,K,r)
  american_value = np.mean(cashflow)
  european_value = black_scholes_put(S0, K, r, sigma, T)
  ee = american_value-european_value
  se = np.std(cashflow)/np.sqrt(N//2)
  results.append({
      'S' : S0, 'sigma' : sigma, 'T' : T,
      'Simulated American': round(american_value, 4),
      's.e.': f"({round(se, 4)})",
      'Closed form European': round(european_value, 4),
      'Early exercise value': round(ee, 4)
  })
df = pd.DataFrame(results)
print(df.to_string(index=False))
