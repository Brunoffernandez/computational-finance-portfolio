"""Exercice 1 - Computational Finance
"""

import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt

def C_bs(S0,K,r,T, sigma):
  '''
    Compute the Black-Scholes price of a European call option.    
  '''
  if sigma <= 0 or T <= 0:
      return max(S0 - K * np.exp(-r * T), 0)
  d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
  d2 = d1-sigma*np.sqrt(T)
  return S0*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)

def implied_volatility_bisection(S0, K, r, T, C_mkt, tol, sigma_low, sigma_high, max_iter):
  '''
    Compute implied volatility by solving C_BS(sigma) = C_mkt using
    the Bisection method.
  '''
  f_low = C_bs(S0, K, r, T, sigma_low)-C_mkt
  f_high = C_bs(S0, K, r, T, sigma_high) - C_mkt
  
  # Check that the root is bracketed; if not, return NaN
  if f_low * f_high > 0:
        return np.nan    
  for i in range(max_iter):
    sigma_mid = (sigma_low + sigma_high)*0.5
    f_mid = C_bs(S0,K, r, T, sigma_mid) - C_mkt
    # Stop if function value is less than the tolerance or the interval is small enough
    if abs(f_mid) < tol or (sigma_high-sigma_low) < tol:
      return sigma_mid
    # we keep the sub-interval where the sign change occurs
    if f_low*f_mid <0:
     sigma_high = sigma_mid ; f_high = f_mid
    else:
      sigma_low = sigma_mid; f_low = f_mid
  return 0.5*(sigma_high + sigma_low)

def Vega(S0, K, r, T, sigma):
  '''
    Compute the Vega of a European call option under Black-Scholes.  
  '''
  #here we use that Vega = dC_bs/dsigma = S0*sqrt(T)N(d1), with N being the pdf of a Normal(0,1)
  d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
  return S0*np.sqrt(T)*norm.pdf(d1)

def implied_volatility_Newton_Ralphson(S0, K, r, T, C_mkt, tol, max_iter):
  '''
    Compute implied volatility by solving C_BS(sigma) = C_mkt using
    the Newton-Raphson method.
  '''
  #we compute the initial guess as the slides in Lecture 3, which
  #guarantees global convergence
  sigma_initial = np.sqrt((1/T)*(2*np.abs(np.log(S0/K)+ r*T)))
  sigma = sigma_initial
  for i in range(max_iter):
    call_price = C_bs(S0,K,r,T,sigma)
    vega = Vega(S0, K, r, T, sigma)
    diff = call_price - C_mkt
    if abs(diff) < tol:
      return sigma
    sigma -= diff/vega
  return sigma

#in order to read the CSV file
df = pd.read_csv("$spx-options-exp-2021-12-01-weekly-near-the-money-stacked-11-21-2021.csv")
df

# we extract the spot price S0 (stored in the 'SPX' row of the data)
S0 = float(df.loc[df['As of Date'] == 'SPX', '2021-Nov-19'].values[0])
T = 12/365
r = 0.0011

#we filter by the call options only
df_calls = df[df['Type'] == 'Call'].copy()

#cleaning strike price and IV
df_calls['Strike'] = df_calls['Strike'].astype(str).str.replace(',', '').astype(float)
df_calls = df_calls.dropna(subset=['Strike', 'Midpoint'])
df_calls['IV'] = df_calls['IV'].astype(str).str.replace('%', '').astype(float)

#setting the parameters
tol = 1e-6
sigma_low = 1e-6
sigma_high = 5.0
max_iter = 200
max_iter2 = 100

#computing IV by bisection and Newton for each value pair in (strike, midpoint) 
IV_bisection = [implied_volatility_bisection(S0, K, r, T, C, tol, sigma_low, sigma_high, max_iter) for K,C in zip(df_calls['Strike'], df_calls['Midpoint'])]
IV_Newton = [implied_volatility_Newton_Ralphson(S0, K, r, T, C, tol, max_iter2)
    for K,C in zip(df_calls['Strike'], df_calls['Midpoint'])]

# we convert from decimal to percentage for plotting
IV_bisection = [i*100 for i in IV_bisection]
IV_Newton = [i*100 for i in IV_Newton]

# Figure 1: bisection method
plt.plot(df_calls['Strike'], IV_bisection , 'o-', label='Bisection')
plt.xlabel('Strike'); plt.ylabel('Implied Volatility (%)')
plt.title('Implied Volatility : Bisection method')
plt.legend()
plt.grid(True)
plt.show()

# Figure 2: bisection + Newton methods
plt.plot(df_calls['Strike'], IV_bisection , 'o-', label='Bisection')
plt.plot(df_calls['Strike'], IV_Newton , 's--', label='Newton')
plt.xlabel('Strike'); plt.ylabel('Implied Volatility (%)')
plt.title('Implied Volatility : Bisection vs Newton')
plt.legend()
plt.grid(True)
plt.show()

# Figure 3: bisection and newton methods vs Reference (Excel IV)
plt.plot(df_calls['Strike'], df_calls['IV'], 'D-', label='CSV IV')
plt.plot(df_calls['Strike'], IV_bisection, 'o-', label='Bisection')
plt.plot(df_calls['Strike'], IV_Newton, 's--', label='Newton')
plt.xlabel('Strike'); plt.ylabel('Implied Volatility (%)')
plt.title('Implied Volatility : Bisection vs Newton vs Reference')
plt.legend()
plt.grid(True)
plt.show()