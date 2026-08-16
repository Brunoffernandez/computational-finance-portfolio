import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq, least_squares
from numpy.polynomial.hermite_e import hermegauss
from math import comb, factorial

# we save all convergence figures next to this script so that the project
# is portable: whoever clones the repo can run the file from anywhere and
# the PNGs land in ./figures/ relative to the script location.
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

#common contract specification (priced under the risk-neutral measure Q)
w1 = w2 = 0.5
T = 1
r = 0.02
K = 100

#Gauss-Hermite nodes and weights for the standard normal (used by the Johnson SU
#pricing and to integrate against N(0,1))
GH_Z, GH_W = hermegauss(120)
GH_W = GH_W/np.sqrt(2*np.pi)

# Moments of the basket B_T = w1 S1(T) + w2 S2(T) under Q
def lognormal_cross(j, m, S10, S20, s1, s2, rho):
    """
    we compute the cross raw moment E[S1(T)^j S2(T)^m] for two correlated
    log-normal underlyings under Q (drift r), using that a sum of jointly normal
    log-returns is normal and E[exp(N)] = exp(mean + 0.5 var).
    """
    muX = np.log(S10) + (r - 0.5*s1**2)*T
    muY = np.log(S20) + (r - 0.5*s2**2)*T
    vX, vY, cXY = s1**2*T, s2**2*T, rho*s1*s2*T
    return np.exp(j*muX + m*muY + 0.5*(j**2*vX + m**2*vY + 2*j*m*cXY))

def basket_moments_LN(S10, S20, s1, s2, rho, nmax=4):
    """
    we compute the first nmax raw moments m_k = E[B_T^k] of the basket variable
    B_T = sum_i w_i S_i(T) when both underlyings are log-normal.
    """
    return [sum(comb(k, j)*w1**j*w2**(k-j)*lognormal_cross(j, k-j, S10, S20, s1, s2, rho)
                for j in range(k+1)) for k in range(1, nmax+1)]

def cir_S2_rawmoments(S20, sig2, nmax=4):
    """
    we compute the first nmax raw moments of S2(T) for the square-root process
    dS2 = r S2 dt + sigma2 sqrt(S2) dW2. By the technical hint S2(T) = c * Z with
    c = exp(rT) sigma2^2 tau(T)/4 and Z ~ chi'^2_0(lambda), a non-central
    chi-square with zero degrees of freedom and lambda = 4 S20/(sigma2^2 tau(T)).
    """
    tau = (1 - np.exp(-r*T))/r
    c = np.exp(r*T)*sig2**2*tau/4
    lam = 4*S20/(sig2**2*tau)
    kappa = [2**(n-1)*factorial(n)*lam for n in range(1, nmax+1)]
    m = [1.0]
    for n in range(1, nmax+1):
        m.append(sum(comb(n-1, kk)*kappa[n-1-kk]*m[kk] for kk in range(n)))
    return [c**(k+1)*m[k+1] for k in range(nmax)]

def basket_moments_CIR(S10, s1, S20, sig2, nmax=4):
    """
    we compute the first nmax raw moments of the basket when S1 is log-normal and
    S2 is the square-root process, using independence (rho = 0) so that the joint
    moments factorise into the product of marginal moments.
    """
    S1m = [1.0] + [lognormal_cross(k, 0, S10, 1.0, s1, 1.0, 0.0) for k in range(1, nmax+1)]
    S2m = [1.0] + cir_S2_rawmoments(S20, sig2, nmax)
    return [sum(comb(k, j)*w1**j*w2**(k-j)*S1m[j]*S2m[k-j] for j in range(k+1))
            for k in range(1, nmax+1)]

def central_from_raw(mom):
    """
    we convert the first four raw moments into mean, variance, skewness and
    kurtosis (the standardized moments that the approximating distributions match).
    """
    m1, m2, m3, m4 = mom[:4]
    c2 = m2 - m1**2
    c3 = m3 - 3*m1*m2 + 2*m1**3
    c4 = m4 - 4*m1*m3 + 6*m1**2*m2 - 3*m1**4
    return m1, c2, c3/c2**1.5, c4/c2**2

# Approximating distributions
def price_2moments(mom):
    """
    we fit a log-normal to B_T matching the expected
    basket value m1 (its forward) and the second moment m2, and price the call
    with a Black-style formula (Black's formula with forward F = m1 and total
    variance s^2 = log(m2/m1^2)).
    """
    m1, m2 = mom[0], mom[1]
    s2 = np.log(m2/m1**2)
    s = np.sqrt(s2)
    d1 = (np.log(m1/K) + 0.5*s2)/s; d2 = d1 - s
    return np.exp(-r*T)*(m1*norm.cdf(d1) - K*norm.cdf(d2))

def price_3moments(mom):
    """
    we match mean, variance and skewness with a shifted log-normal B = gamma + L,
    where L ~ LN. With w = exp(s^2), the log-normal skewness is (w+2) sqrt(w-1),
    which we invert for w.
    We then price the call on B as a shifted-strike log-normal call.
    """
    m1, c2, g1, _ = central_from_raw(mom)
    w = brentq(lambda x: (x + 2)*np.sqrt(x - 1) - g1, 1 + 1e-12, 100)
    s2 = np.log(w); s = np.sqrt(s2)
    M = np.sqrt(c2/(w - 1))           # E[L]
    gamma = m1 - M                    # shift
    Kp = K - gamma                    # shifted strike
    if Kp <= 0:
        return np.exp(-r*T)*(m1 - K)
    d1 = (np.log(M/Kp) + 0.5*s2)/s; d2 = d1 - s
    return np.exp(-r*T)*(M*norm.cdf(d1) - Kp*norm.cdf(d2))

def price_4moments(mom):
    """
    we match mean, variance, skewness and kurtosis with a Johnson SU distribution
    B = xi + lambda*sinh((Z - gamma)/delta), Z ~ N(0,1). The shape parameters
    (delta, gamma) are fitted from the standardized skewness and kurtosis (their
    closed forms are difficult to solve, so we solve the 2x2 system numerically with
    Gauss-Hermite moments of sinh), and (xi, lambda) are then set from the mean
    and variance. We price the call by Gauss-Hermite quadrature.
    """
    m1, c2, g1, b2 = central_from_raw(mom); sd = np.sqrt(c2)

    def U_moments(delta, gamma):
        U = np.sinh((GH_Z - gamma)/delta)
        mu = (GH_W*U).sum(); v = (GH_W*(U - mu)**2).sum()
        sk = (GH_W*(U - mu)**3).sum()/v**1.5
        ku = (GH_W*(U - mu)**4).sum()/v**2
        return mu, v, sk, ku

    def resid(p):
        delta, gamma = p
        if delta <= 0.05 or delta > 50:
            return [1e3, 1e3]
        _, _, sk, ku = U_moments(delta, gamma)
        return [sk - g1, ku - b2]

    sol = least_squares(resid, [2.0, -0.3 if g1 > 0 else 0.3], bounds=([0.1, -10], [50, 10]))
    delta, gamma = sol.x
    muU, vU, _, _ = U_moments(delta, gamma)
    lam = sd/np.sqrt(vU); xi = m1 - lam*muU
    X = xi + lam*np.sinh((GH_Z - gamma)/delta)
    return np.exp(-r*T)*(GH_W*np.maximum(X - K, 0)).sum()


# High-precision Monte Carlo benchmark (exact simulation, antithetic)

def mc_benchmark_LN(S10, S20, s1, s2, rho, M):
    """
    we compute simulations for the log-normal basket using exact
    GBM simulation with antithetic variates.
    """
    rng = np.random.default_rng(99)
    Z1 = rng.standard_normal(M)
    Zc = rng.standard_normal(M)
    Z2 = rho*Z1 + np.sqrt(1 - rho**2)*Zc
    out = []
    for sgn in (1, -1):
        S1 = S10*np.exp((r - 0.5*s1**2)*T + s1*np.sqrt(T)*sgn*Z1)
        S2 = S20*np.exp((r - 0.5*s2**2)*T + s2*np.sqrt(T)*sgn*Z2)
        out.append(np.exp(-r*T)*np.maximum(w1*S1 + w2*S2 - K, 0))
    pair = 0.5*(out[0] + out[1])
    return pair.mean(), pair.std(ddof=1)/np.sqrt(M)

def mc_benchmark_CIR(S10, s1, S20, sig2, M):
    """
    we compute a high-precision benchmark for the GBM + CIR basket with rho = 0
    using exact simulation. S1 from the exact GBM solution and S2 from the
    technical hint, S2(T) = c * Z with Z ~ chi'^2_0(lambda), we use antithetic variates on S1.
    """
    rng = np.random.default_rng(99)
    tau = (1 - np.exp(-r*T))/r
    c = np.exp(r*T)*sig2**2*tau/4
    lam = 4*S20/(sig2**2*tau)
    N = rng.poisson(lam/2, M)
    Zchi = np.where(N > 0, rng.gamma(np.maximum(N, 1), 2.0), 0.0)
    S2 = c*Zchi
    Zg = rng.standard_normal(M)
    out = []
    for sgn in (1, -1):
        S1 = S10*np.exp((r - 0.5*s1**2)*T + s1*np.sqrt(T)*sgn*Zg)
        out.append(np.exp(-r*T)*np.maximum(w1*S1 + w2*S2 - K, 0))
    pair = 0.5*(out[0] + out[1])
    return pair.mean(), pair.std(ddof=1)/np.sqrt(M)


# Q2(a): both log-normal, rho = 0.30

mom_a = basket_moments_LN(100, 100, 0.20, 0.30, 0.30)
m1, c2, g1, b2 = central_from_raw(mom_a)
mc_a, se_mc_a = mc_benchmark_LN(100, 100, 0.20, 0.30, 0.30, 4000000)
p2_a, p3_a, p4_a = price_2moments(mom_a), price_3moments(mom_a), price_4moments(mom_a)

print("Q2(a) both log-normal, rho = 0.30")
print(f"  basket moments: mean = {m1:.4f}, std = {np.sqrt(c2):.4f}, "
      f"skewness = {g1:.4f}, kurtosis = {b2:.4f}")
print(f"  MC benchmark: {mc_a:.5f}  (SE {se_mc_a:.5f})")
print(f"{'moments matched':>16}  {'distribution':>16}  {'price':>9}  {'abs error':>10}")
print("-"*60)
for nm, dist, pr in [(2, 'log-normal', p2_a), (3, 'shifted log-normal', p3_a),
                     (4, 'Johnson SU', p4_a)]:
    print(f"{nm:>16}  {dist:>16}  {pr:>9.5f}  {abs(pr-mc_a):>10.5f}")
print()

# Q2(b): GBM + CIR, rho = 0

mom_b = basket_moments_CIR(100, 0.25, 100, 1.5)
m1b, c2b, g1b, b2b = central_from_raw(mom_b)
mc_b, se_mc_b = mc_benchmark_CIR(100, 0.25, 100, 1.5, 4000000)
p2_b, p3_b, p4_b = price_2moments(mom_b), price_3moments(mom_b), price_4moments(mom_b)

print(" GBM + CIR, rho = 0")
print(f"  basket moments: mean = {m1b:.4f}, std = {np.sqrt(c2b):.4f}, "
      f"skewness = {g1b:.4f}, kurtosis = {b2b:.4f}")
print(f"  MC benchmark: {mc_b:.5f}  (SE {se_mc_b:.5f})")
print(f"{'moments matched':>16}  {'distribution':>16}  {'price':>9}  {'abs error':>10}")
print("-"*60)
for nm, dist, pr in [(2, 'log-normal', p2_b), (3, 'shifted log-normal', p3_b),
                     (4, 'Johnson SU', p4_b)]:
    print(f"{nm:>16}  {dist:>16}  {pr:>9.5f}  {abs(pr-mc_b):>10.5f}")


# Error convergence figure: |error| vs number of matched moments
err_a = [abs(p2_a-mc_a), abs(p3_a-mc_a), abs(p4_a-mc_a)]
err_b = [abs(p2_b-mc_b), abs(p3_b-mc_b), abs(p4_b-mc_b)]
nmoments = [2, 3, 4]

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].semilogy(nmoments, err_a, 'o-', color='blue')
ax[0].set_title(' both log-normal, $\\rho=0.30$')
ax[1].semilogy(nmoments, err_b, 's-', color='red')
ax[1].set_title(' GBM + CIR, $\\rho=0$')
for a in ax:
    a.set_xlabel('number of matched moments'); a.set_ylabel('|price error| vs MC')
    a.set_xticks(nmoments); a.grid(True, which='both', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'q2_moment_error.png'), dpi=150)
plt.show()
