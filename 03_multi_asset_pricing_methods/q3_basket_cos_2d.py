import os
import numpy as np
import matplotlib.pyplot as plt

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

# Two-dimensional COS method from Ruijter and Oosterlee, 2012 paper

#We price V(0) = e^{-rT} E^Q[max(w1 g1(y1) + w2 g2(y2) - K, 0)] by expanding the
#joint density of the two state variables (y1, y2) in a two-dimensional cosine
#series on [a1,b1] x [a2,b2]. The density cosine coefficients factorise into a
#product of one-dimensional coefficients when the two marginals are independent
#(here rho = 0 in both setups), each recovered from the marginal characteristic
#function. The price is then
#   V(0) = e^{-rT} sum_{k1}'' sum_{k2}'' chi_{k1} chi_{k2} V_{k1,k2},
#with chi_k = Re{ phi(omega_k) exp(-i omega_k a) } (first term halved), and the
#payoff coefficients V_{k1,k2} computed by quadrature.

def chi_coeffs(phi, a, b, N):
    """
    we compute the one-dimensional density cosine coefficients
    chi_k = Re{ phi(omega_k) exp(-i omega_k a) }, with omega_k = k pi/(b-a), and
    halve the k = 0 term to implement the primed summation of the COS method.
    """
    k = np.arange(N)
    om = k*np.pi/(b - a)
    chi = np.real(phi(om)*np.exp(-1j*om*a))
    chi[0] *= 0.5
    return chi

def payoff_coeffs_2d(a1, b1, a2, b2, N1, N2, g1, g2, Ngrid):
    """
    we compute the payoff cosine coefficients V_{k1,k2}, defined as the double
    integral over [a1,b1] x [a2,b2] of max(w1 g1(y1) + w2 g2(y2) - K, 0) against
    cos(k1 pi (y1-a1)/(b1-a1)) cos(k2 pi (y2-a2)/(b2-a2)), times the normalisation
    (2/(b1-a1))(2/(b2-a2)). We follow the paper's suggestion to use numerical
    methods to solve this coefficient, we evaluate them with a dense trapezoidal rule written as the matrix
    product C1^T P C2.
    """
    y1 = np.linspace(a1, b1, Ngrid)
    y2 = np.linspace(a2, b2, Ngrid)
    Wy = np.ones(Ngrid)
    Wy[0] = Wy[-1] = 0.5          # trapezoidal weights
    h1 = (b1 - a1)/(Ngrid - 1)
    h2 = (b2 - a2)/(Ngrid - 1)
    P = np.maximum(w1*g1(y1)[:, None] + w2*g2(y2)[None, :] - K, 0.0)
    k1 = np.arange(N1)
    k2 = np.arange(N2)
    C1 = np.cos(np.outer(y1 - a1, k1*np.pi/(b1 - a1)))*(Wy*h1)[:, None]
    C2 = np.cos(np.outer(y2 - a2, k2*np.pi/(b2 - a2)))*(Wy*h2)[:, None]
    return (2.0/(b1 - a1))*(2.0/(b2 - a2))*(C1.T @ P @ C2)

def price_2dCOS(phi1, phi2, a1, b1, a2, b2, N1, N2, g1, g2, Ngrid=4000):
    """
    we assemble the 2D COS price as e^{-rT} chi1^T V chi2.
    """
    chi1 = chi_coeffs(phi1, a1, b1, N1)
    chi2 = chi_coeffs(phi2, a2, b2, N2)
    V = payoff_coeffs_2d(a1, b1, a2, b2, N1, N2, g1, g2, Ngrid)
    return np.exp(-r*T)*(chi1 @ V @ chi2)

def cumulant_range(k1, k2, k4, L=12):
    """
    we choose the truncation interval with the COS-paper cumulant rule
    [a, b] = k1 +/- L sqrt(k2 + sqrt(k4)).
    """
    width = L*np.sqrt(abs(k2) + np.sqrt(abs(k4)))
    return k1 - width, k1 + width

# Q3(a): both log-normal, rho = 0 ; the state variables are (x1, x2) = (log S1, log S2)

s1_a, s2_a = 0.20, 0.30
mu1_a = np.log(100) + (r - 0.5*s1_a**2)*T
mu2_a = np.log(100) + (r - 0.5*s2_a**2)*T
#characteristic functions of the two normal log-prices under Q
phi1_a = lambda om: np.exp(1j*om*mu1_a - 0.5*s1_a**2*T*om**2)
phi2_a = lambda om: np.exp(1j*om*mu2_a - 0.5*s2_a**2*T*om**2)
#cumulant truncation (a normal has k1 = mean, k2 = variance, k4 = 0)
a1_a, b1_a = cumulant_range(mu1_a, s1_a**2*T, 0)
a2_a, b2_a = cumulant_range(mu2_a, s2_a**2*T, 0)
#both underlyings enter the payoff through the exponential map S = exp(x)
g1_a = g2_a = np.exp

#converged COS reference at large N
ref_a = price_2dCOS(phi1_a, phi2_a, a1_a, b1_a, a2_a, b2_a, 256, 256, g1_a, g2_a, Ngrid=12000)

print("Q3(a) both log-normal, rho = 0")
print(f"  truncation domains: x1 in [{a1_a:.3f}, {b1_a:.3f}], x2 in [{a2_a:.3f}, {b2_a:.3f}]")
print(f"  converged COS price (N = 256): {ref_a:.6f}")
print(f"{'N':>4}  {'COS price':>11}  {'|error|':>10}  {'ratio':>8}")
print("-"*40)
prev = None
N_list = [8, 16, 24, 32, 40, 48, 64]
err_a = []
for N in N_list:
    p = price_2dCOS(phi1_a, phi2_a, a1_a, b1_a, a2_a, b2_a, N, N, g1_a, g2_a, Ngrid=8000)
    e = abs(p - ref_a); err_a.append(e)
    ratio = "" if prev is None else f"{prev/e:7.1f}x"
    print(f"{N:>4}  {p:>11.6f}  {e:>10.2e}  {ratio:>8}")
    prev = e
print()

# Q3(b): GBM + CIR, rho = 0 ; the state variables are (x1, S2) = (log S1, level S2)

s1_b = 0.25
sig2_b = 1.5
mu1_b = np.log(100) + (r - 0.5*s1_b**2)*T
#S1 keeps the normal log-price characteristic function
phi1_b = lambda om: np.exp(1j*om*mu1_b - 0.5*s1_b**2*T*om**2)
#S2 we use the hint from the assesment:  S2(T) = c Z with Z ~ chi'^2_0(lambda).
tau = (1 - np.exp(-r*T))/r
c_b = np.exp(r*T)*sig2_b**2*tau/4
lam_b = 4*100/(sig2_b**2*tau)
def phi2_b(om):
    z = om*c_b
    return np.exp(lam_b*1j*z/(1 - 2j*z))

#cumulants of S2(T) = c Z, with chi'^2_0 cumulants kappa_n = 2^(n-1) n! lambda
k1_S2 = c_b*lam_b
k2_S2 = c_b**2*4*lam_b
k4_S2 = c_b**4*192*lam_b
a1_b, b1_b = cumulant_range(mu1_b, s1_b**2*T, 0)
a2_b, b2_b = cumulant_range(k1_S2, k2_S2, k4_S2, L=12)
a2_b = max(a2_b, 0.0)                     # S2 is non-negative
g1_b = np.exp                             # S1 = exp(x1)
g2_b = lambda y: y                        # S2 is already a level

ref_b = price_2dCOS(phi1_b, phi2_b, a1_b, b1_b, a2_b, b2_b, 256, 256, g1_b, g2_b, Ngrid=12000)

print("Q3(b) GBM + CIR, rho = 0")
print(f"  truncation domains: x1 in [{a1_b:.3f}, {b1_b:.3f}], S2 in [{a2_b:.3f}, {b2_b:.3f}]")
print(f"  converged COS price (N = 256): {ref_b:.6f}")
print(f"{'N':>4}  {'COS price':>11}  {'|error|':>10}  {'ratio':>8}")
print("-"*40)
prev = None
err_b = []
for N in N_list:
    p = price_2dCOS(phi1_b, phi2_b, a1_b, b1_b, a2_b, b2_b, N, N, g1_b, g2_b, Ngrid=8000)
    e = abs(p - ref_b); err_b.append(e)
    ratio = "" if prev is None else f"{prev/e:7.1f}x"
    print(f"{N:>4}  {p:>11.6f}  {e:>10.2e}  {ratio:>8}")
    prev = e

# Convergence figures: |error| vs N, one per case

plt.figure(figsize=(8, 5))
plt.semilogy(N_list, err_a, 'o-', color='blue')
plt.xlabel('number of cosine terms N')
plt.ylabel('|error| vs converged COS price')
plt.title('Both log-normal, $\\rho=0$ - COS convergence')
plt.grid(True, which='both', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'q3a_cos_convergence.png'), dpi=150)
plt.show()

plt.figure(figsize=(8, 5))
plt.semilogy(N_list, err_b, 's-', color='red')
plt.xlabel('number of cosine terms N')
plt.ylabel('|error| vs converged COS price')
plt.title('GBM + CIR, $\\rho=0$ - COS convergence')
plt.grid(True, which='both', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'q3b_cos_convergence.png'), dpi=150)
plt.show()
