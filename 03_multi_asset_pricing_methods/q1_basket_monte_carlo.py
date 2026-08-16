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

# Q1(a): both underlyings log-normal, exact simulation, rho = 0.30
#parameters in the physical measure

S10_a, S20_a = 100, 100
sigma1_a, sigma2_a = 0.20, 0.30
rho_a = 0.30

def simulate_basket_lognormal(M):
    """
    we simulate M paths of the basket call under the risk-neutral measure Q,
    assuming both underlyings are log-normal. We use the exact GBM solution at maturity, so no time
    discretization is needed. Returns the MC estimator and its standard error.
    """

    rng = np.random.default_rng(99)

    #we generate two independent standard normals and correlate them via Cholesky decomposition
    Z1 = rng.standard_normal(M)
    Zc = rng.standard_normal(M)
    Z2 = rho_a*Z1 + np.sqrt(1 - rho_a**2)*Zc

    #we simulate the terminal prices using the exact GBM solution under Q
    S1 = S10_a*np.exp((r - 0.5*sigma1_a**2)*T + sigma1_a*np.sqrt(T)*Z1)
    S2 = S20_a*np.exp((r - 0.5*sigma2_a**2)*T + sigma2_a*np.sqrt(T)*Z2)

    #we compute the basket value and the discounted payoff
    B = w1*S1 + w2*S2
    disc_payoff = np.exp(-r*T)*np.maximum(B - K, 0)

    #monte carlo estimator and standard error
    V_aprox = np.mean(disc_payoff)
    SE_aprox = np.std(disc_payoff, ddof=1)/np.sqrt(M)
    return V_aprox, SE_aprox


#convergence test
#the MC error decays as O(1/sqrt(M)), so SE vs M should be a line of
#slope -1/2 in log-log scale. We print a table and estimate the rate.
M_vals = [2000, 5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000]
prices_a, ses_a = [], []
print(f"{'M':>9}  {'price':>9}  {'SE':>9}  {'95% CI half-width':>18}")
print("-"*52)
for M in M_vals:
    p, se = simulate_basket_lognormal(M)
    prices_a.append(p); ses_a.append(se)
    print(f"{M:>9}  {p:>9.5f}  {se:>9.5f}  {1.96*se:>18.5f}")

x = np.log(M_vals)
y = np.log(ses_a)

#we estimate the convergence rate as the slope of log(SE) vs log(M)
#we use ordinary least-squares slope = cov(x,y)/var(x).
slope_ols = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean())**2)
print(f"\n  convergence rate (slope of log SE vs log M) = {slope_ols:.3f}  (expected -0.5)\n")

#figure: SE vs M in log-log, with a 1/sqrt(M) reference line
plt.figure(figsize=(8, 5))
plt.loglog(M_vals, ses_a, 'o-', color='blue', label='Estimated SE')
c_ref = ses_a[0]*np.sqrt(M_vals[0])
plt.loglog(M_vals, c_ref/np.sqrt(np.array(M_vals, dtype=float)), 'k--',
           label=r'$1/\sqrt{M}$ reference')
plt.xlabel('Number of paths M'); plt.ylabel('Standard error')
plt.title(' Monte Carlo convergence of the standard error')
plt.legend(); plt.grid(True, which='both', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'q1a_convergence.png'), dpi=150)
plt.show()


# Q1(b): S1 log-normal, S2 CIR (square-root) process, rho = 0.10
#parameters in the physical measure
S10_b, S20_b = 100, 100
sigma1_b = 0.25
sigma2_b = 1.5
rho_b = 0.10

#note: under Q the second underlying is a tradable asset, so its discounted
#price is a martingale and the physical CIR drift kappa(theta - S2) is replaced
#by r*S2. This is exactly the square-root process of the technical hint,
#dS2 = r*S2 dt + sigma2*sqrt(S2) dW2, for which exact simulation exists only in
#the uncorrelated case. As here rho != 0, exact joint simulation is not
#available, so we use a Euler-Maruyama scheme for S2.

def simulate_basket_cir(M, n_steps):
    """
    we simulate M paths of the basket call under Q with S1 log-normal and S2 a
    CIR process with risk-neutral drift r*S2. The increments of W1
    and W2 are correlated via Cholesky. S1 is stepped with its exact GBM solution
    and S2 with the Euler-Maruyama scheme of Lecture 4,
        X_hat(t_k) = X_hat(t_{k-1}) + mu(t_{k-1}, X_hat) h + sigma(t_{k-1}, X_hat) sqrt(h) Z_k,
    applied to dS2 = mu(t,X) dt + sigma(t,X) dW with mu(t,X) = r*X and
    sigma(t,X) = sigma2*sqrt(X). We add a full truncation X -> max(X,0) inside the
    square root (and drift), which guarantees that the square-root
    diffusion is well defined when a step would drive a path negative.
    Returns the MC estimator and its standard error.
    """
    rng = np.random.default_rng(99)
    h = T/n_steps                       # step size (the slide's h, with K*h = T)
    sq = np.sqrt(h)
    S1 = np.full(M, S10_b)
    S2 = np.full(M, S20_b)
    for _ in range(n_steps):
        #we draw the i.i.d. standard normals Z_k ~ N(0,1) of the scheme and build a
        #correlated normal for W2 via Cholesky decomposition (correlation rho with W1)
        Z1 = rng.standard_normal(M)
        Zc = rng.standard_normal(M)
        Z2 = rho_b*Z1 + np.sqrt(1 - rho_b**2)*Zc
        #S1, we do exact GBM step
        S1 = S1*np.exp((r - 0.5*sigma1_b**2)*h + sigma1_b*sq*Z1)
        #S2, we do Euler-Maruyama step with mu = r*X, sigma = sigma2*sqrt(X), full truncation
        S2p = np.maximum(S2, 0)
        S2 = S2 + r*S2p*h + sigma2_b*np.sqrt(S2p)*sq*Z2
    S2 = np.maximum(S2, 0)
    B = w1*S1 + w2*S2
    disc_payoff = np.exp(-r*T)*np.maximum(B - K, 0)
    return np.mean(disc_payoff), np.std(disc_payoff, ddof=1)/np.sqrt(M)

#we report the estimator and 95% CI at a fine grid and a large M
n_ref = 200
price_b, se_b = simulate_basket_cir(400_000, n_ref)
ci_lo_b, ci_hi_b = price_b - 1.96*se_b, price_b + 1.96*se_b
print("Q1(b) GBM + CIR, rho = 0.10  (full-truncation Euler)")
print(f"  price = {price_b:.5f},  SE = {se_b:.5f},  95% CI = [{ci_lo_b:.5f}, {ci_hi_b:.5f}]\n")

# convergence test 1, time-discretization (weak) error
#we isolate the discretization bias from the MC noise by reusing the SAME
#Brownian increments across all grids. We simulate on a fine grid and build the
#coarser grids by adding increments, so the only difference between grids is
#the discretization. We then measure the bias against a fine reference.
n_fine = 256
n_list = [4, 8, 16, 32, 64, 128]
chunks, M_chunk = 12, 60_000
acc = {n: 0.0 for n in n_list}; acc_ref = 0.0; M_total = 0

def basket_on_grid(dW1f, dW2f, n_steps, M_chunk):
    """
    we price the basket on a coarse grid of n_steps by aggregating the fine
    Brownian increments dW1f, dW2f (shape (M_chunk, n_fine)) into n_steps blocks.
    """
    k = n_fine//n_steps
    h = T/n_steps
    dW1 = dW1f.reshape(M_chunk, n_steps, k).sum(2)
    dW2 = dW2f.reshape(M_chunk, n_steps, k).sum(2)
    S1 = np.full(M_chunk, S10_b); S2 = np.full(M_chunk, S20_b)
    for i in range(n_steps):
        #the block increment dW over one coarse step equals sqrt(h) Z_k in law;
        #S1 is exact GBM and S2 is the Euler-Maruyama step (with full truncation)
        S1 = S1*np.exp((r - 0.5*sigma1_b**2)*h + sigma1_b*dW1[:, i])
        S2p = np.maximum(S2, 0); S2 = S2 + r*S2p*h + sigma2_b*np.sqrt(S2p)*dW2[:, i]
    return np.exp(-r*T)*np.maximum(w1*S1 + w2*np.maximum(S2, 0) - K, 0).sum()


rng = np.random.default_rng(99)
for _ in range(chunks):
    sq = np.sqrt(T/n_fine)
    dW1f = rng.standard_normal((M_chunk, n_fine))*sq
    dW2f = rho_b*dW1f + np.sqrt(1 - rho_b**2)*rng.standard_normal((M_chunk, n_fine))*sq
    acc_ref += basket_on_grid(dW1f, dW2f, n_fine, M_chunk)
    for n in n_list:
        acc[n] += basket_on_grid(dW1f, dW2f, n, M_chunk)
    M_total += M_chunk

ref_price = acc_ref/M_total
print("Q1(b) convergence test 1 - time-discretization weak error")
print(f"  reference price ({n_fine} steps, M = {M_total}): {ref_price:.5f}")
print(f"{'n_steps':>8}  {'price':>9}  {'|bias|':>9}")
print("-"*30)
biases = []
for n in n_list:
    p = acc[n]/M_total; biases.append(abs(p - ref_price))
    print(f"{n:>8}  {p:>9.5f}  {abs(p-ref_price):>9.5f}")
hs = np.array([T/n for n in n_list])
biases = np.array(biases)
#we fit the rate on the coarse grids where the bias dominates the noise floor
weak_order = np.polyfit(np.log(hs[:4]), np.log(biases[:4]), 1)[0]
print(f"\n  weak convergence order (slope of log|bias| vs log h) = {weak_order:.2f}  (expected 1.0)\n")

plt.figure(figsize=(8, 5))
plt.loglog(hs, biases, 's-', color='red', label='Discretization bias')
c_w = biases[0]/hs[0]
plt.loglog(hs, c_w*hs, 'k--', label=r'order-1 reference ($\propto h$)')
plt.xlabel(r'step size $h$'); plt.ylabel('|bias| vs fine reference')
plt.title('time-discretization weak convergence of the Euler scheme')
plt.legend(); plt.grid(True, which='both', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'q1b_time_convergence.png'), dpi=150)
plt.show()

#convergence test 2: simulation error
#we fix the grid (n_ref steps) and vary M to verify the O(1/sqrt(M)) decay.
M_vals_b = [5000, 10000, 25000, 50000, 100000, 200000, 400000]
prices_b, ses_b = [], []
print("convergence test 2 - simulation (statistical) error")
print(f"{'M':>9}  {'price':>9}  {'SE':>9}  {'95% CI half-width':>18}")
print("-"*52)
for M in M_vals_b:
    p, se = simulate_basket_cir(M, n_ref)
    prices_b.append(p); ses_b.append(se)
    print(f"{M:>9}  {p:>9.5f}  {se:>9.5f}  {1.96*se:>18.5f}")
slope_b = np.polyfit(np.log(M_vals_b), np.log(ses_b), 1)[0]
print(f"\n  convergence rate (slope of log SE vs log M) = {slope_b:.3f}  (expected -0.5)")

plt.figure(figsize=(8, 5))
plt.loglog(M_vals_b, ses_b, 'o-', color='green', label='Estimated SE')
c_ref_b = ses_b[0]*np.sqrt(M_vals_b[0])
plt.loglog(M_vals_b, c_ref_b/np.sqrt(np.array(M_vals_b, dtype=float)), 'k--',
           label=r'$1/\sqrt{M}$ reference')
plt.xlabel('Number of paths M'); plt.ylabel('Standard error')
plt.title(' Monte Carlo (statistical) convergence of the standard error')
plt.legend(); plt.grid(True, which='both', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'q1b_mc_convergence.png'), dpi=150)
plt.show()
