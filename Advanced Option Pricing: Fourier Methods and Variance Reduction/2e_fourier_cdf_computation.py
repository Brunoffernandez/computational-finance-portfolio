import numpy as np
from scipy.stats import norm

#parameters
r = 0.03
S0 = 3
sigma = 0.3
T = 1

#we know that the distribution of ln(st/s0) is ~ N(mu_x, sigma_x^2)
mu = (r - 0.5*sigma**2)*T
sigma_aux = sigma * np.sqrt(T)  

a = -2
b = 2
X_test =  np.linspace(-0.5, 0.5, 11)
N_vals = [4,8,16,32,64,128]

def char_function(w):
    """
    we compute the characteristic function of  ln(st/s0)
    """
    return np.exp(1j*w*mu - 0.5*(sigma_aux**2)*w**2)

def F_exact(z):
    """
    we compute the analytical cdf for ln(st/s0)
    """
    return norm.cdf((z - mu) / sigma_aux)

def sin_cdf(z_vals, N, a, b):
    """
    we compute the sin cdf at each point in the z_vals
    """
    
    ks = np.arange(1,N+1)
    wk = ks * np.pi / (b-a)
    
    char = char_function(wk)
    im= np.imag(char* np.exp(-1j * wk * a)) # Im{ phi(omega_k) * exp(-i*omega_k*a) }
    
    z_vals  = np.asarray(z_vals)
    cosine = np.cos(np.outer((z_vals - a) / (b-a), ks) * np.pi) #we employ outer because we need to multiply vectors
    
    # F(z_i) = (2/pi) * sum_k (1/k) * Im_k * (1 - cos_mat[i,k])
    return (2.0/np.pi) * ((im / ks) @ (1.0 - cosine).T)


F_true = F_exact(X_test)


print(f"{'N':>5}  {'Max |CDF error|':>18}")
print("-"*26)
for N in N_vals:
    F_sin = sin_cdf(X_test, N, a, b)
    err   = np.max(np.abs(F_sin - F_true))
    print(f"{N:>5}  {err:>18.2e}")





