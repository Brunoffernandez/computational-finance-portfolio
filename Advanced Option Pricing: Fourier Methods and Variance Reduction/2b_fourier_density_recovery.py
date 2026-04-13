import numpy as np

#parameters of the paper
a = -10
b = 10
X_test = np.arange(-5,6)
N_vals = [4,8,16,32,64]


def normal_density(x):
    #we compute the normal density from the paper
    return (1/np.sqrt(2*np.pi))*np.exp(-0.5*x**2)

def char_function(w):
    #we compute the characteristic function of the normal dist. from the paper
    return np.exp(-0.5 * w**2)

def sin_density(x_vals, a, b, N):
    
    f_recovered = np.zeros_like(x_vals, dtype=float) #we need the same dimension as X_test vector 
    
    k = np.arange(1,N+1) #k goes from 1 to N
    wk = k*np.pi/(b-a) #vector which contains the value w for each k
    
    char = char_function(wk)
    fk = (2.0 / (b - a)) * np.imag(char* np.exp(-1j * wk * a)) #we compute fk
    
    # we evaluate the sum for each x in x_vals
    for i, x in enumerate(x_vals):
        sin_terms = np.sin(k * np.pi * (x - a) / (b - a))
        f_recovered[i] = np.sum(fk * sin_terms)
        
    return f_recovered
    
    

print(f"{'N':<5} | {'Max Absolute Error'}")
print("-" * 25)
for N in N_vals:
    f_approx = sin_density(X_test, a, b, N)
    f_true = normal_density(X_test)
    max_err = np.max(np.abs(f_approx - f_true))
    print(f"{N:<5} | {max_err:.3e}")