import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from models.black_scholes_pde import bsa_pde_solver

def black_scholes_exact(S, K, T, r, sigma):
    """Soluzione analitica esatta di Black-Scholes per una Call."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def run_convergence_study():
    # Parametri fissi
    S0, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    S_max = 300
    
    # Griglie crescenti (raddoppio progressivo: 20 × 2^k, k = 0..5)
    grids = [20, 40, 80, 160, 320, 640]
    errors = []
    
    exact_price = black_scholes_exact(S0, K, T, r, sigma)
    print(f"Exact Price (Analytical): {exact_price:.6f}")

    for N in grids:
        # Risolviamo con la nostra PDE (senza barriera per il confronto con BS)
        s_vals, prices, _, _ = bsa_pde_solver(S_max, K, T, r, sigma, S_grid=N, T_grid=N*2)
        
        # Troviamo il prezzo corrispondente a S0=100
        pde_price = np.interp(S0, s_vals, prices)
        
        err = np.abs(pde_price - exact_price)
        errors.append(err)
        if len(errors) > 1:
            rate = np.log(errors[-2] / err) / np.log(N / grids[len(errors) - 2])
        else:
            rate = float('nan')
        print(f"Grid {N:>4}x{N*2:<5} | Error: {err:.2e} | Conv. rate: {rate:.2f}")

    # Plot Log-Log
    plt.figure(figsize=(8, 6))
    plt.loglog(grids, errors, 'o-', label='PDE Error (Crank-Nicolson)')
    
    # Linea di riferimento O(N^-2) - pendenza -2
    plt.loglog(grids, [errors[0]*(grids[0]/n)**2 for n in grids], '--', label='Theoretical Slope -2')
    
    plt.title("Convergence Analysis")
    plt.xlabel("Number of Grid Points (N)")
    plt.ylabel("Absolute Error")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.show()

if __name__ == "__main__":
    run_convergence_study()