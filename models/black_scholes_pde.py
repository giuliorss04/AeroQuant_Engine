import numpy as np
from core.solvers import thomas_solver

def bsa_pde_solver(S_max, K, T, r, sigma, S_grid, T_grid, barrier=None):
    dt = T / T_grid
    
    # TRUCCO TECNICO: Allineamento della griglia
    # Facciamo in modo che ds sia tale che K sia un multiplo esatto di ds
    # Questo ripristina la convergenza del secondo ordine O(dt^2, ds^2)
    nodes_to_strike = int(S_grid * (K / S_max))
    ds = K / nodes_to_strike
    s_values = np.arange(0, S_grid + 1) * ds
    # Aggiorniamo S_max effettivo per coerenza
    S_max_eff = s_values[-1] 

    v = np.maximum(s_values - K, 0)
    if barrier:
        v[s_values <= barrier] = 0

    indices = np.arange(1, S_grid)
    # Calcolo coefficienti (Crank-Nicolson)
    alpha = 0.25 * dt * (sigma**2 * indices**2 - r * indices)
    beta = -0.5 * dt * (sigma**2 * indices**2 + r)
    gamma = 0.25 * dt * (sigma**2 * indices**2 + r * indices)
    
    A_diag = 1 - beta
    A_sub = -alpha[1:]
    A_sup = -gamma[:-1]
    
    B_diag = 1 + beta
    B_sub = alpha[1:]
    B_sup = gamma[:-1]

    for t in range(T_grid):
        d = B_diag * v[1:-1]
        d[1:] += B_sub * v[1:-2]
        d[:-1] += B_sup * v[2:-1]
        
        # Condizione al contorno a destra
        d[-1] += gamma[-1] * (S_max_eff - K * np.exp(-r * (t * dt)))

        v[1:-1] = thomas_solver(A_sub, A_diag, A_sup, d)
        
        if barrier:
            v[s_values <= barrier] = 0
            
    return s_values, v

# Test rapido
if __name__ == "__main__":
    s, prices = bsa_pde_solver(200, 100, 1, 0.05, 0.2, 100, 1000, barrier=80)
    print("Prezzo al centro (S=100):", prices[50])