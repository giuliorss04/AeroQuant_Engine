import numpy as np
from core.solvers import thomas_solver

def bsa_pde_solver(S_max, K, T, r, sigma, S_grid, T_grid, barrier=None):
    """
    Risolve la PDE di Black-Scholes per un'opzione (Call) con eventuale barriera.Implementa C-N

    """
    dt = T / T_grid
    ds = S_max / S_grid
    s_values = np.linspace(0, S_max, S_grid + 1)
    
    # Condizione finale (Payoff Call: max(S-K, 0))
    v = np.maximum(s_values - K, 0)
    
    # Applica barriera se presente (Down-and-Out)
    if barrier:
        v[s_values <= barrier] = 0

    # Coefficienti della matrice
    def get_coeffs(j):
        alpha = 0.25 * dt * (sigma**2 * j**2 - r * j)
        beta = -0.5 * dt * (sigma**2 * j**2 + r)
        gamma = 0.25 * dt * (sigma**2 * j**2 + r * j)
        return alpha, beta, gamma

    # Prepariamo le matrici per Crank-Nicolson
    # A * V_nuovo = B * V_vecchio
    indices = np.arange(1, S_grid)
    alpha, beta, gamma = get_coeffs(indices)
    
    # Matrice A (implicita)
    A_diag = 1 - beta
    A_sub = -alpha[1:]
    A_sup = -gamma[:-1]
    
    # Matrice B (esplicita)
    B_diag = 1 + beta
    B_sub = alpha[1:]
    B_sup = gamma[:-1]

    # Time Stepping
    for t in range(T_grid):
        # Calcolo termine noto (B * V_vecchio)
        d = B_diag * v[1:-1]
        d[1:] += B_sub * v[1:-2]
        d[:-1] += B_sup * v[2:-1]
        
        # Condizioni al contorno (Dirichlet)
        # S=0 -> V=0; S=S_max -> V = S_max - K*exp(-r*t)
        d[-1] += gamma[-1] * (S_max - K * np.exp(-r * (t * dt)))

        # Risoluzione sistema tridiagonale
        v[1:-1] = thomas_solver(A_sub, A_diag, A_sup, d)
        
        # Riapplica barriera ad ogni step temporale
        if barrier:
            v[s_values <= barrier] = 0
            
    return s_values, v

# Test rapido
if __name__ == "__main__":
    s, prices = bsa_pde_solver(200, 100, 1, 0.05, 0.2, 100, 1000, barrier=80)
    print("Prezzo al centro (S=100):", prices[50])