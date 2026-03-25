import numpy as np
from core.solvers import thomas_solver

def bsa_pde_solver(S_max, K, T, r, sigma, S_grid, T_grid, barrier=None):
    dt = T / T_grid
    s_min = float(barrier) if barrier else 0.0

    # 1. Allineamento della griglia (Ordine 2)
    nodes_to_k = int(S_grid * (K - s_min) / (S_max - s_min))
    nodes_to_k = max(1, nodes_to_k)
    ds = (K - s_min) / nodes_to_k
    s_values = s_min + np.arange(0, S_grid + 1) * ds
    S_max_eff = s_values[-1]

    # 2. Inizializzazione Payoff
    v = np.maximum(s_values - K, 0)
    
    # 3. Coefficienti Crank-Nicolson
    s_over_ds = s_values[1:-1] / ds
    alpha = 0.25 * dt * (sigma**2 * s_over_ds**2 - r * s_over_ds)
    beta = -0.5 * dt * (sigma**2 * s_over_ds**2 + r)
    gamma = 0.25 * dt * (sigma**2 * s_over_ds**2 + r * s_over_ds)
    
    A_diag, A_sub, A_sup = 1 - beta, -alpha[1:], -gamma[:-1]
    B_diag, B_sub, B_sup = 1 + beta, alpha[1:], gamma[:-1]

    # --- 4. RANNACHER STEPPING (Primi 2 sotto-step di Eulero Implicito) ---
    # Sostituiamo il primo step di CN con due step di IE da dt/2
    for step in range(2):
        t_ie = (step + 1) * (dt / 2)
        # Per Eulero Implicito (dt/2), il RHS è semplicemente il valore precedente v
        d_ie = v[1:-1].copy()
        # Condizione al contorno (usiamo 2*gamma perché IE dt/2 richiede il doppio del peso di CN)
        boundary_val = S_max_eff - K * np.exp(-r * t_ie)
        d_ie[-1] += 2 * gamma[-1] * boundary_val
        
        v[1:-1] = thomas_solver(A_sub, A_diag, A_sup, d_ie)

    # --- 5. CICLO CRANK-NICOLSON (I restanti T_grid - 1 step) ---
    for t in range(1, T_grid):
        d = B_diag * v[1:-1]
        d[1:] += B_sub * v[1:-2]
        d[:-1] += B_sup * v[2:-1]
        
        t_mid = (t + 0.5) * dt
        d[-1] += gamma[-1] * (S_max_eff - K * np.exp(-r * t_mid))

        v[1:-1] = thomas_solver(A_sub, A_diag, A_sup, d)
        
    return s_values, v