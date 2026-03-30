import numpy as np
from core.solvers import thomas_solver

def bsa_pde_solver(S_max_user, K, T, r, sigma, S_grid, T_grid, barrier=None):
    dt = T / T_grid
    s_min = float(barrier) if barrier else 0.0

    # 1. S_max Dinamico (Protezione contro alta volatilità/tempo)
    # Calcolo S_max basato sulla deviazione standard log-normale
    s_max_auto = K * np.exp(3.5 * sigma * np.sqrt(T))
    S_max = max(S_max_user, s_max_auto)

    # Allineamento dinamico allo strike
    # Se K è sopra la barriera, allineo il nodo allo strike
    # Se K <= barriera, uso una suddivisione uniforme della griglia
    if K > s_min:
        nodes_to_k = int(S_grid * (K - s_min) / (S_max - s_min))
        nodes_to_k = max(1, nodes_to_k)
        ds = (K - s_min) / nodes_to_k
    else:
        # Se lo strike è alla barriera o sotto, ds non può essere calcolato su (K-s_min)
        ds = (S_max - s_min) / S_grid

    s_values = s_min + np.arange(0, S_grid + 1) * ds
    # Inizializzo matrice per superficie 3D (Tempo x Spazio) nel caso in cui la volessi imlementare in futuro
    v_history = np.zeros((T_grid + 1, S_grid + 1))
    
    # 2. Inizializzo Payoff
    v = np.maximum(s_values - K, 0)
    
    if barrier is not None:
        v[0] = 0
    v_history[0] = v.copy() # Stato iniziale 
    
    # 3. Coefficienti
    s_internal = s_values[1:-1]
    s_over_ds = s_internal / ds
    alpha = 0.25 * dt * (sigma**2 * s_over_ds**2 - r * s_over_ds)
    beta = -0.5 * dt * (sigma**2 * s_over_ds**2 + r)
    gamma = 0.25 * dt * (sigma**2 * s_over_ds**2 + r * s_over_ds)
    
    # Matrici base
    A_diag, A_sub, A_sup = (1 - beta).copy(), -alpha[1:].copy(), -gamma[:-1].copy()
    B_diag, B_sub, B_sup = (1 + beta).copy(), alpha[1:].copy(), gamma[:-1].copy()

    # MODIFICA LINEARITÀ AL BORDO DESTRO (d2V/dS2 = 0)
    # Sostituisco V[N] = 2*V[N-1] - V[N-2] nelle equazioni dell'ultimo nodo interno
    A_diag[-1] -= 2 * gamma[-1]
    A_sub[-1]  += gamma[-1]
    
    B_diag[-1] += 2 * gamma[-1]
    B_sub[-1]  -= gamma[-1]

    # 4. Rannacher Stepping (Eulero Implicito per stabilità iniziale)
    # Nota: Uso la matrice A modificata, ma senza il termine B
    for step in range(2):           # 2 passi Rannacher completi di ampiezza dt
        for _ in range(2):          # ogni passo = 2 mezzi-step IE di ampiezza dt/2
            d_ie = v[1:-1].copy()
            v[1:-1] = thomas_solver(A_sub, A_diag, A_sup, d_ie)
        if barrier is not None: v[0] = 0
        # Aggiorno il nodo fantasma esterno per coerenza
        v[-1] = 2 * v[-2] - v[-3]
        v_history[step + 1] = v.copy()
    

    # 5. Crank-Nicolson
    for t in range(3, T_grid + 1):
        # RHS (lato esplicito)
        d = B_diag * v[1:-1]
        d[1:] += B_sub * v[1:-2]
        d[:-1] += B_sup * v[2:-1]
        
        
        # Solver (lato implicito)
        v[1:-1] = thomas_solver(A_sub, A_diag, A_sup, d)
        
        if barrier is not None: v[0] = 0 
        
        # Aggiorno il valore al bordo destro per il prossimo step
        v[-1] = 2 * v[-2] - v[-3]
        v_history[t] = v.copy() # Salvo per ogni step temporale
        
    return s_values, v, v_history, dt