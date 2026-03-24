import numpy as np
import matplotlib.pyplot as plt

def heston_mc_simulation(S0, K, T, r, kappa, theta, sigma, rho, v0, steps, n_paths):
    """
    Simulazione Monte Carlo del modello di Heston usando RK4 + Milstein.
    Include Antithetic Variates per la riduzione della varianza.
    """
    dt = T / steps
    
    # Inizializzazione matrici (n_paths * 2 perché usiamo Antithetic Variates)
    # Generiamo percorsi speculari per convergere più velocemente
    total_paths = n_paths * 2
    S = np.zeros((steps + 1, total_paths))
    v = np.zeros((steps + 1, total_paths))
    
    S[0] = S0
    v[0] = v0

    for t in range(1, steps + 1):
        # Generazione shock stocastici correlati
        # W1 per il prezzo, W2 per la varianza (correlazione rho)
        z1 = np.random.normal(0, 1, n_paths)
        z2 = np.random.normal(0, 1, n_paths)
        
        # Antithetic Variates: creiamo gli shock opposti
        z1 = np.concatenate([z1, -z1])
        z2 = np.concatenate([z2, -z2])
        
        # Shock correlati: W2 = rho*W1 + sqrt(1-rho^2)*W_indipendente
        dW1 = z1 * np.sqrt(dt)
        dW2 = (rho * z1 + np.sqrt(1 - rho**2) * z2) * np.sqrt(dt)

        # 1. RK4 per la parte deterministica della varianza: kappa*(theta - v)
        # k1, k2, k3, k4 sono i "tentativi" di pendenza tipici di Runge-Kutta
        def drift_v(curr_v): return kappa * (theta - curr_v)
        
        k1 = drift_v(v[t-1])
        k2 = drift_v(v[t-1] + 0.5 * dt * k1)
        k3 = drift_v(v[t-1] + 0.5 * dt * k2)
        k4 = drift_v(v[t-1] + dt * k3)
        
        v_deterministic = (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # 2. Aggiornamento Varianza (Milstein per la parte stocastica)
        # Usiamo np.maximum(v, 0) per evitare che la varianza diventi negativa (problema comune)
        v[t] = v[t-1] + v_deterministic + sigma * np.sqrt(np.maximum(v[t-1], 0)) * dW2 + 0.25 * sigma**2 * (dW2**2 - dt)
        v[t] = np.maximum(v[t], 0.0001) # Floor di sicurezza

        # 3. Aggiornamento Prezzo (Euler-Maruyama)
        S[t] = S[t-1] * np.exp((r - 0.5 * v[t-1]) * dt + np.sqrt(v[t-1]) * dW1)

    return S, v

def plot_spaghetti(S, n_to_plot=10):
    """Crea il famoso 'Spaghetti Plot' per il README."""
    plt.figure(figsize=(10, 6))
    plt.plot(S[:, :n_to_plot])
    plt.title(f"Heston Model: Prime {n_to_plot} traiettorie (Spaghetti Plot)")
    plt.xlabel("Time Steps")
    plt.ylabel("Asset Price S(t)")
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    # Parametri di test (Tipici per l'S&P 500)
    S0, K, T, r = 100, 100, 1, 0.03
    kappa, theta, sigma, rho, v0 = 2.0, 0.04, 0.3, -0.7, 0.04
    
    S_paths, v_paths = heston_mc_simulation(S0, K, T, r, kappa, theta, sigma, rho, v0, 252, 1000)
    
    # Calcolo prezzo opzione Call
    payoffs = np.maximum(S_paths[-1] - K, 0)
    option_price = np.exp(-r * T) * np.mean(payoffs)
    
    print(f"Prezzo Opzione Heston (Monte Carlo): {option_price:.4f}")
    plot_spaghetti(S_paths)