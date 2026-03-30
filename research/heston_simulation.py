import numpy as np
import matplotlib.pyplot as plt

def heston_mc_simulation(S0, K, T, r, kappa, theta, sigma, rho, v0, steps, n_paths):
    """
    Simulazione Monte Carlo del modello di Heston usando RK4 + Milstein.
    Include Antithetic Variates per la riduzione della varianza.
    """
    dt = T / steps
    
    # Inizializzazione matrici (n_paths * 2 perché uso Antithetic Variates)
    # Genero percorsi speculari per convergere più velocemente
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
        
        # Antithetic Variates: creo gli shock opposti
        z1 = np.concatenate([z1, -z1])
        z2 = np.concatenate([z2, -z2])
        
        # Shock correlati: W2 = rho*W1 + sqrt(1-rho^2)*W_indipendente
        dW1 = z1 * np.sqrt(dt)
        dW2 = (rho * z1 + np.sqrt(1 - rho**2) * z2) * np.sqrt(dt)

        # 1. RK4 per la parte deterministica della varianza: kappa*(theta - v)
        # Schema ibrido: RK4 per il drift deterministico, Milstein per il termine stocastico.
        # Nota: l'ordine globale dello schema è comunque determinato dal termine Milstein (O(dt)).
        # Il vantaggio di RK4 sul drift lineare CIR è marginale ma non scorretto.
        def drift_v(curr_v): return kappa * (theta - curr_v)
        
        k1 = drift_v(v[t-1])
        k2 = drift_v(v[t-1] + 0.5 * dt * k1)
        k3 = drift_v(v[t-1] + 0.5 * dt * k2)
        k4 = drift_v(v[t-1] + dt * k3)
        
        v_deterministic = (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # 2. Aggiornamento Varianza (Milstein per la parte stocastica)
        # Usiamo np.maximum(v, 0) per evitare che la varianza diventi negativa
        v[t] = v[t-1] + v_deterministic + sigma * np.sqrt(np.maximum(v[t-1], 0)) * dW2 + 0.25 * sigma**2 * (dW2**2 - dt)
        v[t] = np.maximum(v[t], 0.0001) # Floor di sicurezza

        # 3. Aggiornamento Prezzo (Euler-Maruyama)
        S[t] = S[t-1] * np.exp((r - 0.5 * v[t-1]) * dt + np.sqrt(v[t-1]) * dW1)

    return S, v

def plot_spaghetti(S, n_to_plot=50):
    """Crea lo 'Spaghetti Plot'"""
    plt.figure(figsize=(10, 6))
    plt.plot(S[:, :n_to_plot])
    plt.title(f"Heston Model – First {n_to_plot} Simulated Asset Paths")
    plt.xlabel("Time Steps")
    plt.ylabel("Asset Price S(t)")
    plt.grid(True, alpha=0.3)
    plt.show()

def check_mc_convergence():
    n_list = [100, 500, 1000, 5000, 10000, 50000, 100000]
    prices = []
    ci_widths = []

    print("\nMonte Carlo Convergence Test:")
    for n in n_list:
        S_paths, _ = heston_mc_simulation(100, 100, 1, 0.03, 2.0, 0.04, 0.3, -0.7, 0.04, 252, n)
        payoffs = np.maximum(S_paths[-1] - 100, 0)
        disc_payoffs = np.exp(-0.03 * 1) * payoffs

        p = np.mean(disc_payoffs)
        se = np.std(disc_payoffs) / np.sqrt(len(disc_payoffs))  # Standard error
        ci = 1.96 * se                                          # IC 95%

        prices.append(p)
        ci_widths.append(ci)
        print(f"Paths: {n*2:<6} | Price: {p:.4f} | 95% CI: [{p-ci:.4f}, {p+ci:.4f}]")

    total_paths = [n * 2 for n in n_list]
    plt.figure()
    plt.errorbar(total_paths, prices, yerr=ci_widths, fmt='o-', capsize=4, label='MC Estimate ± 95% CI')
    plt.axhline(y=prices[-1], color='r', linestyle='--', label='Final Estimate')
    plt.xlabel("Total Number of Paths (with Antithetic Variates)")
    plt.ylabel("Option Price")
    plt.title("Monte Carlo Convergence (Heston Model)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)  # Seed fisso per riproducibilità
    # Parametri di test (Tipici per l'S&P 500)
    S0, K, T, r = 100, 100, 1, 0.03
    kappa, theta, sigma, rho, v0 = 2.0, 0.04, 0.3, -0.7, 0.04
    
    S_paths, v_paths = heston_mc_simulation(S0, K, T, r, kappa, theta, sigma, rho, v0, 252, 1000)
    
    # Calcolo prezzo opzione Call
    payoffs = np.maximum(S_paths[-1] - K, 0)
    option_price = np.exp(-r * T) * np.mean(payoffs)
    
    print(f"Heston Option Price (Monte Carlo): {option_price:.4f}")
    plot_spaghetti(S_paths)
    check_mc_convergence()
    