import numpy as np
from models.black_scholes_pde import bsa_pde_solver
from data.market_data import get_spx_data

def find_implied_vol(market_price, S0, K, T, r):
    """
    Trova la Volatilità Implicita usando Newton-Raphson con 
    interpolazione lineare e damping per la massima stabilità.
    """
    sigma = 0.25  # Punto di partenza neutro (25%)
    # Griglia locale: riduce l'errore di discretizzazione concentrando i nodi
    S_max_local = max(S0 * 1.5, K * 1.2) 
    
    for i in range(25):
        # 1. Calcolo Prezzo attuale con la PDE
        # Usiamo una griglia densa 250x500 per bilanciare velocità e precisione
        s_vals, prices = bsa_pde_solver(S_max_local, K, T, r, sigma, 250, 500)
        
        # INTERPOLAZIONE: Fondamentale per evitare salti numerici
        current_price = np.interp(S0, s_vals, prices)
        
        diff = current_price - market_price
        if abs(diff) < 1e-4: # Convergenza raggiunta
            break
        
        # 2. Calcolo Vega Numerico (Derivata del prezzo rispetto a sigma)
        s_vals2, prices2 = bsa_pde_solver(S_max_local, K, T, r, sigma + 0.001, 250, 500)
        price_plus = np.interp(S0, s_vals2, prices2)
        vega = (price_plus - current_price) / 0.001
        
        if abs(vega) < 1e-8: # Evita divisioni per zero in zone "piatte"
            break
        
        # 3. Aggiornamento con Damping (Dampened Newton-Raphson)
        shift = diff / vega
        # Limitiamo lo spostamento per evitare che sigma diventi negativa o esploda
        sigma = sigma - np.clip(shift, -0.08, 0.08)
        
    return max(0.01, min(sigma, 2.0)) # Vincolo di sicurezza: 1% - 200%

if __name__ == "__main__":
    # Carichiamo i dati (Assicurati che market_data restituisca 4 valori!)
    S0, options, T, hist_vol = get_spx_data()
    r = 0.045  # Tasso Risk-free (4.5%)
    
    print(f"\n" + "="*45)
    print(f"       AEROQUANT ENGINE: CALIBRAZIONE")
    print(f"="*45)
    print(f"S&P 500 Spot: {S0:.2f}")
    print(f"Scadenza:     {T*365:.1f} giorni (T={T:.4f})")
    print(f"Vol. Storica: {hist_vol:.2%}")
    print("-" * 45)

    # Analizziamo la prima opzione della lista (solitamente la più ATM)
    sample_opt = options.iloc[0]
    market_p = sample_opt['lastPrice']
    strike = sample_opt['strike']
    yahoo_iv = sample_opt['impliedVolatility']
    
    print(f"ANALISI OPZIONE CALL @ STRIKE {strike}:")
    print(f"Prezzo di Mercato: {market_p:.2f} $")
    
    # Esecuzione del calcolo
    my_iv = find_implied_vol(market_p, S0, strike, T, r)
    
    print(f"\nRISULTATI:")
    print(f"IV AeroQuant (PDE): {my_iv:.2%}")
    print(f"IV Yahoo Finance:   {yahoo_iv:.2%}")
    
    # Calcolo dell'errore e del Volatility Premium
    error = abs(my_iv - yahoo_iv)
    premium = my_iv - hist_vol
    
    print(f"Errore vs Yahoo:    {error:.2%}")
    print(f"Vol. Premium:       {premium:.2%} (Implicita vs Storica)")
    print("="*45)