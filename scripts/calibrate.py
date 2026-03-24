import numpy as np
from models.black_scholes_pde import bsa_pde_solver
from data.market_data import get_spx_data

def find_implied_vol(market_price, S0, K, T, r):
    # Newton-Raphson semplificato (Metodo delle Secanti)
    sigma = 0.2 # Ipotesi iniziale (20%)
    for i in range(15):
        # Calcoliamo il prezzo con la nostra PDE
        s_vals, prices = bsa_pde_solver(S0*3, K, T, r, sigma, 100, 200)
        idx = np.abs(s_vals - S0).argmin()
        diff = prices[idx] - market_price
        
        if abs(diff) < 0.001: break
        
        # Piccola perturbazione per calcolare la derivata (Vega)
        s_vals2, prices2 = bsa_pde_solver(S0*3, K, T, r, sigma + 0.01, 100, 200)
        vega = (prices2[idx] - prices[idx]) / 0.01
        sigma = sigma - diff / vega
        
    return sigma

if __name__ == "__main__":
    S0, options = get_spx_data()
    # Prendiamo la prima opzione della lista
    sample_opt = options.iloc[0]
    
    print(f"\nCalibrando su Strike {sample_opt['strike']}...")
    iv = find_implied_vol(sample_opt['lastPrice'], S0, sample_opt['strike'], 0.1, 0.04)
    print(f"Volatilità Implicita calcolata dalla tua PDE: {iv:.2%}")
    print(f"Volatilità riportata da Yahoo: {sample_opt['impliedVolatility']:.2%}")