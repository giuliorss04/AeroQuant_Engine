import numpy as np
from models.black_scholes_pde import bsa_pde_solver
from data.market_data import get_spx_data

def find_implied_vol(market_price, S0, K, T, r):
    sigma = 0.25 
    S_max_local = max(S0 * 1.7, K * 1.3) # L'ho allargato per sicurezza, l'errore scende
    
    for i in range(25):
        s_vals, prices, _, _ = bsa_pde_solver(S_max_local, K, T, r, sigma, 250, 500)
        current_price = np.interp(S0, s_vals, prices)
        
        diff = current_price - market_price
        if abs(diff) < 1e-4: break
        
        # Vega Numerico con interpolazione
        s_vals2, prices2, _, _ = bsa_pde_solver(S_max_local, K, T, r, sigma + 0.001, 250, 500)
        vega = (np.interp(S0, s_vals2, prices2) - current_price) / 0.001
        
        if abs(vega) < 1e-8: break
        sigma -= np.clip(diff / vega, -0.05, 0.05) 
        
    return max(0.01, min(sigma, 1.5))

if __name__ == "__main__":
    print("\nMARKET CALIBRATION\n")

    S0, options, T, hist_vol = get_spx_data()
    r = 0.045

    opt = options.iloc[0]
    my_iv = find_implied_vol(opt['lastPrice'], S0, opt['strike'], T, r)

    premium = my_iv - hist_vol
    status = "RICH" if premium > 0.04 else "CHEAP" if premium < -0.04 else "FAIR VALUE"

    print(f"Underlying:        S&P 500 (^SPX)")
    print(f"Price:             $ {S0:.2f}")
    print(f"Maturity:          {int(T*365)} days")
    print(f"Historical vol:    {hist_vol:.2%}")
    print("-" * 40)

    print(f"Strike:            $ {opt['strike']}")
    print(f"Market price:      $ {opt['lastPrice']:.2f}")
    print("-" * 40)

    print(f"Model IV:          {my_iv:.2%}")
    print(f"Yahoo IV:          {opt['impliedVolatility']:.2%}")
    print(f"Error:              {abs(my_iv - opt['impliedVolatility']):.2%}")
    print("-" * 40)

    print(f"Market view:       {status}")
    print(f"Risk premium:      {premium:+.2%}")

    print()