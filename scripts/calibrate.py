import numpy as np
from models.black_scholes_pde import bsa_pde_solver
from data.market_data import get_spx_data

def find_implied_vol(market_price, S0, K, T, r, q=0.0):
    sigma = 0.25 
    S_max_local = max(S0 * 1.7, K * 1.3) #occhio ai moltiplicatori
    
    intrinsic = max(S0 - K, 0)
    if market_price <= intrinsic + 1e-4:
        return np.nan  # Prezzo non invertibile: viola il floor dell'intrinsic value
    
    for i in range(25):
        s_vals, prices, _, _ = bsa_pde_solver(S_max_local, K, T, r, sigma, 250, 500, q=q)
        current_price = np.interp(S0, s_vals, prices)
        
        diff = current_price - market_price
        if abs(diff) < 1e-4: break
        
        # Vega Numerico con interpolazione
        vol_bump = 0.005  # bump ridotto: con differenza centrata l'errore è quadratico
        s_up, p_up, _, _ = bsa_pde_solver(S_max_local, K, T, r, sigma + vol_bump, 250, 500)
        s_dn, p_dn, _, _ = bsa_pde_solver(S_max_local, K, T, r, sigma - vol_bump, 250, 500)
        vega = (np.interp(S0, s_up, p_up) - np.interp(S0, s_dn, p_dn)) / (2.0 * vol_bump)


        if abs(vega) < 1e-8: break
        sigma -= np.clip(diff / vega, -0.05, 0.05) 
        
    return max(0.01, min(sigma, 1.5))

if __name__ == "__main__":
    print("\nMARKET CALIBRATION\n")

    S0, options, T, hist_vol = get_spx_data()
    import yfinance as yf  # già importato in market_data, ma serve qui se standalone
    try:
        tbill   = yf.Ticker("^IRX")  # 13-week US T-bill yield (annualizzato, in %)
        r_fetch = tbill.history(period="5d")['Close'].dropna().iloc[-1] / 100.0
        r       = round(r_fetch, 4)
        print(f"Risk-free rate (13W T-bill): {r:.2%}")
    except Exception:
        r = 0.045  # fallback statico se fetch fallisce
        print(f"Risk-free rate (fallback statico): {r:.2%}")

    opt = options.iloc[0]
    mid_price = (opt['bid'] + opt['ask']) / 2
    q_spx = 0.013  # dividend yield continuo SPX (~1.3% storico, aggiornare se necessario)
    my_iv = find_implied_vol(mid_price, S0, opt['strike'], T, r, q=q_spx)

    premium = my_iv - hist_vol
    # Soglia ±4%: threshold empirica sul Volatility Risk Premium storico SPX
    status = "RICH" if premium > 0.04 else "CHEAP" if premium < -0.04 else "FAIR VALUE"

    print(f"Underlying:        S&P 500 (^SPX)")
    print(f"Price:             $ {S0:.2f}")
    print(f"Maturity:          {int(T*365)} days")
    print(f"Historical vol:    {hist_vol:.2%}")
    print("-" * 40)

    print(f"Strike:            $ {opt['strike']}")
    print(f"Market price:      $ {mid_price:.2f}  (mid)")
    print("-" * 40)

    print(f"Model IV:          {my_iv:.2%}")
    print(f"Yahoo IV:          {opt['impliedVolatility']:.2%}")
    print(f"Error:              {abs(my_iv - opt['impliedVolatility']):.2%}")
    print("-" * 40)

    print(f"Market view:       {status}")
    print(f"Risk premium:      {premium:+.2%}")

    print()