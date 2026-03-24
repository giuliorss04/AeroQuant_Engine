import yfinance as yf
import pandas as pd
from scipy.optimize import newton
from models.black_scholes_pde import bsa_pde_solver

def get_spx_data():
    print("Scaricando dati S&P 500...")
    spx = yf.Ticker("^SPX")
    # Prendiamo la prima scadenza disponibile
    expiries = spx.options
    opt_chain = spx.option_chain(expiries[0])
    calls = opt_chain.calls
    
    # Filtriamo per opzioni vicine al prezzo attuale (At-The-Money)
    last_price = spx.history(period="1d")['Close'].iloc[-1]
    calls = calls[(calls['strike'] > last_price * 0.9) & (calls['strike'] < last_price * 1.1)]
    
    return last_price, calls.head(5)

if __name__ == "__main__":
    S, data = get_spx_data()
    print(f"\nPrezzo S&P 500 attuale: {S:.2f}")
    print("Prime 5 opzioni Call trovate:")
    print(data[['strike', 'lastPrice', 'volatility']])