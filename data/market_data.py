import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_spx_data():
    """Recupera dati spot, chain opzioni e volatilità storica per S&P 500."""
    spx = yf.Ticker("^SPX")
    
    # 1. Selezione Scadenza (Target: ~20 giorni per stabilità)
    today = datetime.now()
    available = spx.options
    best_expiry = available[0]
    for d in available:
        if datetime.strptime(d, '%Y-%m-%d') > today + timedelta(days=20):
            best_expiry = d
            break

    T = (datetime.strptime(best_expiry, '%Y-%m-%d') - today).days / 365.0
    
    # 2. Download e Filtro
    opt_chain = spx.option_chain(best_expiry)
    calls = opt_chain.calls
    last_price = spx.history(period="1d")['Close'].iloc[-1]
    
    # Filtro liquidità
    valid = calls[(calls['impliedVolatility'] > 0.01) & (calls['lastPrice'] > 0.1)].copy()
    valid['dist'] = abs(valid['strike'] - last_price)
    
    # ATM selection
    atm_calls = valid.sort_values('dist').head(5)
    
    # 3. Volatilità Storica (Realized 30d)
    hist = spx.history(period="1mo")['Close']
    hist_vol = np.log(hist / hist.shift(1)).std() * np.sqrt(252)
    
    return last_price, atm_calls, T, hist_vol