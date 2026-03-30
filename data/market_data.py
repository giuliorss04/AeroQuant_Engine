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
    target = today + timedelta(days=20)
    best_expiry = next(
        (d for d in available if datetime.strptime(d, '%Y-%m-%d') > target),
        available[-1]  # fallback: scadenza più lontana disponibile
    )

    T = (datetime.strptime(best_expiry, '%Y-%m-%d') - today).days / 365.0
    
    # 2. Download e Filtro
    opt_chain = spx.option_chain(best_expiry)
    calls = opt_chain.calls
    last_price = spx.history(period="5d")['Close'].dropna().iloc[-1]
    
    # Filtro liquidità
    valid = calls[(calls['impliedVolatility'] > 0.01) & (calls['lastPrice'] > 0.1)].copy()
    valid['dist'] = abs(valid['strike'] - last_price)
    
    # Validazione bid/ask (necessaria per mid-price in calibrate.py)
    valid = valid.dropna(subset=['bid', 'ask'])
    valid = valid[valid['bid'] > 0]  # bid = 0 indica opzione illiquida

    # ATM selection
    atm_calls = valid.sort_values('dist').head(5)
    
    # 3. Volatilità Storica (Realized 30d)
    hist = spx.history(period="45d")['Close']  # buffer per garantire 30 giorni di trading
    log_returns = np.log(hist / hist.shift(1)).dropna()
    hist_vol = log_returns.iloc[-30:].std() * np.sqrt(252)  # ultimi 30 ritorni esatti
    
    return last_price, atm_calls, T, hist_vol