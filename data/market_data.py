import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_spx_data():
    print("Scaricando dati S&P 500...")
    spx = yf.Ticker("^SPX")
    
    # 1. Cerchiamo una scadenza "liquida" (almeno tra 20 e 60 giorni)
    today = datetime.now()
    target_date = today + timedelta(days=30)
    
    available_expiries = spx.options
    # Selezioniamo la scadenza più vicina a 30 giorni da oggi
    best_expiry = available_expiries[0]
    for date_str in available_expiries:
        expiry_dt = datetime.strptime(date_str, '%Y-%m-%d')
        if expiry_dt > target_date:
            best_expiry = date_str
            break

    expiry_date = datetime.strptime(best_expiry, '%Y-%m-%d')
    T = (expiry_date - today).days / 365.0
    
    # 2. Scarichiamo la catena di opzioni
    opt_chain = spx.option_chain(best_expiry)
    calls = opt_chain.calls
    
    # 3. Pulizia dati: filtriamo prezzi nulli e volumi bassi
    last_price = spx.history(period="5d")['Close'].iloc[-1]
    valid_calls = calls[(calls['impliedVolatility'] > 0.01) & (calls['lastPrice'] > 0.1)].copy()
    
    # Prendiamo le 5 opzioni più vicine al prezzo attuale (At-The-Money)
    valid_calls['dist'] = abs(valid_calls['strike'] - last_price)
    atm_calls = valid_calls.sort_values('dist').head(5)
    
    # 4. Calcoliamo anche la VOLATILITÀ STORICA (Realized Volatility)
    # Questo serve come benchmark per capire se il mercato è "caro" o "economico"
    hist = spx.history(period="1mo")['Close']
    returns = np.log(hist / hist.shift(1))
    hist_vol = returns.std() * np.sqrt(252) # Annualizzata
    
    return last_price, atm_calls, T, hist_vol

import numpy as np # Aggiunto per il calcolo log