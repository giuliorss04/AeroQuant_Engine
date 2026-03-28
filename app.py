import streamlit as st
import numpy as np
import plotly.graph_objects as go
from models.black_scholes_pde import bsa_pde_solver

st.set_page_config(page_title="AeroQuant Engine", layout="wide")

st.title("AeroQuant: PDE-based European Call & Barrier Option Engine")
st.markdown("---")

# SIDEBAR CON I PARAMETRI
st.sidebar.header("Market Parameters")
S0 = st.sidebar.slider("Underlying Price ($S_0$)", 50, 150, 100)
K = st.sidebar.slider("Strike Price ($K$)", 50, 150, 100)
T = st.sidebar.slider("Time to Maturity ($T$) (years)", 0.1, 2.0, 1.0)
sigma = st.sidebar.slider(r"Volatility ($\sigma$)", 0.05, 0.8, 0.2)
r = st.sidebar.slider("Risk-Free Rate ($r$)", 0.0, 0.1, 0.05)

st.sidebar.header("Barrier Parameters")
use_barrier = st.sidebar.checkbox("Enable Barrier (Down-and-Out)")
barrier = st.sidebar.slider("Barrier Level ($H$)", 0, 100, 80) if use_barrier else None

# LOGICA DI CALCOLO
if st.button("Run PDE Simulation"):
    with st.spinner("Solving Black-Scholes PDE..."):
        
        # 1. S_max dinamico (3 volte lo strike) per evitare il Truncation Error
        S_max_dyn = K * 3 
        
        # Chiamata al solver (S_grid e T_grid alzati per maggiore precisione, anche v_history e dt per calcolare theta ed eventualmente in futuro sup. 3D)
        s_vals, prices, v_hist, dt = bsa_pde_solver(S_max_dyn, K, T, r, sigma, 500, 1000, barrier=barrier)
        
        # 2. Interpolazione lineare per trovare il prezzo esatto allo Spot S0
        # Questo elimina l'errore dell'arrotondamento al nodo più vicino
        current_p = np.interp(S0, s_vals, prices)
        
        # 3. Calcolo delle Greche (Delta, Gamma e Theta) tramite differenze finite sulla griglia
        idx = np.abs(s_vals - S0).argmin()
        if 0 < idx < len(s_vals) - 1:
            ds = s_vals[1] - s_vals[0]
            # Delta: Derivata prima (Differenza centrale)
            delta = (prices[idx+1] - prices[idx-1]) / (2 * ds)
            # Gamma: Derivata seconda
            gamma_g = (prices[idx+1] - 2*prices[idx] + prices[idx-1]) / (ds**2)
            # Theta: Differenza tra oggi (t=0) e il passo precedente (t=dt) -> inverto l'ordine per avere il decadimento (negativo)
            theta = (v_hist[-2, idx] - v_hist[-1, idx]) / dt
        else:
            delta, gamma_g, theta = 0.0, 0.0, 0.0

        # VISUALIZZAZIONE
        col1, col2 = st.columns([2, 1]) # Colonna 1 più larga per il grafico
        
        with col1:
            st.subheader("Option Value Profile")
            fig = go.Figure()
            # Mostro il grafico fino a 2*K per leggibilità, anche se calcolato su 3*K
            mask = s_vals <= K * 2
            fig.add_trace(go.Scatter(x=s_vals[mask], y=prices[mask], name="PDE Solution", line=dict(color='cyan', width=3)))
            fig.add_trace(go.Scatter(x=s_vals[mask], y=np.maximum(s_vals[mask]-K, 0), name="Payoff at Maturity", line=dict(dash='dash', color='gray')))
            fig.update_layout(template="plotly_dark", xaxis_title="Underlying Price (<i>S</i>)", yaxis_title="Option Value (<i>V</i>)")
            st.plotly_chart(fig, width="stretch")

        with col2:
            st.subheader("Risk Metrics")
            st.metric("Option Price", f"{current_p:.4f} $")
            
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Delta (Δ)", f"{delta:.4f}")
            m_col2.metric("Gamma (Γ)", f"{gamma_g:.4f}")
            m_col3.metric("Theta (Θ)", f"{theta:.4g}")
            
            st.write("---")
            st.info(f"**Technical Details:**\n- Numerical Domain Upper Bound $S_{{max}}$: {S_max_dyn}\n- Grid aligned at $K$ = {K}")
            if use_barrier:
                st.warning(f"Down-and-Out Barrier active at {barrier}")
st.markdown("---")
st.caption("AeroQuant v1.1 | Developed by Giulio Rossi")