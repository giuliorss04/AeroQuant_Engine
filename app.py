import streamlit as st
import numpy as np
import plotly.graph_objects as go
from models.black_scholes_pde import bsa_pde_solver

st.set_page_config(page_title="AeroQuant Engine", layout="wide")

st.title("🦅 AeroQuant Engine: PDE Option Pricer")
st.markdown("---")

# Sidebar per i parametri
st.sidebar.header("Parametri di Mercato")
S0 = st.sidebar.slider("Prezzo Asset (S0)", 50, 150, 100)
K = st.sidebar.slider("Strike Price (K)", 50, 150, 100)
T = st.sidebar.slider("Scadenza (Anni)", 0.1, 2.0, 1.0)
sigma = st.sidebar.slider("Volatilità (σ)", 0.05, 0.8, 0.2)
r = st.sidebar.slider("Tasso Risk-free", 0.0, 0.1, 0.05)

st.sidebar.header("Parametri Barriera")
use_barrier = st.sidebar.checkbox("Attiva Barriera (Down-and-Out)")
barrier = st.sidebar.slider("Livello Barriera", 0, 100, 80) if use_barrier else None

# Calcolo
if st.button("Lancia Simulazione PDE"):
    with st.spinner("Risolvendo le equazioni di Black-Scholes..."):
        s_vals, prices = bsa_pde_solver(200, K, T, r, sigma, 100, 500, barrier=barrier)
        
        # Plot Risultati
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Profilo di Prezzo")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s_vals, y=prices, name="Prezzo PDE", line=dict(color='cyan', width=3)))
            fig.add_trace(go.Scatter(x=s_vals, y=np.maximum(s_vals-K, 0), name="Payoff", line=dict(dash='dash', color='gray')))
            fig.update_layout(template="plotly_dark", xaxis_title="Prezzo S", yaxis_title="Valore Opzione")
            st.plotly_chart(fig)

        with col2:
            st.subheader("Analisi Metriche")
            idx = np.abs(s_vals - S0).argmin()
            current_p = prices[idx]
            st.metric("Prezzo Calcolato", f"{current_p:.4f} €")
            st.info(f"Il modello sta usando uno schema Crank-Nicolson con allineamento della griglia su K={K}")

st.markdown("---")
st.caption("Sviluppato per la candidatura EPFL - AeroQuant Engine v1.0")