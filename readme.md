# AeroQuant Engine 🦅
**High-Performance Option Pricing Suite: PDE Solvers & Stochastic Dynamics**

## Project Overview
This engine is a hybrid numerical framework designed to price exotic options (Barrier Options) using both **Finite Difference Methods (CFD)** and **Monte Carlo Simulations**. Developed as a technical showcase for quantitative finance applications.

## Key Features
- **PDE Engine:** Implements the Black-Scholes PDE using a **Crank-Nicolson** scheme.
- **Numerical Stability:** Solved via the **Thomas Algorithm** (Tridiagonal Matrix Solver) in $O(n)$ time.
- **Stochastic Engine:** Heston Model simulation using **Runge-Kutta 4 (RK4)** for variance drift and **Antithetic Variates** for variance reduction.
- **Convergence Proof:** Mathematically verified $O(\Delta x^2)$ second-order convergence via Log-Log error analysis.
- **Real-time Calibration:** Implied Volatility extraction from S&P 500 market data using the **Newton-Raphson** method.

## Mathematical Core
The solver addresses the Black-Scholes PDE:
$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS \frac{\partial V}{\partial S} - rV = 0$$

For the stochastic part, we evolve the Heston variance process $\nu_t$ using RK4:
$$d\nu_t = \kappa(\theta - \nu_t)dt + \sigma_{\nu} \sqrt{\nu_t} dW_t^2$$

## How to Run
1. Install dependencies: `pip install -r requirements.txt` (Coming soon)
2. Launch the dashboard: `streamlit run app.py`
3. Run convergence tests: `python -m tests.convergence_test`

---
*Developed by Giulio Rossi - Engineering Portfolio*