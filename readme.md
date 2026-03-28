# AeroQuant Engine
**High-Performance Quantitative Framework for Option Pricing & Market Calibration**

AeroQuant is a high-performance numerical engine for pricing vanilla and barrier options. Built on a foundation of Aerospace Engineering numerical principles, the solver utilizes Finite Difference Methods (FDM) to solve the Black-Scholes PDE. Key features include Crank-Nicolson integration with Rannacher stepping to ensure stability and eliminate numerical oscillations, and a dynamic grid alignment for precise strike and barrier handling.


## Key Features
- **PDE Pricing Engine**: A second-order accurate solver for the Black-Scholes PDE using the Crank-Nicolson scheme.
- **Numerical Stability**: Implementation of Rannacher Stepping (initial Implicit Euler steps) to suppress spurious oscillations caused by the non-smooth payoff of European and Barrier options.
- **High-Performance Linear Algebra**: Tridiagonal systems are solved in $O(n)$ time using the Thomas Algorithm, ensuring high-speed execution.
- **Dynamic Grid Mapping**: Adaptive spatial discretization that aligns nodes exactly with the Strike ($K$) and Barrier ($H$) to minimize interpolation errors.
- **Market Calibration**: Real-time Implied Volatility (IV) extraction from S&P 500 options data via a clamped Newton-Raphson optimizer.
- **Stochastic Benchmarking**: A Heston Model implementation (Monte Carlo) to explore stochastic volatility effects, integrated with RK4 for the variance process.

- **Scientific Validation & Convergence**
The engine has been tested against the analytical Black-Scholes solution. Log-log error analysis confirms a second-order convergence rate ($O(\Delta x^2, \Delta t^2)$), demonstrating the mathematical consistency of the Crank-Nicolson implementation.

- **Interactive Dashboard**
The project includes a Streamlit web application to visualize the option's price surface, Greeks (Delta, Gamma, Theta), and the impact of barriers in real-time.




## Research & Development
The repository includes a research/ directory for experimental models and benchmarking:
- **Heston Stochastic Volatility (Monte Carlo)**: A framework to account for the "volatility smile" and leverage effects, which constant-volatility PDE models cannot capture.
- **Hybrid Numerical Scheme**: To evolve the variance process, the engine utilizes a Runge-Kutta 4 (RK4) integration for the deterministic mean-reversion drift, combined with a Milstein scheme for the stochastic diffusion.
- **Variance Reduction**: Implements Antithetic Variates to significantly reduce the standard error of the Monte Carlo estimates, ensuring faster convergence.
- **Leverage Effect**: Simulates the correlation (ρ) between asset returns and volatility, providing a more realistic representation of market dynamics

Note: The Heston engine is currently used as a stochastic benchmark for the PDE solver and is undergoing integration into the main AeroQuant engine.



## Mathematical Core

### Black-Scholes PDE
The engine solves the second-order partial differential equation for the option value $V(S, t)$:

$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS \frac{\partial V}{\partial S} - rV = 0$$

### Heston Model (Stochastic Volatility)
In the research module, we evolve the asset price $S_t$ and its variance $v_t$ using the following SDEs:

$$dS_t = r S_t dt + \sqrt{v_t} S_t dW_t^1$$
$$dv_t = \kappa(\theta - v_t)dt + \sigma_{\nu} \sqrt{v_t} dW_t^2$$

*Where $dW_t^1$ and $dW_t^2$ have correlation $\rho$.*

## 📊 Market Intelligence
The calibration module fetches live S&P 500 data to compute the **Volatility Risk Premium (VRP)**:
- **Implied Volatility (IV):** Derived from the PDE model calibrated on market prices.
- **Realized Volatility (RV):** Computed from historical returns to provide a benchmark for "fair" pricing.

## 🛠️ Installation & Usage
1. **Clone the repo**: `git clone https://github.com/tuo-username/AeroQuant.git`
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Launch the Dashboard**: `streamlit run app.py`