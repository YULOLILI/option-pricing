# Option Pricing Models

Python implementations of classic option pricing models, based on Hull's *Options, Futures, and Other Derivatives*.

## Projects

| Module | Description | Methods |
|--------|-------------|---------|
| **Black-Scholes** | European option pricing with Greeks | BS formula, Delta, Gamma, Theta, Vega, Rho |
| **Binomial Tree** | CRR binomial tree pricing | European & American options, convergence analysis |
| **BSM Formula** | Black-Scholes-Merton analytical pricing | Closed-form European call/put |

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib

## Usage

```bash
# Black-Scholes with Greeks
python black_scholes/black_scholes.py

# Binomial Tree pricing
python binomial_tree/binomial_tree.py

# BSM analytical formula
python bsm_formula/bsm_pricing.py
```

## Part of 91-Day Quantitative Finance Study Plan

Phase 1 — Derivatives Pricing Fundamentals (May 2026)
