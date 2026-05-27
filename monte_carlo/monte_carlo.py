"""
Monte Carlo Option Pricing
===========================
European call: standard MC, antithetic variates, control variate (S_T as control)
Asian call:   arithmetic average, geometric control variate
Convergence:  error vs N with O(1/√N) reference line

All methods benchmarked against Black-Scholes closed-form price.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# ═══════════════════════════════════════════════════════════════
# Black-Scholes closed-form (benchmark)
# ═══════════════════════════════════════════════════════════════

def bs_call(S0, K, r, sigma, T):
    """Black-Scholes European call price."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# ═══════════════════════════════════════════════════════════════
# GBM path generation
# ═══════════════════════════════════════════════════════════════

def gbm_paths(S0, r, sigma, T, M, N):
    """
    Generate N paths of GBM with M steps each.
    Returns (M+1, N) array: S[0,:] = S0, S[-1,:] = S_T.
    """
    dt = T / M
    Z = np.random.randn(M, N)
    log_returns = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    log_S = np.log(S0) + np.cumsum(log_returns, axis=0)
    S = np.exp(log_S)
    return np.vstack([np.full(N, S0), S])


# ═══════════════════════════════════════════════════════════════
# 1. Standard Monte Carlo — European Call
# ═══════════════════════════════════════════════════════════════

def mc_european_call(S0, K, r, sigma, T, N, M=252):
    """
    Standard MC for European call.
    M steps default = 252 (daily).
    """
    S_paths = gbm_paths(S0, r, sigma, T, M, N)
    ST = S_paths[-1, :]
    payoff = np.maximum(ST - K, 0)
    price = np.exp(-r * T) * np.mean(payoff)
    std_err = np.std(payoff) / np.sqrt(N) * np.exp(-r * T)
    return price, std_err


# ═══════════════════════════════════════════════════════════════
# 2. Antithetic Variates — European Call
# ═══════════════════════════════════════════════════════════════

def mc_antithetic_european_call(S0, K, r, sigma, T, N, M=252):
    """
    Antithetic variates: average payoff(Z) and payoff(-Z) to reduce variance.
    """
    dt = T / M
    price_sum = 0.0
    payoff_sq_sum = 0.0

    for _ in range(N):
        Z = np.random.randn(M)
        # +Z path
        log_S_pos = np.log(S0) + np.cumsum((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
        payoff_pos = max(np.exp(log_S_pos[-1]) - K, 0)

        # -Z path
        log_S_neg = np.log(S0) + np.cumsum((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * (-Z))
        payoff_neg = max(np.exp(log_S_neg[-1]) - K, 0)

        avg_payoff = (payoff_pos + payoff_neg) / 2
        price_sum += avg_payoff
        payoff_sq_sum += avg_payoff ** 2

    price = np.exp(-r * T) * price_sum / N
    var = (payoff_sq_sum / N - (price_sum / N) ** 2)
    std_err = np.exp(-r * T) * np.sqrt(var / N)
    return price, std_err


# ═══════════════════════════════════════════════════════════════
# 3. Control Variate (S_T as control) — European Call
# ═══════════════════════════════════════════════════════════════

def mc_control_variate_european_call(S0, K, r, sigma, T, N, M=252):
    """
    Use S_T itself as control variate (E[S_T] = S0 * exp(rT) is known exactly).
    """
    dt = T / M
    ST_array = np.zeros(N)
    payoff_array = np.zeros(N)

    for i in range(N):
        Z = np.random.randn(M)
        log_S = np.log(S0) + np.cumsum((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
        ST = np.exp(log_S[-1])
        ST_array[i] = ST
        payoff_array[i] = max(ST - K, 0)

    # Compute optimal β = Cov(payoff, S_T) / Var(S_T)
    cov = np.cov(ST_array, payoff_array)[0, 1]
    beta = cov / np.var(ST_array)

    # E[S_T] under risk-neutral measure
    expected_ST = S0 * np.exp(r * T)

    adjusted_payoff = payoff_array - beta * (ST_array - expected_ST)
    price = np.exp(-r * T) * np.mean(adjusted_payoff)
    std_err = np.std(adjusted_payoff) / np.sqrt(N) * np.exp(-r * T)
    return price, std_err, beta


# ═══════════════════════════════════════════════════════════════
# 4. Asian Option — Arithmetic Average
# ═══════════════════════════════════════════════════════════════

def mc_asian_arithmetic_call(S0, K, r, sigma, T, M, N):
    """
    Asian call with arithmetic average payoff.
    No closed-form — MC is the standard approach.
    """
    dt = T / M
    payoff_sum = 0.0
    payoff_sq_sum = 0.0

    for i in range(N):
        Z = np.random.randn(M)
        log_returns = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
        log_S = np.log(S0) + np.cumsum(log_returns)
        S = np.exp(log_S)

        A = np.mean(S)
        payoff = max(A - K, 0)
        payoff_sum += payoff
        payoff_sq_sum += payoff ** 2

    price = np.exp(-r * T) * payoff_sum / N
    var = (payoff_sq_sum / N - (payoff_sum / N) ** 2)
    std_err = np.exp(-r * T) * np.sqrt(var / N)
    return price, std_err


# ═══════════════════════════════════════════════════════════════
# 4b. Asian Option — Geometric Average (control variate)
# ═══════════════════════════════════════════════════════════════

def asian_geometric_expected_payoff(S0, K, r, sigma, T):
    """
    Closed-form expected payoff for continuously sampled geometric-average
    Asian call — used as control variate for arithmetic average.
    """
    b = 0.5 * (r - 0.5 * sigma ** 2)
    sigma_adj = sigma / np.sqrt(3)

    d1 = (np.log(S0 / K) + (b + 0.5 * sigma_adj ** 2) * T) / (sigma_adj * np.sqrt(T))
    d2 = d1 - sigma_adj * np.sqrt(T)

    price = S0 * np.exp((b - r) * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price * np.exp(r * T)


def mc_asian_with_control(S0, K, r, sigma, T, M, N):
    """
    Arithmetic Asian call using geometric-average as control variate.
    Significantly reduces variance compared to standard MC.
    """
    dt = T / M
    X_arr = np.zeros(N)  # arithmetic-average payoff
    Y_arr = np.zeros(N)  # geometric-average payoff

    for i in range(N):
        Z = np.random.randn(M)
        log_returns = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
        log_S = np.log(S0) + np.cumsum(log_returns)
        S = np.exp(log_S)

        # Arithmetic average payoff (target)
        X_arr[i] = max(np.mean(S) - K, 0)

        # Geometric average payoff (control)
        A_geo = np.exp(np.mean(log_S))
        Y_arr[i] = max(A_geo - K, 0)

    # Compute β
    cov = np.cov(X_arr, Y_arr)[0, 1]
    beta = cov / np.var(Y_arr)
    EY = asian_geometric_expected_payoff(S0, K, r, sigma, T)

    adjusted_payoff = X_arr - beta * (Y_arr - EY)
    price = np.exp(-r * T) * np.mean(adjusted_payoff)
    std_err = np.std(adjusted_payoff) / np.sqrt(N) * np.exp(-r * T)
    return price, std_err, beta


# ═══════════════════════════════════════════════════════════════
# 5. Convergence Analysis
# ═══════════════════════════════════════════════════════════════

def convergence_analysis(S0=100, K=100, r=0.05, sigma=0.2, T=1.0):
    """Error vs N — should follow O(1/√N)."""
    M = 252
    N_list = [100, 500, 1000, 5000, 10000, 50000, 100000]
    true_price = bs_call(S0, K, r, sigma, T)

    errors = []
    std_errors = []

    for N in N_list:
        price, std_err = mc_european_call(S0, K, r, sigma, T, N, M)
        errors.append(abs(price - true_price))
        std_errors.append(std_err)

    return np.array(N_list), np.array(errors), np.array(std_errors)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Parameters
    S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    M, N = 252, 100000

    bs_price = bs_call(S0, K, r, sigma, T)
    print(f"{'='*60}")
    print(f"Black-Scholes (benchmark): {bs_price:.4f}")
    print(f"{'='*60}\n")

    # 1. Standard MC
    p, se = mc_european_call(S0, K, r, sigma, T, N, M)
    print(f"[Standard MC]      price = {p:.4f}  ± {se:.4f}  (error = {abs(p-bs_price):.4f})")

    # 2. Antithetic
    p, se = mc_antithetic_european_call(S0, K, r, sigma, T, N // 2, M)
    print(f"[Antithetic]       price = {p:.4f}  ± {se:.4f}  (error = {abs(p-bs_price):.4f})")

    # 3. Control variate (S_T)
    p, se, beta = mc_control_variate_european_call(S0, K, r, sigma, T, N, M)
    print(f"[Control Variate]  price = {p:.4f}  ± {se:.4f}  (error = {abs(p-bs_price):.4f})  β = {beta:.4f}")

    print(f"\n{'='*60}")
    print("Asian Option (arithmetic average)")
    print(f"{'='*60}\n")

    # 4. Asian — standard
    p, se = mc_asian_arithmetic_call(S0, K, r, sigma, T, M=252, N=N)
    print(f"[Asian Standard]   price = {p:.4f}  ± {se:.4f}")

    # 5. Asian — with control
    p, se, beta = mc_asian_with_control(S0, K, r, sigma, T, M=252, N=N)
    print(f"[Asian + Control]  price = {p:.4f}  ± {se:.4f}  β = {beta:.4f}")

    # ── Convergence plot ──
    print(f"\n{'='*60}")
    print("Convergence Analysis")
    print(f"{'='*60}")

    N_list, errors, std_errors = convergence_analysis(S0, K, r, sigma, T)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(N_list, errors, 'bo-', label='|MC − BS|', linewidth=2)
    # Reference O(1/√N)
    ref = errors[2] * np.sqrt(N_list[2]) / np.sqrt(N_list)
    ax.loglog(N_list, ref, 'r--', label=r'$\propto 1/\sqrt{N}$', linewidth=2)
    ax.set_xlabel('Number of paths (N)', fontsize=12)
    ax.set_ylabel('Absolute error', fontsize=12)
    ax.set_title('Monte Carlo Convergence', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('mc_convergence.png', dpi=150)
    print("Saved: mc_convergence.png")
