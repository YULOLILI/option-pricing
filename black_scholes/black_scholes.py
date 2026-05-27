"""
Black-Scholes 欧式期权定价 & Greeks 计算
==========================================

基于 Black-Scholes-Merton 模型，计算欧式看涨/看跌期权价格，
以及全部五个 Greeks（Delta, Gamma, Vega, Theta, Rho）。

参考：Hull, "Options, Futures, and Other Derivatives" (第 15 章)

用法：
    python black_scholes.py

作者：李嘉珩
日期：2026-05-23
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# ============================================================
# 核心定价函数
# ============================================================

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """
    计算欧式期权的 Black-Scholes 价格。

    Parameters
    ----------
    S : float or np.ndarray
        标的资产当前价格
    K : float
        行权价
    T : float
        到期时间（年）
    r : float
        无风险利率（连续复利）
    sigma : float
        波动率（年化）
    option_type : str, 默认 'call'
        'call' 或 'put'

    Returns
    -------
    price : float or np.ndarray
        期权理论价格

    Example
    -------
    >>> black_scholes_price(S=100, K=105, T=0.5, r=0.05, sigma=0.2, option_type='call')
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type 必须是 'call' 或 'put'")

    return price


# ============================================================
# Greeks 计算
# ============================================================

def delta(S, K, T, r, sigma, option_type='call'):
    """
    计算 Delta — 期权价格对标的资产价格的一阶偏导。
    - Call Delta ∈ (0, 1)
    - Put  Delta ∈ (-1, 0)

    Parameters
    ----------
    同 black_scholes_price

    Returns
    -------
    delta : float or np.ndarray
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    if option_type == 'call':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1


def gamma(S, K, T, r, sigma):
    """
    计算 Gamma — Delta 对标的资产价格的二阶偏导（凸性）。
    Call 与 Put 的 Gamma 相同。

    Returns
    -------
    gamma : float or np.ndarray
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma):
    """
    计算 Vega — 期权价格对波动率的一阶偏导。
    注意：实际返回的是波动率变化 1%（即 0.01）对应的价格变化，
    所以除以 100。

    Call 与 Put 的 Vega 相同。

    Returns
    -------
    vega : float or np.ndarray
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    # 原始公式值对应 sigma 变化 1 单位 → 通常除以 100 表示 1% 变化
    return S * norm.pdf(d1) * np.sqrt(T) / 100


def theta(S, K, T, r, sigma, option_type='call'):
    """
    计算 Theta — 期权价格对时间的一阶偏导（时间衰减）。
    返回的是每日 Theta（除以 365）。

    Call Theta 通常为负（时间越少，期权越不值钱）。
    Put Theta 符号取决于参数。

    Returns
    -------
    theta : float or np.ndarray
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # 原始公式是按年的 Theta
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))

    if option_type == 'call':
        theta_year = term1 - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        theta_year = term1 + r * K * np.exp(-r * T) * norm.cdf(-d2)

    return theta_year / 365


def rho(S, K, T, r, sigma, option_type='call'):
    """
    计算 Rho — 期权价格对无风险利率的一阶偏导。
    返回利率变化 1%（即 0.01）对应的价格变化（除以 100）。

    Returns
    -------
    rho : float or np.ndarray
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        return K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100


# ============================================================
# 汇总函数
# ============================================================

def option_report(S, K, T, r, sigma, option_type='call'):
    """
    一次性输出期权价格和全部五个 Greeks 的汇总表。

    Parameters
    ----------
    同 black_scholes_price

    Returns
    -------
    dict : 包含价格和五个 Greeks
    """
    price = black_scholes_price(S, K, T, r, sigma, option_type)
    greeks = {
        'Price': price,
        'Delta': delta(S, K, T, r, sigma, option_type),
        'Gamma': gamma(S, K, T, r, sigma),
        'Vega':  vega(S, K, T, r, sigma),
        'Theta': theta(S, K, T, r, sigma, option_type),
        'Rho':   rho(S, K, T, r, sigma, option_type),
    }
    return greeks


# ============================================================
# 可视化
# ============================================================

def plot_greeks(S_range, K, T, r, sigma):
    """
    画出 Greeks 随标的资产价格 S 变化的曲线。

    Parameters
    ----------
    S_range : np.ndarray
        标的资产价格的范围
    K, T, r, sigma : 同 black_scholes_price
    """
    call_price = black_scholes_price(S_range, K, T, r, sigma, 'call')
    put_price  = black_scholes_price(S_range, K, T, r, sigma, 'put')

    call_delta  = delta(S_range, K, T, r, sigma, 'call')
    put_delta   = delta(S_range, K, T, r, sigma, 'put')
    gamma_val   = gamma(S_range, K, T, r, sigma)
    vega_val    = vega(S_range, K, T, r, sigma)
    call_theta  = theta(S_range, K, T, r, sigma, 'call')
    put_theta   = theta(S_range, K, T, r, sigma, 'put')
    call_rho    = rho(S_range, K, T, r, sigma, 'call')
    put_rho     = rho(S_range, K, T, r, sigma, 'put')

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(
        f'Black-Scholes Greeks vs Underlying Price\n'
        f'K={K}, T={T:.2f}y, r={r*100:.1f}%, σ={sigma*100:.1f}%',
        fontsize=14, fontweight='bold'
    )

    # (1) 期权价格
    ax = axes[0, 0]
    ax.plot(S_range, call_price, 'b-', label='Call', linewidth=2)
    ax.plot(S_range, put_price, 'r--', label='Put', linewidth=2)
    ax.axvline(K, color='gray', linestyle=':', alpha=0.7, label=f'K={K}')
    ax.set_ylabel('Price ($)')
    ax.set_title('Option Price')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (2) Delta
    ax = axes[0, 1]
    ax.plot(S_range, call_delta, 'b-', label='Call Delta', linewidth=2)
    ax.plot(S_range, put_delta, 'r--', label='Put Delta', linewidth=2)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(K, color='gray', linestyle=':', alpha=0.7)
    ax.set_ylabel('Delta')
    ax.set_title('Delta')
    ax.set_ylim(-1.1, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (3) Gamma
    ax = axes[1, 0]
    ax.plot(S_range, gamma_val, 'g-', label='Gamma', linewidth=2)
    ax.axvline(K, color='gray', linestyle=':', alpha=0.7)
    ax.set_ylabel('Gamma')
    ax.set_title('Gamma (Call = Put)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (4) Vega
    ax = axes[1, 1]
    ax.plot(S_range, vega_val, 'm-', label='Vega', linewidth=2)
    ax.axvline(K, color='gray', linestyle=':', alpha=0.7)
    ax.set_ylabel('Vega ($ per 1% σ change)')
    ax.set_title('Vega (Call = Put)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (5) Theta
    ax = axes[2, 0]
    ax.plot(S_range, call_theta, 'b-', label='Call Theta', linewidth=2)
    ax.plot(S_range, put_theta, 'r--', label='Put Theta', linewidth=2)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(K, color='gray', linestyle=':', alpha=0.7)
    ax.set_xlabel('Underlying Price S ($)')
    ax.set_ylabel('Theta ($ per day)')
    ax.set_title('Theta (daily)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (6) Rho
    ax = axes[2, 1]
    ax.plot(S_range, call_rho, 'b-', label='Call Rho', linewidth=2)
    ax.plot(S_range, put_rho, 'r--', label='Put Rho', linewidth=2)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(K, color='gray', linestyle=':', alpha=0.7)
    ax.set_xlabel('Underlying Price S ($)')
    ax.set_ylabel('Rho ($ per 1% r change)')
    ax.set_title('Rho')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('greeks.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("图表已保存为 greeks.png")


# ============================================================
# 主程序
# ============================================================

if __name__ == '__main__':
    # 典型参数
    S0     = 100.0   # 当前价格
    K      = 105.0   # 行权价
    T      = 0.5     # 6 个月到期
    r      = 0.05    # 5% 无风险利率
    sigma  = 0.20    # 20% 年化波动率

    # --- 1. 单点定价 ---
    print("=" * 55)
    print("Black-Scholes 欧式期权定价")
    print("=" * 55)
    print(f"{'参数':<20} {'值':>10}")
    print("-" * 30)
    print(f"{'标的价格 S':<20} {S0:>10.2f}")
    print(f"{'行权价 K':<20} {K:>10.2f}")
    print(f"{'到期时间 T':<20} {T:>10.2f} y")
    print(f"{'无风险利率 r':<20} {r*100:>9.1f}%")
    print(f"{'波动率 σ':<20} {sigma*100:>9.1f}%")
    print()

    call_report = option_report(S0, K, T, r, sigma, 'call')
    put_report  = option_report(S0, K, T, r, sigma, 'put')

    print(f"{'指标':<15} {'Call':>15} {'Put':>15}")
    print("-" * 45)
    print(f"{'Price ($)':<15} {call_report['Price']:>15.4f} {put_report['Price']:>15.4f}")
    print(f"{'Delta':<15} {call_report['Delta']:>15.4f} {put_report['Delta']:>15.4f}")
    print(f"{'Gamma':<15} {call_report['Gamma']:>15.4f} {put_report['Gamma']:>15.4f}")
    print(f"{'Vega':<15} {call_report['Vega']:>15.4f} {put_report['Vega']:>15.4f}")
    print(f"{'Theta':<15} {call_report['Theta']:>15.4f} {put_report['Theta']:>15.4f}")
    print(f"{'Rho':<15} {call_report['Rho']:>15.4f} {put_report['Rho']:>15.4f}")
    print()

    # --- 2. Put-Call Parity 验证 ---
    parity_lhs = call_report['Price'] + K * np.exp(-r * T)
    parity_rhs = put_report['Price'] + S0
    print(f"Put-Call Parity 验证：")
    print(f"  C + K*e^(-rT) = {parity_lhs:.6f}")
    print(f"  P + S          = {parity_rhs:.6f}")
    print(f"  差异           = {abs(parity_lhs - parity_rhs):.2e}")
    assert abs(parity_lhs - parity_rhs) < 1e-10, "Put-Call Parity 不成立！"
    print("  ✅ Put-Call Parity 成立")
    print()

    # --- 3. Greeks 画图 ---
    print("生成 Greeks 曲线图...")
    S_range = np.linspace(50, 150, 500)
    plot_greeks(S_range, K, T, r, sigma)
    print("完成！")
