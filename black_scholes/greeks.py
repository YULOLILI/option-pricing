# 只允许 import 这两个
import numpy as np
from scipy.stats import norm


def bs_greeks(S, K, T, r, sigma):
    """
    Args:
        S, K, T, r, sigma
    Returns:
        'call_price': call_price,
        'put_price': put_price,
        'call_delta': call_delta,
        'put_delta': put_delta,
        'gamma': gamma,
        'vega': vega,
        'call_theta': call_theta,
        'put_theta': put_theta,
        'call_rho': call_rho,
        'put_rho': put_rho

    """
    d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    

    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    nd1 = norm.pdf(d1)
    df = np.exp(-r * T)
    
  
    call_price = S * Nd1 - K * df * Nd2
    put_price = K * df * norm.cdf(-d2) - S * norm.cdf(-d1)
    

    call_delta = Nd1
    put_delta = Nd1 - 1
    
    gamma = nd1 / (S * sigma * np.sqrt(T))
    
    vega = S * nd1 * np.sqrt(T)
    
    call_theta = -S * nd1 * sigma / (2 * np.sqrt(T)) - r * K * df * Nd2
    put_theta = -S * nd1 * sigma / (2 * np.sqrt(T)) + r * K * df * norm.cdf(-d2)
    
    call_rho = K * T * df * Nd2
    put_rho = -K * T * df * norm.cdf(-d2) 
    

    return {
        'call_price': call_price,
        'put_price': put_price,
        'call_delta': call_delta,
        'put_delta': put_delta,
        'gamma': gamma,
        'vega': vega,
        'call_theta': call_theta,
        'put_theta': put_theta,
        'call_rho': call_rho,
        'put_rho': put_rho
    }

def main():
    S = 100
    K = 105
    T = 0.5
    r = 0.05
    sigma = 0.2
    
    greeks = bs_greeks(S, K, T, r, sigma)


    print(greeks['call_delta'])
    print(greeks['put_delta'])
    print(greeks['gamma'])
    print(greeks['vega'])
    print(greeks['call_theta'])
    print(greeks['put_theta'])
    print(greeks['call_rho'])
    print(greeks['put_rho'])
    print()

if __name__ == "__main__":
    main()