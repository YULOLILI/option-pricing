# 只允许 import 这两个
import numpy as np
from scipy.stats import norm

# 任务 1：定义函数 bs_call(S, K, T, r, sigma) → 返回 call 价格
# 任务 2：定义函数 bs_put(S, K, T, r, sigma) → 返回 put 价格
# 任务 3：用参数 S=100, K=105, T=0.5, r=0.05, σ=0.2 测试
#          正确答案：Call ≈ 4.58, Put ≈ 6.99


#1 

def bs_call(S,K,T,r,sigma):
    d1 = (np.log(S/K)+(r+sigma**2/2)* T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    call_price = S* norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def bs_put(S, K, T, r, sigma):
    d1 = (np.log(S/K)+(r+sigma**2/2)* T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    put_price = K *np.exp(-r*T) * norm.cdf(-d2) - S* norm.cdf(-d1)
    return put_price

def main():
    
    S = 100
    K = 105
    T = 0.5
    r = 0.05
    sigma = 0.2
    call_price = bs_call(S, K, T, r, sigma)
    put_price = bs_put(S, K, T, r, sigma)
    print(call_price)
    print(put_price)

    left_side = call_price - put_price
    right_side = S - K*np.exp(-r*T)
    if (left_side - right_side) < 1e-6:
        print("correct")
    else:
        print("NOt correct")


















if __name__ == "__main__":
    main()