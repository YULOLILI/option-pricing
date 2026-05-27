import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm





def build_crr_tree(S0, T, r, sigma, N):
    """
    Args: 
        S0, T, r, sigma, N

    Returns: 
        dt, u, d, p, price_tree


    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) -d) / (u - d)

    if not (0 < p < 1):
        print(f"Warning: p={p} is not in (0,1) range")

    # 价格树
    price_tree = np.zeros((N + 1, N + 1))

    for i in range(N + 1):
        for j in range(i + 1):
            price_tree[j,i] = S0 * (u ** j) * d ** (i -j)

    return dt, u, d, p, price_tree



def binomial_european(S0, K, T, r, sigma, N, option_type='call'):

    """
    Args: 
        S0, K, T, r, sigma, N, option_type='call'

    Returns:
        option_tree[0, 0]



    """



    dt, u, d, p, price_tree = build_crr_tree(S0, T, r, sigma, N)


    # 期权价值树
    option_tree = np.zeros((N + 1, N + 1))

    # 终期期权价值输入

    for j in range(N + 1):
        S_T = price_tree[j, N]
        if option_type == 'call':

            option_tree[j, N] = max(S_T - K, 0)
        else:
            option_tree[j, N] = max(K - S_T, 0)
    # 完整期权价值树

    for i in range(N - 1, -1 , -1):
        for j in range(i + 1):
            # 如果只计算欧式期权价格可以直接计算出来，但准备美式期权需要完整的树
            hold_value = np.exp(-r * dt) * (p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1])
                                            
           
            option_tree[j, i] = hold_value

    return option_tree[0, 0]

def binomial_american(S0, K, T, r, sigma, N, option_type='call'):
    """
    Args:
        S0, K, T, r, sigma, N, option_type='call'
    Returns:
        option_tree[0, 0], exercise_matrix
    
    """

    # 依旧
    dt, u, d, p, price_tree = build_crr_tree(S0, T, r, sigma, N)
    option_tree = np.zeros((N + 1, N + 1))

    # 多了一个执行树
    exercise_matrix = np.zeros((N + 1, N + 1), dtype=bool)

    for j in range(N + 1):
        S_T = price_tree[j, N]
        if option_type == 'call':
            option_tree[j, N] = max(S_T - K, 0)
        else:  
            option_tree[j, N] = max(K - S_T, 0)

        # 判断终期期权价值是否大于零
        exercise_matrix[j, N] = (option_tree[j, N] > 0)


    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            S_T = price_tree[j, i]


            if option_type == 'call':
                exercise_value = max(S_T - K, 0)
            else:  
                exercise_value = max(K - S_T, 0)
            
            hold_value = np.exp(-r * dt) * (p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1])
            if exercise_value >= hold_value:
                option_tree[j, i] = exercise_value
                exercise_matrix[j, i] = True 
            else:
                option_tree[j, i] = hold_value
                exercise_matrix[j, i] = False  

    return option_tree[0, 0], exercise_matrix



def convergence_analysis(S0=100, K=100, T=1, r=0.05, sigma=0.2):

    """
    Args:
        S0=100, K=100, T=1, r=0.05, sigma=0.2

    Returns:
    
    
    """
    def black_scholes(S0, K, T, r, sigma, option_type='call'):


        d1 = (np.log(S0/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
        return price






    N_values = [10, 50, 100, 200, 500]

    # 依旧矩阵大法
    european_call_prices = []
    european_put_prices = []
    american_call_prices = []
    american_put_prices = []

    for N in N_values:

        # 每种期权五个数据
        ecall = binomial_european(S0, K, T, r, sigma, N, 'call')
        eput = binomial_european(S0, K, T, r, sigma, N, 'put')
        acall, _ = binomial_american(S0, K, T, r, sigma, N, 'call')
        aput, _ = binomial_american(S0, K, T, r, sigma, N, 'put')

        european_call_prices.append(ecall)
        european_put_prices.append(eput)
        american_call_prices.append(acall)
        american_put_prices.append(aput)

    # 作图
    

    # Call option 对比 对数刻度
    plt.figure(figsize = (12, 5))
    plt.title(f'Call Option Convergence (S0={S0}, K={K})', fontsize=14)
    plt.xlabel('Number of Steps N (log scale)', fontsize=12)
    plt.ylabel('Option Price', fontsize=12)    
    plt.semilogx(N_values, european_call_prices, 'bo-', label='European Call (CRR)')
    plt.semilogx(N_values, american_call_prices, 'rs--', label='American Call (CRR)')
    plt.axhline(y=black_scholes(S0, K, T, r, sigma, 'call'), color='grey', linestyle='-')
    plt.grid(True, alpha=0.3)
    plt.legend()    
    plt.show()

    # Put option 对比 对数刻度

    plt.figure(figsize=(12,5))
    plt.title(f'Put Option Convergence (S0={S0}, K={K})', fontsize=14)
    plt.xlabel('Number of Steps N (log scale)', fontsize=12)
    plt.ylabel('Option Price', fontsize=12)
    plt.semilogx(N_values, european_put_prices, 'bo-',label='European Put (CRR)')              
    plt.semilogx(N_values, american_put_prices, 'rs--', label='American Put (CRR)')             
    plt.axhline(y=black_scholes(S0, K, T, r, sigma, 'put'), color='g', linestyle='-.')
    plt.grid(True, alpha=0.3)
    plt.legend()    
    plt.show()

def early_exercise_boundary(S0=100, K=100, T=1, r=0.05, sigma=0.2, N=100):
    """
    Args:
        S0=100, K=100, T=1, r=0.05, sigma=0.2, N=100

    Returns:


    
    """
    # 依旧 得再来一遍

    dt, u, d, p, price_tree = build_crr_tree(S0, T, r, sigma, N)
    option_tree = np.zeros((N + 1, N + 1))
    exercise_matrix = np.zeros((N + 1, N + 1), dtype=bool)
    for j in range(N + 1):
        S_T = price_tree[j, N]
        option_tree[j, N] = max(0, K - S_T)  # 看跌期权
        exercise_matrix[j, N] = (option_tree[j, N] > 0)
    
  
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            S_current = price_tree[j, i]
            exercise_value = max(0, K - S_current)
            hold_value = np.exp(-r * dt) * (p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1])
                                            
            if exercise_value >= hold_value:
                option_tree[j, i] = exercise_value
                exercise_matrix[j, i] = True
            else:
                option_tree[j, i] = hold_value
                exercise_matrix[j, i] = False
    
    # 横坐标
    time_points = np.arange(N + 1) * dt



    # 收集该时间步所有节点的股价和是否行权

    boundary_prices = []

    for i in range(N + 1):
        
        prices_at_time = []
        exercise_at_time = []
        
        for j in range(i + 1):
            prices_at_time.append(price_tree[j, i])
            exercise_at_time.append(exercise_matrix[j, i])


        sorted_indices = np.argsort(prices_at_time)
        sorted_prices = np.array(prices_at_time)[sorted_indices]
        sorted_exercise = np.array(exercise_at_time)[sorted_indices]

        boundary = None
        for k in range(len(sorted_prices) - 1):
            if sorted_exercise[k] and not sorted_exercise[k + 1]:
                boundary = (sorted_prices[k] + sorted_prices[k + 1]) / 2
                break
        if boundary is None:
            if any(sorted_exercise):
                boundary = max([p for p, e in zip(sorted_prices, sorted_exercise) if e])
            else:
                boundary = 0
        
        boundary_prices.append(boundary)

     # 左图：行权边界曲线
    plt.figure(figsize = (12, 5))
    plt.xlabel('Time t (years)', fontsize=12)
    plt.ylabel('Stock Price S', fontsize=12)
    plt.title(f'American Put Option - Early Exercise Boundary (N={N})', fontsize=14)

    plt.plot(time_points, boundary_prices, 'r-', label='Early Exercise Boundary')
    plt.axhline(y=K, color='b', linestyle='--', alpha=0.5, label=f'K={K}')

    plt.fill_between(time_points, 0, boundary_prices, alpha=0.3, color='red', label='Early Exercise Region')
    plt.fill_between(time_points, boundary_prices, max(boundary_prices)*1.2, alpha=0.3, color='green', label='Keep Region')

    
  


    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    



    plt.figure(figsize=(12, 5))
    plt.plot(time_points, boundary_prices, 'r-', linewidth=2, label='Early Exercise Boundary')
    return {'time': time_points, 'boundary': boundary_prices}

if __name__ == "__main__":

    S0, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

    convergence_analysis(S0, K, T, r, sigma)
    early_exercise_boundary(S0, K, T, r, sigma, N=100)

    




          



            

             






