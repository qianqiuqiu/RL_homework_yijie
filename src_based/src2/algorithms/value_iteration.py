def value_iteration(env, gamma, theta, track_convergence=False):
    """值迭代算法"""
    V = {state: 0.0 for state in env.get_all_states()}
    iterations = 0
    convergence_history = []  # 记录每次迭代的状态值
    
    while True:
        delta = 0
        iterations += 1
        
        for state in env.get_all_states():
            v = V[state]
            # 更新值函数：选择最大期望回报的动作
            V[state] = max(env.get_expected_return(state, action, V, gamma) 
                          for action in env.get_all_actions())
            delta = max(delta, abs(v - V[state]))
        
        if track_convergence:
            convergence_history.append(V.copy())
        
        if delta < theta:
            break
    
    # 提取最优策略
    policy = {state: max(env.get_all_actions(), 
                        key=lambda action: env.get_expected_return(state, action, V, gamma)) 
             for state in env.get_all_states()}
    
    if track_convergence:
        return V, iterations, convergence_history
    return V, iterations
