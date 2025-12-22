def policy_iteration(env, gamma, theta, initial_policy, track_convergence=False):
    """策略迭代算法"""
    policy = initial_policy.copy()
    V = {state: 0.0 for state in env.get_all_states()}
    iterations = 0
    convergence_history = []  # 记录每次外层迭代的状态值
    
    while True:
        iterations += 1
        
        # 策略评估（Policy Evaluation）
        while True:
            delta = 0
            for state in env.get_all_states():
                v = V[state]
                action = policy[state]
                V[state] = env.get_expected_return(state, action, V, gamma)
                delta = max(delta, abs(v - V[state]))
            
            if delta < theta:
                break
        
        if track_convergence:
            convergence_history.append(V.copy())
        
        # 策略改进（Policy Improvement）
        policy_stable = True
        for state in env.get_all_states():
            old_action = policy[state]
            policy[state] = max(env.get_all_actions(), 
                              key=lambda action: env.get_expected_return(state, action, V, gamma))
            if old_action != policy[state]:
                policy_stable = False
        
        if policy_stable:
            break
    
    if track_convergence:
        return V, iterations, convergence_history
    return V, iterations
