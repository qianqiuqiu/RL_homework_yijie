def truncated_policy_iteration(env, gamma, theta, initial_policy, x_eval_steps, track_convergence=False):
    """截断式策略迭代算法"""
    policy = initial_policy.copy()
    V = {state: 0.0 for state in env.get_all_states()}
    iterations = 0
    convergence_history = []  # 记录每次外层迭代的状态值
    
    while True:
        iterations += 1
        
        # 截断式策略评估（Truncated Policy Evaluation）
        # 只执行 x_eval_steps 次值更新，而不是完全收敛
        for _ in range(x_eval_steps):
            for state in env.get_all_states():
                action = policy[state]
                V[state] = env.get_expected_return(state, action, V, gamma)
        
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
            # 策略稳定后，完全评估一次以确保收敛
            while True:
                delta = 0
                for state in env.get_all_states():
                    v = V[state]
                    action = policy[state]
                    V[state] = env.get_expected_return(state, action, V, gamma)
                    delta = max(delta, abs(v - V[state]))
                
                if delta < theta:
                    break
            break
    
    if track_convergence:
        return V, iterations, convergence_history
    return V, iterations
