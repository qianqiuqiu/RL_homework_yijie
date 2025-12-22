import numpy as np

def solve_closed_form(env, policy=None, gamma=0.9):
    """
    Evaluate a policy using the closed-form solution: V = (I - gamma * P)^-1 * R
    
    Args:
        env: The GridWorld environment.
        policy: A dictionary {state: action}, a function mapping state to action, 
                or None for Uniform Random Policy.
        gamma: Discount factor.
        
    Returns:
        V: Dictionary mapping state -> value.
    """
    states = env.get_all_states()
    n = len(states)
    state_to_idx = {s: i for i, s in enumerate(states)}
    
    P = np.zeros((n, n))
    R = np.zeros(n)
    
    for s in states:
        s_idx = state_to_idx[s]
        
        if policy is None:
            # Uniform Random Policy
            actions = env.action_space
            prob = 1.0 / len(actions)
            for action in actions:
                next_state, reward = env.get_transition_model(s, action)
                ns_idx = state_to_idx[next_state]
                P[s_idx, ns_idx] += prob
                R[s_idx] += prob * reward
        else:
            # Deterministic Policy
            if isinstance(policy, dict):
                action = policy.get(s)
            elif callable(policy):
                action = policy(s)
            else:
                raise ValueError("Policy must be a dict, callable, or None")
                
            # Get transition dynamics
            next_state, reward = env.get_transition_model(s, action)
            ns_idx = state_to_idx[next_state]
            
            P[s_idx, ns_idx] = 1.0
            R[s_idx] = reward
        
    # Solve (I - gamma * P) * V = R
    I = np.eye(n)
    A = I - gamma * P
    
    # V = A^-1 * R
    try:
        V_vec = np.linalg.solve(A, R)
    except np.linalg.LinAlgError:
        # Fallback if singular
        V_vec = np.linalg.lstsq(A, R, rcond=None)[0]
        
    V = {s: V_vec[state_to_idx[s]] for s in states}
    return V
