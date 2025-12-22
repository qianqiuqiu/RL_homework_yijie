import numpy as np
import os
from envs.grid_world import GridWorld
from utils.plotting import plot_value_function

def evaluate_policy(env, policy=None, gamma=0.9, theta=1e-6, method='iterative'):
    """
    Calculates the state-value function V_pi for a given policy.
    
    Args:
        env: The GridWorld environment.
        policy: A dictionary {state: action} for deterministic policy, 
                or None for Uniform Random Policy.
        gamma: Discount factor.
        theta: Convergence threshold (for iterative method).
        method: 'iterative' or 'closed_form'.
        
    Returns:
        V: Dictionary mapping state -> value.
    """
    if method == 'closed_form':
        from algorithms.dp.closed_form import solve_closed_form
        return solve_closed_form(env, policy, gamma)

    states = env.get_all_states()
    actions = env.action_space
    
    # Initialize V(s) = 0
    V = {state: 0.0 for state in states}
    
    iteration = 0
    while True:
        delta = 0
        iteration += 1
        
        # For each state s
        for state in states:
            v = V[state]
            
            if policy is None:
                # Uniform Random Policy
                num_actions = len(actions)
                prob_action = 1.0 / num_actions
                expected_value = 0
                for action in actions:
                    next_state, reward = env.get_transition_model(state, action)
                    expected_value += prob_action * (reward + gamma * V[next_state])
                V[state] = expected_value
            else:
                # Deterministic Policy
                if isinstance(policy, dict):
                    action = policy.get(state)
                else:
                    # Assume callable
                    action = policy(state)
                    
                next_state, reward = env.get_transition_model(state, action)
                V[state] = reward + gamma * V[next_state]

            delta = max(delta, abs(v - V[state]))
            
        if delta < theta:
            # print(f"Policy Evaluation converged in {iteration} iterations.")
            break
            
    return V

def main():
    print("=== Policy Evaluation Tool ===")
    
    # Setup result directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result_dir = os.path.join(base_dir, 'result')
    os.makedirs(result_dir, exist_ok=True)
    
    # Initialize Environment
    env = GridWorld()
    
    # 1. Calculate Ground Truth for Random Policy (Iterative)
    print("Calculating value for Random Policy (Iterative)...")
    V_random_iter = evaluate_policy(env, policy=None, method='iterative')
    plot_value_function(V_random_iter, title="Random Policy Value (Iterative)", save_path=os.path.join(result_dir, 'random_policy_value_iterative.png'))
    
    # 2. Calculate Ground Truth for Random Policy (Closed Form)
    print("Calculating value for Random Policy (Closed Form)...")
    V_random_closed = evaluate_policy(env, policy=None, method='closed_form')
    plot_value_function(V_random_closed, title="Random Policy Value (Closed Form)", save_path=os.path.join(result_dir, 'random_policy_value_closed_form.png'))
    
    # 手动定义最优策略 (Optimal Policy)
    # 对应图示中的箭头方向
    optimal_policy = {
        # Row 0
        (0, 0): 3, (0, 1): 3, (0, 2): 3, (0, 3): 1, (0, 4): 1,
        # Row 1
        (1, 0): 0, (1, 1): 0, (1, 2): 3, (1, 3): 1, (1, 4): 1,
        # Row 2
        (2, 0): 0, (2, 1): 2, (2, 2): 1, (2, 3): 3, (2, 4): 1,
        # Row 3
        (3, 0): 0, (3, 1): 3, (3, 2): 4, (3, 3): 2, (3, 4): 1,
        # Row 4
        (4, 0): 0, (4, 1): 3, (4, 2): 0, (4, 3): 2, (4, 4): 2,
    }
    
    # 3. Calculate Value Function for Optimal Policy
    print("Calculating value for Optimal Policy...")
    V_optimal = evaluate_policy(env, policy=optimal_policy, method='closed_form')
    # 保存结果
    plot_value_function(
        V_optimal, 
        title="Optimal Policy Value closed-form", 
        save_path=os.path.join(result_dir, 'optimal_policy_value_closed_form.png')
    )
    
    V_optimal = evaluate_policy(env, policy=optimal_policy, method='iterative')
    plot_value_function(
        V_optimal, 
        title="Optimal Policy Value Iterative", 
        save_path=os.path.join(result_dir, 'optimal_policy_value_iterative.png')
    )
    
    
    print("Done.")

if __name__ == "__main__":
    main()
