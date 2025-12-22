import sys
import os
import json
import numpy as np

# Add src_integrated to path
sys.path.append(os.path.join(os.getcwd(), 'src_integrated'))

from envs.grid_world import GridWorld
from algorithms.dp.value_iteration import ValueIterationAgent
from algorithms.dp.policy_iteration import PolicyIterationAgent
from algorithms.dp.truncated_policy_iteration import TruncatedPolicyIterationAgent
from algorithms.temporal_difference.q_learning import QLearningAgent
from algorithms.approximation.td_linear import TDLinearAgent
from utils.features import FeatureExtractor

def serialize_grid(data_dict, rows=5, cols=5):
    """Convert dictionary {(r,c): val} to 2D list."""
    grid = [[0 for _ in range(cols)] for _ in range(rows)]
    for (r, c), val in data_dict.items():
        if isinstance(val, (np.integer, np.floating)):
            val = float(val)
        grid[r][c] = val
    return grid

def get_v_star(env):
    """Calculate V* using high-precision Value Iteration."""
    agent = ValueIterationAgent(env, theta=1e-8)
    agent.train()
    return agent.V

def calculate_error(V, V_star):
    """Calculate max absolute error."""
    max_err = 0
    for s in V:
        err = abs(V[s] - V_star[s])
        if err > max_err:
            max_err = err
    return float(max_err)

def run_vi(env, v_star):
    agent = ValueIterationAgent(env)
    history = []
    errors = []
    
    states = env.get_all_states()
    actions = env.action_space
    
    iteration = 0
    while True:
        delta = 0
        iteration += 1
        
        for state in states:
            v = agent.V[state]
            q_values = []
            for action in actions:
                next_state, reward = env.get_transition_model(state, action)
                q_val = reward + agent.gamma * agent.V[next_state]
                q_values.append(q_val)
            
            agent.V[state] = max(q_values)
            delta = max(delta, abs(v - agent.V[state]))
        
        agent._derive_policy()
        
        history.append({
            "V": serialize_grid(agent.V),
            "policy": serialize_grid(agent.policy)
        })
        errors.append(calculate_error(agent.V, v_star))
        
        if delta < agent.theta:
            break
            
    return {"frames": history, "errors": errors}

def run_pi(env, v_star):
    agent = PolicyIterationAgent(env)
    history = []
    errors = []
    
    while True:
        agent._policy_evaluation()
        policy_stable = agent._policy_improvement()
        
        history.append({
            "V": serialize_grid(agent.V),
            "policy": serialize_grid(agent.policy)
        })
        errors.append(calculate_error(agent.V, v_star))
        
        if policy_stable:
            break
    return {"frames": history, "errors": errors}

def run_tpi(env, k, v_star):
    agent = TruncatedPolicyIterationAgent(env, k=k)
    history = []
    errors = []
    
    while True:
        agent._policy_evaluation(agent.k)
        policy_stable = agent._policy_improvement()
        
        history.append({
            "V": serialize_grid(agent.V),
            "policy": serialize_grid(agent.policy)
        })
        errors.append(calculate_error(agent.V, v_star))
        
        if policy_stable:
            break
    return {"frames": history, "errors": errors}

def run_q_learning(env, epsilon):
    agent = QLearningAgent(env)
    def behavior_policy(agent, state):
        if np.random.rand() < epsilon:
            return np.random.choice(agent.actions)
        else:
            return agent.predict(state)
            
    agent.train(num_episodes=500, behavior_policy=behavior_policy)
    
    V = {}
    policy = {}
    for state in env.get_all_states():
        q_vals = agent.get_q(state)
        V[state] = np.max(q_vals)
        policy[state] = np.argmax(q_vals)
        
    return {
        "V": serialize_grid(V),
        "policy": serialize_grid(policy)
    }

def run_td_linear(env, feature_type, order):
    actual_order = order
    if feature_type == 'polynomial':
        if order == 1: actual_order = 3
        elif order == 2: actual_order = 6
        elif order == 3: actual_order = 10
    
    fe = FeatureExtractor(feature_type=feature_type, order=actual_order)
    agent = TDLinearAgent(env, fe)
    agent.train(num_episodes=500)
    
    V = {}
    for state in env.get_all_states():
        V[state] = agent.get_value(state)
        
    return {
        "V": serialize_grid(V)
    }

def main():
    data = {}
    
    for r_forbidden in [-1, -10]:
        print(f"Processing Reward: {r_forbidden}")
        env = GridWorld()
        env.r_forbidden = r_forbidden
        
        # Calculate Ground Truth
        print("  Calculating V*...")
        v_star = get_v_star(env)
        
        key = str(r_forbidden)
        data[key] = {}
        
        print("  Running VI...")
        data[key]["vi"] = run_vi(env, v_star)
        
        print("  Running PI...")
        data[key]["pi"] = run_pi(env, v_star)
        
        print("  Running TPI...")
        data[key]["tpi"] = {}
        for k in [3, 9, 27, 81]:
            print(f"    k={k}")
            data[key]["tpi"][str(k)] = run_tpi(env, k, v_star)
        
        print("  Running Q-Learning...")
        data[key]["q_learning"] = {}
        for eps in [0.1, 0.2, 0.5]:
            data[key]["q_learning"][str(eps)] = run_q_learning(env, eps)
            
        print("  Running TD-Linear...")
        data[key]["td_linear"] = {}
        for order in [1, 2, 3]:
            data[key]["td_linear"][f"poly_{order}"] = run_td_linear(env, 'polynomial', order)
        for order in [1, 2, 3]:
            data[key]["td_linear"][f"fourier_{order}"] = run_td_linear(env, 'fourier', order)

    with open('web/data.js', 'w') as f:
        f.write("const RL_DATA = ")
        json.dump(data, f)
        f.write(";")
        
    print("Done! Data saved to web/data.js")

if __name__ == "__main__":
    main()
