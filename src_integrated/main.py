import numpy as np
import os
from envs.grid_world import GridWorld
from algorithms.dp.value_iteration import ValueIterationAgent
from algorithms.dp.policy_iteration import PolicyIterationAgent
from algorithms.dp.truncated_policy_iteration import TruncatedPolicyIterationAgent
from algorithms.monte_carlo.mc_agent import MCAgent
from algorithms.temporal_difference.q_learning import QLearningAgent
from algorithms.approximation.td_linear import TDLinearAgent
from algorithms.approximation.sgd_optimizer import SGDOptimizer
from utils.features import FeatureExtractor
from utils.plotting import plot_value_function

def get_v_from_q(agent, env):
    """Helper to extract V from Q-table"""
    V = {}
    for state in env.get_all_states():
        q_values = agent.get_q(state)
        V[state] = np.max(q_values)
    return V

def get_v_from_approx(agent, env):
    """Helper to extract V from approximation"""
    V = {}
    for state in env.get_all_states():
        V[state] = agent.get_value(state)
    return V

def main():
    print("=== Unified Reinforcement Learning Framework ===")
    
    # Setup result directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result_dir = os.path.join(base_dir, 'result')
    os.makedirs(result_dir, exist_ok=True)
    print(f"Results will be saved to: {result_dir}")

    # 1. Initialize Environment
    env = GridWorld()
    print("\nEnvironment Initialized: 5x5 GridWorld")
    
    # 2. Run Value Iteration (DP)
    print("\n--- Running Value Iteration ---")
    vi_agent = ValueIterationAgent(env)
    V_vi, policy_vi = vi_agent.train()
    print("Value Iteration Converged.")
    plot_value_function(V_vi, title="VI Value Function", save_path=os.path.join(result_dir, 'vi_value_function.png'))
    
    # 3. Run Policy Iteration (DP)
    print("\n--- Running Policy Iteration ---")
    pi_agent = PolicyIterationAgent(env)
    V_pi, policy_pi, _ = pi_agent.train()
    print("Policy Iteration Converged.")
    plot_value_function(V_pi, title="PI Value Function", save_path=os.path.join(result_dir, 'pi_value_function.png'))
    
    # 3.5. Run Truncated Policy Iteration (DP)
    print("\n--- Running Truncated Policy Iteration ---")
    tpi_agent = TruncatedPolicyIterationAgent(env, k=10)
    V_tpi, policy_tpi, _ = tpi_agent.train()
    print("Truncated Policy Iteration Converged.")
    plot_value_function(V_tpi, title="Truncated PI Value Function", save_path=os.path.join(result_dir, 'tpi_value_function.png'))
    
    # 4. Run Monte Carlo
    print("\n--- Running Monte Carlo Agent ---")
    # User requested epsilon=0 (Greedy with Exploring Starts) to match original HW3 performance
    # Increased alpha to 0.05 for faster convergence
    mc_agent = MCAgent(env, epsilon=0, alpha=0.05) 
    mc_agent.train(num_episodes=20000, max_steps=200) 
    print("MC Training Completed.")
    V_mc = get_v_from_q(mc_agent, env)
    plot_value_function(V_mc, title="MC Value Function", save_path=os.path.join(result_dir, 'mc_value_function.png'))
    
    # 5. Run Q-Learning
    print("\n--- Running Q-Learning Agent ---")
    ql_agent = QLearningAgent(env)
    # Strategy Pattern: Define a custom behavior policy
    def custom_epsilon_greedy(agent, state):
        epsilon = 0.2 # Higher exploration
        if np.random.rand() < epsilon:
            return np.random.choice(agent.actions)
        return agent.predict(state)
        
    ql_agent.train(num_episodes=5000, behavior_policy=custom_epsilon_greedy)
    print("Q-Learning Training Completed.")
    V_ql = get_v_from_q(ql_agent, env)
    plot_value_function(V_ql, title="Q-Learning Value Function", save_path=os.path.join(result_dir, 'q_learning_value_function.png'))
    
    # 6. Run TD Linear Approximation
    print("\n--- Running TD Linear Approximation ---")
    features = FeatureExtractor(feature_type='fourier', order=3, grid_size=(5,5))
    optimizer = SGDOptimizer(learning_rate=0.01, decay_type='inverse')
    td_agent = TDLinearAgent(env, features, optimizer=optimizer)
    td_agent.train(num_episodes=5000)
    print("TD Linear Training Completed.")
    V_td = get_v_from_approx(td_agent, env)
    plot_value_function(V_td, title="TD Linear Value Function", save_path=os.path.join(result_dir, 'td_linear_value_function.png'))
    
    print("\nAll algorithms executed successfully. Check the 'result' folder for plots.")

if __name__ == "__main__":
    main()
