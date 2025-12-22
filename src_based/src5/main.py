import os
from grid_world import GridWorldEnv
from behavior_policies import BehaviorPolicies
from q_learner import QLearner
from visualizer import Visualizer

def main():
    # 参数设置
    NUM_STEPS = 100000
    
    # 初始化模块
    env = GridWorldEnv()
    policies = BehaviorPolicies()
    learner = QLearner()
    viz = Visualizer()
    
    # 定义要测试的策略
    policy_list = [
        (policies.get_policy_fig1a, "Policy_Fig1a_Epsilon1"),
        (policies.get_policy_fig1b, "Policy_Fig1b_Epsilon0.5"),
        (policies.get_policy_fig1c, "Policy_Fig1c_Epsilon0.1_Var1"),
        (policies.get_policy_fig1d, "Policy_Fig1d_Epsilon0.1_Var2")
    ]
    
    print(f"Starting experiments with {NUM_STEPS} steps per episode...")
    
    for policy_func, policy_name in policy_list:
        print(f"Running {policy_name}...")
        
        # 运行 episode
        trajectory, error_history = learner.run_episode(policy_func, NUM_STEPS)
        
        # 可视化
        print(f"Generating plots for {policy_name}...")
        viz.plot_trajectory(trajectory, title=policy_name, filename=f"{policy_name}_trajectory.png")
        viz.plot_state_value_error(error_history, title=f"Error: {policy_name}", filename=f"{policy_name}_error.png")
        viz.plot_learned_policy(learner, title=f"Learned Policy: {policy_name}", filename=f"{policy_name}_policy.png")
        viz.plot_learned_values(learner, title=f"Learned Values: {policy_name}", filename=f"{policy_name}_values.png")
        
        print(f"Finished {policy_name}. Final Error: {error_history[-1]:.4f}")

    print("All experiments completed.")

if __name__ == "__main__":
    main()
