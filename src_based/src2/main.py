# 这些函数已被废弃，改为直接在main中实现


if __name__ == "__main__":
    from environment.gridworld import GridWorld
    from algorithms.value_iteration import value_iteration
    from algorithms.policy_iteration import policy_iteration
    from algorithms.truncated_policy_iteration import truncated_policy_iteration
    import numpy as np

    # 初始化 GridWorld 环境
    grid_size = (5, 5)  # 示例网格尺寸
    env = GridWorld(grid_size)

    # 设置参数
    gamma = 0.9  # 折扣因子
    theta = 1e-6  # 收敛阈值

    # 使用“最糟糕”的初始策略：在 V=0 时选择期望回报最低的动作（趋向撞墙/禁入/远离目标）
    V0 = {i: 0.0 for i in range(env.num_states)}
    initial_policy = [
        min(env.get_all_actions(), key=lambda a: env.get_expected_return(s, a, V0, gamma))
        for s in range(env.num_states)
    ]

    # 运行实验
    print("\n" + "="*50)
    print("开始实验...")
    print("="*50 + "\n")
    
    # 第一步：运行值迭代获取最优值函数（作为基准）
    print("运行值迭代算法（获取最优值）...")
    vi_values, vi_iterations, vi_history = value_iteration(env, gamma, theta, track_convergence=True)
    
    print("运行策略迭代算法...")
    pi_values, pi_iterations, pi_history = policy_iteration(env, gamma, theta, initial_policy, track_convergence=True)
    
    print("运行截断式策略迭代算法（x=10）...")
    tpi_values, tpi_iterations, tpi_history = truncated_policy_iteration(env, gamma, theta, initial_policy, x_eval_steps=10, track_convergence=True)
    
    # 构建结果字典
    results = {
        'Value Iteration': {
            'optimal_V': vi_values,
            'iterations': vi_iterations,
            'convergence_history': vi_history,
            'value_function': np.array([vi_values[i] for i in range(env.num_states)]).reshape(env.grid_size),
            'policy': np.array([max(env.get_all_actions(), key=lambda a: env.get_expected_return(i, a, vi_values, gamma)) for i in range(env.num_states)]).reshape(env.grid_size)
        },
        'Policy Iteration': {
            'optimal_V': vi_values,
            'iterations': pi_iterations,
            'convergence_history': pi_history,
            'value_function': np.array([pi_values[i] for i in range(env.num_states)]).reshape(env.grid_size),
            'policy': np.array([max(env.get_all_actions(), key=lambda a: env.get_expected_return(i, a, pi_values, gamma)) for i in range(env.num_states)]).reshape(env.grid_size)
        },
        'Truncated Policy Iteration': {
            'optimal_V': vi_values,
            'iterations': tpi_iterations,
            'convergence_history': tpi_history,
            'value_function': np.array([tpi_values[i] for i in range(env.num_states)]).reshape(env.grid_size),
            'policy': np.array([max(env.get_all_actions(), key=lambda a: env.get_expected_return(i, a, tpi_values, gamma)) for i in range(env.num_states)]).reshape(env.grid_size)
        }
    }
    
    print("\n实验结果:")
    for algo_name, data in results.items():
        print(f"{algo_name}: {data['iterations']} 次迭代")

    # 可视化结果
    print("\n" + "="*50)
    print("开始生成可视化结果...")
    print("="*50)
    
    from utils.visualizer import Visualizer
    
    vis = Visualizer(env)
    
    # 1. 三种算法的收敛曲线对比（新图）
    print("\n生成三种算法收敛曲线对比图...")
    vis.plot_convergence_comparison(results)
    
    # 2. 各算法的值函数
    print("生成值函数热力图...")
    for algo_name, data in results.items():
        vis.plot_value_function(data['value_function'], title=f"{algo_name} - 值函数")
    
    # 3. 各算法的策略
    print("生成策略可视化...")
    for algo_name, data in results.items():
        vis.plot_policy(data['policy'], title=f"{algo_name} - 最优策略")
    
    # 4. TPI不同x值的收敛分析（新图：x=1,4,12,80）
    print("\n分析TPI不同截断步数的收敛过程...")
    x_values_to_test = [1, 4, 12, 80]
    tpi_results_dict = {}
    
    for x in x_values_to_test:
        print(f"  运行 TPI (x={x})...")
        tpi_v, tpi_iter, tpi_hist = truncated_policy_iteration(
            env, gamma, theta, initial_policy,
            x_eval_steps=x, track_convergence=True
        )
        tpi_results_dict[x] = {
            'optimal_V': vi_values,
            'iterations': tpi_iter,
            'convergence_history': tpi_hist
        }
        print(f"    完成，迭代次数: {tpi_iter}")
    
    vis.plot_tpi_different_x(tpi_results_dict)
    
    print("\n" + "="*50)
    print("可视化完成！")
    print("="*50)