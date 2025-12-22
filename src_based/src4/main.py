import data_generator
import estimators
import visualizer
import numpy as np

def main():
    print("开始二维随机变量期望估计项目...")
    
    # 1. 数据生成
    print("\n[1] 正在生成数据...")
    data = data_generator.generate_data(num_samples=400)
    print(f"已生成 {data.shape[0]} 个样本。")
    
    # 2. 解析解
    print("\n[2] 正在计算解析均值...")
    true_mean = estimators.calculate_analytical_mean()
    print(f"解析均值 E[X]: {true_mean}")
    
    # 3. SGD 估计
    print("\n[3] 正在运行 SGD 估计...")
    
    # 策略 A: alpha_k = 1/k
    print("  正在运行 SGD 策略 A (alpha_k = 1/k)...")
    sgd_history_a = estimators.sgd_estimator(data, lr_strategy='A', num_iterations=200)
    visualizer.plot_sgd_convergence(data, true_mean, sgd_history_a, strategy_name='A')
    visualizer.plot_sgd_error(true_mean, sgd_history_a, strategy_name='A')
    
    # 策略 B: alpha_k = 0.005
    print("  正在运行 SGD 策略 B (alpha_k = 0.005)...")
    sgd_history_b = estimators.sgd_estimator(data, lr_strategy='B', num_iterations=1000)
    visualizer.plot_sgd_convergence(data, true_mean, sgd_history_b, strategy_name='B')
    visualizer.plot_sgd_error(true_mean, sgd_history_b, strategy_name='B')
    
    # 4. MBGD 估计
    print("\n[4] 正在运行 MBGD 估计...")
    batch_sizes = [1, 10, 50, 100]
    mbgd_histories = {}
    
    for m in batch_sizes:
        print(f"  正在运行批量大小 m={m} 的 MBGD...")
        history = estimators.mbgd_estimator(data, batch_size=m, num_iterations=200)
        mbgd_histories[m] = history
        
    visualizer.plot_all_comparison(data, true_mean, mbgd_histories)
    
    print("\n所有任务已完成。结果保存在 'results/' 目录中。")

if __name__ == "__main__":
    main()
