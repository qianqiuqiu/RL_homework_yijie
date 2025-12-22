import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from environment import GridWorld
from features import FeatureExtractor
from algorithms import compute_ground_truth, generate_episodes, td_linear

def plot_3d_value_function(V, title, filename):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    x = np.arange(5) + 1 # 1 到 5
    y = np.arange(5) + 1 # 1 到 5
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((5, 5))
    
    for r in range(5):
        for c in range(5):
            # V 由 (行, 列) 索引, 它是 0 索引的
            # 绘图使用 1 索引的 X, Y
            # X 对应列, Y 对应行
            Z[r, c] = V[(r, c)]
            
    # 注意: 在 meshgrid 中, X 是列, Y 是行。
    # Z[row, col] 匹配 Y[row, col] 和 X[row, col]
    
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', alpha=0.8)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_zlabel('Value')
    ax.set_title(title)
    
    # 通常反转 Y 轴以匹配矩阵布局 (第 1 行在顶部)?
    # 但 3D 图通常原点在左下角。
    # 让我们保持标准 3D 坐标。
    # 行 1..5, 列 1..5。
    
    ax.set_xticks(np.arange(1, 6))
    ax.set_yticks(np.arange(1, 6))
    
    plt.savefig(filename)
    plt.close()

def print_value_table(V, ground_truth_V, title):
    print(f"\n--- {title} ---")
    print("State (Row, Col) | Ground Truth | Approx Value | Error")
    print("-" * 60)
    
    # 按行然后按列排序
    sorted_states = sorted(V.keys())
    
    # 也打印为 5x5 网格以便于查看
    grid_approx = np.zeros((5, 5))
    grid_error = np.zeros((5, 5))
    
    for s in sorted_states:
        gt = ground_truth_V[s]
        approx = V[s]
        error = abs(gt - approx)
        grid_approx[s] = approx
        grid_error[s] = error
        # print(f"{s} | {gt:.4f} | {approx:.4f} | {error:.4f}")
        
    print("\nApproximate Values (Grid):")
    print(grid_approx)
    print("\nAbsolute Errors (Grid):")
    print(grid_error)

def main():
    # 创建结果目录
    if not os.path.exists('results'):
        os.makedirs('results')
        
    env = GridWorld()
    
    print("Computing Ground Truth...")
    ground_truth_V = compute_ground_truth(env)

    print("\n--- Ground Truth Values (Grid) ---")
    gt_grid = np.zeros((5, 5))
    for r in range(5):
        for c in range(5):
            gt_grid[r, c] = ground_truth_V[(r, c)]
    print(gt_grid)
    print("-" * 30)
    
    # 绘制真实值
    plot_3d_value_function(ground_truth_V, "Ground Truth Value Function", "results/ground_truth_3d.png")
    
    print("Generating Episodes...")
    episodes = generate_episodes(env, num_episodes=500, steps_per_episode=500)
    
    experiments = [
        {'type': 'polynomial', 'order': 3, 'label': 'Poly-3 (R^3)'},
        {'type': 'polynomial', 'order': 6, 'label': 'Poly-6 (R^6)'},
        {'type': 'polynomial', 'order': 10, 'label': 'Poly-10 (R^10)'},
        {'type': 'fourier', 'order': 1, 'label': 'Fourier-1 (R^4)'},
        {'type': 'fourier', 'order': 2, 'label': 'Fourier-2 (R^9)'},
        {'type': 'fourier', 'order': 3, 'label': 'Fourier-3 (R^16)'},
    ]
    
    rmse_results = {}
    
    for exp in experiments:
        print(f"Running {exp['label']}...")
        fe = FeatureExtractor(exp['type'], exp['order'])
        final_V, rmse_hist, w = td_linear(env, episodes, fe, ground_truth_V, alpha=0.0005)
        
        rmse_results[exp['label']] = rmse_hist
        
        # 保存 3D 图
        plot_3d_value_function(final_V, f"Value Function - {exp['label']}", f"results/value_{exp['type']}_{exp['order']}.png")
        
        # 打印表格
        print_value_table(final_V, ground_truth_V, exp['label'])
        
    # 绘制 RMSE 比较
    plt.figure(figsize=(12, 6))
    for label, history in rmse_results.items():
        plt.plot(history, label=label)
    
    plt.xlabel('Episode')
    plt.ylabel('RMSE')
    plt.title('RMSE vs Episode')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/rmse_comparison.png')
    plt.close()
    
    # 分别绘制多项式的 RMSE
    plt.figure(figsize=(12, 6))
    for label, history in rmse_results.items():
        if 'Poly' in label:
            plt.plot(history, label=label)
    plt.xlabel('Episode')
    plt.ylabel('RMSE')
    plt.title('RMSE vs Episode (Polynomial)')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/rmse_poly.png')
    plt.close()

    # 分别绘制傅立叶的 RMSE
    plt.figure(figsize=(12, 6))
    for label, history in rmse_results.items():
        if 'Fourier' in label:
            plt.plot(history, label=label)
    plt.xlabel('Episode')
    plt.ylabel('RMSE')
    plt.title('RMSE vs Episode (Fourier)')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/rmse_fourier.png')
    plt.close()

if __name__ == "__main__":
    main()
