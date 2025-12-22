import matplotlib.pyplot as plt
import numpy as np
import os

def ensure_output_dir(output_dir='results'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def plot_sgd_convergence(data, true_mean, sgd_history, strategy_name):
    """
    绘制 SGD 的收敛轨迹。
    """
    ensure_output_dir()
    
    sgd_history = np.array(sgd_history)
    
    plt.figure(figsize=(8, 8))
    
    # 绘制数据点
    plt.scatter(data[:, 0], data[:, 1], facecolors='none', edgecolors='k', alpha=0.3, label='data')
    
    # 绘制真实均值
    plt.scatter(true_mean[0], true_mean[1], color='k', s=100, label='mean', zorder=5)
    
    # 绘制 SGD 轨迹
    # 使用蓝色线条。箭头在 200 步时可能会显得杂乱，所以我们使用带标记的线条。
    plt.plot(sgd_history[:, 0], sgd_history[:, 1], 'b.-', label='SGD', alpha=0.6)
    
    # 标记起点和终点
    plt.scatter(sgd_history[0, 0], sgd_history[0, 1], color='g', marker='x', s=100, label='Start', zorder=5)
    plt.scatter(sgd_history[-1, 0], sgd_history[-1, 1], color='r', marker='x', s=100, label='End', zorder=5)

    plt.title(f'SGD Convergence Trajectory (Strategy {strategy_name})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal') # 确保纵横比相等，以便正确查看正方形分布
    
    filename = f'results/sgd_convergence_strategy_{strategy_name}.png'
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")

def plot_sgd_error(true_mean, sgd_history, strategy_name):
    """
    绘制 SGD 的误差曲线。
    """
    ensure_output_dir()
    
    sgd_history = np.array(sgd_history)
    # 计算每一步的欧几里得距离误差
    errors = np.linalg.norm(sgd_history - true_mean, axis=1)
    iterations = np.arange(len(errors))
    
    plt.figure(figsize=(8, 6))
    plt.plot(iterations, errors, 'r-', label='Error')
    
    plt.title(f'SGD Error Curve (Strategy {strategy_name})')
    plt.xlabel('Iteration k')
    plt.ylabel('Error ||w_k - E[X]||')
    plt.legend()
    plt.grid(True)
    
    filename = f'results/sgd_error_strategy_{strategy_name}.png'
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")

def plot_all_comparison(data, true_mean, mbgd_histories):
    """
    绘制所有算法比较图（SGD 和不同批量大小的 MBGD）：轨迹和误差曲线。
    mbgd_histories: 字典 {batch_size: history_list}
    注意：m=1 的情况即为 SGD (alpha_k = 1/k)。
    """
    ensure_output_dir()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # --- 左子图：轨迹 ---
    ax1.scatter(data[:, 0], data[:, 1], facecolors='none', edgecolors='k', alpha=0.1, label='data')
    ax1.scatter(true_mean[0], true_mean[1], color='k', s=100, label='mean', zorder=5)
    
    colors = ['b', 'g', 'r', 'm', 'c']
    
    for i, (m, history) in enumerate(mbgd_histories.items()):
        history = np.array(history)
        color = colors[i % len(colors)]
        label = f'SGD (m=1)' if m == 1 else f'MBGD (m={m})'
        ax1.plot(history[:, 0], history[:, 1], f'{color}-', label=label, alpha=0.7)
        # 标记终点
        ax1.scatter(history[-1, 0], history[-1, 1], color=color, marker='x', s=50)

    ax1.set_title('SGD & MBGD Convergence Trajectories')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')

    # --- 右子图：误差曲线 ---
    for i, (m, history) in enumerate(mbgd_histories.items()):
        history = np.array(history)
        errors = np.linalg.norm(history - true_mean, axis=1)
        iterations = np.arange(len(errors))
        color = colors[i % len(colors)]
        label = f'SGD (m=1)' if m == 1 else f'MBGD (m={m})'
        ax2.plot(iterations, errors, f'{color}-', label=label)

    ax2.set_title('SGD & MBGD Error Curves')
    ax2.set_xlabel('Iteration k')
    ax2.set_ylabel('Error ||w_k - E[X]||')
    ax2.legend()
    ax2.grid(True)
    
    filename = 'results/all_comparison.png'
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")
