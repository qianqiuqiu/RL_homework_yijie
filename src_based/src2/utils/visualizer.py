import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
from typing import Dict, List, Tuple
import platform

# 配置中文字体
def setup_chinese_font():
    """配置matplotlib支持中文显示"""
    system = platform.system()
    
    if system == 'Windows':
        # Windows系统常用中文字体
        fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong']
    elif system == 'Darwin':  # macOS
        fonts = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS']
    else:  # Linux
        fonts = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Droid Sans Fallback', 'AR PL UMing CN']
    
    # 尝试设置可用的字体
    for font in fonts:
        try:
            matplotlib.rcParams['font.sans-serif'] = [font] + matplotlib.rcParams['font.sans-serif']
            matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            break
        except:
            continue

# 初始化时配置字体
setup_chinese_font()

class Visualizer:
    def __init__(self, gridworld):
        self.gridworld = gridworld
        self.grid_size = gridworld.grid_size
        
    def plot_convergence(self, results: Dict[str, Dict], save_path: str = None):
        """绘制收敛曲线对比图"""
        plt.figure(figsize=(12, 6))
        
        for algo_name, data in results.items():
            if 'convergence' in data:
                plt.plot(data['convergence'], label=algo_name, linewidth=2, marker='o', markersize=3)
        
        plt.xlabel('迭代次数', fontsize=12)
        plt.ylabel('最大值函数变化', fontsize=12)
        plt.title('三种算法的收敛速度对比', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_value_function(self, value_function: np.ndarray, title: str = "值函数", save_path: str = None):
        """绘制值函数热力图"""
        plt.figure(figsize=(8, 7))
        
        # 创建热力图
        ax = sns.heatmap(value_function, annot=True, fmt='.2f', cmap='RdYlGn',
                        cbar_kws={'label': '状态价值'}, linewidths=0.5,
                        square=True, vmin=-15, vmax=5)
        
        # 标记特殊格子
        for i, j in self.gridworld.forbidden_states:
            ax.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor='orange', lw=3))
        
        for i, j in self.gridworld.target_states:
            ax.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor='green', lw=3))
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('列', fontsize=12)
        plt.ylabel('行', fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_policy(self, policy: np.ndarray, title: str = "策略", save_path: str = None):
        """绘制策略箭头图"""
        fig, ax = plt.subplots(figsize=(8, 7))
        
        # 动作箭头映射
        action_arrows = {
            0: '↑',  # 上
            1: '↓',  # 下
            2: '←',  # 左
            3: '→',  # 右
            4: '●'   # 停留
        }
        
        rows, cols = self.grid_size
        
        # 绘制网格背景
        for i in range(rows):
            for j in range(cols):
                # 根据格子类型设置颜色
                if (i, j) in self.gridworld.forbidden_states:
                    color = 'lightsalmon'
                elif (i, j) in self.gridworld.target_states:
                    color = 'lightgreen'
                else:
                    color = 'lightblue'
                
                rect = Rectangle((j, rows-1-i), 1, 1, 
                                facecolor=color, edgecolor='black', linewidth=1)
                ax.add_patch(rect)
                
                # 添加动作箭头
                action = int(policy[i, j])
                arrow = action_arrows.get(action, '?')
                ax.text(j + 0.5, rows-1-i + 0.5, arrow,
                       ha='center', va='center', fontsize=20, fontweight='bold')
        
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_aspect('equal')
        ax.set_xticks(range(cols + 1))
        ax.set_yticks(range(rows + 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, linewidth=1)
        
        plt.title(title, fontsize=14, fontweight='bold')
        
        # 添加图例
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='lightblue', 
                      markersize=10, label='普通格子'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='lightsalmon', 
                      markersize=10, label='惩罚格子'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='lightgreen', 
                      markersize=10, label='目标格子')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_convergence_comparison(self, results: Dict[str, Dict], save_path: str = None):
        """绘制三种算法的收敛曲线对比图（横轴：迭代次数，纵轴：状态值误差）"""
        plt.figure(figsize=(12, 7))
        
        # 获取最优状态值（使用值迭代的最终结果作为参考）
        optimal_V = None
        for algo_name in ['Value Iteration', '值迭代', 'VI']:
            if algo_name in results:
                optimal_V = results[algo_name]['optimal_V']
                break
        
        if optimal_V is None:
            print("警告：无法找到最优状态值")
            return
        
        colors = {'Value Iteration': 'blue', 'Policy Iteration': 'black', 
                  'Truncated Policy Iteration': 'magenta',
                  '值迭代': 'blue', '策略迭代': 'black', '截断式策略迭代': 'magenta'}
        
        for algo_name, data in results.items():
            if 'convergence_history' not in data or not data['convergence_history']:
                continue
                
            # 计算每次迭代与最优值的最大误差
            errors = []
            for V_iter in data['convergence_history']:
                max_error = max(abs(V_iter[s] - optimal_V[s]) for s in optimal_V.keys())
                errors.append(max_error)
            
            iterations = list(range(1, len(errors) + 1))
            color = colors.get(algo_name, 'gray')
            plt.plot(iterations, errors, marker='o', linewidth=2, markersize=4, 
                    color=color, label=algo_name)
        
        plt.xlabel('迭代次数 k', fontsize=13)
        plt.ylabel('状态值误差 ||V_k - V*||', fontsize=13)
        plt.title('三种算法的收敛速度对比', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # 使用对数坐标更清晰地显示收敛过程
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_tpi_different_x(self, tpi_results_dict: Dict[int, Dict], save_path: str = None):
        """绘制不同x值的TPI收敛曲线（4张子图）"""
        fig, axes = plt.subplots(4, 1, figsize=(10, 12))
        
        # 获取最优状态值
        optimal_V = None
        for x_val, data in tpi_results_dict.items():
            if 'optimal_V' in data:
                optimal_V = data['optimal_V']
                break
        
        if optimal_V is None:
            print("警告：无法找到最优状态值")
            return
        
        x_values = sorted(tpi_results_dict.keys())
        
        for idx, x_val in enumerate(x_values):
            ax = axes[idx]
            data = tpi_results_dict[x_val]
            
            if 'convergence_history' not in data or not data['convergence_history']:
                continue
            
            # 计算每次迭代与最优值的最大误差
            errors = []
            for V_iter in data['convergence_history']:
                max_error = max(abs(V_iter[s] - optimal_V[s]) for s in optimal_V.keys())
                errors.append(max_error)
            
            iterations = list(range(1, len(errors) + 1))
            ax.plot(iterations, errors, marker='o', linewidth=2, markersize=5, color='blue')
            
            ax.set_xlabel('迭代次数', fontsize=11)
            ax.set_ylabel('状态值误差', fontsize=11)
            ax.set_title(f'Truncated policy iteration-{x_val}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, max(iterations) + 5)
            
            # 添加图例
            ax.legend([f'Truncated policy iteration-{x_val}'], loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comprehensive_report(self, results: Dict[str, Dict], save_dir: str = None):
        """创建综合分析报告"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. 收敛曲线
        ax1 = fig.add_subplot(gs[0, :])
        for algo_name, data in results.items():
            if 'convergence' in data:
                ax1.plot(data['convergence'], label=algo_name, linewidth=2, marker='o', markersize=3)
        ax1.set_xlabel('迭代次数')
        ax1.set_ylabel('最大值函数变化 (log scale)')
        ax1.set_title('收敛速度对比', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 2-4. 值函数热力图
        for idx, (algo_name, data) in enumerate(results.items()):
            if 'value_function' in data:
                ax = fig.add_subplot(gs[1, idx])
                sns.heatmap(data['value_function'], annot=True, fmt='.1f', 
                           cmap='RdYlGn', ax=ax, cbar=False, linewidths=0.5,
                           vmin=-15, vmax=5)
                ax.set_title(f'{algo_name} 值函数', fontweight='bold')
        
        # 5-7. 策略图
        action_arrows = {0: '↑', 1: '→', 2: '↓', 3: '←', 4: '●'}
        for idx, (algo_name, data) in enumerate(results.items()):
            if 'policy' in data:
                ax = fig.add_subplot(gs[2, idx])
                policy = data['policy']
                
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        if (i, j) in self.gridworld.forbidden_states:
                            color = 'lightsalmon'
                        elif (i, j) in self.gridworld.target_states:
                            color = 'lightgreen'
                        else:
                            color = 'lightblue'
                        
                        rect = Rectangle((j, self.grid_size-1-i), 1, 1, 
                                        facecolor=color, edgecolor='black', linewidth=0.5)
                        ax.add_patch(rect)
                        
                        action = policy[i, j]
                        arrow = action_arrows.get(action, '?')
                        ax.text(j + 0.5, self.grid_size-1-i + 0.5, arrow,
                               ha='center', va='center', fontsize=12, fontweight='bold')
                
                ax.set_xlim(0, self.grid_size)
                ax.set_ylim(0, self.grid_size)
                ax.set_aspect('equal')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f'{algo_name} 策略', fontweight='bold')
        
        plt.suptitle('GridWorld 强化学习算法综合分析', fontsize=16, fontweight='bold', y=0.98)
        
        if save_dir:
            plt.savefig(f'{save_dir}/comprehensive_report.png', dpi=300, bbox_inches='tight')
        plt.show()