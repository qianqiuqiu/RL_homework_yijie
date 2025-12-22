import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from grid_world import GridWorldEnv

class Visualizer:
    def __init__(self):
        self.env = GridWorldEnv()

    def plot_trajectory(self, trajectory, title, filename):
        """
        在网格上绘制轨迹图。
        绘制连接连续状态的绿色线条。
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        self._draw_grid_base(ax)
        
        # 提取坐标
        # trajectory 是 (row, col) 列表
        # 转换为中心坐标 x = col - 0.5, y = row - 0.5
        xs = [s[1] - 0.5 for s in trajectory]
        ys = [s[0] - 0.5 for s in trajectory]
        
        # 绘制线条
        # 使用绿色线条连接路径点
        ax.plot(xs, ys, color='lime', linewidth=2, alpha=0.1)
        
        # 重新绘制起点和终点标记，或者保持 _draw_grid_base 的标记
        # _draw_grid_base 已经绘制了障碍物和目标
        
        plt.title(f"Trajectory: {title}")
        plt.savefig(filename)
        plt.close()

    def plot_state_value_error(self, error_history, title, filename):
        """绘制误差曲线。"""
        plt.figure(figsize=(10, 6))
        plt.plot(error_history)
        plt.title(title)
        plt.xlabel("Steps")
        plt.ylabel("Mean Absolute Error (MAE)")
        plt.grid(True)
        plt.yscale('log') # 对数刻度可能更适合观察收敛
        plt.savefig(filename)
        plt.close()

    def _draw_grid_base(self, ax):
        """绘制带有障碍物和目标的基础网格的辅助函数。"""
        ax.set_xlim(0, self.env.cols)
        ax.set_ylim(self.env.rows, 0) # 反转 y 轴
        
        # 网格线
        ax.set_xticks(np.arange(0, self.env.cols + 1, 1))
        ax.set_yticks(np.arange(0, self.env.rows + 1, 1))
        ax.grid(which='major', color='black', linestyle='-', linewidth=1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # 绘制障碍物 (橙色)
        for r, c in self.env.obstacle_states:
            rect = patches.Rectangle((c-1, r-1), 1, 1, facecolor='orange', edgecolor='black')
            ax.add_patch(rect)
            
        # 绘制目标 (蓝色)
        tr, tc = self.env.target_state
        rect = patches.Rectangle((tc-1, tr-1), 1, 1, facecolor='skyblue', edgecolor='black')
        ax.add_patch(rect)
        # ax.text(tc-0.5, tr-0.5, 'Target', ha='center', va='center')

    def plot_learned_policy(self, q_learner, title, filename):
        """绘制学习到的最优策略 (箭头)。"""
        fig, ax = plt.subplots(figsize=(6, 6))
        self._draw_grid_base(ax)
        
        for r in range(1, self.env.rows + 1):
            for c in range(1, self.env.cols + 1):
                state = (r, c)
                # 我们为所有状态绘制箭头，包括障碍物，因为它们是可通行的。
                
                # 找到最佳动作
                actions = self.env.actions
                # q_learner.q_table 是一个字典 {(state, action): value}
                q_values = {a: q_learner.q_table.get((state, a), 0.0) for a in actions}
                
                best_action = max(q_values, key=q_values.get)
                
                # 中心坐标
                x, y = c - 0.5, r - 0.5
                dx, dy = 0, 0
                arrow_len = 0.3
                
                if best_action == 'up': dy = -arrow_len
                elif best_action == 'down': dy = arrow_len
                elif best_action == 'left': dx = -arrow_len
                elif best_action == 'right': dx = arrow_len
                elif best_action == 'stay':
                    # 为 stay 绘制一个圆圈
                    circle = patches.Circle((x, y), 0.1, fc='k')
                    ax.add_patch(circle)
                    continue # 跳过箭头绘制
                
                ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, fc='k', ec='k')
        
        plt.title(title)
        plt.savefig(filename)
        plt.close()

    def plot_learned_values(self, q_learner, title, filename):
        """绘制学习到的状态值 (文本)。"""
        fig, ax = plt.subplots(figsize=(6, 6))
        self._draw_grid_base(ax)
        
        for r in range(1, self.env.rows + 1):
            for c in range(1, self.env.cols + 1):
                state = (r, c)
                
                value = q_learner.get_max_q(state)
                ax.text(c-0.5, r-0.5, f"{value:.2f}", ha='center', va='center', fontsize=10)
                
        plt.title(title)
        plt.savefig(filename)
        plt.close()
