import random
import numpy as np
from grid_world import GridWorldEnv
from ground_truth import OPTIMAL_STATE_VALUES_FIG3B

class QLearner:
    def __init__(self):
        self.env = GridWorldEnv()
        self.alpha = 0.1
        self.gamma = self.env.gamma
        self.q_table = {}
        
        # 为所有状态-动作对初始化 Q-table
        for state in self.env.get_all_states():
            for action in self.env.actions:
                self.q_table[(state, action)] = 0.0

    def get_max_q(self, state):
        """返回 max_a Q(s, a)"""
        q_values = [self.q_table[(state, a)] for a in self.env.actions]
        return max(q_values)

    def update(self, state, action, reward, next_state):
        """执行 Q-learning 更新"""
        current_q = self.q_table[(state, action)]
        max_next_q = self.get_max_q(next_state)
        
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[(state, action)] = new_q

    def calculate_state_value_error(self):
        """计算估计的 V(s) 与真实值之间的平均绝对误差。"""
        total_error = 0
        count = 0
        for state, true_value in OPTIMAL_STATE_VALUES_FIG3B.items():
            est_value = self.get_max_q(state)
            total_error += abs(est_value - true_value)
            count += 1
        return total_error / count if count > 0 else 0

    def run_episode(self, behavior_policy_func, num_steps):
        """
        运行单个长 episode。
        返回:
            trajectory: 访问过的状态列表。
            error_history: 随时间变化的状态值误差列表。
        """
        # 每次运行重置 Q-table
        self.q_table = {k: 0.0 for k in self.q_table}
        
        state = (1, 1) # 默认起始状态
        trajectory = [state]
        error_history = []
        
        # 初始误差
        error_history.append(self.calculate_state_value_error())
        
        for _ in range(num_steps):
            # 使用行为策略选择动作
            action = behavior_policy_func(state)
            
            # 执行动作
            next_state, reward = self.env.step(state, action)
            
            # 更新 Q-table
            self.update(state, action, reward, next_state)
            
            # 记录数据
            trajectory.append(next_state)
            error_history.append(self.calculate_state_value_error())
            
            # 移动到下一个状态
            state = next_state
            
        return trajectory, error_history
