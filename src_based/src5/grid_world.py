import random

class GridWorldEnv:
    def __init__(self):
        self.rows = 5
        self.cols = 5
        self.target_state = (4, 3)
        # 正确的障碍物状态（图 3b 中的橙色单元格）
        self.obstacle_states = [
            (2, 2), (2, 3),
            (3, 3),
            (4, 2), (4, 4),
            (5, 2)
        ]
        self.actions = ['up', 'down', 'left', 'right', 'stay']
        self.gamma = 0.9
        
        # 奖励设置
        self.r_boundary = -1
        self.r_target = 1
        self.r_forbidden = -1 # 进入障碍物的惩罚
        self.r_other = 0

    def get_all_states(self):
        """返回所有可能的状态 (row, col) 列表。"""
        states = []
        for r in range(1, self.rows + 1):
            for c in range(1, self.cols + 1):
                states.append((r, c))
        return states

    def get_valid_actions(self, state):
        """返回不会导致撞墙的有效动作列表。"""
        row, col = state
        valid_actions = []
        for action in self.actions:
            next_r, next_c = row, col
            if action == 'up':
                next_r -= 1
            elif action == 'down':
                next_r += 1
            elif action == 'left':
                next_c -= 1
            elif action == 'right':
                next_c += 1
            elif action == 'stay':
                pass
            
            if 1 <= next_r <= self.rows and 1 <= next_c <= self.cols:
                valid_actions.append(action)
        
        return valid_actions

    def step(self, state, action):
        """
        从当前状态执行一个动作。
        返回: next_state, reward
        """
        row, col = state
        next_r, next_c = row, col
        
        if action == 'up':
            next_r -= 1
        elif action == 'down':
            next_r += 1
        elif action == 'left':
            next_c -= 1
        elif action == 'right':
            next_c += 1
        elif action == 'stay':
            pass
            
        # 边界检查
        if not (1 <= next_r <= self.rows and 1 <= next_c <= self.cols):
            return state, self.r_boundary
        
        next_state = (next_r, next_c)
        
        # 目标检查
        if next_state == self.target_state:
            return next_state, self.r_target
            
        # 障碍物检查
        if next_state in self.obstacle_states:
            return next_state, self.r_forbidden
            
        # 普通状态
        return next_state, self.r_other
