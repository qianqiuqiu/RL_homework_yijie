import numpy as np

class GridWorld:
    def __init__(self):
        self.height = 5
        self.width = 5
        self.n_states = self.height * self.width
        self.n_actions = 5 # 上, 下, 左, 右, 原地
        
        # 0: 上, 1: 下, 2: 左, 3: 右, 4: 原地
        self.actions = [0, 1, 2, 3, 4]
        self.action_names = ['Up', 'Down', 'Left', 'Right', 'Stay']
        
        # 禁区 (黄色) - 0索引 (行, 列)
        # 第2行: 第2, 3列 -> (1, 1), (1, 2)
        # 第3行: 第3列 -> (2, 2)
        # 第4行: 第2, 4列 -> (3, 1), (3, 3)
        # 第5行: 第2列 -> (4, 1)
        self.forbidden = [
            (1, 1), (1, 2),
            (2, 2),
            (3, 1), (3, 3),
            (4, 1)
        ]
        
        # 目标 (蓝色)
        # 第4行: 第3列 -> (3, 2)
        self.target = (3, 2)
        
    def step(self, state, action):
        """
        state: (行, 列)
        action: 整数
        Returns: 下一个状态, 奖励
        """
        row, col = state
        
        next_row, next_col = row, col
        
        if action == 0: # 上
            next_row = row - 1
        elif action == 1: # 下
            next_row = row + 1
        elif action == 2: # 左
            next_col = col - 1
        elif action == 3: # 右
            next_col = col + 1
        elif action == 4: # 原地
            pass
            
        # 检查边界
        hit_wall = False
        if next_row < 0 or next_row >= self.height or next_col < 0 or next_col >= self.width:
            next_row, next_col = row, col
            hit_wall = True
            
        next_state = (next_row, next_col)
        
        # 计算奖励
        reward = 0
        if next_state == self.target:
            reward = 1
        elif next_state in self.forbidden:
            reward = -1
        elif hit_wall:
            reward = -1 # 边界奖励
        else:
            reward = 0
            
        return next_state, reward

    def get_all_states(self):
        states = []
        for r in range(self.height):
            for c in range(self.width):
                states.append((r, c))
        return states
