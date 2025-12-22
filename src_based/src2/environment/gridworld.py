class GridWorld:
    def __init__(self, grid_size, special_cells=None):
        self.grid_size = grid_size
        self.special_cells = special_cells if special_cells else {
            'forbidden': [(1,1), (1,2), (2,2), (3,3), (3,1), (4,1)],
            'target': [(3,2)]
        }
        # 添加快捷访问属性
        self.forbidden_states = self.special_cells.get('forbidden', [])
        self.target_states = self.special_cells.get('target', [])
        
        self.num_states = grid_size[0] * grid_size[1]
        self.actions = ['up', 'down', 'left', 'right', 'stay']
        self.num_actions = len(self.actions)
        
        # 奖励设置
        self.r_boundary = -1
        self.r_forbidden = -10
        self.r_target = 1
        self.r_otherstep = 0

    def get_state_id(self, r, c):
        return r * self.grid_size[1] + c

    def get_coords(self, state_id):
        return divmod(state_id, self.grid_size[1])
    
    def get_all_states(self):
        """返回所有状态的列表"""
        return list(range(self.num_states))
    
    def get_all_actions(self):
        """返回所有动作的列表"""
        return list(range(self.num_actions))

    def get_transition_model(self, state_id, action_idx):
        """返回在状态state_id采取动作action_idx后转移到的新状态"""
        r, c = self.get_coords(state_id)
        old_r, old_c = r, c
        
        if action_idx == 0:  # 上
            r = r - 1
        elif action_idx == 1:  # 下
            r = r + 1
        elif action_idx == 2:  # 左
            c = c - 1
        elif action_idx == 3:  # 右
            c = c + 1
        elif action_idx == 4:  # 停留
            pass
        
        # 检查边界
        if r < 0 or r >= self.grid_size[0] or c < 0 or c >= self.grid_size[1]:
            # 撞墙，返回原位置
            return state_id, True  # True 表示撞墙
        
        return self.get_state_id(r, c), False
    
    def get_reward(self, state_id, action_idx):
        """获取在状态state_id采取动作action_idx后的即时奖励"""
        next_state, hit_wall = self.get_transition_model(state_id, action_idx)
        
        if hit_wall:
            return self.r_boundary
        
        next_r, next_c = self.get_coords(next_state)
        
        # 检查是否进入禁止格
        if (next_r, next_c) in self.special_cells.get('forbidden', []):
            return self.r_forbidden
        
        # 检查是否进入目标格
        if (next_r, next_c) in self.special_cells.get('target', []):
            return self.r_target
        
        # 普通格子
        return self.r_otherstep
    
    def get_expected_return(self, state_id, action_idx, V, gamma):
        """计算在状态state采取动作action的期望回报（确定性环境）"""
        next_state, _ = self.get_transition_model(state_id, action_idx)
        reward = self.get_reward(state_id, action_idx)
        return reward + gamma * V[next_state]

    def get_initial_policy(self):
        """初始策略：所有状态均为'停留'（动作4）"""
        return [4] * self.num_states
